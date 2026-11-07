"""
Background worker executing multi-scenario sweeps concurrently via ProcessPoolExecutor.
"""

import os
import time
import atexit
import tempfile
import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from PyQt5.QtCore import QThread, pyqtSignal

from physics.headless_sweep import (
    run_single_sweep_scenario_worker,
    write_scenario_state,
    read_scenario_state,
)

# Global registry of active worker instances for emergency atexit cleanup
_active_workers = set()


def _cleanup_all_active_workers():
    for w in list(_active_workers):
        try:
            w.cancel()
            w._kill_worker_processes()
            w._cleanup_temp_files()
        except Exception:
            pass


atexit.register(_cleanup_all_active_workers)


class ParallelSweepWorker(QThread):
    """
    QThread that orchestrates running multiple simulation scenarios across a ProcessPoolExecutor.
    Reports progress and results back to the GUI via Qt signals without freezing the event loop.
    """

    scenario_completed = pyqtSignal(dict, int, int)  # (result_row, completed_count, total_count)
    progress_updated = pyqtSignal(int, int, str)     # (completed_count, total_count, status_text)
    sweep_finished = pyqtSignal(list)                # (all_sorted_results)
    sweep_error = pyqtSignal(str)                    # (error_message)

    def __init__(self, sweep_config: dict, base_params: dict, n_cores: int = None, scatter_limit=2000, parent=None):
        super().__init__(parent)
        self.sweep_config = sweep_config
        self.base_params = base_params
        self.scatter_limit = scatter_limit
        total_cpus = os.cpu_count() or 16
        self.n_cores = int(n_cores) if n_cores and n_cores > 0 else max(1, total_cpus // 2)
        self.total_computation_time_s = 0.0
        self._canceled = False
        self._executor = None
        worker_token = f"{os.getpid()}_{id(self)}"
        self.cancel_file = os.path.join(tempfile.gettempdir(), f"pybemcs_sweep_{worker_token}.cancel")
        if os.path.exists(self.cancel_file):
            try:
                os.remove(self.cancel_file)
            except Exception:
                pass

        self.active_view_file = os.path.join(tempfile.gettempdir(), f"pybemcs_sweep_{worker_token}.active_view")
        try:
            with open(self.active_view_file, "w") as f:
                f.write("1")
        except Exception:
            pass

        # File-based state and pause tracking per scenario (fast, decoupled, zero-daemon)
        self.state_files = {}
        self.pause_files = {}
        self.final_status_dict = {}
        self._paused_scenarios = set()

        values = self.sweep_config.get("values", [])
        max_iters = int(self.sweep_config.get("max_iterations", 5000))
        for idx, val in enumerate(values):
            scen_num = idx + 1
            sf = os.path.join(tempfile.gettempdir(), f"pybemcs_scen_{worker_token}_{scen_num}.bin")
            pf = os.path.join(tempfile.gettempdir(), f"pybemcs_scen_{worker_token}_{scen_num}.pause")
            self.state_files[scen_num] = sf
            self.pause_files[scen_num] = pf
            if os.path.exists(pf):
                try:
                    os.remove(pf)
                except Exception:
                    pass
            init_st = {
                "status": "queued",
                "iter": 0,
                "max_iters": max_iters,
                "parameter_value": float(val),
                "steady_state_reached": False,
                "computation_time_s": 0.0,
                "t_sim_us": 0.0,
                "div": 0.0,
                "div_mid": 0.0,
                "exit_ion_current_A": 0.0,
                "transparency": 0.0,
                "prim_xy": np.empty((0, 2), dtype=np.float32),
                "e_prim_eV": np.empty(0, dtype=np.float32),
                "cex_xy": np.empty((0, 2), dtype=np.float32),
                "e_cex_eV": np.empty(0, dtype=np.float32),
                "elec_xy": np.empty((0, 2), dtype=np.float32),
                "iter_history": [],
                "div_history": [],
                "div_mid_history": [],
                "ebs_history": [],
                "energy_history_t": [],
                "energy_history_ke_i": [],
                "energy_history_ke_e": [],
                "energy_history_fe": [],
                "energy_history_tot": [],
                "substep_history_iter": [],
                "substep_history_ion": [],
                "substep_history_e": [],
                "current_ppc_map": None,
                "accum_ppc_map": None,
                "ppc_steps_count": 0,
            }
            self.final_status_dict[scen_num] = init_st
            write_scenario_state(sf, init_st)

    def pause_scenario(self, scen_num):
        """Pause a specific worker scenario by creating its pause signal file."""
        if not hasattr(self, '_paused_scenarios'):
            self._paused_scenarios = set()
        self._paused_scenarios.add(scen_num)
        pf = self.pause_files.get(scen_num)
        if pf:
            try:
                with open(pf, "w") as f:
                    f.write("paused")
            except Exception:
                pass
        if scen_num in self.final_status_dict and self.final_status_dict[scen_num].get("status") != "done":
            self.final_status_dict[scen_num]["status"] = "paused"
        sf = self.state_files.get(scen_num)
        if sf and os.path.exists(sf):
            try:
                st = read_scenario_state(sf)
                if st and st.get("status") != "done":
                    st["status"] = "paused"
                    write_scenario_state(sf, st)
            except Exception:
                pass

    def resume_scenario(self, scen_num):
        """Resume a specific worker scenario by removing its pause signal file."""
        if not hasattr(self, '_paused_scenarios'):
            self._paused_scenarios = set()
        self._paused_scenarios.discard(scen_num)
        pf = self.pause_files.get(scen_num)
        if pf and os.path.exists(pf):
            try:
                os.remove(pf)
            except Exception:
                pass
        if scen_num in self.final_status_dict and self.final_status_dict[scen_num].get("status") != "done":
            self.final_status_dict[scen_num]["status"] = "running"
        sf = self.state_files.get(scen_num)
        if sf and os.path.exists(sf):
            try:
                st = read_scenario_state(sf)
                if st and st.get("status") != "done":
                    st["status"] = "running"
                    write_scenario_state(sf, st)
            except Exception:
                pass

    def pause_all_scenarios(self):
        """Pause all running scenarios."""
        for scen_num in list(self.pause_files.keys()):
            if self.final_status_dict.get(scen_num, {}).get("status") != "done":
                self.pause_scenario(scen_num)

    def resume_all_scenarios(self):
        """Resume all paused scenarios."""
        for scen_num in list(self.pause_files.keys()):
            if self.is_scenario_paused(scen_num):
                self.resume_scenario(scen_num)

    def is_scenario_paused(self, scen_num):
        """Check if a specific scenario is currently paused."""
        if not hasattr(self, '_paused_scenarios'):
            self._paused_scenarios = set()
        if scen_num in self._paused_scenarios:
            return True
        pf = self.pause_files.get(scen_num)
        if pf and os.path.exists(pf):
            return True
        return False

    def set_active_view_scenario(self, scen_num):
        """Record which scenario is currently being viewed in the GUI."""
        if hasattr(self, 'active_view_file'):
            try:
                with open(self.active_view_file, "w") as f:
                    f.write(str(scen_num))
            except Exception:
                pass

    def get_status_dict(self):
        """Thread-safe snapshot dictionary of all scenario execution states."""
        res = dict(getattr(self, 'final_status_dict', {}))
        paused_set = getattr(self, '_paused_scenarios', set())
        for scen_num, sf in getattr(self, 'state_files', {}).items():
            st = read_scenario_state(sf)
            if st is not None:
                if st.get("status") != "done":
                    if scen_num in paused_set:
                        st["status"] = "paused"
                    elif st.get("status") == "paused" and not self.is_scenario_paused(scen_num):
                        st["status"] = "running"
                res[scen_num] = st
                self.final_status_dict[scen_num] = st
        return res

    def _cleanup_temp_files(self):
        """Remove temporary state, pause, and cancel files."""
        if hasattr(self, 'cancel_file') and os.path.exists(self.cancel_file):
            try:
                os.remove(self.cancel_file)
            except Exception:
                pass
        for sf in getattr(self, 'state_files', {}).values():
            try:
                if os.path.exists(sf):
                    os.remove(sf)
            except Exception:
                pass
        for pf in getattr(self, 'pause_files', {}).values():
            try:
                if os.path.exists(pf):
                    os.remove(pf)
            except Exception:
                pass
        if hasattr(self, 'active_view_file') and os.path.exists(self.active_view_file):
            try:
                os.remove(self.active_view_file)
            except Exception:
                pass

    def _kill_worker_processes(self):
        """Forcibly and immediately terminate all worker OS processes."""
        ex = self._executor
        if ex is not None:
            procs = list(getattr(ex, '_processes', {}).values()) if getattr(ex, '_processes', None) else []
            for p in procs:
                if p is not None:
                    try:
                        p.kill()
                    except Exception:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                    pid = getattr(p, 'pid', None)
                    if pid:
                        try:
                            os.kill(pid, 9)
                        except Exception:
                            pass

    def cancel(self):
        """Request immediate cancellation of sweep execution."""
        self._canceled = True
        try:
            with open(self.cancel_file, "w") as f:
                f.write("cancelled")
        except Exception:
            pass

        self._kill_worker_processes()
        _active_workers.discard(self)

    def run(self):
        if self._canceled:
            return

        _active_workers.add(self)
        values = self.sweep_config.get("values", [])
        total_count = len(values)
        if total_count == 0:
            _active_workers.discard(self)
            self.sweep_finished.emit([])
            return

        param_id = self.sweep_config.get("param_id", "")
        steady_cfg = {
            "window": self.sweep_config.get("window", 50),
            "threshold": self.sweep_config.get("threshold", 1.0),
            "max_iterations": self.sweep_config.get("max_iterations", 5000),
            "min_iterations": self.sweep_config.get("min_iterations", 500),
        }

        payloads = []
        for idx, val in enumerate(values):
            scen_idx = idx + 1
            payloads.append({
                "scenario_idx": scen_idx,
                "param_id": param_id,
                "parameter_value": float(val),
                "base_params": self.base_params,
                "steady_config": steady_cfg,
                "cancel_file": self.cancel_file,
                "pause_file": self.pause_files.get(scen_idx),
                "state_file": self.state_files.get(scen_idx),
                "active_view_file": self.active_view_file,
                "scatter_limit": getattr(self, 'scatter_limit', 2000),
            })

        results = []
        completed_count = 0
        sweep_t0 = time.perf_counter()

        self.progress_updated.emit(0, total_count, f"Starting parallel sweep with {self.n_cores} worker cores...")

        if self._canceled:
            _active_workers.discard(self)
            return

        executor = ProcessPoolExecutor(max_workers=self.n_cores)
        self._executor = executor

        try:
            future_to_idx = {
                executor.submit(run_single_sweep_scenario_worker, p): p["scenario_idx"]
                for p in payloads
            }

            for future in as_completed(future_to_idx):
                if self._canceled:
                    break
                try:
                    row = future.result()
                    results.append(row)
                    completed_count += 1
                    val_str = f"{row['parameter_value']:.4g}"
                    c_time = row.get("computation_time_s", 0.0)
                    time_info = f" in {c_time:.1f}s" if (isinstance(c_time, (int, float)) and c_time > 0) else ""
                    self.scenario_completed.emit(row, completed_count, total_count)
                    self.progress_updated.emit(
                        completed_count, total_count,
                        f"Completed Scenario {row['scenario_idx']}/{total_count} (value={val_str}{time_info})"
                    )
                except Exception as exc:
                    if not self._canceled:
                        self.sweep_error.emit(f"Scenario error: {exc}")

        except Exception as e:
            if not self._canceled:
                self.sweep_error.emit(f"Process pool error: {e}")
            return
        finally:
            self.total_computation_time_s = time.perf_counter() - sweep_t0
            if self.sweep_config is not None:
                self.sweep_config["total_computation_time_s"] = float(self.total_computation_time_s)
            _active_workers.discard(self)
            try:
                for scen_num, sf in getattr(self, 'state_files', {}).items():
                    st = read_scenario_state(sf)
                    if st is not None:
                        self.final_status_dict[scen_num] = st
            except Exception:
                pass

            if self._canceled:
                self._kill_worker_processes()
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            else:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass
            self._executor = None

            if os.path.exists(self.cancel_file):
                try:
                    os.remove(self.cancel_file)
                except Exception:
                    pass
            self._cleanup_temp_files()

        if not self._canceled:
            # Sort results by scenario_idx so order matches input values
            results.sort(key=lambda r: r.get("scenario_idx", 0))
            self.sweep_finished.emit(results)
