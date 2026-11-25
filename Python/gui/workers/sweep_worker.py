"""
Background worker executing multi-scenario sweeps concurrently via ProcessPoolExecutor.
"""

import os
import copy
import time
import atexit
import tempfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from PyQt5.QtCore import QThread, pyqtSignal

from physics.headless_sweep import (
    run_single_sweep_scenario_worker,
    write_scenario_state,
    read_scenario_state,
)
from physics.hardware import (
    detect_hardware_profile,
    estimate_payload_ram_gb,
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
    scenario_action_needed = pyqtSignal(int, dict)   # (scenario_idx, advisory_info)

    def __init__(self, sweep_config: dict, base_params: dict, n_cores: int = None, scatter_limit=2000, parent=None, remaining_indices: list = None, existing_results: list = None, held_scenarios: list = None, track_particles: bool = False):
        super().__init__(parent)
        self.sweep_config = sweep_config
        self.base_params = base_params
        self.scatter_limit = scatter_limit
        self.remaining_indices = set(remaining_indices) if remaining_indices is not None else None
        self.existing_results = list(existing_results) if existing_results is not None else []
        self.held_scenarios = set(held_scenarios) if held_scenarios is not None else set()
        self._all_payloads_ref = []
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

        self.track_ptcls_file = os.path.join(tempfile.gettempdir(), f"pybemcs_sweep_{worker_token}.track_ptcls")
        self.set_track_particles(bool(track_particles))

        self.sim_mode_file = os.path.join(tempfile.gettempdir(), f"pybemcs_sweep_{worker_token}.sim_mode")
        initial_sim_mode = self.base_params.get("sim_mode", "Both") if isinstance(self.base_params, dict) else "Both"
        self.set_sim_mode(str(initial_sim_mode))

        self.processes_file = os.path.join(tempfile.gettempdir(), f"pybemcs_sweep_{worker_token}.processes")
        initial_see = bool(self.base_params.get("enable_see", True)) if isinstance(self.base_params, dict) else True
        initial_cex = bool(self.base_params.get("enable_cex", True)) if isinstance(self.base_params, dict) else True
        self.set_processes(initial_see, initial_cex)

        # File-based state and pause tracking per scenario (fast, decoupled, zero-daemon)
        self.state_files = {}
        self.pause_files = {}
        self.trk_files = {}
        self.final_status_dict = {}
        self.final_tracking_buffers = {}
        self._paused_scenarios = set()

        values = self.sweep_config.get("values", [])
        max_iters = int(self.sweep_config.get("max_iterations", 5000))
        existing_map = {r.get("scenario_idx"): r for r in self.existing_results if isinstance(r, dict)}

        for idx, val in enumerate(values):
            scen_num = idx + 1
            sf = os.path.join(tempfile.gettempdir(), f"pybemcs_scen_{worker_token}_{scen_num}.bin")
            pf = os.path.join(tempfile.gettempdir(), f"pybemcs_scen_{worker_token}_{scen_num}.pause")
            tf = os.path.join(tempfile.gettempdir(), f"pybemcs_scen_{worker_token}_{scen_num}.trk.bin")
            self.state_files[scen_num] = sf
            self.pause_files[scen_num] = pf
            self.trk_files[scen_num] = tf
            for f_clean in (pf, tf):
                if os.path.exists(f_clean):
                    try:
                        os.remove(f_clean)
                    except Exception:
                        pass

            is_already_done = (self.remaining_indices is not None and scen_num not in self.remaining_indices)
            is_held = scen_num in self.held_scenarios
            ex_row = existing_map.get(scen_num, {})

            # Recover pre-existing histories for already completed scenarios if available
            ex_hist = {}
            if is_already_done:
                if isinstance(ex_row, dict) and len(ex_row.get("iter_history") or []) > 0:
                    ex_hist = ex_row
                elif os.path.exists(sf):
                    try:
                        old_st = read_scenario_state(sf)
                        if isinstance(old_st, dict) and len(old_st.get("iter_history") or []) > 0:
                            ex_hist = old_st
                    except Exception:
                        pass

            init_st = {
                "status": "done" if is_already_done else ("action_required" if is_held else "queued"),
                "held": is_held,
                "iter": ex_row.get("steady_iterations", ex_hist.get("iter", 0)) if is_already_done else 0,
                "max_iters": max_iters,
                "parameter_value": float(val),
                "steady_state_reached": ex_row.get("steady_state_reached", True) if is_already_done else False,
                "computation_time_s": ex_row.get("computation_time_s", ex_hist.get("computation_time_s", 0.0)) if is_already_done else 0.0,
                "t_sim_us": ex_row.get("t_sim_us", ex_hist.get("t_sim_us", 0.0)) if is_already_done else 0.0,
                "div": ex_row.get("divergence_deg", ex_row.get("div", ex_hist.get("div", 0.0))) if is_already_done else 0.0,
                "div_mid": ex_row.get("divergence_2nd_plane_deg", ex_row.get("div_mid", ex_hist.get("div_mid", 0.0))) if is_already_done else 0.0,
                "saddle_potential_V": ex_row.get("saddle_potential_V", ex_row.get("ebs", ex_hist.get("saddle_potential_V", 0.0))) if is_already_done else 0.0,
                "exit_ion_current_A": ex_row.get("exit_ion_current_A", ex_hist.get("exit_ion_current_A", 0.0)) if is_already_done else 0.0,
                "transparency": ex_row.get("transparency", ex_hist.get("transparency", 0.0)) if is_already_done else 0.0,
                "prim_xy": ex_hist.get("prim_xy", np.empty((0, 2), dtype=np.float32)),
                "e_prim_eV": ex_hist.get("e_prim_eV", np.empty(0, dtype=np.float32)),
                "cex_xy": ex_hist.get("cex_xy", np.empty((0, 2), dtype=np.float32)),
                "e_cex_eV": ex_hist.get("e_cex_eV", np.empty(0, dtype=np.float32)),
                "elec_xy": ex_hist.get("elec_xy", np.empty((0, 2), dtype=np.float32)),
                "iter_history": list(ex_hist.get("iter_history", [])),
                "time_history": list(ex_hist.get("time_history", [])),
                "div_history": list(ex_hist.get("div_history", [])),
                "div_mid_history": list(ex_hist.get("div_mid_history", [])),
                "ebs_history": list(ex_hist.get("ebs_history", [])),
                "transparency_history": list(ex_hist.get("transparency_history", [])),
                "ion_current_exit_avg_history": list(ex_hist.get("ion_current_exit_avg_history", [])),
                "active_cells_history": list(ex_hist.get("active_cells_history", [])),
                "low_ppc_cells_history": list(ex_hist.get("low_ppc_cells_history", [])),
                "T_histories": dict(ex_hist.get("T_histories", {})),
                "T_grids": ex_row.get("T_grids_K", []) if is_already_done else [],
                "T_grids_K": ex_row.get("T_grids_K", []) if is_already_done else [],
                "T_grids_C": ex_row.get("T_grids_C", []) if is_already_done else [],
                "Tmap": None,
                "mask_grids": [],
                "isBound": None,
                "energy_history_t": [],
                "energy_history_ke_i": [],
                "energy_history_ke_e": [],
                "energy_history_fe": [],
                "energy_history_tot": [],
                "charge_history_t": [],
                "charge_history_q_i": [],
                "charge_history_q_e": [],
                "charge_history_q_b": [],
                "charge_history_q_free": [],
                "charge_history_q_net": [],
                "e_elec_eV": np.empty(0, dtype=np.float32),
                "substep_history_iter": [],
                "substep_history_ion": [],
                "substep_history_e": [],
                "plume_history_iterations": list(ex_hist.get("plume_history_iterations", [])),
                "plume_history_dx": list(ex_hist.get("plume_history_dx", [])),
                "plume_dx_history": list(ex_hist.get("plume_dx_history", [])),
                "zone_configs": ex_hist.get("zone_configs", None),
                "ion_current_exit_history": [],
                "transmitted_ions_history": [],
                "V": None,
                "current_ppc_map": None,
                "accum_ppc_map": None,
                "ppc_steps_count": 0,
                "macro_weight": float(getattr(self.base_params, 'get', lambda k, d: d)("macro_weight", 1e8) if hasattr(self.base_params, 'get') else 1e8),
                "perf_history": [],
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

    def is_scenario_held(self, scen_num: int) -> bool:
        """Check if scenario is currently held awaiting user decision."""
        return scen_num in getattr(self, 'held_scenarios', set())

    def release_scenario(self, scen_num: int, truncate_plume: bool = False):
        """Release a scenario from the held advisory state so it can execute."""
        if not hasattr(self, 'held_scenarios'):
            self.held_scenarios = set()
        self.held_scenarios.discard(scen_num)

        # Update payload parameters
        for p in getattr(self, '_all_payloads_ref', []):
            if p.get("scenario_idx") == scen_num:
                if not isinstance(p.get("base_params"), dict):
                    p["base_params"] = dict(self.base_params)
                else:
                    p["base_params"] = dict(p["base_params"])
                p["base_params"]["truncate_downstream_plume"] = bool(truncate_plume)
                if not isinstance(p.get("sweep_config"), dict):
                    p["sweep_config"] = {}
                p["sweep_config"]["truncate_downstream_plume"] = bool(truncate_plume)

        if scen_num in self.final_status_dict and self.final_status_dict[scen_num].get("status") not in ("done", "running"):
            self.final_status_dict[scen_num]["status"] = "queued"
            self.final_status_dict[scen_num]["held"] = False
        sf = self.state_files.get(scen_num)
        if sf and os.path.exists(sf):
            try:
                st = read_scenario_state(sf)
                if st and st.get("status") not in ("done", "running"):
                    st["status"] = "queued"
                    st["held"] = False
                    write_scenario_state(sf, st)
            except Exception:
                pass

    def set_active_view_scenario(self, scen_num):
        """Record which scenario is currently being viewed in the GUI."""
        if hasattr(self, 'active_view_file'):
            try:
                with open(self.active_view_file, "w") as f:
                    f.write(str(scen_num))
            except Exception:
                pass

    def set_track_particles(self, enabled: bool):
        """Enable or disable particle kinematics recording across worker processes."""
        self.track_particles_enabled = bool(enabled)
        tf = getattr(self, 'track_ptcls_file', None)
        if not tf:
            return
        if enabled:
            try:
                with open(tf, "w") as f:
                    f.write("1")
            except Exception:
                pass
        else:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except Exception:
                    pass

    def set_sim_mode(self, mode: str):
        """Synchronize simulation mode (Both, Thermal, Erosion, Trajectories) across worker processes."""
        self.sim_mode = str(mode)
        if isinstance(self.base_params, dict):
            self.base_params["sim_mode"] = self.sim_mode
        for p in getattr(self, '_all_payloads_ref', []):
            if isinstance(p.get("base_params"), dict):
                p["base_params"]["sim_mode"] = self.sim_mode
        smf = getattr(self, 'sim_mode_file', None)
        if smf:
            try:
                with open(smf, "w", encoding="utf-8") as f:
                    f.write(self.sim_mode)
            except Exception:
                pass

    def set_processes(self, enable_see: bool, enable_cex: bool):
        """Synchronize Secondary e- emissions and charge exchange flags across worker processes."""
        self.enable_see = bool(enable_see)
        self.enable_cex = bool(enable_cex)
        if isinstance(self.base_params, dict):
            self.base_params["enable_see"] = self.enable_see
            self.base_params["enable_cex"] = self.enable_cex
        for p in getattr(self, '_all_payloads_ref', []):
            if isinstance(p.get("base_params"), dict):
                p["base_params"]["enable_see"] = self.enable_see
                p["base_params"]["enable_cex"] = self.enable_cex
        pf = getattr(self, 'processes_file', None)
        if pf:
            try:
                with open(pf, "w", encoding="utf-8") as f:
                    f.write(f"{int(self.enable_see)},{int(self.enable_cex)}")
            except Exception:
                pass

    def get_scenario_tracking_buffer(self, scen_num: int) -> list:
        """Read and return recorded particle kinematics list for scen_num."""
        if not hasattr(self, 'final_tracking_buffers'):
            self.final_tracking_buffers = {}
        tf = getattr(self, 'trk_files', {}).get(scen_num)
        if tf and os.path.exists(tf):
            buf = read_scenario_state(tf)
            if isinstance(buf, list):
                self.final_tracking_buffers[scen_num] = buf
                return buf
        return list(self.final_tracking_buffers.get(scen_num, []))

    def get_all_tracking_buffers(self) -> dict:
        """Read all available scenario tracking buffers and return mapping {scen_num: list}."""
        if not hasattr(self, 'final_tracking_buffers'):
            self.final_tracking_buffers = {}
        for scen_num, tf in getattr(self, 'trk_files', {}).items():
            if tf and os.path.exists(tf):
                buf = read_scenario_state(tf)
                if isinstance(buf, list):
                    self.final_tracking_buffers[scen_num] = buf
        return dict(self.final_tracking_buffers)

    def get_status_dict(self):
        """Thread-safe snapshot dictionary of all scenario execution states."""
        res = dict(getattr(self, 'final_status_dict', {}))
        paused_set = getattr(self, '_paused_scenarios', set())
        held_set = getattr(self, 'held_scenarios', set())
        for scen_num, sf in getattr(self, 'state_files', {}).items():
            st = read_scenario_state(sf)
            if st is not None:
                if st.get("status") != "done":
                    if scen_num in held_set:
                        st["status"] = "action_required"
                        st["held"] = True
                    elif scen_num in paused_set:
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
        for tf in getattr(self, 'trk_files', {}).values():
            try:
                if os.path.exists(tf):
                    os.remove(tf)
            except Exception:
                pass
        if hasattr(self, 'active_view_file') and os.path.exists(self.active_view_file):
            try:
                os.remove(self.active_view_file)
            except Exception:
                pass
        if hasattr(self, 'track_ptcls_file') and os.path.exists(self.track_ptcls_file):
            try:
                os.remove(self.track_ptcls_file)
            except Exception:
                pass
        if hasattr(self, 'sim_mode_file') and os.path.exists(self.sim_mode_file):
            try:
                os.remove(self.sim_mode_file)
            except Exception:
                pass
        if hasattr(self, 'processes_file') and os.path.exists(self.processes_file):
            try:
                os.remove(self.processes_file)
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
            "window": self.sweep_config.get("window", 100),
            "threshold": self.sweep_config.get("threshold", 1.0),
            "max_iterations": self.sweep_config.get("max_iterations", 5000),
            "min_iterations": self.sweep_config.get("min_iterations", 500),
        }

        payloads = []
        for idx, val in enumerate(values):
            scen_idx = idx + 1
            if self.remaining_indices is not None and scen_idx not in self.remaining_indices:
                continue
            payloads.append({
                "scenario_idx": scen_idx,
                "param_id": param_id,
                "parameter_value": float(val),
                "base_params": copy.deepcopy(self.base_params),
                "sweep_config": {
                    "auto_update_upstream_gap": bool(self.sweep_config.get("auto_update_upstream_gap", False)),
                    "upstream_gap_coupling": self.sweep_config.get("upstream_gap_coupling"),
                },
                "steady_config": steady_cfg,
                "cancel_file": self.cancel_file,
                "pause_file": self.pause_files.get(scen_idx),
                "state_file": self.state_files.get(scen_idx),
                "trk_file": self.trk_files.get(scen_idx),
                "track_ptcls_file": getattr(self, 'track_ptcls_file', None),
                "sim_mode_file": getattr(self, 'sim_mode_file', None),
                "processes_file": getattr(self, 'processes_file', None),
                "track_particles": bool(getattr(self, 'track_particles_enabled', False)),
                "active_view_file": self.active_view_file,
                "scatter_limit": getattr(self, 'scatter_limit', 2000),
            })
        self._all_payloads_ref = payloads

        results = list(self.existing_results)
        completed_count = len(results)
        from physics.active_clock import ActiveDurationTracker
        sweep_tracker = ActiveDurationTracker(max_gap=None)
        sweep_tracker.start()
        sweep_t0 = time.perf_counter()

        if len(payloads) == 0:
            _active_workers.discard(self)
            results.sort(key=lambda r: r.get("scenario_idx", 0))
            self.sweep_finished.emit(results)
            return

        if completed_count > 0:
            self.progress_updated.emit(
                completed_count,
                total_count,
                f"Resuming parallel sweep with {self.n_cores} worker cores ({completed_count}/{total_count} already completed)..."
            )
        else:
            self.progress_updated.emit(0, total_count, f"Starting parallel sweep with {self.n_cores} worker cores...")

        if self._canceled:
            _active_workers.discard(self)
            return

        hw = detect_hardware_profile()
        usable_ram = hw.usable_ram_gb
        max_workers = min(self.n_cores, hw.max_cores_to_use)

        executor = ProcessPoolExecutor(max_workers=max_workers)
        self._executor = executor

        # Dynamic RAM-Aware Scenario Scheduler with Held Scenario Support
        pending_payloads = list(payloads)
        payload_ram = {p["scenario_idx"]: estimate_payload_ram_gb(p) for p in payloads}
        running_futures = {}  # future -> (scenario_idx, estimated_ram)
        current_ram_alloc = 0.0

        def submit_next_if_possible():
            nonlocal current_ram_alloc
            while pending_payloads and len(running_futures) < max_workers:
                chosen_idx = -1
                for idx_cand, cand in enumerate(pending_payloads):
                    if cand["scenario_idx"] not in self.held_scenarios:
                        chosen_idx = idx_cand
                        break

                if chosen_idx < 0:
                    # All remaining pending payloads are currently held
                    break

                next_p = pending_payloads[chosen_idx]
                scen_ram = payload_ram.get(next_p["scenario_idx"], 1.5)
                # Allow launch if memory fits, or if no workers are running (guarantee forward progress)
                if len(running_futures) == 0 or (current_ram_alloc + scen_ram <= usable_ram):
                    p = pending_payloads.pop(chosen_idx)
                    current_ram_alloc += scen_ram
                    fut = executor.submit(run_single_sweep_scenario_worker, p)
                    running_futures[fut] = (p["scenario_idx"], scen_ram)
                else:
                    # Wait for running workers to finish to free memory
                    break

            # Advise user when concurrency is throttled for heavy memory scenarios
            running_count = len(running_futures)
            active_scens = [s[0] for s in running_futures.values()]
            if running_count <= 2 and max_workers > 2 and current_ram_alloc >= usable_ram * 0.35:
                throttle_msg = (
                    f"⚠️ Memory Guard: Throttled to {running_count} concurrent worker(s) for high-RAM scenario(s) {active_scens} "
                    f"({current_ram_alloc:.1f} GB allocated) to prevent RAM saturation."
                )
                self.progress_updated.emit(completed_count, total_count, throttle_msg)

        try:
            submit_next_if_possible()

            last_prompted_held = None
            while running_futures or pending_payloads:
                if self._canceled:
                    break

                submit_next_if_possible()

                done_futures = [f for f in running_futures.keys() if f.done()]
                if not done_futures:
                    if running_futures:
                        try:
                            done_set, _ = wait(running_futures.keys(), return_when=FIRST_COMPLETED, timeout=0.2)
                            done_futures = list(done_set)
                        except Exception:
                            time.sleep(0.05)
                    else:
                        # All running workers completed, but pending payloads are held awaiting user action
                        if pending_payloads:
                            first_held = pending_payloads[0]["scenario_idx"]
                            if first_held != last_prompted_held:
                                last_prompted_held = first_held
                                self.scenario_action_needed.emit(first_held, {})
                                self.progress_updated.emit(
                                    completed_count, total_count,
                                    f"⏸ Sweep paused: Scenario {first_held} requires user decision (Downstream Plume Advisory)."
                                )
                            time.sleep(0.15)
                            continue
                        else:
                            break

                for future in done_futures:
                    if future not in running_futures:
                        continue
                    scen_idx, scen_ram = running_futures.pop(future)
                    current_ram_alloc = max(0.0, current_ram_alloc - scen_ram)

                    try:
                        row = future.result()
                        results.append(row)
                        completed_count += 1
                        sweep_tracker.tick()
                        s_idx = row.get("scenario_idx")
                        if s_idx is not None:
                            sf = self.state_files.get(s_idx)
                            if sf and os.path.exists(sf):
                                st = read_scenario_state(sf)
                                if st is not None:
                                    self.final_status_dict[s_idx] = st
                            if s_idx in self.final_status_dict:
                                self.final_status_dict[s_idx]["status"] = "done"
                                self.final_status_dict[s_idx]["iter"] = row.get("steady_iterations", 0)
                                self.final_status_dict[s_idx]["steady_iterations"] = row.get("steady_iterations", 0)
                                self.final_status_dict[s_idx]["computation_time_s"] = row.get("computation_time_s", 0.0)
                                self.final_status_dict[s_idx]["div"] = row.get("divergence_deg", row.get("div", 0.0))
                                self.final_status_dict[s_idx]["div_mid"] = row.get("divergence_2nd_plane_deg", row.get("div_mid", 0.0))
                                self.final_status_dict[s_idx]["saddle_potential_V"] = row.get("saddle_potential_V", 0.0)
                                self.final_status_dict[s_idx]["exit_ion_current_A"] = row.get("exit_ion_current_A", 0.0)
                                self.final_status_dict[s_idx]["transparency"] = row.get("transparency", 0.0)
                                self.final_status_dict[s_idx]["perveance_effective"] = row.get("perveance_effective", row.get("Peffective", 0.0))
                                self.final_status_dict[s_idx]["t_sim_us"] = row.get("t_sim_us", 0.0)
                                if "T_grids_K" in row:
                                    self.final_status_dict[s_idx]["T_grids"] = list(row["T_grids_K"])
                                    self.final_status_dict[s_idx]["T_grids_K"] = list(row["T_grids_K"])
                                if "T_grids_C" in row:
                                    self.final_status_dict[s_idx]["T_grids_C"] = list(row["T_grids_C"])

                                # Synchronize history lists between row and final_status_dict
                                hist_keys = (
                                    "iter_history", "time_history", "div_history", "div_mid_history",
                                    "ebs_history", "transparency_history", "ion_current_exit_history",
                                    "ion_current_exit_avg_history", "transmitted_ions_history",
                                    "active_cells_history", "low_ppc_cells_history", "T_histories",
                                    "plume_history_iterations", "plume_history_dx", "plume_dx_history",
                                    "zone_configs",
                                )
                                for hk in hist_keys:
                                    if hk in self.final_status_dict[s_idx] and hk not in row:
                                        row[hk] = self.final_status_dict[s_idx][hk]
                                    elif hk in row and (hk not in self.final_status_dict[s_idx] or not self.final_status_dict[s_idx][hk]):
                                        self.final_status_dict[s_idx][hk] = row[hk]
                            self.get_scenario_tracking_buffer(s_idx)
                        val_str = f"{row['parameter_value']:.4g}"
                        c_time = row.get("computation_time_s", 0.0)
                        time_info = f" in {c_time:.1f}s" if (isinstance(c_time, (int, float)) and c_time > 0) else ""
                        self.scenario_completed.emit(row, completed_count, total_count)
                        self.progress_updated.emit(
                            completed_count, total_count,
                            f"Completed Scenario {row['scenario_idx']}/{total_count} (value={val_str}{time_info})"
                        )
                    except Exception as exc:
                        failed_idx = scen_idx
                        if failed_idx and failed_idx in self.final_status_dict:
                            self.final_status_dict[failed_idx]["status"] = "stopped"
                        if not self._canceled:
                            self._canceled = True
                            self.sweep_error.emit(f"Scenario {failed_idx} error: {exc}")
                        break

                submit_next_if_possible()

        except Exception as e:
            if not self._canceled:
                self._canceled = True
                self.sweep_error.emit(f"Process pool error: {e}")
            return
        finally:
            self.total_computation_time_s = sweep_tracker.get_duration()
            if self.sweep_config is not None:
                self.sweep_config["total_computation_time_s"] = float(self.total_computation_time_s)
            _active_workers.discard(self)
            try:
                for scen_num, sf in getattr(self, 'state_files', {}).items():
                    st = read_scenario_state(sf)
                    if st is not None:
                        self.final_status_dict[scen_num] = st
                self.get_all_tracking_buffers()
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
