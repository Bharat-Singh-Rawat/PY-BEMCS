"""
Background worker executing multi-scenario sweeps concurrently via ProcessPoolExecutor.
"""

import os
import atexit
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from PyQt5.QtCore import QThread, pyqtSignal

from physics.headless_sweep import run_single_sweep_scenario_worker

# Global registry of active worker instances for emergency atexit cleanup
_active_workers = set()


def _cleanup_all_active_workers():
    for w in list(_active_workers):
        try:
            w.cancel()
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

    def __init__(self, sweep_config: dict, base_params: dict, n_cores: int = None, parent=None):
        super().__init__(parent)
        self.sweep_config = sweep_config
        self.base_params = base_params
        total_cpus = os.cpu_count() or 16
        self.n_cores = int(n_cores) if n_cores and n_cores > 0 else max(1, total_cpus // 2)
        self._canceled = False
        self._executor = None
        self.cancel_file = os.path.join(tempfile.gettempdir(), f"pybemcs_sweep_{os.getpid()}_{id(self)}.cancel")
        if os.path.exists(self.cancel_file):
            try:
                os.remove(self.cancel_file)
            except Exception:
                pass

    def _kill_worker_processes(self):
        """Forcibly and immediately terminate all worker OS processes."""
        ex = self._executor
        if ex is not None:
            procs = list(getattr(ex, '_processes', {}).values())
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
        """Request immediate cancellation and terminate all running child processes."""
        self._canceled = True
        try:
            with open(self.cancel_file, "w") as f:
                f.write("cancelled")
        except Exception:
            pass

        self._kill_worker_processes()
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
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
            "min_iterations": self.sweep_config.get("min_iterations", 50),
        }

        payloads = []
        for idx, val in enumerate(values):
            payloads.append({
                "scenario_idx": idx + 1,
                "param_id": param_id,
                "parameter_value": float(val),
                "base_params": self.base_params,
                "steady_config": steady_cfg,
                "cancel_file": self.cancel_file,
            })

        results = []
        completed_count = 0

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
                    self.scenario_completed.emit(row, completed_count, total_count)
                    self.progress_updated.emit(
                        completed_count, total_count,
                        f"Completed Scenario {row['scenario_idx']}/{total_count} (value={val_str})"
                    )
                except Exception as exc:
                    if not self._canceled:
                        self.sweep_error.emit(f"Scenario error: {exc}")

        except Exception as e:
            if not self._canceled:
                self.sweep_error.emit(f"Process pool error: {e}")
            return
        finally:
            _active_workers.discard(self)
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

        if not self._canceled:
            # Sort results by scenario_idx so order matches input values
            results.sort(key=lambda r: r.get("scenario_idx", 0))
            self.sweep_finished.emit(results)
