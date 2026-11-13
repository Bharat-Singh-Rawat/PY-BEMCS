"""
DigitalTwinApp main window composed from modular UI and logic mixins.
"""
import os
import sys
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import QTimer

from physics_engine import DigitalTwinSimulator
from physics.active_clock import ActiveDurationTracker
from ..config_io import _config_path, load_json_config
from ..workers.simulation_worker import SimulationWorker
from .ui_builder import UIBuilderMixin
from .config_mixin import ConfigMixin
from .dialogs_mixin import DialogsMixin
from .sim_control import SimControlMixin
from .rendering import RenderingMixin
from .export import ExportMixin
from .sweep_mixin import SweepMixin


class DigitalTwinApp(
    QMainWindow,
    UIBuilderMixin,
    ConfigMixin,
    DialogsMixin,
    SimControlMixin,
    RenderingMixin,
    ExportMixin,
    SweepMixin
):
    """Main window for PY-BEMCS orchestrating the interactive GUI, simulation loop, real-time plotting, and diagnostic windows."""

    def update_config_title(self, config_name=None):
        if config_name:
            self.current_config_name = config_name
        cfg_name = getattr(self, "current_config_name", "config.json")
        self.setWindowTitle(f"PY-BEMCS (Multi-Grid & Co-Extraction) - [{cfg_name}]")
        if hasattr(self, 'lbl_active_config'):
            self.lbl_active_config.setText(f"  Config: {cfg_name}  ")
        if hasattr(self, 'reload_action'):
            self.reload_action.setText(f"Reload {cfg_name}")

    def __init__(self):
        super().__init__()
        self.setGeometry(20, 30, 1500, 800)

        self.current_config_path = _config_path()
        self.current_config_name = os.path.basename(self.current_config_path) if self.current_config_path else "config.json"
        self.update_config_title()
        self.config = load_json_config(self.current_config_path)
        self.sim = DigitalTwinSimulator()
        self.sim_isRunning = False
        self._sim_run_id = 0
        self._pending_snap = None

        self.iter_history = []
        self.ebs_history  = []
        self.div_history  = []
        self.div_mid_history = []
        self._div_curve_visible = {'grid': True, 'mid': True}
        self._div_legend_map = {}
        self._div_live_mode = True
        self.div_method = "95%"
        self.div_percentile = 95.0
        self.div_plane_x = None
        self.ebs_xaxis_unit = "iterations"
        self.div_xaxis_unit = "iterations"
        self.xaxis_unit = "iterations"
        self.time_history = []
        self.transparency_history  = []
        self.transparency3_history = []
        self.ion_current_exit_history = []
        self.ion_current_exit_avg_history = []
        self.transmitted_ions_history = []
        self.active_cells_history  = []
        self.low_ppc_cells_history = []
        self.T_histories = {}
        self.export_selected_params = {
            'iteration': True,
            't_sim_s': True,
            'minpotential': True,
            'beamdivergence': True,
            'beamdivergence_mid': True,
            'transparency': True,
            'ion_current_exit': True,
            'ion_current_exit_avg': True,
            'transmitted_ions': True,
            'total_active_cells': True,
            'low_ppc_cells': True,
            'grid_temps': True,
        }
        self.export_selected_particle_params = {
            'time_s': True,
            'x_mm': True,
            'y_mm': True,
            'vx_ms': True,
            'vy_ms': True,
            'vz_ms': True,
            'energy_eV': True,
            'type': True,
        }
        self.recorded_frames = []
        self.tracking_buffer = []
        self.iedf_window = None
        self.ppc_window  = None
        self.density_window = None
        self.phys_window = None
        self.charge_window = None
        self.perf_window = None
        self.cs_viewer_window = None
        self.mesh_window = None
        self.field_window = None
        self.steady_state_window = None
        self.substep_window = None
        self.is_sweep_active = False
        self.sweep_mode_armed = False
        self._sweep_paused_current = False
        self.sweep_config = None
        self.sweep_results = []
        self.sweep_current_idx = 0
        self.sweep_scenario_start_iter = 0
        self.btn_resume_sweep = None
        self._built_params_snapshot = None
        self._grid_count_modified_since_build = False
        self._loading_config = False
        self.substep_history_iter = []
        self.substep_history_ion = []
        self.substep_history_e = []
        self.sim.enable_perf_monitor(log_every=50)

        self.cbar_temp = None
        self.cbar_energy = None

        self.grid_widgets = []

        self.beam_mass_amu     = None
        self.beam_charge_state = None
        self.cs_store   = {}
        self.mat_name   = None
        self.mat_props  = {}
        self.adv_params = {}
        self.upstream_gap_settings = None

        self.inputs = {}

        self.setup_menu_bar()
        self.setup_ui()
        if self.config is not None:
            self.apply_config(self.config, config_name=self.current_config_name)
        else:
            self._apply_defaults()

        self._active_sim_timer = ActiveDurationTracker(max_gap=2.0)
        self.sim_wall_elapsed = 0.0
        self.sim_wall_start_time = None
        self.duration_clock_timer = QTimer(self)
        self.duration_clock_timer.timeout.connect(self.update_simulation_timer)
        self.duration_clock_timer.start(100)

        # Strategy 1: drop-on-busy guard and adaptive GUI timing
        self._gui_busy = False
        self._pending_snap = None
        self._last_gui_ms = 1.0

        # Strategy 3: diagnostic window throttle
        self._DIAG_MIN_INTERVAL_MS = 500
        self._last_diag_t = 0.0

        # Simulation worker (runs sim.step in background thread)
        self._sim_worker = SimulationWorker(self.sim, parent=self)
        self.scatter_limit = getattr(self._sim_worker, '_scatter_limit', SimulationWorker.MAX_SCATTER)
        self._sim_worker.step_done.connect(self._on_step_result)
        self._sim_worker.sim_finished.connect(self._on_worker_finished)

    def is_simulation_running(self):
        """Check if any simulation is currently running (regular beam simulation or parametric sweep)."""
        if getattr(self, 'sim_isRunning', False):
            return True
        if getattr(self, 'is_sweep_active', False) and not getattr(self, '_sweep_stopped', False) and not getattr(self, '_sweep_paused_current', False):
            return True
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            if self.parallel_sweep_worker.isRunning() and not getattr(self, '_sweep_stopped', False):
                return True
        if hasattr(self, '_sim_worker') and self._sim_worker is not None:
            if self._sim_worker.isRunning() and getattr(self._sim_worker, 'running', False):
                return True
        return False

    def _shutdown_background_tasks(self):
        """Cleanly and forcibly terminate all background worker processes, threads, and timers."""
        # 1. Cancel and kill parallel sweep workers and child processes
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            worker = self.parallel_sweep_worker
            self.parallel_sweep_worker = None
            try:
                worker.cancel()
                worker.wait(2000)
            except Exception:
                pass

        if hasattr(self, 'parallel_sweep_dlg') and self.parallel_sweep_dlg is not None:
            dlg = self.parallel_sweep_dlg
            self.parallel_sweep_dlg = None
            try:
                dlg.reject()
            except Exception:
                pass

        self.is_sweep_active = False

        # 2. Stop regular simulation worker thread cooperatively without dangerous terminate()
        if hasattr(self, '_sim_worker') and self._sim_worker is not None:
            try:
                self._sim_worker.stop(wait_ms=3000)
            except Exception:
                pass
        self.sim_isRunning = False

        # 3. Stop timers
        if hasattr(self, 'sweep_monitor_timer') and self.sweep_monitor_timer is not None:
            try:
                self.sweep_monitor_timer.stop()
            except Exception:
                pass
        if hasattr(self, 'duration_clock_timer') and self.duration_clock_timer is not None:
            try:
                self.duration_clock_timer.stop()
            except Exception:
                pass
        if hasattr(self, 'timer') and self.timer is not None:
            try:
                self.timer.stop()
            except Exception:
                pass

        # 4. Close auxiliary diagnostic windows
        for win_attr in (
            'density_window', 'ebs_window', 'diag_window', 'iedf_window',
            'ppc_window', 'phys_window', 'charge_window', 'perf_window',
            'cs_viewer_window', 'mesh_window', 'field_window',
            'steady_state_window', 'substep_window', 'sweep_results_dialog'
        ):
            w = getattr(self, win_attr, None)
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass

    def closeEvent(self, event):
        """Prompt confirmation if simulation/sweep is running and prompt to save data after a simulation has run."""
        if self.is_simulation_running():
            reply = QMessageBox.question(
                self,
                "Simulation Running",
                "Attention: simulation is running. Are you sure you want to close the tab?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        else:
            # Check if unsaved simulation or sweep data exists after a simulation has run
            has_sim_data = hasattr(self, 'iter_history') and len(self.iter_history) > 0
            has_sweep_data = bool(getattr(self, 'sweep_results', [])) or bool(getattr(self, '_sweep_status_dict', {}))
            is_saved = getattr(self, '_data_saved', False)

            if (has_sim_data or has_sweep_data) and not is_saved:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Save Simulation Data")
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setText("Attention: all unsaved data will be lost. Would you like to save data before closing?")
                btn_save = msg_box.addButton("Save", QMessageBox.AcceptRole)
                btn_dont_save = msg_box.addButton("Don't Save", QMessageBox.DestructiveRole)
                btn_cancel = msg_box.addButton("Cancel", QMessageBox.RejectRole)
                msg_box.setDefaultButton(btn_save)
                msg_box.setEscapeButton(btn_cancel)
                msg_box.exec_()

                clicked = msg_box.clickedButton()
                if clicked == btn_cancel or clicked is None:
                    event.ignore()
                    return
                elif clicked == btn_save:
                    if has_sweep_data and hasattr(self, '_compile_and_show_interim_sweep_results'):
                        self._compile_and_show_interim_sweep_results()
                    elif hasattr(self, 'export_csv'):
                        saved = self.export_csv()
                        if saved is False:
                            event.ignore()
                            return

        self._shutdown_background_tasks()
        event.accept()

    def nativeEvent(self, eventType, message):
        """Handle Windows power broadcast events (suspend/resume) to ensure
        the simulation active duration clock freezes cleanly across sleep."""
        if eventType == b"windows_generic_MSG" and sys.platform == "win32":
            try:
                import ctypes
                import ctypes.wintypes
                msg = ctypes.wintypes.MSG.from_address(message.__int__())
                if msg.message == 0x0218:  # WM_POWERBROADCAST
                    # PBT_APMSUSPEND = 0x0004
                    if msg.wParam == 0x0004:
                        if hasattr(self, '_active_sim_timer') and self._active_sim_timer is not None:
                            self._active_sim_timer.pause()
                    # PBT_APMRESUMEAUTOMATIC = 0x0012, PBT_APMRESUMESUSPEND = 0x0007
                    elif msg.wParam in (0x0007, 0x0012):
                        if self.is_simulation_running() and hasattr(self, '_active_sim_timer') and self._active_sim_timer is not None:
                            self._active_sim_timer.start()
            except Exception:
                pass
        return super().nativeEvent(eventType, message)
