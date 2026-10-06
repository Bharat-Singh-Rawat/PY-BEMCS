"""
DigitalTwinApp main window composed from modular UI and logic mixins.
"""
import os
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer

from physics_engine import DigitalTwinSimulator
from ..config_io import _config_path, load_json_config
from ..workers.simulation_worker import SimulationWorker
from .ui_builder import UIBuilderMixin
from .config_mixin import ConfigMixin
from .dialogs_mixin import DialogsMixin
from .sim_control import SimControlMixin
from .rendering import RenderingMixin
from .export import ExportMixin


class DigitalTwinApp(
    QMainWindow,
    UIBuilderMixin,
    ConfigMixin,
    DialogsMixin,
    SimControlMixin,
    RenderingMixin,
    ExportMixin
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

        self.iter_history = []
        self.ebs_history  = []
        self.div_history  = []
        self.div_mid_history = []
        self._div_curve_visible = {'grid': True, 'mid': True}
        self._div_legend_map = {}
        self.div_method = "95%"
        self.div_percentile = 95.0
        self.time_history = []
        self.transparency_history  = []
        self.transparency3_history = []
        self.ion_current_exit_history = []
        self.ion_current_exit_avg_history = []
        self.transmitted_ions_history = []
        self.active_cells_history  = []
        self.low_ppc_cells_history = []
        self.T_histories = {}
        self.recorded_frames = []
        self.tracking_buffer = []
        self.iedf_window = None
        self.ppc_window  = None
        self.phys_window = None
        self.charge_window = None
        self.perf_window = None
        self.cs_viewer_window = None
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

        self.inputs = {}

        self.setup_menu_bar()
        self.setup_ui()
        if self.config is not None:
            self.apply_config(self.config, config_name=self.current_config_name)
        else:
            self._apply_defaults()

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
        self._sim_worker.step_done.connect(self._on_step_result)
        self._sim_worker.sim_finished.connect(self._on_worker_finished)
