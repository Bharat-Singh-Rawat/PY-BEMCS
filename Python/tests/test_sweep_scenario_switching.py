import numpy as np
import pytest
from unittest.mock import MagicMock
from gui.app.sweep_mixin import SweepMixin


class MockAxes:
    def __init__(self):
        self.xlim = (0, 100)
        self.ylim = (0, 45)
        self.collections = []
        self.title_text = ""
        self.bbox = MagicMock()

    def set_xlim(self, x0, x1):
        self.xlim = (x0, x1)

    def set_ylim(self, y0, y1):
        self.ylim = (y0, y1)

    def set_title(self, t, **kwargs):
        self.title_text = t

    def clear(self):
        pass

    def contourf(self, *args, **kwargs):
        return MagicMock()

    def scatter(self, *args, **kwargs):
        return MagicMock()

    def draw_artist(self, artist):
        pass


class MockLine:
    def __init__(self):
        self.xdata = []
        self.ydata = []

    def set_data(self, x, y):
        self.xdata = list(x)
        self.ydata = list(y)


class MockLabel:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = str(text)


class MockSim:
    def __init__(self):
        self.Lx = 3.0
        self.Ly = 3.0
        self.nx = 20
        self.ny = 20
        self.dx = 0.15
        self.dy = 0.15
        self.dt = 5e-10
        self.iteration = 0
        self.energy_history_tot = []
        self.charge_history_q_net = []
        self.energy_history_fe = []
        self.potential_history = []
        self.ion_current_history = []
        self.exit_ion_current_history = []
        self.damage_map = np.zeros((20, 20), dtype=np.float32)
        self.eroded_depth = np.zeros((20, 20), dtype=np.float32)
        self.mask_grids = [np.zeros((20, 20), dtype=bool)]

    def get_groove_profile(self, groove_idx, thresh=0.05, face="downstream"):
        y_mm = np.linspace(0, 3.0, 20)
        # return depth based on damage_map
        depth_um = np.ones(20) * (np.max(self.damage_map) if self.damage_map is not None else 0.0)
        return y_mm, depth_um


class DummyApp(SweepMixin):
    def __init__(self):
        self.sweep_config = {
            "param_id": "n0_plasma",
            "values": [1e17, 2e17],
            "param_unit": "m^-3",
        }
        self.sweep_results = []
        self._sweep_status_dict = {}
        self._sweep_tracking_buffers = {}
        self._seq_scenario_histories = {}
        self.sim = MockSim()
        self.sim_isRunning = False
        self.is_sweep_active = False

        self.ax_live = MockAxes()
        self.ax_div = MockAxes()
        self.ax_ebs = MockAxes()
        self.ax_grooves = [MockAxes()]
        self.line_div = MockLine()
        self.line_div_mid = MockLine()
        self.line_ebs = MockLine()
        self.line_grooves = [MockLine()]

        self.scat_prim = MagicMock()
        self.scat_cex = MagicMock()
        self.scat_elec = MagicMock()
        self.scat_bound = MagicMock()

        self.lblTime = MockLabel()
        self.lblTransparency = MockLabel()
        self.lbl_top_sim_time = MockLabel()
        self.lblPerfPtcls = MockLabel()
        self.inputs = {"Thresh": MagicMock(value=lambda: 0.05)}
        self.canvas = MagicMock()


def test_scenario_switching_updates_plots_and_telemetry():
    """Verify that switching scenarios updates divergence, ion current, and groove depth when stopped."""
    app = DummyApp()

    # Scenario 1 data
    s1_damage = np.ones((20, 20), dtype=np.float32) * 5.0
    app._sweep_status_dict[1] = {
        "status": "done",
        "iter": 500,
        "div": 12.0,
        "div_mid": 10.0,
        "exit_ion_current_A": 0.05,
        "transparency": 0.85,
        "iter_history": [100, 200, 300, 400, 500],
        "div_history": [15.0, 14.0, 13.0, 12.5, 12.0],
        "div_mid_history": [12.0, 11.5, 11.0, 10.5, 10.0],
        "ebs_history": [-20, -21, -22, -23, -24],
        "damage_map": s1_damage,
        "eroded_depth": s1_damage * 0.1,
        "mask_grids": [np.zeros((20, 20), dtype=bool)],
    }

    # Scenario 2 data
    s2_damage = np.ones((20, 20), dtype=np.float32) * 25.0
    app._sweep_status_dict[2] = {
        "status": "done",
        "iter": 1200,
        "div": 24.0,
        "div_mid": 20.0,
        "exit_ion_current_A": 0.15,
        "transparency": 0.70,
        "iter_history": [200, 400, 600, 800, 1000, 1200],
        "div_history": [30.0, 28.0, 26.0, 25.0, 24.5, 24.0],
        "div_mid_history": [25.0, 23.0, 22.0, 21.0, 20.5, 20.0],
        "ebs_history": [-10, -12, -14, -15, -16, -18],
        "damage_map": s2_damage,
        "eroded_depth": s2_damage * 0.1,
        "mask_grids": [np.zeros((20, 20), dtype=bool)],
    }

    # Render Scenario 1
    app._render_sweep_scenario_snapshot(1)
    assert app.line_div.ydata[-1] == 12.0
    assert "Divergence Exit: 12.00°" in app.lblTransparency.text
    assert "Exit I_ion: 50.000 mA" in app.lblTransparency.text
    assert app.line_grooves[0].ydata[0] == 5.0

    # Render Scenario 2
    app._render_sweep_scenario_snapshot(2)
    assert app.line_div.ydata[-1] == 24.0
    assert "Divergence Exit: 24.00°" in app.lblTransparency.text
    assert "Exit I_ion: 150.000 mA" in app.lblTransparency.text
    assert app.line_grooves[0].ydata[0] == 25.0


def test_steady_state_window_sync_on_scenario_switch():
    """Verify that steady_state_window receives the selected scenario's ion current history."""
    app = DummyApp()
    app.steady_state_window = MagicMock()
    app.steady_state_window.isVisible.return_value = True
    app.steady_state_window.spin_window.value.return_value = 50

    info_scen1 = {
        "iter": 500,
        "energy_history_tot": [1e-3, 1.01e-3],
        "charge_history_q_net": [1e-9, 1.01e-9],
        "ion_current_history": [0.01, 0.0101],
        "ebs_history": [-20.0, -20.1],
    }

    app._update_steady_state_window_for_scenario(1, info_scen1)
    assert app.sim.ion_current_history == [0.01, 0.0101]
    assert app.sim.charge_history_q_net == [1e-9, 1.01e-9]

    info_scen2 = {
        "iter": 800,
        "energy_history_tot": [2e-3, 2.05e-3],
        "charge_history_q_net": [2e-9, 2.08e-9],
        "ion_current_history": [0.05, 0.052],
        "ebs_history": [-15.0, -15.2],
    }

    app._update_steady_state_window_for_scenario(2, info_scen2)
    assert app.sim.ion_current_history == [0.05, 0.052]
    assert app.sim.charge_history_q_net == [2e-9, 2.08e-9]
    assert app.steady_state_window.update_plot.call_count == 2
