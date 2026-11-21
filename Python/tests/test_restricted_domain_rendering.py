import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication

from physics.simulator import DigitalTwinSimulator
from gui.app.main_window import DigitalTwinApp


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_restricted_domain_axes_limits_and_contour(qapp):
    """Verify that viewing a restricted/truncated domain scenario resizes ax_live to match exact truncated Lx."""
    app = DigitalTwinApp()
    app.build_domain(force=True)

    # Base full domain Lx
    full_Lx = float(app.sim.Lx)
    assert full_Lx > 4.0

    # Configure a sweep with 2 scenarios
    app.sweep_config = {
        "param_id": "n0",
        "values": [1e18, 1e19],
        "param_label": "Neutral Gas Density",
        "param_unit": "m⁻³",
        "auto_update_upstream_gap": False,
    }
    app.sweep_mode_armed = True
    app.is_sweep_active = False

    # Scenario 2 is marked as restricted/truncated plume
    app._scenario_advisory_decisions = {2: "truncate"}
    truncated_Lx = 3.688

    # Mock scenario status info where scenario 2 has run with restricted plume
    info_scen2 = {
        "status": "done",
        "iter": 500,
        "parameter_value": 1e19,
        "Lx": truncated_Lx,
        "Ly": float(app.sim.Ly),
        "nx": 125,
        "ny": 118,
        "dx": 0.03,
        "dy": 0.03,
        "truncate_downstream_plume": True,
        "V": np.zeros((118, 125), dtype=np.float32),
        "prim_xy": np.array([[0.5, 0.2], [1.5, 0.3], [3.2, 0.4]], dtype=np.float32),
        "e_prim_eV": np.array([50.0, 500.0, 1100.0], dtype=np.float32),
        "cex_xy": np.empty((0, 2), dtype=np.float32),
        "e_cex_eV": np.empty(0, dtype=np.float32),
        "elec_xy": np.empty((0, 2), dtype=np.float32),
    }

    app._sweep_status_dict = {2: info_scen2}

    # Render snapshot for Scenario 2
    app._render_sweep_scenario_snapshot(2)

    # ax_live xlim must strictly match the truncated Lx (3.688 mm), NOT full_Lx (>4.0 mm)
    xlim = app.ax_live.get_xlim()
    assert abs(xlim[1] - truncated_Lx) < 0.05, f"Expected ax_live xlim upper bound ~{truncated_Lx}, got {xlim[1]}"
    assert abs(xlim[0] - 0.0) < 1e-4

    # ax_temp and ax_dmg must also match truncated_Lx
    assert abs(app.ax_temp.get_xlim()[1] - truncated_Lx) < 0.05
    assert abs(app.ax_dmg.get_xlim()[1] - truncated_Lx) < 0.05

    # Background contour must match truncated domain shape
    assert app.contour_live is not None
    # sim.Lx must be synchronized
    assert abs(app.sim.Lx - truncated_Lx) < 0.05

    # Switch back to Scenario 1 (full domain)
    info_scen1 = {
        "status": "done",
        "iter": 500,
        "parameter_value": 1e18,
        "Lx": full_Lx,
        "Ly": float(app.sim.Ly),
        "nx": 178,
        "ny": 118,
        "dx": 0.03,
        "dy": 0.03,
        "truncate_downstream_plume": False,
        "V": np.zeros((118, 178), dtype=np.float32),
        "prim_xy": np.array([[0.5, 0.2], [1.5, 0.3], [4.5, 0.4]], dtype=np.float32),
        "e_prim_eV": np.array([50.0, 500.0, 1100.0], dtype=np.float32),
        "cex_xy": np.empty((0, 2), dtype=np.float32),
        "e_cex_eV": np.empty(0, dtype=np.float32),
        "elec_xy": np.empty((0, 2), dtype=np.float32),
    }
    app._sweep_status_dict[1] = info_scen1
    app._render_sweep_scenario_snapshot(1)

    xlim1 = app.ax_live.get_xlim()
    assert abs(xlim1[1] - full_Lx) < 0.05, f"Expected ax_live xlim upper bound ~{full_Lx}, got {xlim1[1]}"
    assert abs(app.sim.Lx - full_Lx) < 0.05

    app.close()
