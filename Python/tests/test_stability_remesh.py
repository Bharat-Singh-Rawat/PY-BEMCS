import os
import json
import pytest
import numpy as np

from physics.simulator import DigitalTwinSimulator as Simulator
from physics.mesh_utils import estimate_automatic_plume_mesh_zones, compute_plume_mean_dx
from gui.dialogs import AdvancedSettingsDialog
from diagnostics.mesh_window import PlumeMeshSizeGraphWidget, MeshWindow
from PyQt5.QtWidgets import QApplication

# Ensure QApplication exists for GUI dialog tests
app = QApplication.instance()
if app is None:
    app = QApplication([])


def get_base_params():
    return {
        "grids": [
            {"V": 1000.0, "t": 0.5, "gap": 1.0, "r": 0.8, "cham": 0.0},
            {"V": -200.0, "t": 0.5, "gap": 1.0, "r": 0.7, "cham": 0.0},
        ],
        "n0_plasma": 7e17,
        "Te_up": 4.0,
        "Ti": 0.5,
        "Tn": 300.0,
        "n0": 1e18,
        "Accel": 1.0,
        "Thresh": 1e6,
        "geometry": "half_hole",
        "coordinate_system": "cartesian",
        "sim_mode": "Thermal",
        "upstream_gap_mm": 0.6,
        "Lx": 4.0,
        "Ly": 1.5,
        "target_ppc": 40,
        "adaptive_remesh_enabled": True,
        "adaptive_remesh_settings": {
            "enabled": True,
            "interval_iterations": 150,
        }
    }


def test_estimate_automatic_plume_mesh_zones():
    """Verify zone construction preserves presheath and optics at baseline while adapting only plume."""
    sim = Simulator()
    params = get_base_params()
    sim.build_domain(params)

    zones = estimate_automatic_plume_mesh_zones(
        sim=sim,
        grids=params["grids"],
        upstream_gap_mm=sim.upstream_gap_mm,
        Lx=sim.Lx,
        dx0=sim.dx,
    )

    zone_names = [z['name'] for z in zones]
    # Presheath must be standard single baseline zone (factor = 1.0)
    assert 'Presheath' in zone_names
    assert not any('Presheath_Deep' in z for z in zone_names)
    assert not any('Presheath_Meniscus' in z for z in zone_names)

    # Optics must maintain baseline resolution
    assert any('Grid_1_barrel' in z for z in zone_names)
    assert any('Gap_1_2' in z for z in zone_names)
    assert any('Grid_2_barrel' in z for z in zone_names)

    # Plume must have near and far plume
    assert any('Far_Plume' in z for z in zone_names)

    presheath_zone = [z for z in zones if z['name'] == 'Presheath'][0]
    assert presheath_zone['factor'] == 1.0


def test_execute_plume_remesh_preserves_particles_and_surface():
    """Verify execute_plume_remesh preserves particle coordinates, velocities, and subgrid surface states."""
    sim = Simulator()
    params = get_base_params()
    sim.build_domain(params)

    # Add mock particles
    sim.num_p = 50
    sim.p_x[:50] = np.linspace(0.1, 2.5, 50)
    sim.p_y[:50] = np.linspace(0.05, 0.75, 50)
    sim.p_vx[:50] = 1.5e4
    sim.p_vy[:50] = 500.0
    sim.p_vz[:50] = 0.0
    sim.p_isCEX[:50] = False
    sim.p_isCEX[10] = True

    orig_p_x = sim.p_x[:50].copy()
    orig_p_y = sim.p_y[:50].copy()
    orig_p_vx = sim.p_vx[:50].copy()
    orig_isCEX = sim.p_isCEX[:50].copy()

    # Verify subgrid structural surface model is present
    assert hasattr(sim, 'subgrid_manager')
    assert sim.subgrid_manager is not None

    # Execute adaptive plume remesh
    success = sim.execute_plume_remesh(params=params, force=True)
    assert success is True
    assert sim._has_stability_remeshed is True

    # Particle states must be strictly preserved
    assert sim.num_p == 50
    np.testing.assert_allclose(sim.p_x[:50], orig_p_x, atol=1e-12)
    np.testing.assert_allclose(sim.p_y[:50], orig_p_y, atol=1e-12)
    np.testing.assert_allclose(sim.p_vx[:50], orig_p_vx, atol=1e-12)
    np.testing.assert_array_equal(sim.p_isCEX[:50], orig_isCEX)

    # Continuous structural surface manager must remain intact
    assert sim.subgrid_manager is not None

    # Scalar fields must not contain NaNs
    assert not np.isnan(sim.V).any()
    if hasattr(sim, 'Tmap') and sim.Tmap is not None:
        assert not np.isnan(sim.Tmap).any()

    # Plume cell size history must be tracked
    assert len(sim.plume_history_iterations) > 0
    assert len(sim.plume_history_dx) > 0


def test_periodic_plume_remesh_in_step():
    """Verify that sim.step() automatically triggers plume remeshing periodically every N iterations."""
    sim = Simulator()
    params = get_base_params()
    params["adaptive_remesh_settings"] = {
        "enabled": True,
        "interval_iterations": 10
    }
    sim.build_domain(params)
    sim.adaptive_remesh_enabled = True
    sim.remesh_interval_iterations = 10

    # Advance to iter 9
    sim.iteration = 9
    sim.step(params)
    # At iter 10, remesh triggers
    assert sim.iteration == 10
    assert len(sim.plume_history_iterations) > 0
    assert 10 in sim.plume_history_iterations


def test_advanced_settings_dialog_remesh_controls():
    """Verify that AdvancedSettingsDialog exposes single interval parameter defaulting to 150 and checked."""
    current_params = {
        "V_plasma_offset": 20.0,
        "m_e_ratio": 1000.0,
        "adaptive_remesh_settings": {
            "enabled": True,
            "interval_iterations": 150,
        }
    }
    dlg = AdvancedSettingsDialog(current_params)
    assert dlg.chk_adaptive_remesh.isChecked() is True
    assert dlg.spin_remesh_interval.value() == 150

    # Modify interval and toggle
    dlg.chk_adaptive_remesh.setChecked(False)
    dlg.spin_remesh_interval.setValue(200)

    vals = dlg.get_values()
    assert vals["adaptive_remesh_enabled"] is False
    assert vals["adaptive_remesh_settings"]["enabled"] is False
    assert vals["adaptive_remesh_settings"]["interval_iterations"] == 200


def test_plume_mesh_size_widget_and_window():
    """Verify that PlumeMeshSizeGraphWidget plots history without errors."""
    sim = Simulator()
    params = get_base_params()
    sim.build_domain(params)

    # Populate some plume history
    sim.plume_history_iterations = [0, 150, 300]
    sim.plume_history_dx = [0.04, 0.08, 0.12]
    sim.iteration = 300

    widget = PlumeMeshSizeGraphWidget()
    widget.update_plot(sim)

    assert "300" in widget.lbl_curr_iter.text()
    assert "0.1200" in widget.lbl_curr_dx.text()

    # Check MeshWindow tabs and dynamic live updating
    mesh_win = MeshWindow()
    assert mesh_win.tabs.count() >= 3
    mesh_win.update_plot(sim, params=params)

    # Verify Tab 1 displays updated mesh analytics
    assert "mm" in mesh_win.lbl_dx_nominal.text()
    assert "Total Mesh Cells" in mesh_win.lbl_grid_cells.text()

    # Trigger remesh and verify Tab 1 re-renders accurately on the adapted mesh
    sim.execute_plume_remesh(params=params, force=True)
    mesh_win.update_plot(sim, params=params)
    assert mesh_win.ax.get_xlim() == (0, sim.Lx)


def test_sweep_scenario_mesh_window_history():
    """Verify that MeshWindow correctly displays full multi-thousand iteration plume history in sweep scenarios."""
    from gui.app.sweep_mixin import SweepMixin

    class DummyApp(SweepMixin):
        def __init__(self, sim, mesh_win):
            self.sim = sim
            self.mesh_window = mesh_win
            self.substep_window = None
            self.plume_window = None
            self.sweep_config = {"param_id": "Te_up"}
            self._scenario_domain_cache = {}

    sim = Simulator()
    params = get_base_params()
    sim.build_domain(params)

    mesh_win = MeshWindow()
    app_dummy = DummyApp(sim, mesh_win)

    # Mock scenario info at 1500 iterations with remeshing every 150 iterations
    scenario_info = {
        "status": "done",
        "iter": 1500,
        "steady_iterations": 1500,
        "parameter_value": 4.0,
        "substep_history_iter": list(range(1, 1501)),
        "substep_history_ion": [1] * 1500,
        "substep_history_e": [1] * 1500,
        "plume_history_iterations": [0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500],
        "plume_history_dx": [0.04, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
        "zone_configs": sim.zone_configs,
    }

    # Simulate selecting scenario 1 in the sweep GUI
    app_dummy._update_mesh_window_for_scenario(1, scenario_info, d_data={"params": params}, force=True)

    assert sim.iteration == 1500
    assert len(sim.plume_history_iterations) == 11
    assert 1500 in sim.plume_history_iterations

    # Verify widget displays 1500 iterations
    assert "1500" in mesh_win.plume_mesh_widget.lbl_curr_iter.text()
    assert mesh_win.plume_mesh_widget.ax.get_xlim()[1] >= 1500
