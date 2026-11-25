import os
import json
import pytest
import numpy as np

from physics.simulator import DigitalTwinSimulator as Simulator
from physics.mesh_utils import estimate_stability_optimized_mesh_zones
from gui.dialogs import AdvancedSettingsDialog
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
            "min_iterations": 100,
            "coarsen_presheath_bulk": 1.5,
            "refine_meniscus_zone": 0.8,
            "coarsen_far_plume": 4.0,
        }
    }


def test_estimate_stability_optimized_mesh_zones():
    """Verify zone construction around detected meniscus and preservation of optics bounds."""
    sim = Simulator()
    params = get_base_params()
    sim.build_domain(params)

    # Fake a potential field where sheath forms at x ~ 0.4 mm
    # Plasma potential is ~1020 V, sheath is ~1018 V
    sim.V.fill(1020.0)
    # Beyond x = 0.4, potential drops
    idx_drop = np.where(sim.x_coords >= 0.4)[0]
    if len(idx_drop) > 0:
        sim.V[:, idx_drop] = 500.0

    remesh_cfg = {
        "coarsen_presheath_bulk": 1.8,
        "refine_meniscus_zone": 0.75,
        "coarsen_far_plume": 4.0
    }

    zones = estimate_stability_optimized_mesh_zones(
        sim=sim,
        grids=params["grids"],
        upstream_gap_mm=sim.upstream_gap_mm,
        Lx=sim.Lx,
        dx0=sim.dx,
        remesh_cfg=remesh_cfg
    )

    zone_names = [z['name'] for z in zones]
    assert any('Presheath_Deep' in z for z in zone_names)
    assert any('Presheath_Meniscus' in z for z in zone_names)
    assert any('Grid_1_barrel' in z for z in zone_names)
    assert any('Gap_1_2' in z for z in zone_names)

    # Check factors
    deep_zone = [z for z in zones if z['name'] == 'Presheath_Deep'][0]
    meniscus_zone = [z for z in zones if z['name'] == 'Presheath_Meniscus'][0]
    assert deep_zone['factor'] == 1.8
    assert meniscus_zone['factor'] == 0.75


def test_execute_stability_remesh_preserves_particles_and_surface():
    """Verify execute_stability_remesh preserves particle coordinates, velocities, and subgrid surface states."""
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

    # Record old grid dimensions
    old_nx = sim.nx
    old_ny = sim.ny

    # Execute adaptive remesh
    remesh_cfg = {
        "coarsen_presheath_bulk": 1.5,
        "refine_meniscus_zone": 0.8,
        "coarsen_far_plume": 4.0
    }
    success = sim.execute_stability_remesh(params=params, remesh_cfg=remesh_cfg)
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

    # Second call must be a no-op (remesh triggers only once per steady state milestone)
    second_call = sim.execute_stability_remesh(params=params, remesh_cfg=remesh_cfg)
    assert second_call is False


def test_stability_triggered_remesh_in_step():
    """Verify that sim.step() automatically triggers adaptive remesh upon multi-metric stationarity."""
    sim = Simulator()
    params = get_base_params()
    params["adaptive_remesh_settings"]["min_iterations"] = 105
    sim.build_domain(params)

    sim.adaptive_remesh_enabled = True
    sim.remesh_min_iters = 105
    sim.remesh_window = 20
    sim.remesh_threshold = 1.0

    # Execute one step to get realistic physical baseline values
    sim.step(params)
    pot_val = sim.potential_history[-1]
    curr_val = sim.ion_current_history[-1]
    e_val = sim.energy_history_tot[-1] if sim.energy_history_tot else 1.0
    q_val = sim.charge_history_q_net[-1] if sim.charge_history_q_net else 1e-12

    # Prefill stationary histories with small fluctuations (< 0.1% variation)
    sim.energy_history_tot = [e_val * (1.0 + 0.0001 * np.sin(i)) for i in range(60)]
    sim.charge_history_q_net = [q_val * (1.0 + 0.0001 * np.cos(i)) for i in range(60)]
    sim.potential_history = [pot_val * (1.0 + 0.0001 * np.sin(i)) for i in range(60)]
    sim.ion_current_history = [curr_val] * 60

    # Advance to iteration 100 (< min_iters 105)
    sim.iteration = 100
    sim.step(params)
    assert not sim._has_stability_remeshed

    # Set iteration to 104 so the next step reaches 105 (multiple of 5, >= min_iters 105)
    sim.iteration = 104
    sim.step(params)
    assert sim._has_stability_remeshed
    assert sim._stability_remesh_iter == 105


def test_advanced_settings_dialog_remesh_controls():
    """Verify that AdvancedSettingsDialog exposes and persists adaptive remeshing controls."""
    current_params = {
        "V_plasma_offset": 20.0,
        "m_e_ratio": 1000.0,
        "adaptive_remesh_settings": {
            "enabled": True,
            "min_iterations": 600,
            "coarsen_presheath_bulk": 1.7,
            "refine_meniscus_zone": 0.75,
            "coarsen_far_plume": 4.5,
        }
    }
    dlg = AdvancedSettingsDialog(current_params)
    assert dlg.chk_adaptive_remesh.isChecked() is True
    assert dlg.spin_remesh_min_iters.value() == 600
    assert abs(dlg.spin_remesh_coarsen_presheath.value() - 1.7) < 1e-3
    assert abs(dlg.spin_remesh_refine_meniscus.value() - 0.75) < 1e-3
    assert abs(dlg.spin_remesh_coarsen_plume.value() - 4.5) < 1e-3

    # Toggle values
    dlg.chk_adaptive_remesh.setChecked(False)
    dlg.spin_remesh_min_iters.setValue(800)
    dlg.spin_remesh_coarsen_presheath.setValue(2.0)

    vals = dlg.get_values()
    assert vals["adaptive_remesh_enabled"] is False
    assert vals["adaptive_remesh_settings"]["enabled"] is False
    assert vals["adaptive_remesh_settings"]["min_iterations"] == 800
    assert vals["adaptive_remesh_settings"]["coarsen_presheath_bulk"] == 2.0
