"""
Unit tests for Machine-Aware Resource Management and Presheath Particle Budgeting.
"""
import os
import json
import pytest
import numpy as np

from physics.hardware import (
    HardwareProfile,
    detect_hardware_profile,
    estimate_mesh_memory_gb,
    estimate_payload_ram_gb,
    save_hardware_profile,
)
from physics.simulator import DigitalTwinSimulator


def test_hardware_detection():
    """Verify hardware discovery successfully detects physical RAM and cores."""
    hw = detect_hardware_profile()
    assert hw.total_ram_gb > 1.0, f"Expected total RAM > 1 GB, got {hw.total_ram_gb}"
    assert hw.available_ram_gb > 0.5, f"Expected available RAM > 0.5 GB, got {hw.available_ram_gb}"
    assert hw.logical_cores >= 1, f"Expected logical cores >= 1, got {hw.logical_cores}"
    assert hw.usable_ram_gb > 1.0, f"Expected usable RAM > 1 GB, got {hw.usable_ram_gb}"
    assert hw.max_particle_budget >= 1_000_000.0, "Expected budget ceiling >= 1,000,000"


def test_hardware_profile_persistence(tmp_path):
    """Verify saving and loading hardware profile JSON."""
    test_cfg = str(tmp_path / "hardware_profile.json")
    hw = detect_hardware_profile()
    hw.max_particle_budget = 1_200_000.0
    hw.max_ram_usage_percent = 75.0
    save_hardware_profile(hw, test_cfg)

    loaded = detect_hardware_profile(test_cfg)
    assert loaded.max_particle_budget == 1_200_000.0
    assert loaded.max_ram_usage_percent == 75.0


def test_presheath_budgeting_low_density_unconstrained():
    """
    Verify that at low density (n0 = 5e16), target_ppc = 200 is 100% UNCONSTRAINED
    and not choked down by total bounding-box cells.
    """
    with open("configurations/configRF.json") as f:
        cfg = json.load(f)

    p = {
        **cfg["simulation"],
        "grids": cfg["grids"],
        "discharge_chamber": cfg["discharge_chamber"],
        "mesh_zones": cfg.get("mesh_zones", {}),
        "advanced_settings": cfg.get("advanced_settings", {}),
        "n0_plasma": 5e16,
        "target_ppc": 200.0,
        r"\deltax/debye_length": 0.8,
        "dx_over_debye": 0.8,
    }

    sim = DigitalTwinSimulator()
    sim.build_domain(p)

    # In low density, target_ppc must remain fully unconstrained (effective_ppc == 200)
    assert sim.effective_ppc == 200.0, (
        f"Expected effective_ppc to be unconstrained at 200.0, got {sim.effective_ppc}"
    )
    assert sim.max_particle_budget >= 1_000_000.0


def test_presheath_budgeting_extreme_density_bounded():
    """
    Verify that at extreme density (n0 = 1e19), particle budget protects memory
    by capping expected particles at >= 1,000,000 without crashing.
    """
    with open("configurations/configRF.json") as f:
        cfg = json.load(f)

    p = {
        **cfg["simulation"],
        "grids": cfg["grids"],
        "discharge_chamber": cfg["discharge_chamber"],
        "mesh_zones": cfg.get("mesh_zones", {}),
        "advanced_settings": cfg.get("advanced_settings", {}),
        "n0_plasma": 1e19,
        "target_ppc": 200.0,
        r"\deltax/debye_length": 0.8,
        "dx_over_debye": 0.8,
    }

    sim = DigitalTwinSimulator()
    sim.build_domain(p)

    # In extreme density, effective_ppc should be safely modulated, bounded to at least 1,000,000 budget
    assert sim.effective_ppc < 200.0, "Expected effective_ppc to be modulated for extreme density"
    assert sim.effective_ppc >= 10.0, f"Expected healthy floor >= 10.0, got {sim.effective_ppc}"
    assert sim.max_particle_budget >= 1_000_000.0


def test_plume_preflight_warning_and_truncation():
    """
    Verify pre-flight warning for long plume and truncation support.
    """
    with open("configurations/configRF.json") as f:
        cfg = json.load(f)

    # 1. Without truncation on high density (long 8 mm plume)
    p_long = {
        **cfg["simulation"],
        "grids": cfg["grids"],
        "discharge_chamber": cfg["discharge_chamber"],
        "mesh_zones": cfg.get("mesh_zones", {}),
        "advanced_settings": cfg.get("advanced_settings", {}),
        "n0_plasma": 1e19,
        r"\deltax/debye_length": 0.8,
        "dx_over_debye": 0.8,
        "truncate_downstream_plume": False,
    }
    sim_long = DigitalTwinSimulator()
    sim_long.build_domain(p_long)

    assert sim_long.plume_warning_active is True
    assert sim_long.recommended_Lx < sim_long.Lx
    cells_long = sim_long.nx * sim_long.ny

    # 2. With truncation enabled
    p_trunc = dict(p_long)
    p_trunc["truncate_downstream_plume"] = True
    sim_trunc = DigitalTwinSimulator()
    sim_trunc.build_domain(p_trunc)

    cells_trunc = sim_trunc.nx * sim_trunc.ny
    assert sim_trunc.Lx == sim_long.recommended_Lx
    assert cells_trunc < cells_long * 0.85, (
        f"Truncation should cut cells by >15%: {cells_trunc} vs {cells_long}"
    )


def test_dynamic_plume_reshaping():
    """
    Verify runtime reshape_downstream_plume truncates domain and updates mesh correctly.
    """
    with open("configurations/configRF.json") as f:
        cfg = json.load(f)

    p = {
        **cfg["simulation"],
        "grids": cfg["grids"],
        "discharge_chamber": cfg["discharge_chamber"],
        "mesh_zones": cfg.get("mesh_zones", {}),
        "advanced_settings": cfg.get("advanced_settings", {}),
        "n0_plasma": 1e17,
        r"\deltax/debye_length": 0.8,
        "dx_over_debye": 0.8,
    }
    sim = DigitalTwinSimulator()
    sim.build_domain(p)

    initial_Lx = sim.Lx
    new_Lx = round(sim.Lx - 3.0, 3)
    success = sim.reshape_downstream_plume(new_Lx=new_Lx, params=p)

    assert success is True
    assert sim.Lx == new_Lx
    assert sim.Lx < initial_Lx


def test_payload_memory_estimator():
    """Verify estimate_payload_ram_gb produces reasonable memory estimates."""
    with open("configurations/configRF.json") as f:
        cfg = json.load(f)

    payload_low = {
        "param_id": "n0_plasma",
        "parameter_value": 5e16,
        "base_params": {**cfg["simulation"], "grids": cfg["grids"], "discharge_chamber": cfg["discharge_chamber"]},
    }
    payload_high = {
        "param_id": "n0_plasma",
        "parameter_value": 1e19,
        "base_params": {**cfg["simulation"], "grids": cfg["grids"], "discharge_chamber": cfg["discharge_chamber"]},
    }

    ram_low = estimate_payload_ram_gb(payload_low)
    ram_high = estimate_payload_ram_gb(payload_high)

    assert 0.5 <= ram_low <= 3.0, f"Unexpected ram_low: {ram_low} GB"
    assert ram_high > ram_low, "High density scenario should estimate higher RAM"
    assert ram_high >= 3.0, f"Expected ram_high >= 3.0 GB, got {ram_high}"

