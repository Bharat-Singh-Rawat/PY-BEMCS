"""
Compatibility shim for PY-BEMCS physics engine.

This file maintains full backwards compatibility with external scripts,
benchmarks, test suites, and diagnostic windows. All implementation has been
modularized cleanly into the `physics` package.
"""
from physics import (
    DigitalTwinSimulator,
    compute_debye_upstream_gap,
    estimate_automatic_mesh_zones,
    accumulate_rho_cpu,
    push_particles_boris_cpu,
    accumulate_rho_taichi,
    push_particles_boris_taichi,
    thermal_conduction_taichi,
    USE_TAICHI,
    _NP_FP,
    _TI_FP,
    MATERIAL_PRESETS,
)
from physics.backend import _GPU_POISSON, _USE_GPU_POISSON, _ti_arr, _ti_arr_i32
from physics.constants import Q_ELEM, M_XE, K_B, EPS_0, SIGMA_SB

__all__ = [
    "DigitalTwinSimulator",
    "compute_debye_upstream_gap",
    "estimate_automatic_mesh_zones",
    "accumulate_rho_cpu",
    "push_particles_boris_cpu",
    "accumulate_rho_taichi",
    "push_particles_boris_taichi",
    "thermal_conduction_taichi",
    "USE_TAICHI",
    "_NP_FP",
    "_TI_FP",
    "_GPU_POISSON",
    "_USE_GPU_POISSON",
    "MATERIAL_PRESETS",
    "Q_ELEM",
    "M_XE",
    "K_B",
    "EPS_0",
    "SIGMA_SB",
]