"""
PY-BEMCS Physics Engine Package.
"""
from .backend import USE_TAICHI, _NP_FP, _TI_FP, _USE_GPU_POISSON, _GPU_POISSON, _ti_arr, _ti_arr_i32
from .constants import (
    Q_ELEM, M_XE, K_B, EPS_0, SIGMA_SB, MATERIAL_PRESETS
)
from .kernels_taichi import (
    accumulate_rho_taichi,
    push_particles_boris_taichi,
    thermal_conduction_taichi,
)
from .kernels_cpu import (
    accumulate_rho_cpu,
    push_particles_boris_cpu,
)
from .mesh_utils import (
    compute_debye_upstream_gap,
    estimate_automatic_mesh_zones,
)
from .simulator import DigitalTwinSimulator

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
    "MATERIAL_PRESETS",
]
