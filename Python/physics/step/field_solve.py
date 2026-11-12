"""
Charge deposition and periodic Poisson solve step phase.
"""
import numpy as np

from ..backend import USE_TAICHI
from ..kernels_taichi import accumulate_rho_taichi
from ..kernels_cpu import accumulate_rho_cpu
from .context import StepContext


def step_deposit_and_solve(sim, ctx: StepContext):
    """Deposit charge from ions and electrons to grid, then solve Poisson if scheduled."""
    sim.rho.fill(0.0)
    charge_per_particle = sim.q * sim.macro_weight

    if getattr(sim, 'zone_b_ti', None) is None or getattr(sim, 'cvol_r_ti', None) is None:
        sim._update_taichi_mesh_constants()

    ctx.zone_b_ti = sim.zone_b_ti
    ctx.zone_dx_ti = sim.zone_dx_ti
    ctx.zone_off_ti = sim.zone_off_ti
    ctx.num_zones = sim.num_zones
    ctx._dy = sim._dy_f32
    cvol_r_ti = sim.cvol_r_ti
    cvol_x_ti = sim.cvol_x_ti
    is_axisym = 1 if getattr(sim, 'is_axisymmetric', False) else 0

    if sim.num_p > 0:
        if USE_TAICHI:
            accumulate_rho_taichi(
                sim.p_x[:sim.num_p],
                sim.p_y[:sim.num_p],
                sim.rho,
                cvol_r_ti,
                cvol_x_ti,
                sim.num_p,
                ctx._dy,
                sim.nx,
                sim.ny,
                np.float32(charge_per_particle),
                ctx.zone_b_ti, ctx.zone_dx_ti, ctx.zone_off_ti, ctx.num_zones,
                is_axisym
            )
        else:
            accumulate_rho_cpu(
                sim.p_x[:sim.num_p],
                sim.p_y[:sim.num_p],
                sim.rho,
                sim.num_p,
                sim.dy,
                sim.nx,
                sim.ny,
                charge_per_particle,
                sim.zone_boundaries,
                sim.zone_dx,
                sim.zone_offsets,
                sim.cell_vol_1d,
                cvol_r=sim.cvol_r_1d,
                cvol_x=sim.cvol_x_1d,
                is_axisymmetric=bool(is_axisym)
            )

    if sim.num_e > 0:
        if USE_TAICHI:
            accumulate_rho_taichi(
                sim.e_x[:sim.num_e],
                sim.e_y[:sim.num_e],
                sim.rho,
                cvol_r_ti,
                cvol_x_ti,
                sim.num_e,
                ctx._dy,
                sim.nx,
                sim.ny,
                np.float32(-charge_per_particle),
                ctx.zone_b_ti, ctx.zone_dx_ti, ctx.zone_off_ti, ctx.num_zones,
                is_axisym
            )
        else:
            accumulate_rho_cpu(
                sim.e_x[:sim.num_e],
                sim.e_y[:sim.num_e],
                sim.rho,
                sim.num_e,
                sim.dy,
                sim.nx,
                sim.ny,
                -charge_per_particle,
                sim.zone_boundaries,
                sim.zone_dx,
                sim.zone_offsets,
                sim.cell_vol_1d,
                cvol_r=sim.cvol_r_1d,
                cvol_x=sim.cvol_x_1d,
                is_axisymmetric=bool(is_axisym)
            )

    if sim.iteration % 2 == 0:
        sim.recalc_poisson(iterations=5, params=ctx.params)
