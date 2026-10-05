"""
Particle integration via Boris push with substepping and periodic boundary handling.
"""
import numpy as np

from ..backend import USE_TAICHI, _NP_FP, _ti_arr
from ..kernels_taichi import push_particles_boris_taichi
from ..kernels_cpu import push_particles_boris_cpu
from .context import StepContext


def step_push(sim, ctx: StepContext):
    """Substep and push ions and electrons with the Boris algorithm."""
    num_p_step = int(sim.num_p)
    ctx.num_p_step = num_p_step

    ctx.p_x_old = sim.p_x[:num_p_step].copy()
    ctx.p_y_old = sim.p_y[:num_p_step].copy()

    _qm_ion = np.float32(sim.q_ion / sim.m_ion)
    _qm_e = np.float32(-sim.q / sim.m_e)

    if num_p_step > 0:
        n_sub_ion, ds_ion, ds_lim_ion = sim.compute_particle_substeps(
            sim.p_x[:num_p_step], sim.p_y[:num_p_step],
            sim.p_vx[:num_p_step], sim.p_vy[:num_p_step], sim.p_vz[:num_p_step],
            sim.q_ion / sim.m_ion, sim.dt
        )
    else:
        n_sub_ion = 1

    if n_sub_ion > 1:
        print(
            f"[Warning] Ion displacement criterion violated at iter {sim.iteration}: "
            f"predicted ds_max = {ds_ion:.3e} m, allowed = {ds_lim_ion:.3e} m, "
            f"required substeps = {n_sub_ion}"
        )

    num_e_step = int(sim.num_e)
    if num_e_step > 0:
        n_sub_e, ds_e, ds_lim_e = sim.compute_particle_substeps(
            sim.e_x[:num_e_step], sim.e_y[:num_e_step],
            sim.e_vx[:num_e_step], sim.e_vy[:num_e_step], sim.e_vz[:num_e_step],
            -sim.q / sim.m_e, sim.dt
        )
    else:
        n_sub_e = 1

    if n_sub_e > 1:
        print(
            f"[Warning] Electron displacement criterion violated at iter {sim.iteration}: "
            f"predicted ds_max = {ds_e:.3e} m, allowed = {ds_lim_e:.3e} m, "
            f"required substeps = {n_sub_e}"
        )

    if num_p_step > 0:
        dt_ion = np.float32(sim.dt / n_sub_ion)

        if USE_TAICHI:
            px_ti = _ti_arr(sim.p_x[:num_p_step])
            py_ti = _ti_arr(sim.p_y[:num_p_step])
            pvx_ti = _ti_arr(sim.p_vx[:num_p_step])
            pvy_ti = _ti_arr(sim.p_vy[:num_p_step])
            pvz_ti = _ti_arr(sim.p_vz[:num_p_step])

            for _ in range(n_sub_ion):
                push_particles_boris_taichi(
                    px_ti, py_ti, pvx_ti, pvy_ti, pvz_ti,
                    sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
                    num_p_step, ctx._dy, sim.nx, sim.ny, dt_ion, _qm_ion,
                    ctx.zone_b_ti, ctx.zone_dx_ti, ctx.zone_off_ti, ctx.num_zones
                )

            sim.p_x[:num_p_step]  = px_ti
            sim.p_y[:num_p_step]  = py_ti
            sim.p_vx[:num_p_step] = pvx_ti
            sim.p_vy[:num_p_step] = pvy_ti
            sim.p_vz[:num_p_step] = pvz_ti
        else:
            for _ in range(n_sub_ion):
                push_particles_boris_cpu(
                    sim.p_x[:num_p_step],
                    sim.p_y[:num_p_step],
                    sim.p_vx[:num_p_step],
                    sim.p_vy[:num_p_step],
                    sim.p_vz[:num_p_step],
                    sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
                    num_p_step, sim.dy, sim.nx, sim.ny,
                    sim.dt / n_sub_ion,
                    sim.q_ion / sim.m_ion,
                    sim.zone_boundaries,
                    sim.zone_dx,
                    sim.zone_offsets,
                    sim.x_coords
                )

    if num_e_step > 0:
        dt_e = np.float32(sim.dt / n_sub_e)

        if USE_TAICHI:
            ex_ti = _ti_arr(sim.e_x[:num_e_step])
            ey_ti = _ti_arr(sim.e_y[:num_e_step])
            evx_ti = _ti_arr(sim.e_vx[:num_e_step])
            evy_ti = _ti_arr(sim.e_vy[:num_e_step])
            evz_ti = _ti_arr(sim.e_vz[:num_e_step])

            for _ in range(n_sub_e):
                push_particles_boris_taichi(
                    ex_ti, ey_ti, evx_ti, evy_ti, evz_ti,
                    sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
                    num_e_step, ctx._dy, sim.nx, sim.ny, dt_e, _qm_e,
                    ctx.zone_b_ti, ctx.zone_dx_ti, ctx.zone_off_ti, ctx.num_zones
                )

            sim.e_x[:num_e_step]  = ex_ti
            sim.e_y[:num_e_step]  = ey_ti
            sim.e_vx[:num_e_step] = evx_ti
            sim.e_vy[:num_e_step] = evy_ti
            sim.e_vz[:num_e_step] = evz_ti
        else:
            for _ in range(n_sub_e):
                push_particles_boris_cpu(
                    sim.e_x[:num_e_step],
                    sim.e_y[:num_e_step],
                    sim.e_vx[:num_e_step],
                    sim.e_vy[:num_e_step],
                    sim.e_vz[:num_e_step],
                    sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
                    num_e_step, sim.dy, sim.nx, sim.ny,
                    sim.dt / n_sub_e,
                    -sim.q / sim.m_e,
                    sim.zone_boundaries,
                    sim.zone_dx,
                    sim.zone_offsets,
                    sim.x_coords
                )

    if getattr(sim, 'periodic_y', False):
        if num_p_step > 0:
            sim.p_y[:num_p_step] = np.mod(sim.p_y[:num_p_step], sim.Ly)
        if sim.num_e > 0:
            sim.e_y[:sim.num_e] = np.mod(sim.e_y[:sim.num_e], sim.Ly)
