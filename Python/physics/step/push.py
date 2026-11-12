"""
Particle integration via Boris push with substepping and periodic boundary handling.
"""
import numpy as np

from ..backend import USE_TAICHI
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

    num_e_step = int(sim.num_e)
    if num_e_step > 0:
        n_sub_e, ds_e, ds_lim_e = sim.compute_particle_substeps(
            sim.e_x[:num_e_step], sim.e_y[:num_e_step],
            sim.e_vx[:num_e_step], sim.e_vy[:num_e_step], sim.e_vz[:num_e_step],
            -sim.q / sim.m_e, sim.dt
        )
        dt_max_e = getattr(sim, 'dt_max_electron', None)
        if dt_max_e is not None and dt_max_e > 0:
            n_sub_e = max(n_sub_e, int(np.ceil(sim.dt / dt_max_e)))
    else:
        n_sub_e = 1

    sim.last_n_sub_ion = int(n_sub_ion)
    sim.last_n_sub_e = int(n_sub_e)
    if not hasattr(sim, 'substep_history_iter'):
        sim.substep_history_iter = []
        sim.substep_history_ion = []
        sim.substep_history_e = []
    sim.substep_history_iter.append(int(sim.iteration))
    sim.substep_history_ion.append(int(n_sub_ion))
    sim.substep_history_e.append(int(n_sub_e))

    periodic_y = getattr(sim, 'periodic_y', False)
    geometry = getattr(sim, 'geometry', 'half_hole')
    reflect_y0 = (not periodic_y) and getattr(sim, 'reflect_y0', geometry == 'half_hole')
    reflect_y_top = (not periodic_y) and getattr(sim, 'reflect_y_top', False)
    refl_y0_i = 1 if reflect_y0 else 0
    refl_ytop_i = 1 if reflect_y_top else 0
    Ly_val = float(getattr(sim, 'Ly', 0.0))

    if num_p_step > 0:
        dt_ion = np.float32(sim.dt / n_sub_ion)

        if USE_TAICHI:
            push_particles_boris_taichi(
                sim.p_x[:num_p_step], sim.p_y[:num_p_step],
                sim.p_vx[:num_p_step], sim.p_vy[:num_p_step], sim.p_vz[:num_p_step],
                sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
                num_p_step, ctx._dy, sim.nx, sim.ny, dt_ion, _qm_ion,
                ctx.zone_b_ti, ctx.zone_dx_ti, ctx.zone_off_ti, ctx.num_zones,
                n_sub=n_sub_ion,
                Ly=Ly_val, reflect_y0=refl_y0_i, reflect_y_top=refl_ytop_i
            )
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
                    sim.x_coords,
                    Ly=Ly_val, reflect_y0=reflect_y0, reflect_y_top=reflect_y_top
                )

    if num_e_step > 0:
        dt_e = np.float32(sim.dt / n_sub_e)

        if USE_TAICHI:
            push_particles_boris_taichi(
                sim.e_x[:num_e_step], sim.e_y[:num_e_step],
                sim.e_vx[:num_e_step], sim.e_vy[:num_e_step], sim.e_vz[:num_e_step],
                sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
                num_e_step, ctx._dy, sim.nx, sim.ny, dt_e, _qm_e,
                ctx.zone_b_ti, ctx.zone_dx_ti, ctx.zone_off_ti, ctx.num_zones,
                n_sub=n_sub_e,
                Ly=Ly_val, reflect_y0=refl_y0_i, reflect_y_top=refl_ytop_i
            )
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
                    sim.x_coords,
                    Ly=Ly_val, reflect_y0=reflect_y0, reflect_y_top=reflect_y_top
                )

    if getattr(sim, 'periodic_y', False):
        if num_p_step > 0:
            sim.p_y[:num_p_step] = np.mod(sim.p_y[:num_p_step], sim.Ly)
        if sim.num_e > 0:
            sim.e_y[:sim.num_e] = np.mod(sim.e_y[:sim.num_e], sim.Ly)
