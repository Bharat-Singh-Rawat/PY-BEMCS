"""
Ion wall collisions, sputtering erosion, secondary electron emission, and beam transmission diagnostics.
"""
import numpy as np
from ..backend import _NP_FP
from .context import StepContext


def step_ion_interactions(sim, ctx: StepContext):
    """Handle ion collisions with grid walls, thermal loads, sputter erosion, SEE, and exit diagnostics."""
    num_p_step = ctx.num_p_step
    p_x_old = ctx.p_x_old
    p_y_old = ctx.p_y_old
    grids = ctx.grids
    sim_mode = ctx.sim_mode
    params = ctx.params
    t_current = ctx.t_current

    p_x = sim.p_x[:num_p_step].copy()
    p_y = sim.p_y[:num_p_step].copy()
    p_vx = sim.p_vx[:num_p_step].copy()
    p_vy = sim.p_vy[:num_p_step].copy()
    p_vz = sim.p_vz[:num_p_step].copy()
    p_cex = sim.p_isCEX[:num_p_step].copy()

    if grids and hasattr(sim, "grid_x_starts") and len(sim.grid_x_starts) == len(grids):
        x_entry_first_grid = sim.grid_x_starts[0]
        x_exit_last = sim.grid_x_ends[-1]
    else:
        x_entry_first_grid = 0.5
        x_exit_last = 3.0

    entered_first_grid_mask = (
        (p_x_old < x_entry_first_grid) &
        (p_x >= x_entry_first_grid) &
        (p_vx > 0.0)
    )
    n_entered = int(np.count_nonzero(entered_first_grid_mask))
    sim.entered_optics_step = float(n_entered)
    sim.entered_optics += float(n_entered)
    sim.prev_entered = sim.entered_optics_step

    ix = sim._x_to_ix(p_x)
    iy = np.clip(np.round(p_y / sim.dy).astype(int), 0, sim.ny - 1)

    hit_grid_final = sim.isBound[iy, ix]
    hit_grid_path, path_ix, path_iy = sim._segment_hits_grid(p_x_old, p_y_old, p_x, p_y, samples=8)
    hit_grid = hit_grid_final | hit_grid_path

    impact_ix = np.where(hit_grid_path, path_ix, ix)
    impact_iy = np.where(hit_grid_path, path_iy, iy)

    if getattr(sim, 'periodic_y', False):
        out_of_bounds = (
            (p_x < 0.0) |
            (p_x > sim.Lx) |
            np.isnan(p_x)
        )
    else:
        out_of_bounds = (
            (p_x < 0.0) |
            (p_x > sim.Lx) |
            (p_y < 0.0) |
            (p_y > sim.Ly) |
            np.isnan(p_x)
        )
    sim.lost_to_oob_step = float(np.count_nonzero(out_of_bounds))

    valid_thermal_hit = hit_grid & (p_x > 0.25)
    if sim_mode in ("Thermal", "Both") and np.any(valid_thermal_hit):
        v_mag_sq = (
            p_vx[valid_thermal_hit] ** 2 +
            p_vy[valid_thermal_hit] ** 2 +
            p_vz[valid_thermal_hit] ** 2
        )
        E_joules = 0.5 * sim.m_ion * v_mag_sq * sim.macro_weight
        dT_heat = (E_joules / sim.C_cell) * sim.thermal_accel
        np.add.at(sim.Tmap, (impact_iy[valid_thermal_hit], impact_ix[valid_thermal_hit]), dT_heat)

    valid_see_hit = hit_grid & (p_x > 0.25)
    if np.any(valid_see_hit):
        v_mag_sq = (
            p_vx[valid_see_hit] ** 2 +
            p_vy[valid_see_hit] ** 2 +
            p_vz[valid_see_hit] ** 2
        )
        E_eV = (0.5 * sim.m_ion * v_mag_sq) / sim.q
        see_user = sim.lookup_user_cs('SEE', E_eV)
        gamma = np.clip(see_user if see_user is not None else 0.05 + 1e-4 * E_eV, 0.0, 1.0)
        spawn_mask = np.random.rand(len(gamma)) < gamma

        if np.any(spawn_mask):
            num_see = int(np.sum(spawn_mask))
            see_x = (
                p_x[valid_see_hit][spawn_mask] -
                p_vx[valid_see_hit][spawn_mask] * sim.dt * 1000.0 / 1.5
            ).astype(_NP_FP)
            see_y = (
                p_y[valid_see_hit][spawn_mask] -
                p_vy[valid_see_hit][spawn_mask] * sim.dt * 1000.0 / 1.5
            ).astype(_NP_FP)
            T_see = 2.0
            v_see_th = np.sqrt(2.0 * sim.q * T_see / sim.m_e)
            see_vx = (np.random.randn(num_see) * v_see_th).astype(_NP_FP)
            see_vy = (np.random.randn(num_see) * v_see_th).astype(_NP_FP)
            see_vz = (np.random.randn(num_see) * v_see_th).astype(_NP_FP)
            sim._add_electrons(see_x, see_y, see_vx, see_vy, see_vz)

    is_erosion_hit = hit_grid & (p_x > 0.25)
    if sim_mode in ("Erosion", "Both") and np.any(is_erosion_hit):
        E_eV = (
            0.5 * sim.m_ion * (
                p_vx[is_erosion_hit] ** 2 +
                p_vy[is_erosion_hit] ** 2 +
                p_vz[is_erosion_hit] ** 2
            )
        ) / sim.q

        valid_among_hits = E_eV > sim.sputter_E_th

        if np.any(valid_among_hits):
            E_valid = E_eV[valid_among_hits]
            yield_rate = sim.sputter_Y_coeff * (E_valid - sim.sputter_E_th)
            damage = yield_rate * sim.macro_weight
            iy_hit = impact_iy[is_erosion_hit][valid_among_hits]
            ix_hit = impact_ix[is_erosion_hit][valid_among_hits]
            np.add.at(
                sim.damage_map,
                (iy_hit, ix_hit),
                damage
            )
            sim._damage_version += 1

        broken_cells = (sim.damage_map > params.get('Thresh', 1e5)) & sim.isBound
        if np.any(broken_cells):
            sim.eroded_depth[broken_cells] += sim.dy
            sim.isBound[broken_cells] = False
            sim.damage_map[broken_cells] = 0.0
            sim.build_sparse_matrix()
            ctx.remeshed = True

    exited_mask = (p_x > x_exit_last) & (p_vx > 0.0)
    crossed_mask = (p_x_old <= x_exit_last) & (p_x > x_exit_last) & (p_vx > 0.0)

    if np.count_nonzero(crossed_mask) > 0:
        vx_exit = p_vx[crossed_mask].copy()
        vy_exit = p_vy[crossed_mask].copy()
        vz_exit = p_vz[crossed_mask].copy()
        cex_exit = p_cex[crossed_mask].copy()
        v_exit = np.sqrt(vx_exit ** 2 + vy_exit ** 2 + vz_exit ** 2)
        E_exit_eV = 0.5 * sim.m_ion * v_exit ** 2 / sim.q

        sim.exit_vx_mean = float(np.mean(vx_exit))
        sim.exit_v_mean = float(np.mean(v_exit))
        sim.exit_vx_std = float(np.std(vx_exit))
        sim.exit_v_std = float(np.std(v_exit))
        sim.exit_energy_mean_eV = float(np.mean(E_exit_eV))
        sim.exit_count_step = int(np.count_nonzero(crossed_mask))

        prim_exit = ~cex_exit
        sim.exit_v_mean_primary = float(np.mean(v_exit[prim_exit])) if np.any(prim_exit) else np.nan
        sim.exit_v_mean_cex = float(np.mean(v_exit[cex_exit])) if np.any(cex_exit) else np.nan
    else:
        sim.exit_vx_mean = np.nan
        sim.exit_v_mean = np.nan
        sim.exit_vx_std = np.nan
        sim.exit_v_std = np.nan
        sim.exit_energy_mean_eV = np.nan
        sim.exit_count_step = 0
        sim.exit_v_mean_primary = np.nan
        sim.exit_v_mean_cex = np.nan

    ctx.current_div = (
        np.percentile(
            np.abs(np.arctan2(p_vy[crossed_mask], p_vx[crossed_mask])) * 180.0 / np.pi,
            95
        )
        if np.count_nonzero(crossed_mask) > 5 else np.nan
    )

    if np.any(exited_mask):
        n_transmitted = int(np.count_nonzero(crossed_mask))
        sim.transmitted_ions_step = float(n_transmitted)
        sim.transmitted_ions += float(n_transmitted)
        sim.transmitted3_step = float(n_transmitted)

    charge_per_macro = sim.q_ion * sim.macro_weight
    sim.exit_ion_current_step = (sim.transmitted_ions_step * charge_per_macro) / sim.dt if sim.dt > 0 else 0.0
    sim.exit_ion_current_avg = (sim.transmitted_ions * charge_per_macro) / t_current if t_current > 0 else 0.0

    grid_hit_mask = hit_grid & ~out_of_bounds
    sim.lost_to_grid_step = float(np.count_nonzero(grid_hit_mask))

    dead_mask = hit_grid | out_of_bounds
    alive_mask = ~dead_mask
    n_alive = int(np.sum(alive_mask))

    if n_alive < num_p_step:
        sim.p_x[:n_alive] = p_x[alive_mask]
        sim.p_y[:n_alive] = p_y[alive_mask]
        sim.p_vx[:n_alive] = p_vx[alive_mask]
        sim.p_vy[:n_alive] = p_vy[alive_mask]
        sim.p_vz[:n_alive] = p_vz[alive_mask]
        sim.p_isCEX[:n_alive] = p_cex[alive_mask]
        sim.num_p = n_alive
    else:
        sim.num_p = num_p_step
