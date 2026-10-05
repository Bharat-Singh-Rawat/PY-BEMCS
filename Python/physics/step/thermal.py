"""
Thermal radiative cooling, conductive diffusion, and structural grid deflection.
"""
import numpy as np

from ..backend import USE_TAICHI, _NP_FP
from ..kernels_taichi import thermal_conduction_taichi
from .context import StepContext


def step_thermal(sim, ctx: StepContext):
    """Update grid temperature via Stefan-Boltzmann radiation, conduction, and compute thermoelastic deflection."""
    sim_mode = ctx.sim_mode
    params = ctx.params
    grids = ctx.grids

    if sim_mode in ['Thermal', 'Both']:
        T_bound = sim.Tmap[sim.isBound]
        cooling_factor = (sim.emissivity * sim.sb_sigma * sim.A_cell * sim.dt * sim.thermal_accel) / sim.C_cell
        dT_cool = cooling_factor * (T_bound ** 4 - 300.0 ** 4)
        sim.Tmap[sim.isBound] -= dT_cool
        sim.Tmap = sim.Tmap.astype(_NP_FP)

        alpha_diff = sim.mat_k / (sim.mat_rho * sim.mat_cp)
        dt_thermal = sim.dt * sim.thermal_accel
        dx_m = sim.dx * 1e-3
        dy_m = sim.dy * 1e-3
        Fo_x = alpha_diff * dt_thermal / dx_m ** 2
        Fo_y = alpha_diff * dt_thermal / dy_m ** 2

        max_Fo = 0.2
        if Fo_x > max_Fo or Fo_y > max_Fo:
            scale = max_Fo / max(Fo_x, Fo_y)
            Fo_x *= scale
            Fo_y *= scale

        if USE_TAICHI:
            for _ in range(10):
                thermal_conduction_taichi(
                    sim.Tmap,
                    sim.T_map_new,
                    sim.isBound.astype(np.int32),
                    sim.nx,
                    sim.ny,
                    np.float32(Fo_x),
                    np.float32(Fo_y)
                )
                sim.Tmap, sim.T_map_new = sim.T_map_new, sim.Tmap
        else:
            T = sim.Tmap
            T_new = sim.T_map_new
            for _ in range(10):
                T_new[:] = T[:]
                for iy_ in range(1, sim.ny - 1):
                    for ix_ in range(1, sim.nx - 1):
                        if sim.isBound[iy_, ix_]:
                            T_l = T[iy_, ix_ - 1] if sim.isBound[iy_, ix_ - 1] else T[iy_, ix_]
                            T_r = T[iy_, ix_ + 1] if sim.isBound[iy_, ix_ + 1] else T[iy_, ix_]
                            T_d = T[iy_ - 1, ix_] if sim.isBound[iy_ - 1, ix_] else T[iy_, ix_]
                            T_u = T[iy_ + 1, ix_] if sim.isBound[iy_ + 1, ix_] else T[iy_, ix_]
                            dT = Fo_x * (T_l - 2.0 * T[iy_, ix_] + T_r) + Fo_y * (T_d - 2.0 * T[iy_, ix_] + T_u)
                            T_new[iy_, ix_] = max(T[iy_, ix_] + dT, 300.0)
                T, T_new = T_new, T
            sim.Tmap, sim.T_map_new = T, T_new

        sim.Tmap[sim.isBound] = np.maximum(sim.Tmap[sim.isBound], 300.0)

        needs_remesh = False
        for i, mask in enumerate(sim.mask_grids):
            if np.any(mask):
                sim.T_grids[i] = np.mean(sim.Tmap[mask])

        geometry = getattr(sim, 'geometry', 'half_hole')
        for i, grid in enumerate(grids):
            dT = sim.T_grids[i] - 300.0
            if geometry == 'two_holes':
                pitch_val = params.get("pitch_mm", params.get("discharge_chamber", {}).get("pitch_mm", 3.0) if isinstance(params.get("discharge_chamber"), dict) else 3.0)
                L_cant_mm = max(0.2, pitch_val - 2.0 * grid['r'])
            elif geometry == 'one_hole':
                L_cant_mm = max(0.2, 0.5 * (sim.Ly - 2.0 * grid['r']))
            else:
                L_cant_mm = max(0.2, sim.Ly - grid['r'])

            L_cant_m = L_cant_mm * 1e-3
            t_m = grid['t'] * 1e-3
            if t_m > 0 and L_cant_m > 0:
                delta_m = sim.alpha_thermal * dT * L_cant_m ** 2 / (2.0 * t_m)
                new_defl = delta_m * 1e3
            else:
                new_defl = 0.0

            if abs(new_defl - sim.grid_deflections[i]) > 0.005:
                sim.grid_deflections[i] = new_defl
                needs_remesh = True

        if needs_remesh:
            sim.build_domain(params, preserve_state=True)
            ctx.remeshed = True
