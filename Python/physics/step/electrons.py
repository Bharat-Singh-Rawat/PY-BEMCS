"""
Electron lifetime management, boundary termination, and neutralizer injection.
"""
import numpy as np
from ..backend import _NP_FP
from .context import StepContext


def step_electrons(sim, ctx: StepContext):
    """Purge out-of-bounds/absorbed electrons and inject electrons from neutralizer cathode."""
    params = ctx.params

    if sim.num_e > 0:
        e_x = sim.e_x[:sim.num_e]
        e_y = sim.e_y[:sim.num_e]
        e_vx = sim.e_vx[:sim.num_e]
        e_vy = sim.e_vy[:sim.num_e]
        e_vz = sim.e_vz[:sim.num_e]

        ix_e = sim._x_to_ix(e_x)
        iy_e = np.clip(np.round(e_y / sim.dy).astype(int), 0, sim.ny - 1)
        hit_grid_e = sim.isBound[iy_e, ix_e]
        out_e = (
            (e_x < 0.0) |
            (e_x > sim.Lx) |
            (e_y < 0.0) |
            (e_y > sim.Ly) |
            np.isnan(e_x)
        )
        dead_e = hit_grid_e | out_e
        alive_e = ~dead_e
        n_alive_e = int(np.sum(alive_e))

        if n_alive_e < sim.num_e:
            sim.e_x[:n_alive_e] = e_x[alive_e]
            sim.e_y[:n_alive_e] = e_y[alive_e]
            sim.e_vx[:n_alive_e] = e_vx[alive_e]
            sim.e_vy[:n_alive_e] = e_vy[alive_e]
            sim.e_vz[:n_alive_e] = e_vz[alive_e]
            sim.num_e = n_alive_e

    neut_match_ion = params.get('neut_match_ion', False)
    if neut_match_ion:
        num_e_neut = int(round(sim.transmitted_ions_step))
    else:
        num_e_neut = int(params.get('neut_rate', 30))
    sim._last_neut_injected = num_e_neut

    Te_eV = params.get('Te', 5.0)
    if hasattr(sim, 'grid_x_ends') and sim.grid_x_ends:
        _x_exit = sim.grid_x_ends[-1]
        _neut_default = _x_exit + 0.9 * (sim.Lx - _x_exit)
    else:
        _neut_default = sim.Lx - 0.5
    neut_x_param = params.get('neut_x', _neut_default)
    neut_r_param = params.get('neut_r', sim.Ly)
    if neut_x_param > sim.Lx or neut_x_param < 0.0:
        if not getattr(sim, '_warned_neut_oob', False):
            print(f"[Warning] Neutralizer position x={neut_x_param:.3f} mm is outside x-domain [0, {sim.Lx:.3f} mm]. Setting to x = Lx ({sim.Lx:.3f} mm).")
            sim._warned_neut_oob = True
        neut_x = float(sim.Lx)
    else:
        sim._warned_neut_oob = False
        neut_x = float(neut_x_param)
    neut_r = float(np.clip(neut_r_param, sim.dy, sim.Ly))
    if num_e_neut > 0:
        new_ey = np.random.uniform(0.0, neut_r, num_e_neut).astype(_NP_FP)
        new_ex = np.full(num_e_neut, neut_x, dtype=_NP_FP)
        v_e_th = np.sqrt(2.0 * sim.q * Te_eV / sim.m_e)
        new_evx = (np.random.randn(num_e_neut) * v_e_th).astype(_NP_FP)
        new_evy = (np.random.randn(num_e_neut) * v_e_th).astype(_NP_FP)
        new_evz = (np.random.randn(num_e_neut) * v_e_th).astype(_NP_FP)
        sim._add_electrons(new_ex, new_ey, new_evx, new_evy, new_evz)
