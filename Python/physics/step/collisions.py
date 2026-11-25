"""
Charge-exchange (CEX) collision Monte Carlo model.
"""
import numpy as np
from ..backend import _NP_FP
from .context import StepContext


def step_cex(sim, ctx: StepContext):
    """Perform charge exchange collisions between energetic beam ions and background neutrals."""
    params = ctx.params

    if sim.num_p > 0:
        p_x = sim.p_x[:sim.num_p]
        p_y = sim.p_y[:sim.num_p]
        p_vx = sim.p_vx[:sim.num_p]
        p_vy = sim.p_vy[:sim.num_p]
        p_vz = sim.p_vz[:sim.num_p]
        p_cex = sim.p_isCEX[:sim.num_p]

        primary_mask = (~p_cex) & (p_x >= 1.0) & (p_x <= sim.Lx)
        if np.any(primary_mask):
            n_primary = int(np.sum(primary_mask))
            px_m = p_x[primary_mask]
            py_m = p_y[primary_mask]

            r_T = sim.Ly * 1e-3
            z_m = np.maximum((px_m - 1.0) * 1e-3, 0.0)
            r_m = py_m * 1e-3
            a_corr = 1.0 / (1.0 - 1.0 / np.sqrt(2.0))
            R_dist = np.sqrt(r_m ** 2 + (z_m + r_T) ** 2)
            theta = np.arctan2(r_m, z_m + r_T)

            n_local = params.get('n0', 1e20) * a_corr * (
                1.0 - 1.0 / np.sqrt(1.0 + (r_T / np.maximum(R_dist, 1e-12)) ** 2)
            ) * np.cos(theta)
            n_local = np.maximum(n_local, 0.0)

            v_mag = np.sqrt(
                p_vx[primary_mask] ** 2 +
                p_vy[primary_mask] ** 2 +
                p_vz[primary_mask] ** 2
            )
            g = np.maximum(v_mag, 1.0)
            E_eV_cex = (0.5 * sim.m_ion * g ** 2) / sim.q

            sigma_user = sim.lookup_user_cs('CX', E_eV_cex)
            sigma = sigma_user if sigma_user is not None else ((-0.8821 * np.log(g) + 15.1262) ** 2) * 1e-20
            prob = 1.0 - np.exp(-n_local * sigma * g * sim.dt)
            collided = np.random.rand(n_primary) < prob

            if np.any(collided):
                c_idx = np.where(primary_mask)[0][collided]
                n_coll = len(c_idx)
                neut_vth = np.sqrt(2.0 * sim.kB * params.get('Tn', 300.0) / sim.m_ion)

                fM_x = 2.0 * (
                    np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5
                )
                fM_y = 2.0 * (
                    np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5
                )
                fM_z = 2.0 * (
                    np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5
                )

                sim.p_vx[c_idx] = (neut_vth * fM_x).astype(_NP_FP)
                sim.p_vy[c_idx] = (neut_vth * fM_y).astype(_NP_FP)
                sim.p_vz[c_idx] = (neut_vth * fM_z).astype(_NP_FP)
                sim.p_isCEX[c_idx] = True
