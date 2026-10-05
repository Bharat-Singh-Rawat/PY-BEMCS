"""
Particle injection phase and RF voltage modulation.
"""
import numpy as np
from ..backend import _NP_FP
from .context import StepContext


def step_rf(sim, ctx: StepContext):
    """Apply RF modulated voltage if enabled."""
    params = ctx.params
    grids = ctx.grids
    if params.get('rf_enable') and grids:
        rf_idx = params.get('rf_grid_idx', 0)
        if rf_idx < len(grids):
            f_hz = params.get('rf_freq', 13.56) * 1e6
            v_rf = params.get('rf_amp', 100.0) * np.sin(2.0 * np.pi * f_hz * ctx.t_current)
            sim.V_fixed[sim.mask_grids[rf_idx]] = sim.Vdc[sim.mask_grids[rf_idx]] + v_rf
            sim.recalc_poisson(iterations=2, params=params)


def step_inject(sim, ctx: StepContext):
    """Inject ions and electrons from upstream boundary."""
    params = ctx.params
    grids = ctx.grids
    t_current = ctx.t_current

    if not sim.injection_enabled or (
        sim.injection_stop_time is not None and t_current > sim.injection_stop_time
    ):
        return

    n0 = params.get('n0_plasma', 1e17)
    Te_up = params.get('Te_up', 3.0)
    Ti = params.get('Ti', 0.1)

    v_bohm = np.sqrt(sim.q_ion * Te_up / sim.m_ion)

    if 'injection_area_scale' in params:
        injection_area = sim.Ly * 1e-3 * 1e-3 * params['injection_area_scale']
    else:
        _unit_depth_m = 1e-3
        geometry = getattr(sim, 'geometry', 'half_hole')
        if grids:
            screen_r = grids[0]['r']
            pitch_inj = params.get('pitch_mm', 0.0)
            if geometry == 'two_holes':
                h1 = min(2.5 * screen_r, sim.Ly - sim.dy) - max(sim.dy, 0.5 * screen_r)
                h2 = (min(2.5 * screen_r + pitch_inj, sim.Ly - sim.dy)
                      - max(sim.dy, 0.5 * screen_r + pitch_inj))
                inj_height_mm = max(h1, 0.0) + max(h2, 0.0)
            elif geometry == 'one_hole':
                inj_height_mm = (min(2.5 * screen_r, sim.Ly - sim.dy)
                                 - max(sim.dy, 0.5 * screen_r))
            else:
                inj_height_mm = min(screen_r, sim.Ly - sim.dy)
        else:
            inj_height_mm = sim.Ly - 2.0 * sim.dy

        inj_height_mm = max(inj_height_mm, sim.dy)
        injection_area = inj_height_mm * 1e-3 * _unit_depth_m

    sim._last_injection_area = injection_area
    entire_bulk_plasma = params.get('entire_bulk_plasma', False)
    bohm_factor = 1.0 if entire_bulk_plasma else 0.607
    I_ion = sim.q_ion * bohm_factor * n0 * v_bohm * injection_area
    charge_per_macro = sim.q_ion * sim.macro_weight

    num_inject_float = (I_ion * sim.dt) / charge_per_macro
    num_inject = np.random.poisson(num_inject_float)

    if num_inject > 0:
        geometry = getattr(sim, 'geometry', 'half_hole')
        if grids:
            screen_r = grids[0]['r']
            pitch_inj = params.get('pitch_mm', 0.0)
            if geometry == 'two_holes':
                span_limits = np.array([
                    [max(sim.dy, 0.5 * screen_r),
                     min(2.5 * screen_r, sim.Ly - sim.dy)],
                    [max(sim.dy, 0.5 * screen_r + pitch_inj),
                     min(2.5 * screen_r + pitch_inj, sim.Ly - sim.dy)],
                ], dtype=_NP_FP)
                which_hole = np.random.randint(0, 2, size=num_inject)
                y_lo  = span_limits[which_hole, 0]
                y_hi  = span_limits[which_hole, 1]
                new_y = (y_lo + (y_hi - y_lo) * np.random.rand(num_inject)).astype(_NP_FP)
            elif geometry == 'one_hole':
                ylow  = max(sim.dy, 0.5 * screen_r)
                yhigh = min(2.5 * screen_r, sim.Ly - sim.dy)
                new_y = np.random.uniform(ylow, yhigh, num_inject).astype(_NP_FP)
            else:
                ylow  = 0.0
                yhigh = min(screen_r, sim.Ly - sim.dy)
                new_y = np.random.uniform(ylow, yhigh, num_inject).astype(_NP_FP)
        else:
            new_y = np.random.uniform(sim.dy, sim.Ly - sim.dy, num_inject).astype(_NP_FP)

        new_x = np.full(num_inject, sim.dx * 1.5, dtype=_NP_FP)

        v_spread = np.sqrt(sim.q_ion * Ti / sim.m_ion)
        new_vx = np.full(num_inject, v_bohm, dtype=np.float64) + np.random.randn(num_inject).astype(np.float64) * v_spread
        new_vy = (np.random.randn(num_inject) * v_spread).astype(_NP_FP)
        new_vz = (np.random.randn(num_inject) * v_spread).astype(_NP_FP)
        new_cex = np.zeros(num_inject, dtype=bool)

        sim._add_ions(new_x, new_y, new_vx, new_vy, new_vz, new_cex)
        sim.injected_ions_step += float(num_inject)
        sim.injected_ions += float(num_inject)

    if params.get('rf_enable'):
        v_e_th_source = np.sqrt(2.0 * sim.q * Te_up / sim.m_e)
        I_e = sim.q * 0.25 * n0 * v_e_th_source * injection_area
        num_e_float = (I_e * sim.dt) / charge_per_macro
        num_inj_e = int(num_e_float)
        if np.random.rand() < (num_e_float - num_inj_e):
            num_inj_e += 1

        if num_inj_e > 0:
            y_margin = max(sim.dy, 1e-9)
            y_low = y_margin
            y_high = max(y_low, sim.Ly - y_margin)
            new_ex = np.full(num_inj_e, 0.1, dtype=_NP_FP)
            new_ey = np.random.uniform(y_low, y_high, num_inj_e).astype(_NP_FP)
            new_evx = (np.abs(np.random.randn(num_inj_e)) * v_e_th_source + v_bohm).astype(_NP_FP)
            new_evy = (np.random.randn(num_inj_e) * v_e_th_source).astype(_NP_FP)
            new_evz = (np.random.randn(num_inj_e) * v_e_th_source).astype(_NP_FP)
            sim._add_electrons(new_ex, new_ey, new_evx, new_evy, new_evz)
