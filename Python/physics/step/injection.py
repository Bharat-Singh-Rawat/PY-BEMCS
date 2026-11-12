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

    if sim.injection_stop_time is not None and t_current > sim.injection_stop_time:
        sim.injection_enabled = False
        return

    if not sim.injection_enabled:
        return

    n0 = params.get('n0_plasma', 1e17)
    Te_up = params.get('Te_up', 3.0)
    Ti = params.get('Ti', 0.1)

    v_bohm = np.sqrt(sim.q_ion * Te_up / sim.m_ion)
    is_axisymmetric = getattr(sim, 'is_axisymmetric', False)
    screen_r = grids[0]['r'] if grids else 0.80

    user_inj_y = params.get('inj_y_length_mm', params.get('inj_y_length', None))
    if user_inj_y is not None and float(user_inj_y) > 0.0:
        inj_y_span = min(float(user_inj_y), sim.Ly - sim.dy)
    else:
        inj_y_span = min(screen_r, sim.Ly - sim.dy) if grids else (sim.Ly - sim.dy)

    if 'injection_area_scale' in params:
        if is_axisymmetric:
            r_inj_m = inj_y_span * 1e-3
            injection_area = np.pi * (r_inj_m ** 2) * params['injection_area_scale']
        else:
            injection_area = sim.Ly * 1e-3 * 1e-3 * params['injection_area_scale']
    else:
        if is_axisymmetric:
            r_inj_mm = inj_y_span
            injection_area = np.pi * ((r_inj_mm * 1e-3) ** 2)
        else:
            _unit_depth_m = 1e-3
            geometry = getattr(sim, 'geometry', 'half_hole')
            if grids:
                pitch_inj = params.get('pitch_mm', 0.0)
                if geometry == 'two_holes':
                    hcs = getattr(sim, 'hole_centers', None)
                    yc0 = hcs[0] if hcs and len(hcs) > 0 else 0.5 * (screen_r + pitch_inj)
                    yc1 = hcs[1] if hcs and len(hcs) > 1 else 1.5 * (screen_r + pitch_inj)
                    h1 = max(0.0, min(yc0 + inj_y_span, sim.Ly - sim.dy) - max(sim.dy, yc0 - inj_y_span))
                    h2 = max(0.0, min(yc1 + inj_y_span, sim.Ly - sim.dy) - max(sim.dy, yc1 - inj_y_span))
                    inj_height_mm = h1 + h2
                elif geometry == 'one_hole':
                    hcs = getattr(sim, 'hole_centers', None)
                    yc = hcs[0] if hcs and len(hcs) > 0 else 0.5 * sim.Ly
                    inj_height_mm = max(0.0, min(yc + inj_y_span, sim.Ly - sim.dy) - max(sim.dy, yc - inj_y_span))
                else:
                    inj_height_mm = min(inj_y_span, sim.Ly - sim.dy)
            else:
                inj_height_mm = min(inj_y_span, sim.Ly - 2.0 * sim.dy)

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
        if is_axisymmetric:
            r_inj_mm = inj_y_span
            # Uniform density in circular aperture: r = r_inj * sqrt(U)
            new_y = (r_inj_mm * np.sqrt(np.random.uniform(0.0, 1.0, num_inject))).astype(_NP_FP)
        else:
            geometry = getattr(sim, 'geometry', 'half_hole')
            if grids:
                pitch_inj = params.get('pitch_mm', 0.0)
                if geometry == 'two_holes':
                    hcs = getattr(sim, 'hole_centers', None)
                    yc0 = hcs[0] if hcs and len(hcs) > 0 else 0.5 * (screen_r + pitch_inj)
                    yc1 = hcs[1] if hcs and len(hcs) > 1 else 1.5 * (screen_r + pitch_inj)
                    span_limits = np.array([
                        [max(sim.dy, yc0 - inj_y_span), min(yc0 + inj_y_span, sim.Ly - sim.dy)],
                        [max(sim.dy, yc1 - inj_y_span), min(yc1 + inj_y_span, sim.Ly - sim.dy)],
                    ], dtype=_NP_FP)
                    which_hole = np.random.randint(0, 2, size=num_inject)
                    y_lo  = span_limits[which_hole, 0]
                    y_hi  = span_limits[which_hole, 1]
                    new_y = (y_lo + (y_hi - y_lo) * np.random.rand(num_inject)).astype(_NP_FP)
                elif geometry == 'one_hole':
                    hcs = getattr(sim, 'hole_centers', None)
                    yc = hcs[0] if hcs and len(hcs) > 0 else 0.5 * sim.Ly
                    ylow  = max(sim.dy, yc - inj_y_span)
                    yhigh = min(yc + inj_y_span, sim.Ly - sim.dy)
                    new_y = np.random.uniform(ylow, yhigh, num_inject).astype(_NP_FP)
                else:
                    ylow  = 0.0
                    yhigh = min(inj_y_span, sim.Ly - sim.dy)
                    new_y = np.random.uniform(ylow, yhigh, num_inject).astype(_NP_FP)
            else:
                new_y = np.random.uniform(sim.dy, min(inj_y_span, sim.Ly - sim.dy), num_inject).astype(_NP_FP)

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
            if is_axisymmetric:
                r_inj_mm = inj_y_span
                new_ey = (r_inj_mm * np.sqrt(np.random.uniform(0.0, 1.0, num_inj_e))).astype(_NP_FP)
            else:
                y_margin = max(sim.dy, 1e-9)
                y_low = y_margin
                y_high = max(y_low, sim.Ly - y_margin)
                new_ey = np.random.uniform(y_low, y_high, num_inj_e).astype(_NP_FP)
            new_ex = np.full(num_inj_e, 0.1, dtype=_NP_FP)
            new_evx = (np.abs(np.random.randn(num_inj_e)) * v_e_th_source + v_bohm).astype(_NP_FP)
            new_evy = (np.random.randn(num_inj_e) * v_e_th_source).astype(_NP_FP)
            new_evz = (np.random.randn(num_inj_e) * v_e_th_source).astype(_NP_FP)
            sim._add_electrons(new_ex, new_ey, new_evx, new_evy, new_evz)
