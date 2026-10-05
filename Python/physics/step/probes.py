"""
Centerline and mid-hole saddle point potential evaluation, plus PPC spatial binning.
"""
import numpy as np
from .context import StepContext


def step_midhole_probe(sim, ctx: StepContext):
    """Evaluate centerline/saddle potential at the axial midpoint of the first downstream grid."""
    grids = ctx.grids
    geometry = getattr(sim, 'geometry', 'half_hole')
    hole_centers = getattr(sim, 'hole_centers', None)

    if not hole_centers:
        if grids and geometry in ('one_hole', 'two_holes'):
            hole_centers = [1.5 * grids[0]['r']]
        else:
            hole_centers = [0.0]

    if len(grids) >= 2:
        x_grid_mm = 0.5 * (sim.grid_x_starts[1] + sim.grid_x_ends[1])
    elif len(grids) == 1:
        x_grid_mm = 0.5 * (sim.grid_x_starts[0] + sim.grid_x_ends[0])
    else:
        x_grid_mm = sim.Lx * 0.5

    x_idx = sim._x_to_ix(x_grid_mm)
    y_c_first = hole_centers[0]
    y_idx = int(np.clip(round(y_c_first / sim.dy), 0, sim.ny - 1))
    min_pot = float(sim.V[y_idx, x_idx])
    sim.min_pot = min_pot
    sim.saddle_point_potential = min_pot
    ctx.min_pot = min_pot


def step_end_accounting(sim, ctx: StepContext):
    """Compute per-cell particle counts, transparency, performance metrics, and build return tuple."""
    ctx.trans_last_frame = sim.get_third_grid_transparency_frame()

    p_x_active = sim.p_x[:sim.num_p]
    p_y_active = sim.p_y[:sim.num_p]

    ix = sim._x_to_ix(p_x_active)
    iy = np.clip(np.round(p_y_active / sim.dy).astype(int), 0, sim.ny - 1)

    flat_idx = iy * sim.nx + ix
    ppc_flat = np.bincount(flat_idx, minlength=sim.nx * sim.ny)
    ppc_map = ppc_flat.reshape((sim.ny, sim.nx))
    sim.current_ppc_map = ppc_map
    sim.accum_ppc_map += ppc_map
    sim.ppc_steps_count += 1

    thresh = getattr(sim, 'min_ppc_threshold', 3)
    low_ppc_mask = (ppc_map > 0) & (ppc_map < thresh)
    sim.total_active_cells = int(np.count_nonzero(ppc_map > 0))
    sim.low_ppc_cells = int(np.count_nonzero(low_ppc_mask))

    if sim._perf_monitor is not None:
        import time as _time_mod
        step_wall = _time_mod.perf_counter() - sim._step_t0
        sim._perf_monitor.record_step(sim, step_wall)

    return ctx.remeshed, ctx.min_pot, ctx.current_div, sim.T_grids, ctx.trans_last_frame
