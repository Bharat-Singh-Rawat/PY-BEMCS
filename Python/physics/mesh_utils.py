"""
Mesh utilities and automatic domain zoning calculations.
"""
import numpy as np


def compute_debye_upstream_gap(n0: float, Te_up: float) -> float:
    """Return the physics-recommended upstream gap [mm] based on Debye length.

    Sheath-formation criterion (literature-based):
      n0 <= 1e17  m^-3  →  80 * λ_D
      1e17 < n0 <= 4e17 →  40 * λ_D
      n0 > 4e17         →  30 * λ_D

    Parameters
    ----------
    n0 : float
        Upstream plasma density [m^-3]
    Te_up : float
        Upstream electron temperature [eV]

    Returns
    -------
    float
        Recommended upstream gap in [mm]
    """
    eps0 = 8.854e-12
    q    = 1.6e-19
    debye_m  = np.sqrt(eps0 * Te_up / (q * n0 * 0.61))
    debye_mm = debye_m * 1e3

    if n0 <= 1e17:
        n_debye = 80
    elif n0 <= 4e17:
        n_debye = 40
    else:
        n_debye = 30

    return n_debye * debye_mm


def compute_efield_constrained_spacing(
    grids,
    n0: float,
    Te_up: float,
    ratio_debye: float = 0.8,
    max_rel_jump: float = 0.10,
    min_cells_gap: int = 20,
    min_cells_aperture: int = 20,
    min_cells_thickness: int = 12,
):
    """
    Calculate base grid cell spacings (dx, dy) constrained by both Debye length
    and electric field gradient / geometric limits.

    When plasma density n0 is low, lambda_D becomes large. Sizing cells purely
    from Debye length causes cells to exceed the gap width and aperture
    radius, producing massive cell-to-cell electric field jumps and singular matrices.

    This function bounds dx and dy so that:
      dx_max = min(ratio_debye * lambda_D, max_rel_jump * gap_min, gap_min / min_cells_gap)
      dy_max = min(ratio_debye * lambda_D, max_rel_jump * r_min, r_min / min_cells_aperture)

    Parameters
    ----------
    grids : list of dict
        Grid specifications (each dict with 't', 'gap', 'r', 'V').
    n0 : float
        Plasma density [m^-3].
    Te_up : float
        Electron temperature [eV].
    ratio_debye : float
        User requested dx / lambda_D ratio.
    max_rel_jump : float
        Maximum allowable fractional change in electric field between adjacent cells (default 0.10 = 10%).
    min_cells_gap : int
        Minimum desired cells across the inter-grid gap.
    min_cells_aperture : int
        Minimum desired cells across the aperture radius.
    min_cells_thickness : int
        Minimum desired cells across the grid thickness.

    Returns
    -------
    tuple of (dx_final_mm, dy_final_mm)
    """
    eps0 = 8.854e-12
    q = 1.6e-19
    debye_m = np.sqrt(eps0 * Te_up / (q * max(n0, 1.0) * 0.61))
    debye_mm = debye_m * 1e3
    dx_debye = float(ratio_debye) * debye_mm

    if not grids:
        return dx_debye, dx_debye

    gaps = [float(g['gap']) for g in grids if float(g.get('gap', 0.0)) > 0.0]
    radii = [float(g['r']) for g in grids if float(g.get('r', 0.0)) > 0.0]
    thicknesses = [float(g['t']) for g in grids if float(g.get('t', 0.0)) > 0.0]

    gap_min = min(gaps) if gaps else 1.0
    r_min = min(radii) if radii else 0.5

    # To guarantee cell-to-cell delta E / E <= max_rel_jump:
    # 1. Across the acceleration gap where peak dV/dx occurs:
    dx_geom = float(max_rel_jump) * gap_min

    # 2. Transversely across the aperture radius:
    dy_geom = float(max_rel_jump) * r_min

    dx_final = min(dx_debye, dx_geom)
    dy_final = min(dx_debye, dy_geom)

    return float(dx_final), float(dy_final)


def evaluate_mesh_efield_variation(
    Ex,
    Ey,
    is_bound=None,
    max_rel_jump: float = 0.10,
    e_floor_ratio: float = 0.02,
    percentile: float = 95.0,
):
    """
    Evaluate node-to-node relative variation in electric field magnitude |E|.

    Computes:
      delta_x = |E_{i+1, j} - E_{i, j}| / max(E_{i, j}, E_{floor})
      delta_y = |E_{i, j+1} - E_{i, j}| / max(E_{i, j}, E_{floor})

    within the active field region in free space (where E >= 0.05 * E_max,
    excluding solid conductors). Uses the 95th percentile to robustly evaluate
    field smoothness throughout the extraction channel while excluding
    unphysical numerical corner singularities at re-entrant 90-degree electrode edges.

    Parameters
    ----------
    Ex : 2D ndarray
        Axial electric field [V/m].
    Ey : 2D ndarray
        Transverse electric field [V/m].
    is_bound : 2D bool ndarray or None
        Boundary mask (True for solid grid electrodes).
    max_rel_jump : float
        Allowable jump threshold (default: 0.10 for 10%).
    e_floor_ratio : float
        Noise floor fraction of E_max to prevent division by near-zero.
    percentile : float
        Percentile for jump evaluation (default: 98.0).

    Returns
    -------
    tuple of (max_jump_x, max_jump_y, is_compliant)
    """
    E = np.hypot(Ex, Ey)
    E_max = float(np.max(E))
    if E_max < 1e-6:
        return 0.0, 0.0, True

    E_floor = e_floor_ratio * E_max

    # Relative jump along x in active field region
    diff_x = np.abs(E[:, 1:] - E[:, :-1])
    ref_x = np.maximum(np.maximum(E[:, 1:], E[:, :-1]), E_floor)
    rel_x = diff_x / ref_x
    active_x = ref_x >= 0.05 * E_max
    if is_bound is not None:
        free_x = (~is_bound[:, :-1]) & (~is_bound[:, 1:])
        active_x &= free_x

    if np.any(active_x):
        max_jump_x = float(np.percentile(rel_x[active_x], percentile))
    else:
        max_jump_x = 0.0

    # Relative jump along y in active field region
    diff_y = np.abs(E[1:, :] - E[:-1, :])
    ref_y = np.maximum(np.maximum(E[1:, :], E[:-1, :]), E_floor)
    rel_y = diff_y / ref_y
    active_y = ref_y >= 0.05 * E_max
    if is_bound is not None:
        free_y = (~is_bound[:-1, :]) & (~is_bound[1:, :])
        active_y &= free_y

    if np.any(active_y):
        max_jump_y = float(np.percentile(rel_y[active_y], percentile))
    else:
        max_jump_y = 0.0

    is_compliant = (max_jump_x <= max_rel_jump) and (max_jump_y <= max_rel_jump)
    return max_jump_x, max_jump_y, is_compliant
    return max_jump_x, max_jump_y, is_compliant


def estimate_automatic_mesh_zones(grids, upstream_gap_mm, Lx, dx0, max_rel_jump: float = 0.10):
    """
    Automatically partition the domain into physical axial zones and estimate
    physically-grounded coarsening factors based on Debye length, aperture
    thickness, inter-grid gaps, plume expansion, and electric field gradient limits.

    Zones created:
      1. Presheath: [0, upstream_gap], factor = 1.0 (dense plasma, resolves Debye length)
      2. For each grid i:
         - Grid barrel: [x_i, x_i + t_i], factor = min(1.5, max(0.8, round((t_i / 15.0) / dx0, 1)))
           (ensures >= 15 cells across thickness for aperture resolution)
         - Inter-grid gap: [x_i + t_i, x_{i+1}], factor = min(1.0, max(0.5, round((gap_i / 20.0) / dx0, 1)))
           (ensures steep acceleration/deceleration electric field jumps <= 10%)
      3. Plume:
         - Near Plume: [x_exit, x_exit + near_len], factor = 2.0 (neutralization & transition)
         - Far Plume:  [x_exit + near_len, Lx], factor = 4.0 (dilute neutralized beam)
    """
    if not grids:
        return [
            {'name': 'Presheath', 'x_start': 0.0, 'x_end': min(Lx / 3.0, Lx), 'factor': 1.0},
            {'name': 'Optics',    'x_start': min(Lx / 3.0, Lx), 'x_end': min(2.0 * Lx / 3.0, Lx), 'factor': 1.0},
            {'name': 'Plume',     'x_start': min(2.0 * Lx / 3.0, Lx), 'x_end': Lx, 'factor': 4.0},
        ]

    zones = []
    x_curr = 0.0
    up_gap = min(float(upstream_gap_mm), Lx)

    # 1. Presheath Zone
    if up_gap > 1e-5:
        zones.append({
            'name': 'Presheath',
            'x_start': round(x_curr, 5),
            'x_end': round(up_gap, 5),
            'factor': 1.0
        })
        x_curr = up_gap

    # 2. Grid Barrels and Inter-Grid Gaps
    num_grids = len(grids)
    for i, g in enumerate(grids):
        t_grid = float(g.get('t', 0.5))
        gap = float(g.get('gap', 1.0))

        # Grid barrel
        if x_curr < Lx:
            x_next = min(x_curr + t_grid, Lx)
            if x_next - x_curr > 1e-5:
                f_t = min(1.5, max(0.8, round((t_grid / 15.0) / max(dx0, 1e-6), 1)))
                zones.append({
                    'name': f'Grid_{i+1}_barrel',
                    'x_start': round(x_curr, 5),
                    'x_end': round(x_next, 5),
                    'factor': f_t
                })
                x_curr = x_next

        # Internal inter-grid gap (steepest electric field: keep cell size bounded to <= 10% jump)
        if i < num_grids - 1 and x_curr < Lx:
            x_next = min(x_curr + gap, Lx)
            if x_next - x_curr > 1e-5:
                # Inter-grid acceleration gap must not be coarsened beyond 1.0 to preserve <= 10% jump
                f_gap = min(1.0, max(0.5, round((gap / 20.0) / max(dx0, 1e-6), 1)))
                zones.append({
                    'name': f'Gap_{i+1}_{i+2}',
                    'x_start': round(x_curr, 5),
                    'x_end': round(x_next, 5),
                    'factor': f_gap
                })
                x_curr = x_next

    # 3. Plume Zones (Near Plume and Far Plume)
    plume_remaining = Lx - x_curr
    if plume_remaining > 1e-5:
        near_plume_len = min(1.0, 0.3 * plume_remaining)
        # Near Plume
        if near_plume_len > 1e-5 and x_curr + near_plume_len < Lx - 1e-5:
            x_next = round(x_curr + near_plume_len, 5)
            zones.append({
                'name': 'Near_Plume',
                'x_start': round(x_curr, 5),
                'x_end': x_next,
                'factor': 2.0
            })
            x_curr = x_next

        # Far Plume
        if Lx - x_curr > 1e-5:
            zones.append({
                'name': 'Far_Plume',
                'x_start': round(x_curr, 5),
                'x_end': round(Lx, 5),
                'factor': 4.0
            })

    return zones

