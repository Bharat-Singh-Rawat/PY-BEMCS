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


def estimate_automatic_mesh_zones(grids, upstream_gap_mm, Lx, dx0):
    """
    Automatically partition the domain into physical axial zones and estimate
    physically-grounded coarsening factors based on Debye length, aperture
    thickness, inter-grid gaps, and plume expansion.

    Zones created:
      1. Presheath: [0, upstream_gap], factor = 1.0 (dense plasma, resolves Debye length)
      2. For each grid i:
         - Grid barrel: [x_i, x_i + t_i], factor = min(1.5, max(1.0, round((t_i / 15.0) / dx0, 1)))
           (ensures >= 15 cells across thickness for aperture resolution)
         - Inter-grid gap: [x_i + t_i, x_{i+1}], factor = min(2.0, max(1.0, round((gap_i / 20.0) / dx0, 1)))
           (resolves steep acceleration/deceleration electric fields)
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
                f_t = min(1.5, max(1.0, round((t_grid / 15.0) / max(dx0, 1e-6), 1)))
                zones.append({
                    'name': f'Grid_{i+1}_barrel',
                    'x_start': round(x_curr, 5),
                    'x_end': round(x_next, 5),
                    'factor': f_t
                })
                x_curr = x_next

        # Internal inter-grid gap
        if i < num_grids - 1 and x_curr < Lx:
            x_next = min(x_curr + gap, Lx)
            if x_next - x_curr > 1e-5:
                f_gap = min(2.0, max(1.0, round((gap / 20.0) / max(dx0, 1e-6), 1)))
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
