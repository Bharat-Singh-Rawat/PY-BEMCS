"""
Mesh utilities and automatic domain zoning calculations.
"""
import math
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

    raw_gap = n_debye * debye_mm
    if raw_gap > 4.0:
        print(f"[Warning] Debye-evaluated upstream gap ({raw_gap:.3f} mm) exceeds 4 mm limit. Clamped to 4.0 mm.")
        return 4.0
    return raw_gap


def compute_upstream_gap_value(
    screen_r: float = 0.80,
    n0: float = 1e17,
    Te_up: float = 3.0,
    entire_bulk_plasma: bool = False,
    upstream_gap_settings: dict = None,
    force_base: str = None,
    ratio_override: float = None,
):
    """Compute coupled upstream gap [mm] matching GUI ratio/Debye rules, or None if invalid/custom."""
    if force_base == "screen_r":
        if screen_r <= 0.0:
            return None
        ratio = 0.75 if ratio_override is None else float(ratio_override)
        return round(ratio * float(screen_r), 3)

    if force_base == "debye":
        if n0 <= 0.0 or Te_up <= 0.0:
            return None
        if ratio_override is not None:
            debye_mm = float(np.sqrt(8.854e-12 * float(Te_up) / (1.6e-19 * float(n0) * 0.61)) * 1e3)
            val = float(ratio_override) * debye_mm
            if val > 4.0:
                print(f"[Warning] Debye-evaluated upstream gap ({val:.3f} mm) exceeds 4 mm limit. Clamped to 4.0 mm.")
                val = 4.0
            return round(val, 3)
        return round(float(compute_debye_upstream_gap(float(n0), float(Te_up))), 3)

    if isinstance(upstream_gap_settings, dict):
        method = upstream_gap_settings.get("method")
        if method == "custom":
            return None
        if method == "ratio":
            base = upstream_gap_settings.get("ratio_base", "screen_r")
            ratio = float(upstream_gap_settings.get("ratio", 0.75))
            return compute_upstream_gap_value(
                screen_r=screen_r,
                n0=n0,
                Te_up=Te_up,
                entire_bulk_plasma=entire_bulk_plasma,
                force_base=base,
                ratio_override=ratio,
            )

    if entire_bulk_plasma:
        return compute_upstream_gap_value(n0=n0, Te_up=Te_up, force_base="debye")
    return compute_upstream_gap_value(screen_r=screen_r, force_base="screen_r", ratio_override=0.75)


def get_upstream_gap_coupling_for_param(
    param_id: str,
    upstream_gap_settings: dict = None,
    entire_bulk_plasma: bool = False,
):
    """Return coupling descriptor `{'base', 'ratio'}` if `param_id` influences upstream_gap_mm, else None."""
    settings = upstream_gap_settings if isinstance(upstream_gap_settings, dict) else {}
    method = settings.get("method")
    ratio_base = settings.get("ratio_base")

    if param_id == "grid_1_r":
        ratio = float(settings.get("ratio", 0.75)) if (method == "ratio" and ratio_base == "screen_r") else 0.75
        return {"base": "screen_r", "ratio": ratio}

    if param_id in ("n0_plasma", "Te_up"):
        if method == "ratio" and ratio_base == "debye":
            return {"base": "debye", "ratio": float(settings.get("ratio", 30.0))}
        return {"base": "debye", "ratio": None}

    return None


def resolve_swept_upstream_gap(param_id: str, params: dict, coupling_meta: dict = None):
    """Compute the updated upstream_gap_mm for a swept scenario dict, or None if not applicable."""
    if not isinstance(params, dict):
        return None
    coupling = (
        coupling_meta
        if (isinstance(coupling_meta, dict) and coupling_meta.get("base") in ("screen_r", "debye"))
        else get_upstream_gap_coupling_for_param(
            param_id,
            upstream_gap_settings=params.get("upstream_gap_settings"),
            entire_bulk_plasma=bool(params.get("entire_bulk_plasma", False)),
        )
    )
    if not coupling:
        return None

    base = coupling.get("base")
    if not (
        (param_id == "grid_1_r" and base == "screen_r")
        or (param_id in ("n0_plasma", "Te_up") and base == "debye")
    ):
        return None

    grids = params.get("grids", [])
    screen_r = float(grids[0].get("r", 0.80)) if grids else 0.80
    return compute_upstream_gap_value(
        screen_r=screen_r,
        n0=float(params.get("n0_plasma", 1e17)),
        Te_up=float(params.get("Te_up", 3.0)),
        force_base=base,
        ratio_override=coupling.get("ratio"),
    )


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


def estimate_automatic_mesh_zones(
    grids,
    upstream_gap_mm,
    Lx,
    dx0,
    max_rel_jump: float = 0.10,
    n0: float = 1e17,
    Te_up: float = 3.0,
    V_plasma_offset: float = 20.0,
    m_ion: float = 131.293 * 1.6605e-27,
    q_ion: float = 1.602e-19,
    ratio_debye: float = 0.8,
    entire_bulk_plasma: bool = False,
    is_axisymmetric: bool = False,
):
    """
    Automatically partition the domain into physical axial zones and pre-size
    physically-grounded coarsening factors across all sections (Option B):
      - Far Presheath & Sheath/Meniscus (or unified Presheath if gap is short)
      - Grid Barrels (resolving aperture thickness >= 12 cells without sub-dx0 over-refinement)
      - Inter-Grid Acceleration/Deceleration Gaps (local continuity Debye length & E-field jump bound,
        preventing 0.5x hyper-refinement that triggers unnecessary Boris substepping)
      - Near Plume & Far Plume (pre-sized from 1D continuity ion velocity/density drop and local
        Debye length expansion, bounded by aperture geometry)
    """
    if not grids:
        return [
            {'name': 'Presheath', 'x_start': 0.0, 'x_end': min(Lx / 3.0, Lx), 'factor': 1.0},
            {'name': 'Optics',    'x_start': min(Lx / 3.0, Lx), 'x_end': min(2.0 * Lx / 3.0, Lx), 'factor': 1.0},
            {'name': 'Plume',     'x_start': min(2.0 * Lx / 3.0, Lx), 'x_end': Lx, 'factor': 4.0},
        ]

    eps0 = 8.854e-12
    q_e = 1.602e-19
    n0_safe = max(float(n0), 1e10)
    Te_safe = max(float(Te_up), 0.1)
    m_i_safe = max(float(m_ion), 1e-30)
    q_i_safe = max(float(q_ion), 1e-22)
    dx0_safe = max(float(dx0), 1e-6)
    r_deb = float(ratio_debye) if (ratio_debye is not None and float(ratio_debye) > 0.0) else 0.8
    bohm_fac = 1.0 if entire_bulk_plasma else 0.61

    # Upstream Debye length [mm] and Bohm velocity [m/s]
    debye_bohm_mm = math.sqrt(eps0 * Te_safe / (q_e * n0_safe * bohm_fac)) * 1e3
    debye_unscreened_mm = math.sqrt(eps0 * Te_safe / (q_e * n0_safe)) * 1e3
    v_bohm = math.sqrt(q_i_safe * Te_safe / m_i_safe)
    v_Screen0 = float(grids[0].get('V', 1000.0))
    V_plasma = v_Screen0 + float(V_plasma_offset)

    zones = []
    x_curr = 0.0
    up_gap = min(float(upstream_gap_mm), Lx)

    # 1. Presheath Zone(s): Split into Far_Presheath and Sheath_Meniscus if upstream gap is long enough
    if up_gap > 1e-5:
        r_screen = float(grids[0].get('r', 0.8))
        meniscus_len = min(0.55 * up_gap, max(6.0 * debye_bohm_mm, 0.35 * r_screen, 6.0 * dx0_safe))
        far_pre_len = up_gap - meniscus_len

        # Only split if Far_Presheath has at least 8 cells and allows meaningful coarsening (>= 1.2x)
        dx_far_pre_max = min(max(r_deb, 1.0) * debye_unscreened_mm, far_pre_len / 8.0) if far_pre_len > 0 else dx0_safe
        f_far_pre = round(min(1.5, max(1.0, dx_far_pre_max / dx0_safe)), 1)

        if far_pre_len >= 8.0 * dx0_safe and f_far_pre >= 1.2:
            x_split = round(far_pre_len, 5)
            zones.append({
                'name': 'Far_Presheath',
                'x_start': round(x_curr, 5),
                'x_end': x_split,
                'factor': f_far_pre
            })
            zones.append({
                'name': 'Sheath_Meniscus',
                'x_start': x_split,
                'x_end': round(up_gap, 5),
                'factor': 1.0
            })
        else:
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

        # Grid barrel: decoupled SubgridManager resolves 5-10 um surface normals/erosion,
        # so Eulerian grid needs >= 10-12 cells across thickness without sub-1.0x over-refinement
        # unless the grid is very thin compared to dx0.
        if x_curr < Lx:
            x_next = min(x_curr + t_grid, Lx)
            if x_next - x_curr > 1e-5:
                f_t = min(1.5, max(0.8, round((t_grid / 12.0) / dx0_safe, 1)))
                zones.append({
                    'name': f'Grid_{i+1}_barrel',
                    'x_start': round(x_curr, 5),
                    'x_end': round(x_next, 5),
                    'factor': f_t
                })
                x_curr = x_next

        # Internal inter-grid gap: size via local continuity ion density & E-field jump limit
        if i < num_grids - 1 and x_curr < Lx:
            x_next = min(x_curr + gap, Lx)
            if x_next - x_curr > 1e-5:
                v_g1 = float(g.get('V', 1000.0))
                v_g2 = float(grids[i + 1].get('V', -200.0))
                v_mid = 0.5 * (v_g1 + v_g2)
                dV_mid = max(Te_safe, V_plasma - v_mid)
                v_ion_gap = math.sqrt(v_bohm**2 + 2.0 * q_i_safe * dV_mid / m_i_safe)
                n_gap = max(1e12, bohm_fac * n0_safe * (v_bohm / max(v_ion_gap, v_bohm)))
                lambda_D_gap_mm = math.sqrt(eps0 * Te_safe / (q_e * n_gap)) * 1e3

                # Bound gap cell size by local Debye length, E-field relative jump limit, and >= 16 cells across gap
                dx_gap_max = min(
                    r_deb * lambda_D_gap_mm,
                    float(max_rel_jump) * gap,
                    gap / 16.0
                )
                # Never hyper-refine below 0.8x (prevents 0.5x Boris substep explosion); allow up to 1.2x at ultra-high n0
                f_gap = min(1.2, max(0.8, round(dx_gap_max / dx0_safe, 1)))
                zones.append({
                    'name': f'Gap_{i+1}_{i+2}',
                    'x_start': round(x_curr, 5),
                    'x_end': round(x_next, 5),
                    'factor': f_gap
                })
                x_curr = x_next

    # 3. Plume Zones (Near Plume and Far Plume): Analytical continuity & local Debye pre-sizing
    plume_remaining = Lx - x_curr
    if plume_remaining > 1e-5:
        r_screen = max(float(grids[0].get('r', 0.8)), 0.1)
        r_last = max(float(grids[-1].get('r', r_screen)), 0.1)
        v_last = float(grids[-1].get('V', -180.0))
        dV_exit = max(50.0, V_plasma - max(v_last, -500.0))
        v_exit = math.sqrt(v_bohm**2 + 2.0 * q_i_safe * dV_exit / m_i_safe)

        # Geometric expansion factor in plume (conservative core radius ~ max(r_last, 0.85 * r_screen))
        r_core_est = max(r_last, 0.85 * r_screen)
        area_ratio = (r_screen / r_core_est)**2 if is_axisymmetric else (r_screen / r_core_est)
        T_opt_est = 0.85

        # Expected steady-state beam core ion density [m^-3] and local plume Debye length [mm]
        n_plume_est = max(1e12, bohm_fac * n0_safe * T_opt_est * (v_bohm / max(v_exit, v_bohm)) * area_ratio)
        lambda_D_plume_mm = math.sqrt(eps0 * Te_safe / (q_e * n_plume_est)) * 1e3

        near_plume_len = min(1.0, 0.3 * plume_remaining)
        far_plume_len = max(0.2, plume_remaining - near_plume_len)

        dx_far_debye = r_deb * lambda_D_plume_mm
        dx_far_geom = min(0.45 * r_last, far_plume_len / 10.0)
        dx_far_target = min(dx_far_debye, dx_far_geom)

        f_far = max(2.0, min(8.0, round(dx_far_target / dx0_safe, 1)))
        f_near = round(min(3.5, max(1.5, 0.5 * (1.0 + f_far))), 1)

        # Near Plume
        if near_plume_len > 1e-5 and x_curr + near_plume_len < Lx - 1e-5:
            x_next = round(x_curr + near_plume_len, 5)
            zones.append({
                'name': 'Near_Plume',
                'x_start': round(x_curr, 5),
                'x_end': x_next,
                'factor': f_near
            })
            x_curr = x_next

        # Far Plume
        if Lx - x_curr > 1e-5:
            zones.append({
                'name': 'Far_Plume',
                'x_start': round(x_curr, 5),
                'x_end': round(Lx, 5),
                'factor': f_far
            })

    return zones


def estimate_automatic_plume_mesh_zones(
    sim,
    grids: list,
    upstream_gap_mm: float,
    Lx: float,
    dx0: float,
    remesh_cfg: dict = None,
    max_rel_jump: float = 0.10,
) -> list:
    """
    Generate an adapted non-uniform axial mesh partitioned dynamically where:
    - Upstream presheath strictly maintains baseline factor = 1.0 (no coarsening, no meniscus refining).
    - Grid barrels and inter-grid optics gaps strictly maintain baseline resolution.
    - Plume zones (Near Plume and Far Plume) are automatically adapted based on local
      plume ion density, Debye length expansion, and beam kinematics.
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

    # Preserve upstream presheath and optics zones from existing zone_configs if available
    existing_zones = getattr(sim, 'zone_configs', None) if sim is not None else None
    x_exit_expected = up_gap + (sum(float(g.get('t', 0.5)) + float(g.get('gap', 1.0)) for g in grids) - float(grids[-1].get('gap', 1.0)))
    x_exit_expected = min(x_exit_expected, Lx)

    non_plume_existing = (
        [dict(z) for z in existing_zones if 'plume' not in z.get('name', '').lower()]
        if isinstance(existing_zones, list) and len(existing_zones) > 0
        else []
    )

    if non_plume_existing and abs(float(non_plume_existing[-1].get('x_end', 0.0)) - x_exit_expected) <= 1e-3:
        zones.extend(non_plume_existing)
        x_curr = float(non_plume_existing[-1]['x_end'])
    else:
        # Fallback: compute upstream and optics zones via estimate_automatic_mesh_zones
        base_zones = estimate_automatic_mesh_zones(
            grids=grids,
            upstream_gap_mm=upstream_gap_mm,
            Lx=Lx,
            dx0=dx0,
            max_rel_jump=max_rel_jump,
            n0=float(getattr(sim, 'n0_plasma', getattr(sim, 'n0', 1e17))) if sim is not None else 1e17,
            Te_up=float(getattr(sim, 'Te_up', 3.0)) if sim is not None else 3.0,
            m_ion=float(getattr(sim, 'm_ion', 131.293 * 1.6605e-27)) if sim is not None else 131.293 * 1.6605e-27,
            q_ion=float(getattr(sim, 'q_ion', 1.602e-19)) if sim is not None else 1.602e-19,
            is_axisymmetric=bool(getattr(sim, 'is_axisymmetric', False)) if sim is not None else False,
        )
        for z in base_zones:
            if 'plume' not in z.get('name', '').lower():
                zones.append(z)
                x_curr = float(z['x_end'])

    # 3. Plume Zones: Automatic determination based on local beam core density / Debye ratio
    x_exit = x_curr
    f_far = getattr(sim, '_current_plume_factor', 4.0) if sim is not None else 4.0
    if sim is not None and getattr(sim, 'num_p', 0) > 5:
        try:
            num_p = sim.num_p
            p_x = sim.p_x[:num_p]
            plume_mask = (p_x >= x_exit) & (p_x <= Lx)
            n_in_plume = int(np.sum(plume_mask))

            # Wait until ions have populated at least 60% of the plume length before adapting
            plume_span_mm = max(1e-4, Lx - x_exit)
            max_px_plume = float(np.max(p_x[plume_mask])) if n_in_plume > 0 else x_exit
            plume_fill_frac = (max_px_plume - x_exit) / plume_span_mm

            if n_in_plume < 15 or plume_fill_frac < 0.60:
                sim._plume_remesh_deferred = True
            else:
                sim._plume_remesh_deferred = False
                plume_len_m = max(1e-4, (max_px_plume - x_exit) * 1e-3)
                p_y = sim.p_y[:num_p][plume_mask]

                # Evaluate actual beam envelope radius r_core (90th percentile of |y|)
                # Bounded by screen grid aperture radius and domain boundary Ly
                r_screen = float(grids[0].get('r', 1.0)) if grids else 1.0
                r_last = float(grids[-1].get('r', r_screen)) if grids else r_screen
                if len(p_y) > 0:
                    r_core_mm = max(float(np.percentile(np.abs(p_y), 90)), r_last * 0.5, 0.3)
                else:
                    r_core_mm = r_screen
                Ly = float(getattr(sim, 'Ly', r_core_mm))
                r_core_mm = min(r_core_mm, Ly)

                is_axi = getattr(sim, 'is_axisymmetric', False)
                if is_axi:
                    vol_core_m3 = math.pi * (r_core_mm * 1e-3)**2 * plume_len_m
                else:
                    vol_core_m3 = (r_core_mm * 1e-3) * 1e-3 * plume_len_m

                macro_weight = float(getattr(sim, 'macro_weight', 1.0))
                n_phys_ions = n_in_plume * macro_weight
                n_core_avg = n_phys_ions / max(vol_core_m3, 1e-15)

                if n_core_avg > 0:
                    eps0 = 8.854e-12
                    q = 1.6e-19
                    Te_plume = float(getattr(sim, 'Te_up', 3.0))
                    # Local plume Debye length [mm] from actual beam core density
                    lambda_D_plume_mm = math.sqrt(eps0 * Te_plume / (q * max(n_core_avg, 1e12))) * 1e3
                    ratio_debye = float(remesh_cfg.get("dx_over_debye", 0.8)) if isinstance(remesh_cfg, dict) else 0.8

                    # Maximum allowable cell size in far plume bounded by local Debye length and geometric smoothness
                    far_len_mm = max(0.2, plume_span_mm * 0.7)
                    dx_far_debye = ratio_debye * lambda_D_plume_mm
                    dx_far_geom = min(0.45 * r_last, far_len_mm / 10.0)
                    dx_far_target = min(dx_far_debye, dx_far_geom)

                    auto_factor = dx_far_target / max(dx0, 1e-6)
                    # Clamped sensibly between 1.5x and 8.0x
                    f_far = max(1.5, min(8.0, round(auto_factor, 1)))
        except Exception as exc:
            print(f"[Plume Remesh Warning] Failed to estimate live plume factor: {exc}")
            f_far = getattr(sim, '_current_plume_factor', 4.0)

    f_near = round(min(3.5, max(1.2, 0.5 * (1.0 + f_far))), 1)

    plume_remaining = Lx - x_curr
    if plume_remaining > 1e-5:
        near_plume_len = min(1.0, 0.3 * plume_remaining)
        if near_plume_len > 1e-5 and x_curr + near_plume_len < Lx - 1e-5:
            x_next = round(x_curr + near_plume_len, 5)
            zones.append({
                'name': 'Near_Plume',
                'x_start': round(x_curr, 5),
                'x_end': x_next,
                'factor': f_near
            })
            x_curr = x_next

        if Lx - x_curr > 1e-5:
            zones.append({
                'name': 'Far_Plume',
                'x_start': round(x_curr, 5),
                'x_end': round(Lx, 5),
                'factor': f_far
            })

    return zones


def estimate_stability_optimized_mesh_zones(
    sim,
    grids: list,
    upstream_gap_mm: float,
    Lx: float,
    dx0: float,
    remesh_cfg: dict = None,
    max_rel_jump: float = 0.10,
) -> list:
    """Backward-compatible alias pointing to estimate_automatic_plume_mesh_zones."""
    return estimate_automatic_plume_mesh_zones(
        sim=sim,
        grids=grids,
        upstream_gap_mm=upstream_gap_mm,
        Lx=Lx,
        dx0=dx0,
        remesh_cfg=remesh_cfg,
        max_rel_jump=max_rel_jump,
    )


def compute_plume_mean_dx(sim, zone_configs=None) -> float:
    """Compute the weighted mean cell size Delta x [mm] across downstream plume zones."""
    if sim is None:
        return 0.04
    Lx = float(getattr(sim, 'Lx', 10.0))
    dx0 = float(getattr(sim, 'dx', 0.04))
    cfg = zone_configs or getattr(sim, 'zone_configs', None)
    if not cfg:
        return dx0

    plume_zones = [z for z in cfg if 'plume' in z.get('name', '').lower()]
    if not plume_zones:
        return dx0

    total_len = 0.0
    weighted_dx = 0.0
    for z in plume_zones:
        z_len = max(0.0, float(z.get('x_end', 0.0)) - float(z.get('x_start', 0.0)))
        z_dx = dx0 * float(z.get('factor', 1.0))
        total_len += z_len
        weighted_dx += z_len * z_dx

    if total_len > 0:
        return weighted_dx / total_len
    return dx0

