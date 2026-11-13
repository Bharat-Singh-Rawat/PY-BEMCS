"""
Ion wall collisions, sputtering erosion, secondary electron emission, and beam transmission diagnostics.
"""
import numpy as np
from ..backend import USE_TAICHI, _NP_FP
from ..kernels_taichi import check_particle_boundaries_taichi
from .context import StepContext


def _get_boundary_scratch(sim, n: int):
    """Retrieve or allocate pre-allocated GPU scratch buffers on simulator instance."""
    if not hasattr(sim, '_boundary_scratch_size') or sim._boundary_scratch_size < n:
        alloc_size = max(n, getattr(sim, 'max_particles', 200000), 10000)
        sim._boundary_status_buf = np.empty(alloc_size, dtype=np.uint8)
        sim._boundary_impact_ix_buf = np.empty(alloc_size, dtype=np.int16)
        sim._boundary_impact_iy_buf = np.empty(alloc_size, dtype=np.int16)
        sim._boundary_counts_buf = np.zeros(8, dtype=np.int32)
        sim._boundary_scratch_size = alloc_size
    sim._boundary_counts_buf.fill(0)
    return (
        sim._boundary_status_buf[:n],
        sim._boundary_impact_ix_buf[:n],
        sim._boundary_impact_iy_buf[:n],
        sim._boundary_counts_buf
    )


def _get_is_bound_int(sim):
    """Retrieve or cached int32 version of isBound array."""
    ver = getattr(sim, '_damage_version', 0)
    if (not hasattr(sim, '_isBound_int') or
        sim._isBound_int is None or
        getattr(sim, '_isBound_int_ver', -1) != ver or
        sim._isBound_int.shape != sim.isBound.shape):
        sim._isBound_int = sim.isBound.astype(np.int32)
        sim._isBound_int_ver = ver
    return sim._isBound_int


def compute_beam_divergence(
    p_y: np.ndarray,
    p_vx: np.ndarray,
    p_vy: np.ndarray,
    mask: np.ndarray,
    method: str = "95%",
    div_percentile: float = 95.0,
    n_sigmas: float = 1.0,
    symmetric_half_domain: bool = False,
    is_axisymmetric: bool = False,
    p_vz: np.ndarray = None,
) -> float:
    """Compute the beam divergence (half-angle in degrees) for ions crossing a plane.

    Supported evaluation methods:
      - '95%': 95th percentile of |theta|
      - '90%': 90th percentile of |theta|
      - 'Direct RMS': raw Root Mean Square of particle angles: sqrt(mean(theta^2))
      - 'RMS (Twiss)': n_sigmas * sqrt(epsilon * gamma) from Courant-Snyder emittance ellipse
      - 'FWHM': half the full width at half-maximum of the angular distribution histogram peak
    """
    n_pts = int(np.count_nonzero(mask))
    if n_pts < 5:
        return np.nan

    y = p_y[mask]
    vx = p_vx[mask]
    vy = p_vy[mask]

    if is_axisymmetric and p_vz is not None:
        vz = p_vz[mask]
        v_perp = np.sqrt(vy ** 2 + vz ** 2)
        angles_deg = np.arctan2(v_perp, vx) * (180.0 / np.pi)
        angles_planar = np.arctan2(vy, vx) * (180.0 / np.pi)
    else:
        angles_deg = np.arctan2(vy, vx) * (180.0 / np.pi)
        angles_planar = angles_deg

    method_clean = str(method).strip().lower() if method is not None else "95%"

    if "direct" in method_clean and "rms" in method_clean:
        # Direct raw RMS of particle angles, skipping emittance ellipse fit
        return float(np.sqrt(np.mean(angles_deg ** 2)))

    elif "twiss" in method_clean:
        # RMS (Twiss): Divergence = n * sigmas * sqrt(epsilon * gamma)
        # In half_hole or axisymmetric symmetry (y >= 0), mirror {(y, theta)} -> {(-y, -theta)} so moments are centered.
        if symmetric_half_domain:
            y_eff = np.concatenate((y, -y))
            th_eff = np.concatenate((angles_planar, -angles_planar))
        else:
            y_eff = y
            th_eff = angles_planar

        y_mean = np.mean(y_eff)
        y_dev = y_eff - y_mean
        theta_mean = np.mean(th_eff)
        theta_dev = th_eff - theta_mean

        sigma_y_sq = np.mean(y_dev ** 2)
        sigma_theta_sq = np.mean(theta_dev ** 2)
        sigma_y_theta = np.mean(y_dev * theta_dev)

        # Statistical Courant-Snyder emittance
        eps_sq = max(0.0, sigma_y_sq * sigma_theta_sq - sigma_y_theta ** 2)
        eps = np.sqrt(eps_sq)

        if eps > 1e-15:
            gamma = sigma_theta_sq / eps
            rms_div = np.sqrt(eps * gamma)
        else:
            rms_div = np.sqrt(sigma_theta_sq)

        return float(n_sigmas * rms_div)

    elif "fwhm" in method_clean:
        # FWHM: Half the full width at half-maximum of angular distribution histogram peak
        # In half_hole or axisymmetric symmetry, mirror planar angles symmetrically around 0 deg.
        th_eff = np.concatenate((angles_planar, -angles_planar)) if symmetric_half_domain else angles_planar
        n_eff = len(th_eff)
        if n_pts < 15:
            # For small sample sizes, empirical histogram is sparse;
            # use Gaussian equivalent half-FWHM: sqrt(2 * ln(2)) * sigma
            sigma = np.std(th_eff)
            return float(np.sqrt(2.0 * np.log(2.0)) * sigma)

        theta_min = float(np.min(th_eff))
        theta_max = float(np.max(th_eff))
        if theta_max - theta_min < 1e-4:
            return 0.0

        n_bins = int(np.clip(np.sqrt(n_eff) * 1.5, 12, 50))
        counts, bin_edges = np.histogram(th_eff, bins=n_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 3-point moving average smoothing to eliminate discrete binning noise spikes
        kernel = np.array([0.2, 0.6, 0.2])
        smooth = np.convolve(counts, kernel, mode="same")
        peak_idx = int(np.argmax(smooth))
        max_val = float(smooth[peak_idx])

        if max_val <= 0:
            return 0.0

        half_max = 0.5 * max_val

        # Find left crossing via linear interpolation
        left_x = float(bin_centers[0])
        for i in range(peak_idx, 0, -1):
            if smooth[i - 1] <= half_max:
                denom = smooth[i] - smooth[i - 1]
                frac = (half_max - smooth[i - 1]) / denom if denom > 0 else 0.5
                left_x = float(bin_centers[i - 1] + frac * (bin_centers[i] - bin_centers[i - 1]))
                break

        # Find right crossing via linear interpolation
        right_x = float(bin_centers[-1])
        for i in range(peak_idx, len(smooth) - 1):
            if smooth[i + 1] <= half_max:
                denom = smooth[i] - smooth[i + 1]
                frac = (smooth[i] - half_max) / denom if denom > 0 else 0.5
                right_x = float(bin_centers[i] + frac * (bin_centers[i + 1] - bin_centers[i]))
                break

        fwhm = max(0.0, right_x - left_x)
        return float(0.5 * fwhm)

    else:
        # Percentile method (default '95%' or '90%')
        pct = 95.0
        if "90" in method_clean:
            pct = 90.0
        elif div_percentile is not None:
            pct = float(div_percentile)
        return float(np.percentile(np.abs(angles_deg), pct))


def step_ion_interactions(sim, ctx: StepContext):
    """Handle ion collisions with grid walls, thermal loads, sputter erosion, SEE, and exit diagnostics."""
    num_p_step = ctx.num_p_step
    p_x_old = ctx.p_x_old
    p_y_old = ctx.p_y_old
    grids = ctx.grids
    sim_mode = ctx.sim_mode
    params = ctx.params
    t_current = ctx.t_current

    p_x = sim.p_x[:num_p_step]
    p_y = sim.p_y[:num_p_step]
    p_vx = sim.p_vx[:num_p_step]
    p_vy = sim.p_vy[:num_p_step]
    p_vz = sim.p_vz[:num_p_step]
    p_cex = sim.p_isCEX[:num_p_step]
    

    if grids and hasattr(sim, "grid_x_starts") and len(sim.grid_x_starts) == len(grids):
        x_entry_first_grid = sim.grid_x_starts[0]
        x_exit_last = sim.grid_x_ends[-1]
    else:
        x_entry_first_grid = 0.5
        x_exit_last = 3.0

    periodic_y = getattr(sim, 'periodic_y', False)
    geometry = getattr(sim, 'geometry', 'half_hole')
    is_half_hole = (geometry == 'half_hole') and not periodic_y
    reflect_y0 = (not periodic_y) and getattr(sim, 'reflect_y0', is_half_hole)
    reflect_y_top = (not periodic_y) and getattr(sim, 'reflect_y_top', False)

    if USE_TAICHI and check_particle_boundaries_taichi is not None and num_p_step > 0:
        grid_x_starts = getattr(sim, 'grid_x_starts', None)
        grid_x_ends = getattr(sim, 'grid_x_ends', None)
        if grid_x_starts and grid_x_ends:
            grid_x_min = max(0.0, min(grid_x_starts) - 0.2)
            grid_x_max = min(sim.Lx, max(grid_x_ends) + 0.2)
        else:
            grid_x_min = 0.0
            grid_x_max = sim.Lx

        x_plane = getattr(sim, 'div_plane_x', None)
        if x_plane is None:
            x_plane = params.get('div_plane_x', None)
        if x_plane is None or not (x_exit_last <= x_plane <= sim.Lx):
            x_plane = x_exit_last + 0.5 * (sim.Lx - x_exit_last)

        status, impact_ix, impact_iy, counts = _get_boundary_scratch(sim, num_p_step)
        is_bound_int = _get_is_bound_int(sim)

        check_particle_boundaries_taichi(
            p_x, p_y, p_vx, p_x_old, p_y_old,
            is_bound_int, sim.zone_boundaries, sim.zone_dx, sim.zone_offsets,
            len(sim.zone_boundaries) - 1, num_p_step,
            sim.dy, sim.nx, sim.ny, sim.Lx, sim.Ly,
            1 if periodic_y else 0,
            grid_x_min, grid_x_max, x_entry_first_grid, x_exit_last, x_plane,
            8, status, impact_ix, impact_iy, counts,
            reflect_y0=(1 if reflect_y0 else 0),
            reflect_y_top=(1 if reflect_y_top else 0)
        )

        n_dead = int(counts[0])
        n_lost_to_grid = int(counts[1])
        n_lost_to_oob = int(counts[2])
        n_entered = int(counts[3])
        n_crossed_exit = int(counts[4])
        n_crossed_mid = int(counts[5])
        n_exited = int(counts[6])

        sim.entered_optics_step = float(n_entered)
        sim.entered_optics += float(n_entered)
        sim.prev_entered = sim.entered_optics_step
        if getattr(sim, 'first_grid_entry_iter', None) is None and sim.entered_optics > 0:
            sim.first_grid_entry_iter = getattr(sim, 'iteration', 0)

        sim.lost_to_oob_step = float(n_lost_to_oob)
        sim.lost_to_grid_step = float(n_lost_to_grid)

        # Thermal, SEE, and Erosion only if particles hit grid walls
        if n_lost_to_grid > 0:
            hit_grid_mask = (status & 1) != 0
            valid_thermal_hit = hit_grid_mask & (p_x > 0.25)
            if sim_mode in ("Thermal", "Both") and np.any(valid_thermal_hit):
                v_mag_sq = (
                    p_vx[valid_thermal_hit] ** 2 +
                    p_vy[valid_thermal_hit] ** 2 +
                    p_vz[valid_thermal_hit] ** 2
                )
                E_joules = 0.5 * sim.m_ion * v_mag_sq * sim.macro_weight
                dT_heat = (E_joules / sim.C_cell) * sim.thermal_accel
                np.add.at(sim.Tmap, (impact_iy[valid_thermal_hit], impact_ix[valid_thermal_hit]), dT_heat)
                sim._thermal_heat_added = True

            valid_see_hit = hit_grid_mask & (p_x > 0.25)
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

            is_erosion_hit = hit_grid_mask & (p_x > 0.25)
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
                    sim._isBound_int = None
                    sim.damage_map[broken_cells] = 0.0
                    sim.build_sparse_matrix()
                    ctx.remeshed = True

        # Exit diagnostics & Beam divergence
        if n_crossed_exit > 0:
            crossed_mask = (status & 16) != 0
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
            sim.exit_count_step = n_crossed_exit

            prim_exit = ~cex_exit
            sim.exit_v_mean_primary = float(np.mean(v_exit[prim_exit])) if np.any(prim_exit) else np.nan
            sim.exit_v_mean_cex = float(np.mean(v_exit[cex_exit])) if np.any(cex_exit) else np.nan

            div_method = getattr(sim, 'div_method', params.get('div_method', '95%'))
            div_pct = getattr(sim, 'div_percentile', params.get('div_percentile', 95.0))
            n_sigmas = getattr(sim, 'twiss_n_sigma', params.get('twiss_n_sigma', 1.0))
            is_axisym = bool(getattr(sim, 'is_axisymmetric', False))
            symmetric_half = is_half_hole or is_axisym

            ctx.current_div = compute_beam_divergence(
                p_y, p_vx, p_vy, crossed_mask,
                method=div_method, div_percentile=div_pct, n_sigmas=n_sigmas,
                symmetric_half_domain=symmetric_half,
                is_axisymmetric=is_axisym,
                p_vz=p_vz
            )
        else:
            sim.exit_vx_mean = np.nan
            sim.exit_v_mean = np.nan
            sim.exit_vx_std = np.nan
            sim.exit_v_std = np.nan
            sim.exit_energy_mean_eV = np.nan
            sim.exit_count_step = 0
            sim.exit_v_mean_primary = np.nan
            sim.exit_v_mean_cex = np.nan
            ctx.current_div = np.nan

        if n_crossed_mid > 0:
            crossed_mid_mask = (status & 32) != 0
            div_method = getattr(sim, 'div_method', params.get('div_method', '95%'))
            div_pct = getattr(sim, 'div_percentile', params.get('div_percentile', 95.0))
            n_sigmas = getattr(sim, 'twiss_n_sigma', params.get('twiss_n_sigma', 1.0))
            is_axisym = bool(getattr(sim, 'is_axisymmetric', False))
            symmetric_half = is_half_hole or is_axisym

            ctx.current_div_mid = compute_beam_divergence(
                p_y, p_vx, p_vy, crossed_mid_mask,
                method=div_method, div_percentile=div_pct, n_sigmas=n_sigmas,
                symmetric_half_domain=symmetric_half,
                is_axisymmetric=is_axisym,
                p_vz=p_vz
            )
        else:
            ctx.current_div_mid = np.nan

        sim.current_div = ctx.current_div
        sim.current_div_mid = ctx.current_div_mid

        if n_exited > 0:
            n_transmitted = n_crossed_exit
            sim.transmitted_ions_step = float(n_transmitted)
            sim.transmitted_ions += float(n_transmitted)
            sim.transmitted3_step = float(n_transmitted)

        charge_per_macro = sim.q_ion * sim.macro_weight
        sim.exit_ion_current_step = (sim.transmitted_ions_step * charge_per_macro) / sim.dt if sim.dt > 0 else 0.0
        sim.exit_ion_current_avg = (sim.transmitted_ions * charge_per_macro) / t_current if t_current > 0 else 0.0

        # Alive compaction
        n_alive = num_p_step - n_dead
        if n_alive < num_p_step:
            alive_mask = (status & 4) == 0
            sim.p_x[:n_alive] = p_x[alive_mask]
            sim.p_y[:n_alive] = p_y[alive_mask]
            sim.p_vx[:n_alive] = p_vx[alive_mask]
            sim.p_vy[:n_alive] = p_vy[alive_mask]
            sim.p_vz[:n_alive] = p_vz[alive_mask]
            sim.p_isCEX[:n_alive] = p_cex[alive_mask]
            sim.num_p = n_alive
        else:
            sim.num_p = num_p_step
        return

    entered_first_grid_mask = (
        (p_x_old < x_entry_first_grid) &
        (p_x >= x_entry_first_grid) &
        (p_vx > 0.0)
    )
    n_entered = int(np.count_nonzero(entered_first_grid_mask))
    sim.entered_optics_step = float(n_entered)
    sim.entered_optics += float(n_entered)
    sim.prev_entered = sim.entered_optics_step
    if getattr(sim, 'first_grid_entry_iter', None) is None and sim.entered_optics > 0:
        sim.first_grid_entry_iter = getattr(sim, 'iteration', 0)

    ix = sim._x_to_ix(p_x)
    iy = np.clip(np.round(p_y / sim.dy).astype(int), 0, sim.ny - 1)

    hit_grid_final = sim.isBound[iy, ix]
    hit_grid_path, path_ix, path_iy = sim._segment_hits_grid(p_x_old, p_y_old, p_x, p_y, samples=8)
    hit_grid = hit_grid_final | hit_grid_path

    impact_ix = np.where(hit_grid_path, path_ix, ix)
    impact_iy = np.where(hit_grid_path, path_iy, iy)

    out_of_bounds = (
        (p_x < 0.0) |
        (p_x > sim.Lx) |
        np.isnan(p_x) |
        np.isnan(p_y)
    )
    if not periodic_y:
        if not reflect_y0:
            out_of_bounds = out_of_bounds | (p_y < 0.0)
        if not reflect_y_top:
            out_of_bounds = out_of_bounds | (p_y > sim.Ly)
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
        sim._thermal_heat_added = True

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

    div_method = getattr(sim, 'div_method', params.get('div_method', '95%'))
    div_pct = getattr(sim, 'div_percentile', params.get('div_percentile', 95.0))
    n_sigmas = getattr(sim, 'twiss_n_sigma', params.get('twiss_n_sigma', 1.0))
    is_axisym = bool(getattr(sim, 'is_axisymmetric', False))
    symmetric_half = is_half_hole or is_axisym

    ctx.current_div = compute_beam_divergence(
        p_y, p_vx, p_vy, crossed_mask,
        method=div_method, div_percentile=div_pct, n_sigmas=n_sigmas,
        symmetric_half_domain=symmetric_half,
        is_axisymmetric=is_axisym,
        p_vz=p_vz
    )

    x_plane = getattr(sim, 'div_plane_x', None)
    if x_plane is None:
        x_plane = params.get('div_plane_x', None)
    if x_plane is None or not (x_exit_last <= x_plane <= sim.Lx):
        x_plane = x_exit_last + 0.5 * (sim.Lx - x_exit_last)

    crossed_mid_mask = (p_x_old <= x_plane) & (p_x > x_plane) & (p_vx > 0.0)
    ctx.current_div_mid = compute_beam_divergence(
        p_y, p_vx, p_vy, crossed_mid_mask,
        method=div_method, div_percentile=div_pct, n_sigmas=n_sigmas,
        symmetric_half_domain=symmetric_half,
        is_axisymmetric=is_axisym,
        p_vz=p_vz
    )

    sim.current_div = ctx.current_div
    sim.current_div_mid = ctx.current_div_mid

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
