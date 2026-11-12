"""
Vectorized NumPy CPU fallbacks for charge deposition and Boris pusher.
"""
import numpy as np


def accumulate_rho_cpu(x, y, rho, num_p, dy, nx, ny, charge_per_particle,
                       zone_boundaries, zone_dx, zone_offsets, cell_vol_1d,
                       cvol_r=None, cvol_x=None, is_axisymmetric=False):
    """Vectorised NGP charge deposition on 3-zone piecewise-uniform mesh."""
    n = int(num_p)
    if n == 0:
        return
    z = np.searchsorted(zone_boundaries[1:-1], x[:n])
    ix = zone_offsets[z] + np.round((x[:n] - zone_boundaries[z]) / zone_dx[z]).astype(np.int32)
    ix = np.clip(ix, 1, nx - 2)
    min_iy = 0 if is_axisymmetric else 1
    iy = np.clip(np.round(y[:n] / dy).astype(np.int32), min_iy, ny - 2)
    if cvol_r is not None and cvol_x is not None:
        np.add.at(rho, (iy, ix), charge_per_particle / (cvol_r[iy] * cvol_x[ix]))
    elif getattr(cell_vol_1d, 'ndim', 1) == 2:
        np.add.at(rho, (iy, ix), charge_per_particle / cell_vol_1d[iy, ix])
    else:
        np.add.at(rho, (iy, ix), charge_per_particle / cell_vol_1d[ix])


def push_particles_boris_cpu(x, y, vx, vy, vz,
                             Ex, Ey, Bx, By, Bz,
                             num_p, dy, nx, ny, dt, q_m,
                             zone_boundaries, zone_dx, zone_offsets, x_coords,
                             Ly=0.0, reflect_y0=False, reflect_y_top=False,
                             is_axisymmetric=False):
    """Vectorised 2D3V Boris pusher on 3-zone piecewise-uniform mesh."""
    n = int(num_p)
    if n == 0:
        return

    nx_m1 = nx - 1
    ny_m1 = ny - 1

    px = x[:n]
    py = y[:n]

    z = np.searchsorted(zone_boundaries[1:-1], px)
    ix0 = zone_offsets[z] + np.floor((px - zone_boundaries[z]) / zone_dx[z]).astype(np.int32)
    ix0 = np.clip(ix0, 0, nx_m1 - 1)

    idx_y = py / dy
    iy0 = np.clip(np.floor(idx_y).astype(np.int32), 0, ny_m1 - 1)

    dx_local = x_coords[ix0 + 1] - x_coords[ix0]
    fx  = np.clip((px - x_coords[ix0]) / dx_local, 0.0, 1.0)
    fy  = idx_y - iy0
    ix1 = np.minimum(ix0 + 1, nx_m1)
    iy1 = np.minimum(iy0 + 1, ny_m1)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx          * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx          * fy

    def interp(F):
        return F[iy0, ix0]*w00 + F[iy0, ix1]*w10 + F[iy1, ix0]*w01 + F[iy1, ix1]*w11

    Ex_p = interp(Ex);  Ey_p = interp(Ey)

    # —- Boris algorithm (vectorised) —-
    hqmdt = 0.5 * q_m * dt          # half charge-mass-time factor

    # Step 1: first half E-field kick  ->  v_minus
    vmx = vx[:n] + hqmdt * Ex_p
    vmy = vy[:n] + hqmdt * Ey_p
    vmz = vz[:n]  # no Ez in 2D

    # Step 2: magnetic rotation (skipped if B is zero)
    has_B = (Bx is not None and np.any(Bx)) or (By is not None and np.any(By)) or (Bz is not None and np.any(Bz))
    if has_B:
        Bx_p = interp(Bx);  By_p = interp(By);  Bz_p = interp(Bz)
        tx = hqmdt * Bx_p
        ty = hqmdt * By_p
        tz = hqmdt * Bz_p
        t2 = tx*tx + ty*ty + tz*tz
        sx = 2.0 * tx / (1.0 + t2)
        sy = 2.0 * ty / (1.0 + t2)
        sz = 2.0 * tz / (1.0 + t2)

        # v_prime = v_minus + v_minus x t
        vpx = vmx + (vmy*tz - vmz*ty)
        vpy = vmy + (vmz*tx - vmx*tz)
        vpz = vmz + (vmx*ty - vmy*tx)

        # v_plus = v_minus + v_prime x s
        vpx2 = vmx + (vpy*sz - vpz*sy)
        vpy2 = vmy + (vpz*sx - vpx*sz)
        vpz2 = vmz + (vpx*sy - vpy*sx)
    else:
        vpx2 = vmx
        vpy2 = vmy
        vpz2 = vmz

    # Step 3: second half E-field kick
    vx[:n] = vpx2 + hqmdt * Ex_p
    vy[:n] = vpy2 + hqmdt * Ey_p
    vz[:n] = vpz2

    # Position update (mm units — same convention as Taichi kernel)
    x[:n] += vx[:n] * dt * 1000.0

    if is_axisymmetric:
        # Cylindrical 2D3V coordinate rotation (Birdsall & Langdon local-Cartesian rotation)
        # Advance in particle's local cross-sectional plane: Y = r, Z = 0
        y_tmp = y[:n] + vy[:n] * dt * 1000.0
        z_tmp = vz[:n] * dt * 1000.0
        r_new = np.sqrt(y_tmp ** 2 + z_tmp ** 2)
        nz = r_new > 1e-12
        if np.any(nz):
            cos_a = y_tmp[nz] / r_new[nz]
            sin_a = z_tmp[nz] / r_new[nz]
            vr_rot = vy[:n][nz] * cos_a + vz[:n][nz] * sin_a
            vt_rot = -vy[:n][nz] * sin_a + vz[:n][nz] * cos_a
            vy[:n][nz] = vr_rot
            vz[:n][nz] = vt_rot
        z_exact = ~nz
        if np.any(z_exact):
            vy[:n][z_exact] = -vy[:n][z_exact]
        y[:n] = r_new
        if reflect_y_top and Ly > 0.0:
            over = y[:n] > Ly
            if np.any(over):
                y[:n][over] = 2.0 * Ly - y[:n][over]
                vy[:n][over] = -vy[:n][over]
    else:
        y[:n] += vy[:n] * dt * 1000.0
        # Specular reflection at transverse symmetry boundaries
        if reflect_y0 or (reflect_y_top and Ly > 0.0):
            py_sl = y[:n]
            vy_sl = vy[:n]
            for _ in range(2):
                if reflect_y0:
                    neg = py_sl < 0.0
                    if np.any(neg):
                        py_sl[neg] = -py_sl[neg]
                        vy_sl[neg] = -vy_sl[neg]
                if reflect_y_top and Ly > 0.0:
                    over = py_sl > Ly
                    if np.any(over):
                        py_sl[over] = 2.0 * Ly - py_sl[over]
                        vy_sl[over] = -vy_sl[over]
