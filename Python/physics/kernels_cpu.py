"""
Vectorized NumPy CPU fallbacks for charge deposition and Boris pusher.
"""
import numpy as np


def accumulate_rho_cpu(x, y, rho, num_p, dy, nx, ny, charge_per_particle,
                       zone_boundaries, zone_dx, zone_offsets, cell_vol_1d):
    """Vectorised NGP charge deposition on 3-zone piecewise-uniform mesh."""
    n = int(num_p)
    if n == 0:
        return
    z = np.searchsorted(zone_boundaries[1:-1], x[:n])
    ix = zone_offsets[z] + np.round((x[:n] - zone_boundaries[z]) / zone_dx[z]).astype(np.int32)
    ix = np.clip(ix, 1, nx - 2)
    iy = np.clip(np.round(y[:n] / dy).astype(np.int32), 1, ny - 2)
    np.add.at(rho, (iy, ix), charge_per_particle / cell_vol_1d[ix])


def push_particles_boris_cpu(x, y, vx, vy, vz,
                             Ex, Ey, Bx, By, Bz,
                             num_p, dy, nx, ny, dt, q_m,
                             zone_boundaries, zone_dx, zone_offsets, x_coords):
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
    Bx_p = interp(Bx);  By_p = interp(By);  Bz_p = interp(Bz)

    # —- Boris algorithm (vectorised) —-
    hqmdt = 0.5 * q_m * dt          # half charge-mass-time factor

    # Step 1: first half E-field kick  ->  v_minus
    vmx = vx[:n] + hqmdt * Ex_p
    vmy = vy[:n] + hqmdt * Ey_p
    vmz = vz[:n]  # no Ez in 2D

    # Step 2: magnetic rotation
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

    # Step 3: second half E-field kick
    vx[:n] = vpx2 + hqmdt * Ex_p
    vy[:n] = vpy2 + hqmdt * Ey_p
    vz[:n] = vpz2

    # Position update (mm units — same convention as Taichi kernel)
    x[:n] += vx[:n] * dt * 1000.0
    y[:n] += vy[:n] * dt * 1000.0
