"""
Taichi GPU / accelerated kernels for particle deposition, Boris push, and thermal conduction.
"""
from .backend import USE_TAICHI, _TI_FP, _NP_FP
import numpy as np

if USE_TAICHI:
    import taichi as ti

    @ti.kernel
    def accumulate_rho_taichi(
        x: ti.types.ndarray(dtype=_TI_FP),
        y: ti.types.ndarray(dtype=_TI_FP),
        rho: ti.types.ndarray(dtype=_TI_FP),
        cell_vol: ti.types.ndarray(dtype=_TI_FP),
        num_p: ti.i32,
        dy: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        charge_per_particle: ti.f32,
        zone_b: ti.types.ndarray(dtype=_TI_FP),
        zone_dx: ti.types.ndarray(dtype=_TI_FP),
        zone_off: ti.types.ndarray(dtype=ti.i32),
        num_zones: ti.i32
    ):
        """Parallel density accumulation with generic N-zone O(1) cell lookup."""
        for i in range(num_p):
            px = x[i]
            py = y[i]
            z = 0
            while z < num_zones - 1 and px >= zone_b[z + 1]:
                z += 1

            ix = ti.cast(ti.round((px - zone_b[z]) / zone_dx[z]), ti.i32) + zone_off[z]
            iy = ti.cast(ti.round(py / dy), ti.i32)

            # Clamp to avoid out-of-bounds access
            ix = ti.max(1, ti.min(ix, nx - 2))
            iy = ti.max(1, ti.min(iy, ny - 2))

            # Taichi handles atomic additions automatically on the GPU
            rho[iy, ix] += charge_per_particle / cell_vol[ix]

    @ti.kernel
    def _push_particles_boris_taichi_kernel(
        x:   ti.types.ndarray(dtype=_TI_FP),
        y:   ti.types.ndarray(dtype=_TI_FP),
        vx:  ti.types.ndarray(dtype=_TI_FP),
        vy:  ti.types.ndarray(dtype=_TI_FP),
        vz:  ti.types.ndarray(dtype=_TI_FP),
        Ex:  ti.types.ndarray(dtype=_TI_FP),
        Ey:  ti.types.ndarray(dtype=_TI_FP),
        Bx:  ti.types.ndarray(dtype=_TI_FP),
        By:  ti.types.ndarray(dtype=_TI_FP),
        Bz:  ti.types.ndarray(dtype=_TI_FP),
        num_p: ti.i32,
        dy: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        dt: ti.f32,
        q_m: ti.f32,
        zone_b: ti.types.ndarray(dtype=_TI_FP),
        zone_dx: ti.types.ndarray(dtype=_TI_FP),
        zone_off: ti.types.ndarray(dtype=ti.i32),
        num_zones: ti.i32,
        n_sub: ti.i32
    ):
        """Parallel bilinear interpolation with 2D3V Boris Algorithm on generic N-zone mesh with fused in-register sub-stepping."""
        for i in range(num_p):
            px = x[i]
            py = y[i]
            cur_vx = vx[i]
            cur_vy = vy[i]
            cur_vz = vz[i]

            for _ in range(n_sub):
                z = 0
                while z < num_zones - 1 and px >= zone_b[z + 1]:
                    z += 1

                local_i = ti.cast(ti.floor((px - zone_b[z]) / zone_dx[z]), ti.i32)
                ix0 = local_i + zone_off[z]
                fx = (px - (zone_b[z] + ti.cast(local_i, ti.f32) * zone_dx[z])) / zone_dx[z]

                idx_y = py / dy
                iy0 = ti.cast(ti.floor(idx_y), ti.i32)

                ix0 = ti.max(0, ti.min(ix0, nx - 2))
                iy0 = ti.max(0, ti.min(iy0, ny - 2))
                fx = ti.max(0.0, ti.min(fx, 1.0))
                fy = idx_y - ti.cast(iy0, ti.f32)

                Ex_p = Ex[iy0, ix0]*(1.0-fx)*(1.0-fy) + Ex[iy0, ix0+1]*fx*(1.0-fy) + \
                       Ex[iy0+1, ix0]*(1.0-fx)*fy      + Ex[iy0+1, ix0+1]*fx*fy
                Ey_p = Ey[iy0, ix0]*(1.0-fx)*(1.0-fy) + Ey[iy0, ix0+1]*fx*(1.0-fy) + \
                       Ey[iy0+1, ix0]*(1.0-fx)*fy      + Ey[iy0+1, ix0+1]*fx*fy

                Bx_p = Bx[iy0, ix0]*(1.0-fx)*(1.0-fy) + Bx[iy0, ix0+1]*fx*(1.0-fy) + \
                       Bx[iy0+1, ix0]*(1.0-fx)*fy      + Bx[iy0+1, ix0+1]*fx*fy
                By_p = By[iy0, ix0]*(1.0-fx)*(1.0-fy) + By[iy0, ix0+1]*fx*(1.0-fy) + \
                       By[iy0+1, ix0]*(1.0-fx)*fy      + By[iy0+1, ix0+1]*fx*fy
                Bz_p = Bz[iy0, ix0]*(1.0-fx)*(1.0-fy) + Bz[iy0, ix0+1]*fx*(1.0-fy) + \
                       Bz[iy0+1, ix0]*(1.0-fx)*fy      + Bz[iy0+1, ix0+1]*fx*fy

                # Boris pusher
                v_minus_x = cur_vx + (q_m * Ex_p * dt) / 2.0
                v_minus_y = cur_vy + (q_m * Ey_p * dt) / 2.0
                v_minus_z = cur_vz

                t_x = (q_m * Bx_p * dt) / 2.0
                t_y = (q_m * By_p * dt) / 2.0
                t_z = (q_m * Bz_p * dt) / 2.0
                t_mag_sq = t_x**2 + t_y**2 + t_z**2

                s_x = 2.0 * t_x / (1.0 + t_mag_sq)
                s_y = 2.0 * t_y / (1.0 + t_mag_sq)
                s_z = 2.0 * t_z / (1.0 + t_mag_sq)

                v_prime_x = v_minus_x + (v_minus_y * t_z - v_minus_z * t_y)
                v_prime_y = v_minus_y + (v_minus_z * t_x - v_minus_x * t_z)
                v_prime_z = v_minus_z + (v_minus_x * t_y - v_minus_y * t_x)

                # Cross product 2: v_plus = v_minus + (v_prime x s)
                v_plus_x = v_minus_x + (v_prime_y * s_z - v_prime_z * s_y)
                v_plus_y = v_minus_y + (v_prime_z * s_x - v_prime_x * s_z)
                v_plus_z = v_minus_z + (v_prime_x * s_y - v_prime_y * s_x)

                cur_vx = v_plus_x + (q_m * Ex_p * dt) / 2.0
                cur_vy = v_plus_y + (q_m * Ey_p * dt) / 2.0
                cur_vz = v_plus_z

                # Kinematic update in mm units
                px += cur_vx * dt * 1000.0
                py += cur_vy * dt * 1000.0

            x[i] = px
            y[i] = py
            vx[i] = cur_vx
            vy[i] = cur_vy
            vz[i] = cur_vz

    def push_particles_boris_taichi(
        x, y, vx, vy, vz,
        Ex, Ey, Bx, By, Bz,
        num_p, dy, nx, ny, dt, q_m,
        zone_b, zone_dx, zone_off, num_zones,
        n_sub: int = 1
    ):
        """Python wrapper for Taichi Boris pusher with optional fused in-register sub-stepping (default n_sub=1)."""
        return _push_particles_boris_taichi_kernel(
            x, y, vx, vy, vz,
            Ex, Ey, Bx, By, Bz,
            int(num_p), float(dy), int(nx), int(ny), float(dt), float(q_m),
            zone_b, zone_dx, zone_off, int(num_zones),
            int(n_sub)
        )

    @ti.kernel
    def _check_particle_boundaries_taichi_kernel(
        x: ti.types.ndarray(dtype=_TI_FP),
        y: ti.types.ndarray(dtype=_TI_FP),
        vx: ti.types.ndarray(dtype=_TI_FP),
        x_old: ti.types.ndarray(dtype=_TI_FP),
        y_old: ti.types.ndarray(dtype=_TI_FP),
        is_bound: ti.types.ndarray(dtype=ti.i32),
        zone_b: ti.types.ndarray(dtype=_TI_FP),
        zone_dx: ti.types.ndarray(dtype=_TI_FP),
        zone_off: ti.types.ndarray(dtype=ti.i32),
        num_zones: ti.i32,
        num_p: ti.i32,
        dy: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        Lx: ti.f32,
        Ly: ti.f32,
        periodic_y: ti.i32,
        grid_x_min: ti.f32,
        grid_x_max: ti.f32,
        x_entry: ti.f32,
        x_exit: ti.f32,
        x_mid: ti.f32,
        samples: ti.i32,
        status: ti.types.ndarray(dtype=ti.u8),
        impact_ix: ti.types.ndarray(dtype=ti.i16),
        impact_iy: ti.types.ndarray(dtype=ti.i16),
        counts: ti.types.ndarray(dtype=ti.i32)
    ):
        """Parallel particle wall collision, ray segment testing, and domain boundary checking."""
        for i in range(num_p):
            px = x[i]
            py = y[i]
            cur_vx = vx[i]
            px_old = x_old[i]
            py_old = y_old[i]

            is_oob = 0
            if px < 0.0 or px > Lx or ti.math.isnan(px) or ti.math.isnan(py):
                is_oob = 1
            elif periodic_y == 0 and (py < 0.0 or py > Ly):
                is_oob = 1

            is_hit = 0
            imp_x = 0
            imp_y = 0

            if is_oob == 0:
                seg_min_x = ti.min(px, px_old)
                seg_max_x = ti.max(px, px_old)

                if seg_max_x >= grid_x_min and seg_min_x <= grid_x_max and samples > 1:
                    dx = px - px_old
                    dy_p = py - py_old

                    for s in range(1, samples + 1):
                        f = ti.cast(s, ti.f32) / ti.cast(samples, ti.f32)
                        xi = px_old + dx * f
                        yi = py_old + dy_p * f

                        z = 0
                        while z < num_zones - 1 and xi >= zone_b[z + 1]:
                            z += 1
                        c_x = ti.cast(ti.round((xi - zone_b[z]) / zone_dx[z]), ti.i32) + zone_off[z]
                        c_y = ti.cast(ti.round(yi / dy), ti.i32)

                        c_x = ti.max(0, ti.min(c_x, nx - 1))
                        c_y = ti.max(0, ti.min(c_y, ny - 1))

                        if is_bound[c_y, c_x] != 0:
                            is_hit = 1
                            imp_x = c_x
                            imp_y = c_y
                            break
                else:
                    z = 0
                    while z < num_zones - 1 and px >= zone_b[z + 1]:
                        z += 1
                    c_x = ti.cast(ti.round((px - zone_b[z]) / zone_dx[z]), ti.i32) + zone_off[z]
                    c_y = ti.cast(ti.round(py / dy), ti.i32)

                    c_x = ti.max(0, ti.min(c_x, nx - 1))
                    c_y = ti.max(0, ti.min(c_y, ny - 1))

                    if is_bound[c_y, c_x] != 0:
                        is_hit = 1
                        imp_x = c_x
                        imp_y = c_y

            impact_ix[i] = ti.cast(imp_x, ti.i16)
            impact_iy[i] = ti.cast(imp_y, ti.i16)

            st = ti.cast(0, ti.u8)
            if is_hit == 1:
                st |= ti.cast(1, ti.u8)   # bit 0: hit_grid
            if is_oob == 1:
                st |= ti.cast(2, ti.u8)   # bit 1: oob

            is_dead = is_hit | is_oob
            if is_dead == 1:
                st |= ti.cast(4, ti.u8)   # bit 2: dead
                ti.atomic_add(counts[0], 1)

            if is_hit == 1 and is_oob == 0:
                ti.atomic_add(counts[1], 1)  # lost to grid

            if is_oob == 1:
                ti.atomic_add(counts[2], 1)  # lost to oob

            if px_old < x_entry and px >= x_entry and cur_vx > 0.0:
                st |= ti.cast(8, ti.u8)   # bit 3: entered optics
                ti.atomic_add(counts[3], 1)

            if px > x_exit and cur_vx > 0.0:
                st |= ti.cast(64, ti.u8)  # bit 6: exited
                ti.atomic_add(counts[6], 1)
                if px_old <= x_exit:
                    st |= ti.cast(16, ti.u8)  # bit 4: crossed exit
                    ti.atomic_add(counts[4], 1)

            if px_old <= x_mid and px > x_mid and cur_vx > 0.0:
                st |= ti.cast(32, ti.u8)  # bit 5: crossed mid plane
                ti.atomic_add(counts[5], 1)

            status[i] = st

    def check_particle_boundaries_taichi(
        x, y, vx, x_old, y_old,
        is_bound, zone_b, zone_dx, zone_off, num_zones,
        num_p, dy, nx, ny, Lx, Ly, periodic_y,
        grid_x_min, grid_x_max, x_entry, x_exit, x_mid,
        samples, status, impact_ix, impact_iy, counts
    ):
        """Python wrapper for Taichi boundary and wall collision checking."""
        if num_p <= 0:
            return
        if x.dtype != _NP_FP:
            x = x.astype(_NP_FP)
        if y.dtype != _NP_FP:
            y = y.astype(_NP_FP)
        if vx.dtype != _NP_FP:
            vx = vx.astype(_NP_FP)
        if x_old.dtype != _NP_FP:
            x_old = x_old.astype(_NP_FP)
        if y_old.dtype != _NP_FP:
            y_old = y_old.astype(_NP_FP)
        if zone_b.dtype != _NP_FP:
            zone_b = zone_b.astype(_NP_FP)
        if zone_dx.dtype != _NP_FP:
            zone_dx = zone_dx.astype(_NP_FP)
        if zone_off.dtype != np.int32:
            zone_off = zone_off.astype(np.int32)
        if is_bound.dtype != np.int32:
            is_bound = is_bound.astype(np.int32)

        _check_particle_boundaries_taichi_kernel(
            x, y, vx, x_old, y_old,
            is_bound, zone_b, zone_dx, zone_off, int(num_zones),
            int(num_p), float(dy), int(nx), int(ny),
            float(Lx), float(Ly), int(periodic_y),
            float(grid_x_min), float(grid_x_max),
            float(x_entry), float(x_exit), float(x_mid),
            int(samples), status, impact_ix, impact_iy, counts
        )

    @ti.kernel
    def thermal_conduction_taichi(
        T:    ti.types.ndarray(dtype=_TI_FP),
        T_new:ti.types.ndarray(dtype=_TI_FP),
        mask: ti.types.ndarray(dtype=ti.i32),
        nx: ti.i32,
        ny: ti.i32,
        Fo_x: ti.f32,
        Fo_y: ti.f32
    ):
        """Parallel 2D Finite Difference Thermal Conduction"""
        for iy, ix in ti.ndrange(ny, nx):
            if mask[iy, ix] == 1:
                T_l = T[iy, ix]
                if ix > 0 and mask[iy, ix-1] == 1: T_l = T[iy, ix-1]
                T_r = T[iy, ix]
                if ix < nx-1 and mask[iy, ix+1] == 1: T_r = T[iy, ix+1]
                T_u = T[iy, ix]
                if iy < ny-1 and mask[iy+1, ix] == 1: T_u = T[iy+1, ix]
                T_d = T[iy, ix]
                if iy > 0 and mask[iy-1, ix] == 1: T_d = T[iy-1, ix]
                dT = Fo_x*(T_l - 2.0*T[iy, ix] + T_r) + Fo_y*(T_d - 2.0*T[iy, ix] + T_u)
                T_new[iy, ix] = ti.max(T[iy, ix] + dT, 300.0)
            else:
                T_new[iy, ix] = T[iy, ix]

else:
    accumulate_rho_taichi = None
    push_particles_boris_taichi = None
    check_particle_boundaries_taichi = None
    thermal_conduction_taichi = None
