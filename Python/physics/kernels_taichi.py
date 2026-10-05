"""
Taichi GPU / accelerated kernels for particle deposition, Boris push, and thermal conduction.
"""
from .backend import USE_TAICHI, _TI_FP

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
    def push_particles_boris_taichi(
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
        num_zones: ti.i32
    ):
        """Parallel bilinear interpolation with 2D3V Boris Algorithm on generic N-zone mesh"""
        for i in range(num_p):
            px = x[i]
            py = y[i]

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
            v_minus_x = vx[i] + (q_m * Ex_p * dt) / 2.0
            v_minus_y = vy[i] + (q_m * Ey_p * dt) / 2.0
            v_minus_z = vz[i]

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

            # STEP 3: Second half E-field acceleration
            vx[i] = v_plus_x + (q_m * Ex_p * dt) / 2.0
            vy[i] = v_plus_y + (q_m * Ey_p * dt) / 2.0
            vz[i] = v_plus_z

            # =====================================================================
            # KINEMATIC UPDATE
            # =====================================================================
            x[i] += vx[i] * dt * 1000.0
            y[i] += vy[i] * dt * 1000.0

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
    thermal_conduction_taichi = None
