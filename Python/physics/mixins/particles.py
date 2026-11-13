"""
Particle buffers and substep estimation mixin for DigitalTwinSimulator.
"""
import numpy as np


class ParticlesMixin:
    """Particle buffer growth, particle injection buffers, and adaptive substepping."""

    MAX_HARD_PARTICLES = 1_500_000
    MAX_HARD_ELECTRONS = 1_500_000

    def _add_ions(self, x, y, vx, vy, vz, is_cex):
        n_new = len(x)
        if self.num_p + n_new > self.MAX_HARD_PARTICLES:
            allowed = max(0, self.MAX_HARD_PARTICLES - self.num_p)
            if allowed <= 0:
                return
            n_new = allowed
            x = x[:n_new]
            y = y[:n_new]
            vx = vx[:n_new]
            vy = vy[:n_new]
            vz = vz[:n_new]
            is_cex = is_cex[:n_new]

        if self.num_p + n_new > self.max_p:
            new_max = max(self.max_p * 2, self.num_p + n_new)
            self.p_x     = np.pad(self.p_x,     (0, new_max - self.max_p))
            self.p_y     = np.pad(self.p_y,     (0, new_max - self.max_p))
            self.p_vx    = np.pad(self.p_vx,    (0, new_max - self.max_p))
            self.p_vy    = np.pad(self.p_vy,    (0, new_max - self.max_p))
            self.p_vz    = np.pad(self.p_vz,    (0, new_max - self.max_p))
            self.p_isCEX = np.pad(self.p_isCEX, (0, new_max - self.max_p))
            self.max_p   = new_max
        s = self.num_p
        e = s + n_new
        self.p_x[s:e]     = x
        self.p_y[s:e]     = y
        self.p_vx[s:e]    = vx
        self.p_vy[s:e]    = vy
        self.p_vz[s:e]    = vz
        self.p_isCEX[s:e] = is_cex
        self.num_p += n_new

    def _add_electrons(self, x, y, vx, vy, vz):
        n_new = len(x)
        if self.num_e + n_new > self.MAX_HARD_ELECTRONS:
            allowed = max(0, self.MAX_HARD_ELECTRONS - self.num_e)
            if allowed <= 0:
                return
            n_new = allowed
            x = x[:n_new]
            y = y[:n_new]
            vx = vx[:n_new]
            vy = vy[:n_new]
            vz = vz[:n_new]

        if self.num_e + n_new > self.max_e:
            new_max = max(self.max_e * 2, self.num_e + n_new)
            self.e_x  = np.pad(self.e_x,  (0, new_max - self.max_e))
            self.e_y  = np.pad(self.e_y,  (0, new_max - self.max_e))
            self.e_vx = np.pad(self.e_vx, (0, new_max - self.max_e))
            self.e_vy = np.pad(self.e_vy, (0, new_max - self.max_e))
            self.e_vz = np.pad(self.e_vz, (0, new_max - self.max_e))
            self.max_e = new_max
        s = self.num_e
        e = s + n_new
        self.e_x[s:e]  = x
        self.e_y[s:e]  = y
        self.e_vx[s:e] = vx
        self.e_vy[s:e] = vy
        self.e_vz[s:e] = vz
        self.num_e += n_new

    def compute_particle_substeps(self, x, y, vx, vy, vz, qm, dt, frac=0.25):
        """
        Dynamically calculates the number of sub-steps (n_sub) required for particle pushing.
        
        Sub-stepping ensures that fast-moving or strongly accelerated particles do not travel
        more than a fraction of a grid cell (frac * min(dx, dy)) in a single push. This prevents
        particles from skipping cells, avoids non-physical trajectory errors, ensures accurate
        electric field interpolation, and prevents tunneling through grid boundaries.
        """
        if len(x) == 0:
            return 1, 0.0, frac * min(getattr(self, 'dx_min', self.dx), self.dy) * 1e-3

        ds_lim_min = float(frac * min(getattr(self, 'dx_min', self.dx), self.dy) * 1e-3)

        # Fast upper-bound check: if the maximum theoretical displacement in the domain
        # cannot exceed ds_lim_min, skip expensive per-particle 2D field interpolation.
        v_sq_max = float(np.max(vx * vx + vy * vy + vz * vz))
        v_max = np.sqrt(v_sq_max) if v_sq_max > 0.0 else 0.0
        
        E_max_sq = float(getattr(self, '_cached_E_max_sq', -1.0))
        if E_max_sq < 0.0 and hasattr(self, 'Ex') and hasattr(self, 'Ey') and self.Ex is not None and self.Ey is not None:
            E_max_sq = float(np.max(self.Ex ** 2 + self.Ey ** 2))
            self._cached_E_max_sq = E_max_sq
        elif E_max_sq < 0.0:
            E_max_sq = 0.0

        a_max = abs(qm) * np.sqrt(E_max_sq)
        ds_upper_bound = v_max * dt + 0.5 * a_max * (dt ** 2)
        if ds_upper_bound <= ds_lim_min:
            return 1, float(ds_upper_bound), ds_lim_min

        ix0 = self._x_to_ix_floor(x)
        iy0 = np.clip(np.floor(y / self.dy).astype(int), 0, self.ny - 2)
        dx_local = self.x_coords[ix0 + 1] - self.x_coords[ix0]
        fx = np.clip((x - self.x_coords[ix0]) / dx_local, 0.0, 1.0)
        fy = y / self.dy - iy0
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        def interp(F):
            return (
                F[iy0, ix0] * (1 - fx) * (1 - fy) +
                F[iy0, ix1] * fx * (1 - fy) +
                F[iy1, ix0] * (1 - fx) * fy +
                F[iy1, ix1] * fx * fy
            )

        Ex_p = interp(self.Ex)
        Ey_p = interp(self.Ey)

        a_mag = np.sqrt((qm * Ex_p) ** 2 + (qm * Ey_p) ** 2)
        v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)

        ds_pred = v_mag * dt + 0.5 * a_mag * dt ** 2
        ds_lim = frac * np.minimum(dx_local, self.dy) * 1e-3
        ds_lim_min = float(np.min(ds_lim))
        ds_max = float(np.max(ds_pred))

        n_sub = min(50, max(1, int(np.ceil(np.max(ds_pred / np.maximum(ds_lim, 1e-30))))))
        return n_sub, ds_max, ds_lim_min
