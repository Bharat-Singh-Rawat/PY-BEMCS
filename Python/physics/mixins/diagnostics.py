"""
Diagnostics and physical observables mixin for DigitalTwinSimulator.
"""
import numpy as np

from diagnostics.metrics import compute_energy_budget, compute_charge_budget
from diagnostics.performance_monitor import PerformanceMonitor


class DiagnosticsMixin:
    """Methods for retrieving diagnostic maps, energy/charge balance, erosion profiles, and particle kinematics."""

    def reset_ppc_accumulator(self):
        """Resets the time-averaged PPC spatial accumulator."""
        if hasattr(self, 'accum_ppc_map'):
            self.accum_ppc_map.fill(0.0)
        self.ppc_steps_count = 0

    def get_avg_ppc_map(self):
        """Returns the time-averaged 2D PPC map across accumulated steps."""
        if hasattr(self, 'accum_ppc_map') and self.accum_ppc_map is not None and getattr(self, 'ppc_steps_count', 0) > 0:
            return self.accum_ppc_map / float(self.ppc_steps_count)
        if hasattr(self, 'current_ppc_map') and self.current_ppc_map is not None:
            return self.current_ppc_map.astype(np.float64)
        ny = int(getattr(self, 'ny', 1) or 1)
        nx = int(getattr(self, 'nx', 1) or 1)
        return np.zeros((ny, nx), dtype=np.float64)

    def get_density_map(self, species: str = 'ions') -> np.ndarray:
        """
        Compute and return the 2D spatial number density map [m^-3] of shape (ny, nx).
        Supported species:
          - 'ions' or 'total_ions': all active macro-ions (primary + CEX)
          - 'primary' or 'primary_ions': primary non-CEX ions only
          - 'cex' or 'cex_ions': charge-exchange ions only
          - 'electrons' or 'electron': total electrons (kinetic + Boltzmann fluid)
          - 'kinetic_electrons': kinetic electron macroparticles only
          - 'boltzmann_electrons': fluid/Boltzmann continuum electrons only
        """
        cvol = getattr(self, 'cell_vol_2d', None)
        if cvol is None or cvol.shape != (self.ny, self.nx):
            if hasattr(self, 'cell_vol_1d') and len(self.cell_vol_1d) == self.nx:
                cvol = np.tile(self.cell_vol_1d, (self.ny, 1))
            else:
                cvol = np.full((self.ny, self.nx), (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3)

        sp = species.lower().strip()

        if sp in ('electrons', 'electron', 'kinetic_electrons', 'boltzmann_electrons'):
            density = np.zeros((self.ny, self.nx), dtype=np.float64)

            # 1. Kinetic electrons
            if sp != 'boltzmann_electrons' and getattr(self, 'num_e', 0) > 0:
                n_e = int(self.num_e)
                ex = self.e_x[:n_e]
                ey = self.e_y[:n_e]
                ix = self._x_to_ix(ex)
                iy = np.clip(np.round(ey / self.dy).astype(int), 0, self.ny - 1)
                flat_idx = iy * self.nx + ix
                counts = np.bincount(flat_idx, minlength=self.nx * self.ny).reshape((self.ny, self.nx))
                density += (counts * self.macro_weight) / cvol

            # 2. Boltzmann fluid electrons
            if sp != 'kinetic_electrons' and hasattr(self, 'V') and self.V is not None and self.V.shape == (self.ny, self.nx):
                grids = getattr(self, 'grids', [{'V': 1000}])
                v_offset = getattr(self, 'V_plasma_offset', 20.0)
                V_p = grids[0]['V'] + v_offset if grids else 1020.0
                Te = getattr(self, 'Te_up', 3.0)
                n0 = getattr(self, 'n0_plasma', 1e17)
                bf = np.exp((np.minimum(self.V, V_p) - V_p) / max(Te, 1e-4))
                density += n0 * bf

            # Zero out density inside solid grid electrodes
            if hasattr(self, 'isBound') and self.isBound is not None and self.isBound.shape == (self.ny, self.nx):
                density[self.isBound] = 0.0

            return density

        # Ion species
        density = np.zeros((self.ny, self.nx), dtype=np.float64)
        n_p = int(getattr(self, 'num_p', 0))
        if n_p == 0:
            return density

        px = self.p_x[:n_p]
        py = self.p_y[:n_p]

        if sp in ('cex', 'cex_ions'):
            if not hasattr(self, 'p_isCEX'):
                return density
            mask = self.p_isCEX[:n_p]
            px = px[mask]
            py = py[mask]
        elif sp in ('primary', 'primary_ions'):
            if hasattr(self, 'p_isCEX'):
                mask = ~self.p_isCEX[:n_p]
                px = px[mask]
                py = py[mask]

        if len(px) == 0:
            return density

        ix = self._x_to_ix(px)
        iy = np.clip(np.round(py / self.dy).astype(int), 0, self.ny - 1)
        flat_idx = iy * self.nx + ix
        counts = np.bincount(flat_idx, minlength=self.nx * self.ny).reshape((self.ny, self.nx))
        density = (counts * self.macro_weight) / cvol

        if hasattr(self, 'isBound') and self.isBound is not None and self.isBound.shape == (self.ny, self.nx):
            density[self.isBound] = 0.0

        return density

    def get_divergence_plane_locations(self) -> tuple[float, float]:
        """
        Return (x1_exit_mm, x2_custom_mm) defining the two axial planes where divergence
        and radial density profiles are evaluated.
        """
        if hasattr(self, 'grid_x_ends') and len(self.grid_x_ends) > 0:
            x1 = float(self.grid_x_ends[-1])
        else:
            x1 = float(getattr(self, 'Lx', 20.0) * 0.5)

        x2 = getattr(self, 'div_plane_x', None)
        if x2 is None or not (x1 <= x2 <= getattr(self, 'Lx', 20.0)):
            x2 = round(x1 + 0.5 * (getattr(self, 'Lx', 20.0) - x1), 3)

        return float(x1), float(x2)

    def get_radial_density_profile(self, x_plane_mm: float, species: str = 'ions', density_map: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (y_coords_mm, density_profile_m3) along the radial direction at axial position x_plane_mm.
        Uses linear interpolation between adjacent mesh columns for smooth profiles.
        """
        if density_map is None:
            density_map = self.get_density_map(species=species)
        y_coords = getattr(self, 'y_coords', getattr(self, 'ypts', None))
        if y_coords is None:
            y_coords = np.arange(self.ny) * self.dy

        x_coords = getattr(self, 'x_coords', getattr(self, 'xpts', None))
        x_clamped = float(np.clip(x_plane_mm, 0.0, getattr(self, 'Lx', 20.0)))

        if x_coords is not None and len(x_coords) >= 2:
            ix1 = int(np.searchsorted(x_coords, x_clamped))
            ix1 = int(np.clip(ix1, 1, len(x_coords) - 1))
            ix0 = ix1 - 1
            x0 = x_coords[ix0]
            x1 = x_coords[ix1]
            if x1 > x0:
                w = float(np.clip((x_clamped - x0) / (x1 - x0), 0.0, 1.0))
                profile = (1.0 - w) * density_map[:, ix0] + w * density_map[:, ix1]
            else:
                profile = density_map[:, ix0]
        else:
            ix = self._x_to_ix(x_clamped) if hasattr(self, '_x_to_ix') else int(np.clip(round(x_clamped / self.dx), 0, self.nx - 1))
            profile = density_map[:, ix]

        return y_coords.copy(), profile.copy()

    def get_centerline_potential_profile(self):
        """
        Returns (x_coords_mm, V_centerline_V) along the axial direction through the center
        of the first hole.
        Consistent across half_hole (y=0), one_hole (y=1.5*r_s), and two_holes (y=1.5*r_s).
        """
        geometry = getattr(self, 'geometry', 'half_hole')
        hole_centers = getattr(self, 'hole_centers', None)
        if not hole_centers:
            if hasattr(self, 'grids') and self.grids and geometry in ('one_hole', 'two_holes'):
                y_c = 0.5 * getattr(self, 'Ly', 3.0) if geometry == 'one_hole' else 0.25 * getattr(self, 'Ly', 3.0)
            else:
                y_c = 0.0
        else:
            y_c = hole_centers[0]

        y_idx = int(np.clip(round(y_c / self.dy), 0, self.ny - 1))
        x_pts = getattr(self, 'xpts', np.linspace(0, self.Lx, self.nx))
        return x_pts.copy(), self.V[y_idx, :].copy()

    def enable_perf_monitor(self, **kwargs):
        """
        Enable the runtime performance monitor.

        Keyword arguments are forwarded to PerformanceMonitor.__init__
        (log_every, warn_particles, warn_memory_mb, warn_step_time_ms).
        """
        self._perf_monitor = PerformanceMonitor(**kwargs)
        return self._perf_monitor

    def disable_perf_monitor(self):
        """Disable and return the performance monitor (for final report / export)."""
        mon = self._perf_monitor
        self._perf_monitor = None
        return mon

    def get_total_energy(self):
        """
        Compute and store the instantaneous total energy of the PIC system:
            E_total = E_kinetic_ions + E_kinetic_electrons + E_field
        Delegates calculation and warning logging to diagnostics.metrics.compute_energy_budget.
        """
        return compute_energy_budget(self)

    def get_total_charge(self):
        """
        Compute and store the instantaneous charge budget of the PIC system:
            Q_net = Q_ions + Q_electrons + Q_boltzmann
        Delegates calculation and history logging to diagnostics.metrics.compute_charge_budget.
        """
        return compute_charge_budget(self)

    def get_third_grid_transparency_frame(self):
        if self.entered_optics_step <= 0.0:
            return 0.0
        return self.transmitted3_step / self.entered_optics_step

    def get_transparency(self):
        if self.entered_optics <= 0.0:
            return 0.0
        return self.transmitted_ions / self.entered_optics

    def get_exit_ion_current(self):
        """Return (instantaneous_step_A, cumulative_average_A) ion current exiting the grids."""
        return self.exit_ion_current_step, self.exit_ion_current_avg

    def has_active_particles(self):
        return (self.num_p > 0) or (self.num_e > 0)

    def _segment_hits_grid(self, x0, y0, x1, y1, samples=8):
        n = len(x0)
        hit = np.zeros(n, dtype=bool)
        hit_ix = np.zeros(n, dtype=np.int32)
        hit_iy = np.zeros(n, dtype=np.int32)
        if n == 0:
            return hit, hit_ix, hit_iy

        grid_x_starts = getattr(self, 'grid_x_starts', None)
        grid_x_ends = getattr(self, 'grid_x_ends', None)
        if grid_x_starts and grid_x_ends:
            grid_x_min = max(0.0, min(grid_x_starts) - 0.2)
            grid_x_max = min(self.Lx, max(grid_x_ends) + 0.2)
        else:
            grid_x_min = 0.0
            grid_x_max = self.Lx

        seg_min_x = np.minimum(x0, x1)
        seg_max_x = np.maximum(x0, x1)
        cand_mask = (seg_max_x >= grid_x_min) & (seg_min_x <= grid_x_max)

        cand_indices = np.where(cand_mask)[0]
        if len(cand_indices) == 0:
            return hit, hit_ix, hit_iy

        x0_c = x0[cand_indices]
        y0_c = y0[cand_indices]
        x1_c = x1[cand_indices]
        y1_c = y1[cand_indices]

        hit_c = np.zeros(len(cand_indices), dtype=bool)
        hit_ix_c = self._x_to_ix(x1_c)
        hit_iy_c = np.clip(np.round(y1_c / self.dy).astype(int), 0, self.ny - 1)

        dx = x1_c - x0_c
        dy = y1_c - y0_c
        for i in range(1, samples + 1):
            f = i / float(samples)
            xi = x0_c + dx * f
            yi = y0_c + dy * f
            c_x = self._x_to_ix(xi)
            c_y = np.clip(np.round(yi / self.dy).astype(int), 0, self.ny - 1)
            step_hit = self.isBound[c_y, c_x]

            # Record coordinates only for the FIRST hit
            new_hits = step_hit & ~hit_c
            hit_ix_c[new_hits] = c_x[new_hits]
            hit_iy_c[new_hits] = c_y[new_hits]

            hit_c |= step_hit

        hit[cand_indices] = hit_c
        hit_ix[cand_indices] = hit_ix_c
        hit_iy[cand_indices] = hit_iy_c
        return hit, hit_ix, hit_iy

    def get_groove_profile(self, grid_idx, thresh=None, accumulate_subcell=True, face='upstream'):
        if grid_idx < 0 or grid_idx >= len(self.mask_grids):
            return np.array([]), np.array([])

        mask = self.mask_grids[grid_idx]
        if not np.any(mask):
            return np.array([]), np.array([])

        depth = self.eroded_depth
        if accumulate_subcell and thresh and thresh > 0:
            depth = depth + (self.damage_map / thresh) * self.dy

        y_mm = np.arange(self.ny) * self.dy
        per_y = np.zeros(self.ny, dtype=np.float64)
        has_any = mask.any(axis=1)

        if face == 'any':
            cols = np.any(mask, axis=0)
            if not np.any(cols):
                return np.array([]), np.array([])
            per_y[has_any] = depth[has_any][:, cols].max(axis=1)
        elif face == 'upstream':
            first_idx = np.argmax(mask, axis=1)
            per_y[has_any] = depth[np.where(has_any)[0], first_idx[has_any]]
        elif face == 'downstream':
            last_idx = self.nx - 1 - np.argmax(mask[:, ::-1], axis=1)
            per_y[has_any] = depth[np.where(has_any)[0], last_idx[has_any]]
        else:
            raise ValueError(f"face must be 'upstream','downstream', or 'any' (got {face!r})")

        return y_mm, per_y * 1000.0

    def get_particle_kinematics(self):
        t_current = self.iteration * self.dt

        if self.num_p > 0:
            p_x, p_y = self.p_x[:self.num_p], self.p_y[:self.num_p]
            p_vx, p_vy, p_vz = self.p_vx[:self.num_p], self.p_vy[:self.num_p], self.p_vz[:self.num_p]
            p_cex = self.p_isCEX[:self.num_p]
            v_sq_i = p_vx ** 2 + p_vy ** 2 + p_vz ** 2
            energy_eV_i = (0.5 * self.m_ion * v_sq_i) / self.q
            ions = np.column_stack((
                np.full(self.num_p, t_current),
                p_x, p_y, p_vx, p_vy, p_vz,
                energy_eV_i, p_cex.astype(int)
            ))
        else:
            ions = np.empty((0, 8))

        if self.num_e > 0:
            e_x, e_y = self.e_x[:self.num_e], self.e_y[:self.num_e]
            e_vx, e_vy, e_vz = self.e_vx[:self.num_e], self.e_vy[:self.num_e], self.e_vz[:self.num_e]
            v_sq_e = e_vx ** 2 + e_vy ** 2 + e_vz ** 2
            energy_eV_e = (0.5 * self.m_e * v_sq_e) / self.q
            type_e = np.where(e_x <= 4.0, 2, 3)
            elecs = np.column_stack((
                np.full(self.num_e, t_current),
                e_x, e_y, e_vx, e_vy, e_vz,
                energy_eV_e, type_e
            ))
        else:
            elecs = np.empty((0, 8))

        return np.vstack((ions, elecs))
