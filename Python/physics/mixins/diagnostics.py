"""
Diagnostics and physical observables mixin for DigitalTwinSimulator.
"""
import numpy as np


class DiagnosticsMixin:
    """Methods for retrieving diagnostic maps, energy/charge balance, erosion profiles, and particle kinematics."""

    def reset_ppc_accumulator(self):
        """Resets the time-averaged PPC spatial accumulator."""
        if hasattr(self, 'accum_ppc_map'):
            self.accum_ppc_map.fill(0.0)
        self.ppc_steps_count = 0

    def get_avg_ppc_map(self):
        """Returns the time-averaged 2D PPC map across accumulated steps."""
        if hasattr(self, 'accum_ppc_map') and self.ppc_steps_count > 0:
            return self.accum_ppc_map / float(self.ppc_steps_count)
        if hasattr(self, 'current_ppc_map'):
            return self.current_ppc_map.astype(np.float64)
        return np.zeros((self.ny, self.nx), dtype=np.float64)

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
                y_c = 1.5 * self.grids[0]['r']
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
        from diagnostics.performance_monitor import PerformanceMonitor
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
        from diagnostics.metrics import compute_energy_budget
        return compute_energy_budget(self)

    def get_total_charge(self):
        """
        Compute and store the instantaneous charge budget of the PIC system:
            Q_net = Q_ions + Q_electrons + Q_boltzmann
        Delegates calculation and history logging to diagnostics.metrics.compute_charge_budget.
        """
        from diagnostics.metrics import compute_charge_budget
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
        hit = np.zeros(len(x0), dtype=bool)
        # Default to ending cell (will be overwritten for hits)
        hit_ix = self._x_to_ix(x1)
        hit_iy = np.clip(np.round(y1 / self.dy).astype(int), 0, self.ny - 1)

        for i in range(1, samples + 1):
            f = i / float(samples)
            xi = x0 + (x1 - x0) * f
            yi = y0 + (y1 - y0) * f
            c_x = self._x_to_ix(xi)
            c_y = np.clip(np.round(yi / self.dy).astype(int), 0, self.ny - 1)
            step_hit = self.isBound[c_y, c_x]

            # Record coordinates only for the FIRST hit
            new_hits = step_hit & ~hit
            hit_ix[new_hits] = c_x[new_hits]
            hit_iy[new_hits] = c_y[new_hits]

            hit |= step_hit

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
