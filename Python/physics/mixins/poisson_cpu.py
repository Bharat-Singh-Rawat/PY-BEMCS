"""
CPU non-linear Poisson-Boltzmann solver mixin for DigitalTwinSimulator.
"""
import numpy as np
import scipy.sparse as sp


class PoissonCPUMixin:
    """CPU Picard-Anderson solver and Newton-Raphson fallback."""

    def _recalc_poisson_cpu(self, max_iters, coeff, V_plasma, Te_up, n0, omega_min=0.2, omega_max=0.35, tol_V=0.05, min_iters=3, tol_rms=0.01, peak_guard=0.25):
        b = np.zeros(self.nx * self.ny, dtype=np.float64)
        V_fixed_flat = self.V_fixed.flatten()
        rhs_rho_mask = getattr(self, 'is_rhs_rho_mask', self.is_interior_mask)

        diff_history = []
        V_best = self.V.copy()
        best_diff = float('inf')
        status = 'max_iters'
        last_diff = 0.0
        last_rms = 0.0
        iters_done = 0
        anderson_history = []
        last_was_anderson = False
        anderson_count = 0
        backtrack_count = 0
        omega_scale = 1.0
        min_omega_scale = 0.15

        # Asymmetric Clamping bounds: prevent exponential Boltzmann surges (upward in V)
        # while permitting rapid field updates in the linear gap / plume.
        te_val = float(Te_up)
        up_limit = min(1.5 * te_val, 15.0)
        down_limit = min(3.0 * te_val, 20.0)

        for it in range(1, max_iters + 1):
            iters_done = it
            boltzmann_factor = np.exp((np.minimum(self.V, V_plasma) - V_plasma) / Te_up)
            rho_e     = -self.q * n0 * boltzmann_factor
            rho_total = self.rho + rho_e
            rho_flat  = rho_total.flatten()

            b.fill(0.0)
            b[self.is_bound_mask] = V_fixed_flat[self.is_bound_mask]
            b[rhs_rho_mask]       = -coeff * rho_flat[rhs_rho_mask]

            V_new_flat = self.laplacian_lu(b)
            V_new      = V_new_flat.reshape((self.ny, self.nx))
            
            delta_V_raw = V_new - self.V
            last_diff  = float(np.max(np.abs(delta_V_raw)))
            last_rms   = float(np.sqrt(np.mean(delta_V_raw ** 2)))
            diff_history.append(last_diff)

            # Track best state seen so far
            if last_diff < best_diff:
                best_diff = last_diff
                V_best = self.V.copy()

            # Record state before update for Anderson history
            V_curr = self.V.copy()
            R_curr = delta_V_raw.copy()

            # Divergence & Surge Detection with Adaptive Backtracking Line Search
            if it >= 4:
                if np.isnan(last_diff) or last_diff > 50000.0:
                    status = 'diverged'
                    self.V = V_best.copy()
                    break

                baseline = diff_history[1] if len(diff_history) > 1 and diff_history[0] > 50.0 else diff_history[0]
                is_surging = (
                    (last_diff > 2.5 * best_diff and last_diff > 1.5) or
                    (last_diff > 3.0 * baseline and last_diff > 2.0) or
                    (
                        len(diff_history) >= 4 and
                        diff_history[-1] > diff_history[-2] > diff_history[-3] > diff_history[-4] and
                        last_diff > 1.0
                    )
                )
                if is_surging:
                    if omega_scale > min_omega_scale:
                        # Backtrack: halve relaxation factor, restore best potential state, flush memory
                        omega_scale *= 0.5
                        self.V = V_best.copy()
                        anderson_history.clear()
                        last_was_anderson = False
                        backtrack_count += 1
                        diff_history.pop()
                        continue
                    else:
                        status = 'diverged'
                        self.V = V_best.copy()
                        break

            # Asymmetric Spatially-Aware Clamping Maps:
            # - In plasma (boltzmann_factor -> 1): bound upward jumps to <= 1.5*Te to eliminate
            #   exponential charge explosions (e^(1.5) <= 4.5x charge surge max).
            # - In vacuum / gap (boltzmann_factor -> 0): allow full step size up to 25 V.
            max_up_map = 25.0 - (25.0 - up_limit) * boltzmann_factor
            max_down_map = 25.0 - (25.0 - down_limit) * boltzmann_factor

            # Adaptive relaxation map scaled by backtracking damping
            omega_map = (omega_max - (omega_max - omega_min) * boltzmann_factor) * omega_scale

            # Method 1 & 4: Trust-region step-clamping with Spatially-Adaptive Omega & On-Demand Anderson Acceleration
            if it == 1 and last_diff > 50.0:
                # Cold start: first iteration seeds the macroscopic Laplace potential field
                self.V = V_new.astype(np.float64)
                anderson_history.clear()
                last_was_anderson = False
            else:
                # On-Demand Anderson Acceleration trigger
                use_anderson = False
                if it >= 4 and not last_was_anderson and last_diff > 0.25 and len(anderson_history) >= 1:
                    if len(diff_history) >= 4:
                        prog = (diff_history[-4] - diff_history[-1]) / max(diff_history[-4], 1e-12)
                        abs_ch = diff_history[-4] - diff_history[-1]
                        is_bouncing = (diff_history[-1] >= diff_history[-2] and diff_history[-2] <= diff_history[-3])
                        if prog < 0.10 or abs_ch < 0.05 or is_bouncing:
                            use_anderson = True
                    elif len(diff_history) >= 3:
                        prog = (diff_history[-3] - diff_history[-1]) / max(diff_history[-3], 1e-12)
                        is_bouncing = (diff_history[-1] >= diff_history[-2] and diff_history[-2] <= diff_history[-3])
                        if prog < 0.07 or is_bouncing:
                            use_anderson = True

                if use_anderson:
                    V_prev, R_prev = anderson_history[-1]
                    delta_V_hist = V_curr - V_prev
                    delta_R_hist = R_curr - R_prev
                    dot_f_df = float(np.sum(R_curr * delta_R_hist))
                    norm_df_sq = float(np.sum(delta_R_hist * delta_R_hist))
                    gamma = dot_f_df / (norm_df_sq + 1e-16)
                    gamma = float(np.clip(gamma, -1.5, 1.5))

                    V_AA = V_curr - gamma * delta_V_hist
                    R_AA = R_curr - gamma * delta_R_hist

                    # Damped extrapolation step using scaled spatially-adaptive omega map
                    delta_V_update = (V_AA - V_curr) + omega_map * R_AA
                    delta_V_clamped = np.clip(delta_V_update, -max_down_map, max_up_map)
                    self.V = (V_curr + delta_V_clamped).astype(np.float64)
                    last_was_anderson = True
                    anderson_count += 1
                else:
                    delta_V_clamped = np.clip(delta_V_raw, -max_down_map, max_up_map)
                    self.V = (self.V + omega_map * delta_V_clamped).astype(np.float64)
                    last_was_anderson = False

                if len(anderson_history) >= 2:
                    anderson_history.pop(0)
                anderson_history.append((V_curr, R_curr))

            # Check 1: Dual-norm convergence (Method 3)
            converged_linf = (last_diff <= tol_V)
            converged_rms  = (last_rms <= tol_rms and last_diff <= peak_guard)
            if it >= min_iters and (converged_linf or converged_rms):
                status = 'converged'
                break

            # Check 3: Stagnation detection (over a 5-iteration window)
            if it >= min_iters + 5:
                w = diff_history[-5:]
                progress = (w[0] - w[-1]) / max(w[0], 1e-12)
                abs_change = w[0] - w[-1]
                # Stagnation occurs if progress has genuinely flatlined (< 1% and < 50 mV change)
                # or if hovering at the macroparticle shot-noise floor (< 5 mV change and delta_V <= 0.30 V)
                if (progress < 0.01 and abs_change < 0.05) or (abs_change < 0.005 and last_diff <= 0.30):
                    status = 'stagnated_noise_floor' if last_diff <= 0.30 else 'stagnated'
                    break

        if status in ['diverged', 'stagnated']:
            print(f"[Poisson Warning] Picard {status} at iter {iters_done} (diff={last_diff:.3f}V). Triggering Newton-Raphson fallback...")
            V_NR, iters_NR, diff_NR, status_NR = self._solve_poisson_newton_cpu(coeff, V_plasma, Te_up, n0, tol_V=tol_V, max_iters=15)
            self.V = V_NR
            self._last_anderson_count = anderson_count
            self._last_backtrack_count = backtrack_count
            return iters_done + iters_NR, diff_NR, diff_NR, status_NR

        self._last_anderson_count = anderson_count
        self._last_backtrack_count = backtrack_count
        return iters_done, last_diff, last_rms, status

    def _solve_poisson_newton_cpu(self, coeff, V_plasma, Te_up, n0, tol_V=0.05, max_iters=15):
        import scipy.sparse.linalg as spla
        V_current = self.V.copy()
        V_fixed_flat = self.V_fixed.flatten()
        rhs_rho_mask = getattr(self, 'is_rhs_rho_mask', self.is_interior_mask)
        last_diff = 0.0

        for it in range(max_iters):
            boltzmann_factor = np.exp((np.minimum(V_current, V_plasma) - V_plasma) / Te_up)
            rho_e = -self.q * n0 * boltzmann_factor
            rho_total = self.rho + rho_e
            rho_flat = rho_total.flatten()

            b = np.zeros(self.nx * self.ny, dtype=np.float64)
            b[self.is_bound_mask] = V_fixed_flat[self.is_bound_mask]
            b[rhs_rho_mask] = -coeff * rho_flat[rhs_rho_mask]

            # F(V) = A * V - b
            F = self.laplacian_matrix.dot(V_current.flatten()) - b

            # Jacobian J = A - d(b)/dV
            drho_e_dV = np.zeros_like(rho_e)
            mask_less = V_current < V_plasma
            drho_e_dV[mask_less] = -(self.q * n0 / Te_up) * boltzmann_factor[mask_less]

            db_dV = np.zeros(self.nx * self.ny, dtype=np.float64)
            db_dV[rhs_rho_mask] = -coeff * drho_e_dV.flatten()[rhs_rho_mask]

            J = self.laplacian_matrix - sp.diags(db_dV)

            # Solve J * delta_V = -F
            try:
                delta_V_flat, exitCode = spla.bicgstab(J, -F, tol=1e-5, maxiter=200)
                if exitCode != 0:
                    delta_V_flat = spla.spsolve(J, -F)
            except Exception:
                delta_V_flat = spla.spsolve(J, -F)

            delta_V = delta_V_flat.reshape((self.ny, self.nx))
            
            # Limit the Newton step size to prevent overshoot
            max_step = 25.0
            delta_V = np.clip(delta_V, -max_step, max_step)
            
            V_current += delta_V
            last_diff = float(np.max(np.abs(delta_V)))

            if last_diff <= tol_V:
                return V_current, it + 1, last_diff, 'converged_newton'

        return V_current, max_iters, last_diff, 'diverged_newton'
