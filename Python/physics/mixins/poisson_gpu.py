"""
GPU-accelerated non-linear Poisson solver mixin for DigitalTwinSimulator using CuPy.
"""
from ..backend import cp, cp_sp


class PoissonGPUMixin:
    """Optional CuPy GPU Picard and Newton solvers."""

    def _recalc_poisson_gpu(self, max_iters, coeff, V_plasma, Te_up, n0, omega_min=0.2, omega_max=0.35, tol_V=0.05, min_iters=3, tol_rms=0.01, peak_guard=0.25):
        V_gpu            = cp.asarray(self.V, dtype=cp.float64)
        rho_gpu          = cp.asarray(self.rho).astype(cp.float64)
        V_fixed_flat_gpu = cp.asarray(self.V_fixed.ravel(), dtype=cp.float64)
        b_gpu            = cp.zeros(self.nx * self.ny, dtype=cp.float64)
        q                = self.q

        rhs_rho_mask = getattr(self, 'is_rhs_rho_mask', self.is_interior_mask)
        rhs_rho_mask_gpu = cp.asarray(rhs_rho_mask)
        bound_mask_gpu   = cp.asarray(self.is_bound_mask)

        diff_history = []
        V_best_gpu = V_gpu.copy()
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

        te_val = float(Te_up)
        up_limit = min(1.5 * te_val, 15.0)
        down_limit = min(3.0 * te_val, 20.0)

        for it in range(1, max_iters + 1):
            iters_done = it
            boltzmann_factor_gpu = cp.exp((cp.minimum(V_gpu, V_plasma) - V_plasma) / Te_up)
            rho_e_gpu    = -q * n0 * boltzmann_factor_gpu
            rho_flat_gpu = (rho_gpu + rho_e_gpu).ravel()
            b_gpu[:] = 0.0
            b_gpu[bound_mask_gpu]   = V_fixed_flat_gpu[bound_mask_gpu]
            b_gpu[rhs_rho_mask_gpu] = -coeff * rho_flat_gpu[rhs_rho_mask_gpu]
            V_new_flat = self.laplacian_lu_gpu.solve(b_gpu)
            V_new_gpu  = V_new_flat.reshape((self.ny, self.nx))
            
            delta_V_raw_gpu = V_new_gpu - V_gpu
            diff_gpu   = cp.max(cp.abs(delta_V_raw_gpu))
            rms_gpu    = cp.sqrt(cp.mean(delta_V_raw_gpu ** 2))
            last_diff  = float(cp.asnumpy(diff_gpu))
            last_rms   = float(cp.asnumpy(rms_gpu))
            diff_history.append(last_diff)

            if last_diff < best_diff:
                best_diff = last_diff
                V_best_gpu = V_gpu.copy()

            # Record state before update for Anderson history
            V_curr_gpu = V_gpu.copy()
            R_curr_gpu = delta_V_raw_gpu.copy()

            # Divergence & Surge Detection with Adaptive Backtracking Line Search
            if it >= 4:
                if cp.isnan(diff_gpu) or last_diff > 50000.0:
                    status = 'diverged'
                    V_gpu = V_best_gpu.copy()
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
                        omega_scale *= 0.5
                        V_gpu = V_best_gpu.copy()
                        anderson_history.clear()
                        last_was_anderson = False
                        backtrack_count += 1
                        diff_history.pop()
                        continue
                    else:
                        status = 'diverged'
                        V_gpu = V_best_gpu.copy()
                        break

            # Asymmetric Spatially-Aware Clamping Maps
            max_up_map_gpu = 25.0 - (25.0 - up_limit) * boltzmann_factor_gpu
            max_down_map_gpu = 25.0 - (25.0 - down_limit) * boltzmann_factor_gpu

            # Adaptive relaxation map scaled by backtracking damping
            omega_map_gpu = (omega_max - (omega_max - omega_min) * boltzmann_factor_gpu) * omega_scale

            # Method 1 & 4: Trust-region step-clamping with Spatially-Adaptive Omega & On-Demand Anderson Acceleration
            if it == 1 and last_diff > 50.0:
                V_gpu = V_new_gpu
                anderson_history.clear()
                last_was_anderson = False
            else:
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
                    V_prev_gpu, R_prev_gpu = anderson_history[-1]
                    delta_V_hist_gpu = V_curr_gpu - V_prev_gpu
                    delta_R_hist_gpu = R_curr_gpu - R_prev_gpu
                    dot_f_df = cp.sum(R_curr_gpu * delta_R_hist_gpu)
                    norm_df_sq = cp.sum(delta_R_hist_gpu * delta_R_hist_gpu)
                    gamma_gpu = cp.clip(dot_f_df / (norm_df_sq + 1e-16), -1.5, 1.5)

                    V_AA_gpu = V_curr_gpu - gamma_gpu * delta_V_hist_gpu
                    R_AA_gpu = R_curr_gpu - gamma_gpu * delta_R_hist_gpu

                    delta_V_update_gpu = (V_AA_gpu - V_curr_gpu) + omega_map_gpu * R_AA_gpu
                    delta_V_clamped_gpu = cp.clip(delta_V_update_gpu, -max_down_map_gpu, max_up_map_gpu)
                    V_gpu = V_curr_gpu + delta_V_clamped_gpu
                    last_was_anderson = True
                    anderson_count += 1
                else:
                    delta_V_clamped_gpu = cp.clip(delta_V_raw_gpu, -max_down_map_gpu, max_up_map_gpu)
                    V_gpu = V_gpu + omega_map_gpu * delta_V_clamped_gpu
                    last_was_anderson = False

                if len(anderson_history) >= 2:
                    anderson_history.pop(0)
                anderson_history.append((V_curr_gpu, R_curr_gpu))

            # Check 1: Dual-norm convergence (Method 3)
            converged_linf = (last_diff <= tol_V)
            converged_rms  = (last_rms <= tol_rms and last_diff <= peak_guard)
            if it >= min_iters and (converged_linf or converged_rms):
                status = 'converged'
                break

            # Check 3: Stagnation detection
            if it >= min_iters + 5:
                w = diff_history[-5:]
                progress = (w[0] - w[-1]) / max(w[0], 1e-12)
                abs_change = w[0] - w[-1]
                if (progress < 0.01 and abs_change < 0.05) or (abs_change < 0.005 and last_diff <= 0.30):
                    status = 'stagnated_noise_floor' if last_diff <= 0.30 else 'stagnated'
                    break

        if status in ['diverged', 'stagnated']:
            print(f"[Poisson Warning] GPU Picard {status} at iter {iters_done} (diff={last_diff:.3f}V). Triggering Newton-Raphson fallback...")
            self.V = cp.asnumpy(V_best_gpu)
            V_NR, iters_NR, diff_NR, status_NR = self._solve_poisson_newton_gpu(coeff, V_plasma, Te_up, n0, tol_V=tol_V, max_iters=15)
            self.V = V_NR
            self._last_anderson_count = anderson_count
            self._last_backtrack_count = backtrack_count
            return iters_done + iters_NR, diff_NR, diff_NR, status_NR

        self._last_anderson_count = anderson_count
        self._last_backtrack_count = backtrack_count
        self.V = cp.asnumpy(V_gpu)
        return iters_done, last_diff, last_rms, status

    def _solve_poisson_newton_gpu(self, coeff, V_plasma, Te_up, n0, tol_V=0.05, max_iters=15):
        if cp is None or getattr(self, 'laplacian_matrix_gpu', None) is None:
            # Fall back to CPU Newton-Raphson if GPU/CuPy is unavailable
            return self._solve_poisson_newton_cpu(coeff, V_plasma, Te_up, n0, tol_V=tol_V, max_iters=max_iters)

        try:
            import cupyx.scipy.sparse.linalg as cp_spla
        except ImportError:
            cp_spla = None

        V_current = cp.asarray(self.V, dtype=cp.float64)
        V_fixed_flat = cp.asarray(self.V_fixed.flatten(), dtype=cp.float64)
        
        rhs_rho_mask = getattr(self, 'is_rhs_rho_mask', self.is_interior_mask)
        rhs_rho_mask_gpu = getattr(self, 'is_rhs_rho_mask_gpu', None)
        if rhs_rho_mask_gpu is None:
            rhs_rho_mask_gpu = cp.asarray(rhs_rho_mask)
            
        bound_mask_gpu = getattr(self, 'is_bound_mask_gpu', None)
        if bound_mask_gpu is None:
            bound_mask_gpu = cp.asarray(self.is_bound_mask)
            
        rho_gpu = cp.asarray(self.rho, dtype=cp.float64)
        last_diff = 0.0

        for it in range(max_iters):
            boltzmann_factor = cp.exp((cp.minimum(V_current, V_plasma) - V_plasma) / Te_up)
            rho_e = -self.q * n0 * boltzmann_factor
            rho_total = rho_gpu + rho_e
            rho_flat = rho_total.flatten()

            b = cp.zeros(self.nx * self.ny, dtype=cp.float64)
            b[bound_mask_gpu] = V_fixed_flat[bound_mask_gpu]
            b[rhs_rho_mask_gpu] = -coeff * rho_flat[rhs_rho_mask_gpu]

            # F(V) = A * V - b
            F = self.laplacian_matrix_gpu.dot(V_current.flatten()) - b

            # Jacobian J = A - d(b)/dV
            drho_e_dV = cp.zeros_like(rho_e)
            mask_less = V_current < V_plasma
            drho_e_dV[mask_less] = -(self.q * n0 / Te_up) * boltzmann_factor[mask_less]

            db_dV = cp.zeros(self.nx * self.ny, dtype=cp.float64)
            db_dV[rhs_rho_mask_gpu] = -coeff * drho_e_dV.flatten()[rhs_rho_mask_gpu]

            J = self.laplacian_matrix_gpu - cp_sp.diags(db_dV)

            # Solve J * delta_V = -F
            delta_V_flat = None
            if cp_spla is not None:
                try:
                    delta_V_flat, exitCode = cp_spla.gmres(J, -F, tol=1e-5, maxiter=200)
                    if exitCode != 0:
                        delta_V_flat = None
                except Exception:
                    delta_V_flat = None

            if delta_V_flat is None:
                # Direct solve fallback via SciPy SuperLU
                try:
                    import scipy.sparse.linalg as spla
                    J_cpu = J.get()
                    F_cpu = F.get()
                    delta_V_flat_cpu = spla.spsolve(J_cpu, -F_cpu)
                    delta_V_flat = cp.asarray(delta_V_flat_cpu)
                except Exception as e:
                    print(f"[Newton GPU] Linear solver fallback failed: {e}")
                    break

            delta_V = delta_V_flat.reshape((self.ny, self.nx))
            
            max_step = 25.0
            delta_V = cp.clip(delta_V, -max_step, max_step)
            
            V_current += delta_V
            last_diff = float(cp.max(cp.abs(delta_V)))

            if last_diff <= tol_V:
                return cp.asnumpy(V_current), it + 1, last_diff, 'converged_newton'

        return cp.asnumpy(V_current), max_iters, last_diff, 'diverged_newton'
