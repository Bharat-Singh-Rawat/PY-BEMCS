"""
Poisson solver dispatcher and electric field calculation mixin for DigitalTwinSimulator.
"""
import numpy as np

from ..backend import _NP_FP, _GPU_POISSON
from .poisson_cpu import PoissonCPUMixin
from .poisson_gpu import PoissonGPUMixin


class PoissonMixin(PoissonCPUMixin, PoissonGPUMixin):
    """Main Poisson solver dispatcher, tolerance management, and electric field gradient computation."""

    def recalc_poisson(self, iterations=None, params=None, tol_V=None, min_iters=3):
        if self.laplacian_lu is None:
            return

        dy_m2 = (self.dy * 1e-3) ** 2
        coeff = dy_m2 / self.eps0

        if params is None:
            params = {}
        grids = params.get('grids', [{'V': 1000}])
        v_offset = params.get('V_plasma_offset', 20.0)
        V_plasma = grids[0]['V'] + v_offset
        Te_up = params.get('Te_up', 3.0)
        n0 = params.get('n0_plasma', 1e17)
        omega_min = float(params.get('poisson_omega', 0.2))
        omega_max = float(params.get('poisson_omega_max', 0.35)) if params.get('poisson_adaptive_omega', False) else omega_min

        target_tol = float(tol_V if tol_V is not None else params.get('poisson_tol_V', 0.05))
        tol_rms = float(params.get('poisson_tol_V_rms', target_tol * 0.2))
        peak_guard = float(params.get('poisson_tol_peak_guard', target_tol * 5.0))
        min_it = int(min_iters if min_iters is not None else params.get('poisson_min_iters', 3))

        default_max = 100 if getattr(self, 'iteration', 0) == 0 else 100
        if iterations is not None and iterations > default_max:
            max_it = iterations
        elif iterations is not None and not params.get('poisson_adaptive', True):
            max_it = iterations
            target_tol = 0.0
        else:
            max_it = int(params.get('poisson_max_iters', default_max))

        if _GPU_POISSON and self.laplacian_lu_gpu is not None:
            n_done, delta_V, rms_V, status = self._recalc_poisson_gpu(
                max_it, coeff, V_plasma, Te_up, n0, omega_min, omega_max, target_tol, min_it, tol_rms, peak_guard
            )
        else:
            n_done, delta_V, rms_V, status = self._recalc_poisson_cpu(
                max_it, coeff, V_plasma, Te_up, n0, omega_min, omega_max, target_tol, min_it, tol_rms, peak_guard
            )

        self.last_poisson_iters = n_done
        self.last_poisson_delta_V = delta_V
        self.last_poisson_rms = rms_V
        self.last_poisson_status = status
        self.last_poisson_converged = (status in ['converged', 'stagnated_noise_floor', 'converged_newton'])
        self.last_poisson_anderson_steps = getattr(self, '_last_anderson_count', 0)
        self.last_poisson_backtracks = getattr(self, '_last_backtrack_count', 0)

        if status == 'diverged':
            bt_info = f", backtracks={self.last_poisson_backtracks}" if self.last_poisson_backtracks > 0 else ""
            print(
                f"[Poisson Warning] Iter {self.iteration}: Divergence detected in Poisson solver "
                f"(error grew to {delta_V:.2f} V, rms {rms_V*1000:.1f} mV{bt_info}). Terminated early at Picard iter {n_done}."
            )
        elif status == 'stagnated':
            details = []
            if self.last_poisson_anderson_steps > 0:
                details.append(f"AA steps={self.last_poisson_anderson_steps}")
            if self.last_poisson_backtracks > 0:
                details.append(f"backtracks={self.last_poisson_backtracks}")
            extra_info = f", {', '.join(details)}" if details else ""
            print(
                f"[Poisson Warning] Iter {self.iteration}: Stagnation detected in Poisson solver "
                f"(stuck at delta_V = {delta_V:.2f} V, rms {rms_V*1000:.1f} mV, progress < 2% over 5 iters{extra_info}). Terminated at Picard iter {n_done}."
            )
        elif status == 'converged_newton':
            print(
                f"[Poisson Info] Iter {self.iteration}: Successfully converged via Newton-Raphson fallback "
                f"(final delta_V = {delta_V:.4f} V across {n_done} total iters)."
            )
        elif status == 'diverged_newton':
            print(
                f"[Poisson Warning] Iter {self.iteration}: Newton-Raphson fallback failed to reach tolerance "
                f"(final delta_V = {delta_V:.4f} V)."
            )

        self.Ey, self.Ex = np.gradient(-self.V, self.y_coords * 1e-3, self.x_coords * 1e-3)

        if getattr(self, 'periodic_y', False):
            two_dy_m = 2.0 * self.dy * 1e-3
            self.Ey[0, :] = (self.V[self.ny - 1, :] - self.V[1, :]) / two_dy_m
            self.Ey[self.ny - 1, :] = (self.V[self.ny - 2, :] - self.V[0, :]) / two_dy_m

        self.Ey = self.Ey.astype(_NP_FP)
        self.Ex = self.Ex.astype(_NP_FP)
