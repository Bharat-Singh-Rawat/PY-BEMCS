"""
Runtime Performance Diagnostics for PY-BEMCS.

Tracks per-step simulation metrics (particle counts, injection/loss balance,
PPC statistics, memory usage, wall-clock timing) and provides periodic console
summaries, threshold-based warnings, and CSV export for post-processing.

Usage:
    from diagnostics.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor(log_every=50)
    # ... inside simulation loop:
    monitor.record_step(sim, step_wall_time)
    # ... after simulation:
    monitor.export_csv("perf_log.csv")
"""

import csv
import os
import time
from dataclasses import dataclass, field, fields
from typing import List, Optional

import numpy as np

# Optional: psutil for RSS memory tracking (graceful fallback if missing)
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False


@dataclass
class StepDiagnostics:
    """Container for all per-step performance metrics."""
    iteration: int = 0
    sim_time_s: float = 0.0
    wall_time_ms: float = 0.0

    # Particle counts
    num_ions: int = 0
    num_electrons: int = 0

    # Injection / loss balance
    injected_this_step: int = 0
    lost_to_grid: int = 0
    lost_to_oob: int = 0
    transmitted: int = 0
    net_balance: int = 0               # injected - (grid + oob + transmitted)

    # PPC statistics
    mean_ppc_all: float = 0.0          # mean over all active (non-zero) cells
    mean_ppc_upstream: float = 0.0     # mean over cells before first grid
    mean_ppc_plume: float = 0.0        # mean over cells after last grid
    min_ppc_nonzero: int = 0           # lowest non-zero PPC
    low_ppc_cell_count: int = 0        # cells with 0 < PPC < 3
    total_active_cells: int = 0

    # Macro-weight and physical consistency
    macro_weight: float = 0.0
    total_physical_ions: float = 0.0   # num_ions * macro_weight
    injection_area_m2: float = 0.0     # logged for sanity checking

    # Memory
    memory_rss_mb: float = 0.0
    ion_array_mb: float = 0.0

    # Array reallocation events (cumulative)
    array_realloc_count: int = 0

    # Poisson solver convergence
    poisson_iters: int = 0
    poisson_delta_V: float = 0.0
    poisson_rms: float = 0.0
    poisson_converged: bool = True
    poisson_status: str = 'converged'
    poisson_anderson_steps: int = 0
    poisson_backtracks: int = 0


class PerformanceMonitor:
    """
    Lightweight runtime diagnostics for DigitalTwinSimulator.

    Parameters
    ----------
    log_every : int
        Print a summary line to the console every N steps.
    warn_particles : int
        Emit a warning when num_ions or num_electrons exceeds this.
    warn_memory_mb : float
        Emit a warning when process RSS exceeds this (MB).
    warn_step_time_ms : float
        Emit a warning when a single step exceeds this (ms).
    """

    def __init__(
        self,
        log_every: int = 50,
        warn_particles: int = 500_000,
        warn_memory_mb: float = 2000.0,
        warn_step_time_ms: float = 1000.0,
    ):
        self.log_every = log_every
        self.warn_particles = warn_particles
        self.warn_memory_mb = warn_memory_mb
        self.warn_step_time_ms = warn_step_time_ms

        self.history: List[StepDiagnostics] = []
        self._process = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None
        self._prev_max_p: int = 0   # track array reallocations

        # Warnings are throttled: print at most once every N steps
        self._warn_throttle_steps = 50
        self._last_particle_warn_iter = -999
        self._last_memory_warn_iter = -999

    # ------------------------------------------------------------------ #
    #  Main recording entry point                                         #
    # ------------------------------------------------------------------ #
    def record_step(self, sim, step_wall_time_s: float):
        """
        Record diagnostics for the current step.

        Parameters
        ----------
        sim : DigitalTwinSimulator
            The simulator instance (read-only access to its state).
        step_wall_time_s : float
            Wall-clock time for this step, in seconds.
        """
        d = StepDiagnostics()
        d.iteration = sim.iteration
        d.sim_time_s = sim.iteration * sim.dt
        d.wall_time_ms = step_wall_time_s * 1000.0

        # Particle counts
        d.num_ions = int(sim.num_p)
        d.num_electrons = int(sim.num_e)

        # Injection / loss balance
        d.injected_this_step = int(getattr(sim, 'injected_ions_step', 0))
        d.lost_to_grid = int(getattr(sim, 'lost_to_grid_step', 0))
        d.lost_to_oob = int(getattr(sim, 'lost_to_oob_step', 0))
        d.transmitted = int(getattr(sim, 'transmitted_ions_step', 0))
        d.net_balance = d.injected_this_step - (d.lost_to_grid + d.lost_to_oob + d.transmitted)

        # PPC statistics (use current_ppc_map if available)
        ppc_map = getattr(sim, 'current_ppc_map', None)
        if ppc_map is not None and ppc_map.size > 0:
            active_mask = ppc_map > 0
            d.total_active_cells = int(np.count_nonzero(active_mask))
            if d.total_active_cells > 0:
                d.mean_ppc_all = float(np.mean(ppc_map[active_mask]))
                d.min_ppc_nonzero = int(np.min(ppc_map[active_mask]))
                d.low_ppc_cell_count = int(np.count_nonzero(
                    (ppc_map > 0) & (ppc_map < 3)
                ))

            # Regional PPC: upstream (before first grid) and plume (after last grid)
            grid_x_starts = getattr(sim, 'grid_x_starts', [])
            grid_x_ends = getattr(sim, 'grid_x_ends', [])
            if grid_x_starts and grid_x_ends:
                x_first = grid_x_starts[0]
                x_last = grid_x_ends[-1]
                # Build x-coordinate for each cell column
                x_cell = getattr(sim, 'x_coords', getattr(sim, 'xpts', np.arange(sim.nx) * sim.dx))
                upstream_cols = x_cell < x_first
                plume_cols = x_cell > x_last

                if np.any(upstream_cols):
                    upstream_ppc = ppc_map[:, upstream_cols]
                    up_active = upstream_ppc > 0
                    d.mean_ppc_upstream = float(np.mean(upstream_ppc[up_active])) if np.any(up_active) else 0.0

                if np.any(plume_cols):
                    plume_ppc = ppc_map[:, plume_cols]
                    pl_active = plume_ppc > 0
                    d.mean_ppc_plume = float(np.mean(plume_ppc[pl_active])) if np.any(pl_active) else 0.0

        # Macro-weight
        d.macro_weight = float(getattr(sim, 'macro_weight', 0))
        d.total_physical_ions = d.num_ions * d.macro_weight
        d.injection_area_m2 = float(getattr(sim, '_last_injection_area', 0.0))

        # Poisson solver
        d.poisson_iters = int(getattr(sim, 'last_poisson_iters', 0))
        d.poisson_delta_V = float(getattr(sim, 'last_poisson_delta_V', 0.0))
        d.poisson_rms = float(getattr(sim, 'last_poisson_rms', 0.0))
        d.poisson_converged = bool(getattr(sim, 'last_poisson_converged', True))
        d.poisson_status = str(getattr(sim, 'last_poisson_status', 'converged'))
        d.poisson_anderson_steps = int(getattr(sim, 'last_poisson_anderson_steps', 0))
        d.poisson_backtracks = int(getattr(sim, 'last_poisson_backtracks', 0))

        # Memory
        if self._process is not None:
            try:
                d.memory_rss_mb = self._process.memory_info().rss / 1e6
            except Exception:
                d.memory_rss_mb = 0.0
        d.ion_array_mb = d.num_ions * 6 * 4 / 1e6   # 6 float32 arrays

        # Array reallocation detection
        current_max_p = getattr(sim, 'max_p', 0)
        if self._prev_max_p > 0 and current_max_p > self._prev_max_p:
            d.array_realloc_count = (
                self.history[-1].array_realloc_count + 1
                if self.history else 1
            )
        elif self.history:
            d.array_realloc_count = self.history[-1].array_realloc_count
        self._prev_max_p = current_max_p

        self.history.append(d)

        # Console output
        self._check_warnings(d)
        if sim.iteration % self.log_every == 0:
            self._print_summary(d)

    # ------------------------------------------------------------------ #
    #  Console output                                                     #
    # ------------------------------------------------------------------ #
    def _print_summary(self, d: StepDiagnostics):
        parts = [
            f"[PERF] Iter {d.iteration:>6d}",
            f"ions={d.num_ions:>7,}",
            f"e-={d.num_electrons:>6,}",
            f"inj={d.injected_this_step:>5d}",
            f"lost_g={d.lost_to_grid:>4d}",
            f"lost_o={d.lost_to_oob:>4d}",
            f"tx={d.transmitted:>4d}",
            f"net={d.net_balance:>+5d}",
            f"PPC_up={d.mean_ppc_upstream:>5.1f}",
            f"PPC_pl={d.mean_ppc_plume:>5.1f}",
            f"P_it={d.poisson_iters:>2d}(dV={d.poisson_delta_V:.2f}V,rms={d.poisson_rms*1000:.1f}mV)",
            f"dt={d.wall_time_ms:>6.1f}ms",
        ]
        if d.memory_rss_mb > 0:
            parts.append(f"RSS={d.memory_rss_mb:>.0f}MB")
        print(" | ".join(parts))

    def _check_warnings(self, d: StepDiagnostics):
        it = d.iteration

        if (d.num_ions > self.warn_particles or d.num_electrons > self.warn_particles):
            if it - self._last_particle_warn_iter >= self._warn_throttle_steps:
                print(
                    f"[PERF WARNING] Iter {it}: "
                    f"ions={d.num_ions:,}, electrons={d.num_electrons:,} "
                    f"(threshold: {self.warn_particles:,})"
                )
                self._last_particle_warn_iter = it

        if d.memory_rss_mb > self.warn_memory_mb:
            if it - self._last_memory_warn_iter >= self._warn_throttle_steps:
                print(
                    f"[PERF WARNING] Iter {it}: "
                    f"RSS = {d.memory_rss_mb:.0f} MB "
                    f"(threshold: {self.warn_memory_mb:.0f} MB)"
                )
                self._last_memory_warn_iter = it

        if d.wall_time_ms > self.warn_step_time_ms:
            print(
                f"[PERF WARNING] Iter {it}: "
                f"step took {d.wall_time_ms:.0f} ms "
                f"(threshold: {self.warn_step_time_ms:.0f} ms)"
            )

        if d.poisson_status == 'diverged':
            print(
                f"[PERF WARNING] Iter {it}: "
                f"Poisson solver DIVERGED (error grew to {d.poisson_delta_V:.2f} V). Terminated early."
            )
        elif d.poisson_status == 'stagnated':
            print(
                f"[PERF WARNING] Iter {it}: "
                f"Poisson solver STAGNATED (stuck at delta_V = {d.poisson_delta_V:.2f} V after {d.poisson_iters} iters)."
            )
        elif d.poisson_delta_V > 1.0:
            print(
                f"[PERF WARNING] Iter {it}: "
                f"Poisson under-converged (delta_V = {d.poisson_delta_V:.2f} V after {d.poisson_iters} iters)"
            )

        if d.array_realloc_count > 0 and (
            not self.history or
            d.array_realloc_count > self.history[-2].array_realloc_count
            if len(self.history) >= 2 else d.array_realloc_count > 0
        ):
            print(
                f"[PERF INFO] Iter {it}: "
                f"ion array reallocated (now max_p={self._prev_max_p:,}, "
                f"total reallocs={d.array_realloc_count})"
            )

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #
    def export_csv(self, path: str):
        """Export the full diagnostics history to a CSV file."""
        if not self.history:
            print("[PERF] No data to export.")
            return

        field_names = [f.name for f in fields(StepDiagnostics)]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(field_names)
            for d in self.history:
                writer.writerow([getattr(d, name) for name in field_names])
        print(f"[PERF] Exported {len(self.history)} steps to {path}")

    def get_summary_dict(self) -> dict:
        """Return a summary of the latest diagnostics as a dictionary."""
        if not self.history:
            return {}
        d = self.history[-1]
        return {f.name: getattr(d, f.name) for f in fields(StepDiagnostics)}

    def print_final_report(self):
        """Print a final summary after the simulation ends."""
        if not self.history:
            print("[PERF] No steps recorded.")
            return

        n = len(self.history)
        last = self.history[-1]
        avg_dt = np.mean([d.wall_time_ms for d in self.history])
        max_ions = max(d.num_ions for d in self.history)
        total_inj = sum(d.injected_this_step for d in self.history)
        total_lost_g = sum(d.lost_to_grid for d in self.history)
        total_lost_o = sum(d.lost_to_oob for d in self.history)
        total_tx = sum(d.transmitted for d in self.history)

        print("\n" + "=" * 70)
        print("  PY-BEMCS PERFORMANCE REPORT")
        print("=" * 70)
        print(f"  Steps recorded:        {n}")
        print(f"  Sim time:              {last.sim_time_s:.4e} s")
        print(f"  Avg step time:         {avg_dt:.1f} ms")
        print(f"  Peak ion count:        {max_ions:,}")
        print(f"  Final ion count:       {last.num_ions:,}")
        print(f"  Final electron count:  {last.num_electrons:,}")
        print(f"  Total injected:        {total_inj:,}")
        print(f"  Total lost to grids:   {total_lost_g:,}")
        print(f"  Total lost to OOB:     {total_lost_o:,}")
        print(f"  Total transmitted:     {total_tx:,}")
        print(f"  Macro-weight:          {last.macro_weight:.2e}")
        print(f"  Injection area:        {last.injection_area_m2:.4e} m^2")
        print(f"  Array reallocs:        {last.array_realloc_count}")
        if last.memory_rss_mb > 0:
            print(f"  Final RSS:             {last.memory_rss_mb:.0f} MB")
        print(f"  Final PPC upstream:    {last.mean_ppc_upstream:.1f}")
        print(f"  Final PPC plume:       {last.mean_ppc_plume:.1f}")
        avg_p_iter = np.mean([d.poisson_iters for d in self.history])
        max_p_iter = max(d.poisson_iters for d in self.history)
        max_p_dv   = max(d.poisson_delta_V for d in self.history)
        print(f"  Poisson iters (avg/max): {avg_p_iter:.1f} / {max_p_iter}")
        print(f"  Max Poisson delta V:   {max_p_dv:.4f} V")
        print("=" * 70 + "\n")
