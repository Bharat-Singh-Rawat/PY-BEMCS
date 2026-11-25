"""
Unit and integration tests for:
1. Exit ion current stability criteria (headless sweep, GUI sweep, and SteadyStateWindow).
2. 2D Density Diagnostics using full PPC grids, species breakdown, and volume scaling.
3. Midline Potential graph autoscaling beyond +100V and below -100V.
Verifies proper functionality in both sweep mode and standalone / single-simulation mode.
"""
import unittest
import numpy as np
import os
import sys

# Ensure offscreen Qt platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt5.QtWidgets import QApplication

_qapp = QApplication.instance()
if _qapp is None:
    _qapp = QApplication(sys.argv)

from physics.simulator import DigitalTwinSimulator
from physics.mixins.diagnostics import DiagnosticsMixin
from physics.step.context import StepContext
from physics.step.probes import step_end_accounting
from diagnostics.density_window import DensityWindow
from diagnostics.steady_state_window import SteadyStateWindow
from diagnostics.metrics import compute_rolling_variation


class MockSimForDensity(DiagnosticsMixin):
    """Mock simulator for validating density calculations."""
    def __init__(self, nx=50, ny=30):
        self.nx = nx
        self.ny = ny
        self.dx = 0.05
        self.dy = 0.05
        self.Lx = nx * self.dx
        self.Ly = ny * self.dy
        self.macro_weight = 1e8
        self.num_p = 0
        self.num_e = 0
        self.p_x = np.empty(0, dtype=np.float32)
        self.p_y = np.empty(0, dtype=np.float32)
        self.p_isCEX = np.empty(0, dtype=bool)
        self.current_ppc_map = np.zeros((ny, nx), dtype=int)
        self.accum_ppc_map = np.zeros((ny, nx), dtype=float)
        self.current_cex_ppc_map = np.zeros((ny, nx), dtype=int)
        self.accum_cex_ppc_map = np.zeros((ny, nx), dtype=float)
        self.ppc_steps_count = 0
        self.cell_vol_2d = np.full((ny, nx), (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3)
        self.isBound = np.zeros((ny, nx), dtype=bool)
        self.x_coords = np.linspace(0, self.Lx, nx)
        self.y_coords = np.linspace(0, self.Ly, ny)

    def _x_to_ix(self, x):
        return np.clip(np.round(x / self.dx).astype(int), 0, self.nx - 1)


class TestStabilityCriteria(unittest.TestCase):
    """Test exit ion current integration into stability checks."""

    def test_ion_current_history_tracked_on_sim(self):
        """Verify simulator tracks ion_current_history during step accounting."""
        sim = DigitalTwinSimulator()
        sim.nx = 50
        sim.ny = 30
        sim.dx = 0.05
        sim.dy = 0.05
        sim.num_p = 0
        sim.p_x = np.empty(0, dtype=np.float32)
        sim.p_y = np.empty(0, dtype=np.float32)
        self.assertTrue(hasattr(sim, 'ion_current_history'))
        self.assertEqual(len(sim.ion_current_history), 0)

        # Simulate step_end_accounting
        sim.exit_ion_current_step = 0.015
        sim.exit_ion_current_avg = 0.012
        ctx = StepContext(params={}, sim_mode='Both', t_current=0.0, grids=[])
        ctx.min_pot = -50.0
        ctx.current_div = 10.0
        step_end_accounting(sim, ctx)

        self.assertEqual(len(sim.ion_current_history), 1)
        self.assertAlmostEqual(sim.ion_current_history[0], 0.012)
        self.assertIs(sim.exit_ion_current_history, sim.ion_current_history)

    def test_steady_state_window_reads_sim_ion_current_without_app(self):
        """Verify SteadyStateWindow extracts ion_current directly from sim in standalone mode."""
        sim = DigitalTwinSimulator()
        sim.iteration = 100
        sim.ion_current_history = [0.010 * (1.0 + 0.001 * np.sin(k * 0.1)) for k in range(100)]
        sim.energy_history_tot = [1.0e-3] * 100
        sim.charge_history_q_net = [1.0e-9] * 100
        sim.potential_history = [-60.0] * 100

        win = SteadyStateWindow()
        raw_series = win._get_raw_series(sim, app=None)
        self.assertIn("ion_current", raw_series)
        self.assertEqual(len(raw_series["ion_current"]), 100)

        # Verify rolling variation detects stationarity
        iters, var = compute_rolling_variation(raw_series["ion_current"], window=50, mode="rel_change")
        self.assertTrue(len(var) > 0)
        self.assertLess(abs(var[-1]), 0.5)  # Less than 0.5% variation -> stable

    def test_transient_ion_current_prevents_premature_steady_state(self):
        """Verify that if energy/potential are stable but ion current is trending, steady state is not declared."""
        window = 50
        thresh = 1.0  # 1% threshold
        iters = 80

        stable_energy = [1.0e-3] * iters
        stable_charge = [1.0e-9] * iters
        stable_pot = [-60.0] * iters
        # Ion current still charging up / transient (e.g. 5% growth per 50 iterations)
        transient_current = [0.005 + 0.0002 * k for k in range(iters)]

        histories_to_check = [stable_energy, stable_charge, stable_pot, transient_current]
        all_under = True
        for h in histories_to_check:
            _, var = compute_rolling_variation(h, window=window, mode="rel_change")
            if len(var) == 0 or abs(var[-1]) > thresh:
                all_under = False
                break

        # Must NOT declare steady state because current has not stabilized!
        self.assertFalse(all_under)


class TestDensityDiagnostics(unittest.TestCase):
    """Test 2D Density Diagnostics with full PPC grids and species separation."""

    def test_density_map_uses_full_ppc_grid(self):
        """Verify get_density_map uses current_ppc_map directly to yield smooth non-empty density."""
        sim = MockSimForDensity(nx=40, ny=20)
        # Put 10 particles per cell in beam core
        sim.current_ppc_map[5:15, 10:30] = 10
        sim.current_cex_ppc_map[5:15, 10:30] = 2  # 2 CEX, 8 primary

        # Total ions
        dens_tot = sim.get_density_map(species="ions")
        self.assertEqual(dens_tot.shape, (20, 40))
        self.assertGreater(np.max(dens_tot), 0.0)
        self.assertEqual(dens_tot[0, 0], 0.0)  # Empty edge
        self.assertAlmostEqual(dens_tot[10, 20], 10 * sim.macro_weight / sim.cell_vol_2d[10, 20])

        # Primary ions
        dens_prim = sim.get_density_map(species="primary")
        self.assertAlmostEqual(dens_prim[10, 20], 8 * sim.macro_weight / sim.cell_vol_2d[10, 20])

        # CEX ions
        dens_cex = sim.get_density_map(species="cex")
        self.assertAlmostEqual(dens_cex[10, 20], 2 * sim.macro_weight / sim.cell_vol_2d[10, 20])

        # Conservation: primary + cex == total
        np.testing.assert_allclose(dens_prim + dens_cex, dens_tot)

    def test_density_window_instant_and_time_averaged(self):
        """Verify DensityWindow updates smoothly with both instantaneous and time-averaged maps."""
        sim = MockSimForDensity(nx=30, ny=20)
        sim.current_ppc_map[5:15, 5:25] = 12
        sim.accum_ppc_map[5:15, 5:25] = 120.0
        sim.ppc_steps_count = 10
        sim.iteration = 100

        win = DensityWindow()
        # Instantaneous mode
        win.combo_mode.setCurrentIndex(0)
        win.update_plot(sim, is_new_step=True)
        self.assertIn("Instantaneous", win.lbl_stats.text())

        # Time-averaged mode
        win.combo_mode.setCurrentIndex(1)
        win.update_plot(sim, is_new_step=False)
        self.assertIn("Time-Averaged", win.lbl_stats.text())


class TestMidlinePotentialAutoscaling(unittest.TestCase):
    """Test Midline / Saddle-point potential graph autoscaling beyond +100V and below -100V."""

    def test_rendering_mixin_ebs_ylim_autoscales_high_positive_potential(self):
        """Verify _update_ebs_ylim scales properly when potential exceeds +100V."""
        from gui.app.rendering import RenderingMixin
        import matplotlib.pyplot as plt

        class MockApp(RenderingMixin):
            def __init__(self):
                self.fig, self.ax_ebs = plt.subplots()
                self.ebs_history = []

        app = MockApp()
        # High density space-charge choking: saddle potential rises from -200V to +180V
        app.ebs_history = [-200.0, -100.0, 0.0, 50.0, 120.0, 180.0]
        app._update_ebs_ylim()

        ymin, ymax = app.ax_ebs.get_ylim()
        self.assertLess(ymin, -200.0)
        self.assertGreater(ymax, 180.0)  # +180V is fully within the plot!

    def test_rendering_mixin_ebs_ylim_empty_fallback(self):
        """Verify _update_ebs_ylim falls back to [-100, 100] when history is empty."""
        from gui.app.rendering import RenderingMixin
        import matplotlib.pyplot as plt

        class MockApp(RenderingMixin):
            def __init__(self):
                self.fig, self.ax_ebs = plt.subplots()
                self.ebs_history = []

        app = MockApp()
        app._update_ebs_ylim()
        ymin, ymax = app.ax_ebs.get_ylim()
        self.assertEqual((ymin, ymax), (-100.0, 100.0))


if __name__ == "__main__":
    unittest.main()
