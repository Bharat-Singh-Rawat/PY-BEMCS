import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication

from physics.simulator import DigitalTwinSimulator
from diagnostics.metrics import compute_rolling_variation
from diagnostics.steady_state_window import SteadyStateWindow
from gui.app.main_window import DigitalTwinApp


class TestSteadyStateMetrics(unittest.TestCase):
    def test_compute_rolling_variation_empty_or_short(self):
        iters, var = compute_rolling_variation([], window=50)
        self.assertEqual(len(iters), 0)
        self.assertEqual(len(var), 0)

        iters, var = compute_rolling_variation([10.0], window=50)
        self.assertEqual(len(iters), 0)
        self.assertEqual(len(var), 0)

    def test_compute_rolling_variation_constant_series(self):
        # 100 iterations of constant 42.0
        const_series = [42.0] * 100
        iters, var = compute_rolling_variation(const_series, window=50, mode="rel_change")
        self.assertEqual(len(iters), 50)
        self.assertEqual(iters[0], 50)
        np.testing.assert_allclose(var, 0.0, atol=1e-10)

        iters_p2p, var_p2p = compute_rolling_variation(const_series, window=50, mode="p2p_spread")
        self.assertEqual(len(var_p2p), 50)
        np.testing.assert_allclose(var_p2p, 0.0, atol=1e-10)

    def test_compute_rolling_variation_linear_growth(self):
        # Linear growth: Q[k] = 100 + k
        # Over 50 iterations: Q[k] - Q[k-50] = 50.
        # At k=50: base = 100, diff = 50 -> 50%
        # At k=100: base = 150, diff = 50 -> 33.333%
        series = [100.0 + k for k in range(101)]
        iters, var = compute_rolling_variation(series, window=50, mode="rel_change")
        self.assertEqual(len(iters), 51)
        self.assertAlmostEqual(var[0], 50.0, places=4)
        self.assertAlmostEqual(var[-1], (50.0 / 150.0) * 100.0, places=4)

    def test_compute_rolling_variation_abs_change(self):
        # Decreasing series: Q[k] = 200 - k
        series = [200.0 - k for k in range(101)]
        iters, var_abs = compute_rolling_variation(series, window=50, mode="abs_change")
        self.assertTrue(np.all(var_abs > 0))


class TestSteadyStateWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self):
        self.sim = DigitalTwinSimulator()
        self.sim.iteration = 100
        # Populate mock history of 100 steps
        self.sim.energy_history_tot = [1.0e-3 * (1.0 + 0.001 * np.sin(k * 0.1)) for k in range(100)]
        self.sim.charge_history_q_net = [1.0e-9 * (1.0 + 0.002 * np.cos(k * 0.1)) for k in range(100)]
        self.sim.energy_history_fe = [5.0e-4 * (1.0 + 0.001 * np.sin(k * 0.05)) for k in range(100)]
        self.sim.potential_history = [-800.0 + 0.1 * np.sin(k * 0.2) for k in range(100)]

    def test_window_initialization(self):
        dlg = SteadyStateWindow()
        self.assertEqual(dlg.spin_window.value(), 50)
        self.assertEqual(dlg.spin_thresh.value(), 1.0)
        self.assertIn("energy_tot", dlg.chk_quantities)
        self.assertIn("charge_net", dlg.chk_quantities)
        self.assertIn("field_energy", dlg.chk_quantities)
        self.assertIn("potential", dlg.chk_quantities)
        dlg.close()

    def test_update_plot_and_steady_state_detection(self):
        dlg = SteadyStateWindow()
        dlg.update_plot(self.sim)

        # Since fluctuation amplitude is <= 0.2%, with thresh = 1.0% it should be steady state
        dlg.spin_thresh.setValue(1.0)
        dlg._replot()
        self.assertIn("STEADY STATE ACHIEVED", dlg.lbl_status_text.text())

        # If threshold is extremely tight (0.01%), it should report transient
        dlg.spin_thresh.setValue(0.01)
        dlg._replot()
        self.assertIn("TRANSIENT", dlg.lbl_status_text.text())

        dlg.close()

    def test_mode_and_quantity_toggling(self):
        dlg = SteadyStateWindow()
        dlg.update_plot(self.sim)

        # Toggle modes
        for idx in range(dlg.combo_mode.count()):
            dlg.combo_mode.setCurrentIndex(idx)
            dlg._replot()

        # Select all / Deselect all
        dlg._set_all_quantities(False)
        for chk in dlg.chk_quantities.values():
            self.assertFalse(chk.isChecked())

        dlg._set_all_quantities(True)
        for chk in dlg.chk_quantities.values():
            self.assertTrue(chk.isChecked())

        dlg.close()

    def test_gui_menu_action_opens_steady_state_window(self):
        win = DigitalTwinApp()
        self.assertIsNone(win.steady_state_window)

        # Trigger open_steady_state_window
        win.open_steady_state_window()
        self.assertIsNotNone(win.steady_state_window)
        self.assertTrue(win.steady_state_window.isVisible())

        # Call update_plot
        win.steady_state_window.update_plot(win.sim, win)

        win.steady_state_window.close()
        win.close()


if __name__ == "__main__":
    unittest.main()
