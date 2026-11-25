import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication

from physics.step.ion_walls import compute_beam_divergence, step_ion_interactions
from physics.step.context import StepContext
from physics.simulator import DigitalTwinSimulator
from gui.app.main_window import DigitalTwinApp


class TestDivergenceEvaluationMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_direct_rms(self):
        # 100 particles with known angles +/- 10 deg
        N = 100
        angles = np.array([-10.0, 10.0] * 50)
        p_vx = np.full(N, 20000.0)
        p_vy = p_vx * np.tan(angles * np.pi / 180.0)
        p_y = np.linspace(0.0, 1.0, N)
        mask = np.ones(N, dtype=bool)

        div_val = compute_beam_divergence(p_y, p_vx, p_vy, mask, method="Direct RMS")
        self.assertAlmostEqual(div_val, 10.0, places=2)

    def test_rms_twiss(self):
        # For a centered distribution, 1-sigma Twiss RMS should equal std(angles)
        N = 200
        np.random.seed(123)
        angles = np.random.normal(0, 8.0, N)
        p_vx = np.full(N, 20000.0)
        p_vy = p_vx * np.tan(angles * np.pi / 180.0)
        p_y = np.linspace(0.0, 1.5, N)
        mask = np.ones(N, dtype=bool)

        div_val = compute_beam_divergence(p_y, p_vx, p_vy, mask, method="RMS (Twiss)", n_sigmas=1.0)
        expected_std = np.std(angles)
        self.assertAlmostEqual(div_val, expected_std, places=2)

        # n_sigmas = 2.0 should scale by 2
        div_2sigma = compute_beam_divergence(p_y, p_vx, p_vy, mask, method="RMS (Twiss)", n_sigmas=2.0)
        self.assertAlmostEqual(div_2sigma, 2.0 * div_val, places=2)

    def test_fwhm(self):
        # Symmetric distribution with peak at 0
        N = 400
        np.random.seed(42)
        true_sigma = 6.0
        angles = np.random.normal(0, true_sigma, N)
        p_vx = np.full(N, 20000.0)
        p_vy = p_vx * np.tan(angles * np.pi / 180.0)
        p_y = np.linspace(0.0, 1.0, N)
        mask = np.ones(N, dtype=bool)

        div_fwhm = compute_beam_divergence(p_y, p_vx, p_vy, mask, method="FWHM")
        # Theoretical half-FWHM of Gaussian is sqrt(2*ln(2)) * sigma ~ 1.1774 * 6.0 = 7.06
        self.assertGreater(div_fwhm, 4.0)
        self.assertLess(div_fwhm, 10.0)

    def test_insufficient_particles(self):
        p_y = np.array([0.1, 0.2, 0.3])
        p_vx = np.array([1000.0, 1000.0, 1000.0])
        p_vy = np.array([100.0, 100.0, 100.0])
        mask = np.ones(3, dtype=bool)

        div_val = compute_beam_divergence(p_y, p_vx, p_vy, mask, method="Direct RMS")
        self.assertTrue(np.isnan(div_val))

    def test_gui_menu_and_method_switching(self):
        win = DigitalTwinApp()
        self.assertIn("RMS (Twiss)", win.DIVERGENCE_METHODS)
        self.assertIn("Direct RMS", win.DIVERGENCE_METHODS)
        self.assertIn("FWHM", win.DIVERGENCE_METHODS)

        # Check menu actions exist
        self.assertIn("RMS (Twiss)", win._div_method_actions)
        self.assertIn("Direct RMS", win._div_method_actions)
        self.assertIn("FWHM", win._div_method_actions)

        # Switch to Direct RMS
        win.set_divergence_method("Direct RMS")
        self.assertEqual(win.div_method, "Direct RMS")
        self.assertEqual(win.sim.div_method, "Direct RMS")
        self.assertTrue(win._div_method_actions["Direct RMS"].isChecked())
        self.assertFalse(win._div_method_actions["95%"].isChecked())

        # Switch to RMS (Twiss)
        win.set_divergence_method("RMS (Twiss)")
        self.assertEqual(win.div_method, "RMS (Twiss)")
        self.assertEqual(win.sim.div_method, "RMS (Twiss)")
        self.assertTrue(win._div_method_actions["RMS (Twiss)"].isChecked())
        self.assertFalse(win._div_method_actions["Direct RMS"].isChecked())

        # Switch to FWHM
        win.set_divergence_method("FWHM")
        self.assertEqual(win.div_method, "FWHM")
        self.assertEqual(win.sim.div_method, "FWHM")
        self.assertTrue(win._div_method_actions["FWHM"].isChecked())

        win.close()


if __name__ == "__main__":
    unittest.main()
