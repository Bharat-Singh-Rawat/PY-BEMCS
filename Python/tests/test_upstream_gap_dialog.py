import os
import sys
import unittest
from PyQt5.QtWidgets import QApplication

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.dialogs import UpstreamGapDialog
from gui.app.main_window import DigitalTwinApp


class TestUpstreamGapDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(["--platform", "offscreen"])

    def test_presheath_default_ratio(self):
        screen_r = 0.80
        dlg = UpstreamGapDialog(
            current_gap=0.60,
            screen_r=screen_r,
            n0=1e18,
            Te_up=3.0,
            is_bulk=False
        )
        self.assertTrue(dlg.radio_ratio.isChecked())
        self.assertEqual(dlg.combo_base.currentData(), "screen_r")
        self.assertAlmostEqual(dlg.spin_ratio.value(), 0.75, places=3)
        gap, settings = dlg.get_result()
        self.assertEqual(settings["method"], "ratio")
        self.assertAlmostEqual(gap, 0.60, places=3)
        dlg.close()

    def test_modify_ratio(self):
        screen_r = 0.80
        dlg = UpstreamGapDialog(
            current_gap=0.60,
            screen_r=screen_r,
            n0=1e18,
            Te_up=3.0,
            is_bulk=False
        )
        dlg.spin_ratio.setValue(1.20)
        gap, settings = dlg.get_result()
        self.assertEqual(settings["method"], "ratio")
        self.assertAlmostEqual(settings["ratio"], 1.20, places=3)
        self.assertAlmostEqual(gap, 0.96, places=3)
        dlg.close()

    def test_custom_value_without_ratio(self):
        dlg = UpstreamGapDialog(
            current_gap=0.60,
            screen_r=0.80,
            n0=1e18,
            Te_up=3.0,
            is_bulk=False
        )
        dlg.radio_custom.setChecked(True)
        dlg.spin_custom.setValue(1.85)
        gap, settings = dlg.get_result()
        self.assertEqual(settings["method"], "custom")
        self.assertAlmostEqual(gap, 1.85, places=3)
        dlg.close()

    def test_bulk_plasma_debye_ratio(self):
        n0 = 1e18
        Te_up = 3.0
        dlg = UpstreamGapDialog(
            current_gap=0.0,
            screen_r=0.80,
            n0=n0,
            Te_up=Te_up,
            is_bulk=True
        )
        self.assertTrue(dlg.radio_ratio.isChecked())
        self.assertEqual(dlg.combo_base.currentData(), "debye")
        self.assertAlmostEqual(dlg.spin_ratio.value(), 30.0, places=1)
        expected_debye = (8.854e-12 * Te_up / (1.6e-19 * n0 * 0.61))**0.5 * 1e3
        expected_gap = round(30.0 * expected_debye, 3)
        gap, settings = dlg.get_result()
        self.assertAlmostEqual(gap, expected_gap, places=3)
        dlg.close()

    def test_main_window_settings_button_and_custom_gap(self):
        win = DigitalTwinApp()
        self.assertTrue(hasattr(win, "btn_upstream_gap_settings"))
        self.assertEqual(win.btn_upstream_gap_settings.text(), "Settings")

        # Simulate applying custom settings
        win.upstream_gap_settings = {
            "method": "custom",
            "ratio_base": "screen_r",
            "ratio": 0.75,
            "custom_value": 2.50,
            "final_gap": 2.50,
        }
        win.inputs["upstream_gap_mm"].setValue(2.50)
        win._last_auto_gap = 2.50

        # Change plasma density which would normally trigger _update_debye_gap
        win.inputs["n0_plasma"].setValue(5e17)
        win._update_debye_gap()
        # Custom gap should NOT be overwritten
        self.assertAlmostEqual(win.inputs["upstream_gap_mm"].value(), 2.50, places=3)
        win.close()


if __name__ == "__main__":
    unittest.main()
