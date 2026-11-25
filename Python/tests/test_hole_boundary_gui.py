"""
GUI integration tests for Hole-Boundary distance spinbox under Upstream Gap,
auto-tracking, domain size computation, and config save/load.
"""
import os
import unittest

os.environ["QT_API"] = "pyqt5"
from PyQt5.QtWidgets import QApplication

from gui.app.main_window import DigitalTwinApp


class TestHoleBoundaryGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(["--platform", "offscreen"])

    def test_hole_boundary_controls_and_domain_computation(self):
        win = DigitalTwinApp()
        try:
            # 1. Check spinbox exists
            self.assertIn("hole_boundary_dist_mm", win.inputs)
            spin = win.inputs["hole_boundary_dist_mm"]

            screen_r = win.grid_widgets[0]["r"].value()
            pitch = getattr(win, "pitch_mm", 3.0)
            expected_d = round(0.5 * (screen_r + pitch), 3)
            self.assertAlmostEqual(spin.value(), expected_d, places=3)

            # 2. Check Ly for half_hole, one_hole, two_holes
            win.combo_geometry.setCurrentText("half_hole")
            Lx, Ly = win.compute_grid_domain_size()
            self.assertAlmostEqual(Ly, expected_d, places=3)

            win.combo_geometry.setCurrentText("one_hole")
            Lx, Ly = win.compute_grid_domain_size()
            self.assertAlmostEqual(Ly, 2.0 * expected_d, places=3)

            win.combo_geometry.setCurrentText("two_holes")
            Lx, Ly = win.compute_grid_domain_size()
            self.assertAlmostEqual(Ly, 4.0 * expected_d, places=3)

            # 3. Manual override
            win.combo_geometry.setCurrentText("half_hole")
            spin.setValue(2.75)
            Lx, Ly = win.compute_grid_domain_size()
            self.assertAlmostEqual(Ly, 2.75, places=3)

            # 4. Auto reset
            win.btn_reset_hole_bound.click()
            self.assertAlmostEqual(spin.value(), expected_d, places=3)

            # 5. Config serialization
            spin.setValue(3.14)
            cfg = win.get_current_config_dict()
            self.assertIn("hole_boundary_dist_mm", cfg["simulation"])
            self.assertAlmostEqual(cfg["simulation"]["hole_boundary_dist_mm"], 3.14, places=3)

            # 6. Apply config
            spin.setValue(1.0)
            win.apply_config(cfg)
            self.assertAlmostEqual(spin.value(), 3.14, places=3)
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
