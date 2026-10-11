import os
import sys
import unittest
from unittest.mock import patch
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYBEMCS_FORCE_CPU"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog
from gui.app import DigitalTwinApp
from gui.dialogs import DivergencePlaneDialog
from physics_engine import DigitalTwinSimulator
from physics.step.context import StepContext
from physics.step.ion_walls import step_ion_interactions

app = QApplication.instance() or QApplication(sys.argv)


class TestDivergencePlane(unittest.TestCase):
    def test_plane_dialog_controls(self):
        # Test valid range [4.5, 10.0]
        x_min = 4.5
        x_max = 10.0
        dialog = DivergencePlaneDialog(6.0, x_min, x_max)

        self.assertEqual(dialog.spin_x.minimum(), x_min)
        self.assertEqual(dialog.spin_x.maximum(), x_max)
        self.assertAlmostEqual(dialog.get_plane_x(), 6.0, places=3)

        # Test preset buttons
        # 1. Last Grid Exit
        dialog.spin_x.setValue(x_min)
        self.assertAlmostEqual(dialog.get_plane_x(), x_min, places=3)

        # 2. Mid Plume
        mid_val = round(x_min + 0.5 * (x_max - x_min), 3)
        dialog.spin_x.setValue(mid_val)
        self.assertAlmostEqual(dialog.get_plane_x(), mid_val, places=3)

        # 3. End Boundary
        dialog.spin_x.setValue(x_max)
        self.assertAlmostEqual(dialog.get_plane_x(), x_max, places=3)

        # 4. Out-of-bounds validation check
        dialog.spin_x.setRange(0.0, 20.0) # artificially widen spinbox range to test validation guard
        dialog.spin_x.setValue(2.0) # below x_min
        with patch.object(QMessageBox, "warning") as mock_warn:
            dialog.validate_and_accept()
            mock_warn.assert_called_once()
            self.assertIn("Invalid Plane Position", mock_warn.call_args[0][1])

        dialog.spin_x.setValue(12.0) # above x_max
        with patch.object(QMessageBox, "warning") as mock_warn:
            dialog.validate_and_accept()
            mock_warn.assert_called_once()
            self.assertIn("Invalid Plane Position", mock_warn.call_args[0][1])

    def test_menu_plane_button_and_dialog_integration(self):
        win = DigitalTwinApp()
        self.assertTrue(hasattr(win, "action_div_plane"))
        self.assertEqual(win.action_div_plane.text(), "Plane")
        self.assertIn(win.action_div_plane, win.menu_div.actions())

        # Verify line_div_mid uses solid continuous line
        self.assertEqual(win.line_div_mid.get_linestyle(), "-")

        # Test limits calculation and domain build
        win.build_domain()
        self.assertIsNotNone(win.line_div_plane)
        self.assertIn(win.line_div_plane, win.ax_live.lines)
        self.assertEqual(win.line_div_plane.get_linestyle(), ":")
        self.assertEqual(win.line_div_plane.get_color(), "#e67e22")

        x_min, x_max = win.get_divergence_plane_limits()
        self.assertGreater(x_max, x_min)

        # Test changing plane via dialog
        target_plane = round(x_min + 0.7 * (x_max - x_min), 3)
        with patch.object(DivergencePlaneDialog, "exec_", return_value=QDialog.Accepted), \
             patch.object(DivergencePlaneDialog, "get_plane_x", return_value=target_plane):
            win.open_divergence_plane_dialog()

        self.assertAlmostEqual(win.div_plane_x, target_plane, places=3)
        self.assertAlmostEqual(win.sim.div_plane_x, target_plane, places=3)

        # Verify dotted plane line in ax_live updated to target_plane
        plane_xdata = list(win.line_div_plane.get_xdata())
        self.assertAlmostEqual(plane_xdata[0], target_plane, places=3)
        self.assertAlmostEqual(plane_xdata[1], target_plane, places=3)

        # Verify legend updated with custom plane position
        leg_texts = [t.get_text() for t in win.ax_div.get_legend().get_texts()]
        expected_substr = f"Plane x={target_plane:.2f}mm"
        self.assertTrue(any(expected_substr in t for t in leg_texts), f"Expected {expected_substr} in {leg_texts}")

        # Verify get_params includes div_plane_x
        params = win.get_params()
        self.assertIn("div_plane_x", params)
        self.assertAlmostEqual(params["div_plane_x"], target_plane, places=3)
        win.close()

    def test_physics_divergence_at_custom_plane(self):
        sim = DigitalTwinSimulator()
        sim.Lx = 10.0
        sim.Ly = 3.0
        sim.dx = 0.1
        sim.dy = 0.1
        sim.nx = int(sim.Lx / sim.dx) + 1
        sim.ny = int(sim.Ly / sim.dy) + 1
        sim.grid_x_starts = [1.0]
        sim.grid_x_ends = [2.0]
        sim.isBound = np.zeros((sim.ny, sim.nx), dtype=bool)

        custom_plane = 6.0
        sim.div_plane_x = custom_plane

        # Particles crossing custom_plane
        num_p = 10
        sim.p_x = np.full(sim.max_p, 6.05)
        sim.p_y = np.full(sim.max_p, 1.5)
        sim.p_vx = np.full(sim.max_p, 20000.0)
        sim.p_vy = np.full(sim.max_p, 2000.0) # arctan(2000/20000) ~ 5.71 deg
        sim.p_vz = np.zeros(sim.max_p)
        sim.p_isCEX = np.zeros(sim.max_p, dtype=bool)

        ctx = StepContext(
            params={"div_percentile": 95.0, "div_plane_x": custom_plane},
            sim_mode="Thermal",
            t_current=1e-9,
            grids=[{"t": 1.0, "gap": 0.0, "r": 0.8}],
            num_p_step=num_p,
            p_x_old=np.full(num_p, 5.95),
            p_y_old=np.full(num_p, 1.5)
        )

        step_ion_interactions(sim, ctx)

        expected_deg = np.arctan(2000.0 / 20000.0) * 180.0 / np.pi
        self.assertFalse(np.isnan(ctx.current_div_mid))
        self.assertAlmostEqual(ctx.current_div_mid, expected_deg, places=2)


if __name__ == "__main__":
    unittest.main()
