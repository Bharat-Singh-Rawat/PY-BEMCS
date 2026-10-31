import os
import sys
import unittest
import numpy as np

os.environ["PYBEMCS_FORCE_CPU"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt5.QtWidgets import QApplication
from physics_engine import (
    DigitalTwinSimulator,
    compute_efield_constrained_spacing,
    evaluate_mesh_efield_variation,
)
from gui.dialogs import AdvancedSettingsDialog

app = QApplication.instance()
if app is None:
    app = QApplication([""])


class TestEfieldMeshControl(unittest.TestCase):
    def test_advanced_dialog_efield_controls(self):
        # 1. Defaults
        dialog = AdvancedSettingsDialog({})
        self.assertTrue(dialog.chk_efield_limit.isChecked())
        self.assertAlmostEqual(dialog.spin_max_efield_jump.value(), 10.0, places=1)

        vals = dialog.get_values()
        self.assertIn("enforce_efield_mesh_limit", vals)
        self.assertTrue(vals["enforce_efield_mesh_limit"])
        self.assertIn("max_efield_jump", vals)
        self.assertAlmostEqual(vals["max_efield_jump"], 0.10, places=3)

        # 2. Modify to 15% and uncheck
        dialog.chk_efield_limit.setChecked(False)
        dialog.spin_max_efield_jump.setValue(15.0)
        vals2 = dialog.get_values()
        self.assertFalse(vals2["enforce_efield_mesh_limit"])
        self.assertAlmostEqual(vals2["max_efield_jump"], 0.15, places=3)

    def test_compute_efield_constrained_spacing(self):
        grids = [
            {"V": 1100.0, "t": 0.38, "gap": 0.58, "r": 0.95, "cham": 0.0},
            {"V": -180.0, "t": 0.51, "gap": 3.0, "r": 0.57, "cham": 0.0},
        ]
        Te_up = 4.3

        # At high density, Debye length is tiny (~0.02 mm), so Debye scale governs
        dx_high, dy_high = compute_efield_constrained_spacing(
            grids=grids, n0=7e17, Te_up=Te_up, ratio_debye=0.8, max_rel_jump=0.10
        )
        self.assertLess(dx_high, 0.03)
        self.assertLess(dy_high, 0.03)

        # At low density, Debye length explodes (~0.62 mm), but E-field/gap geometry bounds it
        dx_low, dy_low = compute_efield_constrained_spacing(
            grids=grids, n0=1e15, Te_up=Te_up, ratio_debye=0.8, max_rel_jump=0.10
        )
        # Without constraint, 0.8 * lambda_D would be ~0.50 mm.
        # With constraint, dx must be <= 0.10 * 0.58 = 0.058 mm
        self.assertLessEqual(dx_low, 0.058)
        self.assertLessEqual(dy_low, 0.057)

    def test_low_density_domain_stability_and_jump_bound(self):
        sim = DigitalTwinSimulator()
        params = {
            "n0_plasma": 1e15,
            "Te_up": 4.3,
            "Ti": 0.75,
            "Lx": 5.0,
            "Ly": 1.6,
            "grids": [
                {"V": 1100.0, "t": 0.38, "gap": 0.58, "r": 0.95, "cham": 0.0},
                {"V": -180.0, "t": 0.51, "gap": 3.0, "r": 0.57, "cham": 0.0},
            ],
            "dx_over_debye": 0.8,
            "enforce_efield_mesh_limit": True,
            "max_efield_jump": 0.10,
        }

        # Build domain - should succeed without singular matrix
        sim.build_domain(params)

        # Grid dimensions should be well-formed
        self.assertGreaterEqual(sim.ny, 12)
        self.assertGreaterEqual(sim.nx, 30)

        # Electric field jumps must satisfy <= 10% tolerance
        self.assertTrue(sim.last_efield_jump_compliant)
        self.assertLessEqual(sim.last_max_efield_jump_x, 0.105)
        self.assertLessEqual(sim.last_max_efield_jump_y, 0.105)

    def test_low_density_pic_step_execution(self):
        sim = DigitalTwinSimulator()
        params = {
            "n0_plasma": 1e15,
            "Te_up": 4.3,
            "Ti": 0.75,
            "Lx": 5.0,
            "Ly": 1.6,
            "grids": [
                {"V": 1100.0, "t": 0.38, "gap": 0.58, "r": 0.95, "cham": 0.0},
                {"V": -180.0, "t": 0.51, "gap": 3.0, "r": 0.57, "cham": 0.0},
            ],
            "dx_over_debye": 0.8,
            "enforce_efield_mesh_limit": True,
            "max_efield_jump": 0.10,
            "target_ppc": 20,
        }
        sim.build_domain(params)

        # Run 5 PIC steps
        for _ in range(5):
            sim.step(params)

        # Confirm particle storage and metrics remain healthy
        self.assertTrue(np.all(np.isfinite(sim.Ex)))
        self.assertTrue(np.all(np.isfinite(sim.Ey)))
        self.assertTrue(np.all(np.isfinite(sim.V)))
        if sim.num_p > 0:
            self.assertTrue(np.all(np.isfinite(sim.p_x[:sim.num_p])))
            self.assertTrue(np.all(np.isfinite(sim.p_vx[:sim.num_p])))


if __name__ == "__main__":
    unittest.main()
