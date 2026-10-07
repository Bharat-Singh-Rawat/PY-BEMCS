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
from physics_engine import DigitalTwinSimulator
from gui.dialogs import AdvancedSettingsDialog

app = QApplication.instance()
if app is None:
    app = QApplication([""])


class TestMeshSizeSetting(unittest.TestCase):
    def test_advanced_dialog_default_and_modify(self):
        # 1. Default should be 0.8
        dialog = AdvancedSettingsDialog({})
        self.assertIn("dx_over_debye", dialog.inputs)
        spin = dialog.inputs["dx_over_debye"]
        self.assertAlmostEqual(spin.value(), 0.8, places=3)
        
        vals = dialog.get_values()
        self.assertIn(r"\deltax/debye_length", vals)
        self.assertAlmostEqual(vals[r"\deltax/debye_length"], 0.8, places=3)

        # 2. Modify to 0.5
        spin.setValue(0.5)
        vals2 = dialog.get_values()
        self.assertAlmostEqual(vals2[r"\deltax/debye_length"], 0.5, places=3)
        self.assertAlmostEqual(vals2["dx_over_debye"], 0.5, places=3)

    def test_domain_mesh_scaling(self):
        sim = DigitalTwinSimulator()
        base_params = {
            "n0_plasma": 1e17,
            "Te_up": 3.0,
            "Lx": 5.0,
            "Ly": 2.0,
            "grids": [
                {"V": 1100.0, "t": 0.38, "gap": 0.58, "r": 0.95, "cham": 0.0},
                {"V": -180.0, "t": 0.51, "gap": 3.0, "r": 0.57, "cham": 0.0},
            ]
        }

        # Build with ratio 0.8
        params_08 = dict(base_params)
        params_08[r"\deltax/debye_length"] = 0.8
        sim.build_domain(params_08)
        dx_08 = sim.dx
        dy_08 = sim.dy
        nx_08 = sim.nx
        ny_08 = sim.ny

        # Build with ratio 0.4 (finer mesh: should halve cell size, roughly double cell count)
        params_04 = dict(base_params)
        params_04[r"\deltax/debye_length"] = 0.4
        sim.build_domain(params_04)
        dx_04 = sim.dx
        dy_04 = sim.dy
        nx_04 = sim.nx
        ny_04 = sim.ny

        self.assertAlmostEqual(dx_04, dx_08 * 0.5, places=5)
        self.assertAlmostEqual(dy_04, dy_08 * 0.5, places=5)
        self.assertGreater(nx_04, nx_08)
        self.assertGreater(ny_04, ny_08)

    def test_coarse_mesh_override_no_crash(self):
        sim = DigitalTwinSimulator()
        params_coarse = {
            "n0_plasma": 1e17,
            "Te_up": 3.0,
            "Lx": 5.0,
            "Ly": 2.0,
            r"\deltax/debye_length": 1.5,
            "grids": [
                {"V": 1100.0, "t": 0.38, "gap": 0.58, "r": 0.95, "cham": 0.0},
            ]
        }
        # Should NOT raise ValueError
        sim.build_domain(params_coarse)
        self.assertGreater(sim.dx, 0.0)


if __name__ == "__main__":
    unittest.main()
