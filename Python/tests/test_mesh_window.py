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
from diagnostics.mesh_window import MeshWindow
from main import DigitalTwinApp

app = QApplication.instance()
if app is None:
    app = QApplication([""])


class TestMeshWindow(unittest.TestCase):
    def setUp(self):
        self.sim = DigitalTwinSimulator()
        self.params = {
            "n0_plasma": 1e17,
            "Te_up": 3.0,
            "Lx": 6.0,
            "Ly": 2.5,
            r"\deltax/debye_length": 0.8,
            "grids": [
                {"V": 1100.0, "t": 0.38, "gap": 0.58, "r": 0.95, "cham": 0.0},
                {"V": -180.0, "t": 0.51, "gap": 3.0, "r": 0.57, "cham": 0.0},
            ]
        }

    def test_unbuilt_domain_graceful(self):
        win = MeshWindow()
        win.update_plot(None, params=self.params)
        self.assertIn("—", win.lbl_dx_nominal.text())
        win.close()

    def test_built_domain_rendering_and_metrics(self):
        self.sim.build_domain(self.params)
        win = MeshWindow()
        win.update_plot(self.sim, params=self.params)

        # Check Δx labels
        self.assertIn(f"{self.sim.dx:.4f}", win.lbl_dx_nominal.text())
        self.assertIn(f"{self.sim.dy:.4f}", win.lbl_dy_nominal.text())
        self.assertIn("Δx / λ_D ratio", win.lbl_ratio.text())
        self.assertIn("0.800", win.lbl_ratio.text())

        # Check Debye length label
        self.assertIn("λ_D (unscreened):", win.lbl_debye_std.text())
        self.assertIn("λ_D (Bohm presheath):", win.lbl_debye_bohm.text())

        # Check mesh sizes and counts
        self.assertIn(f"{self.sim.nx} × {self.sim.ny}", win.lbl_grid_nodes.text())
        self.assertIn(f"{(self.sim.nx - 1) * (self.sim.ny - 1):,}", win.lbl_grid_cells.text())
        self.assertIn(f"{self.sim.Lx:.3f} × {self.sim.Ly:.3f}", win.lbl_domain_size.text())

        # Test toggling controls and replot
        win.chk_mesh.setChecked(False)
        win.chk_potential.setChecked(False)
        win.chk_electrodes.setChecked(False)
        win.chk_zones.setChecked(False)
        win.btn_refresh.click()

        win.close()

    def test_substep_graph_widget_and_window(self):
        from diagnostics.mesh_window import DisplacementSubstepsWindow, SubstepGraphWidget
        self.sim.build_domain(self.params)
        self.sim.substep_history_iter = [1, 2, 3, 4]
        self.sim.substep_history_ion  = [1, 1, 2, 1]
        self.sim.substep_history_e    = [1, 4, 8, 2]

        sub_win = DisplacementSubstepsWindow()
        sub_win.update_plot(self.sim)

        widget = sub_win.substep_widget
        self.assertIn("4", widget.lbl_curr_iter.text())
        self.assertIn("1", widget.lbl_curr_ion.text())
        self.assertIn("2", widget.lbl_curr_elec.text())
        self.assertIn("Ion Violations:", widget.lbl_viol_ion.text())
        self.assertIn("Electron Violations:", widget.lbl_viol_elec.text())

        # Test filter violations toggle
        widget.chk_violations.setChecked(True)
        widget.chk_violations.setChecked(False)

        sub_win.close()

    def test_mesh_window_second_window_popup(self):
        win = MeshWindow()
        self.assertIsNone(win._substeps_dialog)
        win.btn_open_substeps.click()
        self.assertIsNotNone(win._substeps_dialog)
        self.assertTrue(win._substeps_dialog.isVisible())
        win._substeps_dialog.close()
        win.close()

    def test_gui_integration_open_mesh_window(self):
        gui = DigitalTwinApp()
        self.assertIsNone(gui.mesh_window)
        gui.open_mesh_window()
        self.assertIsNotNone(gui.mesh_window)
        self.assertTrue(gui.mesh_window.isVisible())
        gui.mesh_window.close()
        gui.close()

    def test_gui_integration_open_substep_window(self):
        gui = DigitalTwinApp()
        self.assertIsNone(gui.substep_window)
        gui.open_substep_window()
        self.assertIsNotNone(gui.substep_window)
        self.assertTrue(gui.substep_window.isVisible())
        gui.substep_window.close()
        gui.close()


if __name__ == "__main__":
    unittest.main()

