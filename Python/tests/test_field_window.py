import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication

from physics.simulator import DigitalTwinSimulator
from diagnostics.field_window import FieldDiagnosticWindow
from gui.app.main_window import DigitalTwinApp


class TestFieldDiagnosticWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self):
        self.sim = DigitalTwinSimulator()
        self.sim.Lx = 5.0
        self.sim.Ly = 2.0
        self.sim.dx = 0.1
        self.sim.dy = 0.1
        self.sim.nx = int(self.sim.Lx / self.sim.dx) + 1
        self.sim.ny = int(self.sim.Ly / self.sim.dy) + 1
        x = np.linspace(0, self.sim.Lx, self.sim.nx)
        y = np.linspace(0, self.sim.Ly, self.sim.ny)
        self.sim.X, self.sim.Y = np.meshgrid(x, y)
        self.sim.V = np.full((self.sim.ny, self.sim.nx), 500.0)
        self.sim.Vdc = np.full((self.sim.ny, self.sim.nx), 450.0)
        self.sim.Ex = np.full((self.sim.ny, self.sim.nx), 1e5)
        self.sim.Ey = np.full((self.sim.ny, self.sim.nx), 1e4)
        self.sim.isBound = np.zeros((self.sim.ny, self.sim.nx), dtype=bool)
        self.sim.grids = [{"V": 1000.0, "t": 0.5, "gap": 1.0, "r": 0.8}]

    def test_window_modes_and_rendering(self):
        dlg = FieldDiagnosticWindow()
        dlg.update_plot(self.sim)

        for mode in FieldDiagnosticWindow.MODES:
            idx = dlg.combo_mode.findText(mode)
            self.assertGreaterEqual(idx, 0, f"Mode '{mode}' not found in combo_mode")
            dlg.combo_mode.setCurrentIndex(idx)
            # Replot should run cleanly without error
            dlg._replot()

        dlg.close()

    def test_unit_switching(self):
        dlg = FieldDiagnosticWindow()
        dlg.update_plot(self.sim)

        for unit in FieldDiagnosticWindow.UNITS_E:
            idx = dlg.combo_unit.findText(unit)
            self.assertGreaterEqual(idx, 0)
            dlg.combo_unit.setCurrentIndex(idx)
            dlg._replot()

        dlg.close()

    def test_gui_menu_action_opens_field_window(self):
        win = DigitalTwinApp()
        self.assertIsNone(win.field_window)

        # Trigger open_field_window
        win.open_field_window()
        self.assertIsNotNone(win.field_window)
        self.assertTrue(win.field_window.isVisible())

        # Call update_plot via field_window
        win.field_window.update_plot(win.sim)

        win.field_window.close()
        win.close()

    def test_streamplot_nonuniform_grid(self):
        # Create non-uniform grid matching realistic zone adaptive mesh
        sim_nonuni = DigitalTwinSimulator()
        sim_nonuni.build_domain({})
        dlg = FieldDiagnosticWindow()
        dlg.update_plot(sim_nonuni)

        idx = dlg.combo_mode.findText("E Field Streamlines (Flux Lines)")
        self.assertGreaterEqual(idx, 0)
        dlg.combo_mode.setCurrentIndex(idx)
        dlg._replot()

        # Ensure no error text is drawn on the plot
        texts = [t.get_text() for t in dlg.ax.texts]
        for t in texts:
            self.assertNotIn("Streamplot error", t)

        dlg.close()


if __name__ == "__main__":
    unittest.main()
