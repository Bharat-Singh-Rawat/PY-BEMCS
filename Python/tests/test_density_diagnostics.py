"""
Unit tests for the Density Diagnostic Window and underlying physics calculations.
"""

import sys
import unittest
import numpy as np

try:
    from PyQt5.QtWidgets import QApplication, QAction
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from physics.simulator import DigitalTwinSimulator
from diagnostics.density_window import DensityWindow
from gui.app.main_window import DigitalTwinApp


class TestDensityDiagnostics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if HAS_PYQT:
            cls.app = QApplication.instance()
            if cls.app is None:
                cls.app = QApplication(sys.argv)

    def setUp(self):
        self.sim = DigitalTwinSimulator()
        self.sim.Lx = 18.0
        self.sim.Ly = 3.0
        self.sim.dx = 0.1
        self.sim.dy = 0.1
        self.sim.nx = int(self.sim.Lx / self.sim.dx) + 1
        self.sim.ny = int(self.sim.Ly / self.sim.dy) + 1
        self.sim.x_coords = np.linspace(0, self.sim.Lx, self.sim.nx)
        self.sim.y_coords = np.linspace(0, self.sim.Ly, self.sim.ny)
        self.sim.X, self.sim.Y = np.meshgrid(self.sim.x_coords, self.sim.y_coords)
        self.sim.isBound = np.zeros((self.sim.ny, self.sim.nx), dtype=bool)
        self.sim.grid_x_starts = [2.0, 5.0]
        self.sim.grid_x_ends = [3.0, 6.0]
        self.sim.div_plane_x = 10.0
        self.sim.V = np.full((self.sim.ny, self.sim.nx), 200.0)
        self.sim.n0_plasma = 1e17
        self.sim.Te_up = 3.0
        self.sim.macro_weight = 1e6

    def test_density_maps_shape_and_empty(self):
        """Verify density maps have correct shape (ny, nx) and zero/positive values."""
        rho_ions = self.sim.get_density_map('ions')
        self.assertEqual(rho_ions.shape, (self.sim.ny, self.sim.nx))
        self.assertTrue(np.all(rho_ions >= 0.0))

        rho_cex = self.sim.get_density_map('cex')
        self.assertEqual(rho_cex.shape, (self.sim.ny, self.sim.nx))
        self.assertTrue(np.all(rho_cex >= 0.0))

        rho_elec = self.sim.get_density_map('electrons')
        self.assertEqual(rho_elec.shape, (self.sim.ny, self.sim.nx))
        self.assertTrue(np.all(rho_elec >= 0.0))

    def test_density_with_mock_particles(self):
        """Verify particle deposition into density map with primary and CEX ions."""
        self.sim.num_p = 10
        self.sim.p_x[:10] = np.linspace(1.0, 10.0, 10)
        self.sim.p_y[:10] = np.full(10, 1.5)
        self.sim.p_isCEX[:10] = np.array([False, True, False, True, False, False, True, False, False, False])

        dens_all = self.sim.get_density_map('ions')
        dens_cex = self.sim.get_density_map('cex')
        dens_prim = self.sim.get_density_map('primary')

        self.assertGreater(np.max(dens_all), 0.0)
        self.assertGreater(np.max(dens_cex), 0.0)
        self.assertGreater(np.max(dens_prim), 0.0)

        # CEX and Primary should sum up to total ions
        np.testing.assert_allclose(dens_all, dens_cex + dens_prim, rtol=1e-5, atol=1e-8)

    def test_divergence_planes_and_radial_profiles(self):
        """Verify divergence planes retrieval and radial profile 1D slices."""
        x1, x2 = self.sim.get_divergence_plane_locations()
        self.assertGreater(x1, 0.0)
        self.assertLessEqual(x1, self.sim.Lx)
        self.assertGreaterEqual(x2, x1)
        self.assertLessEqual(x2, self.sim.Lx)

        # Add particles so there is signal
        self.sim.num_p = 20
        self.sim.p_x[:20] = np.linspace(0.5, 15.0, 20)
        self.sim.p_y[:20] = np.linspace(0.2, 2.8, 20)
        self.sim.p_isCEX[:20] = False

        y_pts, prof_p1 = self.sim.get_radial_density_profile(x1, species='ions')
        _, prof_p2 = self.sim.get_radial_density_profile(x2, species='ions')

        self.assertEqual(len(y_pts), self.sim.ny)
        self.assertEqual(len(prof_p1), self.sim.ny)
        self.assertEqual(len(prof_p2), self.sim.ny)
        self.assertTrue(np.all(prof_p1 >= 0.0))
        self.assertTrue(np.all(prof_p2 >= 0.0))

    @unittest.skipUnless(HAS_PYQT, "PyQt5 not available")
    def test_density_window_ui_and_update(self):
        """Verify DensityWindow initializes and updates without exceptions."""
        win = DensityWindow()
        self.assertIsNotNone(win)
        self.assertEqual(win.combo_species.count(), 4)

        # Test instantaneous and time-averaged update
        win.update_plot(self.sim)
        self.assertIn("Peak", win.lbl_stats.text())

        # Test species change
        win.combo_species.setCurrentIndex(2)  # CEX
        win.update_plot(self.sim)

        # Test overlay all species
        win.chk_overlay_all.setChecked(True)
        win.update_plot(self.sim)

        win.close()

    @unittest.skipUnless(HAS_PYQT, "PyQt5 not available")
    def test_menu_density_action_position_and_open(self):
        """Verify that Density action is in Diagnostics menu directly below PPC and opens DensityWindow."""
        app_win = DigitalTwinApp()

        # Find Diagnostics menu
        menubar = app_win.menuBar()
        diag_action = None
        for action in menubar.actions():
            if action.text() == "Diagnostics":
                diag_action = action
                break

        self.assertIsNotNone(diag_action, "Diagnostics menu action not found in menubar")
        diag_menu = diag_action.menu()
        self.assertIsNotNone(diag_menu, "Diagnostics QMenu is None")

        actions = diag_menu.actions()
        action_texts = [a.text() for a in actions]

        self.assertIn("PPC Distribution...", action_texts)
        self.assertIn("Density...", action_texts)

        ppc_idx = action_texts.index("PPC Distribution...")
        density_idx = action_texts.index("Density...")
        self.assertEqual(density_idx, ppc_idx + 1, "Density action must be directly under PPC Distribution")

        # Test triggering Density action
        density_action = actions[density_idx]
        density_action.trigger()

        self.assertIsNotNone(app_win.density_window)
        self.assertTrue(app_win.density_window.isVisible())

        # Close opened windows
        app_win.density_window.close()
        app_win.close()

    @unittest.skipUnless(HAS_PYQT, "PyQt5 not available")
    def test_sweep_mesh_resize_resets_accumulator_cleanly(self):
        """Verify that when mesh dimensions change during sweep scenarios, the accumulator resets without broadcast error."""
        win = DensityWindow()

        # Step 1: Initial mesh (93, 104)
        self.sim.ny, self.sim.nx = 93, 104
        self.sim.x_coords = np.linspace(0, self.sim.Lx, self.sim.nx)
        self.sim.y_coords = np.linspace(0, self.sim.Ly, self.sim.ny)
        self.sim.X, self.sim.Y = np.meshgrid(self.sim.x_coords, self.sim.y_coords)
        self.sim.isBound = np.zeros((self.sim.ny, self.sim.nx), dtype=bool)
        self.sim.V = np.full((self.sim.ny, self.sim.nx), 200.0)
        self.sim.iteration = 10

        win.update_plot(self.sim, is_new_step=True)
        self.assertEqual(win._last_mesh_shape, (93, 104))
        self.assertIn("ions", win._accum_maps)
        self.assertEqual(win._accum_maps["ions"].shape, (93, 104))

        # Step 2: Mesh resize during sweep (231, 227)
        self.sim.ny, self.sim.nx = 231, 227
        self.sim.x_coords = np.linspace(0, self.sim.Lx, self.sim.nx)
        self.sim.y_coords = np.linspace(0, self.sim.Ly, self.sim.ny)
        self.sim.X, self.sim.Y = np.meshgrid(self.sim.x_coords, self.sim.y_coords)
        self.sim.isBound = np.zeros((self.sim.ny, self.sim.nx), dtype=bool)
        self.sim.V = np.full((self.sim.ny, self.sim.nx), 200.0)
        self.sim.iteration = 0  # iteration resets at scenario start

        # This previously raised ValueError: operands could not be broadcast together with shapes (93,104) (231,227) (93,104)
        win.update_plot(self.sim, is_new_step=True)
        self.assertEqual(win._last_mesh_shape, (231, 227))
        self.assertEqual(win._accum_maps["ions"].shape, (231, 227))

        win.close()


if __name__ == '__main__':
    unittest.main()
