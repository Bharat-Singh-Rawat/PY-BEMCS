"""Tests for multi-grid erosion profile subplots in Row 2 of the main window."""
import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from gui.app.main_window import DigitalTwinApp


class TestMultiGridErosionPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_groove_axes_match_grid_count_on_add_remove_and_config(self):
        win = DigitalTwinApp()
        try:
            n_init = len(win.grid_widgets)
            self.assertEqual(len(win.ax_grooves), n_init)
            self.assertEqual(len(win.line_grooves), n_init)
            # 5 non-groove axes (live, temp, dmg, ebs, div) + n_init groove axes
            self.assertEqual(len(win.fig.axes), 5 + n_init)

            # Remove down to 2 grids -> 1x2 layout
            while len(win.grid_widgets) > 2:
                win.remove_grid_ui()
            self.assertEqual(len(win.ax_grooves), 2)
            self.assertEqual(len(win.line_grooves), 2)
            self.assertEqual(len(win.fig.axes), 5 + 2)
            self.assertEqual(win.ax_grooves[0].get_subplotspec().get_gridspec().nrows, 1)
            self.assertEqual(win.ax_grooves[0].get_subplotspec().get_gridspec().ncols, 2)

            # Remove down to 1 grid -> 1x1 layout
            win.remove_grid_ui()
            self.assertEqual(len(win.ax_grooves), 1)
            self.assertEqual(len(win.line_grooves), 1)
            self.assertEqual(len(win.fig.axes), 5 + 1)
            self.assertEqual(win.ax_grooves[0].get_subplotspec().get_gridspec().nrows, 1)
            self.assertEqual(win.ax_grooves[0].get_subplotspec().get_gridspec().ncols, 1)

            # Add up to 4 grids -> 2x2 layout
            for _ in range(3):
                win.add_grid_ui(0.0, 0.38, 1.0, 0.70, 0.0)
            self.assertEqual(len(win.grid_widgets), 4)
            self.assertEqual(len(win.ax_grooves), 4)
            self.assertEqual(len(win.line_grooves), 4)
            self.assertEqual(len(win.fig.axes), 5 + 4)
            self.assertEqual(win.ax_grooves[0].get_subplotspec().get_gridspec().nrows, 2)
            self.assertEqual(win.ax_grooves[0].get_subplotspec().get_gridspec().ncols, 2)
            # Verify vertical clearance between row 0 and row 1
            gap = win.ax_grooves[0].get_position().y0 - win.ax_grooves[2].get_position().y1
            self.assertGreater(gap, 0.04, f"Expected gap > 0.04 to prevent overlap, got {gap:.3f}")

            # apply_config with 2 grids should rebuild to 2 groove axes
            cfg = win.get_current_config_dict()
            cfg["grids"] = cfg["grids"][:2]
            win.apply_config(cfg, config_name="test_2grids.json")
            self.assertEqual(len(win.grid_widgets), 2)
            self.assertEqual(len(win.ax_grooves), 2)
            self.assertEqual(len(win.line_grooves), 2)
            self.assertEqual(len(win.fig.axes), 5 + 2)

            # Verify that Rows 0 & 1 (ax_live, ax_div) positions are identical between 2 and 3 grids
            pos2_live = win.ax_live.get_position()
            pos2_div = win.ax_div.get_position()
            win.add_grid_ui(0.0, 0.38, 1.0, 0.70, 0.0)
            self.assertEqual(len(win.grid_widgets), 3)
            pos3_live = win.ax_live.get_position()
            pos3_div = win.ax_div.get_position()
            self.assertAlmostEqual(pos2_live.y0, pos3_live.y0, delta=0.005)
            self.assertAlmostEqual(pos2_live.height, pos3_live.height, delta=0.005)
            self.assertAlmostEqual(pos2_div.y0, pos3_div.y0, delta=0.005)
            self.assertAlmostEqual(pos2_div.height, pos3_div.height, delta=0.005)
        finally:
            win.close()

    def test_draw_static_domain_and_process_snap_update_all_grooves(self):
        win = DigitalTwinApp()
        try:
            win.build_domain(force=True)
            n_grids = len(win.sim.mask_grids)
            self.assertGreaterEqual(n_grids, 2)
            self.assertEqual(len(win.ax_grooves), n_grids)
            for i, ax in enumerate(win.ax_grooves):
                self.assertIn(f"Grid {i + 1} Erosion Profile", ax.get_title())

            # Simulate non-zero erosion depth on each grid and verify _process_snap plots all of them
            for idx, mg in enumerate(win.sim.mask_grids):
                ys, xs = np.where(mg)
                if ys.size > 0:
                    win.sim.eroded_depth[ys[0], xs[-1]] = (idx + 1) * 1e-3  # mm -> (idx+1) um

            captured = []
            win._sim_worker.step_done.connect(captured.append)
            win._sim_worker._emit_snapshot(
                win.get_params(), win._sim_run_id, False, 0.0, 10.0, [299.15] * n_grids, 0.9
            )
            self.app.processEvents()
            self.assertEqual(len(captured), 1)
            win._process_snap(captured[0])

            for i in range(n_grids):
                x_data, y_data = win.line_grooves[i].get_data()
                self.assertGreater(len(x_data), 0)
                self.assertGreater(len(y_data), 0)
                self.assertIn(f"Grid {i + 1} Erosion Profile — downstream face", win.ax_grooves[i].get_title())
                self.assertGreaterEqual(float(np.max(y_data)), float(i + 1) - 1e-6)
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
