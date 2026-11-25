"""Tests for the saddle-point potential (ax_ebs) scrollbar and its independence from ax_div scrollbar."""
import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from gui.app.main_window import DigitalTwinApp


class TestEBSScrollbar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_ebs_and_div_scrollbars_exist_and_position(self):
        win = DigitalTwinApp()
        try:
            self.assertTrue(hasattr(win, "ebs_scrollbar"))
            self.assertTrue(hasattr(win, "btn_ebs_live"))
            self.assertTrue(hasattr(win, "div_scrollbar"))
            self.assertTrue(hasattr(win, "btn_div_live"))

            win.show()
            self.app.processEvents()

            # Verify geometry under ax_ebs and ax_div
            ebs_geom = win.ebs_scrollbar.geometry()
            div_geom = win.div_scrollbar.geometry()

            # ebs is in col 1, div is in col 2 -> ebs x must be strictly to the left of div x
            self.assertLess(ebs_geom.x() + ebs_geom.width(), div_geom.x())
            # both scrollbars should have identical height and matching style
            self.assertEqual(win.ebs_scrollbar.height(), win.div_scrollbar.height())
            self.assertEqual(win.btn_ebs_live.width(), win.btn_div_live.width())
        finally:
            win.close()

    def test_scrollbars_operate_independently(self):
        win = DigitalTwinApp()
        try:
            # Populate 1000 points of synthetic data
            win.iter_history = list(range(1, 1001))
            win.ebs_history = [float(v) for v in range(1, 1001)]
            win.div_history = [float(v * 0.1) for v in range(1, 1001)]
            win.time_history = [v * 5e-10 for v in range(1, 1001)]

            win._update_ebs_scroll_range(1000)
            win._update_div_scroll_range(1000)

            # Both should start in live mode at maximum
            self.assertTrue(getattr(win, "_ebs_live_mode", False))
            self.assertTrue(getattr(win, "_div_live_mode", False))
            self.assertEqual(win.ebs_scrollbar.maximum(), 600)  # 1000 - 400 window
            self.assertEqual(win.div_scrollbar.maximum(), 600)
            self.assertEqual(win.ebs_scrollbar.value(), 600)
            self.assertEqual(win.div_scrollbar.value(), 600)

            div_xlim_before = win.ax_div.get_xlim()
            ebs_xlim_before = win.ax_ebs.get_xlim()

            # 1. Scroll ebs_scrollbar back to 100
            win.ebs_scrollbar.setValue(100)
            self.assertFalse(win._ebs_live_mode)
            self.assertEqual(win.btn_ebs_live.text(), "▶ Live")
            # ax_ebs must have shifted to [100, 500] (0-based indices)
            ebs_xlim_after = win.ax_ebs.get_xlim()
            self.assertAlmostEqual(ebs_xlim_after[0], 101, delta=2)
            self.assertAlmostEqual(ebs_xlim_after[1], 501, delta=2)

            # ax_div must remain COMPLETELY UNTOUCHED and still in live mode!
            self.assertTrue(win._div_live_mode)
            self.assertEqual(win.btn_div_live.text(), "📌 Live")
            self.assertEqual(win.div_scrollbar.value(), 600)
            self.assertEqual(win.ax_div.get_xlim(), div_xlim_before)

            # 2. Scroll div_scrollbar back to 250
            win.div_scrollbar.setValue(250)
            self.assertFalse(win._div_live_mode)
            self.assertEqual(win.btn_div_live.text(), "▶ Live")
            div_xlim_after = win.ax_div.get_xlim()
            self.assertAlmostEqual(div_xlim_after[0], 251, delta=2)
            self.assertAlmostEqual(div_xlim_after[1], 651, delta=2)

            # ax_ebs must remain at its own scrolled position [101, 501]!
            self.assertEqual(win.ax_ebs.get_xlim(), ebs_xlim_after)

            # 3. Toggle ebs live button -> ebs returns to live [600..1000]
            win.btn_ebs_live.setChecked(True)
            self.assertTrue(win._ebs_live_mode)
            self.assertEqual(win.btn_ebs_live.text(), "📌 Live")
            self.assertEqual(win.ebs_scrollbar.value(), 600)
            self.assertEqual(win.ax_ebs.get_xlim(), ebs_xlim_before)

            # div must still be in history mode at 250!
            self.assertFalse(win._div_live_mode)
            self.assertEqual(win.div_scrollbar.value(), 250)
            self.assertEqual(win.ax_div.get_xlim(), div_xlim_after)

            # 4. Toggle div live button -> div returns to live
            win.btn_div_live.setChecked(True)
            self.assertTrue(win._div_live_mode)
            self.assertEqual(win.btn_div_live.text(), "📌 Live")
            self.assertEqual(win.div_scrollbar.value(), 600)
            self.assertEqual(win.ax_div.get_xlim(), div_xlim_before)
        finally:
            win.close()

    def test_particles_remain_visible_during_scroll(self):
        win = DigitalTwinApp()
        try:
            win.build_domain(force=True)
            win.show()
            self.app.processEvents()

            # Set synthetic particles on scat_prim
            test_offsets = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 0.8]])
            win.scat_prim.set_offsets(test_offsets)
            win.scat_prim.set_array(np.array([100.0, 200.0, 300.0]))

            # Populate history and update ranges
            win.iter_history = list(range(1, 601))
            win.ebs_history = [float(v) for v in range(1, 601)]
            win.div_history = [float(v * 0.1) for v in range(1, 601)]
            win._update_ebs_scroll_range(600)

            # Trigger scroll on ebs_scrollbar
            win._on_ebs_scroll(50)
            self.app.processEvents()

            # Verify scat_prim still has its offsets
            self.assertEqual(len(win.scat_prim.get_offsets()), 3)

            # Trigger scroll on div_scrollbar
            win._on_div_scroll(50)
            self.app.processEvents()
            self.assertEqual(len(win.scat_prim.get_offsets()), 3)
        finally:
            win.close()

    def test_splitter_resize_and_horizontal_scrollbar(self):
        win = DigitalTwinApp()
        try:
            win.show()
            self.app.processEvents()

            splitter = win.content_splitter
            scroll_area = win.control_scroll_area

            # Initial state: width ~330, horizontal scrollbar not needed
            self.assertFalse(scroll_area.horizontalScrollBar().isVisible())

            # Shrink left side menu to 200px
            total_w = splitter.width()
            splitter.setSizes([200, total_w - 200])
            self.app.processEvents()

            self.assertLess(scroll_area.width(), 300)
            # Horizontal downstream scrollbar must appear
            self.assertTrue(scroll_area.horizontalScrollBar().isVisible())

            # Expand left side menu back to 400px
            splitter.setSizes([400, total_w - 400])
            self.app.processEvents()

            self.assertGreaterEqual(scroll_area.width(), 320)
            # Horizontal scrollbar should hide when menu is wide enough
            self.assertFalse(scroll_area.horizontalScrollBar().isVisible())
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
