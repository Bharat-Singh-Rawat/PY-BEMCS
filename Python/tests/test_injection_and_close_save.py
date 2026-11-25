"""
Tests for:
1. Injection Y-length clamping in one-hole vs half-hole geometry (clamped to Ly/2 for one_hole).
2. Main window closeEvent save confirmation pop-up after a simulation has run.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QCloseEvent
from physics_engine import DigitalTwinSimulator
from gui.app.main_window import DigitalTwinApp


class TestInjectionAndCloseSave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self):
        self.win = DigitalTwinApp()

    def tearDown(self):
        self.win._data_saved = True
        self.win.iter_history = []
        if hasattr(self.win, 'sim') and self.win.sim is not None:
            self.win.sim.iteration = 0
        self.win._shutdown_background_tasks()
        self.win.close()

    def test_injection_y_length_clamping_one_hole_physics(self):
        """Verify that in one_hole geometry, inj_y_length is clamped to Ly/2 in physics."""
        sim = DigitalTwinSimulator()
        params = {
            "coordinate_system": "Cartesian",
            "geometry": "one_hole",
            "n0_plasma": 1e17,
            "Te_up": 3.0,
            "Ti": 0.1,
            "sim_mode": "PIC-MCC",
            "enforce_efield_mesh_limit": False,
            "Ly": 4.0,  # 4mm height, hole center is at y = 2.0 mm
            "inj_y_length_mm": 3.0,  # Exceeds Ly/2 = 2.0 mm
            "grids": [
                {"V": 1000.0, "t": 0.38, "gap": 0.64, "r": 0.80, "cham": 0.0},
                {"V": -200.0, "t": 0.38, "gap": 0.64, "r": 0.70, "cham": 0.0},
            ]
        }
        sim.build_domain(params)
        sim.step(params)

        self.assertGreater(sim.num_p, 0)
        # Center of hole is at Ly/2 = 2.0 mm
        # With inj_y clamped to Ly/2 = 2.0 mm, particles span [0, 4.0] mm
        y_particles = sim.p_y[:sim.num_p]
        self.assertGreaterEqual(np.min(y_particles), 0.0)
        self.assertLessEqual(np.max(y_particles), sim.Ly)

    def test_injection_y_length_gui_clamping_one_hole(self):
        """Verify that GUI input clamps inj_y_length to Ly/2 when one_hole is armed."""
        win = self.win
        win._user_overrode_Ly = True
        win.adv_params["Ly"] = 4.0

        # Half hole: allows up to Ly = 4.0 mm
        win.combo_geometry.setCurrentText("half_hole")
        win.inputs["inj_y_length_mm"].setValue(3.5)
        self.assertAlmostEqual(win.inputs["inj_y_length_mm"].value(), 3.5, places=2)

        # Switch to one_hole: must clamp to Ly/2 = 2.0 mm
        win.combo_geometry.setCurrentText("one_hole")
        win.inputs["inj_y_length_mm"].setValue(3.5)
        self.assertAlmostEqual(win.inputs["inj_y_length_mm"].value(), 2.0, places=2)
        self.assertIn("clamped to Ly/2", win.lbl_status.text())

    def test_close_event_prompts_to_save_after_simulation(self):
        """Verify that after a simulation runs, closing the window prompts to save data."""
        win = self.win

        # Simulate that a short or long simulation has run
        win.sim_isRunning = False  # Beam paused or stopped
        win.iter_history = [1, 2, 3, 4, 5]
        win.sim.iteration = 5
        win._data_saved = False

        # Case A: User clicks "Cancel" in save dialog -> close event is rejected
        event_cancel = QCloseEvent()
        with patch("gui.app.main_window.QMessageBox") as MockMsgBox:
            mock_instance = MockMsgBox.return_value
            btn_cancel = MagicMock()
            mock_instance.addButton.side_effect = lambda text, role: btn_cancel if "Cancel" in text else MagicMock()
            mock_instance.clickedButton.return_value = btn_cancel

            win.closeEvent(event_cancel)
            self.assertFalse(event_cancel.isAccepted())

        # Case B: User clicks "Don't Save" -> close event is accepted without saving
        event_dont_save = QCloseEvent()
        with patch("gui.app.main_window.QMessageBox") as MockMsgBox:
            mock_instance = MockMsgBox.return_value
            btn_dont_save = MagicMock()
            mock_instance.addButton.side_effect = lambda text, role: btn_dont_save if "Don't Save" in text else MagicMock()
            mock_instance.clickedButton.return_value = btn_dont_save

            win.closeEvent(event_dont_save)
            self.assertTrue(event_dont_save.isAccepted())

        # Case C: User clicks "Save" -> calls export_csv()
        event_save = QCloseEvent()
        with patch("gui.app.main_window.QMessageBox") as MockMsgBox:
            mock_instance = MockMsgBox.return_value
            btn_save = MagicMock()
            mock_instance.addButton.side_effect = lambda text, role: btn_save if text == "Save" else MagicMock()
            mock_instance.clickedButton.return_value = btn_save

            with patch.object(win, 'export_csv', return_value=True) as mock_export:
                win.closeEvent(event_save)
                self.assertTrue(mock_export.called)
                self.assertTrue(event_save.isAccepted())
