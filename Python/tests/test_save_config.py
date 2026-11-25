import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

os.environ["PYBEMCS_FORCE_CPU"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt5.QtWidgets import QApplication, QMessageBox
from gui.config_io import save_json_config, load_json_config, _sanitize_for_json
from gui.app.main_window import DigitalTwinApp

app = QApplication.instance()
if app is None:
    app = QApplication([""])


class TestSaveConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sanitize_and_save_json_config(self):
        test_dict = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.1415),
            "bool_val": np.bool_(True),
            "arr_val": np.array([1, 2, 3]),
            "nested": {
                "val": np.float32(1.23)
            }
        }
        sanitized = _sanitize_for_json(test_dict)
        self.assertIsInstance(sanitized["int_val"], int)
        self.assertIsInstance(sanitized["float_val"], float)
        self.assertIsInstance(sanitized["bool_val"], bool)
        self.assertIsInstance(sanitized["arr_val"], list)
        self.assertIsInstance(sanitized["nested"]["val"], float)

        save_path = os.path.join(self.temp_dir.name, "subdir", "test_config.json")
        save_json_config(test_dict, save_path)
        self.assertTrue(os.path.isfile(save_path))

        loaded = load_json_config(save_path)
        self.assertEqual(loaded["int_val"], 42)
        self.assertAlmostEqual(loaded["float_val"], 3.1415, places=4)
        self.assertEqual(loaded["bool_val"], True)
        self.assertEqual(loaded["arr_val"], [1, 2, 3])

    def test_menu_actions_exist(self):
        win = DigitalTwinApp()
        self.assertTrue(hasattr(win, "save_action"))
        self.assertTrue(hasattr(win, "save_as_action"))
        self.assertEqual(win.save_action.text(), "Save")
        self.assertEqual(win.save_as_action.text(), "Save as")

    def test_save_config_confirmation_yes_and_no(self):
        win = DigitalTwinApp()
        save_path = os.path.join(self.temp_dir.name, "saved_config.json")
        win.current_config_path = save_path
        win.current_config_name = "saved_config.json"

        # Change a UI value
        win.inputs["n0_plasma"].setValue(8.5e17)
        win.inputs["Te_up"].setValue(5.2)

        # 1. Test clicking "No" on confirmation pop-up
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No) as mock_q:
            win.save_config()
            mock_q.assert_called_once()
            args, kwargs = mock_q.call_args
            self.assertIn("Are you sure you want to save this configuration?", args[2])
            # File should not exist
            self.assertFalse(os.path.exists(save_path))

        # 2. Test clicking "Yes" on confirmation pop-up
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes) as mock_q, \
             patch.object(QMessageBox, "information") as mock_info:
            win.save_config()
            mock_q.assert_called_once()
            args, _ = mock_q.call_args
            self.assertIn("Are you sure you want to save this configuration?", args[2])
            self.assertTrue(os.path.isfile(save_path))
            mock_info.assert_called_once()

            # Verify saved content
            loaded = load_json_config(save_path)
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded["simulation"]["n0_plasma"], 8.5e17)
            self.assertAlmostEqual(loaded["simulation"]["Te_up"], 5.2)
            self.assertEqual(len(loaded["grids"]), len(win.grid_widgets))

    def test_save_as_config(self):
        win = DigitalTwinApp()
        orig_path = os.path.join(self.temp_dir.name, "orig_config.json")
        win.current_config_path = orig_path
        win.current_config_name = "orig_config.json"
        win.update_config_title()

        new_path = os.path.join(self.temp_dir.name, "new_config.json")

        # 1. Test cancelling file dialog directly (no pop-up shown)
        with patch.object(QMessageBox, "question") as mock_q, \
             patch("gui.app.config_mixin.QFileDialog.getSaveFileName", return_value=("", "")):
            win.save_as_config()
            mock_q.assert_not_called()
            self.assertFalse(os.path.exists(new_path))
            self.assertEqual(win.current_config_path, orig_path)

        # 2. Test choosing new file directly (no pop-up shown)
        with patch.object(QMessageBox, "question") as mock_q, \
             patch("gui.app.config_mixin.QFileDialog.getSaveFileName", return_value=(new_path, "JSON Files (*.json)")), \
             patch.object(QMessageBox, "information"):
            win.save_as_config()
            mock_q.assert_not_called()
            self.assertTrue(os.path.isfile(new_path))
            self.assertEqual(win.current_config_path, new_path)
            self.assertEqual(win.current_config_name, "new_config.json")
            self.assertIn("new_config.json", win.lbl_active_config.text())
            self.assertEqual(win.reload_action.text(), "Reload new_config.json")

            # Verify the saved config can be reloaded
            loaded = load_json_config(new_path)
            self.assertIsNotNone(loaded)
            self.assertIn("simulation", loaded)
            self.assertIn("grids", loaded)


if __name__ == "__main__":
    unittest.main()
