import os
import sys
import unittest
import csv
import tempfile
from PyQt5.QtWidgets import QApplication

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.dialogs import (
    ExportParametersDialog,
    EXPORT_PARAMETER_DEFS,
    ExportParticleParametersDialog,
    EXPORT_PARTICLE_PARAMETER_DEFS,
)
from gui.app.main_window import DigitalTwinApp


class TestExportParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(["--platform", "offscreen"])

    def test_export_parameters_dialog_defaults(self):
        dlg = ExportParametersDialog()
        sel = dlg.get_selected_params()
        for pdef in EXPORT_PARAMETER_DEFS:
            self.assertIn(pdef["id"], sel)
            self.assertTrue(sel[pdef["id"]])
        dlg.close()

    def test_export_parameters_dialog_toggle_all(self):
        dlg = ExportParametersDialog()
        dlg._deselect_all()
        sel = dlg.get_selected_params()
        self.assertFalse(any(sel.values()))

        dlg._select_all()
        sel = dlg.get_selected_params()
        self.assertTrue(all(sel.values()))

        dlg._deselect_all()
        dlg._reset_defaults()
        sel = dlg.get_selected_params()
        self.assertTrue(all(sel.values()))
        dlg.close()

    def test_export_particle_parameters_dialog_defaults(self):
        dlg = ExportParticleParametersDialog()
        sel = dlg.get_selected_params()
        for pdef in EXPORT_PARTICLE_PARAMETER_DEFS:
            self.assertIn(pdef["id"], sel)
            self.assertTrue(sel[pdef["id"]])
        dlg.close()

    def test_export_particle_parameters_dialog_toggle_all(self):
        dlg = ExportParticleParametersDialog()
        dlg._deselect_all()
        sel = dlg.get_selected_params()
        self.assertFalse(any(sel.values()))

        dlg._select_all()
        sel = dlg.get_selected_params()
        self.assertTrue(all(sel.values()))

        dlg._deselect_all()
        dlg._reset_defaults()
        sel = dlg.get_selected_params()
        self.assertTrue(all(sel.values()))
        dlg.close()

    def test_main_window_has_parameters_buttons(self):
        win = DigitalTwinApp()
        self.assertTrue(hasattr(win, 'btn_export_params'))
        self.assertEqual(win.btn_export_params.text(), "Parameters")
        self.assertTrue(hasattr(win, 'export_selected_params'))
        self.assertTrue(win.export_selected_params['iteration'])

        self.assertTrue(hasattr(win, 'btn_export_trk_params'))
        self.assertEqual(win.btn_export_trk_params.text(), "Parameters")
        self.assertTrue(hasattr(win, 'export_selected_particle_params'))
        self.assertTrue(win.export_selected_particle_params['time_s'])
        win.close()


if __name__ == "__main__":
    unittest.main()
