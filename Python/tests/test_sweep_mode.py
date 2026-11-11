import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import csv
import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QToolButton, QWidget

from gui.dialogs import (
    SweepSettingsDialog,
    SweepResultsDialog,
    SWEEP_PARAMETERS,
    build_sweep_parameters,
    find_sweep_param_def,
)
from gui.app.main_window import DigitalTwinApp


class TestSweepDialogs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_sweep_settings_dialog_linear_values(self):
        dlg = SweepSettingsDialog()
        dlg.spin_start.setValue(2.0)
        dlg.spin_end.setValue(10.0)
        dlg.spin_n.setValue(5)
        # Select linear spacing
        dlg.combo_scale.setCurrentIndex(0)

        vals = dlg.get_values_array()
        self.assertEqual(len(vals), 5)
        np.testing.assert_allclose(vals, [2.0, 4.0, 6.0, 8.0, 10.0])

        cfg = dlg.get_config()
        self.assertEqual(cfg["max_iterations"], 5000)
        self.assertEqual(cfg["min_iterations"], 500)
        self.assertEqual(cfg["window"], 50)
        self.assertEqual(cfg["threshold"], 1.0)
        dlg.close()

    def test_sweep_settings_dialog_log_values(self):
        dlg = SweepSettingsDialog()
        dlg.spin_start.setValue(1e16)
        dlg.spin_end.setValue(1e18)
        dlg.spin_n.setValue(3)
        idx_log = dlg.combo_scale.findData("log")
        dlg.combo_scale.setCurrentIndex(idx_log)

        vals = dlg.get_values_array()
        self.assertEqual(len(vals), 3)
        np.testing.assert_allclose(vals, [1e16, 1e17, 1e18])
        dlg.close()

    def test_sweep_results_dialog_and_export_warning(self):
        cfg = {
            "param_id": "Te_up",
            "param_label": "Upstream Electron Temp Te_up (eV)",
            "param_unit": "eV",
            "max_iterations": 5000,
        }
        mock_results = [
            {
                "scenario_idx": 1,
                "parameter_value": 2.0,
                "steady_state_reached": True,
                "max_iters_hit": None,
                "steady_iterations": 240,
                "computation_time_s": 5.42,
                "t_sim_us": 1.2,
                "saddle_potential_V": -650.0,
                "divergence_deg": 12.3,
                "divergence_2nd_plane_deg": 10.1,
                "exit_ion_current_A": 0.045,
                "perveance_effective": 1.2e-6,
                "perveance_max": 5.0e-6,
                "perveance_fraction": 0.24,
                "transparency": 0.82,
                "total_energy_J": 1.2e-3,
                "total_charge_C": 1.1e-9,
                "field_energy_J": 4.5e-4,
            },
            {
                "scenario_idx": 2,
                "parameter_value": 4.0,
                "steady_state_reached": False,  # Hit safeguard limit!
                "max_iters_hit": 5000,
                "steady_iterations": 5000,
                "computation_time_s": 12.80,
                "t_sim_us": 25.0,
                "saddle_potential_V": -620.0,
                "divergence_deg": 14.1,
                "divergence_2nd_plane_deg": 11.5,
                "exit_ion_current_A": 0.041,
                "perveance_effective": 1.1e-6,
                "perveance_max": 5.0e-6,
                "perveance_fraction": 0.22,
                "transparency": 0.79,
                "total_energy_J": 1.5e-3,
                "total_charge_C": 1.4e-9,
                "field_energy_J": 4.8e-4,
            },
        ]

        dlg = SweepResultsDialog(cfg, mock_results)
        self.assertEqual(dlg.table.rowCount(), 2)
        self.assertEqual(dlg.table.columnCount(), 12)

        # Check table contents
        item_ss1 = dlg.table.item(0, 2)
        item_ss2 = dlg.table.item(1, 2)
        self.assertIn("YES", item_ss1.text())
        self.assertIn("NO", item_ss2.text())

        # Check computation time column (col 4)
        self.assertEqual(dlg.table.horizontalHeaderItem(4).text(), "Computation Time")
        self.assertEqual(dlg.table.item(0, 4).text(), "5.42 s")
        self.assertEqual(dlg.table.item(1, 4).text(), "12.80 s")

        # Check divergence columns in table (grid exit and 2nd plane)
        self.assertEqual(dlg.table.item(0, 6).text(), "12.30")
        self.assertEqual(dlg.table.item(0, 7).text(), "10.10")
        self.assertEqual(dlg.table.item(1, 6).text(), "14.10")
        self.assertEqual(dlg.table.item(1, 7).text(), "11.50")

        # Check Perveance Fraction column (col 9)
        self.assertEqual(dlg.table.item(0, 9).text(), "0.2400")
        self.assertEqual(dlg.table.item(1, 9).text(), "0.2200")

        # Test CSV export content directly
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        try:
            # Emulate export to tmp_path
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                unconverged = [r for r in mock_results if not r.get("steady_state_reached", True)]
                if unconverged:
                    writer.writerow([f"# WARNING: {len(unconverged)} of {len(mock_results)} scenarios DID NOT reach steady state within {cfg['max_iterations']} iterations!"])
                writer.writerow(["scenario_idx", "parameter_value", "steady_state_reached", "warning_note"])
                for r in mock_results:
                    reached = r.get("steady_state_reached", True)
                    warn = "" if reached else f"WARNING: Max iterations ({r.get('max_iters_hit', 5000)}) reached without steady state"
                    writer.writerow([r["scenario_idx"], r["parameter_value"], "YES" if reached else "NO", warn])

            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("WARNING", content)
            self.assertIn("5000", content)
            self.assertIn("DID NOT reach steady state", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Test TXT tab-delimited export content
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp_txt:
            tmp_txt_path = tmp_txt.name

        try:
            dlg._write_tabular(tmp_txt_path, is_txt=True, notify=False)

            with open(tmp_txt_path, "r", encoding="utf-8") as f:
                txt_content = f.read()

            self.assertIn("\t", txt_content)
            self.assertIn("WARNING", txt_content)
            self.assertIn("5000", txt_content)
            self.assertIn("computation_time_s", txt_content)
            self.assertIn("5.4200", txt_content)
            self.assertIn("beam_divergence_deg", txt_content)
            self.assertIn("beam_divergence_2nd_plane_deg", txt_content)
            self.assertIn("perveance_effective", txt_content)
            self.assertIn("perveance_max", txt_content)
            self.assertIn("perveance_fraction", txt_content)
            self.assertIn("0.2400", txt_content)
            self.assertIn("0.2200", txt_content)
            self.assertIn("10.1", txt_content)
            self.assertIn("11.5", txt_content)
        finally:
            if os.path.exists(tmp_txt_path):
                os.remove(tmp_txt_path)

        # Test JSON export content
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp_json:
            tmp_json_path = tmp_json.name

        try:
            dlg._write_json(tmp_json_path, notify=False)
            import json
            with open(tmp_json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            self.assertEqual(json_data["results"][0]["divergence_deg"], 12.3)
            self.assertEqual(json_data["results"][0]["divergence_2nd_plane_deg"], 10.1)
            self.assertEqual(json_data["results"][1]["divergence_2nd_plane_deg"], 11.5)
            self.assertEqual(json_data["results"][0]["computation_time_s"], 5.42)
            self.assertEqual(json_data["results"][0]["perveance_fraction"], 0.24)
            self.assertEqual(json_data["results"][1]["perveance_fraction"], 0.22)
            self.assertEqual(json_data["results"][0]["perveance_effective"], 1.2e-6)
            self.assertEqual(json_data["results"][0]["perveance_max"], 5.0e-6)
        finally:
            if os.path.exists(tmp_json_path):
                os.remove(tmp_json_path)

        dlg.close()

    def test_density_scientific_notation_in_sweep_dialogs(self):
        # 1. Test SweepSettingsDialog when plasma density is selected
        dlg = SweepSettingsDialog()
        idx_plasma = dlg.combo_param.findData("n0_plasma")
        dlg.combo_param.setCurrentIndex(idx_plasma)

        self.assertTrue(dlg.spin_start.scientific)
        self.assertTrue(dlg.spin_end.scientific)

        # Start value should format as 1e16 without all zeros
        dlg.spin_start.setValue(1e16)
        dlg.spin_end.setValue(2.5e16)
        start_txt = dlg.spin_start.text()
        end_txt = dlg.spin_end.text()

        self.assertIn("1e16", start_txt)
        self.assertNotIn("0000000000000000", start_txt)
        self.assertIn("2.5e16", end_txt)

        # Preview label should also use clean scientific notation
        self.assertIn("1e16", dlg.lbl_preview.text())
        self.assertNotIn("0000000000000000", dlg.lbl_preview.text())

        # 2. Test Neutral Gas Density selection
        idx_n0 = dlg.combo_param.findData("n0")
        dlg.combo_param.setCurrentIndex(idx_n0)
        self.assertTrue(dlg.spin_start.scientific)
        dlg.spin_start.setValue(1e17)
        self.assertIn("1e17", dlg.spin_start.text())
        self.assertNotIn("00000000000000000", dlg.spin_start.text())
        dlg.close()

        # 3. Test SweepResultsDialog table display with density in scientific notation
        cfg_density = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density n0_plasma (m⁻³)",
            "param_unit": "m⁻³",
            "max_iterations": 5000,
        }
        res = [
            {
                "scenario_idx": 1,
                "parameter_value": 1e16,
                "steady_state_reached": True,
                "steady_iterations": 100,
                "saddle_potential_V": -600.0,
                "divergence_deg": 10.0,
                "exit_ion_current_A": 0.05,
                "transparency": 0.85,
                "total_energy_J": 1e-3,
            }
        ]
        res_dlg = SweepResultsDialog(cfg_density, res)
        table_val_txt = res_dlg.table.item(0, 1).text()
        self.assertEqual(table_val_txt, "1e16")
        self.assertNotIn("0000000000000000", table_val_txt)

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
        try:
            res_dlg._write_tabular(tmp_path, is_txt=False, notify=False)
            with open(tmp_path, "r", encoding="utf-8") as f:
                csv_txt = f.read()
            self.assertIn("1e16", csv_txt)
            self.assertNotIn("10000000000000000", csv_txt)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        res_dlg.close()

    def test_sweep_dialog_upstream_gap_parameter(self):
        """Verify Upstream Gap (mm) is selectable in SweepSettingsDialog with 3-decimal mm precision."""
        self.assertIn("upstream_gap_mm", [p["id"] for p in SWEEP_PARAMETERS])

        dlg = SweepSettingsDialog()
        idx = dlg.combo_param.findData("upstream_gap_mm")
        self.assertGreaterEqual(idx, 0)
        dlg.combo_param.setCurrentIndex(idx)

        self.assertFalse(dlg.spin_start.scientific)
        self.assertEqual(dlg.spin_start.decimals(), 3)
        self.assertEqual(dlg.spin_start.suffix(), " mm")
        self.assertEqual(dlg.combo_scale.currentData(), "linear")

        dlg.spin_start.setValue(0.375)
        dlg.spin_end.setValue(1.625)
        dlg.spin_n.setValue(3)

        cfg = dlg.get_config()
        self.assertEqual(cfg["param_id"], "upstream_gap_mm")
        self.assertEqual(cfg["param_unit"], "mm")
        self.assertEqual(cfg["values"], [0.375, 1.0, 1.625])
        self.assertIn("0.375", dlg.lbl_preview.text())
        self.assertIn("1.62", dlg.lbl_preview.text())
        dlg.close()

    def test_build_sweep_parameters_dynamic_5_props_per_grid(self):
        """Verify build_sweep_parameters exposes all 5 properties for grids 1..N."""
        for n_grids in (1, 3, 4):
            catalog = build_sweep_parameters(n_grids)
            ids = [p["id"] for p in catalog]
            for g in range(1, n_grids + 1):
                for prop in ("V", "t", "gap", "r", "cham"):
                    self.assertIn(f"grid_{g}_{prop}", ids)
            self.assertNotIn(f"grid_{n_grids + 1}_V", ids)
            self.assertIn("upstream_gap_mm", ids)

        dummy_parent = QWidget()
        dummy_parent.grid_widgets = [{}, {}, {}]
        dlg = SweepSettingsDialog(dummy_parent)
        combo_ids = [dlg.combo_param.itemData(i) for i in range(dlg.combo_param.count())]
        for g in (1, 2, 3):
            for prop in ("V", "t", "gap", "r", "cham"):
                self.assertIn(f"grid_{g}_{prop}", combo_ids)
        self.assertNotIn("grid_4_V", combo_ids)

        idx_cham = dlg.combo_param.findData("grid_3_cham")
        self.assertGreaterEqual(idx_cham, 0)
        dlg.combo_param.setCurrentIndex(idx_cham)
        self.assertEqual(dlg.spin_start.suffix(), " °")
        self.assertEqual(dlg.spin_start.decimals(), 2)
        dlg.close()

        pdef_g4 = find_sweep_param_def("grid_4_cham")
        self.assertIsNotNone(pdef_g4)
        self.assertEqual(pdef_g4["unit"], "°")

        res_dlg = SweepResultsDialog(
            {"param_id": "grid_3_cham", "param_label": "Grid 3 Chamfer (°)", "param_unit": "°"},
            [{
                "scenario_idx": 1, "parameter_value": 15.0, "steady_state_reached": True,
                "steady_iterations": 100, "saddle_potential_V": -10.0, "divergence_deg": 5.0,
                "exit_ion_current_A": 0.02, "transparency": 0.9, "total_energy_J": 1e-4,
            }],
        )
        self.assertEqual(res_dlg.table.rowCount(), 1)
        res_dlg.close()

    def test_sweep_dialog_workflow_help_dot(self):
        """Verify Group 1 has a '?' help dot explaining the grid-build/sweep workflow."""
        dlg = SweepSettingsDialog()
        self.assertTrue(hasattr(dlg, "btn_param_help"))
        self.assertIsInstance(dlg.btn_param_help, QToolButton)
        self.assertEqual(dlg.btn_param_help.text(), "?")
        self.assertIn("1. BUILD DOMAIN", dlg.btn_param_help.toolTip())
        with patch.object(QMessageBox, "information") as mock_info:
            dlg.btn_param_help.click()
            mock_info.assert_called_once()
            self.assertIn("1. BUILD DOMAIN", mock_info.call_args[0][2])
        dlg.close()

    def test_sweep_dialog_validation_blocks_invalid_geometric_ranges(self):
        """Verify saving invalid geometric bounds in SweepSettingsDialog is blocked via warning popup."""
        dummy_parent = QWidget()
        dummy_parent.grid_widgets = [{}, {}, {}]
        dlg = SweepSettingsDialog(dummy_parent)

        invalid_cases = [
            ("upstream_gap_mm", 0.0, 1.0),
            ("grid_2_t", 0.0, 0.5),
            ("grid_2_r", -0.1, 0.5),
            ("grid_2_gap", 0.0, 0.5),   # internal gap <= 0
            ("grid_3_gap", -0.2, 0.5),  # final gap < 0
            ("grid_3_cham", 0.0, 50.0), # chamfer > 45
        ]
        for param_id, start, end in invalid_cases:
            idx = dlg.combo_param.findData(param_id)
            dlg.combo_param.setCurrentIndex(idx)
            dlg.spin_start.setValue(start)
            dlg.spin_end.setValue(end)
            with patch.object(QMessageBox, "warning") as mock_warn:
                dlg.btn_start.click()
                mock_warn.assert_called_once()
                self.assertNotEqual(dlg.result(), QDialog.Accepted)

        # Valid downstream gap == 0.0 on final grid (Grid 3 of 3) is allowed
        dlg.combo_param.setCurrentIndex(dlg.combo_param.findData("grid_3_gap"))
        dlg.spin_start.setValue(0.0)
        dlg.spin_end.setValue(0.5)
        with patch.object(QMessageBox, "warning") as mock_warn:
            dlg.btn_start.click()
            mock_warn.assert_not_called()
            self.assertEqual(dlg.result(), QDialog.Accepted)
        dlg.close()

    def test_sweep_dialog_contextual_auto_upstream_gap_toggle(self):
        """Verify auto-update upstream gap checkbox appears only for influencing params and defaults to True."""
        # 1. Presheath mode (default): visible for grid_1_r, hidden for Te_up and grid_1_V / grid_2_r
        parent_pre = QWidget()
        parent_pre.grid_widgets = [{}, {}]
        parent_pre.adv_params = {"entire_bulk_plasma": False}
        parent_pre.upstream_gap_settings = {"method": "ratio", "ratio_base": "screen_r", "ratio": 0.75}

        dlg = SweepSettingsDialog(parent_pre)
        for pid in ("Te_up", "grid_1_V", "grid_2_r"):
            dlg.combo_param.setCurrentIndex(dlg.combo_param.findData(pid))
            self.assertTrue(dlg.box_auto_upstream_gap.isHidden())
            self.assertFalse(dlg.get_config()["auto_update_upstream_gap"])

        dlg.combo_param.setCurrentIndex(dlg.combo_param.findData("grid_1_r"))
        self.assertFalse(dlg.box_auto_upstream_gap.isHidden())
        self.assertTrue(dlg.chk_auto_upstream_gap.isChecked())
        self.assertIn("Variable", dlg.lbl_auto_gap_hint.text())
        cfg_var = dlg.get_config()
        self.assertTrue(cfg_var["auto_update_upstream_gap"])
        self.assertEqual(cfg_var["upstream_gap_coupling"], {"base": "screen_r", "ratio": 0.75})

        dlg.chk_auto_upstream_gap.setChecked(False)
        self.assertIn("Fixed", dlg.lbl_auto_gap_hint.text())
        cfg_fixed = dlg.get_config()
        self.assertFalse(cfg_fixed["auto_update_upstream_gap"])
        dlg.close()

        # 2. Debye bulk-plasma mode: visible for n0_plasma & Te_up too
        parent_debye = QWidget()
        parent_debye.grid_widgets = [{}, {}]
        parent_debye.adv_params = {"entire_bulk_plasma": True}
        dlg_d = SweepSettingsDialog(parent_debye)
        for pid in ("n0_plasma", "Te_up", "grid_1_r"):
            dlg_d.combo_param.setCurrentIndex(dlg_d.combo_param.findData(pid))
            self.assertFalse(dlg_d.box_auto_upstream_gap.isHidden())
            self.assertTrue(dlg_d.chk_auto_upstream_gap.isChecked())
            self.assertTrue(dlg_d.get_config()["auto_update_upstream_gap"])
        dlg_d.close()

    def test_sweep_divergence_second_plane_output(self):
        cfg = {
            "param_id": "grid_2_V",
            "param_label": "Accel Grid Voltage (V)",
            "param_unit": "V",
            "max_iterations": 2000,
        }
        res = [
            {
                "scenario_idx": 1,
                "parameter_value": -200.0,
                "steady_state_reached": True,
                "steady_iterations": 150,
                "saddle_potential_V": -620.0,
                "divergence_deg": 15.2,
                "divergence_2nd_plane_deg": 11.4,
                "exit_ion_current_A": 0.048,
                "transparency": 0.81,
                "total_energy_J": 1.1e-3,
            },
            {
                "scenario_idx": 2,
                "parameter_value": -300.0,
                "steady_state_reached": True,
                "steady_iterations": 160,
                "saddle_potential_V": -650.0,
                "divergence_deg": 13.8,
                "divergence_plane2_deg": 10.2,  # test alternative key fallback
                "exit_ion_current_A": 0.050,
                "transparency": 0.83,
                "total_energy_J": 1.2e-3,
            },
            {
                "scenario_idx": 3,
                "parameter_value": -400.0,
                "steady_state_reached": True,
                "steady_iterations": 170,
                "saddle_potential_V": -680.0,
                "divergence_deg": 12.5,
                # scenario without 2nd plane key to test fallback
                "exit_ion_current_A": 0.052,
                "transparency": 0.85,
                "total_energy_J": 1.3e-3,
            },
        ]
        dlg = SweepResultsDialog(cfg, res)
        self.assertEqual(dlg.table.columnCount(), 12)
        self.assertEqual(dlg.table.horizontalHeaderItem(4).text(), "Computation Time")
        self.assertEqual(dlg.table.horizontalHeaderItem(6).text(), "Divergence Grid Exit (°)")
        self.assertEqual(dlg.table.horizontalHeaderItem(7).text(), "Divergence 2nd Plane (°)")
        self.assertEqual(dlg.table.horizontalHeaderItem(9).text(), "Perveance Fraction")

        # Verify values in table
        self.assertEqual(dlg.table.item(0, 6).text(), "15.20")
        self.assertEqual(dlg.table.item(0, 7).text(), "11.40")
        self.assertEqual(dlg.table.item(1, 6).text(), "13.80")
        self.assertEqual(dlg.table.item(1, 7).text(), "10.20")
        self.assertEqual(dlg.table.item(2, 6).text(), "12.50")
        self.assertEqual(dlg.table.item(2, 7).text(), "0.00")

        # Test CSV export
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
        try:
            dlg._write_tabular(tmp_path, is_txt=False, notify=False)
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("beam_divergence_deg,beam_divergence_2nd_plane_deg", content)
            self.assertIn("perveance_effective,perveance_max,perveance_fraction", content)
            self.assertIn("15.2,11.4", content)
            self.assertIn("13.8,10.2", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        dlg.close()


class TestSweepMixinAppIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self):
        self.win = DigitalTwinApp()

    def tearDown(self):
        self.win.is_sweep_active = False
        self.win.sweep_mode_armed = False
        self.win._sweep_stopped = True
        if hasattr(self.win, 'parallel_sweep_worker') and self.win.parallel_sweep_worker is not None:
            worker = self.win.parallel_sweep_worker
            self.win.parallel_sweep_worker = None
            try:
                worker.cancel()
                worker.wait(1000)
            except Exception:
                pass
        if hasattr(self.win, '_sim_worker') and self.win._sim_worker.isRunning():
            self.win._sim_worker.stop(wait_ms=1000)
        self.win.close()

    def test_sweep_ui_elements_exist(self):
        self.assertFalse(self.win.is_sweep_active)
        self.assertIsNotNone(self.win.lbl_sweep_badge)
        self.assertFalse(self.win.lbl_sweep_badge.isVisible())
        self.assertIsNotNone(self.win.btn_stop_sweep)
        self.assertFalse(self.win.btn_stop_sweep.isVisible())

    def test_apply_sweep_value(self):
        # Test applying Te_up value
        self.win._apply_sweep_value("Te_up", 7.5)
        self.assertAlmostEqual(self.win.inputs["Te_up"].value(), 7.5)

        # Test applying n0_plasma value
        self.win._apply_sweep_value("n0_plasma", 5e17)
        self.assertAlmostEqual(self.win.inputs["n0_plasma"].value(), 5e17)

    def test_sweep_progression_logic(self):
        cfg = {
            "param_id": "Te_up",
            "param_label": "Upstream Electron Temp Te_up (eV)",
            "param_unit": "eV",
            "values": [2.0, 5.0],
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 50,
        }

        self.win.sweep_config = cfg
        self.win.sweep_results = []
        self.win.sweep_current_idx = 0
        self.win.is_sweep_active = True
        self.win.sweep_scenario_start_iter = 0

        # Simulate 60 iterations in scenario 1 with stable metrics
        self.win.sim.iteration = 60
        self.win.sim.energy_history_tot = [1e-3] * 60
        self.win.sim.charge_history_q_net = [1e-9] * 60
        self.win.sim.potential_history = [-700.0] * 60
        self.win.div_history = [12.5] * 60
        self.win.div_mid_history = [9.4] * 60

        # Check steady state should detect stationarity and record result
        self.win.check_sweep_steady_state()

        self.assertEqual(len(self.win.sweep_results), 1)
        self.assertTrue(self.win.sweep_results[0]["steady_state_reached"])
        self.assertEqual(self.win.sweep_results[0]["parameter_value"], 2.0)
        self.assertEqual(self.win.sweep_results[0]["divergence_deg"], 12.5)
        self.assertEqual(self.win.sweep_results[0]["divergence_2nd_plane_deg"], 9.4)
        self.assertIn("perveance_effective", self.win.sweep_results[0])
        self.assertIn("perveance_max", self.win.sweep_results[0])
        self.assertIn("perveance_fraction", self.win.sweep_results[0])
        self.assertEqual(self.win.sweep_current_idx, 1)

        # Now simulate scenario 2 hitting safeguard 5000 iterations without convergence
        self.win.sweep_scenario_start_iter = 60
        self.win.sim.iteration = 60 + 5000
        # Oscillating/drifting metrics
        self.win.sim.energy_history_tot = [1e-3 * (1.0 + 0.1 * (k % 2)) for k in range(5060)]
        self.win.sim.charge_history_q_net = [1e-9] * 5060
        self.win.sim.potential_history = [-700.0] * 5060

        self.win.check_sweep_steady_state()

        self.assertEqual(len(self.win.sweep_results), 2)
        # Second scenario hit max iters, so steady_state_reached should be False
        self.assertFalse(self.win.sweep_results[1]["steady_state_reached"])
        self.assertEqual(self.win.sweep_results[1]["max_iters_hit"], 5000)
        self.assertFalse(self.win.is_sweep_active)

    def test_stop_sweep(self):
        self.win.is_sweep_active = True
        self.win.btn_stop_sweep.setVisible(True)
        self.win.stop_sweep()
        self.assertFalse(self.win.is_sweep_active)
        self.assertFalse(self.win.btn_stop_sweep.isVisible())
        self.assertIn("stopped", self.win.lbl_sweep_badge.text())

    def test_build_domain_clears_graphs_and_zeroes_initial_conditions(self):
        # Populate artificial histories and graph lines as if simulation ran to 400 iters
        self.win.iter_history = list(range(1, 401))
        self.win.ebs_history = [-600.0] * 400
        self.win.div_history = [12.0] * 400
        self.win.line_ebs.set_data(self.win.iter_history, self.win.ebs_history)
        self.win.line_div.set_data(self.win.iter_history, self.win.div_history)
        self.win.sim.iteration = 400
        self.win.sim.num_p = 500

        # Execute build_domain (with patched warning popup to prevent blocking modal)
        with patch('PyQt5.QtWidgets.QMessageBox.warning'):
            self.win.build_domain()

        # Check initial conditions are zero
        self.assertEqual(self.win.sim.iteration, 0)
        self.assertEqual(self.win.sim.num_p, 0)
        self.assertEqual(len(self.win.sim.energy_history_tot), 0)
        self.assertEqual(len(self.win.sim.charge_history_q_net), 0)

        # Check histories and graph lines are clear
        self.assertEqual(len(self.win.iter_history), 0)
        self.assertEqual(len(self.win.ebs_history), 0)
        self.assertEqual(len(self.win.div_history), 0)
        self.assertEqual(len(self.win.line_ebs.get_xdata()), 0)
        self.assertEqual(len(self.win.line_div.get_xdata()), 0)

        # Verify that a stale snapshot from previous run_id is discarded
        stale_snap = {
            'run_id': self.win._sim_run_id - 1,
            'iteration': 400,
            'min_pot': -600.0,
            'current_div': 12.0,
            'current_div_mid': 12.0,
        }
        self.win._on_step_result(stale_snap)
        self.assertEqual(len(self.win.iter_history), 0, "Stale snapshot should be discarded!")

        # Verify that backward jumps in iteration clear histories rather than connecting back
        self.win.iter_history = [400]
        self.win.ebs_history = [-600.0]
        self.win.div_history = [12.0]
        # Simulate arrival of step 1 from a new run
        snap_step1 = {
            'params': {},
            'remeshed': False,
            'min_pot': -550.0,
            'current_div': 11.0,
            'current_div_mid': 11.0,
            'T_grids': [],
            'trans_last_frame': 0.8,
            'transparency': 0.8,
            'iteration': 1,
            'dt': 1e-9,
            'num_p': 10,
            'num_e': 10,
            'exit_vx_mean': 1e4,
            'exit_v_mean': 1e4,
            'exit_energy_mean_eV': 1000.0,
            'exit_count_step': 5,
            'exit_ion_current_step': 0.01,
            'exit_ion_current_avg': 0.01,
            'transmitted_ions_step': 5,
            'total_active_cells': 20,
            'low_ppc_cells': 2,
            'injection_enabled': True,
            'has_active_particles': True,
            'run_id': self.win._sim_run_id,
            'prim_xy': np.empty((0, 2)),
            'cex_xy': np.empty((0, 2)),
            'elec_xy': np.empty((0, 2)),
            'e_prim_eV': np.empty(0),
            'e_cex_eV': np.empty(0),
            'perf_monitor': None,
            'X': self.win.sim.X,
            'Y': self.win.sim.Y,
            'V': self.win.sim.V,
            'damage_map': self.win.sim.damage_map,
            'Tmap': None,
            'isBound': self.win.sim.isBound,
            'mask_grids': [],
            'x_coords': self.win.sim.x_coords,
            'y_coords': self.win.sim.y_coords,
            'p_isCEX': np.empty(0, dtype=bool),
            'p_vx': np.empty(0),
            'p_vy': np.empty(0),
            'e_x': np.empty(0),
            'e_vx': np.empty(0),
            'e_vy': np.empty(0),
            'm_ion': self.win.sim.m_ion,
            'm_e': self.win.sim.m_e,
            'q': self.win.sim.q,
            'Lx': self.win.sim.Lx,
            'Ly': self.win.sim.Ly,
            'dx': self.win.sim.dx,
            'dy': self.win.sim.dy,
            'nx': self.win.sim.nx,
            'ny': self.win.sim.ny,
        }
        self.win._process_snap(snap_step1)
        # Should now have only iteration 1, not [400, 1]!
        self.assertEqual(self.win.iter_history, [1])
        self.assertEqual(self.win.div_history, [11.0])

    def test_perveance_computation_metrics(self):
        # Configure known grid geometry and voltage
        # Screen grid: 1100 V, gap: 1.0 mm, hole radius: 1.0 mm
        # Accel grid: -200 V
        # Total voltage = 1300 V
        self.win.sim.grids = [
            {'V': 1100.0, 't': 0.5, 'gap': 1.0, 'r': 1.0, 'cham': 0.0},
            {'V': -200.0, 't': 0.5, 'gap': 1.0, 'r': 0.6, 'cham': 0.0},
        ]
        self.win.sim._last_injection_area = 1.0e-6  # 1 mm^2 in m^2
        self.win.sim.q_ion = 1.602e-19
        self.win.sim.m_ion = 131.293 * 1.6605e-27

        i_exit = 5.0e-5  # 0.05 mA exit current (typical single aperture)
        p_eff, p_max, perv_frac = self.win._compute_perveance_metrics(i_exit)

        v_tot = 1300.0
        expected_p_eff = 5.0e-5 / (v_tot ** 1.5)
        eps_0 = 8.8541878128e-12
        # Goebel & Katz equivalent length: l_e^2 = (l_g + t_s)^2 + r_s^2
        l_e_sq = (1.0e-3 + 0.5e-3) ** 2 + (1.0e-3) ** 2
        expected_p_max = (4.0 * eps_0 / 9.0) * np.sqrt(2.0 * self.win.sim.q_ion / self.win.sim.m_ion) * (1.0e-6 / l_e_sq)
        expected_frac = expected_p_eff / expected_p_max

        self.assertAlmostEqual(p_eff, expected_p_eff, places=12)
        self.assertAlmostEqual(p_max, expected_p_max, places=12)
        self.assertAlmostEqual(perv_frac, expected_frac, places=6)
        self.assertTrue(0.0 < perv_frac < 1.0)

    def test_perveance_edge_cases(self):
        self.win.sim.grids = [
            {'V': 100.0, 'gap': 1.0, 'r': 1.0},
            {'V': 200.0, 'gap': 1.0, 'r': 1.0},  # V_total <= 0
        ]
        p_eff, p_max, perv_frac = self.win._compute_perveance_metrics(0.05)
        self.assertEqual(p_eff, 0.0)
        self.assertEqual(perv_frac, 0.0)
        self.assertGreater(p_max, 0.0)

        # Zero or negative exit current
        self.win.sim.grids = [
            {'V': 1000.0, 'gap': 1.0, 'r': 1.0},
            {'V': -200.0, 'gap': 1.0, 'r': 1.0},
        ]
        p_eff_zero, _, perv_frac_zero = self.win._compute_perveance_metrics(0.0)
        self.assertEqual(p_eff_zero, 0.0)
        self.assertEqual(perv_frac_zero, 0.0)

        p_eff_nan, _, perv_frac_nan = self.win._compute_perveance_metrics(float('nan'))
        self.assertEqual(p_eff_nan, 0.0)
        self.assertEqual(perv_frac_nan, 0.0)

    def test_sweep_dialog_save_settings_button_text(self):
        dlg = SweepSettingsDialog()
        self.assertEqual(dlg.btn_start.text(), "Save Sweep Settings")
        dlg.close()

    def test_save_sweep_settings_arms_mode_without_running(self):
        cfg = {
            "param_id": "Te_up",
            "param_label": "Upstream Electron Temp Te_up (eV)",
            "param_unit": "eV",
            "values": [2.0, 5.0, 8.0],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 50,
        }
        self.win.save_sweep_settings(cfg)
        self.assertTrue(self.win.sweep_mode_armed)
        self.assertFalse(self.win.is_sweep_active)
        self.assertFalse(self.win.sim_isRunning)
        self.assertEqual(self.win.combo_sweep_scenario.count(), 3)
        self.assertFalse(self.win.combo_sweep_scenario.isHidden())
        self.assertFalse(self.win.btn_resume_sweep.isHidden())
        self.assertTrue(self.win.btn_stop_sweep.isHidden())
        self.assertIn("Sweep mode armed", self.win.lbl_sweep_badge.text())
        self.assertEqual(self.win.btn_toggle.text(), "2. START BEAM")

    def test_start_sweep_with_normal_buttons_and_stop_single_sweep(self):
        cfg = {
            "param_id": "Te_up",
            "param_label": "Upstream Electron Temp Te_up (eV)",
            "param_unit": "eV",
            "values": [3.0, 6.0],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 50,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Starting sweep via top-bar Start Sweep button
        with patch.object(self.win._sim_worker, "start_sim") as mock_start:
            self.win.on_sweep_action_clicked()
            self.assertTrue(self.win.is_sweep_active)
            self.assertFalse(self.win.sweep_mode_armed)
            self.assertTrue(self.win.sim_isRunning)
            self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")
            self.assertEqual(self.win.btn_resume_sweep.text(), "Pause Sweep")

        # Click Stop: should stop ONLY the current sweep, not all of them
        with patch.object(self.win._sim_worker, "pause") as mock_pause:
            self.win.toggle_sim()
            self.assertFalse(self.win.sim_isRunning)
            self.assertTrue(self.win._sweep_paused_current)
            self.assertTrue(self.win.is_sweep_active)  # All of them are NOT stopped!
            self.assertEqual(self.win.btn_toggle.text(), "RESUME BEAM")

        # Steady state check must not advance while current sweep is stopped/paused
        self.win.sim.iteration = 100
        self.win.sim.energy_history_tot = [1e-3] * 100
        self.win.sim.charge_history_q_net = [1e-9] * 100
        self.win.sim.potential_history = [-700.0] * 100
        self.win.div_history = [12.0] * 100
        self.win.check_sweep_steady_state()
        self.assertEqual(len(self.win.sweep_results), 0)

        # Click Resume: resumes the current sweep
        with patch.object(self.win._sim_worker, "start_sim") as mock_start_resume:
            self.win.toggle_sim()
            self.assertTrue(self.win.sim_isRunning)
            self.assertFalse(self.win._sweep_paused_current)
            self.assertTrue(self.win.is_sweep_active)
            self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")

        # Stop simulation worker cleanly
        self.win.stop_sweep()

    def test_stop_all_sweeps_and_resume_with_selected_parameters(self):
        cfg = {
            "param_id": "Te_up",
            "param_label": "Upstream Electron Temp Te_up (eV)",
            "param_unit": "eV",
            "values": [2.0, 5.0, 9.0],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 50,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()
        self.win.is_sweep_active = True

        # Stop all of them using btn_stop_sweep
        self.win.stop_sweep()
        self.assertFalse(self.win.is_sweep_active)
        self.assertFalse(self.win.btn_stop_sweep.isVisible())
        self.assertFalse(self.win.btn_resume_sweep.isHidden())

        # Select Scenario 2 (Te_up = 5.0) in the scenario selector
        self.win.combo_sweep_scenario.setCurrentIndex(1)

        # Resuming with selected parameters applies parameter for scenario 2 directly without blocking popup
        self.win.resume_sweep_with_selected()
        self.assertEqual(self.win.inputs["Te_up"].value(), 5.0)
        self.assertTrue(self.win.is_sweep_active)
        self.assertEqual(self.win.sweep_current_idx, 1)

        # Cleanly stop simulation worker
        self.win.stop_sweep()

    def test_gui_settings_change_at_runtime_without_rebuild_popup(self):
        self.win.build_domain()

        # Modify an input without building domain - should run at runtime without popup dialog
        self.win.inputs["Te_up"].setValue(99.0)

        with patch("gui.app.sim_control.QMessageBox.warning") as mock_warn:
            with patch.object(self.win._sim_worker, "start_sim"):
                self.win.toggle_sim()
                mock_warn.assert_not_called()
                self.assertTrue(self.win.sim_isRunning)
                self.win.toggle_sim()



    def test_tent_menu_scenario_switch_updates_grids_and_visuals(self):
        cfg = {
            "param_id": "grid_1_V",
            "param_label": "Grid 1 DC Voltage (V)",
            "param_unit": "V",
            "values": [800.0, 1200.0, 1600.0],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Initially on scenario 1 (index 0)
        self.assertEqual(self.win.grid_widgets[0]["V"].value(), 800.0)

        # Switch to scenario 2 (index 1) via tent menu
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.assertEqual(self.win.grid_widgets[0]["V"].value(), 1200.0)
        self.assertIsNotNone(self.win.scat_bound)
        self.assertIsNotNone(self.win.contour_live)

        # Switch to scenario 3 (index 2) via tent menu
        self.win.combo_sweep_scenario.setCurrentIndex(2)
        self.assertEqual(self.win.grid_widgets[0]["V"].value(), 1600.0)
        self.assertIsNotNone(self.win.scat_bound)
        self.assertIsNotNone(self.win.contour_live)

    def test_pause_resume_beam_for_selected_scenario_parallel(self):
        from unittest.mock import MagicMock
        cfg = {
            "param_id": "grid_1_V",
            "param_label": "Grid 1 DC Voltage (V)",
            "param_unit": "V",
            "values": [800.0, 1200.0],
            "parallel": True,
            "n_cores": 2,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Start sweep via top-bar Start Sweep action
        with patch.object(self.win, "_start_parallel_sweep") as mock_start_parallel:
            self.win.on_sweep_action_clicked()
            mock_start_parallel.assert_called_once()
            self.win.is_sweep_active = True

        # Mock parallel_sweep_worker
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        paused_scens = set()
        mock_worker.is_scenario_paused.side_effect = lambda num: num in paused_scens
        def pause_scen(num): paused_scens.add(num)
        def resume_scen(num): paused_scens.discard(num)
        mock_worker.pause_scenario.side_effect = pause_scen
        mock_worker.resume_scenario.side_effect = resume_scen
        mock_worker.get_status_dict.return_value = {
            1: {"status": "running" if 1 not in paused_scens else "paused", "iter": 10},
            2: {"status": "running" if 2 not in paused_scens else "paused", "iter": 10},
        }
        self.win.parallel_sweep_worker = mock_worker

        # Select Scenario 2 (index 1) in tent menu
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")

        # Click btn_toggle to pause ONLY Scenario 2
        self.win.toggle_sim()
        self.assertIn(2, paused_scens)
        self.assertNotIn(1, paused_scens)
        self.assertEqual(self.win.btn_toggle.text(), "RESUME BEAM")

        # Switch tent menu to Scenario 1 (index 0) -> should show PAUSE BEAM since it's running
        self.win.combo_sweep_scenario.setCurrentIndex(0)
        self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")

        # Switch tent menu back to Scenario 2 (index 1) -> should show RESUME BEAM since it's paused
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.assertEqual(self.win.btn_toggle.text(), "RESUME BEAM")

        # Click btn_toggle to resume Scenario 2
        self.win.toggle_sim()
        self.assertNotIn(2, paused_scens)
        self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")

        self.win.stop_sweep()

    def test_gui_settings_dialog_custom_particles_default(self):
        from gui.dialogs import GUISettingsDialog
        dlg = GUISettingsDialog(self.win)
        self.assertTrue(dlg._rb_custom.isChecked())
        self.assertEqual(dlg._spin_custom.value(), 2000)
        self.assertEqual(dlg.get_scatter_limit(), 2000)
        dlg.close()

    def test_parallel_sweep_active_gui_view_tracking(self):
        cfg = {
            "param_id": "grid_1_V",
            "param_label": "Grid 1 DC Voltage (V)",
            "param_unit": "V",
            "values": [800.0, 1200.0, 1600.0],
            "parallel": True,
            "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)
        from gui.workers.sweep_worker import ParallelSweepWorker
        worker = ParallelSweepWorker(cfg, self.win.get_params(), n_cores=2, scatter_limit=1000)
        self.win.parallel_sweep_worker = worker

        # Initially active scenario is 1
        with open(worker.active_view_file, "r") as f:
            self.assertEqual(f.read().strip(), "1")

        # Switching tent menu to scenario 2 updates active_view_file so scenario 2 extracts particles and scenario 1 speeds up
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        with open(worker.active_view_file, "r") as f:
            self.assertEqual(f.read().strip(), "2")

        # Switching tent menu to scenario 3 updates active_view_file to 3
        self.win.combo_sweep_scenario.setCurrentIndex(2)
        with open(worker.active_view_file, "r") as f:
            self.assertEqual(f.read().strip(), "3")

        worker._cleanup_temp_files()
        self.win.stop_sweep()

    def test_exit_sweep_button_exists_and_workflow(self):
        self.assertTrue(hasattr(self.win, 'btn_exit_sweep'))
        self.assertEqual(self.win.btn_exit_sweep.text(), "Exit sweep")

        cfg = {
            "param_id": "grid_1_V",
            "param_label": "Grid 1 DC Voltage (V)",
            "param_unit": "V",
            "values": [800.0, 1200.0],
            "parallel": True,
            "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)
        self.assertFalse(self.win.btn_exit_sweep.isHidden())
        self.assertFalse(self.win.btn_resume_sweep.isHidden())
        self.assertFalse(self.win.combo_sweep_scenario.isHidden())

        # 1. Test Cancel: should stay in armed/sweep mode
        self.win.exit_sweep(confirm_choice="cancel")
        self.assertFalse(self.win.btn_exit_sweep.isHidden())
        self.assertTrue(self.win.sweep_mode_armed)

        # 2. Test Yes: saves data until now, opens SweepResultsDialog, and returns to normal simulation
        from unittest.mock import MagicMock
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        mock_worker.get_status_dict.return_value = {
            1: {"status": "running", "iter": 42, "div": 12.5, "parameter_value": 800.0},
            2: {"status": "paused", "iter": 30, "div": 14.1, "parameter_value": 1200.0},
        }
        self.win.parallel_sweep_worker = mock_worker
        self.win.is_sweep_active = True

        self.win.exit_sweep(confirm_choice="yes")
        self.assertFalse(self.win.is_sweep_active)
        self.assertFalse(self.win.sweep_mode_armed)
        self.assertTrue(self.win.btn_exit_sweep.isHidden())
        self.assertTrue(self.win.combo_sweep_scenario.isHidden())
        self.assertTrue(self.win.btn_stop_sweep.isHidden())
        self.assertTrue(self.win.btn_resume_sweep.isHidden())
        self.assertEqual(self.win.btn_toggle.text(), "2. START BEAM")
        self.assertTrue(hasattr(self.win, 'sweep_results_dialog'))
        self.assertEqual(len(self.win.sweep_results_dialog.results), 2)
        self.assertEqual(self.win.sweep_results_dialog.results[0]["steady_iterations"], 42)
        self.win.sweep_results_dialog.close()

        # 3. Test No: exits directly without saving
        self.win.save_sweep_settings(cfg)
        self.assertFalse(self.win.btn_exit_sweep.isHidden())
        self.win.sweep_results_dialog = None
        self.win.exit_sweep(confirm_choice="no")
        self.assertFalse(self.win.is_sweep_active)
        self.assertFalse(self.win.sweep_mode_armed)
        self.assertTrue(self.win.btn_exit_sweep.isHidden())
        self.assertIsNone(self.win.sweep_results_dialog)

    def test_combo_sweep_scenario_size_stability(self):
        """Verify combo_sweep_scenario has fixed width and does not resize when item texts change."""
        self.assertEqual(self.win.combo_sweep_scenario.minimumWidth(), self.win.combo_sweep_scenario.maximumWidth())
        initial_width = self.win.combo_sweep_scenario.width() or self.win.combo_sweep_scenario.minimumWidth()

        cfg = {
            "param_id": "grid_1_V",
            "values": [800.0, 1200.0],
            "parallel": True,
            "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)
        self.assertEqual(self.win.combo_sweep_scenario.minimumWidth(), initial_width)

        # Update items with longer/shorter text
        self.win._update_sweep_scenario_combo_items()
        self.assertEqual(self.win.combo_sweep_scenario.minimumWidth(), initial_width)
        self.assertEqual(self.win.combo_sweep_scenario.maximumWidth(), initial_width)
        self.win.exit_sweep(confirm_choice="no")

    def test_stop_sweep_and_resume_with_selected_preserves_iterations(self):
        """Verify stop_sweep followed by resume_sweep_with_selected resumes worker from paused state without restarting from 0."""
        from unittest.mock import MagicMock
        cfg = {
            "param_id": "grid_1_V",
            "param_label": "Grid 1 DC Voltage (V)",
            "param_unit": "V",
            "values": [800.0, 1200.0],
            "parallel": True,
            "n_cores": 2,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        paused_scens = set()
        mock_worker.is_scenario_paused.side_effect = lambda num: num in paused_scens
        def pause_scen(num): paused_scens.add(num)
        def resume_scen(num): paused_scens.discard(num)
        mock_worker.pause_scenario.side_effect = pause_scen
        mock_worker.resume_scenario.side_effect = resume_scen
        def pause_all():
            paused_scens.add(1)
            paused_scens.add(2)
        def resume_all():
            paused_scens.clear()
        mock_worker.pause_all_scenarios.side_effect = pause_all
        mock_worker.resume_all_scenarios.side_effect = resume_all
        mock_worker.get_status_dict.return_value = {
            1: {"status": "paused" if 1 in paused_scens else "running", "iter": 150, "parameter_value": 800.0},
            2: {"status": "paused" if 2 in paused_scens else "running", "iter": 120, "parameter_value": 1200.0},
        }

        self.win.parallel_sweep_worker = mock_worker
        self.win.is_sweep_active = True

        # Stop the sweep: pauses all scenarios
        self.win.stop_sweep()
        self.assertFalse(self.win.is_sweep_active)
        self.assertTrue(self.win._sweep_stopped)
        self.assertIn(1, paused_scens)
        self.assertIn(2, paused_scens)
        self.assertIsNotNone(self.win.parallel_sweep_worker)
        self.assertEqual(self.win.btn_toggle.text(), "RESUME BEAM")

        # Select Scenario 2 in combo_sweep_scenario
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.assertEqual(self.win.grid_widgets[0]["V"].value(), 1200.0)

        # Resume with selected parameters
        with patch.object(self.win, "toggle_sim") as mock_toggle_sim:
            self.win.resume_sweep_with_selected()
            # Must NOT call toggle_sim, which would restart single sim from iter 0
            mock_toggle_sim.assert_not_called()

        self.assertTrue(self.win.is_sweep_active)
        self.assertFalse(self.win._sweep_stopped)
        self.assertNotIn(2, paused_scens)
        mock_worker.resume_all_scenarios.assert_called_once()
        mock_worker.set_active_view_scenario.assert_called_with(2)
        self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")
        self.assertTrue(self.win.btn_stop_sweep.isHidden())
        self.assertEqual(self.win.btn_resume_sweep.text(), "Pause Sweep")

        # Cleanup
        self.win.exit_sweep(confirm_choice="no")

    def test_exit_sweep_dialog_prompt_after_stop_sweep(self):
        """Verify exit_sweep prompts the user modal after stop_sweep with options Yes, No, exit, Cancel."""
        from unittest.mock import MagicMock
        from PyQt5.QtWidgets import QMessageBox
        cfg = {
            "param_id": "grid_1_V",
            "values": [800.0, 1200.0],
            "parallel": True,
            "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        mock_worker.get_status_dict.return_value = {
            1: {"status": "paused", "iter": 75, "parameter_value": 800.0},
            2: {"status": "paused", "iter": 50, "parameter_value": 1200.0},
        }
        self.win.parallel_sweep_worker = mock_worker
        self.win.is_sweep_active = True

        # Stop sweep
        self.win.stop_sweep()
        self.assertTrue(self.win._sweep_stopped)
        self.assertFalse(self.win.is_sweep_active)

        # 1. Trigger exit_sweep with Cancel simulated via QMessageBox (testing button click in UI)
        with patch('gui.app.sweep_mixin.QMessageBox.exec_') as mock_exec, \
             patch('gui.app.sweep_mixin.QMessageBox.clickedButton') as mock_clicked:
            mock_clicked.return_value = None  # Simulates pressing Cancel or closing dialog
            self.win.btn_exit_sweep.click()
            mock_exec.assert_called_once()
            # Remains in stopped sweep mode
            self.assertTrue(self.win._sweep_stopped)
            self.assertFalse(self.win.btn_exit_sweep.isHidden())

        # 1b. Trigger exit_sweep directly passing False (as Qt clicked(bool) would do)
        with patch('gui.app.sweep_mixin.QMessageBox.exec_') as mock_exec, \
             patch('gui.app.sweep_mixin.QMessageBox.clickedButton') as mock_clicked:
            mock_clicked.return_value = None
            self.win.exit_sweep(False)
            mock_exec.assert_called_once()
            self.assertTrue(self.win._sweep_stopped)
            self.assertFalse(self.win.btn_exit_sweep.isHidden())

        # 2. Trigger exit_sweep with "Yes" simulated
        with patch('gui.app.sweep_mixin.QMessageBox.exec_') as mock_exec, \
             patch.object(QMessageBox, 'clickedButton') as mock_clicked:
            # Re-call exit_sweep with confirm_choice="yes"
            self.win.exit_sweep(confirm_choice="yes")
            self.assertFalse(self.win._sweep_stopped)
            self.assertFalse(self.win.is_sweep_active)
            self.assertTrue(self.win.btn_exit_sweep.isHidden())
            self.assertTrue(hasattr(self.win, 'sweep_results_dialog'))
            self.assertEqual(len(self.win.sweep_results_dialog.results), 2)
            self.assertEqual(self.win.sweep_results_dialog.results[0]["steady_iterations"], 75)
            self.win.sweep_results_dialog.close()

    def test_sweep_build_rebuild_start_beam_sequence(self):
        """Verify the exact sequence:
        1. Build domain
        2. Arm a sweep on n0
        3. Build again the domain
        4. Start beam
        Must start beam without 'Build Domain first!' warning and populate sim.Ex.
        """
        # Step 1: Initial domain build
        self.win.build_domain()
        self.assertTrue(np.any(self.win.sim.Ex), "sim.Ex must be populated after build_domain")

        # Step 2: Choose a sweep on n0 (n0_plasma)
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (n0)",
            "param_unit": "m⁻³",
            "values": [1e17, 2e17, 3e17],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.assertTrue(self.win.sweep_mode_armed)

        # Step 3: Build again the domain
        self.win.build_domain()
        self.assertTrue(np.any(self.win.sim.Ex), "sim.Ex must remain populated after rebuild")
        self.assertIn("Sweep mode armed", self.win.lbl_sweep_badge.text())

        # Step 4: Start beam (starts only the CURRENT scenario)
        with patch("gui.app.sim_control.QMessageBox.warning") as mock_warn, \
             patch.object(self.win._sim_worker, "start_sim") as mock_start:
            self.win.toggle_sim()
            mock_warn.assert_not_called()
            # Left button starts beam only on current scenario, sweep mode remains armed
            self.assertFalse(self.win.is_sweep_active)
            self.assertTrue(self.win.sweep_mode_armed)
            self.assertTrue(self.win.sim_isRunning)
            self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")
            mock_start.assert_called_once()

            # Pause current scenario
            self.win.toggle_sim()
            self.assertFalse(self.win.sim_isRunning)
            self.assertEqual(self.win.btn_toggle.text(), "RESUME BEAM")

        # Step 5: Start sweep via top-bar Start Sweep button
        with patch.object(self.win._sim_worker, "start_sim") as mock_sweep_start:
            self.win.on_sweep_action_clicked()
            self.assertTrue(self.win.is_sweep_active)
            self.assertFalse(self.win.sweep_mode_armed)
            self.assertTrue(self.win.sim_isRunning)
            self.assertEqual(self.win.btn_resume_sweep.text(), "Pause Sweep")
            self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")

        # Stop simulation cleanly
        self.win.stop_sweep()

    def test_build_domain_during_simulation_warning(self):
        """Verify that clicking Build domain during simulation shows 'Stop the sweep before building' and aborts."""
        self.win.build_domain()
        self.win.sim_isRunning = True

        with patch("gui.app.sim_control.QMessageBox.warning") as mock_warn:
            self.win.build_domain()
            mock_warn.assert_called_once()
            args = mock_warn.call_args[0]
            self.assertEqual(args[1], "Warning")
            self.assertEqual(args[2], "Stop the sweep before building")

        self.win.sim_isRunning = False

    def test_multiple_build_domain_parameters_last_counts(self):
        """Verify that building multiple times with different parameters uses the last build."""
        self.win.inputs["n0_plasma"].setValue(1e17)
        self.win.build_domain()
        dx1 = self.win.sim.dx

        self.win.inputs["n0_plasma"].setValue(4e17)
        self.win.build_domain()
        dx2 = self.win.sim.dx

        # Debye length is shorter for higher density, so dx should be smaller
        self.assertNotEqual(dx1, dx2)
        params = self.win.get_params()
        self.assertEqual(params["n0_plasma"], 4e17)

    def test_lbl_build_domain_info_visibility_single_vs_sweep(self):
        """Verify lbl_build_domain_info is hidden in single mode and only visible in sweep mode."""
        # Initial single mode state
        self.assertTrue(self.win.lbl_build_domain_info.isHidden())

        # Single build domain: must NOT show the label
        self.win.build_domain()
        self.assertTrue(self.win.lbl_build_domain_info.isHidden())

        # Arm sweep: label must become visible
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (n0)",
            "param_unit": "m⁻³",
            "values": [1e17, 2e17],
            "parallel": False,
        }
        self.win.save_sweep_settings(cfg)
        self.assertFalse(self.win.lbl_build_domain_info.isHidden())

        # Rebuild domain during sweep: label remains visible
        self.win.build_domain()
        self.assertFalse(self.win.lbl_build_domain_info.isHidden())

        # Exit sweep: label must be hidden again
        self.win.exit_sweep(confirm_choice="no")
        self.assertTrue(self.win.lbl_build_domain_info.isHidden())

        # Subsequent single build: label remains hidden
        self.win.build_domain()
        self.assertTrue(self.win.lbl_build_domain_info.isHidden())

    def test_sweep_pause_resume_no_frozen_particle_ghosting(self):
        """Verify that scatter artists stay animated=True and _bg_live is reset on pause/resume."""
        self.win.build_domain()
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (n0)",
            "param_unit": "m⁻³",
            "values": [1e17, 2e17],
            "parallel": True,
            "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)

        # Mock parallel_sweep_worker
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        self.win.parallel_sweep_worker = mock_worker
        self.win.is_sweep_active = True
        self.win._sweep_stopped = False

        # Pre-set _bg_live to something
        self.win._bg_live = "dummy_background"

        # Pause sweep via on_sweep_action_clicked
        self.win.on_sweep_action_clicked()
        self.assertIsNone(self.win._bg_live, "_bg_live must be cleared on pause")
        self.assertTrue(self.win._sweep_stopped)

        # Resume sweep via on_sweep_action_clicked
        self.win._bg_live = "dummy_background_2"
        self.win.on_sweep_action_clicked()
        # _bg_live was cleared and then cleanly re-captured by _render_sweep_scenario_snapshot (not dummy_background_2)
        self.assertNotEqual(self.win._bg_live, "dummy_background_2")

        # Snapshot rendering must ensure scatter artists remain animated
        mock_worker.get_scenario_snapshot.return_value = {
            "scenario_index": 1,
            "t_us": 1.5,
            "iteration": 100,
            "transparency": 0.9,
            "divergence": 5.0,
            "prim_xy": np.array([[1.0, 2.0]], dtype=np.float32),
            "cex_xy": np.empty((0, 2), dtype=np.float32),
            "elec_xy": np.empty((0, 2), dtype=np.float32),
        }
        self.win._render_sweep_scenario_snapshot(1)
        self.assertTrue(self.win.scat_prim.get_animated(), "scat_prim must remain animated=True")
        self.assertTrue(self.win.scat_cex.get_animated(), "scat_cex must remain animated=True")
        self.assertTrue(self.win.scat_elec.get_animated(), "scat_elec must remain animated=True")

        self.win.exit_sweep(confirm_choice="no")

    def test_sweep_armed_left_button_and_start_sweep_progression(self):
        """Verify:
        1. When sweep is armed, lbl_beam_info is shown and left toggle controls only the selected scenario.
        2. Starting beam on left marks ONLY scenario 1 as Running in combo; others remain Queued.
        3. Pausing beam on left marks scenario 1 as Paused; others remain Queued.
        4. Clicking Start Sweep launches all scenarios in parallel with correct core-compatible labels.
        """
        # Step 1: Arm sweep with 3 scenarios
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (n0)",
            "param_unit": "m⁻³",
            "values": [1e17, 2e17, 3e17],
            "parallel": True,
            "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)
        self.assertTrue(self.win.sweep_mode_armed)
        self.assertFalse(self.win.lbl_beam_info.isHidden())
        self.assertIn("Start/pause applies only to current scenario", self.win.lbl_beam_info.text())

        # Step 2: Build domain
        self.win.build_domain()

        # Step 3: Click left Start Beam -> runs only current scenario (Scenario 1)
        with patch.object(self.win._sim_worker, "start_sim"):
            self.win.toggle_sim()
            self.assertTrue(self.win.sim_isRunning)
            self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")
            self.assertIn("[▶ Running", self.win.combo_sweep_scenario.itemText(0))
            self.assertIn("[⏳ Queued]", self.win.combo_sweep_scenario.itemText(1))
            self.assertIn("[⏳ Queued]", self.win.combo_sweep_scenario.itemText(2))

            # Step 4: Click left Pause Beam -> pauses only current scenario
            self.win.toggle_sim()
            self.assertFalse(self.win.sim_isRunning)
            self.assertEqual(self.win.btn_toggle.text(), "RESUME BEAM")
            self.assertIn("[⏸ Paused", self.win.combo_sweep_scenario.itemText(0))
            self.assertIn("[⏳ Queued]", self.win.combo_sweep_scenario.itemText(1))
            self.assertIn("[⏳ Queued]", self.win.combo_sweep_scenario.itemText(2))

        # Step 5: Click top 'Start Sweep' -> starts sweep across all scenarios
        with patch.object(self.win, "_start_parallel_sweep") as mock_parallel:
            self.win.on_sweep_action_clicked()
            mock_parallel.assert_called_once()

        self.win.exit_sweep(confirm_choice="no")
        self.assertTrue(self.win.lbl_beam_info.isHidden())

    def test_start_sweep_requires_build_domain_first(self):
        """Verify that clicking Start Sweep before Build Domain shows 'Attention: build domain first' and does not start."""
        from PyQt5.QtWidgets import QMessageBox

        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (n0)",
            "param_unit": "m⁻³",
            "values": [1e17, 2e17],
            "parallel": True,
            "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)
        self.assertTrue(self.win.sweep_mode_armed)
        self.assertFalse(self.win.is_sweep_active)

        # 1. Attempt to start sweep directly before build_domain
        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.on_sweep_action_clicked()
            mock_warn.assert_called_once_with(self.win, "Attention", "Attention: build domain first")
            self.assertFalse(self.win.is_sweep_active)
            self.assertTrue(self.win.sweep_mode_armed)

        # 2. Build domain
        self.win.build_domain()
        self.assertTrue(self.win._sweep_domain_built)

        # 3. Now start sweep -> succeeds
        with patch.object(self.win, "_start_parallel_sweep") as mock_parallel, \
             patch.object(QMessageBox, "warning") as mock_warn:
            self.win.on_sweep_action_clicked()
            mock_warn.assert_not_called()
            mock_parallel.assert_called_once()

        self.win.exit_sweep(confirm_choice="no")

    def test_ppc_window_live_and_post_run_in_sweep_mode(self):
        """Verify PPC window plots live snapshots and post-run completed scenarios in sweep mode without modifying PPCWindow."""
        self.win.build_domain()
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (n0)",
            "param_unit": "m⁻³",
            "values": [1e17, 2e17],
            "parallel": False,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open PPC Window
        self.win.open_ppc_window()
        self.assertIsNotNone(self.win.ppc_window)
        self.assertTrue(self.win.ppc_window.isVisible())

        # 1. Simulate live snapshot for Scenario 1
        ny, nx = self.win.sim.ny, self.win.sim.nx
        ppc_s1 = np.zeros((ny, nx), dtype=int)
        ppc_s1[10:20, 10:20] = 5  # 100 active cells with PPC=5
        info_s1 = {
            "status": "running",
            "iter": 50,
            "parameter_value": 1e17,
            "current_ppc_map": ppc_s1,
            "accum_ppc_map": ppc_s1.astype(float),
            "ppc_steps_count": 1,
            "Lx": self.win.sim.Lx,
            "Ly": self.win.sim.Ly,
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.ppc_window.windowTitle())
        self.assertIn("Active Cells: 100", self.win.ppc_window.lbl_stats.text())

        # 2. Simulate switching to Scenario 2 with different post-run completed data
        ppc_s2 = np.zeros((ny, nx), dtype=int)
        ppc_s2[5:10, 5:10] = 2  # 25 active cells with PPC=2 (Low PPC < 3)
        info_s2 = {
            "status": "done",
            "iter": 120,
            "parameter_value": 2e17,
            "current_ppc_map": ppc_s2,
            "accum_ppc_map": ppc_s2.astype(float),
            "ppc_steps_count": 10,
            "Lx": self.win.sim.Lx,
            "Ly": self.win.sim.Ly,
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.ppc_window.windowTitle())
        self.assertIn("Active Cells: 25", self.win.ppc_window.lbl_stats.text())
        self.assertIn("Low PPC (< 3): 25 (100.0%)", self.win.ppc_window.lbl_stats.text())

        # 3. Simulate user closing PPC window, switching scenario, then re-opening
        self.win.ppc_window.close()
        self.assertFalse(self.win.ppc_window.isVisible())

        # Switch to scenario 1
        self.win.combo_sweep_scenario.setCurrentIndex(0)

        # User re-opens PPC window
        self.win.open_ppc_window()
        self.assertTrue(self.win.ppc_window.isVisible())
        self.assertIn("Scenario 1", self.win.ppc_window.windowTitle())
        self.assertIn("Active Cells: 100", self.win.ppc_window.lbl_stats.text())

        # Mode switching works without crash
        self.win.ppc_window.combo_mode.setCurrentIndex(1)  # Instantaneous
        self.assertIn("Active Cells: 100", self.win.ppc_window.lbl_stats.text())
        self.win.ppc_window.combo_mode.setCurrentIndex(2)  # Low PPC Mask
        self.assertIn("Active Cells: 100", self.win.ppc_window.lbl_stats.text())
        self.win.ppc_window.combo_mode.setCurrentIndex(0)  # Time-Averaged

        # 4. Clean exit
        self.win.ppc_window.close()
        self.win.exit_sweep(confirm_choice="no")

    def test_energy_window_live_and_post_run_in_sweep_mode(self):
        """Verify Total Energy Conservation monitor window functions in sweep mode (live & post-run)."""
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (m^-3)",
            "param_unit": "m^-3",
            "values": [1e17, 2e17],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open Physical Constraints / Total Energy window
        self.win.open_phys_window()
        self.assertIsNotNone(self.win.phys_window)
        self.assertTrue(self.win.phys_window.isVisible())

        # 1. Simulate Scenario 1 live status
        info_s1 = {
            "status": "running",
            "iter": 50,
            "parameter_value": 1e17,
            "t_sim_us": 0.05,
            "energy_history_t": [0.0, 2.5e-8, 5.0e-8],
            "energy_history_ke_i": [1e-4, 1.1e-4, 1.2e-4],
            "energy_history_ke_e": [5e-5, 5.1e-5, 5.2e-5],
            "energy_history_fe": [2e-4, 1.9e-4, 1.8e-4],
            "energy_history_tot": [3.5e-4, 3.51e-4, 3.52e-4],
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.phys_window.windowTitle())
        self.assertIn("Total Energy:", self.win.phys_window.lbl_stats.text())

        # 2. Simulate switching to Scenario 2 with completed post-run data
        info_s2 = {
            "status": "done",
            "iter": 100,
            "parameter_value": 2e17,
            "t_sim_us": 0.10,
            "energy_history_t": [0.0, 5.0e-8, 1.0e-7],
            "energy_history_ke_i": [2e-4, 2.2e-4, 2.4e-4],
            "energy_history_ke_e": [1e-4, 1.05e-4, 1.1e-4],
            "energy_history_fe": [4e-4, 3.8e-4, 3.7e-4],
            "energy_history_tot": [7.0e-4, 7.05e-4, 7.2e-4],
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.phys_window.windowTitle())
        self.assertIn("Total Energy:", self.win.phys_window.lbl_stats.text())

        # 3. Simulate closing and re-opening window via menu
        self.win.phys_window.close()
        self.assertFalse(self.win.phys_window.isVisible())

        self.win.open_phys_window()
        self.assertTrue(self.win.phys_window.isVisible())
        self.assertIn("Scenario 2", self.win.phys_window.windowTitle())

        # Toggle checkboxes without error
        self.win.phys_window.chk_normalize.setChecked(True)
        self.win.phys_window.chk_normalize.setChecked(False)

        # 4. Clean exit
        self.win.phys_window.close()
        self.win.exit_sweep(confirm_choice="no")
        self.assertEqual(self.win.phys_window.windowTitle(), "Total Energy Conservation Monitor")

    def test_mesh_window_live_and_post_run_in_sweep_mode(self):
        """Verify Computational Domain & Mesh Grid inspector and Substeps monitor in sweep mode."""
        cfg = {
            "param_id": "grid_1_gap",
            "param_label": "Grid 1-2 Gap (mm)",
            "param_unit": "mm",
            "values": [1.5, 2.5],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open Mesh window
        self.win.open_mesh_window()
        self.assertIsNotNone(self.win.mesh_window)
        self.assertTrue(self.win.mesh_window.isVisible())

        # 1. Simulate Scenario 1 live status with substep history
        info_s1 = {
            "status": "running",
            "iter": 30,
            "parameter_value": 1.5,
            "substep_history_iter": [10, 20, 30],
            "substep_history_ion": [1, 1, 2],
            "substep_history_e": [1, 2, 4],
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.mesh_window.windowTitle())
        self.assertIn("Domain (Lx × Ly):", self.win.mesh_window.lbl_domain_size.text())

        # 2. Simulate switching to Scenario 2
        info_s2 = {
            "status": "done",
            "iter": 60,
            "parameter_value": 2.5,
            "substep_history_iter": [10, 20, 30, 40, 50, 60],
            "substep_history_ion": [1, 1, 1, 1, 1, 1],
            "substep_history_e": [1, 1, 2, 1, 2, 1],
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.mesh_window.windowTitle())

        # 3. Test Displacement Substeps floating window
        self.win.open_substep_window()
        self.assertIsNotNone(self.win.substep_window)
        self.assertTrue(self.win.substep_window.isVisible())
        self.assertIn("Scenario 2", self.win.substep_window.windowTitle())

        # 4. Clean exit restores window titles
        self.win.mesh_window.close()
        self.win.substep_window.close()
        self.win.exit_sweep(confirm_choice="no")
        self.assertEqual(self.win.mesh_window.windowTitle(), "Computational Domain & Mesh Grid Inspector")
        self.assertEqual(self.win.substep_window.windowTitle(), "Displacement Substeps Monitor — (Iterations vs Required Substeps)")

    def test_perf_window_live_and_post_run_in_sweep_mode(self):
        """Verify Runtime Performance & Profiling monitor window in sweep mode (live & post-run)."""
        from diagnostics.performance_monitor import StepDiagnostics

        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (m^-3)",
            "param_unit": "m^-3",
            "values": [1e17, 2e17],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open Performance window
        self.win.open_perf_window()
        self.assertIsNotNone(self.win.perf_window)
        self.assertTrue(self.win.perf_window.isVisible())

        # 1. Simulate Scenario 1 live status
        diag1 = StepDiagnostics(iteration=25, sim_time_s=2.5e-8, num_ions=100, num_electrons=50, wall_time_ms=2.0, memory_rss_mb=120.0)
        diag2 = StepDiagnostics(iteration=50, sim_time_s=5.0e-8, num_ions=200, num_electrons=100, wall_time_ms=2.1, memory_rss_mb=125.0)
        info_s1 = {
            "status": "running",
            "iter": 50,
            "parameter_value": 1e17,
            "t_sim_us": 0.05,
            "perf_history": [diag1, diag2],
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.perf_window.windowTitle())
        self.assertIn("Iter: 50", self.win.perf_window.lbl_stats.text())
        self.assertIn("Ions: 200", self.win.perf_window.lbl_stats.text())

        # 2. Simulate switching to Scenario 2 with post-run completed data
        diag3 = StepDiagnostics(iteration=100, sim_time_s=1.0e-7, num_ions=400, num_electrons=200, wall_time_ms=3.0, memory_rss_mb=150.0)
        info_s2 = {
            "status": "done",
            "iter": 100,
            "parameter_value": 2e17,
            "t_sim_us": 0.10,
            "perf_history": [diag1, diag2, diag3],
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.perf_window.windowTitle())
        self.assertIn("Iter: 100", self.win.perf_window.lbl_stats.text())
        self.assertIn("Ions: 400", self.win.perf_window.lbl_stats.text())

        # 3. Re-opening via menu refreshes to currently selected scenario
        self.win.perf_window.close()
        self.assertFalse(self.win.perf_window.isVisible())
        self.win.open_perf_window()
        self.assertTrue(self.win.perf_window.isVisible())
        self.assertIn("Scenario 2", self.win.perf_window.windowTitle())

        # 4. Clean exit
        self.win.perf_window.close()
        self.win.exit_sweep(confirm_choice="no")
        self.assertEqual(self.win.perf_window.windowTitle(), "Runtime Performance & Particle Diagnostics Monitor")

    def test_density_window_live_and_post_run_in_sweep_mode(self):
        """Verify Plasma & Particle Density Diagnostics window in sweep mode (live & post-run)."""
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (m^-3)",
            "param_unit": "m^-3",
            "values": [1e17, 2e17],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open Density window
        self.win.open_density_window()
        self.assertIsNotNone(self.win.density_window)
        self.assertTrue(self.win.density_window.isVisible())

        # 1. Simulate Scenario 1 live status with particles
        prim_s1 = np.array([[0.5, 0.5], [1.0, 0.8], [1.5, 1.0]], dtype=np.float32)
        cex_s1 = np.array([[1.2, 0.6]], dtype=np.float32)
        elec_s1 = np.array([[0.2, 0.3], [0.4, 0.4]], dtype=np.float32)

        info_s1 = {
            "status": "running",
            "iter": 40,
            "parameter_value": 1e17,
            "t_sim_us": 0.04,
            "macro_weight": 1e8,
            "prim_xy": prim_s1,
            "cex_xy": cex_s1,
            "elec_xy": elec_s1,
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.density_window.windowTitle())
        self.assertIn("Total Ion Density", self.win.density_window.lbl_stats.text())

        # Switch species dropdown to Electron Density
        self.win.density_window.combo_species.setCurrentIndex(3)
        self.assertIn("Electron Density", self.win.density_window.lbl_stats.text())

        # 2. Simulate switching to Scenario 2 with completed post-run data
        prim_s2 = np.array([[0.6, 0.7], [1.1, 0.9]], dtype=np.float32)
        info_s2 = {
            "status": "done",
            "iter": 80,
            "parameter_value": 2e17,
            "t_sim_us": 0.08,
            "macro_weight": 2e8,
            "prim_xy": prim_s2,
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.density_window.windowTitle())

        # 3. Re-opening via menu refreshes to currently selected scenario
        self.win.density_window.close()
        self.assertFalse(self.win.density_window.isVisible())
        self.win.open_density_window()
        self.assertTrue(self.win.density_window.isVisible())
        self.assertIn("Scenario 2", self.win.density_window.windowTitle())

        # 4. Clean exit
        self.win.density_window.close()
        self.win.exit_sweep(confirm_choice="no")
        self.assertEqual(self.win.density_window.windowTitle(), "Plasma & Particle Density Diagnostics")

    def test_charge_window_live_and_post_run_in_sweep_mode(self):
        """Verify Total Charge Monitor window in sweep mode (live & post-run)."""
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (m^-3)",
            "param_unit": "m^-3",
            "values": [1e17, 2e17],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open Total Charge window
        self.win.open_charge_window()
        self.assertIsNotNone(self.win.charge_window)
        self.assertTrue(self.win.charge_window.isVisible())

        # 1. Live status for Scenario 1
        t1 = [0.0, 1e-7, 2e-7, 3e-7]
        qi1 = [0.0, 1e-10, 2e-10, 3e-10]
        qe1 = [0.0, -0.5e-10, -1e-10, -1.5e-10]
        qb1 = [0.0, 0.0, 0.0, 0.0]
        qfree1 = [0.0, 0.5e-10, 1e-10, 1.5e-10]
        qnet1 = [0.0, 0.5e-10, 1e-10, 1.5e-10]

        info_s1 = {
            "status": "running",
            "iter": 3,
            "parameter_value": 1e17,
            "charge_history_t": t1,
            "charge_history_q_i": qi1,
            "charge_history_q_e": qe1,
            "charge_history_q_b": qb1,
            "charge_history_q_free": qfree1,
            "charge_history_q_net": qnet1,
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.charge_window.windowTitle())
        self.assertIn("+150.00 pC", self.win.charge_window.lbl_stats.text())

        # 2. Switch to Scenario 2 (post-run done)
        t2 = [0.0, 1e-7, 2e-7, 3e-7, 4e-7]
        qi2 = [0.0, 2e-10, 4e-10, 6e-10, 8e-10]
        qe2 = [0.0, -1e-10, -2e-10, -3e-10, -4e-10]
        qb2 = [0.0, 0.0, 0.0, 0.0, 0.0]
        qfree2 = [0.0, 1e-10, 2e-10, 3e-10, 4e-10]
        qnet2 = [0.0, 1e-10, 2e-10, 3e-10, 4e-10]

        info_s2 = {
            "status": "done",
            "iter": 5,
            "parameter_value": 2e17,
            "charge_history_t": t2,
            "charge_history_q_i": qi2,
            "charge_history_q_e": qe2,
            "charge_history_q_b": qb2,
            "charge_history_q_free": qfree2,
            "charge_history_q_net": qnet2,
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.charge_window.windowTitle())
        self.assertIn("+400.00 pC", self.win.charge_window.lbl_stats.text())

        # 3. Re-open via menu refreshes to currently selected scenario (Scenario 2)
        self.win.charge_window.close()
        self.assertFalse(self.win.charge_window.isVisible())
        self.win.open_charge_window()
        self.assertTrue(self.win.charge_window.isVisible())
        self.assertIn("Scenario 2", self.win.charge_window.windowTitle())
        self.assertIn("+400.00 pC", self.win.charge_window.lbl_stats.text())

        # 4. Clean exit
        self.win.charge_window.close()
        self.win.exit_sweep(confirm_choice="no")
        self.assertEqual(self.win.charge_window.windowTitle(), "Total Charge Monitor")

    def test_iedf_window_live_and_post_run_in_sweep_mode(self):
        """Verify Energy Distribution Function (IEDF & EEDF) window in sweep mode (live & post-run)."""
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (m^-3)",
            "param_unit": "m^-3",
            "values": [1e17, 2e17],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open IEDF window
        self.win.open_iedf_window()
        self.assertIsNotNone(self.win.iedf_window)
        self.assertTrue(self.win.iedf_window.isVisible())

        # 1. Live status for Scenario 1
        e_prim1 = np.array([500.0, 520.0, 510.0], dtype=np.float32)
        e_cex1  = np.array([50.0, 70.0], dtype=np.float32)
        elec_xy1 = np.array([[1.0, 0.5], [5.0, 0.5]], dtype=np.float32)
        e_elec1  = np.array([2.5, 3.5], dtype=np.float32)

        info_s1 = {
            "status": "running",
            "iter": 30,
            "parameter_value": 1e17,
            "prim_xy": np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]], dtype=np.float32),
            "e_prim_eV": e_prim1,
            "cex_xy": np.array([[1.2, 0.6], [1.3, 0.7]], dtype=np.float32),
            "e_cex_eV": e_cex1,
            "elec_xy": elec_xy1,
            "e_elec_eV": e_elec1,
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.iedf_window.windowTitle())
        self.assertIn("Ion Energy Distribution", self.win.iedf_window.ax.get_title())

        # Switch to Electrons in combo_type
        self.win.iedf_window.combo_type.setCurrentIndex(3)  # "All Electrons"
        self.assertIn("Electron Energy Distribution (All)", self.win.iedf_window.ax.get_title())

        # Switch to Grid SEE
        self.win.iedf_window.combo_type.setCurrentIndex(4)  # SEE <= 4mm
        self.assertIn("Grid/SEE Zone", self.win.iedf_window.ax.get_title())

        # 2. Switch to Scenario 2 (post-run done)
        e_prim2 = np.array([600.0, 610.0], dtype=np.float32)
        e_cex2  = np.empty(0, dtype=np.float32)
        info_s2 = {
            "status": "done",
            "iter": 60,
            "parameter_value": 2e17,
            "prim_xy": np.array([[0.8, 0.8], [0.9, 0.9]], dtype=np.float32),
            "e_prim_eV": e_prim2,
            "cex_xy": np.empty((0, 2), dtype=np.float32),
            "e_cex_eV": e_cex2,
            "elec_xy": np.empty((0, 2), dtype=np.float32),
            "e_elec_eV": np.empty(0, dtype=np.float32),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.iedf_window.windowTitle())

        # 3. Re-open via menu refreshes to currently selected scenario (Scenario 2)
        self.win.iedf_window.close()
        self.assertFalse(self.win.iedf_window.isVisible())
        self.win.open_iedf_window()
        self.assertTrue(self.win.iedf_window.isVisible())
        self.assertIn("Scenario 2", self.win.iedf_window.windowTitle())

        # 4. Clean exit
        self.win.iedf_window.close()
        self.win.exit_sweep(confirm_choice="no")
        self.assertEqual(self.win.iedf_window.windowTitle(), "Energy Distribution Function (IEDF & EEDF)")

    def test_field_window_live_and_post_run_in_sweep_mode(self):
        """Verify Electric Field & Potential Diagnostics window in sweep mode (live & post-run)."""
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (m^-3)",
            "param_unit": "m^-3",
            "values": [1e17, 2e17],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open Field window
        self.win.open_field_window()
        self.assertIsNotNone(self.win.field_window)
        self.assertTrue(self.win.field_window.isVisible())

        # 1. Live status for Scenario 1
        ny, nx = self.win.sim.ny, self.win.sim.nx
        V1 = np.ones((ny, nx), dtype=np.float32) * 500.0
        V1[:, -1] = 0.0

        info_s1 = {
            "status": "running",
            "iter": 25,
            "parameter_value": 1e17,
            "V": V1,
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.field_window.windowTitle())
        self.assertIn("Peak |E| (total):", self.win.field_window.lbl_e_max.text())
        self.assertIsNotNone(self.win.sim.Ex)
        self.assertIsNotNone(self.win.sim.Ey)
        # Calling get_total_energy must not crash with TypeError
        self.win.sim.get_total_energy()

        # 2. Switch to Scenario 2 (post-run done)
        V2 = np.ones((ny, nx), dtype=np.float32) * 1000.0
        V2[:, -1] = -200.0
        info_s2 = {
            "status": "done",
            "iter": 50,
            "parameter_value": 2e17,
            "V": V2,
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.field_window.windowTitle())
        self.assertIsNotNone(self.win.sim.Ex)
        self.assertIsNotNone(self.win.sim.Ey)
        self.win.sim.get_total_energy()

        # 3. Re-open via menu refreshes to currently selected scenario (Scenario 2)
        self.win.field_window.close()
        self.assertFalse(self.win.field_window.isVisible())
        self.win.open_field_window()
        self.assertTrue(self.win.field_window.isVisible())
        self.assertIn("Scenario 2", self.win.field_window.windowTitle())

        # 4. Clean exit
        self.win.field_window.close()
        self.win.exit_sweep(confirm_choice="no")
        self.assertEqual(self.win.field_window.windowTitle(), "Electric Field & Potential Diagnostics")

    def test_steady_state_window_live_and_post_run_in_sweep_mode(self):
        """Verify Steady State Diagnostics window in sweep mode (live & post-run)."""
        cfg = {
            "param_id": "n0_plasma",
            "param_label": "Plasma Density (m^-3)",
            "param_unit": "m^-3",
            "values": [1e17, 2e17],
            "parallel": False,
            "window": 50,
            "threshold": 1.0,
            "max_iterations": 5000,
            "min_iterations": 500,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()

        # Open Steady State window
        self.win.open_steady_state_window()
        self.assertIsNotNone(self.win.steady_state_window)
        self.assertTrue(self.win.steady_state_window.isVisible())

        # 1. Live status for Scenario 1
        N = 60
        e_tot1 = [1e-6 * (1.0 + 0.05 * np.exp(-i / 10.0)) for i in range(N)]
        q_net1 = [1e-11 * (1.0 + 0.02 * np.exp(-i / 10.0)) for i in range(N)]
        fe1 = [5e-7 * (1.0 + 0.01 * np.exp(-i / 10.0)) for i in range(N)]
        ebs1 = [150.0 * (1.0 - 0.03 * np.exp(-i / 10.0)) for i in range(N)]
        i_exit1 = [0.05 * (1.0 - 0.02 * np.exp(-i / 10.0)) for i in range(N)]
        div1 = [12.0 * (1.0 + 0.01 * np.exp(-i / 10.0)) for i in range(N)]
        parts1 = [500.0 * (1.0 + 0.01 * np.exp(-i / 10.0)) for i in range(N)]

        info_s1 = {
            "status": "running",
            "iter": N,
            "parameter_value": 1e17,
            "energy_history_tot": e_tot1,
            "charge_history_q_net": q_net1,
            "energy_history_fe": fe1,
            "ebs_history": ebs1,
            "ion_current_exit_history": i_exit1,
            "div_history": div1,
            "transmitted_ions_history": parts1,
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict = {1: info_s1}
        self.win._render_sweep_scenario_snapshot(1)

        self.assertIn("Scenario 1", self.win.steady_state_window.windowTitle())
        self.assertIn("TRANSIENT / DRIFTING", self.win.steady_state_window.lbl_status_text.text())

        # 2. Switch to Scenario 2 (post-run done with flat stationary values)
        info_s2 = {
            "status": "done",
            "iter": N,
            "parameter_value": 2e17,
            "energy_history_tot": [1.5e-6] * N,
            "charge_history_q_net": [1.5e-11] * N,
            "energy_history_fe": [7.5e-7] * N,
            "ebs_history": [180.0] * N,
            "ion_current_exit_history": [0.08] * N,
            "div_history": [12.0] * N,
            "transmitted_ions_history": [750.0] * N,
            "prim_xy": np.empty((0, 2)),
            "cex_xy": np.empty((0, 2)),
            "elec_xy": np.empty((0, 2)),
        }
        self.win._sweep_status_dict[2] = info_s2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._render_sweep_scenario_snapshot(2)

        self.assertIn("Scenario 2", self.win.steady_state_window.windowTitle())
        self.assertIn("STEADY STATE ACHIEVED", self.win.steady_state_window.lbl_status_text.text())

        # 3. Re-open via menu refreshes to currently selected scenario (Scenario 2)
        self.win.steady_state_window.close()
        self.assertFalse(self.win.steady_state_window.isVisible())
        self.win.open_steady_state_window()
        self.assertTrue(self.win.steady_state_window.isVisible())
        self.assertIn("Scenario 2", self.win.steady_state_window.windowTitle())

        # 4. Clean exit
        self.win.steady_state_window.close()
        self.win.exit_sweep(confirm_choice="no")
        self.assertEqual(self.win.steady_state_window.windowTitle(), "Steady State Diagnostics — 50-Iteration Variation")

    def test_compute_energy_budget_defensive_against_none_efield(self):
        """Verify compute_energy_budget does not crash with TypeError if sim.Ex or sim.Ey is None."""
        from diagnostics.metrics import compute_energy_budget
        self.win.build_domain()
        self.win.sim.Ex = None
        self.win.sim.Ey = None
        budget = compute_energy_budget(self.win.sim)
        self.assertIsInstance(budget, dict)
        self.assertIn("field_J", budget)
        self.assertIsNotNone(self.win.sim.Ex)
        self.assertIsNotNone(self.win.sim.Ey)

    def test_upstream_gap_sweep_gui_preview_and_sequential_transition(self):
        """Verify sweeping upstream_gap_mm updates GUI inputs, domain geometry, Lx, and auto neut_x consistently."""
        from PyQt5.QtWidgets import QMessageBox

        # Start with auto-tracked neut_x (not manually locked to config.json's neut_x)
        self.win._last_auto_neut_x = round(self.win.inputs["neut_x"].value(), 3)

        cfg = {
            "param_id": "upstream_gap_mm",
            "param_label": "Upstream Gap (mm)",
            "param_unit": "mm",
            "start": 1.255,
            "end": 0.455,
            "n_sims": 2,
            "scale": "linear",
            "values": [1.255, 0.455],
            "window": 10,
            "threshold": 5.0,
            "max_iterations": 20,
            "min_iterations": 10,
            "parallel": False,
            "n_cores": 1,
        }

        self.win.save_sweep_settings(cfg)
        self.assertAlmostEqual(self.win.inputs["upstream_gap_mm"].value(), 1.255, places=3)

        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.build_domain(force=True)
            mock_warn.assert_not_called()

        self.assertAlmostEqual(self.win.sim.upstream_gap_mm, 1.255, places=3)
        self.assertAlmostEqual(self.win.sim.grid_x_starts[0], 1.255, places=3)
        lx_1 = self.win.sim.Lx
        expected_neut_1 = round(
            self.win.sim.grid_x_ends[-1] + 0.9 * (lx_1 - self.win.sim.grid_x_ends[-1]), 3
        )
        self.assertAlmostEqual(self.win.inputs["neut_x"].value(), expected_neut_1, places=3)

        # Preview Scenario 2 (smaller gap = 0.455 mm): domain geometry & neut_x shrink cleanly
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._on_sweep_scenario_selected(1)
        self.assertAlmostEqual(self.win.inputs["upstream_gap_mm"].value(), 0.455, places=3)
        d2 = self.win._get_scenario_domain_data(2)
        self.assertAlmostEqual(d2["upstream_gap_mm"], 0.455, places=3)
        self.assertAlmostEqual(d2["grid_x_starts"][0], 0.455, places=3)
        self.assertAlmostEqual(lx_1 - d2["Lx"], 0.8, places=3)
        self.assertAlmostEqual(self.win.ax_live.get_xlim()[1], d2["Lx"], places=3)
        expected_neut_2 = round(
            d2["grid_x_ends"][-1] + 0.9 * (d2["Lx"] - d2["grid_x_ends"][-1]), 3
        )
        self.assertAlmostEqual(d2["params"]["neut_x"], expected_neut_2, places=3)
        self.assertAlmostEqual(self.win.inputs["neut_x"].value(), expected_neut_2, places=3)

        # Switch back to Scenario 1 and verify sequential advance_sweep_scenario downward transition
        self.win.combo_sweep_scenario.setCurrentIndex(0)
        self.win._on_sweep_scenario_selected(0)
        self.win.sweep_current_idx = 0
        self.win.is_sweep_active = True
        self.win.sweep_mode_armed = False

        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.advance_sweep_scenario()
            mock_warn.assert_not_called()

        self.assertAlmostEqual(self.win.sim.upstream_gap_mm, 0.455, places=3)
        self.assertAlmostEqual(self.win.sim.grid_x_starts[0], 0.455, places=3)
        self.assertAlmostEqual(lx_1 - self.win.sim.Lx, 0.8, places=3)
        self.assertAlmostEqual(self.win._last_auto_neut_x, expected_neut_2, places=3)
        self.assertAlmostEqual(self.win.inputs["neut_x"].value(), expected_neut_2, places=3)
        self.win.exit_sweep(confirm_choice="no")

    def test_build_domain_validates_all_grids_uniformly(self):
        """Verify SimControlMixin.build_domain blocks invalid geometry on any grid 1..N with QMessageBox."""
        n_before = len(self.win.grid_widgets)
        self.win.btn_add_grid.click()
        self.assertEqual(len(self.win.grid_widgets), n_before + 1)
        new_gw = self.win.grid_widgets[-1]
        self.assertEqual(new_gw["t"].value(), 0.0)
        self.assertEqual(new_gw["r"].value(), 0.0)

        # 1. Newly added zeroed Grid N+1 must fail validation without mutating domain readiness
        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.build_domain()
            mock_warn.assert_called_once()
            warn_msg = mock_warn.call_args[0][2]
            self.assertIn(f"Grid {n_before + 1}", warn_msg)
            self.assertFalse(self.win._sweep_domain_built)

        # 2. Give new grid valid geometry, but break Grid 2 internal gap & Grid 3 aperture radius
        new_gw["V"].setValue(0.0)
        new_gw["t"].setValue(0.38)
        new_gw["gap"].setValue(1.5)
        new_gw["r"].setValue(0.75)
        new_gw["cham"].setValue(0.0)
        orig_g2_gap = self.win.grid_widgets[1]["gap"].value()
        orig_g3_r = self.win.grid_widgets[2]["r"].value()
        self.win.grid_widgets[1]["gap"].setValue(0.0)
        self.win.grid_widgets[2]["r"].setValue(0.0)

        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.build_domain()
            mock_warn.assert_called_once()
            warn_msg = mock_warn.call_args[0][2]
            self.assertIn("Grid 2", warn_msg)
            self.assertIn("Grid 3", warn_msg)

        # 3. Restore valid values across all grids -> build succeeds cleanly
        self.win.grid_widgets[1]["gap"].setValue(orig_g2_gap)
        self.win.grid_widgets[2]["r"].setValue(orig_g3_r)
        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.build_domain()
            mock_warn.assert_not_called()
        self.assertTrue(self.win._sweep_domain_built)
        self.assertFalse(self.win._grid_count_modified_since_build)
        self.assertEqual(len(self.win.sim.grids), n_before + 1)

    def test_add_remove_grid_requires_build_domain_before_sweep_and_blocks_discrepancies(self):
        """Verify adding/removing grids enforces build-domain-first and blocks deleted-grid sweeps."""
        self.win.build_domain()
        self.assertFalse(self.win._grid_count_modified_since_build)

        # 1. Removing a grid marks domain stale -> open_sweep_dialog and save_sweep_settings block until rebuild
        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.btn_remove_grid.click()
            mock_warn.assert_not_called()
        self.assertTrue(self.win._grid_count_modified_since_build)
        self.assertFalse(self.win._sweep_domain_built)

        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.open_sweep_dialog()
            mock_warn.assert_called_once()
            self.assertIn("1. BUILD DOMAIN", mock_warn.call_args[0][2])

        n_grids_now = len(self.win.grid_widgets)
        cfg_grid2 = {
            "param_id": f"grid_{n_grids_now}_cham",
            "param_label": f"Grid {n_grids_now} Chamfer (°)",
            "param_unit": "°",
            "values": [0.0, 15.0],
            "parallel": True,
            "n_cores": 2,
        }
        with patch.object(QMessageBox, "warning") as mock_warn:
            saved = self.win.save_sweep_settings(cfg_grid2)
            mock_warn.assert_called_once()
            self.assertFalse(saved)
            self.assertIsNone(self.win.sweep_config)

        # 2. Build domain first -> sweep settings accepted
        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.build_domain()
            mock_warn.assert_not_called()
        self.assertFalse(self.win._grid_count_modified_since_build)

        self.win.save_sweep_settings(cfg_grid2)
        self.assertTrue(self.win.sweep_mode_armed)
        self.assertEqual(self.win.sweep_config["param_id"], f"grid_{n_grids_now}_cham")

        # 3. Removing the targeted grid immediately warns and disarms the invalid sweep target
        if n_grids_now > 1:
            with patch.object(QMessageBox, "warning") as mock_warn:
                self.win.btn_remove_grid.click()
                mock_warn.assert_called_once()
                self.assertIn("removed", mock_warn.call_args[0][2].lower())
            self.assertIsNone(self.win.sweep_config)
            self.assertFalse(self.win.sweep_mode_armed)

        # 4. Arm valid global sweep after rebuilding, then add another grid without rebuilding -> Start Sweep blocked
        self.win.grid_widgets[-1]["t"].setValue(0.38)
        self.win.grid_widgets[-1]["r"].setValue(0.75)
        self.win.grid_widgets[-1]["gap"].setValue(1.5)
        self.win.build_domain()
        cfg_global = {
            "param_id": "Te_up", "param_label": "Te_up (eV)", "param_unit": "eV",
            "values": [2.0, 4.0], "parallel": True, "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg_global)
        self.win.build_domain()

        self.win.btn_add_grid.click()
        self.win.grid_widgets[-1]["t"].setValue(0.38)
        self.win.grid_widgets[-1]["r"].setValue(0.75)
        self.win.grid_widgets[-1]["gap"].setValue(1.5)
        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.on_sweep_action_clicked()
            mock_warn.assert_called_once()
            self.assertFalse(self.win.is_sweep_active)

        self.win.build_domain()
        with patch.object(self.win, "_start_parallel_sweep") as mock_parallel, \
             patch.object(QMessageBox, "warning") as mock_warn:
            self.win.on_sweep_action_clicked()
            mock_warn.assert_not_called()
            mock_parallel.assert_called_once()
        self.win.exit_sweep(confirm_choice="no")

    def test_grid_edit_buttons_disabled_during_active_sweep(self):
        """Verify + Add Grid and - Remove Grid lock while sweeping and unlock on stop/finish/exit."""
        cfg = {
            "param_id": "Te_up", "param_label": "Te_up (eV)", "param_unit": "eV",
            "values": [2.0, 4.0], "parallel": True, "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)
        self.win.build_domain()
        self.assertTrue(self.win.btn_add_grid.isEnabled())
        self.assertTrue(self.win.btn_remove_grid.isEnabled())

        with patch.object(self.win, "_start_parallel_sweep"):
            self.win.start_sweep(cfg)
        self.assertTrue(self.win.is_sweep_active)
        self.assertFalse(self.win.btn_add_grid.isEnabled())
        self.assertFalse(self.win.btn_remove_grid.isEnabled())

        self.win.stop_sweep()
        self.assertTrue(self.win.btn_add_grid.isEnabled())
        self.assertTrue(self.win.btn_remove_grid.isEnabled())

        self.win.exit_sweep(confirm_choice="no")
        self.assertTrue(self.win.btn_add_grid.isEnabled())
        self.assertTrue(self.win.btn_remove_grid.isEnabled())

    def test_grid1_r_sweep_auto_updates_upstream_gap_toggle_and_fixed(self):
        """Verify grid_1_r sweep updates upstream_gap_mm dynamically when checked and keeps it fixed when unchecked."""
        self.win.upstream_gap_settings = {"method": "ratio", "ratio_base": "screen_r", "ratio": 0.75}
        self.win.grid_widgets[0]["r"].setValue(0.8)
        self.win.inputs["upstream_gap_mm"].setValue(0.6)
        self.win._last_auto_gap = 0.6
        self.win._user_overrode_Lx = False
        self.win._last_auto_neut_x = self.win.inputs["neut_x"].value()

        # Case A: auto_update_upstream_gap = True -> varies 0.6 -> 1.2 mm
        cfg_var = {
            "param_id": "grid_1_r",
            "param_label": "Grid 1 Aperture Radius (mm)",
            "param_unit": "mm",
            "values": [0.8, 1.6],
            "auto_update_upstream_gap": True,
            "upstream_gap_coupling": {"base": "screen_r", "ratio": 0.75},
            "window": 10, "threshold": 5.0, "max_iterations": 20, "min_iterations": 10,
            "parallel": False, "n_cores": 1,
        }
        self.win.save_sweep_settings(cfg_var)
        self.win.build_domain(force=True)
        lx_1 = self.win.sim.Lx
        self.assertAlmostEqual(self.win.sim.upstream_gap_mm, 0.6, places=3)

        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._on_sweep_scenario_selected(1)
        d2_var = self.win._get_scenario_domain_data(2)
        self.assertAlmostEqual(d2_var["upstream_gap_mm"], 1.2, places=3)
        self.assertAlmostEqual(d2_var["Lx"] - lx_1, 0.6, places=3)
        self.assertAlmostEqual(self.win.inputs["upstream_gap_mm"].value(), 1.2, places=3)

        self.win.combo_sweep_scenario.setCurrentIndex(0)
        self.win._on_sweep_scenario_selected(0)
        self.win.sweep_current_idx = 0
        self.win.is_sweep_active = True
        self.win.sweep_mode_armed = False
        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.advance_sweep_scenario()
            mock_warn.assert_not_called()
        self.assertAlmostEqual(self.win.inputs["upstream_gap_mm"].value(), 1.2, places=3)
        self.assertAlmostEqual(self.win.sim.upstream_gap_mm, 1.2, places=3)
        self.win.exit_sweep(confirm_choice="no")

        # Case B: auto_update_upstream_gap = False -> stays strictly fixed at 0.6 mm across preview & sequential step
        self.win.grid_widgets[0]["r"].setValue(0.8)
        self.win.inputs["upstream_gap_mm"].setValue(0.6)
        self.win._last_auto_gap = 0.6
        cfg_fixed = dict(cfg_var)
        cfg_fixed["auto_update_upstream_gap"] = False
        self.win.save_sweep_settings(cfg_fixed)
        self.win.build_domain(force=True)
        lx_fixed_1 = self.win.sim.Lx

        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._on_sweep_scenario_selected(1)
        d2_fixed = self.win._get_scenario_domain_data(2)
        self.assertAlmostEqual(d2_fixed["upstream_gap_mm"], 0.6, places=3)
        self.assertAlmostEqual(d2_fixed["Lx"], lx_fixed_1, places=3)
        self.assertAlmostEqual(self.win.inputs["upstream_gap_mm"].value(), 0.6, places=3)

        self.win.combo_sweep_scenario.setCurrentIndex(0)
        self.win._on_sweep_scenario_selected(0)
        self.win.sweep_current_idx = 0
        self.win.is_sweep_active = True
        self.win.sweep_mode_armed = False
        with patch.object(QMessageBox, "warning") as mock_warn:
            self.win.advance_sweep_scenario()
            mock_warn.assert_not_called()
        self.assertAlmostEqual(self.win.inputs["upstream_gap_mm"].value(), 0.6, places=3)
        self.assertAlmostEqual(self.win.sim.upstream_gap_mm, 0.6, places=3)
        self.win.exit_sweep(confirm_choice="no")


if __name__ == "__main__":
    unittest.main()
