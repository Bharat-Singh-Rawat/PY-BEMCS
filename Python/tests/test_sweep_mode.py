import unittest
from unittest.mock import patch
import os
import tempfile
import csv
import numpy as np
from PyQt5.QtWidgets import QApplication

from gui.dialogs import SweepSettingsDialog, SweepResultsDialog, SWEEP_PARAMETERS
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
        self.assertEqual(dlg.table.columnCount(), 11)

        # Check table contents
        item_ss1 = dlg.table.item(0, 2)
        item_ss2 = dlg.table.item(1, 2)
        self.assertIn("YES", item_ss1.text())
        self.assertIn("NO", item_ss2.text())

        # Check divergence columns in table (grid exit and 2nd plane)
        self.assertEqual(dlg.table.item(0, 5).text(), "12.30")
        self.assertEqual(dlg.table.item(0, 6).text(), "10.10")
        self.assertEqual(dlg.table.item(1, 5).text(), "14.10")
        self.assertEqual(dlg.table.item(1, 6).text(), "11.50")

        # Check Perveance Fraction column (col 8)
        self.assertEqual(dlg.table.item(0, 8).text(), "0.2400")
        self.assertEqual(dlg.table.item(1, 8).text(), "0.2200")

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
        self.assertEqual(dlg.table.columnCount(), 11)
        self.assertEqual(dlg.table.horizontalHeaderItem(5).text(), "Divergence Grid Exit (°)")
        self.assertEqual(dlg.table.horizontalHeaderItem(6).text(), "Divergence 2nd Plane (°)")
        self.assertEqual(dlg.table.horizontalHeaderItem(8).text(), "Perveance Fraction")

        # Verify values in table
        self.assertEqual(dlg.table.item(0, 5).text(), "15.20")
        self.assertEqual(dlg.table.item(0, 6).text(), "11.40")
        self.assertEqual(dlg.table.item(1, 5).text(), "13.80")
        self.assertEqual(dlg.table.item(1, 6).text(), "10.20")
        self.assertEqual(dlg.table.item(2, 5).text(), "12.50")
        self.assertEqual(dlg.table.item(2, 6).text(), "0.00")

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
        if hasattr(self.win, '_sim_worker') and self.win._sim_worker.isRunning():
            self.win._sim_worker.pause()
            self.win._sim_worker.wait(2000)
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


if __name__ == "__main__":
    unittest.main()
