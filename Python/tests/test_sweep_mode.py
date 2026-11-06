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
        self.assertFalse(self.win.btn_stop_sweep.isHidden())
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

        # Starting sweep via normal simulation toggle button
        with patch.object(self.win._sim_worker, "start_sim") as mock_start:
            self.win.toggle_sim()
            self.assertTrue(self.win.is_sweep_active)
            self.assertFalse(self.win.sweep_mode_armed)
            self.assertTrue(self.win.sim_isRunning)
            self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")

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

        # Start sweep via toggle_sim
        with patch.object(self.win, "_start_parallel_sweep") as mock_start_parallel:
            self.win.toggle_sim()
            mock_start_parallel.assert_called_once()
            self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")
            self.assertTrue(self.win.is_sweep_active)

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

    def test_gui_settings_dialog_custom_1000_particles_default(self):
        from gui.dialogs import GUISettingsDialog
        dlg = GUISettingsDialog(self.win)
        self.assertTrue(dlg._rb_custom.isChecked())
        self.assertEqual(dlg._spin_custom.value(), 1000)
        self.assertEqual(dlg.get_scatter_limit(), 1000)
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
        self.assertFalse(self.win.btn_stop_sweep.isHidden())
        self.assertTrue(self.win.btn_resume_sweep.isHidden())

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



if __name__ == "__main__":
    unittest.main()
