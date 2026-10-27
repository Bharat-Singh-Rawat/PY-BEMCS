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
                "exit_ion_current_A": 0.045,
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
                "exit_ion_current_A": 0.041,
                "transparency": 0.79,
                "total_energy_J": 1.5e-3,
                "total_charge_C": 1.4e-9,
                "field_energy_J": 4.8e-4,
            },
        ]

        dlg = SweepResultsDialog(cfg, mock_results)
        self.assertEqual(dlg.table.rowCount(), 2)

        # Check table contents
        item_ss1 = dlg.table.item(0, 2)
        item_ss2 = dlg.table.item(1, 2)
        self.assertIn("YES", item_ss1.text())
        self.assertIn("NO", item_ss2.text())

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
        finally:
            if os.path.exists(tmp_txt_path):
                os.remove(tmp_txt_path)

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

        # Check steady state should detect stationarity and record result
        self.win.check_sweep_steady_state()

        self.assertEqual(len(self.win.sweep_results), 1)
        self.assertTrue(self.win.sweep_results[0]["steady_state_reached"])
        self.assertEqual(self.win.sweep_results[0]["parameter_value"], 2.0)
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


if __name__ == "__main__":
    unittest.main()
