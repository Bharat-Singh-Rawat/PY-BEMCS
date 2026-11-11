"""
Unit tests for parallel multi-core sweep execution, core count controls, and worker integration.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEventLoop

from gui.dialogs import SweepSettingsDialog, ParallelSweepProgressDialog
from gui.workers.sweep_worker import ParallelSweepWorker
from physics.headless_sweep import (
    apply_sweep_param_to_dict,
    compute_perveance_metrics_static,
    run_single_sweep_scenario_worker,
)
from gui.app.main_window import DigitalTwinApp


class TestSweepSettingsDialogParallelConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_default_cores_and_buttons(self):
        dlg = SweepSettingsDialog()
        total_cpus = os.cpu_count() or 16
        expected_half = max(1, total_cpus // 2)

        # 1. Parallel checkbox enabled by default
        self.assertTrue(dlg.chk_parallel.isChecked())

        # 2. Spin box configured with 1 to total_cpus and half as default
        self.assertEqual(dlg.spin_cores.minimum(), 1)
        self.assertEqual(dlg.spin_cores.maximum(), total_cpus)
        self.assertEqual(dlg.spin_cores.value(), expected_half)

        # 3. Test "Max" button sets cores to total_cpus
        dlg.btn_max_cores.click()
        self.assertEqual(dlg.spin_cores.value(), total_cpus)

        # 4. Test "Set to Half" button sets cores back to half
        dlg.btn_half_cores.click()
        self.assertEqual(dlg.spin_cores.value(), expected_half)

        # 5. Check config export
        cfg = dlg.get_config()
        self.assertTrue(cfg["parallel"])
        self.assertEqual(cfg["n_cores"], expected_half)

        # 6. Test toggling parallel checkbox disables/enables core inputs
        dlg.chk_parallel.setChecked(False)
        self.assertFalse(dlg.spin_cores.isEnabled())
        self.assertFalse(dlg.btn_half_cores.isEnabled())
        self.assertFalse(dlg.btn_max_cores.isEnabled())
        cfg_disabled = dlg.get_config()
        self.assertFalse(cfg_disabled["parallel"])

        dlg.chk_parallel.setChecked(True)
        self.assertTrue(dlg.spin_cores.isEnabled())
        self.assertTrue(dlg.btn_half_cores.isEnabled())
        self.assertTrue(dlg.btn_max_cores.isEnabled())

        dlg.close()


class TestHeadlessSweepUtilities(unittest.TestCase):
    def test_apply_sweep_param_scalar(self):
        base = {
            "Te_up": 2.0,
            "n0_plasma": 1e16,
            "Ti": 0.5,
            "n0": 1e17,
            "Tn": 300.0,
            "Te": 2.0,
            "grids": [{"V": 1000.0}, {"V": -200.0}],
        }
        p1 = apply_sweep_param_to_dict(base, "Te_up", 8.5)
        self.assertEqual(p1["Te_up"], 8.5)
        self.assertEqual(base["Te_up"], 2.0)  # Original unchanged

        p2 = apply_sweep_param_to_dict(base, "n0_plasma", 5e17)
        self.assertEqual(p2["n0_plasma"], 5e17)

    def test_apply_sweep_param_grid(self):
        base = {
            "grids": [
                {"V": 1000.0, "gap": 1.0},
                {"V": -200.0, "gap": 1.0},
            ]
        }
        p1 = apply_sweep_param_to_dict(base, "grid_1_V", 1400.0)
        self.assertEqual(p1["grids"][0]["V"], 1400.0)
        self.assertEqual(base["grids"][0]["V"], 1000.0)

        p2 = apply_sweep_param_to_dict(base, "grid_2_V", -350.0)
        self.assertEqual(p2["grids"][1]["V"], -350.0)

    def test_compute_perveance_metrics_static(self):
        grids = [
            {"V": 1000.0, "gap": 1.0, "t": 0.5, "r": 1.0},
            {"V": -200.0, "gap": 1.0, "t": 1.0, "r": 0.8},
        ]
        i_exit = 0.05  # 50 mA
        p_eff, p_max, perv_frac = compute_perveance_metrics_static(grids, i_exit)

        v_tot = 1200.0
        expected_p_eff = 0.05 / (v_tot ** 1.5)
        self.assertAlmostEqual(p_eff, expected_p_eff, places=10)
        self.assertGreater(p_max, 0.0)
        self.assertAlmostEqual(perv_frac, p_eff / p_max, places=6)

    def test_run_single_sweep_scenario_worker(self):
        base_params = {
            "Te_up": 2.0, "n0_plasma": 1e16, "Ti": 0.5, "n0": 1e17, "Tn": 300.0,
            "grid_1_V": 1000.0, "grid_2_V": -200.0, "Te": 2.0,
            "grids": [
                {"V": 1000.0, "t": 0.5, "gap": 1.0, "r": 1.0, "cham": 0.0},
                {"V": -200.0, "t": 1.0, "gap": 1.0, "r": 0.8, "cham": 0.0},
            ]
        }
        payload = {
            "scenario_idx": 1,
            "param_id": "Te_up",
            "parameter_value": 3.0,
            "base_params": base_params,
            "steady_config": {"window": 10, "threshold": 5.0, "max_iterations": 15, "min_iterations": 10}
        }
        res = run_single_sweep_scenario_worker(payload)
        self.assertEqual(res["scenario_idx"], 1)
        self.assertEqual(res["parameter_value"], 3.0)
        self.assertIn("steady_state_reached", res)
        self.assertIn("steady_iterations", res)
        self.assertIn("divergence_deg", res)
        self.assertIn("divergence_2nd_plane_deg", res)
        self.assertIn("perveance_effective", res)
        self.assertIn("perveance_fraction", res)
        self.assertIn("computation_time_s", res)
        self.assertGreater(res["computation_time_s"], 0.0)

    def test_apply_sweep_upstream_gap_mm_updates_auto_geometry(self):
        grids = [
            {"V": 1000.0, "t": 0.4, "gap": 0.0, "r": 0.8, "cham": 0.0},
            {"V": -200.0, "t": 0.5, "gap": 1.1, "r": 0.7, "cham": 0.0},
        ]
        # total_grid_thickness = 0.4 + 0.5 + 1.1 = 2.0, x_exit - gap = 0.9

        # 1. Auto Lx and auto neut_x shrink and grow symmetrically with upstream_gap_mm
        base_auto = {
            "upstream_gap_mm": 1.0,
            "Lx": 3.0,
            "neut_x": 2.89,
            "user_overrode_Lx": False,
            "user_overrode_neut_x": False,
            "grids": grids,
        }
        shrunk = apply_sweep_param_to_dict(base_auto, "upstream_gap_mm", 0.375)
        self.assertAlmostEqual(shrunk["upstream_gap_mm"], 0.375, places=6)
        self.assertAlmostEqual(shrunk["Lx"], 2.375, places=3)
        # x_exit = 0.375 + 0.9 = 1.275, plume = 1.1 -> neut_x = 1.275 + 0.99 = 2.265
        self.assertAlmostEqual(shrunk["neut_x"], 2.265, places=3)
        self.assertEqual(base_auto["upstream_gap_mm"], 1.0)

        grown = apply_sweep_param_to_dict(base_auto, "upstream_gap_mm", 2.5)
        self.assertAlmostEqual(grown["Lx"], 4.5, places=3)
        self.assertAlmostEqual(grown["neut_x"], round(3.4 + 0.9 * 1.1, 3), places=3)

        # 2. User-overridden Lx and neut_x remain respected
        base_fixed = {
            "upstream_gap_mm": 0.6,
            "Lx": 6.0,
            "neut_x": 4.25,
            "user_overrode_Lx": True,
            "user_overrode_neut_x": True,
            "grids": grids,
        }
        fixed = apply_sweep_param_to_dict(base_fixed, "upstream_gap_mm", 0.3)
        self.assertAlmostEqual(fixed["upstream_gap_mm"], 0.3, places=6)
        self.assertEqual(fixed["Lx"], 6.0)
        self.assertEqual(fixed["neut_x"], 4.25)

    def test_run_single_sweep_scenario_worker_upstream_gap_mm(self):
        base_params = {
            "Te_up": 2.0, "n0_plasma": 1e16, "Ti": 0.5, "n0": 1e17, "Tn": 300.0,
            "Te": 2.0, "upstream_gap_mm": 0.8, "Lx": 3.8,
            "user_overrode_Lx": False, "user_overrode_neut_x": False, "neut_x": 3.65,
            "grids": [
                {"V": 1000.0, "t": 0.5, "gap": 1.0, "r": 1.0, "cham": 0.0},
                {"V": -200.0, "t": 0.5, "gap": 1.0, "r": 0.8, "cham": 0.0},
            ],
        }
        status_dict = {}
        res1 = run_single_sweep_scenario_worker({
            "scenario_idx": 1,
            "param_id": "upstream_gap_mm",
            "parameter_value": 0.45,
            "base_params": base_params,
            "steady_config": {"window": 5, "threshold": 5.0, "max_iterations": 6, "min_iterations": 5},
            "status_dict": status_dict,
        })
        res2 = run_single_sweep_scenario_worker({
            "scenario_idx": 2,
            "param_id": "upstream_gap_mm",
            "parameter_value": 1.35,
            "base_params": base_params,
            "steady_config": {"window": 5, "threshold": 5.0, "max_iterations": 6, "min_iterations": 5},
            "status_dict": status_dict,
        })
        self.assertAlmostEqual(res1["parameter_value"], 0.45, places=6)
        self.assertAlmostEqual(res2["parameter_value"], 1.35, places=6)
        self.assertAlmostEqual(status_dict[1]["Lx"], 3.45, places=3)
        self.assertAlmostEqual(status_dict[2]["Lx"], 4.35, places=3)

    def test_apply_sweep_param_dynamic_grids_1_and_3(self):
        """Verify 1-grid and 3-grid parameter mapping & auto Lx/neut_x synchronization."""
        single_base = {
            "upstream_gap_mm": 0.6,
            "Lx": 2.0,
            "neut_x": 1.9,
            "user_overrode_Lx": False,
            "user_overrode_neut_x": False,
            "grids": [{"V": 1000.0, "t": 0.4, "gap": 1.0, "r": 0.8, "cham": 0.0}],
        }
        s1_cham = apply_sweep_param_to_dict(single_base, "grid_1_cham", 22.5)
        self.assertAlmostEqual(s1_cham["grids"][0]["cham"], 22.5)
        self.assertEqual(s1_cham["Lx"], 2.0)

        s1_gap = apply_sweep_param_to_dict(single_base, "grid_1_gap", 2.0)
        self.assertAlmostEqual(s1_gap["grids"][0]["gap"], 2.0)
        self.assertAlmostEqual(s1_gap["Lx"], 3.0, places=3)
        self.assertAlmostEqual(s1_gap["neut_x"], 2.8, places=3)

        three_base = {
            "upstream_gap_mm": 0.6,
            "Lx": 4.28,
            "neut_x": 4.03,
            "user_overrode_Lx": False,
            "user_overrode_neut_x": False,
            "grids": [
                {"V": 1100.0, "t": 0.38, "gap": 0.64, "r": 0.80, "cham": 0.0},
                {"V": -200.0, "t": 0.38, "gap": 0.64, "r": 0.70, "cham": 0.0},
                {"V": 0.0, "t": 0.38, "gap": 1.26, "r": 0.75, "cham": 0.0},
            ],
        }
        g3_v = apply_sweep_param_to_dict(three_base, "grid_3_V", -75.0)
        self.assertAlmostEqual(g3_v["grids"][2]["V"], -75.0)
        self.assertAlmostEqual(g3_v["grids"][0]["V"], 1100.0)

        g3_r = apply_sweep_param_to_dict(three_base, "grid_3_r", 0.95)
        self.assertAlmostEqual(g3_r["grids"][2]["r"], 0.95)

        g3_cham = apply_sweep_param_to_dict(three_base, "grid_3_cham", 30.0)
        self.assertAlmostEqual(g3_cham["grids"][2]["cham"], 30.0)

        g3_t = apply_sweep_param_to_dict(three_base, "grid_3_t", 0.58)
        self.assertAlmostEqual(g3_t["grids"][2]["t"], 0.58)
        self.assertAlmostEqual(g3_t["Lx"], 4.48, places=3)

    def test_run_single_sweep_scenario_worker_1_and_3_grids(self):
        """Run headless scenario worker with 1-grid chamfer sweep and 3-grid gap sweep."""
        one_grid_base = {
            "Te_up": 2.0, "n0_plasma": 1e16, "Ti": 0.5, "n0": 1e17, "Tn": 300.0,
            "Te": 2.0, "upstream_gap_mm": 0.6, "Lx": 2.5,
            "user_overrode_Lx": False, "user_overrode_neut_x": False, "neut_x": 2.35,
            "grids": [{"V": 1000.0, "t": 0.5, "gap": 1.4, "r": 0.8, "cham": 0.0}],
        }
        status_1 = {}
        res_1 = run_single_sweep_scenario_worker({
            "scenario_idx": 1,
            "param_id": "grid_1_cham",
            "parameter_value": 25.0,
            "base_params": one_grid_base,
            "steady_config": {"window": 5, "threshold": 5.0, "max_iterations": 6, "min_iterations": 5},
            "status_dict": status_1,
        })
        self.assertEqual(res_1["scenario_idx"], 1)
        self.assertAlmostEqual(res_1["parameter_value"], 25.0, places=6)
        self.assertTrue(np.isfinite(res_1["total_energy_J"]))

        three_grid_base = {
            "Te_up": 2.0, "n0_plasma": 1e16, "Ti": 0.5, "n0": 1e17, "Tn": 300.0,
            "Te": 2.0, "upstream_gap_mm": 0.6, "Lx": 4.28,
            "user_overrode_Lx": False, "user_overrode_neut_x": False, "neut_x": 4.03,
            "grids": [
                {"V": 1100.0, "t": 0.38, "gap": 0.64, "r": 0.80, "cham": 0.0},
                {"V": -200.0, "t": 0.38, "gap": 0.64, "r": 0.70, "cham": 0.0},
                {"V": 0.0, "t": 0.38, "gap": 1.26, "r": 0.75, "cham": 0.0},
            ],
        }
        status_3 = {}
        res_3 = run_single_sweep_scenario_worker({
            "scenario_idx": 1,
            "param_id": "grid_3_gap",
            "parameter_value": 2.26,
            "base_params": three_grid_base,
            "steady_config": {"window": 5, "threshold": 5.0, "max_iterations": 6, "min_iterations": 5},
            "status_dict": status_3,
        })
        self.assertAlmostEqual(res_3["parameter_value"], 2.26, places=6)
        self.assertAlmostEqual(status_3[1]["Lx"], 5.28, places=3)
        self.assertTrue(np.isfinite(res_3["total_energy_J"]))

    def test_apply_sweep_param_and_worker_auto_upstream_gap(self):
        """Verify apply_sweep_param_to_dict and worker honor auto_update_upstream_gap True vs False."""
        base_params = {
            "Te_up": 3.0, "n0_plasma": 1e16, "Ti": 0.5, "n0": 1e17, "Tn": 300.0,
            "Te": 2.0, "upstream_gap_mm": 0.6, "Lx": 3.6, "neut_x": 3.45,
            "user_overrode_Lx": False, "user_overrode_neut_x": False,
            "upstream_gap_settings": {"method": "ratio", "ratio_base": "screen_r", "ratio": 0.75},
            "grids": [
                {"V": 1000.0, "t": 0.5, "gap": 1.0, "r": 0.8, "cham": 0.0},
                {"V": -200.0, "t": 0.5, "gap": 1.0, "r": 0.7, "cham": 0.0},
            ],
        }
        # 1. Auto-update enabled -> r1=1.6 => upstream_gap_mm=1.2, Lx=4.2
        sw_var = {
            "auto_update_upstream_gap": True,
            "upstream_gap_coupling": {"base": "screen_r", "ratio": 0.75},
        }
        out_var = apply_sweep_param_to_dict(base_params, "grid_1_r", 1.6, sweep_config=sw_var)
        self.assertAlmostEqual(out_var["upstream_gap_mm"], 1.2, places=3)
        self.assertAlmostEqual(out_var["Lx"], 4.2, places=3)
        self.assertAlmostEqual(out_var["neut_x"], 4.1, places=3)

        # 2. Auto-update disabled -> upstream_gap_mm stays 0.6, Lx stays 3.6
        sw_fixed = {
            "auto_update_upstream_gap": False,
            "upstream_gap_coupling": {"base": "screen_r", "ratio": 0.75},
        }
        out_fixed = apply_sweep_param_to_dict(base_params, "grid_1_r", 1.6, sweep_config=sw_fixed)
        self.assertAlmostEqual(out_fixed["upstream_gap_mm"], 0.6, places=3)
        self.assertAlmostEqual(out_fixed["Lx"], 3.6, places=3)

        # 3. Worker end-to-end executes with synced upstream_gap_mm
        status_var, status_fixed = {}, {}
        run_single_sweep_scenario_worker({
            "scenario_idx": 1, "param_id": "grid_1_r", "parameter_value": 1.6,
            "base_params": base_params, "sweep_config": sw_var,
            "steady_config": {"window": 5, "threshold": 5.0, "max_iterations": 6, "min_iterations": 5},
            "status_dict": status_var,
        })
        run_single_sweep_scenario_worker({
            "scenario_idx": 1, "param_id": "grid_1_r", "parameter_value": 1.6,
            "base_params": base_params, "sweep_config": sw_fixed,
            "steady_config": {"window": 5, "threshold": 5.0, "max_iterations": 6, "min_iterations": 5},
            "status_dict": status_fixed,
        })
        self.assertAlmostEqual(status_var[1]["upstream_gap_mm"], 1.2, places=3)
        self.assertAlmostEqual(status_var[1]["Lx"], 4.2, places=3)
        self.assertAlmostEqual(status_fixed[1]["upstream_gap_mm"], 0.6, places=3)
        self.assertAlmostEqual(status_fixed[1]["Lx"], 3.6, places=3)


class TestParallelSweepWorkerExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_worker_execution(self):
        base_params = {
            "Te_up": 2.0, "n0_plasma": 1e16, "Ti": 0.5, "n0": 1e17, "Tn": 300.0,
            "grid_1_V": 1000.0, "grid_2_V": -200.0, "Te": 2.0,
            "grids": [
                {"V": 1000.0, "t": 0.5, "gap": 1.0, "r": 1.0, "cham": 0.0},
                {"V": -200.0, "t": 1.0, "gap": 1.0, "r": 0.8, "cham": 0.0},
            ]
        }
        config = {
            "param_id": "Te_up",
            "values": [2.0, 4.0],
            "window": 10,
            "threshold": 5.0,
            "max_iterations": 15,
            "min_iterations": 10,
        }

        worker = ParallelSweepWorker(config, base_params, n_cores=2)
        self.assertEqual(worker.n_cores, 2)

        results_collected = []
        progress_events = []

        finished_results = []
        loop = QEventLoop()
        worker.scenario_completed.connect(lambda row, c, t: results_collected.append(row))
        worker.progress_updated.connect(lambda c, t, s: progress_events.append((c, t, s)))
        worker.sweep_finished.connect(lambda res: (finished_results.extend(res), loop.quit()))
        worker.sweep_error.connect(lambda err: loop.quit())

        worker.start()
        loop.exec_()

        self.assertEqual(len(results_collected), 2)
        self.assertGreater(len(progress_events), 0)
        self.assertEqual(sorted([r["scenario_idx"] for r in results_collected]), [1, 2])
        self.assertEqual(len(finished_results), 2)
        self.assertEqual(finished_results[0]["scenario_idx"], 1)
        self.assertEqual(finished_results[1]["scenario_idx"], 2)


class TestSweepMixinParallelAppIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self):
        self.win = DigitalTwinApp()
        self.win.inputs["neut_x"].setValue(5.0)

    def tearDown(self):
        if hasattr(self.win, '_sim_worker') and self.win._sim_worker.isRunning():
            self.win._sim_worker.stop(wait_ms=1000)
        if hasattr(self.win, 'parallel_sweep_worker') and self.win.parallel_sweep_worker is not None:
            if self.win.parallel_sweep_worker.isRunning():
                self.win.parallel_sweep_worker.cancel()
                self.win.parallel_sweep_worker.wait(2000)
        self.win.close()

    def test_start_and_stop_parallel_sweep(self):
        cfg = {
            "param_id": "Te_up",
            "param_label": "Upstream Electron Temp Te_up (eV)",
            "param_unit": "eV",
            "values": [2.0, 4.0],
            "parallel": True,
            "n_cores": 2,
            "window": 10,
            "threshold": 5.0,
            "max_iterations": 15,
            "min_iterations": 10,
        }

        self.win.build_domain()
        with patch.object(ParallelSweepWorker, "start"), \
             patch.object(ParallelSweepWorker, "isRunning", return_value=True):
            self.win.start_sweep(cfg)
            self.assertTrue(self.win.is_sweep_active)
            self.assertIsNotNone(self.win.parallel_sweep_worker)
            self.assertIsNotNone(self.win.combo_sweep_scenario)
            self.assertFalse(self.win.combo_sweep_scenario.isHidden())
            self.assertEqual(self.win.combo_sweep_scenario.count(), 2)
            self.assertFalse(self.win.lbl_sweep_badge.isHidden())
            self.assertIn("Parallel Sweep", self.win.lbl_sweep_badge.text())

            # Test stop sweep pauses scenarios and keeps worker ready for resume
            self.win.stop_sweep()
            self.assertFalse(self.win.is_sweep_active)
            self.assertTrue(self.win._sweep_stopped)
            self.assertIsNotNone(self.win.parallel_sweep_worker)
            self.assertTrue(self.win.parallel_sweep_worker.is_scenario_paused(1))
            self.assertTrue(self.win.parallel_sweep_worker.is_scenario_paused(2))
            self.assertEqual(self.win.lbl_sweep_badge.text(), "Sweep mode stopped")
            self.assertFalse(self.win.btn_resume_sweep.isHidden())
            self.assertEqual(self.win.btn_toggle.text(), "RESUME BEAM")
            self.assertIn("Paused", self.win.combo_sweep_scenario.itemText(0))
            self.assertIn("Paused", self.win.combo_sweep_scenario.itemText(1))

            # Test resume via top-bar action button (on_sweep_action_clicked)
            self.win.on_sweep_action_clicked()
            self.assertTrue(self.win.is_sweep_active)
            self.assertFalse(self.win._sweep_stopped)
            self.assertEqual(self.win.btn_resume_sweep.text(), "Pause Sweep")
            self.assertEqual(self.win.btn_toggle.text(), "PAUSE BEAM")
            self.assertTrue(self.win.btn_stop_sweep.isHidden())
            self.assertFalse(self.win.parallel_sweep_worker.is_scenario_paused(1))
            self.assertFalse(self.win.parallel_sweep_worker.is_scenario_paused(2))
            self.assertIn("Running", self.win.combo_sweep_scenario.itemText(0))
            self.assertIn("Running", self.win.combo_sweep_scenario.itemText(1))

            # Test stop again and resume via resume_sweep_with_selected
            self.win.stop_sweep()
            self.assertTrue(self.win._sweep_stopped)
            self.assertEqual(self.win.btn_resume_sweep.text(), "Resume Sweep")
            self.assertFalse(self.win.btn_resume_sweep.isHidden())
            self.win.resume_sweep_with_selected()
            self.assertTrue(self.win.is_sweep_active)
            self.assertFalse(self.win._sweep_stopped)
            self.assertEqual(self.win.btn_resume_sweep.text(), "Pause Sweep")
            self.assertFalse(self.win.parallel_sweep_worker.is_scenario_paused(1))
            self.assertFalse(self.win.parallel_sweep_worker.is_scenario_paused(2))
            self.assertIn("Running", self.win.combo_sweep_scenario.itemText(0))
            self.assertIn("Running", self.win.combo_sweep_scenario.itemText(1))

            # Test exit sweep cleans up worker
            self.win.exit_sweep(confirm_choice="no")
            self.assertIsNone(self.win.parallel_sweep_worker)

    def test_scenario_dropdown_switch_updates_canvas(self):
        cfg = {
            "param_id": "Te_up",
            "param_label": "Upstream Electron Temp Te_up (eV)",
            "param_unit": "eV",
            "values": [2.0, 4.0, 6.0],
            "parallel": True,
            "n_cores": 2,
            "window": 10,
            "threshold": 5.0,
            "max_iterations": 15,
            "min_iterations": 10,
        }
        self.win.build_domain()
        with patch.object(ParallelSweepWorker, "start"):
            self.win.start_sweep(cfg)
        self.assertEqual(self.win.combo_sweep_scenario.count(), 3)

        # Manually inject mock scenario status
        pts1 = np.array([[1.0, 0.5], [2.0, 0.6]], dtype=np.float32)
        pts2 = np.array([[3.0, 0.8], [4.0, 0.9]], dtype=np.float32)
        mock_status = {
            1: {"status": "done", "iter": 120, "prim_xy": pts1, "e_prim_eV": np.array([10.0, 12.0]), "t_sim_us": 1.2, "div": 8.5},
            2: {"status": "running", "iter": 60, "prim_xy": pts2, "e_prim_eV": np.array([15.0, 18.0]), "t_sim_us": 0.6, "div": 11.2},
            3: {"status": "queued", "iter": 0},
        }
        self.win._sweep_status_dict = mock_status
        if self.win.parallel_sweep_worker:
            self.win.parallel_sweep_worker.final_status_dict = dict(mock_status)
            for s_idx, st in mock_status.items():
                sf = self.win.parallel_sweep_worker.state_files.get(s_idx)
                if sf:
                    from physics.headless_sweep import write_scenario_state
                    write_scenario_state(sf, st)

        # 1. Switch to Scenario 1
        self.win.combo_sweep_scenario.setCurrentIndex(0)
        self.win._on_sweep_scenario_selected(0)
        np.testing.assert_allclose(self.win.scat_prim.get_offsets(), pts1)
        self.assertIn("Scenario 1", self.win.lblTime.text())

        # 2. Switch to Scenario 2
        self.win.combo_sweep_scenario.setCurrentIndex(1)
        self.win._on_sweep_scenario_selected(1)
        np.testing.assert_allclose(self.win.scat_prim.get_offsets(), pts2)
        self.assertIn("Scenario 2", self.win.lblTime.text())

        self.win.stop_sweep()
        self.win.exit_sweep(confirm_choice="no")

    def test_close_event_prompt_when_sweep_running(self):
        from PyQt5.QtGui import QCloseEvent
        from PyQt5.QtWidgets import QMessageBox

        self.win.is_sweep_active = True
        self.assertTrue(self.win.is_simulation_running())

        # 1. User selects "No" -> event should be ignored
        event_no = QCloseEvent()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.No) as mock_q:
            self.win.closeEvent(event_no)
            self.assertFalse(event_no.isAccepted())
            mock_q.assert_called_once()
            call_text = mock_q.call_args[0][2]
            self.assertIn("Attention: simulation is running", call_text)
            self.assertIn("close the tab", call_text)

        # 2. User selects "Yes" -> event should be accepted and background tasks shut down
        event_yes = QCloseEvent()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes) as mock_q:
            self.win.closeEvent(event_yes)
            self.assertTrue(event_yes.isAccepted())
            self.assertFalse(self.win.is_sweep_active)

    def test_close_event_prompt_when_sim_running(self):
        from PyQt5.QtGui import QCloseEvent
        from PyQt5.QtWidgets import QMessageBox

        self.win.sim_isRunning = True
        self.assertTrue(self.win.is_simulation_running())

        event_no = QCloseEvent()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.No) as mock_q:
            self.win.closeEvent(event_no)
            self.assertFalse(event_no.isAccepted())
            mock_q.assert_called_once()

        event_yes = QCloseEvent()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes) as mock_q:
            self.win.closeEvent(event_yes)
            self.assertTrue(event_yes.isAccepted())
            self.assertFalse(self.win.sim_isRunning)

    def test_close_event_no_prompt_when_idle(self):
        from PyQt5.QtGui import QCloseEvent
        from PyQt5.QtWidgets import QMessageBox

        self.win.sim_isRunning = False
        self.win.is_sweep_active = False
        self.assertFalse(self.win.is_simulation_running())

        event = QCloseEvent()
        with patch.object(QMessageBox, 'question') as mock_q:
            self.win.closeEvent(event)
            mock_q.assert_not_called()
            self.assertTrue(event.isAccepted())

    def test_progress_dialog_close_event_prompt(self):
        from PyQt5.QtGui import QCloseEvent
        from PyQt5.QtWidgets import QMessageBox

        cfg = {"param_id": "Te_up", "values": [2.0, 4.0]}
        dlg = ParallelSweepProgressDialog(cfg, n_cores=2)

        # 1. User says No
        ev_no = QCloseEvent()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.No) as mock_q:
            dlg.closeEvent(ev_no)
            self.assertFalse(ev_no.isAccepted())

        # 2. User says Yes
        ev_yes = QCloseEvent()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes) as mock_q:
            dlg.closeEvent(ev_yes)
            self.assertTrue(ev_yes.isAccepted())
            self.assertEqual(dlg.result(), ParallelSweepProgressDialog.Rejected)

    def test_sweep_scenario_render_divergence_and_particles_custom_and_all_limits(self):
        """Verify particles and divergence graphs update correctly under custom limit, All particles, and None."""
        self.win.build_domain()

        pts = np.array([[0.5, 0.2], [1.0, 0.4], [1.5, 0.6], [2.0, 0.8], [2.5, 1.0]], dtype=np.float32)
        e_pts = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        iter_hist = [10, 20, 30, 40, 50]
        div_hist = [15.2, 14.1, 13.0, 12.5, 12.1]
        div_mid_hist = [10.2, 9.8, 9.5, 9.1, 8.8]
        ebs_hist = [-650.0, -640.0, -630.0, -625.0, -620.0]

        mock_status = {
            1: {
                "status": "done",
                "iter": 50,
                "prim_xy": pts,
                "e_prim_eV": e_pts,
                "iter_history": iter_hist,
                "div_history": div_hist,
                "div_mid_history": div_mid_hist,
                "ebs_history": ebs_hist,
                "div": 12.1,
                "div_mid": 8.8,
                "t_sim_us": 1.5,
                "Lx": 3.0,
                "Ly": 3.0,
            }
        }
        self.win._sweep_status_dict = mock_status

        # 1. Custom limit = 3 particles
        self.win.scatter_limit = 3
        self.win._render_sweep_scenario_snapshot(1)
        sampled = self.win.scat_prim.get_offsets()
        self.assertEqual(len(sampled), 3)
        # Verify sampled points are from original pts
        for p in sampled:
            self.assertTrue(any(np.allclose(p, orig) for orig in pts))

        # Verify divergence graph was updated
        div_x, div_y = self.win.line_div.get_data()
        np.testing.assert_allclose(div_x, iter_hist)
        np.testing.assert_allclose(div_y, div_hist)
        div_mid_x, div_mid_y = self.win.line_div_mid.get_data()
        np.testing.assert_allclose(div_mid_y, div_mid_hist)
        ebs_x, ebs_y = self.win.line_ebs.get_data()
        np.testing.assert_allclose(ebs_y, ebs_hist)

        # 2. All particles (limit = 0)
        self.win.scatter_limit = 0
        self.win._render_sweep_scenario_snapshot(1)
        self.assertEqual(len(self.win.scat_prim.get_offsets()), 5)
        np.testing.assert_allclose(self.win.scat_prim.get_offsets(), pts)

        # 3. None (limit = None -> hide all particles)
        self.win.scatter_limit = None
        self.win._render_sweep_scenario_snapshot(1)
        self.assertEqual(len(self.win.scat_prim.get_offsets()), 0)

    def test_extract_display_particles_units_and_limits(self):
        """Verify extract_display_particles does not scale mm coords and obeys scatter_limit."""
        from physics.headless_sweep import extract_display_particles

        class DummySim:
            num_p = 100
            num_e = 50
            p_x = np.linspace(0.1, 2.9, 100)  # already mm
            p_y = np.linspace(0.1, 1.4, 100)
            p_vx = np.full(100, 1000.0)
            p_vy = np.zeros(100)
            p_vz = np.zeros(100)
            p_isCEX = np.zeros(100, dtype=bool)
            p_isCEX[80:] = True  # 20 CEX ions
            e_x = np.linspace(0.2, 2.8, 50)
            e_y = np.linspace(0.2, 1.2, 50)
            m_ion = 131.293 * 1.6605e-27
            q_ion = 1.602e-19

        sim = DummySim()

        # All particles
        prim_xy, e_prim, cex_xy, e_cex, elec_xy = extract_display_particles(sim, scatter_limit=0)
        self.assertEqual(len(prim_xy), 80)
        self.assertEqual(len(cex_xy), 20)
        self.assertEqual(len(elec_xy), 50)
        # Verify coordinates are in mm (< 3.0), NOT multiplied by 1e3
        self.assertTrue(np.all(prim_xy[:, 0] <= 3.0))
        self.assertTrue(np.all(prim_xy[:, 0] >= 0.0))

        # Custom limit = 20
        prim_xy_lim, _, cex_xy_lim, _, elec_xy_lim = extract_display_particles(sim, scatter_limit=20)
        self.assertLessEqual(len(prim_xy_lim) + len(cex_xy_lim), 20)
        self.assertLessEqual(len(elec_xy_lim), 20)

        # Limit = None (hide all)
        prim_xy_none, _, cex_xy_none, _, elec_xy_none = extract_display_particles(sim, scatter_limit=None)
        self.assertEqual(len(prim_xy_none), 0)
        self.assertEqual(len(cex_xy_none), 0)
        self.assertEqual(len(elec_xy_none), 0)

    def test_newly_injected_particles_preserved_during_subsampling(self):
        """Verify newly injected particles at the array tail are not truncated away."""
        from physics.headless_sweep import extract_display_particles

        class InjectedSim:
            num_p = 500
            num_e = 0
            p_x = np.linspace(1.0, 3.0, 500)
            p_x[480:] = 0.03  # newest 20 particles at injection boundary
            p_y = np.full(500, 0.5)
            p_vx = np.full(500, 1000.0)
            p_vy = np.zeros(500)
            p_vz = np.zeros(500)
            p_isCEX = np.zeros(500, dtype=bool)
            m_ion = 131.293 * 1.6605e-27
            q = 1.602e-19

        sim = InjectedSim()
        found_injected = False
        for _ in range(20):
            prim_xy, _, _, _, _ = extract_display_particles(sim, scatter_limit=50)
            if np.any(prim_xy[:, 0] <= 0.05):
                found_injected = True
                break
        self.assertTrue(found_injected, "Newly injected particles at array tail must be sampled, not cut off")

    def test_contour_cleanup_on_scenario_switch_prevents_fog(self):
        """Verify switching scenarios in tent menu cleanly removes old contours so only 1 ContourSet remains."""
        import matplotlib.contour
        self.win.build_domain()
        cfg = {
            "param_id": "grid_1_V",
            "values": [800.0, 1200.0],
            "parallel": True,
            "n_cores": 2,
        }
        self.win.save_sweep_settings(cfg)

        # Render scenario 1 snapshot
        self.win._on_sweep_scenario_selected(0)
        c_count_1 = sum(1 for c in self.win.ax_live.collections if isinstance(c, matplotlib.contour.ContourSet))
        self.assertEqual(c_count_1, 1)

        # Switch to scenario 2 snapshot
        self.win._on_sweep_scenario_selected(1)
        c_count_2 = sum(1 for c in self.win.ax_live.collections if isinstance(c, matplotlib.contour.ContourSet))
        self.assertEqual(c_count_2, 1, "Switching scenario must remove previous contour and not accumulate yellow fog")

        # Switch back to scenario 1
        self.win._on_sweep_scenario_selected(0)
        c_count_3 = sum(1 for c in self.win.ax_live.collections if isinstance(c, matplotlib.contour.ContourSet))
        self.assertEqual(c_count_3, 1)

    def test_parallel_sweep_worker_pause_and_resume_state(self):
        """Verify ParallelSweepWorker pause_scenario and resume_scenario update in-memory and disk states correctly."""
        cfg = {
            "param_id": "grid_1_V",
            "values": [900.0, 1100.0],
            "parallel": True,
            "n_cores": 2,
        }
        worker = ParallelSweepWorker(cfg, self.win.get_params(), n_cores=2)

        # Scenario 1 initially not paused
        self.assertFalse(worker.is_scenario_paused(1))

        # Pause Scenario 1
        worker.pause_scenario(1)
        self.assertTrue(worker.is_scenario_paused(1))
        self.assertEqual(worker.get_status_dict()[1]["status"], "paused")

        # Resume Scenario 1
        worker.resume_scenario(1)
        self.assertFalse(worker.is_scenario_paused(1))
        self.assertEqual(worker.get_status_dict()[1]["status"], "running")

        worker._cleanup_temp_files()


if __name__ == "__main__":
    unittest.main()
