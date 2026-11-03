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

        loop = QEventLoop()
        worker.scenario_completed.connect(lambda row, c, t: results_collected.append(row))
        worker.progress_updated.connect(lambda c, t, s: progress_events.append((c, t, s)))
        worker.sweep_finished.connect(lambda res: loop.quit())
        worker.sweep_error.connect(lambda err: loop.quit())

        worker.start()
        loop.exec_()

        self.assertEqual(len(results_collected), 2)
        self.assertGreater(len(progress_events), 0)
        # Results should be sorted by scenario_idx
        self.assertEqual(results_collected[0]["scenario_idx"], 1)
        self.assertEqual(results_collected[1]["scenario_idx"], 2)


class TestSweepMixinParallelAppIntegration(unittest.TestCase):
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

        self.win.start_sweep(cfg)
        self.assertTrue(self.win.is_sweep_active)
        self.assertIsNotNone(self.win.parallel_sweep_worker)
        self.assertIsNotNone(self.win.parallel_sweep_dlg)
        self.assertFalse(self.win.lbl_sweep_badge.isHidden())
        self.assertIn("Parallel Sweep", self.win.lbl_sweep_badge.text())

        # Test early cancellation
        self.win.stop_sweep()
        self.assertFalse(self.win.is_sweep_active)
        self.assertIsNone(self.win.parallel_sweep_worker)
        self.assertEqual(self.win.lbl_sweep_badge.text(), "Sweep mode stopped")


if __name__ == "__main__":
    unittest.main()
