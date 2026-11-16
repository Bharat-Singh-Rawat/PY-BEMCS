"""
Unit tests for per-scenario downstream plume advisory hold and interactive configuration.
"""
import os
import sys
import copy
import pytest
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYBEMCS_FORCE_CPU"] = "1"

from PyQt5.QtWidgets import QApplication

from physics.hardware import check_scenario_plume_advisory
from gui.dialogs import ScenarioPlumeAdvisoryDialog
from gui.workers.sweep_worker import ParallelSweepWorker


def test_scenario_plume_advisory_detection():
    """Verify check_scenario_plume_advisory correctly flags high-mesh plume scenarios."""
    grids = [
        {"V": 1300.0, "t": 1.2, "gap": 0.9, "r": 1.3, "cham": 20.0},
        {"V": -150.0, "t": 0.6, "gap": 0.5, "r": 0.8, "cham": 0.0},
        {"V": 50.0,   "t": 0.5, "gap": 8.0, "r": 0.8, "cham": 0.0},
    ]

    # Case A: Low density n0=1e16 -> Few cells (< 300,000) -> No advisory
    low_params = {
        "n0_plasma": 1e16,
        "Te_up": 2.9,
        "dx_over_debye": 0.8,
        "upstream_gap_mm": 0.8,
        "grids": copy.deepcopy(grids),
        "target_ppc": 40.0,
    }
    adv_low = check_scenario_plume_advisory(low_params)
    assert adv_low["has_advisory"] is False

    # Case B: Extreme density n0=1e19 with 8mm plume gap -> > 300,000 cells -> Advisory triggered
    high_params = {
        "n0_plasma": 1e19,
        "Te_up": 2.9,
        "dx_over_debye": 0.8,
        "upstream_gap_mm": 0.8,
        "grids": copy.deepcopy(grids),
        "target_ppc": 200.0,
    }
    adv_high = check_scenario_plume_advisory(high_params)
    assert adv_high["has_advisory"] is True
    assert adv_high["plume_len"] == 8.0
    assert adv_high["r_accel"] == 0.8
    assert adv_high["total_cells"] > 300_000
    assert adv_high["excess_cells"] > 0
    assert adv_high["recommended_Lx"] < adv_high["Lx"]
    assert adv_high["modulated_budget"] is True

    # Case C: If truncate_downstream_plume is already True -> No advisory
    high_params["truncate_downstream_plume"] = True
    adv_trunc = check_scenario_plume_advisory(high_params)
    assert adv_trunc["has_advisory"] is False


def test_scenario_plume_advisory_dialog():
    """Verify interactive dialog properties and user choice extraction."""
    qt_app = QApplication.instance() or QApplication(sys.argv)

    adv_info = {
        "n0": 1.12e18,
        "plume_len": 8.0,
        "r_accel": 0.8,
        "Lx": 12.94,
        "recommended_Lx": 7.44,
        "total_cells": 1_540_000,
        "excess_cells": 350_000,
        "pct_reduction": 23.5,
        "est_ram_gb": 1.85,
        "target_ppc": 200.0,
        "effective_ppc": 22.0,
    }

    # Test dialog with 3 remaining held scenarios
    dlg = ScenarioPlumeAdvisoryDialog(scen_num=2, advisory_info=adv_info, remaining_held_count=3)
    assert dlg.chk_apply_all is not None
    assert dlg.chk_apply_all.isChecked() is True

    # Simulate clicking "Truncate Plume"
    dlg._on_truncate()
    assert dlg.get_choice() == "truncate"
    assert dlg.apply_to_all() is True

    # Simulate clicking "Keep Full Domain"
    dlg2 = ScenarioPlumeAdvisoryDialog(scen_num=3, advisory_info=adv_info, remaining_held_count=1)
    assert dlg2.chk_apply_all is None  # Only 1 scenario remaining, checkbox not shown
    dlg2._on_full()
    assert dlg2.get_choice() == "full"
    assert dlg2.apply_to_all() is False

    # Simulate clicking "Keep on Hold"
    dlg3 = ScenarioPlumeAdvisoryDialog(scen_num=4, advisory_info=adv_info, remaining_held_count=2)
    dlg3._on_hold()
    assert dlg3.get_choice() == "hold"


def test_parallel_worker_held_scenarios():
    """Verify ParallelSweepWorker holds flagged scenarios until released."""
    grids = [
        {"V": 1300.0, "t": 1.2, "gap": 0.9, "r": 1.3, "cham": 20.0},
        {"V": -150.0, "t": 0.6, "gap": 0.5, "r": 0.8, "cham": 0.0},
        {"V": 50.0,   "t": 0.5, "gap": 8.0, "r": 0.8, "cham": 0.0},
    ]
    base_params = {
        "n0_plasma": 1e16,
        "Te_up": 2.9,
        "dx_over_debye": 0.8,
        "upstream_gap_mm": 0.8,
        "grids": grids,
        "max_iterations": 10,
    }
    sweep_cfg = {
        "param_id": "n0_plasma",
        "values": [1e16, 1.12e18, 2.23e18],
        "max_iterations": 10,
    }

    # Hold scenarios 2 and 3
    worker = ParallelSweepWorker(
        sweep_config=sweep_cfg,
        base_params=base_params,
        n_cores=2,
        held_scenarios=[2, 3],
    )

    assert worker.is_scenario_held(1) is False
    assert worker.is_scenario_held(2) is True
    assert worker.is_scenario_held(3) is True

    status_dict = worker.get_status_dict()
    assert status_dict[1]["status"] == "queued"
    assert status_dict[2]["status"] == "action_required"
    assert status_dict[3]["status"] == "action_required"

    # Release scenario 2 with truncation
    worker.release_scenario(2, truncate_plume=True)
    assert worker.is_scenario_held(2) is False
    assert worker.is_scenario_held(3) is True

    status_dict2 = worker.get_status_dict()
    assert status_dict2[2]["status"] == "queued"

    # Cleanup temp worker files
    worker._cleanup_temp_files()

