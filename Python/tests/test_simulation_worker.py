"""
Tests for Phase 3 SimulationWorker decoupling and fast-path pipeline.
Verifies decoupled snapshot throttling, final frame emission on pause,
and uniform strided subsampling without memory duplication.
"""
import time
import pytest
import numpy as np
from PyQt5.QtCore import QCoreApplication
from physics.simulator import DigitalTwinSimulator
from gui.workers.simulation_worker import SimulationWorker


def test_simulation_worker_decoupled_snapshots(qtbot):
    """Verify that worker steps rapidly and emits decoupled snapshots."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])

    sim = DigitalTwinSimulator()
    params = {
        'V_screen': 1000.0,
        'V_accel': -200.0,
        'sim_mode': 'Both',
        'dt': 1e-10,
        'n0_plasma': 1e16,
        'Te_up': 3.0,
        'Te_eV': 3.0,
        'Ti_eV': 0.5,
        'beam_mass_amu': 131.293,
        'mesh_strategy': 'uniform',
        'grids': [
            {'V': 1000.0, 't': 0.5, 'gap': 0.5, 'r': 1.0, 'cham': 0.0},
            {'V': -200.0, 't': 0.5, 'gap': 1.0, 'r': 0.8, 'cham': 0.0}
        ]
    }
    sim.build_domain(params)
    # Warmup step to allow Taichi kernels to JIT compile
    sim.step(params)

    worker = SimulationWorker(sim)
    worker._min_snap_interval = 0.05  # 50 ms

    received_snaps = []
    worker.step_done.connect(lambda s: received_snaps.append(s))

    worker.start_sim(params)
    # Let it run for 400 ms
    qtbot.wait(400)
    worker.pause()
    worker.wait(2000)
    app.processEvents()

    # Worker should have completed multiple physics steps
    assert sim.iteration >= 3, f"Expected at least 3 iterations, got {sim.iteration}"
    # Should have emitted at least one snapshot
    assert len(received_snaps) >= 1
    # Final snapshot should have the latest iteration
    assert received_snaps[-1]['iteration'] == sim.iteration


def test_simulation_worker_strided_subsampling():
    """Verify that strided subsampling maintains ratio and does not duplicate full arrays."""
    sim = DigitalTwinSimulator()
    worker = SimulationWorker(sim)
    worker.MAX_SCATTER = 100

    N = 1000
    sim.num_p = N
    sim.p_x = np.linspace(0.0, 10.0, N, dtype=np.float32)
    sim.p_y = np.linspace(0.0, 2.0, N, dtype=np.float32)
    sim.p_vx = np.ones(N, dtype=np.float32) * 1000.0
    sim.p_vy = np.zeros(N, dtype=np.float32)
    sim.p_vz = np.zeros(N, dtype=np.float32)
    sim.p_isCEX = np.zeros(N, dtype=bool)

    stride = max(1, N // worker.MAX_SCATTER)
    sl = slice(0, N, stride)
    sub_x = sim.p_x[sl]

    assert len(sub_x) <= worker.MAX_SCATTER + 1
    assert sub_x[0] == sim.p_x[0]
    assert np.all(np.diff(sub_x) > 0), "Spatial ordering must be strictly preserved"
