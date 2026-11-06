import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics.backend import USE_TAICHI
from physics.kernels_cpu import push_particles_boris_cpu
from physics_engine import DigitalTwinSimulator


def test_fused_boris_substeps_equivalence():
    """Verify that fused in-register Taichi Boris pusher matches sequential pushing bit-for-bit."""
    if not USE_TAICHI:
        pytest.skip("Taichi GPU backend not available in this environment")

    from physics.kernels_taichi import push_particles_boris_taichi
    import taichi as ti

    num_p = 5000
    nx, ny = 146, 118
    np.random.seed(42)

    px1 = np.random.uniform(0.1, 2.5, num_p).astype(np.float32)
    py1 = np.random.uniform(0.1, 2.5, num_p).astype(np.float32)
    pvx1 = np.random.randn(num_p).astype(np.float32) * 1000.0
    pvy1 = np.random.randn(num_p).astype(np.float32) * 1000.0
    pvz1 = np.random.randn(num_p).astype(np.float32) * 1000.0

    px2 = px1.copy()
    py2 = py1.copy()
    pvx2 = pvx1.copy()
    pvy2 = pvy1.copy()
    pvz2 = pvz1.copy()

    Ex = np.random.randn(ny, nx).astype(np.float32)
    Ey = np.random.randn(ny, nx).astype(np.float32)
    Bx = np.zeros((ny, nx), dtype=np.float32)
    By = np.zeros((ny, nx), dtype=np.float32)
    Bz = np.zeros((ny, nx), dtype=np.float32)

    zone_b = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    zone_dx = np.array([0.02, 0.02, 0.02], dtype=np.float32)
    zone_off = np.array([0, 50, 100], dtype=np.int32)

    n_sub = 5
    dt_sub = 1e-10

    # 1. Sequential calls
    for _ in range(n_sub):
        push_particles_boris_taichi(
            px1, py1, pvx1, pvy1, pvz1,
            Ex, Ey, Bx, By, Bz,
            num_p, 0.02, nx, ny, dt_sub, 1.0,
            zone_b, zone_dx, zone_off, 3, n_sub=1
        )
    ti.sync()

    # 2. Fused call
    push_particles_boris_taichi(
        px2, py2, pvx2, pvy2, pvz2,
        Ex, Ey, Bx, By, Bz,
        num_p, 0.02, nx, ny, dt_sub, 1.0,
        zone_b, zone_dx, zone_off, 3, n_sub=n_sub
    )
    ti.sync()

    diff_x = np.max(np.abs(px1 - px2))
    diff_vx = np.max(np.abs(pvx1 - pvx2))

    assert diff_x < 1e-5, f"Position mismatch: max diff = {diff_x}"
    assert diff_vx < 1e-5, f"Velocity mismatch: max diff = {diff_vx}"


def test_fast_path_substep_bounding():
    """Verify that the fast upper-bound in compute_particle_substeps returns n_sub=1 correctly."""
    sim = DigitalTwinSimulator()
    sim.dx = 0.02
    sim.dy = 0.02
    sim.dx_min = 0.02
    sim.nx = 100
    sim.ny = 100
    sim.x_coords = np.linspace(0, 2.0, 100)
    sim.Ex = np.zeros((100, 100), dtype=np.float32)
    sim.Ey = np.zeros((100, 100), dtype=np.float32)

    # Slow particles where ds_pred << ds_lim
    N = 1000
    vx = np.full(N, 100.0, dtype=np.float32)
    vy = np.zeros(N, dtype=np.float32)
    vz = np.zeros(N, dtype=np.float32)
    x = np.full(N, 0.5, dtype=np.float32)
    y = np.full(N, 0.5, dtype=np.float32)

    n_sub, ds_max, ds_lim = sim.compute_particle_substeps(x, y, vx, vy, vz, 1.0, dt=1e-9)
    assert n_sub == 1
    assert ds_max < ds_lim
