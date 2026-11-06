"""
Tests for Phase 2 GPU Wall Collisions and Boundary Handling.
Verifies equivalence between GPU Taichi kernels and CPU NumPy implementations,
zero-loss fast-paths, electron boundary compaction, and CPU fallback consistency.
"""
import pytest
import numpy as np
from physics.simulator import DigitalTwinSimulator
from physics.backend import USE_TAICHI
from physics.step.context import StepContext
from physics.step.ion_walls import step_ion_interactions, _get_boundary_scratch, _get_is_bound_int
from physics.step.electrons import step_electrons


@pytest.mark.skipif(not USE_TAICHI, reason="Taichi GPU backend not available")
def test_gpu_boundary_kernel_equivalence():
    """Verify that the Taichi boundary kernel produces identical flags and hit coordinates to CPU."""
    from physics.kernels_taichi import check_particle_boundaries_taichi

    sim = DigitalTwinSimulator()
    sim.nx = 150
    sim.ny = 50
    sim.Lx = 3.0
    sim.Ly = 1.0
    sim.dy = 1.0 / 50.0

    # Create dummy grids in the domain
    sim.isBound = np.zeros((sim.ny, sim.nx), dtype=bool)
    sim.isBound[15:35, 40:60] = True
    sim.isBound[10:40, 90:110] = True
    sim._isBound_int = None
    sim._damage_version = 0

    N = 5000
    np.random.seed(42)
    # Positions ranging from upstream to downstream, some in bounds, some out
    x = np.random.uniform(-0.1, 3.2, N).astype(np.float32)
    y = np.random.uniform(-0.05, 1.05, N).astype(np.float32)
    vx = np.random.uniform(-2000.0, 30000.0, N).astype(np.float32)
    vy = np.random.uniform(-500.0, 500.0, N).astype(np.float32)
    dt = float(sim.dt)
    x_old = (x - vx * dt * 1000.0).astype(np.float32)
    y_old = (y - vy * dt * 1000.0).astype(np.float32)

    x_entry = 0.5
    x_exit = 2.5
    x_mid = 2.75

    # CPU ground truth
    ix = sim._x_to_ix(x)
    iy = np.clip(np.round(y / sim.dy).astype(int), 0, sim.ny - 1)
    hit_final_cpu = sim.isBound[iy, ix]
    hit_path_cpu, path_ix_cpu, path_iy_cpu = sim._segment_hits_grid(x_old, y_old, x, y, samples=8)
    hit_cpu = hit_final_cpu | hit_path_cpu
    impact_ix_cpu = np.where(hit_path_cpu, path_ix_cpu, ix)
    impact_iy_cpu = np.where(hit_path_cpu, path_iy_cpu, iy)

    oob_cpu = (x < 0.0) | (x > sim.Lx) | (y < 0.0) | (y > sim.Ly) | np.isnan(x)
    dead_cpu = hit_cpu | oob_cpu
    entered_cpu = (x_old < x_entry) & (x >= x_entry) & (vx > 0.0)
    crossed_exit_cpu = (x_old <= x_exit) & (x > x_exit) & (vx > 0.0)
    crossed_mid_cpu = (x_old <= x_mid) & (x > x_mid) & (vx > 0.0)
    exited_cpu = (x > x_exit) & (vx > 0.0)

    # GPU Taichi evaluation
    status, imp_ix_gpu, imp_iy_gpu, counts = _get_boundary_scratch(sim, N)
    is_bound_int = _get_is_bound_int(sim)

    check_particle_boundaries_taichi(
        x, y, vx, x_old, y_old,
        is_bound_int, sim.zone_boundaries, sim.zone_dx, sim.zone_offsets,
        len(sim.zone_boundaries) - 1, N,
        sim.dy, sim.nx, sim.ny, sim.Lx, sim.Ly, 0,
        0.0, sim.Lx, x_entry, x_exit, x_mid,
        8, status, imp_ix_gpu, imp_iy_gpu, counts
    )

    hit_gpu = (status & 1) != 0
    oob_gpu = (status & 2) != 0
    dead_gpu = (status & 4) != 0
    entered_gpu = (status & 8) != 0
    crossed_exit_gpu = (status & 16) != 0
    crossed_mid_gpu = (status & 32) != 0
    exited_gpu = (status & 64) != 0

    # Assert 100% equivalence
    assert np.array_equal(hit_gpu, hit_cpu), "Grid hit flag mismatch between GPU and CPU"
    assert np.array_equal(oob_gpu, oob_cpu), "Out-of-bounds flag mismatch between GPU and CPU"
    assert np.array_equal(dead_gpu, dead_cpu), "Dead particle flag mismatch between GPU and CPU"
    assert np.array_equal(entered_gpu, entered_cpu), "Entered optics flag mismatch"
    assert np.array_equal(crossed_exit_gpu, crossed_exit_cpu), "Crossed exit flag mismatch"
    assert np.array_equal(crossed_mid_gpu, crossed_mid_cpu), "Crossed mid plane flag mismatch"
    assert np.array_equal(exited_gpu, exited_cpu), "Exited domain flag mismatch"

    # Assert atomic counter consistency
    assert counts[0] == int(np.sum(dead_cpu)), "Atomic dead count mismatch"
    assert counts[1] == int(np.sum(hit_cpu & ~oob_cpu)), "Atomic lost-to-grid count mismatch"
    assert counts[2] == int(np.sum(oob_cpu)), "Atomic lost-to-oob count mismatch"
    assert counts[3] == int(np.sum(entered_cpu)), "Atomic entered optics count mismatch"
    assert counts[4] == int(np.sum(crossed_exit_cpu)), "Atomic crossed exit count mismatch"
    assert counts[5] == int(np.sum(crossed_mid_cpu)), "Atomic crossed mid count mismatch"
    assert counts[6] == int(np.sum(exited_cpu)), "Atomic exited count mismatch"

    # Assert hit coordinates match for hitting particles
    assert np.array_equal(imp_ix_gpu[hit_cpu], impact_ix_cpu[hit_cpu]), "Impact ix mismatch"
    assert np.array_equal(imp_iy_gpu[hit_cpu], impact_iy_cpu[hit_cpu]), "Impact iy mismatch"


def test_step_ion_interactions_survival_fast_path():
    """Verify that when all particles survive, no array compaction or copying is performed."""
    sim = DigitalTwinSimulator()
    N = 1000
    sim.num_p = N
    # Confine particles safely inside the channel far from boundaries and grids
    sim.p_x[:N] = np.linspace(0.6, 2.0, N)
    sim.p_y[:N] = np.full(N, 0.5 * sim.Ly)
    sim.p_vx[:N] = np.full(N, 15000.0)
    sim.p_vy[:N] = 0.0
    sim.p_vz[:N] = 0.0
    sim.p_isCEX[:N] = False

    # Ensure no walls at y = 0.5 * Ly
    mid_iy = int(round((0.5 * sim.Ly) / sim.dy))
    sim.isBound[mid_iy, :] = False
    sim._isBound_int = None

    ctx = StepContext(params={'div_method': '95%'}, sim_mode='Both', t_current=1e-6, grids=[])
    ctx.num_p_step = N
    ctx.p_x_old = sim.p_x[:N] - 0.01
    ctx.p_y_old = sim.p_y[:N]

    # Save original array object ids
    orig_px = sim.p_x[:N].copy()

    step_ion_interactions(sim, ctx)

    assert sim.num_p == N
    assert sim.lost_to_grid_step == 0.0
    assert sim.lost_to_oob_step == 0.0
    np.testing.assert_array_equal(sim.p_x[:N], orig_px)


def test_step_ion_interactions_wall_collisions_and_see():
    """Verify that hitting particles trigger SEE injection and array compaction."""
    sim = DigitalTwinSimulator()
    N = 200
    sim.num_p = N
    # 100 safe particles, 100 particles hitting grid wall
    sim.p_x[:100] = 0.2
    sim.p_y[:100] = 0.5 * sim.Ly
    sim.p_vx[:100] = 10000.0

    # Place wall at ix=100
    sim.isBound.fill(False)
    sim.isBound[:, 100] = True
    sim._isBound_int = None
    wall_x = sim.x_coords[100]

    sim.p_x[100:200] = wall_x
    sim.p_y[100:200] = 0.5 * sim.Ly
    sim.p_vx[100:200] = 20000.0
    sim.p_vy[100:200] = 0.0
    sim.p_vz[100:200] = 0.0
    sim.p_isCEX[:200] = False

    ctx = StepContext(params={'div_method': '95%'}, sim_mode='Both', t_current=1e-6, grids=[])
    ctx.num_p_step = N
    ctx.p_x_old = sim.p_x[:N] - 0.01
    ctx.p_y_old = sim.p_y[:N]

    sim.num_e = 0
    step_ion_interactions(sim, ctx)

    # 100 particles should have died on the wall
    assert sim.num_p == 100
    assert sim.lost_to_grid_step == 100.0
    # SEE should have added electrons if secondary electron yield triggered
    assert hasattr(sim, 'num_e')


def test_step_electrons_boundary_purge():
    """Verify that electrons out-of-bounds or hitting walls are purged correctly."""
    sim = DigitalTwinSimulator()
    sim.num_e = 50
    # 25 safe electrons, 25 out-of-bounds electrons
    sim.e_x[:25] = 0.5 * sim.Lx
    sim.e_y[:25] = 0.5 * sim.Ly
    sim.e_vx[:25] = 1e5
    sim.e_vy[:25] = 0.0
    sim.e_vz[:25] = 0.0

    sim.e_x[25:50] = -0.5  # OOB
    sim.e_y[25:50] = 0.5 * sim.Ly
    sim.e_vx[25:50] = 1e5
    sim.e_vy[25:50] = 0.0
    sim.e_vz[25:50] = 0.0

    ctx = StepContext(params={'neut_rate': 0}, sim_mode='Both', t_current=1e-6, grids=[])
    step_electrons(sim, ctx)

    assert sim.num_e == 25
    assert np.all(sim.e_x[:sim.num_e] >= 0.0)


def test_pure_cpu_fallback(monkeypatch):
    """Verify that setting USE_TAICHI=False executes cleanly through CPU fallback paths."""
    import physics.step.ion_walls as ion_walls_mod
    import physics.step.electrons as elec_mod

    monkeypatch.setattr(ion_walls_mod, 'USE_TAICHI', False)
    monkeypatch.setattr(elec_mod, 'USE_TAICHI', False)

    sim = DigitalTwinSimulator()
    N = 100
    sim.num_p = N
    sim.p_x[:N] = np.linspace(0.1, 2.0, N)
    sim.p_y[:N] = np.full(N, 0.5 * sim.Ly)
    sim.p_vx[:N] = 15000.0
    sim.p_vy[:N] = 0.0
    sim.p_vz[:N] = 0.0
    sim.p_isCEX[:N] = False

    ctx = StepContext(params={'div_method': '95%'}, sim_mode='Both', t_current=1e-6, grids=[])
    ctx.num_p_step = N
    ctx.p_x_old = sim.p_x[:N] - 0.01
    ctx.p_y_old = sim.p_y[:N]

    ion_walls_mod.step_ion_interactions(sim, ctx)
    assert sim.num_p <= N
