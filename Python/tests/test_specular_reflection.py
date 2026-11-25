"""
Tests for specular reflection at y=0 / y=Ly in half_hole geometry and half-domain divergence mirroring.
"""
import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physics.backend import USE_TAICHI
from physics.kernels_cpu import push_particles_boris_cpu
from physics.simulator import DigitalTwinSimulator
from physics.step.context import StepContext
from physics.step.ion_walls import compute_beam_divergence, step_ion_interactions


class TestSpecularReflection(unittest.TestCase):
    def _make_dummy_sim(self, geometry="half_hole"):
        sim = DigitalTwinSimulator()
        sim.Lx = 4.0
        sim.Ly = 2.0
        sim.dx = 0.1
        sim.dy = 0.1
        sim.nx = int(sim.Lx / sim.dx) + 1
        sim.ny = int(sim.Ly / sim.dy) + 1
        sim.x_coords = np.linspace(0.0, sim.Lx, sim.nx)
        sim.y_coords = np.linspace(0.0, sim.Ly, sim.ny)
        sim.zone_boundaries = np.array([0.0, sim.Lx / 3.0, 2.0 * sim.Lx / 3.0, sim.Lx], dtype=np.float64)
        sim.zone_dx = np.array([sim.dx, sim.dx, sim.dx], dtype=np.float64)
        sim.zone_offsets = np.array([0, int(sim.nx / 3), int(2 * sim.nx / 3)], dtype=np.int32)
        sim.Ex = np.zeros((sim.ny, sim.nx), dtype=np.float32)
        sim.Ey = np.zeros((sim.ny, sim.nx), dtype=np.float32)
        sim.Bx = np.zeros((sim.ny, sim.nx), dtype=np.float32)
        sim.By = np.zeros((sim.ny, sim.nx), dtype=np.float32)
        sim.Bz = np.zeros((sim.ny, sim.nx), dtype=np.float32)
        sim.isBound = np.zeros((sim.ny, sim.nx), dtype=bool)
        sim.geometry = geometry
        sim.periodic_y = geometry in ("one_hole", "two_holes")
        sim.reflect_y0 = (geometry == "half_hole")
        sim.reflect_y_top = (geometry == "half_hole")
        sim.grid_x_starts = [0.5]
        sim.grid_x_ends = [1.0]
        sim._update_taichi_mesh_constants()
        return sim

    def test_cpu_pusher_reflects_y0_and_ly(self):
        sim = self._make_dummy_sim("half_hole")
        px = np.array([1.0, 1.0], dtype=np.float32)
        py = np.array([0.05, 1.95], dtype=np.float32)
        vx = np.array([10000.0, 10000.0], dtype=np.float32)
        vy = np.array([-200000.0, 200000.0], dtype=np.float32)  # moves 0.1 mm in 0.5 ns -> crosses y=0 and y=Ly
        vz = np.zeros(2, dtype=np.float32)
        dt = 5e-10

        push_particles_boris_cpu(
            px, py, vx, vy, vz,
            sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
            2, sim.dy, sim.nx, sim.ny, dt, 1.0,
            sim.zone_boundaries, sim.zone_dx, sim.zone_offsets, sim.x_coords,
            Ly=sim.Ly, reflect_y0=True, reflect_y_top=True
        )

        # Particle 0: 0.05 - 0.10 = -0.05 -> reflects to +0.05, vy -> +200000
        self.assertAlmostEqual(float(py[0]), 0.05, places=4)
        self.assertAlmostEqual(float(vy[0]), 200000.0, places=1)
        # Particle 1: 1.95 + 0.10 = 2.05 -> reflects to 1.95, vy -> -200000
        self.assertAlmostEqual(float(py[1]), 1.95, places=4)
        self.assertAlmostEqual(float(vy[1]), -200000.0, places=1)

    @unittest.skipUnless(USE_TAICHI, "Taichi GPU backend not available")
    def test_taichi_pusher_matches_cpu_specular_reflection(self):
        from physics.kernels_taichi import push_particles_boris_taichi
        import taichi as ti

        sim = self._make_dummy_sim("half_hole")
        px_cpu = np.array([1.0, 1.0], dtype=np.float32)
        py_cpu = np.array([0.05, 1.95], dtype=np.float32)
        vx_cpu = np.array([10000.0, 10000.0], dtype=np.float32)
        vy_cpu = np.array([-200000.0, 200000.0], dtype=np.float32)
        vz_cpu = np.zeros(2, dtype=np.float32)

        px_gpu = px_cpu.copy()
        py_gpu = py_cpu.copy()
        vx_gpu = vx_cpu.copy()
        vy_gpu = vy_cpu.copy()
        vz_gpu = vz_cpu.copy()

        dt = 5e-10
        n_sub = 4
        for _ in range(n_sub):
            push_particles_boris_cpu(
                px_cpu, py_cpu, vx_cpu, vy_cpu, vz_cpu,
                sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
                2, sim.dy, sim.nx, sim.ny, dt / n_sub, 1.0,
                sim.zone_boundaries, sim.zone_dx, sim.zone_offsets, sim.x_coords,
                Ly=sim.Ly, reflect_y0=True, reflect_y_top=True
            )

        push_particles_boris_taichi(
            px_gpu, py_gpu, vx_gpu, vy_gpu, vz_gpu,
            sim.Ex, sim.Ey, sim.Bx, sim.By, sim.Bz,
            2, sim._dy_f32, sim.nx, sim.ny, np.float32(dt / n_sub), np.float32(1.0),
            sim.zone_b_ti, sim.zone_dx_ti, sim.zone_off_ti, sim.num_zones,
            n_sub=n_sub, Ly=float(sim.Ly), reflect_y0=1, reflect_y_top=1
        )
        ti.sync()

        np.testing.assert_allclose(py_gpu, py_cpu, atol=1e-4)
        np.testing.assert_allclose(vy_gpu, vy_cpu, atol=1.0)

    def test_reflected_particles_not_killed_as_oob(self):
        sim = self._make_dummy_sim("half_hole")
        N = 20
        sim.num_p = N
        sim.p_x[:N] = 1.5
        sim.p_y[:N] = np.linspace(0.01, 0.2, N)
        sim.p_vx[:N] = 20000.0
        sim.p_vy[:N] = -5000.0
        sim.p_vz[:N] = 0.0
        sim.p_isCEX[:N] = False

        ctx = StepContext(params={"geometry": "half_hole"}, sim_mode="Trajectories", t_current=sim.dt, grids=[])
        ctx.num_p_step = N
        ctx.p_x_old = sim.p_x[:N].copy()
        ctx.p_y_old = sim.p_y[:N].copy()

        # Simulate step_push + step_ion_interactions when particles would cross y=0
        sim.p_y[:N] = np.abs(sim.p_y[:N] + sim.p_vy[:N] * sim.dt * 1000.0)
        step_ion_interactions(sim, ctx)

        self.assertEqual(sim.num_p, N)
        self.assertEqual(sim.lost_to_oob_step, 0.0)

    def test_half_domain_divergence_mirroring_matches_full_domain(self):
        rng = np.random.default_rng(7)
        N = 500
        vx = np.full(N, 30000.0)
        angles_full = rng.normal(0.0, 7.5, N)
        y_full = rng.uniform(-1.0, 1.0, N)
        # Ensure perfect sign symmetry for exact check
        y_pos = np.abs(y_full)
        angles_pos = np.abs(angles_full)
        vy_pos = vx * np.tan(np.deg2rad(angles_pos))

        y_sym = np.concatenate((y_pos, -y_pos))
        vx_sym = np.concatenate((vx, vx))
        vy_sym = np.concatenate((vy_pos, -vy_pos))

        mask_half = np.ones(N, dtype=bool)
        mask_full = np.ones(2 * N, dtype=bool)

        for method in ("95%", "90%", "Direct RMS", "RMS (Twiss)", "FWHM"):
            d_half = compute_beam_divergence(
                y_pos, vx, vy_pos, mask_half, method=method, symmetric_half_domain=True
            )
            d_full = compute_beam_divergence(
                y_sym, vx_sym, vy_sym, mask_full, method=method, symmetric_half_domain=False
            )
            self.assertAlmostEqual(d_half, d_full, places=5, msg=f"Mismatch for method={method}")


if __name__ == "__main__":
    unittest.main()

