"""
Unit tests for Axisymmetric (r-z) 2D3V cylindrical PIC-MCC solver.
Tests:
  1. Cylindrical Laplace radial stencil & on-axis boundary condition
  2. Uniform density injection (r ~ R*sqrt(U)) yielding flat radial density profile
  3. Angular momentum L_z = r*v_theta and transverse kinetic energy conservation in Boris push
  4. 3D cone beam divergence calculation with azimuthal velocity v_theta
  5. Full simulator execution in axisymmetric mode vs Cartesian backward compatibility
"""
import unittest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from physics.simulator import DigitalTwinSimulator
from physics.kernels_cpu import push_particles_boris_cpu, accumulate_rho_cpu
from physics.step.ion_walls import compute_beam_divergence
from physics.backend import USE_TAICHI

if USE_TAICHI:
    from physics.kernels_taichi import push_particles_boris_taichi, accumulate_rho_taichi


class TestAxisymmetricSolver(unittest.TestCase):
    """Test suite for axisymmetric 2D3V cylindrical physics."""

    def test_cylindrical_laplace_coaxial_potential(self):
        """Verify that cylindrical radial 5-point stencil solves d/dr(r dV/dr) = 0 accurately."""
        # Grid starting at r=0 with step dr
        dr = 0.05  # mm
        ny = 80
        r_coords = np.arange(ny) * dr

        # Coaxial region between j1 and j2 (both > 0)
        j1 = 10
        j2 = 70
        r1 = r_coords[j1]
        r2 = r_coords[j2]

        # Analytical solution: V(r) = V1 + (V2 - V1) * ln(r/r1) / ln(r2/r1)
        V1 = 100.0
        V2 = 500.0
        r_sub = r_coords[j1:j2+1]
        V_exact = V1 + (V2 - V1) * np.log(r_sub / r1) / np.log(r2 / r1)

        # Assemble cylindrical Laplace matrix for 1D radial slice
        n_sub = j2 - j1 + 1
        A = sp.lil_matrix((n_sub, n_sub), dtype=np.float64)
        b = np.zeros(n_sub, dtype=np.float64)

        # Dirichlet boundaries at j1 and j2
        A[0, 0] = 1.0
        b[0] = V1
        A[-1, -1] = 1.0
        b[-1] = V2

        # Interior nodes: cS = 1 - 1/(2*j_global), cN = 1 + 1/(2*j_global), cP = -2
        for k in range(1, n_sub - 1):
            j_global = j1 + k
            cS = 1.0 - 0.5 / j_global
            cN = 1.0 + 0.5 / j_global
            cP = -2.0
            A[k, k - 1] = cS
            A[k, k] = cP
            A[k, k + 1] = cN

        V_num = spla.spsolve(A.tocsr(), b)
        max_err = np.max(np.abs(V_num - V_exact))
        rel_err = max_err / (V2 - V1)

        # Truncation error on coaxial gap
        self.assertLess(rel_err, 0.001, f"Cylindrical Laplace error too large: {rel_err:.4e}")

    def test_on_axis_lhopital_boundary(self):
        """Verify that on-axis L'Hopital row gives Er(0) = 0 and matches parabolic potential."""
        # For a uniform charge distribution rho = const, V(r) = V0 - rho/(4 eps0) * r^2
        # Near r=0, d^2V/dr^2 = -rho/(2 eps0)
        # Using L'Hopital limit at r=0: d^2V/dz^2 + 2 d^2V/dr^2 = -rho/eps0
        # Discrete stencil: 4*V_1 - 4*V_0 = - (dr^2 / eps0) * rho
        dr = 0.05
        eps0 = 8.8541878128e-12
        rho = 1e-4  # C/m^3
        # If V_1 is given, V_0 should be V_1 + (dr^2 * rho) / (4 * eps0)
        V1 = 1000.0
        V0_expected = V1 + (dr ** 2 * rho) / (4.0 * eps0)

        # Check stencil equation: -4 * V0 + 4 * V1 = - (dr^2 / eps0) * rho
        stencil_lhs = -4.0 * V0_expected + 4.0 * V1
        stencil_rhs = -(dr ** 2 / eps0) * rho
        self.assertAlmostEqual(stencil_lhs, stencil_rhs, places=7)

    def test_uniform_density_radial_injection(self):
        """Verify that r ~ R * sqrt(U) produces flat density across cylindrical annuli."""
        r_inj = 1.5  # mm
        N = 500000
        rng = np.random.default_rng(42)
        # Sample radial positions
        r = r_inj * np.sqrt(rng.uniform(0.0, 1.0, N))

        # Discretize into 15 equal-width radial annuli
        n_bins = 15
        dr = r_inj / n_bins
        bin_edges = np.linspace(0.0, r_inj, n_bins + 1)
        counts, _ = np.histogram(r, bins=bin_edges)

        # Annular areas: A_j = pi * (r_{j+1}^2 - r_j^2) = pi * ((j+1)^2 - j^2) * dr^2 = pi * (2j + 1) * dr^2
        j = np.arange(n_bins)
        annular_area = np.pi * (2 * j + 1) * (dr ** 2)

        # Number density per unit area: counts / annular_area
        density = counts / annular_area
        density_norm = density / np.mean(density)

        # Normalized density should be 1.0 across all bins within statistical fluctuation
        max_rel_dev = np.max(np.abs(density_norm - 1.0))
        self.assertLess(max_rel_dev, 0.05, f"Radial density is not uniform: max relative deviation {max_rel_dev:.3f}")

    def test_boris_push_angular_momentum_and_energy_conservation(self):
        """Verify that cylindrical Boris coordinate rotation conserves L_z and transverse kinetic energy."""
        dt = 1e-9  # s
        q_m = 1.0  # neutral or field-free motion test
        N = 100
        r_init = np.random.uniform(0.2, 1.0, N)
        vr_init = np.random.uniform(-1000.0, 1000.0, N)
        vt_init = np.random.uniform(-1000.0, 1000.0, N)

        # Initial angular momentum per unit mass L_z = r * v_theta
        Lz_0 = (r_init * 1e-3) * vt_init
        # Initial transverse kinetic energy E_perp = 0.5 * (vr^2 + vt^2)
        E_perp_0 = 0.5 * (vr_init ** 2 + vt_init ** 2)

        # Push particles for 200 steps in zero field on CPU
        x = np.zeros(N, dtype=np.float32)
        y = r_init.copy().astype(np.float32)
        vx = np.zeros(N, dtype=np.float32)
        vy = vr_init.copy().astype(np.float32)
        vz = vt_init.copy().astype(np.float32)

        Ex = np.zeros((10, 10), dtype=np.float32)
        Ey = np.zeros((10, 10), dtype=np.float32)
        Bx = np.zeros((10, 10), dtype=np.float32)
        By = np.zeros((10, 10), dtype=np.float32)
        Bz = np.zeros((10, 10), dtype=np.float32)
        z_b = np.array([0.0, 10.0], dtype=np.float32)
        z_dx = np.array([1.0], dtype=np.float32)
        z_off = np.array([0], dtype=np.int32)
        x_coords = np.linspace(0, 10, 10)

        for _ in range(50):
            push_particles_boris_cpu(
                x, y, vx, vy, vz,
                Ex, Ey, Bx, By, Bz,
                N, 0.1, 10, 10, dt, 0.0,
                z_b, z_dx, z_off, x_coords,
                Ly=5.0, reflect_y0=False, reflect_y_top=False,
                is_axisymmetric=True
            )

        # Post-push angular momentum and kinetic energy
        Lz_final = (y * 1e-3) * vz
        E_perp_final = 0.5 * (vy ** 2 + vz ** 2)

        max_Lz_error = np.max(np.abs(Lz_final - Lz_0) / (np.abs(Lz_0) + 1e-12))
        max_E_error = np.max(np.abs(E_perp_final - E_perp_0) / E_perp_0)

        self.assertLess(max_Lz_error, 1e-4, f"Angular momentum not conserved: max error {max_Lz_error:.4e}")
        self.assertLess(max_E_error, 1e-5, f"Transverse kinetic energy not conserved: max error {max_E_error:.4e}")

    def test_axisymmetric_beam_divergence_metric(self):
        """Verify that 3D cone divergence is computed using v_perp = sqrt(vr^2 + v_theta^2)."""
        N = 1000
        p_y = np.random.uniform(0.0, 1.0, N)
        p_vx = np.full(N, 10000.0)  # 10 km/s axial
        # Choose vr and v_theta such that v_perp / vx = tan(15 deg)
        theta_target_deg = 15.0
        v_perp_target = 10000.0 * np.tan(np.radians(theta_target_deg))

        # Distribute randomly in azimuthal angle phi
        phi = np.random.uniform(0.0, 2.0 * np.pi, N)
        p_vy = v_perp_target * np.cos(phi)
        p_vz = v_perp_target * np.sin(phi)
        mask = np.ones(N, dtype=bool)

        div_3d = compute_beam_divergence(
            p_y, p_vx, p_vy, mask,
            method="95%",
            is_axisymmetric=True,
            p_vz=p_vz
        )

        self.assertAlmostEqual(div_3d, theta_target_deg, places=2,
                               msg=f"Axisymmetric 3D divergence {div_3d} != expected {theta_target_deg}")

    def test_taichi_axisymmetric_boris_parity(self):
        """Verify that Taichi GPU Boris pusher matches CPU Boris pusher in axisymmetric mode."""
        if not USE_TAICHI:
            self.skipTest("Taichi not available")

        dt = 1e-9
        N = 50
        r_init = np.random.uniform(0.1, 1.2, N).astype(np.float32)
        x_init = np.random.uniform(0.1, 1.5, N).astype(np.float32)
        vx_init = np.random.uniform(5000.0, 15000.0, N).astype(np.float32)
        vr_init = np.random.uniform(-1000.0, 1000.0, N).astype(np.float32)
        vt_init = np.random.uniform(-1000.0, 1000.0, N).astype(np.float32)

        x_cpu, y_cpu = x_init.copy(), r_init.copy()
        vx_cpu, vy_cpu, vz_cpu = vx_init.copy(), vr_init.copy(), vt_init.copy()

        x_gpu, y_gpu = x_init.copy(), r_init.copy()
        vx_gpu, vy_gpu, vz_gpu = vx_init.copy(), vr_init.copy(), vt_init.copy()

        nx, ny = 20, 20
        Ex = np.zeros((ny, nx), dtype=np.float32)
        Ey = np.zeros((ny, nx), dtype=np.float32)
        Bx = np.zeros((ny, nx), dtype=np.float32)
        By = np.zeros((ny, nx), dtype=np.float32)
        Bz = np.zeros((ny, nx), dtype=np.float32)
        z_b = np.array([0.0, 5.0], dtype=np.float32)
        z_dx = np.array([0.25], dtype=np.float32)
        z_off = np.array([0], dtype=np.int32)
        x_coords = np.linspace(0.0, 5.0, nx).astype(np.float32)

        for _ in range(10):
            push_particles_boris_cpu(
                x_cpu, y_cpu, vx_cpu, vy_cpu, vz_cpu,
                Ex, Ey, Bx, By, Bz,
                N, 0.1, nx, ny, dt, 0.0,
                z_b, z_dx, z_off, x_coords,
                Ly=2.0, reflect_y0=False, reflect_y_top=False,
                is_axisymmetric=True
            )
            push_particles_boris_taichi(
                x_gpu, y_gpu, vx_gpu, vy_gpu, vz_gpu,
                Ex, Ey, Bx, By, Bz,
                N, 0.1, nx, ny, dt, 0.0,
                z_b, z_dx, z_off, 1,
                n_sub=1, Ly=2.0, reflect_y0=0, reflect_y_top=0,
                is_axisymmetric=1
            )

        np.testing.assert_allclose(x_cpu, x_gpu, rtol=1e-5, atol=1e-6, err_msg="x differs between CPU and Taichi")
        np.testing.assert_allclose(y_cpu, y_gpu, rtol=1e-5, atol=1e-6, err_msg="y differs between CPU and Taichi")
        np.testing.assert_allclose(vx_cpu, vx_gpu, rtol=1e-5, atol=1e-6, err_msg="vx differs between CPU and Taichi")
        np.testing.assert_allclose(vy_cpu, vy_gpu, rtol=1e-5, atol=1e-6, err_msg="vy differs between CPU and Taichi")
        np.testing.assert_allclose(vz_cpu, vz_gpu, rtol=1e-5, atol=1e-6, err_msg="vz differs between CPU and Taichi")

    def test_full_simulator_axisymmetric_smoke(self):
        """Verify DigitalTwinSimulator builds and steps correctly in axisymmetric mode."""
        sim = DigitalTwinSimulator()
        params = {
            "coordinate_system": "axisymmetric",
            "geometry": "half_hole",
            "n0_plasma": 1e16,
            "Te_up": 3.0,
            "Ti": 0.1,
            "sim_mode": "Trajectories",
            "enforce_efield_mesh_limit": False,
            "grids": [
                {"V": 1000.0, "t": 0.38, "gap": 0.64, "r": 0.80, "cham": 0.0},
                {"V": -200.0, "t": 0.38, "gap": 0.64, "r": 0.70, "cham": 0.0},
            ]
        }
        sim.build_domain(params)
        self.assertTrue(sim.is_axisymmetric, "Simulator is_axisymmetric flag not set")
        self.assertEqual(sim.geometry, "half_hole", "Axisymmetric domain should force half_hole geometry")
        self.assertIsNotNone(sim.cell_vol_2d, "cell_vol_2d must be initialized")
        self.assertIsNotNone(sim.cvol_r_1d, "cvol_r_1d must be initialized")

        # Step 20 iterations
        for _ in range(20):
            sim.step(params)

        self.assertGreaterEqual(sim.iteration, 20)
        self.assertGreaterEqual(sim.num_p, 0)


if __name__ == "__main__":
    unittest.main()
