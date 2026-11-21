"""
Unit tests for Plan A: Sub-Grid Structural Surface & Thermal Representation.
"""
import unittest
import numpy as np

from physics.simulator import DigitalTwinSimulator
from physics.structural_surface import (
    SubgridSegment, SubgridApertureProfile, SubgridThermalModel, SubgridManager
)


class TestSubgridSurface(unittest.TestCase):

    def test_subgrid_segment_normals(self):
        """Verify unit normal vectors for upstream, downstream, and chamfer segments."""
        # Upstream face
        s_up = SubgridSegment(1.0, 0.5, 1.0, 1.5, face='upstream', grid_idx=0)
        self.assertAlmostEqual(s_up.nx, -1.0)
        self.assertAlmostEqual(s_up.ny, 0.0)

        # Downstream face
        s_dn = SubgridSegment(2.0, 0.5, 2.0, 1.5, face='downstream', grid_idx=0)
        self.assertAlmostEqual(s_dn.nx, 1.0)
        self.assertAlmostEqual(s_dn.ny, 0.0)

        # Straight barrel (chamfer = 0)
        s_bar = SubgridSegment(1.0, 0.8, 2.0, 0.8, face='barrel', grid_idx=0)
        self.assertAlmostEqual(s_bar.nx, 0.0)
        self.assertAlmostEqual(s_bar.ny, -1.0)  # Inward normal points towards hole axis

    def test_aperture_profile_and_hit_recording(self):
        """Verify continuous erosion deposition and Yamamura angular scaling on subgrid segments."""
        grid = {'t': 0.5, 'gap': 1.0, 'r': 0.8, 'cham': 15.0, 'V': 1000.0}
        prof = SubgridApertureProfile(
            grid_idx=0,
            grid_dict=grid,
            gstart=1.0,
            gend=1.5,
            dxbow=0.0,
            Ly=2.0,
            hole_centers=[0.0],
            n_sub=40
        )
        self.assertGreater(len(prof.segments), 80)

        # Record a normal impact on downstream face at x=1.5, y=1.2
        # vx = -10000 (moving upstream, hitting back face), vy=0, vz=0
        delta_d, seg = prof.record_hit(
            px=1.5, py=1.2,
            vx=-10000.0, vy=0.0, vz=0.0,
            energy_eV=500.0,
            macro_weight=1e4,
            Y_coeff=1e-4,
            E_th=30.0,
            thresh=1e5,
            dy=0.02
        )
        self.assertGreater(delta_d, 0.0)
        self.assertIsNotNone(seg)
        self.assertGreater(seg.eroded_depth_mm, 0.0)
        self.assertEqual(seg.face, 'downstream')

        # Test profile extraction
        y_pts, d_pts = prof.get_face_profile('downstream')
        self.assertGreater(len(y_pts), 0)
        self.assertGreater(np.max(d_pts), 0.0)

    def test_thermal_model_conduction_and_deflection(self):
        """Verify 1D radial heat conduction and cantilever thermal deflection."""
        therm = SubgridThermalModel(
            grid_idx=0,
            r_min=0.8,
            r_max=2.0,
            thickness_mm=0.5,
            k=138.0,
            rho=10280.0,
            cp=250.0,
            emissivity=0.8,
            alpha=4.8e-6,
            n_r=30
        )
        self.assertAlmostEqual(therm.T_mean, 300.0)

        # Deposit 10 Joules at inner aperture edge
        therm.add_heat(r_mm=0.85, energy_joules=10.0)
        t_mean = therm.solve_step(dt=1e-9, thermal_accel=1e7)
        self.assertGreater(t_mean, 300.0)

        # Check deflection
        defl = therm.compute_deflection(L_cant_mm=1.2)
        self.assertGreater(defl, 0.0)

    def test_subgrid_manager_integration_in_simulator(self):
        """Verify that SubgridManager is initialized in build_domain and tracks erosion in step()."""
        sim = DigitalTwinSimulator()
        params = {
            'grids': [
                {'t': 0.5, 'gap': 1.0, 'r': 0.8, 'cham': 0.0, 'V': 1000.0},
                {'t': 0.5, 'gap': 1.0, 'r': 0.6, 'cham': 0.0, 'V': -200.0}
            ],
            'upstream_gap_mm': 1.0,
            'sim_mode': 'Both',
            'Thresh': 1e4
        }
        sim.build_domain(params)

        self.assertIsNotNone(getattr(sim, 'subgrid_manager', None))
        self.assertEqual(len(sim.subgrid_manager.profiles), 2)
        self.assertEqual(len(sim.subgrid_manager.thermal_models), 2)

        # Simulate an erosion impact via SubgridManager
        px = np.array([1.5], dtype=np.float64)
        py = np.array([1.0], dtype=np.float64)
        vx = np.array([50000.0], dtype=np.float64)
        vy = np.array([0.0], dtype=np.float64)
        vz = np.array([0.0], dtype=np.float64)

        sim.subgrid_manager.record_erosion_hits(px, py, vx, vy, vz, sim, params)

        # Verify that sim.eroded_depth received the continuous depth
        self.assertGreater(np.max(sim.eroded_depth), 0.0)

        # Verify get_groove_profile returns continuous curve with positive erosion
        y_data, d_data = sim.get_groove_profile(grid_idx=0, face='downstream')
        self.assertGreater(len(y_data), 0)
        self.assertGreater(np.max(d_data), 0.0)

    def test_get_groove_profile_after_remesh_preserve_state(self):
        """Verify that remeshing with preserve_state (e.g. changing cell counts) does not cause shape mismatch."""
        sim = DigitalTwinSimulator()
        params = {
            'grids': [
                {'t': 0.5, 'gap': 1.0, 'r': 0.8, 'cham': 0.0, 'V': 1000.0},
                {'t': 0.5, 'gap': 1.0, 'r': 0.6, 'cham': 0.0, 'V': -200.0}
            ],
            'upstream_gap_mm': 1.0,
            'sim_mode': 'Both',
            'Thresh': 1e4
        }
        sim.build_domain(params)
        orig_shape = sim.isBound.shape

        # Simulate changing mesh resolution (e.g., e-field refinement or plume reshape)
        sim.dx = sim.dx * 0.8
        sim.dy = sim.dy * 0.8
        sim.build_domain(params, preserve_state=True)
        new_shape = sim.isBound.shape
        self.assertNotEqual(orig_shape, new_shape)

        # get_groove_profile should succeed without shape mismatch broadcasting errors
        y_data, d_data = sim.get_groove_profile(grid_idx=0, thresh=1e4, face='downstream')
        self.assertGreater(len(y_data), 0)
        self.assertEqual(len(y_data), len(d_data))


if __name__ == '__main__':
    unittest.main()
