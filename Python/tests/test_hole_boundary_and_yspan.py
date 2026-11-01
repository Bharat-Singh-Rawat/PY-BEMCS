"""
Tests for domain y-span calculation, hole-to-boundary distance configuration,
and multi-geometry hole positioning.
"""
import unittest
import numpy as np

from physics.simulator import DigitalTwinSimulator


class TestHoleBoundaryAndYSpan(unittest.TestCase):
    def setUp(self):
        self.sim = DigitalTwinSimulator()
        self.base_params = {
            "n0_plasma": 1e17,
            "Te_up": 3.0,
            "Lx": 6.0,
            "pitch_mm": 3.0,
            r"\deltax/debye_length": 0.8,
            "grids": [
                {"V": 1100.0, "t": 0.38, "gap": 0.58, "r": 0.80, "cham": 0.0},
                {"V": -180.0, "t": 0.51, "gap": 2.0,  "r": 0.60, "cham": 0.0},
            ]
        }

    def test_default_half_hole_yspan(self):
        """half_hole default Ly = 0.5 * (r_screen + pitch)"""
        p = dict(self.base_params)
        p["geometry"] = "half_hole"
        self.sim.build_domain(p)

        screen_r = 0.80
        pitch = 3.0
        expected_Ly = 0.5 * (screen_r + pitch)  # 1.90 mm
        self.assertAlmostEqual(self.sim.Ly, expected_Ly, places=3)
        self.assertEqual(self.sim.hole_centers, [0.0])
        self.assertAlmostEqual(self.sim.hole_boundary_dist_mm, expected_Ly, places=3)

    def test_default_one_hole_yspan_and_centering(self):
        """one_hole default Ly = r_screen + pitch with hole centered"""
        p = dict(self.base_params)
        p["geometry"] = "one_hole"
        self.sim.build_domain(p)

        screen_r = 0.80
        pitch = 3.0
        expected_Ly = screen_r + pitch  # 3.80 mm
        self.assertAlmostEqual(self.sim.Ly, expected_Ly, places=3)
        self.assertEqual(len(self.sim.hole_centers), 1)
        self.assertAlmostEqual(self.sim.hole_centers[0], 0.5 * expected_Ly, places=3)
        self.assertAlmostEqual(self.sim.hole_boundary_dist_mm, 0.5 * (screen_r + pitch), places=3)

    def test_default_two_holes_yspan_and_positions(self):
        """two_holes default Ly = 2 * (r_screen + pitch) with dual unit-cell positioning"""
        p = dict(self.base_params)
        p["geometry"] = "two_holes"
        self.sim.build_domain(p)

        screen_r = 0.80
        pitch = 3.0
        expected_Ly = 2.0 * (screen_r + pitch)  # 7.60 mm
        self.assertAlmostEqual(self.sim.Ly, expected_Ly, places=3)
        self.assertEqual(len(self.sim.hole_centers), 2)
        d_bound = 0.5 * (screen_r + pitch)  # 1.90 mm
        self.assertAlmostEqual(self.sim.hole_centers[0], d_bound, places=3)
        self.assertAlmostEqual(self.sim.hole_centers[1], 3.0 * d_bound, places=3)

    def test_custom_hole_boundary_distance(self):
        """User-specified distance between hole center and boundary overrides default Ly across all modes."""
        custom_d = 2.5  # mm

        # 1. half_hole
        p_half = dict(self.base_params)
        p_half["geometry"] = "half_hole"
        p_half["hole_boundary_dist_mm"] = custom_d
        self.sim.build_domain(p_half)
        self.assertAlmostEqual(self.sim.Ly, custom_d, places=3)
        self.assertEqual(self.sim.hole_centers, [0.0])

        # 2. one_hole (Ly = 2 * d, hole centered at d)
        p_one = dict(self.base_params)
        p_one["geometry"] = "one_hole"
        p_one["hole_boundary_dist_mm"] = custom_d
        self.sim.build_domain(p_one)
        self.assertAlmostEqual(self.sim.Ly, 2.0 * custom_d, places=3)
        self.assertAlmostEqual(self.sim.hole_centers[0], custom_d, places=3)

        # 3. two_holes (Ly = 4 * d, holes at d and 3*d)
        p_two = dict(self.base_params)
        p_two["geometry"] = "two_holes"
        p_two["hole_boundary_dist_mm"] = custom_d
        self.sim.build_domain(p_two)
        self.assertAlmostEqual(self.sim.Ly, 4.0 * custom_d, places=3)
        self.assertAlmostEqual(self.sim.hole_centers[0], custom_d, places=3)
        self.assertAlmostEqual(self.sim.hole_centers[1], 3.0 * custom_d, places=3)


if __name__ == "__main__":
    unittest.main()

