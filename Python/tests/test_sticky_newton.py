import os
import sys
import json
import numpy as np
import pytest

# Ensure Python root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics_engine import DigitalTwinSimulator


@pytest.fixture
def base_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configurations", "configNSTAR.json"))
    if not os.path.isfile(config_path):
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configNSTAR.json"))
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def test_sticky_newton_streak_and_direct_activation(base_config):
    """Verify that consecutive Newton fallbacks trigger direct Newton mode."""
    engine = DigitalTwinSimulator()
    engine.build_domain(base_config)

    assert engine._poisson_nr_streak == 0
    assert engine._poisson_direct_nr_remaining == 0

    # Mock Picard to fail and Newton to succeed
    picard_calls = []
    newton_calls = []

    def mock_recalc_cpu(max_iters, coeff, V_plasma, Te_up, n0, *args, **kwargs):
        picard_calls.append(len(picard_calls) + 1)
        # Simulate Picard returning converged_newton via internal fallback
        return 12, 0.01, 0.01, 'converged_newton'

    def mock_newton_cpu(coeff, V_plasma, Te_up, n0, tol_V=0.05, max_iters=15):
        newton_calls.append(len(newton_calls) + 1)
        return engine.V.copy(), 1, 0.01, 'converged_newton'

    engine._recalc_poisson_cpu = mock_recalc_cpu
    engine._solve_poisson_newton_cpu = mock_newton_cpu

    # Test parameters: threshold=3, duration=5
    sim_params = {
        'poisson_sticky_nr': True,
        'poisson_nr_streak_threshold': 3,
        'poisson_nr_direct_duration': 5,
        'n0_plasma': 1e18,
        'grids': [{'V': 1000}],
        'V_plasma_offset': 26.0,
        'Te_up': 4.0
    }

    # Step 1: Fallback #1 -> streak = 1
    engine.recalc_poisson(params=sim_params)
    assert engine._poisson_nr_streak == 1
    assert engine._poisson_direct_nr_remaining == 0
    assert len(picard_calls) == 1
    assert len(newton_calls) == 0

    # Step 2: Fallback #2 -> streak = 2
    engine.recalc_poisson(params=sim_params)
    assert engine._poisson_nr_streak == 2
    assert engine._poisson_direct_nr_remaining == 0
    assert len(picard_calls) == 2

    # Step 3: Fallback #3 -> streak hits 3 -> activates direct NR for 5 steps!
    engine.recalc_poisson(params=sim_params)
    assert engine._poisson_nr_streak == 0
    assert engine._poisson_direct_nr_remaining == 5
    assert len(picard_calls) == 3

    # Now the next 5 calls should go DIRECTLY to Newton without touching Picard!
    picard_calls_before = len(picard_calls)
    newton_calls_before = len(newton_calls)

    for i in range(5):
        expected_remaining = 5 - (i + 1)
        engine.recalc_poisson(params=sim_params)
        assert engine._poisson_direct_nr_remaining == expected_remaining
        assert engine.last_poisson_status == 'converged_newton'
        assert engine.last_poisson_converged is True

    # Picard was NOT called during the direct phase
    assert len(picard_calls) == picard_calls_before
    # Direct Newton was called exactly 5 times
    assert len(newton_calls) == newton_calls_before + 5

    # On step 9, remaining is 0 -> it should probe Picard again!
    engine.recalc_poisson(params=sim_params)
    assert len(picard_calls) == picard_calls_before + 1


def test_sticky_newton_resets_on_picard_convergence(base_config):
    """Verify that if Picard converges, the streak is reset to 0."""
    engine = DigitalTwinSimulator()
    engine.build_domain(base_config)

    sim_params = {
        'poisson_sticky_nr': True,
        'poisson_nr_streak_threshold': 3,
        'poisson_nr_direct_duration': 5,
    }

    # Set streak to 2 manually
    engine._poisson_nr_streak = 2

    # Mock Picard to converge
    def mock_recalc_converged(*args, **kwargs):
        return 5, 0.02, 0.005, 'converged'

    engine._recalc_poisson_cpu = mock_recalc_converged

    engine.recalc_poisson(params=sim_params)
    assert engine._poisson_nr_streak == 0
    assert engine._poisson_direct_nr_remaining == 0
    assert engine.last_poisson_status == 'converged'


def test_sticky_newton_disabled_flag(base_config):
    """Verify that when poisson_sticky_nr is False, streak is not tracked and direct NR is not activated."""
    engine = DigitalTwinSimulator()
    engine.build_domain(base_config)

    sim_params = {
        'poisson_sticky_nr': False,
        'poisson_nr_streak_threshold': 2,
        'poisson_nr_direct_duration': 5,
    }

    def mock_recalc_fallback(*args, **kwargs):
        return 12, 0.01, 0.01, 'converged_newton'

    engine._recalc_poisson_cpu = mock_recalc_fallback

    # Run 5 times with fallback
    for _ in range(5):
        engine.recalc_poisson(params=sim_params)

    assert engine._poisson_nr_streak == 0
    assert engine._poisson_direct_nr_remaining == 0


def test_domain_rebuild_resets_sticky_counters(base_config):
    """Verify that rebuilding the domain resets sticky state."""
    engine = DigitalTwinSimulator()
    engine.build_domain(base_config)

    engine._poisson_nr_streak = 4
    engine._poisson_direct_nr_remaining = 25

    engine.build_domain(base_config)
    assert engine._poisson_nr_streak == 0
    assert engine._poisson_direct_nr_remaining == 0


def test_real_poisson_sticky_newton_integration(base_config):
    """Integration test with actual physics equations triggering sticky Newton."""
    base_config["simulation"]["n0_plasma"] = 2.0e18
    base_config["simulation"]["n0"] = 2.0e18
    base_config["simulation"]["Te_up"] = 4.3

    engine = DigitalTwinSimulator()
    engine.build_domain(base_config)

    sim_params = {
        'poisson_sticky_nr': True,
        'poisson_nr_streak_threshold': 2,
        'poisson_nr_direct_duration': 3,
        'poisson_max_iters': 4, # Force early Picard exit so it triggers Newton fallback
        'n0_plasma': 2.0e18,
        'n0': 2.0e18,
        'Te_up': 4.3,
        'grids': base_config['grids']
    }

    # Perturb V to ensure Picard cannot converge in 4 iters
    engine.V += np.random.normal(0.0, 3.0, size=engine.V.shape)
    engine.V[engine.isBound] = engine.V_fixed[engine.isBound]

    # Step 1: Picard hits max_iters / stagnates -> triggers Newton fallback -> streak = 1
    engine.recalc_poisson(params=sim_params)
    assert engine.last_poisson_converged is True
    assert engine._poisson_nr_streak == 1
    assert engine._poisson_direct_nr_remaining == 0

    # Perturb again to force fallback #2
    engine.V += np.random.normal(0.0, 3.0, size=engine.V.shape)
    engine.V[engine.isBound] = engine.V_fixed[engine.isBound]

    # Step 2: Fallback #2 -> streak hits 2 -> activates direct NR for 3 steps!
    engine.recalc_poisson(params=sim_params)
    assert engine.last_poisson_converged is True
    assert engine._poisson_nr_streak == 0
    assert engine._poisson_direct_nr_remaining == 3

    # Step 3, 4, 5: Direct NR
    for expected_left in [2, 1, 0]:
        engine.recalc_poisson(params=sim_params)
        assert engine.last_poisson_converged is True
        assert engine.last_poisson_status == 'converged_newton'
        assert engine._poisson_direct_nr_remaining == expected_left
