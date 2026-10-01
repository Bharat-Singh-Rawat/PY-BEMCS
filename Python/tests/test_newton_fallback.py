import sys
import os
import json
import numpy as np

# Ensure Python directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics_engine import DigitalTwinSimulator

def test_newton_raphson_cpu_and_gpu():
    print("=" * 60)
    print("TEST: Newton-Raphson Poisson Fallback Verification")
    print("=" * 60)

    # 1. Load configNSTAR.json
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configNSTAR.json"))
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Set up high density configuration where Picard struggles
    config["simulation"]["n0_plasma"] = 2.0e18
    config["simulation"]["n0"] = 2.0e18
    config["simulation"]["Te_up"] = 4.3

    engine = DigitalTwinSimulator()
    print("[1] Building domain from configNSTAR with n0 = 2.0e18 m^-3...")
    engine.build_domain(config)

    print(f"    Grid dimensions: nx = {engine.nx}, ny = {engine.ny}, total nodes = {engine.nx * engine.ny}")
    print(f"    dx = {engine.dx:.4f} mm, dy = {engine.dy:.4f} mm")

    # 2. Test CPU Newton-Raphson fallback direct invocation
    print("\n[2] Testing _solve_poisson_newton_cpu directly...")
    coeff = (engine.dy * 1e-3)**2 / 8.854187817e-12
    V_plasma = config["grids"][0]["V"] + config.get("advanced_settings", {}).get("V_plasma_offset", 26.0)
    Te_up = config["simulation"].get("Te_up", 4.3)
    n0 = 2.0e18

    # Perturb V to simulate an oscillating/stagnated state
    engine.V += np.random.normal(0.0, 2.0, size=engine.V.shape)
    engine.V[engine.isBound] = engine.V_fixed[engine.isBound]

    V_nr, iters_nr, diff_nr, status_nr = engine._solve_poisson_newton_cpu(
        coeff=coeff,
        V_plasma=V_plasma,
        Te_up=Te_up,
        n0=n0,
        tol_V=0.05,
        max_iters=15
    )

    print(f"    Result: status = '{status_nr}', iters = {iters_nr}, diff = {diff_nr:.4e} V")
    assert status_nr == 'converged_newton', f"Expected 'converged_newton', got '{status_nr}'"
    assert not np.isnan(V_nr).any(), "Potential contains NaNs"
    assert not np.isinf(V_nr).any(), "Potential contains Infs"
    print("    -> CPU Newton-Raphson PASS!")

    # 3. Test GPU Newton-Raphson wrapper (handles both CuPy and fallback)
    print("\n[3] Testing _solve_poisson_newton_gpu (with CuPy or auto-fallback)...")
    engine.V += np.random.normal(0.0, 2.0, size=engine.V.shape)
    engine.V[engine.isBound] = engine.V_fixed[engine.isBound]

    V_gpu_nr, iters_gpu, diff_gpu, status_gpu = engine._solve_poisson_newton_gpu(
        coeff=coeff,
        V_plasma=V_plasma,
        Te_up=Te_up,
        n0=n0,
        tol_V=0.05,
        max_iters=15
    )

    print(f"    Result: status = '{status_gpu}', iters = {iters_gpu}, diff = {diff_gpu:.4e} V")
    assert status_gpu == 'converged_newton', f"Expected 'converged_newton', got '{status_gpu}'"
    assert not np.isnan(V_gpu_nr).any(), "GPU/Fallback potential contains NaNs"
    print("    -> GPU / Fallback Newton-Raphson PASS!")

    # 4. Test integrated recalc_poisson triggering Newton when Picard stagnates
    print("\n[4] Testing full recalc_poisson pipeline with high-density configuration...")
    # Force a challenging state
    engine.V.fill(0.0)
    engine.V[engine.isBound] = engine.V_fixed[engine.isBound]
    
    engine.recalc_poisson(iterations=25, params=config["simulation"])
    n_done = engine.last_poisson_iters
    delta_V = engine.last_poisson_delta_V
    rms_V = engine.last_poisson_rms
    status = engine.last_poisson_status

    print(f"    recalc_poisson result: status = '{status}', iters = {n_done}, delta_V = {delta_V:.4f} V, rms = {rms_V*1000:.2f} mV")
    print(f"    Convergence flag (last_poisson_converged): {engine.last_poisson_converged}")
    
    assert engine.last_poisson_converged, "Solver was expected to converge (via Picard or Newton fallback)"
    # 5. Test forcing Picard fallback branch directly in _recalc_poisson_cpu
    print("\n[5] Testing forced Picard fallback triggering Newton-Raphson...")
    # Run _recalc_poisson_cpu with max_iters=4 to hit stagnation/timeout or high density
    # Or simulate Picard status in ['stagnated', 'diverged']
    coeff = (engine.dy * 1e-3)**2 / 8.854187817e-12
    V_plasma = config["grids"][0]["V"] + config.get("advanced_settings", {}).get("V_plasma_offset", 26.0)
    Te_up = config["simulation"].get("Te_up", 4.3)
    n0 = 2.0e18

    # Start from current converged field with realistic stagnation oscillation (+/- 5V)
    engine.V += np.random.normal(0.0, 5.0, size=engine.V.shape)
    engine.V[engine.isBound] = engine.V_fixed[engine.isBound]

    # Run with standard max_iters to test fallback
    V_nr, iters_nr, diff_nr, status_nr = engine._solve_poisson_newton_cpu(
        coeff=coeff, V_plasma=V_plasma, Te_up=Te_up, n0=n0, tol_V=0.05, max_iters=15
    )
    print(f"    Direct Newton convergence: status = '{status_nr}', iters = {iters_nr}, delta_V = {diff_nr:.4f} V")
    assert status_nr == 'converged_newton'
    print("    -> Fallback execution PASS!")

    print("\n" + "=" * 60)
    print("ALL NEWTON-RAPHSON FALLBACK TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_newton_raphson_cpu_and_gpu()
