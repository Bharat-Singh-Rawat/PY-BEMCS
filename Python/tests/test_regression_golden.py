"""
Golden-file regression test for the PY-BEMCS physics engine.

Purpose
-------
Guarantees that code re-organisation (modularisation, refactoring) does NOT
change the physics.  A set of short, seeded scenarios is run on the
deterministic NumPy CPU path (PYBEMCS_FORCE_CPU=1) and every relevant state
array / counter / step() return value is compared **bit-for-bit** against a
stored baseline.

Usage
-----
    # 1. Create the baseline (only once, on known-good code):
    python tests/test_regression_golden.py --generate

    # 2. Verify after any change:
    python tests/test_regression_golden.py
    #   or: python -m pytest tests/test_regression_golden.py
"""
import os
import sys

# Must be set BEFORE the physics engine is imported.
os.environ["PYBEMCS_FORCE_CPU"] = "1"

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physics_engine import DigitalTwinSimulator  # noqa: E402  (compat shim)

GOLDEN_DIR = os.path.join(HERE, "golden")
N_STEPS = 40

_BASE_GRIDS = [
    {"V": 1100.0, "t": 0.38, "gap": 0.58, "r": 0.95, "cham": 0.0},
    {"V": -180.0, "t": 0.51, "gap": 3.0, "r": 0.57, "cham": 0.0},
]

_BASE_PARAMS = {
    "neut_rate": 0, "neut_match_ion": False, "Te": 5.0,
    "neut_x": 10.0, "neut_r": 1.5,
    "V_plasma_offset": 26.0, "m_e_ratio": 1000.0,
    "n0_plasma": 7e17, "Te_up": 4.3, "Ti": 0.75, "Tn": 534.0, "n0": 1e18,
    "Accel": 0.5, "Thresh": 10000.0, "sim_mode": "Both",
    "geometry": "half_hole", "target_ppc": 250,
    "rf_enable": False, "rf_grid_idx": 0, "rf_freq": 13.56, "rf_amp": 500.0,
}

# Scenarios chosen to exercise every phase of step():
# injection, Poisson, push, hits/erosion/SEE, electrons/neutralizer,
# thermal, mid-hole probe, CEX, RF co-extraction, periodic BCs.
SCENARIOS = {
    "nstar_half_hole": dict(),
    "two_holes_rf_neut": dict(geometry="two_holes", rf_enable=True,
                              neut_rate=20, neut_match_ion=False),
    "one_hole_erosion_match": dict(geometry="one_hole", sim_mode="Erosion",
                                   Accel=1e4, Thresh=50.0,
                                   neut_match_ion=True, neut_rate=10),
}

_STATE_ARRAYS = [
    "V", "rho", "Ex", "Ey", "Tmap", "damage_map", "eroded_depth",
    "isBound", "V_fixed", "accum_ppc_map", "current_ppc_map",
    "x_coords", "y_coords",
]
_SCALARS = [
    "num_p", "num_e", "iteration", "injected_ions", "transmitted_ions",
    "entered_optics", "exit_ion_current_avg", "exit_vx_mean", "exit_v_mean",
    "exit_energy_mean_eV", "exit_count_step", "total_active_cells",
    "low_ppc_cells", "last_poisson_iters", "last_poisson_delta_V",
    "last_poisson_rms", "nx", "ny",
]


def _make_params(overrides):
    p = dict(_BASE_PARAMS)
    p.update(overrides)
    p["grids"] = [dict(g) for g in _BASE_GRIDS]
    return p


def run_scenario(name):
    """Run one seeded scenario and return a flat dict of numpy arrays."""
    np.random.seed(12345)
    sim = DigitalTwinSimulator()
    sim.set_material(name="Titanium")
    params = _make_params(SCENARIOS[name])
    sim.build_domain(params)

    out = {}
    step_rets = []
    for _ in range(N_STEPS):
        ret = sim.step(params)
        remeshed, min_pot, cdiv, T_grids, trans = ret
        step_rets.append([float(bool(remeshed)), float(min_pot), float(cdiv),
                          float(trans)] + [float(t) for t in T_grids])
    width = max(len(r) for r in step_rets)
    out["step_returns"] = np.array(
        [r + [np.nan] * (width - len(r)) for r in step_rets], dtype=np.float64)

    for a in _STATE_ARRAYS:
        if hasattr(sim, a):
            out["arr_" + a] = np.asarray(getattr(sim, a))
    n, ne = int(sim.num_p), int(sim.num_e)
    for a in ("p_x", "p_y", "p_vx", "p_vy", "p_vz", "p_isCEX"):
        out["p_" + a] = np.asarray(getattr(sim, a)[:n])
    for a in ("e_x", "e_y", "e_vx", "e_vy", "e_vz"):
        out["e_" + a] = np.asarray(getattr(sim, a)[:ne])
    out["scalars"] = np.array(
        [float(getattr(sim, s, np.nan)) for s in _SCALARS], dtype=np.float64)
    out["energy_tot"] = np.asarray(sim.energy_history_tot, dtype=np.float64)
    out["charge_net"] = np.asarray(sim.charge_history_q_net, dtype=np.float64)
    out["rng_after"] = np.random.rand(4)   # catches changed RNG call order
    return out


def _golden_path(name):
    return os.path.join(GOLDEN_DIR, f"{name}.npz")


def generate():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name in SCENARIOS:
        data = run_scenario(name)
        np.savez_compressed(_golden_path(name), **data)
        print(f"[golden] wrote {name}: num_p={int(data['scalars'][0])}, "
              f"num_e={int(data['scalars'][1])}")


def compare(name):
    ref = np.load(_golden_path(name))
    new = run_scenario(name)
    problems = []
    if set(ref.files) != set(new.keys()):
        problems.append(f"key mismatch: {set(ref.files) ^ set(new.keys())}")
    for k in ref.files:
        if k not in new:
            continue
        a, b = ref[k], new[k]
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=(a.dtype.kind == "f")):
            diff = ""
            if a.shape == b.shape and a.dtype.kind in "fi":
                diff = f" max|diff|={np.nanmax(np.abs(a.astype(float) - b.astype(float)))}"
            problems.append(f"{k}: shape {a.shape} vs {b.shape}{diff}")
    return problems


def test_golden_regression():
    failures = {}
    for name in SCENARIOS:
        probs = compare(name)
        if probs:
            failures[name] = probs
    assert not failures, f"Physics regression detected: {failures}"


if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate()
    else:
        ok = True
        for name in SCENARIOS:
            probs = compare(name)
            if probs:
                ok = False
                print(f"[FAIL] {name}")
                for p in probs:
                    print("    ", p)
            else:
                print(f"[ OK ] {name}")
        sys.exit(0 if ok else 1)
