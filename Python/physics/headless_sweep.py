"""
Headless parametric sweep worker and execution utilities.

Enables running simulation scenarios concurrently across all CPU cores
without GUI dependencies, bypassing Python's GIL.
"""

import os
import copy
import numpy as np

from physics.simulator import DigitalTwinSimulator
from diagnostics.metrics import compute_rolling_variation


def apply_sweep_param_to_dict(base_params: dict, param_id: str, value: float) -> dict:
    """
    Apply a single swept parameter value to a copy of the simulation parameters dict.
    Handles grid voltages, geometry, plasma densities, and temperatures.
    """
    params = copy.deepcopy(base_params)

    # 1. Direct scalar parameters
    if param_id in ("Te_up", "n0_plasma", "Ti", "n0", "Tn", "Te", "upstream_gap_mm", "neut_x", "neut_r", "target_ppc"):
        params[param_id] = float(value)

    # 2. Grid-specific parameters
    elif param_id.startswith("grid_"):
        parts = param_id.split("_")  # e.g. ["grid", "1", "V"] or ["grid", "2", "gap"]
        if len(parts) >= 3:
            grid_idx = int(parts[1]) - 1
            prop_key = parts[2]
            grids = params.get("grids", [])
            if 0 <= grid_idx < len(grids):
                grids[grid_idx][prop_key] = float(value)
                params["grids"] = grids

    return params


def compute_perveance_metrics_static(grids: list, i_exit: float, q_ion: float = 1.602e-19, m_ion: float = 131.293 * 1.6605e-27, injection_area: float = None) -> tuple:
    """
    Calculates effective perveance, maximum Child-Langmuir perveance (Goebel-Katz),
    and perveance fraction.
    """
    if not np.isfinite(i_exit) or i_exit < 0.0:
        i_exit = 0.0

    v_screen = 0.0
    v_accel = 0.0
    gap_mm = 1.0
    t_screen_mm = 0.5
    r_screen_mm = 1.0

    if grids and len(grids) > 0:
        v_screen = float(grids[0].get('V', 0.0))
        gap_mm = float(grids[0].get('gap', 1.0))
        t_screen_mm = float(grids[0].get('t', 0.5))
        r_screen_mm = float(grids[0].get('r', 1.0))
        if len(grids) > 1:
            v_accel = float(grids[1].get('V', 0.0))

    v_total = max(v_screen - v_accel, 0.0)

    # Peffective = I_exit / (V_total^1.5)
    if v_total > 0.0 and i_exit > 0.0:
        p_eff = float(i_exit / (v_total ** 1.5))
    else:
        p_eff = 0.0

    eps_0 = 8.8541878128e-12
    if q_ion <= 0.0 or m_ion <= 0.0:
        q_ion = 1.602e-19
        m_ion = 131.293 * 1.6605e-27

    l_g = max(gap_mm * 1e-3, 1e-6)
    t_s = max(t_screen_mm * 1e-3, 0.0)
    r_s = max(r_screen_mm * 1e-3, 1e-6)
    l_e_sq = (l_g + t_s) ** 2 + (r_s ** 2)

    if injection_area is not None and injection_area > 0.0:
        area = float(injection_area)
    else:
        area = max(r_screen_mm * 1e-3 * 1e-3, 1e-8)

    p_max = float((4.0 * eps_0 / 9.0) * np.sqrt(2.0 * q_ion / m_ion) * (area / l_e_sq))

    if p_max > 0.0 and np.isfinite(p_eff):
        perv_frac = float(p_eff / p_max)
    else:
        perv_frac = 0.0

    return p_eff, p_max, perv_frac


def run_single_sweep_scenario_worker(payload: dict) -> dict:
    """
    Worker function executed in an isolated OS process.
    Instantiates DigitalTwinSimulator, steps to steady state, and returns full metrics row.
    """
    scenario_idx = payload["scenario_idx"]
    param_id = payload["param_id"]
    parameter_value = payload["parameter_value"]
    base_params = payload["base_params"]
    steady_config = payload["steady_config"]

    # Apply parameter variation
    scenario_params = apply_sweep_param_to_dict(base_params, param_id, parameter_value)

    sim = DigitalTwinSimulator()
    sim.build_domain(scenario_params)

    window = int(steady_config.get("window", 50))
    thresh = float(steady_config.get("threshold", 1.0))
    max_iters = int(steady_config.get("max_iterations", 5000))
    min_iters = int(steady_config.get("min_iterations", 50))

    scenario_iters = 0
    steady_reached = False
    max_hit = False

    while True:
        sim.step(scenario_params)
        scenario_iters += 1

        if scenario_iters >= max_iters:
            max_hit = True
            break

        if scenario_iters >= max(min_iters, window):
            # Evaluate stationarity every 5 iterations
            if scenario_iters % 5 == 0:
                histories_to_check = []
                if hasattr(sim, 'energy_history_tot') and len(sim.energy_history_tot) >= window:
                    histories_to_check.append(sim.energy_history_tot[-scenario_iters:])
                if hasattr(sim, 'charge_history_q_net') and len(sim.charge_history_q_net) >= window:
                    histories_to_check.append(sim.charge_history_q_net[-scenario_iters:])
                if hasattr(sim, 'potential_history') and len(sim.potential_history) >= window:
                    histories_to_check.append(sim.potential_history[-scenario_iters:])

                if histories_to_check:
                    all_under = True
                    for h in histories_to_check:
                        _, var = compute_rolling_variation(h, window=window, mode="rel_change")
                        if len(var) == 0 or abs(var[-1]) > thresh:
                            all_under = False
                            break
                    if all_under:
                        steady_reached = True
                        break

    # Extract final metrics
    t_sim_us = (sim.iteration * sim.dt * 1e6) if hasattr(sim, 'dt') else 0.0
    e_tot = sim.energy_history_tot[-1] if hasattr(sim, 'energy_history_tot') and sim.energy_history_tot else 0.0
    q_net = sim.charge_history_q_net[-1] if hasattr(sim, 'charge_history_q_net') and sim.charge_history_q_net else 0.0
    fe = sim.energy_history_fe[-1] if hasattr(sim, 'energy_history_fe') and sim.energy_history_fe else 0.0
    saddle_v = sim.min_pot if hasattr(sim, 'min_pot') else 0.0

    div_val = getattr(sim, 'current_div', 0.0)
    div = float(div_val) if np.isfinite(div_val) else 0.0

    div_plane2_val = getattr(sim, 'current_div_mid', 0.0)
    div_plane2 = float(div_plane2_val) if np.isfinite(div_plane2_val) else 0.0
    i_exit = float(getattr(sim, 'exit_ion_current_avg', getattr(sim, 'exit_ion_current_step', 0.0)))
    trans = float(sim.get_transparency()) if hasattr(sim, 'get_transparency') else 0.0

    p_eff, p_max, perv_frac = compute_perveance_metrics_static(
        grids=scenario_params.get("grids", []),
        i_exit=i_exit,
        q_ion=getattr(sim, 'q_ion', 1.602e-19),
        m_ion=getattr(sim, 'm_ion', 131.293 * 1.6605e-27),
        injection_area=getattr(sim, '_last_injection_area', None)
    )

    return {
        "scenario_idx": scenario_idx,
        "parameter_value": parameter_value,
        "steady_state_reached": bool(steady_reached),
        "max_iters_hit": max_iters if max_hit else None,
        "steady_iterations": int(scenario_iters),
        "t_sim_us": float(t_sim_us),
        "saddle_potential_V": float(saddle_v),
        "divergence_deg": float(div),
        "divergence_2nd_plane_deg": float(div_plane2),
        "divergence_plane2_deg": float(div_plane2),
        "exit_ion_current_A": float(i_exit),
        "perveance_effective": float(p_eff),
        "Peffective": float(p_eff),
        "perveance_max": float(p_max),
        "Pmax": float(p_max),
        "perveance_fraction": float(perv_frac),
        "P_fraction": float(perv_frac),
        "transparency": float(trans),
        "total_energy_J": float(e_tot),
        "total_charge_C": float(q_net),
        "field_energy_J": float(fe),
    }
