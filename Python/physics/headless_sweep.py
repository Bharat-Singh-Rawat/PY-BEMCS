"""
Headless parametric sweep worker and execution utilities.

Enables running simulation scenarios concurrently across all CPU cores
without GUI dependencies, bypassing Python's GIL.
"""

import os
import copy
import time
import pickle
import numpy as np

from physics.simulator import DigitalTwinSimulator
from diagnostics.metrics import compute_rolling_variation


def write_scenario_state(state_file: str, data: dict):
    """Atomically write scenario display snapshot to disk for real-time GUI tracking."""
    if not state_file:
        return
    tmp_path = state_file + f".{os.getpid()}.tmp"
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, state_file)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def read_scenario_state(state_file: str):
    """Read scenario display snapshot from disk."""
    if not state_file or not os.path.exists(state_file):
        return None
    try:
        with open(state_file, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


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


def extract_display_particles(sim, scatter_limit=1000):
    """
    Extract particle coordinates and energy arrays for GUI visualization.
    Coordinates (px, py) are in millimeters matching the simulator domain [0, Lx] and [0, Ly].
    """
    num_p = getattr(sim, 'num_p', 0)
    num_e = getattr(sim, 'num_e', 0)
    m_ion = getattr(sim, 'm_ion', 131.293 * 1.6605e-27)
    q     = getattr(sim, 'q_ion', getattr(sim, 'q', 1.602e-19))

    prim_xy = np.empty((0, 2), dtype=np.float32)
    e_prim = np.empty(0, dtype=np.float32)
    cex_xy = np.empty((0, 2), dtype=np.float32)
    e_cex = np.empty(0, dtype=np.float32)
    elec_xy = np.empty((0, 2), dtype=np.float32)

    if scatter_limit is None:
        # None - hide all particles
        return prim_xy, e_prim, cex_xy, e_cex, elec_xy

    # Extract ions
    if num_p > 0 and hasattr(sim, 'p_x') and hasattr(sim, 'p_y'):
        p_isCEX_raw = sim.p_isCEX[:num_p].copy() if hasattr(sim, 'p_isCEX') else np.zeros(num_p, dtype=bool)
        p_x_raw = sim.p_x[:num_p]
        p_y_raw = sim.p_y[:num_p]
        p_vx_raw = sim.p_vx[:num_p] if hasattr(sim, 'p_vx') else np.zeros(num_p)
        p_vy_raw = sim.p_vy[:num_p] if hasattr(sim, 'p_vy') else np.zeros(num_p)
        p_vz_raw = sim.p_vz[:num_p] if hasattr(sim, 'p_vz') else np.zeros(num_p)

        if scatter_limit == 0 or num_p <= scatter_limit:
            p_isCEX = p_isCEX_raw
            px, py = p_x_raw, p_y_raw
            pvx, pvy, pvz = p_vx_raw, p_vy_raw, p_vz_raw
        else:
            ion_idx = np.random.choice(num_p, scatter_limit, replace=False)
            ion_idx.sort()
            p_isCEX = p_isCEX_raw[ion_idx]
            px, py = p_x_raw[ion_idx], p_y_raw[ion_idx]
            pvx = p_vx_raw[ion_idx]
            pvy = p_vy_raw[ion_idx]
            pvz = p_vz_raw[ion_idx]

        prim_mask = ~p_isCEX
        cex_mask = p_isCEX

        if np.any(prim_mask):
            prim_xy = np.column_stack((px[prim_mask], py[prim_mask])).astype(np.float32)
            vsq = pvx[prim_mask]**2 + pvy[prim_mask]**2 + pvz[prim_mask]**2
            e_prim = (0.5 * m_ion * vsq / max(q, 1e-25)).astype(np.float32)

        if np.any(cex_mask):
            cex_xy = np.column_stack((px[cex_mask], py[cex_mask])).astype(np.float32)
            vsq = pvx[cex_mask]**2 + pvy[cex_mask]**2 + pvz[cex_mask]**2
            e_cex = (0.5 * m_ion * vsq / max(q, 1e-25)).astype(np.float32)

    # Extract electrons
    if num_e > 0 and hasattr(sim, 'e_x') and hasattr(sim, 'e_y'):
        e_x_raw = sim.e_x[:num_e]
        e_y_raw = sim.e_y[:num_e]
        if scatter_limit == 0 or num_e <= scatter_limit:
            ex, ey = e_x_raw, e_y_raw
        else:
            e_idx = np.random.choice(num_e, scatter_limit, replace=False)
            e_idx.sort()
            ex, ey = e_x_raw[e_idx], e_y_raw[e_idx]
        elec_xy = np.column_stack((ex, ey)).astype(np.float32)

    return prim_xy, e_prim, cex_xy, e_cex, elec_xy


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
    cancel_file = payload.get("cancel_file")
    pause_file = payload.get("pause_file")
    active_view_file = payload.get("active_view_file")
    status_dict = payload.get("status_dict")
    state_file = payload.get("state_file")
    scatter_limit = payload.get("scatter_limit", 1000)

    t_start = time.perf_counter()

    # Apply parameter variation
    scenario_params = apply_sweep_param_to_dict(base_params, param_id, parameter_value)

    sim = DigitalTwinSimulator()
    sim.build_domain(scenario_params)

    window = int(steady_config.get("window", 50))
    thresh = float(steady_config.get("threshold", 1.0))
    max_iters = int(steady_config.get("max_iterations", 5000))
    min_iters = int(steady_config.get("min_iterations", 500))

    scenario_iters = 0
    steady_reached = False
    max_hit = False

    prim_xy = np.empty((0, 2), dtype=np.float32)
    e_prim = np.empty(0, dtype=np.float32)
    cex_xy = np.empty((0, 2), dtype=np.float32)
    e_cex = np.empty(0, dtype=np.float32)
    elec_xy = np.empty((0, 2), dtype=np.float32)

    iter_hist = []
    div_hist = []
    div_mid_hist = []
    ebs_hist = []
    last_valid_div = 0.0
    last_valid_div_mid = 0.0

    init_state = {
        "status": "running",
        "iter": 0,
        "max_iters": max_iters,
        "parameter_value": parameter_value,
        "steady_state_reached": False,
        "computation_time_s": 0.0,
        "t_sim_us": 0.0,
        "div": 0.0,
        "div_mid": 0.0,
        "exit_ion_current_A": 0.0,
        "transparency": 0.0,
        "prim_xy": prim_xy,
        "e_prim_eV": e_prim,
        "cex_xy": cex_xy,
        "e_cex_eV": e_cex,
        "elec_xy": elec_xy,
        "iter_history": [],
        "div_history": [],
        "div_mid_history": [],
        "ebs_history": [],
        "Lx": float(getattr(sim, 'Lx', 3.0)),
        "Ly": float(getattr(sim, 'Ly', 3.0)),
        "grids": scenario_params.get("grids", []),
    }
    if status_dict is not None:
        status_dict[scenario_idx] = init_state
    if state_file:
        write_scenario_state(state_file, init_state)

    while True:
        if cancel_file and (scenario_iters % 5 == 0) and os.path.exists(cancel_file):
            break

        if pause_file and os.path.exists(pause_file):
            t_pause_start = time.perf_counter()
            try:
                prim_xy, e_prim, cex_xy, e_cex, elec_xy = extract_display_particles(sim, scatter_limit)
            except Exception:
                pass
            paused_state = dict(run_state) if 'run_state' in locals() else dict(init_state)
            paused_state["status"] = "paused"
            paused_state["iter"] = scenario_iters
            paused_state["prim_xy"] = prim_xy
            paused_state["e_prim_eV"] = e_prim
            paused_state["cex_xy"] = cex_xy
            paused_state["e_cex_eV"] = e_cex
            paused_state["elec_xy"] = elec_xy
            if status_dict is not None:
                status_dict[scenario_idx] = paused_state
            if state_file:
                write_scenario_state(state_file, paused_state)

            while os.path.exists(pause_file):
                if cancel_file and os.path.exists(cancel_file):
                    break
                time.sleep(0.05)
            t_start += (time.perf_counter() - t_pause_start)
            resumed_state = dict(paused_state)
            resumed_state["status"] = "running"
            if status_dict is not None:
                status_dict[scenario_idx] = resumed_state
            if state_file:
                write_scenario_state(state_file, resumed_state)

        sim.step(scenario_params)
        scenario_iters += 1

        div_val = getattr(sim, 'current_div', np.nan)
        div_c = float(div_val) if (div_val is not None and np.isfinite(div_val)) else np.nan
        div_mid_val = getattr(sim, 'current_div_mid', np.nan)
        div_m = float(div_mid_val) if (div_mid_val is not None and np.isfinite(div_mid_val)) else np.nan
        min_pot_val = getattr(sim, 'min_pot', getattr(sim, 'saddle_point_potential', np.nan))
        min_pot_c = float(min_pot_val) if (min_pot_val is not None and np.isfinite(min_pot_val)) else 0.0

        if np.isfinite(div_c):
            last_valid_div = div_c
        if np.isfinite(div_m):
            last_valid_div_mid = div_m

        iter_hist.append(scenario_iters)
        div_hist.append(div_c)
        div_mid_hist.append(div_m)
        ebs_hist.append(min_pot_c)

        if (status_dict is not None or state_file is not None) and (scenario_iters % 10 == 0 or scenario_iters <= 5):
            # Check if this scenario is currently being viewed in the GUI
            is_active_view = True
            if active_view_file and os.path.exists(active_view_file):
                try:
                    with open(active_view_file, "r") as f:
                        is_active_view = (f.read().strip() == str(scenario_idx))
                except Exception:
                    is_active_view = False

            if is_active_view:
                try:
                    prim_xy, e_prim, cex_xy, e_cex, elec_xy = extract_display_particles(sim, scatter_limit)
                except Exception:
                    pass

            t_us = (scenario_iters * sim.dt * 1e6) if hasattr(sim, 'dt') else 0.0
            i_step_c = float(getattr(sim, 'exit_ion_current_avg', getattr(sim, 'exit_ion_current_step', 0.0)))
            trans_c = float(sim.get_transparency()) if hasattr(sim, 'get_transparency') else 0.0
            cur_comp_time = time.perf_counter() - t_start

            run_state = {
                "status": "running",
                "iter": scenario_iters,
                "max_iters": max_iters,
                "parameter_value": parameter_value,
                "steady_state_reached": False,
                "computation_time_s": float(cur_comp_time),
                "t_sim_us": float(t_us),
                "div": last_valid_div,
                "div_mid": last_valid_div_mid,
                "exit_ion_current_A": i_step_c,
                "transparency": trans_c,
                "prim_xy": prim_xy,
                "e_prim_eV": e_prim,
                "cex_xy": cex_xy,
                "e_cex_eV": e_cex,
                "elec_xy": elec_xy,
                "iter_history": list(iter_hist),
                "div_history": list(div_hist),
                "div_mid_history": list(div_mid_hist),
                "ebs_history": list(ebs_hist),
                "Lx": float(getattr(sim, 'Lx', 3.0)),
                "Ly": float(getattr(sim, 'Ly', 3.0)),
                "nx": int(getattr(sim, 'nx', 0)),
                "ny": int(getattr(sim, 'ny', 0)),
                "dx": float(getattr(sim, 'dx', 0.05)),
                "dy": float(getattr(sim, 'dy', 0.05)),
                "grids": scenario_params.get("grids", []),
                "current_ppc_map": getattr(sim, 'current_ppc_map', None).copy() if getattr(sim, 'current_ppc_map', None) is not None else None,
                "accum_ppc_map": getattr(sim, 'accum_ppc_map', None).copy() if getattr(sim, 'accum_ppc_map', None) is not None else None,
                "ppc_steps_count": int(getattr(sim, 'ppc_steps_count', 0)),
            }
            if status_dict is not None:
                status_dict[scenario_idx] = run_state
            if state_file:
                write_scenario_state(state_file, run_state)

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

    # Extract final particles and metrics
    try:
        prim_xy, e_prim, cex_xy, e_cex, elec_xy = extract_display_particles(sim, scatter_limit)
    except Exception:
        pass

    comp_time_s = time.perf_counter() - t_start
    t_sim_us = (sim.iteration * sim.dt * 1e6) if hasattr(sim, 'dt') else 0.0
    e_tot = sim.energy_history_tot[-1] if hasattr(sim, 'energy_history_tot') and sim.energy_history_tot else 0.0
    q_net = sim.charge_history_q_net[-1] if hasattr(sim, 'charge_history_q_net') and sim.charge_history_q_net else 0.0
    fe = sim.energy_history_fe[-1] if hasattr(sim, 'energy_history_fe') and sim.energy_history_fe else 0.0
    saddle_v = sim.min_pot if hasattr(sim, 'min_pot') else 0.0

    div_val = getattr(sim, 'current_div', 0.0)
    div = float(div_val) if np.isfinite(div_val) else last_valid_div

    div_plane2_val = getattr(sim, 'current_div_mid', 0.0)
    div_plane2 = float(div_plane2_val) if np.isfinite(div_plane2_val) else last_valid_div_mid
    i_exit = float(getattr(sim, 'exit_ion_current_avg', getattr(sim, 'exit_ion_current_step', 0.0)))
    trans = float(sim.get_transparency()) if hasattr(sim, 'get_transparency') else 0.0

    p_eff, p_max, perv_frac = compute_perveance_metrics_static(
        grids=scenario_params.get("grids", []),
        i_exit=i_exit,
        q_ion=getattr(sim, 'q_ion', 1.602e-19),
        m_ion=getattr(sim, 'm_ion', 131.293 * 1.6605e-27),
        injection_area=getattr(sim, '_last_injection_area', None)
    )

    res_row = {
        "scenario_idx": scenario_idx,
        "parameter_value": parameter_value,
        "steady_state_reached": bool(steady_reached),
        "max_iters_hit": max_iters if max_hit else None,
        "steady_iterations": int(scenario_iters),
        "computation_time_s": float(comp_time_s),
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

    done_state = {
        "status": "done",
        "iter": scenario_iters,
        "max_iters": max_iters,
        "parameter_value": parameter_value,
        "steady_state_reached": bool(steady_reached),
        "computation_time_s": float(comp_time_s),
        "t_sim_us": float(t_sim_us),
        "div": div,
        "div_mid": div_plane2,
        "exit_ion_current_A": i_exit,
        "perveance_effective": float(p_eff),
        "perveance_max": float(p_max),
        "perveance_fraction": float(perv_frac),
        "transparency": float(trans),
        "prim_xy": prim_xy,
        "e_prim_eV": e_prim,
        "cex_xy": cex_xy,
        "e_cex_eV": e_cex,
        "elec_xy": elec_xy,
        "iter_history": list(iter_hist),
        "div_history": list(div_hist),
        "div_mid_history": list(div_mid_hist),
        "ebs_history": list(ebs_hist),
        "Lx": float(getattr(sim, 'Lx', 3.0)),
        "Ly": float(getattr(sim, 'Ly', 3.0)),
        "nx": int(getattr(sim, 'nx', 0)),
        "ny": int(getattr(sim, 'ny', 0)),
        "dx": float(getattr(sim, 'dx', 0.05)),
        "dy": float(getattr(sim, 'dy', 0.05)),
        "grids": scenario_params.get("grids", []),
        "current_ppc_map": getattr(sim, 'current_ppc_map', None).copy() if getattr(sim, 'current_ppc_map', None) is not None else None,
        "accum_ppc_map": getattr(sim, 'accum_ppc_map', None).copy() if getattr(sim, 'accum_ppc_map', None) is not None else None,
        "ppc_steps_count": int(getattr(sim, 'ppc_steps_count', 0)),
        "result_row": res_row,
    }

    if status_dict is not None:
        status_dict[scenario_idx] = done_state
    if state_file:
        write_scenario_state(state_file, done_state)

    return res_row
