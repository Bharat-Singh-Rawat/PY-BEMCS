"""
Parametric Sweep Mixin for DigitalTwinApp.
Coordinates automated execution across multiple physical scenarios,
steady-state condition checks, scenario transitions, and results collection.
"""

import os
import sys
import time
import copy
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import QTimer

from ..dialogs import (
    SweepSettingsDialog,
    SweepResultsDialog,
    format_sci_value,
    parse_grid_sweep_param,
    format_comp_duration,
)
from ..workers import ParallelSweepWorker
from diagnostics.metrics import compute_rolling_variation
from physics.active_clock import ActiveDurationTracker, get_active_clock


class SweepMixin:
    """Mixin managing the Parametric Sweep simulation workflow."""

    def _get_sweep_total_timer(self):
        if not hasattr(self, '_sweep_total_timer') or self._sweep_total_timer is None:
            self._sweep_total_timer = ActiveDurationTracker(max_gap=None)
        return self._sweep_total_timer

    def _get_sweep_scenario_timer(self):
        if not hasattr(self, '_sweep_scenario_timer') or self._sweep_scenario_timer is None:
            self._sweep_scenario_timer = ActiveDurationTracker(max_gap=None)
        return self._sweep_scenario_timer

    def _is_in_sweep_mode(self) -> bool:
        """Return True if sweep mode is armed, active, or has an active/completed sweep configuration."""
        if getattr(self, 'is_sweep_active', False) or getattr(self, 'sweep_mode_armed', False):
            return True
        if getattr(self, 'sweep_config', None) is not None:
            return True
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None and self.parallel_sweep_worker.isRunning():
            return True
        return False

    def _on_track_ptcls_toggled(self, checked: bool):
        """Synchronize Record Kinematics checkbox state with parallel sweep worker processes."""
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            if hasattr(self.parallel_sweep_worker, 'set_track_particles'):
                self.parallel_sweep_worker.set_track_particles(bool(checked))

    def _on_sim_mode_changed(self, mode_text: str):
        """Synchronize left-panel Simulation Mode selection across sweep base params, cache, and workers."""
        mode = str(mode_text).strip() if mode_text else (self.combo_mode.currentText() if hasattr(self, 'combo_mode') else "Both")
        if hasattr(self, '_sweep_base_params') and isinstance(self._sweep_base_params, dict):
            self._sweep_base_params["sim_mode"] = mode
        if hasattr(self, '_scenario_domain_cache') and isinstance(self._scenario_domain_cache, dict):
            for d in self._scenario_domain_cache.values():
                if isinstance(d, dict) and isinstance(d.get("params"), dict):
                    d["params"]["sim_mode"] = mode
        if getattr(self, 'sim_isRunning', False) and hasattr(self, '_sim_worker') and self._sim_worker is not None:
            if hasattr(self, 'get_params'):
                self._sim_worker.update_params(self.get_params())
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            if hasattr(self.parallel_sweep_worker, 'set_sim_mode'):
                self.parallel_sweep_worker.set_sim_mode(mode)

    def _get_merged_sweep_status_dict(self) -> dict:
        """Return merged scenario status dictionary from cached state and parallel worker."""
        status_dict = {}
        if hasattr(self, '_sweep_status_dict') and self._sweep_status_dict:
            status_dict.update(self._sweep_status_dict)
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            worker_dict = self.parallel_sweep_worker.get_status_dict()
            for k, v in worker_dict.items():
                if v and (v.get("iter", 0) > 0 or v.get("status") in ("running", "paused", "done")):
                    status_dict[k] = v
                elif k not in status_dict:
                    status_dict[k] = v

        # Supplement with completed results from self.sweep_results if any status entry is missing/empty
        for res_row in getattr(self, 'sweep_results', []):
            if not isinstance(res_row, dict):
                continue
            s_idx = res_row.get("scenario_idx")
            if s_idx is None:
                continue
            if s_idx not in status_dict or not status_dict[s_idx]:
                status_dict[s_idx] = {}
            st = status_dict[s_idx]
            if not st.get("status") or st["status"] in ("queued", "unknown"):
                st["status"] = "done"
            if not st.get("iter"):
                st["iter"] = res_row.get("steady_iterations", res_row.get("iter", 0))
            if st.get("div") is None or st.get("div") == 0.0:
                st["div"] = res_row.get("divergence_deg", res_row.get("div", 0.0))
            if st.get("div_mid") is None or st.get("div_mid") == 0.0:
                st["div_mid"] = res_row.get("divergence_2nd_plane_deg", res_row.get("div_mid", 0.0))
            if not st.get("exit_ion_current_A"):
                st["exit_ion_current_A"] = res_row.get("exit_ion_current_A", 0.0)
            if not st.get("transparency"):
                st["transparency"] = res_row.get("transparency", 0.0)
            if not st.get("perveance_effective"):
                st["perveance_effective"] = res_row.get("perveance_effective", res_row.get("Peffective", 0.0))
            if not st.get("t_sim_us"):
                st["t_sim_us"] = res_row.get("t_sim_us", 0.0)
            if not st.get("computation_time_s"):
                st["computation_time_s"] = res_row.get("computation_time_s", 0.0)
            if not st.get("T_grids_K") and "T_grids_K" in res_row:
                st["T_grids_K"] = list(res_row["T_grids_K"])
                st["T_grids"] = list(res_row["T_grids_K"])
            if not st.get("T_grids_C") and "T_grids_C" in res_row:
                st["T_grids_C"] = list(res_row["T_grids_C"])
            if "steady_state_reached" in res_row and "steady_state_reached" not in st:
                st["steady_state_reached"] = res_row["steady_state_reached"]
            if "saddle_potential_V" not in st and "saddle_potential_V" in res_row:
                st["saddle_potential_V"] = res_row["saddle_potential_V"]
            for hk in ("iter_history", "time_history", "div_history", "div_mid_history", "ebs_history", "transparency_history", "ion_current_exit_history", "ion_current_exit_avg_history", "transmitted_ions_history", "active_cells_history", "low_ppc_cells_history", "T_histories"):
                if not st.get(hk) and res_row.get(hk):
                    st[hk] = list(res_row[hk]) if isinstance(res_row[hk], list) else dict(res_row[hk])
        return status_dict

    def _record_seq_scenario_step(
        self,
        snap,
        t_sim,
        min_pot,
        current_div,
        current_div_mid,
        transparency,
        trans_last_frame,
        T_grids,
        ptcl_data=None,
    ):
        """Record a live sequential/armed simulation step into per-scenario history & tracking stores."""
        scen_idx = getattr(self, 'sweep_current_idx', 0) + 1
        if not hasattr(self, '_seq_scenario_histories') or not isinstance(self._seq_scenario_histories, dict):
            self._seq_scenario_histories = {}
        if not hasattr(self, '_sweep_tracking_buffers') or not isinstance(self._sweep_tracking_buffers, dict):
            self._sweep_tracking_buffers = {}
        if not hasattr(self, '_sweep_status_dict') or not isinstance(self._sweep_status_dict, dict):
            self._sweep_status_dict = {}

        h = self._seq_scenario_histories.get(scen_idx)
        if h is not None and h.get("iter_history") and snap['iteration'] <= h["iter_history"][-1]:
            h = None

        n_grids = max(len(T_grids), len(getattr(self, 'grid_widgets', [])))
        if h is None:
            h = {
                "iter_history": [],
                "time_history": [],
                "ebs_history": [],
                "div_history": [],
                "div_mid_history": [],
                "transparency_history": [],
                "transparency3_history": [],
                "ion_current_exit_history": [],
                "ion_current_exit_avg_history": [],
                "transmitted_ions_history": [],
                "active_cells_history": [],
                "low_ppc_cells_history": [],
                "T_histories": {i: [] for i in range(n_grids)},
            }
            self._seq_scenario_histories[scen_idx] = h

        h["iter_history"].append(snap['iteration'])
        h["time_history"].append(t_sim)
        h["ebs_history"].append(min_pot)
        h["div_history"].append(current_div)
        h["div_mid_history"].append(current_div_mid)
        h["transparency_history"].append(transparency)
        h["transparency3_history"].append(trans_last_frame)
        h["ion_current_exit_history"].append(snap['exit_ion_current_step'])
        h["ion_current_exit_avg_history"].append(snap['exit_ion_current_avg'])
        h["transmitted_ions_history"].append(int(round(snap['transmitted_ions_step'])))
        h["active_cells_history"].append(snap['total_active_cells'])
        h["low_ppc_cells_history"].append(snap['low_ppc_cells'])
        for i, T in enumerate(T_grids):
            h["T_histories"].setdefault(i, []).append(T)

        if ptcl_data is not None and ptcl_data.size > 0:
            self._sweep_tracking_buffers.setdefault(scen_idx, []).append(ptcl_data)

        cfg = getattr(self, 'sweep_config', {}) or {}
        values = cfg.get("values", [])
        param_val = values[scen_idx - 1] if 0 <= (scen_idx - 1) < len(values) else 0.0
        div_val = float(current_div) if (current_div is not None and np.isfinite(current_div)) else 0.0
        div_mid_val = float(current_div_mid) if (current_div_mid is not None and np.isfinite(current_div_mid)) else 0.0

        existing = self._sweep_status_dict.get(scen_idx, {})
        st_name = "running" if getattr(self, 'sim_isRunning', False) else ("paused" if getattr(self, '_sweep_paused_current', False) else existing.get("status", "running"))
        sim = getattr(self, 'sim', None)
        t_grids_list = [float(t) for t in T_grids]
        tmap_snap = snap.get('Tmap')
        if tmap_snap is None and sim is not None and getattr(sim, 'Tmap', None) is not None:
            tmap_snap = sim.Tmap.copy()
        elif tmap_snap is not None:
            tmap_snap = tmap_snap.copy()
        mask_grids_snap = snap.get('mask_grids') if snap.get('mask_grids') is not None else getattr(sim, 'mask_grids', [])
        is_bound_snap = snap.get('isBound') if snap.get('isBound') is not None else getattr(sim, 'isBound', None)
        self._sweep_status_dict[scen_idx] = {
            **existing,
            "status": st_name,
            "iter": int(snap['iteration']),
            "parameter_value": param_val,
            "t_sim_us": float(t_sim * 1e6),
            "div": div_val,
            "div_mid": div_mid_val,
            "exit_ion_current_A": float(snap['exit_ion_current_step']),
            "transparency": float(transparency),
            "prim_xy": snap.get('prim_xy', np.empty((0, 2))),
            "e_prim_eV": snap.get('e_prim_eV', np.empty(0)),
            "cex_xy": snap.get('cex_xy', np.empty((0, 2))),
            "e_cex_eV": snap.get('e_cex_eV', np.empty(0)),
            "elec_xy": snap.get('elec_xy', np.empty((0, 2))),
            "e_elec_eV": snap.get('e_elec_eV', np.empty(0)),
            "iter_history": list(h["iter_history"]),
            "time_history": list(h["time_history"]),
            "ebs_history": list(h["ebs_history"]),
            "div_history": list(h["div_history"]),
            "div_mid_history": list(h["div_mid_history"]),
            "transparency_history": list(h["transparency_history"]),
            "transparency3_history": list(h["transparency3_history"]),
            "ion_current_exit_history": list(h["ion_current_exit_history"]),
            "ion_current_exit_avg_history": list(h["ion_current_exit_avg_history"]),
            "transmitted_ions_history": list(h["transmitted_ions_history"]),
            "active_cells_history": list(h["active_cells_history"]),
            "low_ppc_cells_history": list(h["low_ppc_cells_history"]),
            "T_histories": {k: list(v) for k, v in h["T_histories"].items()},
            "T_grids": t_grids_list,
            "T_grids_K": t_grids_list,
            "T_grids_C": [float(t - 273.15) for t in t_grids_list],
            "Tmap": tmap_snap,
            "mask_grids": [m.copy() for m in mask_grids_snap] if mask_grids_snap else [],
            "isBound": is_bound_snap.copy() if is_bound_snap is not None else None,
            "X": snap.get('X').copy() if snap.get('X') is not None else (getattr(sim, 'X', None).copy() if getattr(sim, 'X', None) is not None else None),
            "Y": snap.get('Y').copy() if snap.get('Y') is not None else (getattr(sim, 'Y', None).copy() if getattr(sim, 'Y', None) is not None else None),
            "V": snap.get('V').copy() if snap.get('V') is not None else (getattr(sim, 'V', None).copy() if getattr(sim, 'V', None) is not None else None),
            "damage_map": getattr(sim, 'damage_map', None).copy() if getattr(sim, 'damage_map', None) is not None else None,
            "eroded_depth": getattr(sim, 'eroded_depth', None).copy() if getattr(sim, 'eroded_depth', None) is not None else None,
            "energy_history_tot": list(getattr(sim, 'energy_history_tot', [])),
            "charge_history_q_net": list(getattr(sim, 'charge_history_q_net', [])),
            "energy_history_fe": list(getattr(sim, 'energy_history_fe', [])),
            "ion_current_history": list(getattr(sim, 'ion_current_history', [])),
            "Lx": float(snap.get('Lx', getattr(sim, 'Lx', 3.0))),
            "Ly": float(snap.get('Ly', getattr(sim, 'Ly', 3.0))),
            "nx": int(snap.get('nx', getattr(sim, 'nx', 0))),
            "ny": int(snap.get('ny', getattr(sim, 'ny', 0))),
            "dx": float(snap.get('dx', getattr(sim, 'dx', 0.05))),
            "dy": float(snap.get('dy', getattr(sim, 'dy', 0.05))),
        }

    def _get_sweep_scenario_tracking_buffer(self, scenario_idx: int) -> list:
        """Retrieve the particle kinematics tracking buffer for a specific sweep scenario."""
        if not hasattr(self, '_sweep_tracking_buffers') or not isinstance(self._sweep_tracking_buffers, dict):
            self._sweep_tracking_buffers = {}

        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            if hasattr(self.parallel_sweep_worker, 'get_scenario_tracking_buffer'):
                w_buf = self.parallel_sweep_worker.get_scenario_tracking_buffer(scenario_idx)
                if w_buf:
                    self._sweep_tracking_buffers[scenario_idx] = list(w_buf)
                    return list(w_buf)

        if scenario_idx in self._sweep_tracking_buffers and len(self._sweep_tracking_buffers[scenario_idx]) > 0:
            return list(self._sweep_tracking_buffers[scenario_idx])

        status_dict = self._get_merged_sweep_status_dict()
        info = status_dict.get(scenario_idx, {})
        if isinstance(info.get("tracking_buffer"), list) and len(info["tracking_buffer"]) > 0:
            self._sweep_tracking_buffers[scenario_idx] = list(info["tracking_buffer"])
            return list(info["tracking_buffer"])

        is_parallel = (
            hasattr(self, 'parallel_sweep_worker')
            and self.parallel_sweep_worker is not None
        )
        if (
            not is_parallel
            and scenario_idx == getattr(self, 'sweep_current_idx', 0) + 1
            and info.get("status") not in ("queued", "action_required")
            and scenario_idx not in self._sweep_tracking_buffers
            and len(getattr(self, 'tracking_buffer', [])) > 0
        ):
            return list(self.tracking_buffer)

        return []

    def _sync_sweep_scenario_data_to_app(self, scenario_idx: int, info: dict = None):
        """Synchronize app history lists and particle tracking buffer with the specified sweep scenario."""
        if info is None:
            status_dict = self._get_merged_sweep_status_dict()
            info = dict(status_dict.get(scenario_idx, {}))
        else:
            info = dict(info)

        seq_h = getattr(self, '_seq_scenario_histories', {}).get(scenario_idx)
        if isinstance(seq_h, dict) and len(seq_h.get("iter_history", [])) >= len(info.get("iter_history") or []):
            for k, v in seq_h.items():
                if k == "T_histories":
                    info[k] = {int(gk): list(gv) for gk, gv in v.items()}
                else:
                    info[k] = list(v)

        iters = info.get("iter_history")
        divs = info.get("div_history")
        divs_mid = info.get("div_mid_history")
        ebs = info.get("ebs_history")

        if not iters and info.get("iter", 0) > 0 and (info.get("div") is not None or info.get("div_mid") is not None):
            iters = [info.get("iter", 0)]
            divs = [info.get("div", 0.0)]
            divs_mid = [info.get("div_mid", 0.0)]
            ebs = [info.get("saddle_potential_V", info.get("min_pot", 0.0))]

        n_grids = len(info.get("grids", []) or getattr(self, 'grid_widgets', []))

        if iters and len(iters) > 0:
            n = len(iters)
            self.iter_history = list(iters)
            self.div_history = list(divs) if divs is not None else [info.get("div", 0.0)] * n
            self.div_mid_history = list(divs_mid) if divs_mid is not None else [info.get("div_mid", 0.0)] * n
            self.ebs_history = list(ebs) if ebs is not None else [info.get("saddle_potential_V", 0.0)] * n
            t_hist = info.get("time_history")
            if t_hist is not None and len(t_hist) == n:
                self.time_history = list(t_hist)
            else:
                dt = getattr(self.sim, 'dt', 5e-10) if hasattr(self, 'sim') and hasattr(self.sim, 'dt') and self.sim.dt > 0 else 5e-10
                self.time_history = [it * dt for it in iters]

            trans_hist = info.get("transparency_history")
            if trans_hist is not None and len(trans_hist) > 0:
                self.transparency_history = list(trans_hist)
            elif "transparency" in info:
                self.transparency_history = [float(info.get("transparency", 0.0))] * n
            else:
                self.transparency_history = []

            self.transparency3_history = list(info.get("transparency3_history") or [])

            i_exit_hist = info.get("ion_current_exit_history")
            if i_exit_hist is not None and len(i_exit_hist) > 0:
                self.ion_current_exit_history = list(i_exit_hist)
            elif "exit_ion_current_A" in info:
                self.ion_current_exit_history = [float(info.get("exit_ion_current_A", 0.0))] * n
            else:
                self.ion_current_exit_history = []

            i_avg_hist = info.get("ion_current_exit_avg_history")
            if i_avg_hist is not None and len(i_avg_hist) > 0:
                self.ion_current_exit_avg_history = list(i_avg_hist)
            else:
                self.ion_current_exit_avg_history = list(self.ion_current_exit_history)

            self.transmitted_ions_history = list(info.get("transmitted_ions_history") or [])
            self.active_cells_history = list(info.get("active_cells_history") or [])
            self.low_ppc_cells_history = list(info.get("low_ppc_cells_history") or [])

            t_hists = info.get("T_histories")
            if isinstance(t_hists, dict) and len(t_hists) > 0:
                self.T_histories = {int(k): list(v) for k, v in t_hists.items()}
            else:
                self.T_histories = {i: [] for i in range(n_grids)}
        else:
            is_parallel = (
                hasattr(self, 'parallel_sweep_worker')
                and self.parallel_sweep_worker is not None
            )
            preserve_live_active = (
                not info
                and not is_parallel
                and scenario_idx == getattr(self, 'sweep_current_idx', 0) + 1
                and scenario_idx not in getattr(self, '_seq_scenario_histories', {})
                and len(getattr(self, 'iter_history', [])) > 0
            )
            if not preserve_live_active:
                self.iter_history = []
                self.time_history = []
                self.ebs_history = []
                self.div_history = []
                self.div_mid_history = []
                self.transparency_history = []
                self.transparency3_history = []
                self.ion_current_exit_history = []
                self.ion_current_exit_avg_history = []
                self.transmitted_ions_history = []
                self.active_cells_history = []
                self.low_ppc_cells_history = []
                self.T_histories = {i: [] for i in range(n_grids)}

        self.tracking_buffer = self._get_sweep_scenario_tracking_buffer(scenario_idx)

    def _sync_selected_sweep_scenario_for_export(self):
        """Ensure history series and tracking_buffer match the selected sweep scenario right before export."""
        if not self._is_in_sweep_mode():
            return
        if (
            hasattr(self, 'combo_sweep_scenario')
            and self.combo_sweep_scenario is not None
            and self.combo_sweep_scenario.count() > 0
            and self.combo_sweep_scenario.currentIndex() >= 0
        ):
            scenario_idx = self.combo_sweep_scenario.currentIndex() + 1
        else:
            scenario_idx = getattr(self, 'sweep_current_idx', 0) + 1
        self._sync_sweep_scenario_data_to_app(scenario_idx)

    def _set_grid_edit_buttons_enabled(self, enabled: bool):
        """Enable or disable `+ Add Grid` / `- Remove Grid` buttons during active sweeps."""
        for attr in ("btn_add_grid", "btn_remove_grid"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(bool(enabled))

    def _hide_sweep_top_bar_controls(self):
        for attr in (
            "lbl_sweep_badge",
            "sweep_control_group",
            "combo_sweep_scenario",
            "btn_stop_sweep",
            "btn_resume_sweep",
            "btn_exit_sweep",
            "lbl_build_domain_info",
            "lbl_beam_info",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(False)

    def _validate_sweep_grid_consistency(self, config=None, require_built_domain=False):
        """Verify UI grids, built simulator grids, and sweep target agree."""
        n_widgets = len(getattr(self, 'grid_widgets', []))
        if getattr(self, '_grid_count_modified_since_build', False):
            return (
                "Grid configuration changed after adding or removing grids.\n\n"
                "Please click '1. BUILD DOMAIN' first so the computational domain matches the current grids."
            )

        cfg = config if config is not None else getattr(self, 'sweep_config', None)
        sim = getattr(self, 'sim', None)
        sim_grids = getattr(sim, 'grids', None)
        domain_checked = require_built_domain or getattr(self, '_sweep_domain_built', False)

        if cfg is not None:
            param_id = cfg.get("param_id", "")
            grid_idx, _ = parse_grid_sweep_param(param_id)
            if grid_idx is not None:
                if n_widgets > 0 and grid_idx > n_widgets:
                    return (
                        f"Sweep target '{param_id}' refers to Grid {grid_idx}, "
                        f"which does not exist in the current {n_widgets}-grid configuration."
                    )
                if domain_checked and sim_grids is not None and len(sim_grids) > 0 and grid_idx > len(sim_grids):
                    return (
                        f"Sweep target '{param_id}' refers to Grid {grid_idx}, "
                        f"but the built computational domain only has {len(sim_grids)} grid(s).\n\n"
                        "Please click '1. BUILD DOMAIN' and reconfigure the sweep."
                    )

        if n_widgets > 0 and domain_checked and sim_grids is not None and len(sim_grids) > 0:
            if len(sim_grids) != n_widgets:
                return (
                    f"Grid count discrepancy detected: UI defines {n_widgets} grid(s), "
                    f"while the built computational domain has {len(sim_grids)} grid(s).\n\n"
                    "Please click '1. BUILD DOMAIN' before running the sweep."
                )

        if require_built_domain and n_widgets > 0:
            base_params = getattr(self, '_sweep_base_params', None)
            if isinstance(base_params, dict) and isinstance(base_params.get("grids"), list):
                if len(base_params["grids"]) != n_widgets:
                    return (
                        f"Grid count discrepancy detected: sweep baseline has {len(base_params['grids'])} grid(s), "
                        f"while UI defines {n_widgets} grid(s).\n\n"
                        "Please click '1. BUILD DOMAIN' first."
                    )
        return None

    def _warn_sweep_grid_discrepancy(self, msg: str, title: str = "Sweep / Domain Discrepancy"):
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText("Status: Grid/sweep discrepancy — rebuild domain or update sweep settings.")
        QMessageBox.warning(self, title, msg)

    def on_grid_count_changed(self):
        """Handle grid additions/removals outside config loading."""
        n_grids = len(getattr(self, 'grid_widgets', []))
        cfg = getattr(self, 'sweep_config', None)
        if cfg is None:
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(
                    f"Grid count changed ({n_grids} grid(s)). Click '1. BUILD DOMAIN' before opening Sweep settings."
                )
            return

        param_id = cfg.get("param_id", "")
        grid_idx, _ = parse_grid_sweep_param(param_id)
        if grid_idx is not None and grid_idx > n_grids:
            self.sweep_mode_armed = False
            self.is_sweep_active = False
            self._sweep_domain_built = False
            self._sweep_paused_current = False
            self._sweep_stopped = False
            self.sweep_config = None
            self.sweep_results = []
            self._sweep_status_dict = {}
            self._hide_sweep_top_bar_controls()
            self._set_grid_edit_buttons_enabled(True)
            msg = (
                f"Grid {grid_idx} was removed, so the configured sweep on '{param_id}' is no longer valid "
                f"for the current {n_grids}-grid setup.\n\n"
                "The invalid sweep has been disarmed. Please click '1. BUILD DOMAIN' and then open "
                "'Settings -> Sweep...' to configure a valid sweep."
            )
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(
                    f"Grid {grid_idx} removed — invalid sweep '{param_id}' disarmed. Click '1. BUILD DOMAIN' first."
                )
            QMessageBox.warning(self, "Sweep Target Invalidated", msg)
            return

        self.is_sweep_active = False
        self.sweep_mode_armed = True
        self._sweep_domain_built = False
        self._sweep_stopped = False
        self._set_grid_edit_buttons_enabled(True)
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText("⚡ Sweep invalidated — click 1. BUILD DOMAIN")
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #fef3c7; color: #b45309; font-weight: bold; "
                "border: 1px solid #fde68a; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(
                f"Grid count changed ({n_grids} grid(s)). Click '1. BUILD DOMAIN' to rebuild before starting sweep."
            )
        self._update_sweep_action_button_state()

    def open_sweep_dialog(self):
        """Open modal configuration dialog to save sweep settings."""
        if getattr(self, 'is_sweep_active', False):
            QMessageBox.information(
                self,
                "Sweep in Progress",
                "A parametric sweep is already active. Stop or wait for it to complete."
            )
            return

        if getattr(self, '_grid_count_modified_since_build', False):
            msg = (
                "Grid configuration changed after adding or removing grids.\n\n"
                "Please click '1. BUILD DOMAIN' first before setting sweep parameters."
            )
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText("Status: Attention: build domain first")
            QMessageBox.warning(self, "Build Domain First", msg)
            return

        dialog = SweepSettingsDialog(self)
        if dialog.exec_() == SweepSettingsDialog.Accepted:
            cfg = dialog.get_config()
            self.save_sweep_settings(cfg)

    def save_sweep_settings(self, config):
        """Save sweep configuration and arm sweep mode without immediately launching."""
        err = self._validate_sweep_grid_consistency(config, require_built_domain=False)
        if err:
            self._warn_sweep_grid_discrepancy(err, title="Sweep Configuration Discrepancy")
            return False

        self.sweep_config = config
        self.sweep_results = []
        self._sweep_status_dict = {}
        self._sweep_tracking_buffers = {}
        self._seq_scenario_histories = {}
        self.sweep_current_idx = 0
        self.sweep_mode_armed = True
        self.is_sweep_active = False
        self._sweep_domain_built = False
        self._sweep_paused_current = False
        self._sweep_base_params = copy.deepcopy(self.get_params()) if hasattr(self, 'get_params') else {}
        self._scenario_domain_cache = {}
        self._current_rendered_scenario_grid_idx = None

        total_scenarios = len(config.get("values", []))
        param_label = config.get("param_label", "Parameter")
        param_id = config.get("param_id", "")
        param_unit = config.get("param_unit", "")

        first_val = config["values"][0] if total_scenarios > 0 else 0.0
        first_val_str = (
            format_sci_value(first_val)
            if (param_id in ("n0_plasma", "n0") or abs(first_val) >= 1e4 or (0 < abs(first_val) <= 1e-3))
            else f"{first_val:.3g}"
        )

        # Apply initial scenario value to GUI inputs
        self._apply_sweep_value(param_id, first_val)

        # Initialize and display top-bar scenario selector dropdown
        if hasattr(self, 'combo_sweep_scenario'):
            self.combo_sweep_scenario.blockSignals(True)
            self.combo_sweep_scenario.clear()
            for idx, val in enumerate(config.get("values", [])):
                val_str = (
                    format_sci_value(val)
                    if (param_id in ("n0_plasma", "n0") or abs(val) >= 1e4 or (0 < abs(val) <= 1e-3))
                    else f"{val:.3g}"
                )
                unit_str = f" {param_unit}" if param_unit else ""
                self.combo_sweep_scenario.addItem(f"Scenario {idx + 1}: {param_id} = {val_str}{unit_str}  [⏳ Queued]")
            self.combo_sweep_scenario.setCurrentIndex(0)
            self.combo_sweep_scenario.blockSignals(False)
            if hasattr(self, 'sweep_control_group'):
                self.sweep_control_group.setVisible(True)
            self.combo_sweep_scenario.setVisible(True)

        # Update Top Bar Badge & Buttons
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep mode armed [1/{total_scenarios}]"
            )
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #f3e8ff; color: #7e22ce; font-weight: bold; "
                "border: 1px solid #c084fc; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(False)
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("2. START BEAM")
        if hasattr(self, 'lbl_build_domain_info'):
            self.lbl_build_domain_info.setVisible(True)
        if hasattr(self, 'lbl_beam_info'):
            self.lbl_beam_info.setVisible(True)

        self._update_sweep_action_button_state()

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(
                f"Sweep settings saved for {param_label} ({total_scenarios} scenarios). "
                f"Click 'Start Sweep' to begin, or '2. START BEAM' to simulate the current scenario."
            )

    def _update_sweep_action_button_state(self):
        """Update text, tooltip, and style of top-bar sweep action button (Start / Pause / Resume)."""
        btn = getattr(self, 'btn_resume_sweep', None)
        if btn is None:
            return

        if getattr(self, 'sweep_config', None) is None:
            btn.setVisible(False)
            return

        # If sweep is armed but not started:
        if getattr(self, 'sweep_mode_armed', False) and not getattr(self, 'is_sweep_active', False):
            btn.setText("Start Sweep")
            btn.setToolTip("Start parametric sweep simulation across all scenarios")
            btn.setStyleSheet(
                "background-color: #f0fdf4; color: #15803d; font-weight: bold; "
                "border: 1px solid #86efac; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
            btn.setVisible(True)
            return

        # If sweep is running:
        is_parallel_running = (
            hasattr(self, 'parallel_sweep_worker')
            and self.parallel_sweep_worker is not None
            and self.parallel_sweep_worker.isRunning()
            and not getattr(self, '_sweep_stopped', False)
        )
        is_sequential_running = (
            getattr(self, 'is_sweep_active', False)
            and getattr(self, 'sim_isRunning', False)
            and not getattr(self, '_sweep_stopped', False)
            and not getattr(self, '_sweep_paused_current', False)
        )

        if is_parallel_running or is_sequential_running:
            btn.setText("Pause Sweep")
            btn.setToolTip("Pause all scenarios in the active parametric sweep")
            btn.setStyleSheet(
                "background-color: #fef3c7; color: #b45309; font-weight: bold; "
                "border: 1px solid #fde68a; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
            btn.setVisible(True)
            return

        # If sweep is paused / stopped:
        btn.setText("Resume Sweep")
        btn.setToolTip("Resume the parametric sweep simulation")
        btn.setStyleSheet(
            "background-color: #f0fdf4; color: #15803d; font-weight: bold; "
            "border: 1px solid #86efac; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
        )
        btn.setVisible(True)

    def is_domain_ready_for_sweep(self):
        """Check if domain has been built after configuring the sweep and has valid field data."""
        if not getattr(self, '_sweep_domain_built', False):
            return False
        if not hasattr(self, 'sim') or not hasattr(self.sim, 'Ex') or self.sim.Ex is None or not np.any(self.sim.Ex):
            return False
        return True

    def on_sweep_action_clicked(self):
        """Handler for top-bar sweep action button (Start Sweep / Pause Sweep / Resume Sweep)."""
        if getattr(self, 'sweep_config', None) is None:
            return

        # Case 1: Sweep is armed (not yet started)
        if getattr(self, 'sweep_mode_armed', False) and not getattr(self, 'is_sweep_active', False):
            err = self._validate_sweep_grid_consistency(self.sweep_config, require_built_domain=False)
            if err:
                self._warn_sweep_grid_discrepancy(err)
                return
            if not self.is_domain_ready_for_sweep():
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText("Status: Attention: build domain first")
                QMessageBox.warning(self, "Attention", "Attention: build domain first")
                return
            err = self._validate_sweep_grid_consistency(self.sweep_config, require_built_domain=True)
            if err:
                self._warn_sweep_grid_discrepancy(err)
                return
            self.start_sweep(self.sweep_config)
            self._update_sweep_action_button_state()
            return

        # Case 2: Sweep is actively running -> PAUSE it
        is_parallel_running = (
            hasattr(self, 'parallel_sweep_worker')
            and self.parallel_sweep_worker is not None
            and self.parallel_sweep_worker.isRunning()
            and not getattr(self, '_sweep_stopped', False)
        )
        is_sequential_running = (
            getattr(self, 'is_sweep_active', False)
            and getattr(self, 'sim_isRunning', False)
            and not getattr(self, '_sweep_stopped', False)
            and not getattr(self, '_sweep_paused_current', False)
        )

        if is_parallel_running:
            self._bg_live = None
            self.parallel_sweep_worker.pause_all_scenarios()
            self._sweep_stopped = True
            self._pause_simulation_timer()
            if hasattr(self, 'lbl_sweep_badge'):
                self.lbl_sweep_badge.setText("⚡ Sweep mode paused")
                self.lbl_sweep_badge.setStyleSheet(
                    "background-color: #fef3c7; color: #b45309; font-weight: bold; "
                    "border: 1px solid #fde68a; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
                )
            if hasattr(self, 'btn_toggle'):
                self.btn_toggle.setText("RESUME BEAM")
            self._update_sweep_action_button_state()
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText("Status: Parallel sweep paused.")
            return

        if is_sequential_running:
            self._bg_live = None
            self.sim_isRunning = False
            self._sweep_paused_current = True
            self._sweep_stopped = True
            if hasattr(self, '_sim_worker'):
                self._sim_worker.pause()
            self._pause_simulation_timer()
            if hasattr(self, 'lbl_sweep_badge'):
                self.lbl_sweep_badge.setText("⚡ Sweep mode paused")
                self.lbl_sweep_badge.setStyleSheet(
                    "background-color: #fef3c7; color: #b45309; font-weight: bold; "
                    "border: 1px solid #fde68a; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
                )
            if hasattr(self, 'btn_toggle'):
                self.btn_toggle.setText("RESUME BEAM")
            self.update_sim_status_badge("PAUSED")
            self._update_sweep_action_button_state()
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText("Status: Sweep paused.")
            return

        # Case 3: Sweep was stopped or paused -> RESUME it
        if hasattr(self, 'resume_sweep_with_selected'):
            self.resume_sweep_with_selected()
            self._update_sweep_action_button_state()

    def resume_sweep_with_selected(self):
        """Resume simulation / sweep with parameters from the selected scenario in combo_sweep_scenario."""
        if not hasattr(self, 'combo_sweep_scenario') or self.sweep_config is None:
            return

        is_parallel_config = bool(self.sweep_config.get("parallel", False))
        err = self._validate_sweep_grid_consistency(
            self.sweep_config, require_built_domain=(not is_parallel_config)
        )
        if err:
            self._warn_sweep_grid_discrepancy(err)
            return

        cur_idx = self.combo_sweep_scenario.currentIndex()
        values = self.sweep_config.get("values", [])
        if cur_idx < 0 or cur_idx >= len(values):
            return

        param_id = self.sweep_config.get("param_id", "")
        param_unit = self.sweep_config.get("param_unit", "")
        selected_val = values[cur_idx]
        total_scenarios = len(values)
        scen_num = cur_idx + 1

        # Apply the selected scenario parameter value
        self._apply_sweep_value(param_id, selected_val)
        self.sweep_current_idx = cur_idx
        self._bg_live = None

        # Case A: Parallel sweep worker running/paused
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None and self.parallel_sweep_worker.isRunning():
            self.is_sweep_active = True
            self._sweep_stopped = False
            self._sweep_paused_current = False
            self._set_grid_edit_buttons_enabled(False)
            self.parallel_sweep_worker.set_active_view_scenario(scen_num)
            self.parallel_sweep_worker.resume_all_scenarios()

            # Clear cached snapshots so live running worker status flows directly to GUI
            if hasattr(self, '_sweep_status_dict') and self._sweep_status_dict:
                self._sweep_status_dict.clear()

            if hasattr(self, '_resume_simulation_timer'):
                self._resume_simulation_timer()

            if not hasattr(self, 'sweep_monitor_timer') or self.sweep_monitor_timer is None:
                self.sweep_monitor_timer = QTimer(self)
                self.sweep_monitor_timer.timeout.connect(self._on_sweep_monitor_tick)
            if not self.sweep_monitor_timer.isActive():
                self.sweep_monitor_timer.start(150)

            val_str = (
                format_sci_value(selected_val)
                if (param_id in ("n0_plasma", "n0") or abs(selected_val) >= 1e4 or (0 < abs(selected_val) <= 1e-3))
                else f"{selected_val:.3g}"
            )
            if hasattr(self, 'lbl_sweep_badge'):
                self.lbl_sweep_badge.setVisible(True)
                self.lbl_sweep_badge.setText(
                    f"⚡ Parallel Sweep [{scen_num}/{total_scenarios}]"
                )
                self.lbl_sweep_badge.setStyleSheet(
                    "background-color: #f3e8ff; color: #7e22ce; font-weight: bold; "
                    "border: 1px solid #d8b4fe; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
                )
            if hasattr(self, 'btn_stop_sweep'):
                self.btn_stop_sweep.setVisible(False)
            self._update_sweep_action_button_state()
            if hasattr(self, 'btn_exit_sweep'):
                self.btn_exit_sweep.setVisible(True)
            if hasattr(self, 'btn_toggle'):
                self.btn_toggle.setText("PAUSE BEAM")
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(
                    f"Resumed parallel sweep for Scenario {scen_num} ({param_id} = {val_str} {param_unit})."
                )

            self._update_sweep_scenario_combo_items()
            self._render_sweep_scenario_snapshot(scen_num)
            return

        # Case B: Parallel sweep configured, but worker is NOT running (e.g. stopped/errored)
        if is_parallel_config:
            completed_indices = {
                r["scenario_idx"]
                for r in getattr(self, 'sweep_results', [])
                if isinstance(r, dict) and "scenario_idx" in r
            }
            remaining_indices = [i for i in range(1, total_scenarios + 1) if i not in completed_indices]
            if not remaining_indices:
                QMessageBox.information(
                    self,
                    "Sweep Completed",
                    "All scenarios in this sweep have already completed successfully."
                )
                self.finish_sweep()
                return

            import os
            n_cores = self.sweep_config.get("n_cores")
            if not n_cores and hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
                n_cores = getattr(self.parallel_sweep_worker, 'n_cores', None)
            if not n_cores:
                n_cores = max(1, (os.cpu_count() or 16) // 2)

            self._start_parallel_sweep(self.sweep_config, n_cores, remaining_indices=remaining_indices)
            return

        # Case C: Sequential sweep mode
        domain_unbuilt = not hasattr(self.sim, 'Ex') or self.sim.Ex is None or not np.any(self.sim.Ex)
        needs_rebuild = domain_unbuilt

        if needs_rebuild:
            try:
                self.build_domain(force=True)
            except Exception as e:
                self.sweep_mode_armed = True
                self.is_sweep_active = False
                self._sweep_stopped = True
                self._sweep_paused_current = False
                if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
                    worker = self.parallel_sweep_worker
                    self.parallel_sweep_worker = None
                    try:
                        worker.cancel()
                        worker.wait(2000)
                    except Exception:
                        pass
                if hasattr(self, 'btn_toggle'):
                    self.btn_toggle.setText("2. START BEAM")
                if hasattr(self, 'lbl_sweep_badge'):
                    self.lbl_sweep_badge.setVisible(True)
                    self.lbl_sweep_badge.setText(
                        f"⚡ Sweep armed [Scenario {scen_num} | Build Domain required]"
                    )
                if hasattr(self, 'btn_stop_sweep'):
                    self.btn_stop_sweep.setVisible(False)
                if hasattr(self, 'btn_resume_sweep'):
                    self.btn_resume_sweep.setVisible(True)
                if hasattr(self, 'btn_exit_sweep'):
                    self.btn_exit_sweep.setVisible(True)
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText(
                        f"Parameters for Scenario {scen_num} applied ({param_id} = {selected_val}). "
                        f"Please click '1. BUILD DOMAIN' and then '2. START BEAM' to resume."
                    )
                return

        self.sweep_mode_armed = False
        self.is_sweep_active = True
        self._sweep_stopped = False
        self._sweep_paused_current = False
        self._set_grid_edit_buttons_enabled(False)
        self._get_sweep_scenario_timer().reset(start_running=True)
        self.sweep_scenario_start_wall_time = get_active_clock()
        self.sweep_scenario_start_iter = getattr(self.sim, 'iteration', 0)
        if hasattr(self, '_sweep_status_dict') and self._sweep_status_dict:
            self._sweep_status_dict.clear()

        val_str = (
            format_sci_value(selected_val)
            if (param_id in ("n0_plasma", "n0") or abs(selected_val) >= 1e4 or (0 < abs(selected_val) <= 1e-3))
            else f"{selected_val:.3g}"
        )
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep mode active [{cur_idx + 1}/{total_scenarios}]"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(False)
        self._update_sweep_action_button_state()
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)

        if not getattr(self, 'sim_isRunning', False):
            self.sim_isRunning = True
            if hasattr(self, '_resume_simulation_timer'):
                self._resume_simulation_timer()
            if hasattr(self, '_sim_worker'):
                self._sim_worker.start_sim(self.get_params(), run_id=getattr(self, '_sim_run_id', 0))
            if hasattr(self, 'update_sim_status_badge'):
                self.update_sim_status_badge("RUNNING")
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("PAUSE BEAM")

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(
                f"Resumed sweep simulation for Scenario {cur_idx + 1} ({param_id} = {val_str} {param_unit})."
            )
        self._update_sweep_scenario_combo_items()

    def start_sweep(self, config):
        """Initialize and launch the multi-scenario sweep."""
        err = self._validate_sweep_grid_consistency(config, require_built_domain=False)
        if err:
            self._warn_sweep_grid_discrepancy(err)
            return

        if not self.is_domain_ready_for_sweep():
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText("Status: Attention: build domain first")
            QMessageBox.warning(self, "Attention", "Attention: build domain first")
            return

        err = self._validate_sweep_grid_consistency(config, require_built_domain=True)
        if err:
            self._warn_sweep_grid_discrepancy(err)
            return

        if hasattr(self, '_sim_worker') and self._sim_worker.isRunning():
            self._sim_worker.stop(wait_ms=1000)
        self.sim_isRunning = False
        self._sweep_paused_current = False
        self._sweep_stopped = False

        self.sweep_config = config
        self.sweep_results = []
        self._sweep_status_dict = {}
        self._sweep_tracking_buffers = {}
        self._seq_scenario_histories = {}
        self.sweep_current_idx = 0
        self.is_sweep_active = True
        self.sweep_mode_armed = False
        self._sweep_paused_current = False
        self._set_grid_edit_buttons_enabled(False)
        self._sweep_base_params = copy.deepcopy(self.get_params()) if hasattr(self, 'get_params') else {}
        self._scenario_domain_cache = {}
        self._current_rendered_scenario_grid_idx = None
        self._get_sweep_total_timer().reset(start_running=True)
        self.sweep_start_wall_time = get_active_clock()
        self._get_sweep_scenario_timer().reset(start_running=True)
        self.sweep_scenario_start_wall_time = get_active_clock()
        if hasattr(self, '_resume_simulation_timer'):
            self._resume_simulation_timer()

        # Scan scenarios for downstream plume memory advisories
        from physics.hardware import check_scenario_plume_advisory
        from physics.headless_sweep import apply_sweep_param_to_dict

        self._scenarios_pending_advisory = {}
        self._scenario_advisory_decisions = {}

        base_params = copy.deepcopy(self.get_params()) if hasattr(self, 'get_params') else {}
        for idx, val in enumerate(config.get("values", [])):
            scen_num = idx + 1
            scen_params = apply_sweep_param_to_dict(base_params, config.get("param_id", ""), val, sweep_config=config)
            adv = check_scenario_plume_advisory(scen_params)
            if adv.get("has_advisory", False):
                adv["scenario_idx"] = scen_num
                adv["parameter_value"] = val
                adv["param_id"] = config.get("param_id", "")
                self._scenarios_pending_advisory[scen_num] = adv

        total_scenarios = len(config.get("values", []))
        param_label = config.get("param_label", "Parameter")
        use_parallel = config.get("parallel", False) and total_scenarios > 0

        # Initialize and display top-bar scenario selector dropdown
        if hasattr(self, 'combo_sweep_scenario'):
            self.combo_sweep_scenario.blockSignals(True)
            self.combo_sweep_scenario.clear()
            for idx, val in enumerate(config.get("values", [])):
                scen_num = idx + 1
                val_str = (
                    format_sci_value(val)
                    if (config.get("param_id") in ("n0_plasma", "n0") or abs(val) >= 1e4 or (0 < abs(val) <= 1e-3))
                    else f"{val:.3g}"
                )
                unit_str = f" {config.get('param_unit', '')}" if config.get('param_unit') else ""
                tag = "[⚠️ Action Required - Plume Advisory]" if scen_num in self._scenarios_pending_advisory else "[⏳ Queued]"
                self.combo_sweep_scenario.addItem(f"Scenario {scen_num}: {config.get('param_id', 'param')} = {val_str}{unit_str}  {tag}")
            self.combo_sweep_scenario.setCurrentIndex(0)
            self.combo_sweep_scenario.blockSignals(False)
            if hasattr(self, 'sweep_control_group'):
                self.sweep_control_group.setVisible(True)
            self.combo_sweep_scenario.setVisible(True)

        if hasattr(self, 'btn_resume_sweep'):
            self.btn_resume_sweep.setVisible(True)

        if use_parallel:
            n_cores = int(config.get("n_cores", 8))
            self._start_parallel_sweep(config, n_cores)
            return

        # Check if Scenario 1 in sequential sweep triggers the plume advisory
        if 1 in self._scenarios_pending_advisory and 1 not in self._scenario_advisory_decisions:
            self._prompt_scenario_plume_advisory(1)
            if 1 not in self._scenario_advisory_decisions:
                self._sweep_paused_current = True
                self.sim_isRunning = False
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText("Status: Scenario 1 held awaiting plume decision. Select it in dropdown to choose.")
                return

        first_val = config["values"][0] if total_scenarios > 0 else 0.0
        first_val_str = (
            format_sci_value(first_val)
            if (config.get("param_id") in ("n0_plasma", "n0") or abs(first_val) >= 1e4 or (0 < abs(first_val) <= 1e-3))
            else f"{first_val:.3g}"
        )

        # Update Top Bar Badge & Button
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep mode active [1/{total_scenarios}]"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(False)
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(
                f"Starting Parametric Sweep: {param_label} ({total_scenarios} scenarios)..."
            )

        # Apply initial scenario value
        self._apply_sweep_value(config["param_id"], first_val)
        is_trunc = (self._scenario_advisory_decisions.get(1) == "truncate")
        self.truncate_downstream_plume = is_trunc
        if hasattr(self, 'sim') and self.sim is not None:
            self.sim.truncate_downstream_plume = is_trunc
        self.build_domain(force=True)

        self.sweep_scenario_start_iter = getattr(self.sim, 'iteration', 0)
        self.sweep_scenario_entry_iter = None

        if not getattr(self, 'sim_isRunning', False):
            self.sim.injection_enabled = True
            self.sim_isRunning = True
            self._resume_simulation_timer()
            self._sim_worker.start_sim(self.get_params(), run_id=getattr(self, '_sim_run_id', 0))
            self.update_sim_status_badge("RUNNING")
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("PAUSE BEAM")
        self._update_sweep_action_button_state()

    def _start_parallel_sweep(self, config, n_cores, remaining_indices=None):
        """Launch parallel multi-core sweep using ProcessPoolExecutor and top-bar scenario selector."""
        self.is_sweep_active = True
        self.sweep_mode_armed = False
        self._sweep_paused_current = False
        total_scenarios = len(config.get("values", []))
        param_label = config.get("param_label", "Parameter")

        # Stop live single simulator if currently running
        if hasattr(self, '_sim_worker') and self._sim_worker.isRunning():
            self._sim_worker.stop(wait_ms=1000)
        self.sim_isRunning = False
        self._sweep_paused_current = False
        self._sweep_stopped = False
        if not hasattr(self, '_sweep_status_dict'):
            self._sweep_status_dict = {}

        if remaining_indices is None:
            self.sweep_results = []
            self._sweep_status_dict.clear()
            if hasattr(self, '_sweep_tracking_buffers') and isinstance(self._sweep_tracking_buffers, dict):
                self._sweep_tracking_buffers.clear()
            if hasattr(self, '_seq_scenario_histories') and isinstance(self._seq_scenario_histories, dict):
                self._seq_scenario_histories.clear()
            done_count = 0
        else:
            if not hasattr(self, 'sweep_results'):
                self.sweep_results = []
            done_count = len(self.sweep_results)

        if hasattr(self, '_resume_simulation_timer'):
            self._resume_simulation_timer()

        # Update Top Bar Badge & Button
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText(
                f"⚡ Parallel Sweep [{n_cores} cores | {done_count}/{total_scenarios} Done]"
            )
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #f3e8ff; color: #7e22ce; font-weight: bold; "
                "border: 1px solid #d8b4fe; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )

        # Initialize and display top-bar scenario selector dropdown
        if hasattr(self, 'combo_sweep_scenario'):
            self.combo_sweep_scenario.blockSignals(True)
            if remaining_indices is None or self.combo_sweep_scenario.count() != total_scenarios:
                self.combo_sweep_scenario.clear()
                for idx, val in enumerate(config.get("values", [])):
                    val_str = (
                        format_sci_value(val)
                        if (config.get("param_id") in ("n0_plasma", "n0") or abs(val) >= 1e4 or (0 < abs(val) <= 1e-3))
                        else f"{val:.3g}"
                    )
                    unit_str = f" {config.get('param_unit', '')}" if config.get('param_unit') else ""
                    self.combo_sweep_scenario.addItem(f"Scenario {idx + 1}: {config.get('param_id', 'param')} = {val_str}{unit_str}  [⏳ Queued]")

            sel_idx = (remaining_indices[0] - 1) if remaining_indices else 0
            sel_idx = max(0, min(sel_idx, self.combo_sweep_scenario.count() - 1))
            self.combo_sweep_scenario.setCurrentIndex(sel_idx)
            self.combo_sweep_scenario.blockSignals(False)
            if hasattr(self, 'sweep_control_group'):
                self.sweep_control_group.setVisible(True)
            self.combo_sweep_scenario.setVisible(True)

        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(False)
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("PAUSE BEAM")
        self._update_sweep_action_button_state()

        if hasattr(self, 'lbl_status'):
            rem_str = f" (resuming {len(remaining_indices)} remaining)" if remaining_indices else ""
            self.lbl_status.setText(
                f"Running Parallel Sweep: {param_label} ({total_scenarios} scenarios on {n_cores} cores{rem_str}). Switch active views via top dropdown."
            )

        base_params = self.get_params() if hasattr(self, 'get_params') else {}
        self._sweep_base_params = copy.deepcopy(base_params)
        self._scenario_domain_cache = {}
        self._current_rendered_scenario_grid_idx = None
        scatter_limit = getattr(self, 'scatter_limit', getattr(getattr(self, '_sim_worker', None), '_scatter_limit', 2000))

        held_indices = [
            s for s in getattr(self, '_scenarios_pending_advisory', {}).keys()
            if s not in getattr(self, '_scenario_advisory_decisions', {})
        ]

        track_ptcls = bool(self.chk_track_ptcls.isChecked()) if hasattr(self, 'chk_track_ptcls') and self.chk_track_ptcls is not None else False
        self.parallel_sweep_worker = ParallelSweepWorker(
            config,
            base_params,
            n_cores=n_cores,
            scatter_limit=scatter_limit,
            parent=self,
            remaining_indices=remaining_indices,
            existing_results=self.sweep_results,
            held_scenarios=held_indices,
            track_particles=track_ptcls,
        )

        self.parallel_sweep_worker.progress_updated.connect(self._on_parallel_progress_updated)
        self.parallel_sweep_worker.scenario_completed.connect(self._on_parallel_scenario_completed)
        self.parallel_sweep_worker.sweep_finished.connect(self._on_parallel_sweep_finished)
        self.parallel_sweep_worker.sweep_error.connect(self._on_parallel_sweep_error)
        self.parallel_sweep_worker.scenario_action_needed.connect(self._on_scenario_action_needed)

        # Start live GUI monitor timer to sync top-bar dropdown items and active canvas view
        if not hasattr(self, 'sweep_monitor_timer') or self.sweep_monitor_timer is None:
            self.sweep_monitor_timer = QTimer(self)
            self.sweep_monitor_timer.timeout.connect(self._on_sweep_monitor_tick)
        self.sweep_monitor_timer.start(150)

        self.parallel_sweep_worker.start()
        active_view_idx = remaining_indices[0] if remaining_indices else 1
        self.parallel_sweep_worker.set_active_view_scenario(active_view_idx)

    def _on_sweep_monitor_tick(self):
        """Periodic tick to update dropdown labels and render active scenario view."""
        self._update_sweep_scenario_combo_items()
        if hasattr(self, 'combo_sweep_scenario') and self.combo_sweep_scenario is not None:
            active_idx = self.combo_sweep_scenario.currentIndex()
            if active_idx >= 0:
                self._update_toggle_btn_for_scenario(active_idx + 1)
                self._render_sweep_scenario_snapshot(active_idx + 1)

    def _on_scenario_action_needed(self, scenario_idx: int, adv_info: dict):
        """Worker reached a held scenario; prompt user for decision."""
        if scenario_idx not in getattr(self, '_scenario_advisory_decisions', {}):
            self._prompt_scenario_plume_advisory(scenario_idx)

    def _prompt_scenario_plume_advisory(self, scenario_idx: int):
        """Show interactive modal dialog for a held scenario and apply choice."""
        try:
            adv_info = getattr(self, '_scenarios_pending_advisory', {}).get(scenario_idx)
            if not adv_info:
                return

            from gui.dialogs import ScenarioPlumeAdvisoryDialog

            remaining_held = [
                s for s in getattr(self, '_scenarios_pending_advisory', {}).keys()
                if s not in getattr(self, '_scenario_advisory_decisions', {})
            ]

            # In automated test environments (offscreen), default to truncate without blocking
            if os.environ.get("QT_QPA_PLATFORM") == "offscreen" or os.environ.get("PYTEST_CURRENT_TEST"):
                choice = "truncate"
                apply_all = True
            else:
                dlg = ScenarioPlumeAdvisoryDialog(
                    scen_num=scenario_idx,
                    advisory_info=adv_info,
                    remaining_held_count=len(remaining_held),
                    parent=self,
                )
                dlg.exec_()
                choice = dlg.get_choice()
                apply_all = dlg.apply_to_all()

            targets = remaining_held if apply_all else [scenario_idx]

            if choice in ("truncate", "full"):
                truncate_plume = (choice == "truncate")
                for s_num in targets:
                    self._scenario_advisory_decisions[s_num] = choice
                    if hasattr(self, '_scenario_domain_cache') and isinstance(self._scenario_domain_cache, dict):
                        self._scenario_domain_cache.pop(s_num, None)

                    # Release in parallel sweep worker
                    if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
                        self.parallel_sweep_worker.release_scenario(s_num, truncate_plume=truncate_plume)

                    # In sequential sweep if this is the active scenario:
                    if not getattr(self, 'is_parallel_sweep', False) and getattr(self, 'sweep_current_idx', 0) + 1 == s_num:
                        if hasattr(self, 'sim'):
                            self.sim.truncate_downstream_plume = truncate_plume
                        self._apply_sweep_value(self.sweep_config["param_id"], adv_info.get("parameter_value"))
                        self.build_domain(force=True)
                        if not getattr(self, 'sim_isRunning', False):
                            self.sim.injection_enabled = True
                            self.sim_isRunning = True
                            self._resume_simulation_timer()
                            self._sim_worker.start_sim(self.get_params(), run_id=getattr(self, '_sim_run_id', 0))
                            self.update_sim_status_badge("RUNNING")

                self._update_scenario_selector_dropdown()
                strategy_name = "Truncated Plume (High PPC)" if truncate_plume else "Full Domain (Modulated PPC)"
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText(f"Status: Scenario(s) {targets} released with {strategy_name}.")
        except Exception as err:
            import traceback
            traceback.print_exc()
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(f"Status: Error in plume advisory dialog: {err}")

    def _on_sweep_scenario_selected(self, index):
        """User switched scenario in top dropdown: immediately refresh canvas, inputs, and toggle button."""
        if index < 0:
            return
        scenario_idx = index + 1

        try:
            # Check if selected scenario requires user action on downstream plume advisory
            if (
                scenario_idx in getattr(self, '_scenarios_pending_advisory', {})
                and scenario_idx not in getattr(self, '_scenario_advisory_decisions', {})
            ):
                self._prompt_scenario_plume_advisory(scenario_idx)
        except Exception as err:
            import traceback
            traceback.print_exc()

        cfg = getattr(self, 'sweep_config', {})
        values = cfg.get("values", [])
        param_id = cfg.get("param_id", "")

        # 1. Update left panel parameter inputs
        if 0 <= index < len(values):
            val = values[index]
            self._apply_sweep_value(param_id, val)

        if getattr(self, 'sweep_mode_armed', False) and not getattr(self, 'is_sweep_active', False):
            if getattr(self, 'sim_isRunning', False):
                if hasattr(self, '_sim_worker'):
                    self._sim_worker.pause()
                self.sim_isRunning = False
                self._sweep_paused_current = True
            self.sweep_current_idx = index

        # 2. Sync grid spinboxes and auto-computed inputs from cached scenario domain data
        d_data = self._get_scenario_domain_data(scenario_idx)
        scenario_params = d_data.get("params", {}) if d_data else {}
        if (
            scenario_params
            and "upstream_gap_mm" in scenario_params
            and hasattr(self, 'inputs')
            and "upstream_gap_mm" in self.inputs
        ):
            try:
                gap_val = float(scenario_params["upstream_gap_mm"])
                self.inputs["upstream_gap_mm"].blockSignals(True)
                self.inputs["upstream_gap_mm"].setValue(gap_val)
                self.inputs["upstream_gap_mm"].blockSignals(False)
                if cfg.get("auto_update_upstream_gap", False):
                    self._last_auto_gap = round(gap_val, 3)
            except Exception:
                pass
        if (
            scenario_params
            and not scenario_params.get("user_overrode_neut_x", True)
            and "neut_x" in scenario_params
            and hasattr(self, 'inputs')
            and "neut_x" in self.inputs
        ):
            try:
                neut_val = float(scenario_params["neut_x"])
                self.inputs["neut_x"].blockSignals(True)
                self.inputs["neut_x"].setValue(neut_val)
                self.inputs["neut_x"].blockSignals(False)
                self._last_auto_neut_x = round(neut_val, 3)
            except Exception:
                pass
        scenario_grids = d_data.get("grids", [])
        if hasattr(self, 'grid_widgets') and scenario_grids:
            for g_idx, g_dict in enumerate(scenario_grids):
                if g_idx < len(self.grid_widgets):
                    gw = self.grid_widgets[g_idx]
                    for k in ("V", "t", "gap", "r", "cham"):
                        if k in gw and k in g_dict:
                            try:
                                gw[k].blockSignals(True)
                                gw[k].setValue(float(g_dict[k]))
                                gw[k].blockSignals(False)
                            except Exception:
                                pass

        # 3. Inform parallel worker which scenario is active so it extracts particles for GUI while non-viewed scenarios speed up
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            self.parallel_sweep_worker.set_active_view_scenario(scenario_idx)

        # 4. Update Pause/Resume toggle button text based on this scenario's state
        self._update_toggle_btn_for_scenario(scenario_idx)

        # 5. Render snapshot on canvas
        self._render_sweep_scenario_snapshot(scenario_idx)
        self._update_sweep_scenario_combo_items()

    def _update_sweep_scenario_combo_items(self):
        """Update top-bar scenario dropdown items with real-time status and iteration count."""
        if not hasattr(self, 'combo_sweep_scenario') or self.sweep_config is None:
            return

        cfg = self.sweep_config
        values = cfg.get("values", [])
        param_id = cfg.get("param_id", "")
        param_unit = cfg.get("param_unit", "")

        status_dict = {}
        if hasattr(self, '_sweep_status_dict') and self._sweep_status_dict:
            status_dict.update(self._sweep_status_dict)
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            worker_dict = self.parallel_sweep_worker.get_status_dict()
            for k, v in worker_dict.items():
                if v and (v.get("iter", 0) > 0 or v.get("status") in ("running", "paused", "done")):
                    status_dict[k] = v
                elif k not in status_dict:
                    status_dict[k] = v

        is_parallel = (
            hasattr(self, 'parallel_sweep_worker')
            and self.parallel_sweep_worker is not None
            and self.parallel_sweep_worker.isRunning()
        )
        completed_map = {
            r["scenario_idx"]: r
            for r in getattr(self, 'sweep_results', [])
            if isinstance(r, dict) and "scenario_idx" in r
        }
        current_idx = self.combo_sweep_scenario.currentIndex()
        self.combo_sweep_scenario.blockSignals(True)
        try:
            for idx, val in enumerate(values):
                scen_num = idx + 1
                info = status_dict.get(scen_num, {})
                status = info.get("status", "queued")
                scen_iter = info.get("iter", 0)

                if scen_num in completed_map:
                    status = "done"
                    ex_row = completed_map[scen_num]
                    scen_iter = ex_row.get("steady_iterations", ex_row.get("iter", scen_iter))
                    if "computation_time_s" in ex_row:
                        info["computation_time_s"] = ex_row["computation_time_s"]
                elif is_parallel:
                    if self.parallel_sweep_worker.is_scenario_paused(scen_num):
                        status = "paused"
                    elif status == "paused":
                        status = "running"
                elif status not in ("done", "stopped"):
                    cur_seq_idx = getattr(self, 'sweep_current_idx', 0)
                    if idx == cur_seq_idx:
                        if getattr(self, 'sim_isRunning', False):
                            status = "running"
                            scen_iter = getattr(self.sim, 'iteration', 0)
                        elif getattr(self, '_sweep_paused_current', False) or getattr(self, '_sweep_stopped', False):
                            status = "paused"
                            scen_iter = getattr(self.sim, 'iteration', 0)
                        elif getattr(self, 'is_sweep_active', False):
                            status = "running"
                            scen_iter = getattr(self.sim, 'iteration', 0)
                        else:
                            status = "queued"
                    else:
                        status = "queued"

                val_str = (
                    format_sci_value(val)
                    if (param_id in ("n0_plasma", "n0") or abs(val) >= 1e4 or (0 < abs(val) <= 1e-3))
                    else f"{val:.3g}"
                )
                unit_str = f" {param_unit}" if param_unit else ""

                comp_s = info.get("computation_time_s")
                comp_tag = f", {format_comp_duration(comp_s)}" if (comp_s is not None and np.isfinite(comp_s) and comp_s > 0) else ""

                is_held_advisory = (
                    scen_num in getattr(self, '_scenarios_pending_advisory', {})
                    and scen_num not in getattr(self, '_scenario_advisory_decisions', {})
                )

                if status == "running":
                    tag = f"[▶ Running - {scen_iter} iters]"
                elif status == "paused":
                    tag = f"[⏸ Paused - {scen_iter} iters]"
                elif status == "done":
                    tag = f"[✔ Done - {scen_iter} iters{comp_tag}]"
                elif status == "stopped":
                    tag = f"[⏹ Stopped]"
                elif is_held_advisory or status == "action_required":
                    tag = "[⚠️ Action Required - Plume Advisory]"
                else:
                    tag = "[⏳ Queued]"

                item_text = f"Scenario {scen_num}: {param_id} = {val_str}{unit_str}  {tag}"

                if idx < self.combo_sweep_scenario.count():
                    if self.combo_sweep_scenario.itemText(idx) != item_text:
                        self.combo_sweep_scenario.setItemText(idx, item_text)
                else:
                    self.combo_sweep_scenario.addItem(item_text)

            if current_idx >= 0 and current_idx < self.combo_sweep_scenario.count():
                self.combo_sweep_scenario.setCurrentIndex(current_idx)
        finally:
            self.combo_sweep_scenario.blockSignals(False)

    def _get_scenario_domain_data(self, scenario_idx):
        """Retrieve or compute domain geometry, grid electrode boundaries, and potential contour for scenario_idx."""
        if not hasattr(self, '_scenario_domain_cache'):
            self._scenario_domain_cache = {}

        status_dict = self._get_merged_sweep_status_dict() if hasattr(self, '_get_merged_sweep_status_dict') else {}
        info = dict(status_dict.get(scenario_idx, {})) if status_dict else {}

        cfg = getattr(self, 'sweep_config', {})
        if not cfg:
            return {}

        values = cfg.get("values", [])
        if scenario_idx < 1 or scenario_idx > len(values):
            return {}

        param_id = cfg.get("param_id", "")
        val = values[scenario_idx - 1]
        base_params = getattr(self, '_sweep_base_params', None)
        if base_params is None and hasattr(self, 'get_params'):
            base_params = self.get_params()
        elif base_params is None:
            base_params = {}

        is_trunc = (
            getattr(self, '_scenario_advisory_decisions', {}).get(scenario_idx) == "truncate"
            or bool(info.get("truncate_downstream_plume", False))
            or bool(cfg.get("truncate_downstream_plume", False))
            or (info.get("Lx") is not None and abs(info.get("Lx") - float(base_params.get("Lx", 5.26))) > 0.5)
        )

        if scenario_idx in self._scenario_domain_cache:
            cached = self._scenario_domain_cache[scenario_idx]
            if cached.get("truncate_downstream_plume", False) == is_trunc:
                return cached

        from physics.headless_sweep import apply_sweep_param_to_dict
        scenario_params = apply_sweep_param_to_dict(base_params, param_id, val, sweep_config=cfg)
        if is_trunc:
            scenario_params["truncate_downstream_plume"] = True

        try:
            from physics.simulator import DigitalTwinSimulator
            sim_temp = DigitalTwinSimulator()
            sim_temp.build_domain(scenario_params)

            gy, gx = np.where(sim_temp.isBound)
            x_pts = getattr(sim_temp, 'x_coords', getattr(sim_temp, 'xpts', None))
            y_pts = getattr(sim_temp, 'y_coords', getattr(sim_temp, 'ypts', None))
            if x_pts is not None and y_pts is not None and len(x_pts) > 0 and len(y_pts) > 0:
                bound_xy = np.column_stack((x_pts[gx], y_pts[gy])).astype(np.float32)
            else:
                bound_xy = np.column_stack((gx * sim_temp.dx, gy * sim_temp.dy)).astype(np.float32)

            data = {
                "truncate_downstream_plume": is_trunc,
                "bound_xy": bound_xy,
                "X": sim_temp.X,
                "Y": sim_temp.Y,
                "V": sim_temp.V,
                "Vdc": getattr(sim_temp, 'Vdc', sim_temp.V),
                "Ex": getattr(sim_temp, 'Ex', None),
                "Ey": getattr(sim_temp, 'Ey', None),
                "Lx": float(sim_temp.Lx),
                "Ly": float(sim_temp.Ly),
                "dx": float(sim_temp.dx),
                "dy": float(sim_temp.dy),
                "nx": int(sim_temp.nx),
                "ny": int(sim_temp.ny),
                "isBound": sim_temp.isBound,
                "Tmap": sim_temp.Tmap.copy() if getattr(sim_temp, 'Tmap', None) is not None else None,
                "mask_grids": [m.copy() for m in getattr(sim_temp, 'mask_grids', [])] if getattr(sim_temp, 'mask_grids', None) else [],
                "T_grids": [float(t) for t in getattr(sim_temp, 'T_grids', [])],
                "x_coords": getattr(sim_temp, 'x_coords', None),
                "y_coords": getattr(sim_temp, 'y_coords', None),
                "zone_configs": getattr(sim_temp, 'zone_configs', None),
                "upstream_gap_mm": float(getattr(sim_temp, 'upstream_gap_mm', 0.0)),
                "grid_x_starts": getattr(sim_temp, 'grid_x_starts', None),
                "grid_x_ends": getattr(sim_temp, 'grid_x_ends', None),
                "grids": scenario_params.get("grids", []),
                "params": scenario_params,
            }
            self._scenario_domain_cache[scenario_idx] = data
            return data
        except Exception:
            return {}

    def _sync_sim_domain_from_scenario_data(self, d_data, include_fields=False):
        """Synchronize self.sim geometry and mesh metadata from a cached scenario domain dict."""
        if not d_data or not hasattr(self, 'sim') or self.sim is None:
            return
        if d_data.get("X") is not None and d_data.get("Y") is not None:
            self.sim.X = d_data["X"]
            self.sim.Y = d_data["Y"]
            self.sim.nx = d_data["X"].shape[1]
            self.sim.ny = d_data["X"].shape[0]
        for k in ("dx", "dy", "upstream_gap_mm"):
            if d_data.get(k) is not None:
                setattr(self.sim, k, float(d_data[k]))
        for k in ("Lx", "Ly"):
            if d_data.get(k) is not None:
                setattr(self.sim, k, float(d_data[k]))
        for k in ("V", "isBound", "Tmap", "mask_grids", "T_grids", "x_coords", "y_coords", "zone_configs", "grid_x_starts", "grid_x_ends", "grids"):
            if d_data.get(k) is not None:
                setattr(self.sim, k, d_data[k])
        if "truncate_downstream_plume" in d_data:
            self.sim.truncate_downstream_plume = bool(d_data["truncate_downstream_plume"])
        if include_fields:
            if d_data.get("Vdc") is not None:
                self.sim.Vdc = d_data["Vdc"]
            elif d_data.get("V") is not None:
                self.sim.Vdc = d_data["V"]

    def _update_toggle_btn_for_scenario(self, scenario_idx):
        """Update btn_toggle text ('PAUSE BEAM' or 'RESUME BEAM') based on selected scenario state."""
        if not hasattr(self, 'btn_toggle'):
            return

        is_held = (
            scenario_idx in getattr(self, '_scenarios_pending_advisory', {})
            and scenario_idx not in getattr(self, '_scenario_advisory_decisions', {})
        )
        if is_held:
            self.btn_toggle.setText("⚠️ CONFIGURE")
            return

        # Parallel sweep worker running (either active or paused/stopped)
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None and self.parallel_sweep_worker.isRunning():
            is_paused = self.parallel_sweep_worker.is_scenario_paused(scenario_idx)
            if is_paused or getattr(self, '_sweep_stopped', False) or not getattr(self, 'is_sweep_active', False):
                self.btn_toggle.setText("RESUME BEAM")
            else:
                status_dict = self.parallel_sweep_worker.get_status_dict()
                st = status_dict.get(scenario_idx, {}).get("status", "running")
                if st in ("done", "stopped", "paused"):
                    self.btn_toggle.setText("RESUME BEAM")
                else:
                    self.btn_toggle.setText("PAUSE BEAM")
            return

        if getattr(self, 'sweep_mode_armed', False) and not getattr(self, 'is_sweep_active', False):
            cur_seq_idx = getattr(self, 'sweep_current_idx', 0)
            if (scenario_idx - 1) == cur_seq_idx:
                if getattr(self, 'sim_isRunning', False):
                    self.btn_toggle.setText("PAUSE BEAM")
                elif getattr(self, '_sweep_paused_current', False):
                    self.btn_toggle.setText("RESUME BEAM")
                else:
                    self.btn_toggle.setText("2. START BEAM")
            else:
                self.btn_toggle.setText("2. START BEAM")
            return

        if not getattr(self, 'is_sweep_active', False):
            if getattr(self, '_sweep_stopped', False):
                self.btn_toggle.setText("RESUME BEAM")
            else:
                self.btn_toggle.setText("2. START BEAM")
            return

        # Sequential sweep
        cur_idx = getattr(self, 'sweep_current_idx', 0)
        if (scenario_idx - 1) == cur_idx:
            if getattr(self, 'sim_isRunning', False):
                self.btn_toggle.setText("PAUSE BEAM")
            else:
                self.btn_toggle.setText("RESUME BEAM")
        else:
            self.btn_toggle.setText("RESUME BEAM")

    def toggle_sweep_scenario_pause(self):
        """Toggle pause/resume for the scenario currently selected in combo_sweep_scenario."""
        if not hasattr(self, 'combo_sweep_scenario') or self.sweep_config is None:
            return

        # If the entire sweep was stopped, resume the whole sweep
        if getattr(self, '_sweep_stopped', False):
            if hasattr(self, 'resume_sweep_with_selected'):
                self.resume_sweep_with_selected()
                return

        cur_idx = self.combo_sweep_scenario.currentIndex()
        if cur_idx < 0:
            cur_idx = 0
        scen_num = cur_idx + 1

        # If scenario is held on downstream plume advisory, prompt user choice
        if (
            scen_num in getattr(self, '_scenarios_pending_advisory', {})
            and scen_num not in getattr(self, '_scenario_advisory_decisions', {})
        ):
            self._prompt_scenario_plume_advisory(scen_num)
            return

        # Case A: Parallel sweep active
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None and self.parallel_sweep_worker.isRunning():
            if self.parallel_sweep_worker.is_scenario_paused(scen_num):
                self.parallel_sweep_worker.resume_scenario(scen_num)
                self.is_sweep_active = True
                self._sweep_stopped = False
                if hasattr(self, '_sweep_status_dict') and self._sweep_status_dict and scen_num in self._sweep_status_dict:
                    del self._sweep_status_dict[scen_num]
                if hasattr(self, '_resume_simulation_timer'):
                    self._resume_simulation_timer()
                if hasattr(self, 'btn_stop_sweep'):
                    self.btn_stop_sweep.setVisible(False)
                if hasattr(self, '_update_sweep_action_button_state'):
                    self._update_sweep_action_button_state()
                if hasattr(self, 'btn_toggle'):
                    self.btn_toggle.setText("PAUSE BEAM")
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText(f"Resumed simulation for Scenario {scen_num}.")
                if not hasattr(self, 'sweep_monitor_timer') or self.sweep_monitor_timer is None:
                    self.sweep_monitor_timer = QTimer(self)
                    self.sweep_monitor_timer.timeout.connect(self._on_sweep_monitor_tick)
                if not self.sweep_monitor_timer.isActive():
                    self.sweep_monitor_timer.start(150)
            else:
                self.parallel_sweep_worker.pause_scenario(scen_num)
                if hasattr(self, 'btn_toggle'):
                    self.btn_toggle.setText("RESUME BEAM")
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText(f"Paused simulation for Scenario {scen_num}.")
                if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker:
                    all_paused = all(self.parallel_sweep_worker.is_scenario_paused(s) for s in self.parallel_sweep_worker.pause_files)
                    if all_paused and hasattr(self, '_pause_simulation_timer'):
                        self._pause_simulation_timer()
            self._update_sweep_scenario_combo_items()
            return

        # Case B: Sequential sweep active
        current_running_idx = getattr(self, 'sweep_current_idx', 0)
        if cur_idx == current_running_idx:
            if getattr(self, 'sim_isRunning', False):
                self.sim_isRunning = False
                self._sweep_paused_current = True
                if hasattr(self, '_sim_worker'):
                    self._sim_worker.pause()
                self._pause_simulation_timer()
                if hasattr(self, 'btn_toggle'):
                    self.btn_toggle.setText("RESUME BEAM")
                self.update_sim_status_badge("PAUSED")
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText(f"Paused simulation for Scenario {scen_num}.")
            else:
                self.sim_isRunning = True
                self._sweep_paused_current = False
                self._resume_simulation_timer()
                if hasattr(self, 'btn_toggle'):
                    self.btn_toggle.setText("PAUSE BEAM")
                if hasattr(self, '_sim_worker'):
                    self._sim_worker.start_sim(self.get_params(), run_id=getattr(self, '_sim_run_id', 0))
                self.update_sim_status_badge("RUNNING")
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText(f"Resumed simulation for Scenario {scen_num}.")
            self.update_simulation_timer()
            self._update_sweep_scenario_combo_items()
        else:
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(
                    f"Scenario {scen_num} is not currently running. Click 'Resume with Selected Parameters' to run it."
                )

    def _render_sweep_scenario_snapshot(self, scenario_idx):
        """Render the chosen sweep scenario's particle data and physical observables on the canvas."""
        status_dict = self._get_merged_sweep_status_dict()
        info = dict(status_dict.get(scenario_idx, {}))

        # 0. Sync live axes limits, grid electrode boundaries, and potential contour
        d_data = self._get_scenario_domain_data(scenario_idx)
        lx = info.get("Lx") if info.get("Lx") is not None else d_data.get("Lx")
        ly = info.get("Ly") if info.get("Ly") is not None else d_data.get("Ly")
        if lx is not None and ly is not None:
            lx_f = float(lx)
            ly_f = float(ly)
            if hasattr(self, 'ax_live'):
                self.ax_live.set_xlim(0, lx_f)
                self.ax_live.set_ylim(0, ly_f)
            if hasattr(self, 'ax_temp'):
                self.ax_temp.set_xlim(0, lx_f)
                self.ax_temp.set_ylim(0, ly_f)
            if hasattr(self, 'ax_dmg'):
                self.ax_dmg.set_xlim(0, lx_f)
                self.ax_dmg.set_ylim(0, ly_f)

        bound_xy = d_data.get("bound_xy")
        live_is_bound = info.get("isBound")
        live_X = info.get("X", d_data.get("X"))
        live_Y = info.get("Y", d_data.get("Y"))
        if (
            live_is_bound is not None
            and live_X is not None
            and live_Y is not None
            and getattr(live_is_bound, "shape", None) == getattr(live_X, "shape", None)
        ):
            gy_l, gx_l = np.where(live_is_bound)
            bound_xy = np.column_stack((live_X[gy_l, gx_l], live_Y[gy_l, gx_l])).astype(np.float32)

        if bound_xy is not None and hasattr(self, 'scat_bound') and self.scat_bound is not None:
            self.scat_bound.set_offsets(bound_xy)
            self.scat_bound.set_zorder(2)
        elif bound_xy is not None and hasattr(self, 'ax_live'):
            self.scat_bound = self.ax_live.scatter(bound_xy[:, 0], bound_xy[:, 1], s=4, c="k", alpha=0.8, zorder=2)

        if getattr(self, '_current_rendered_scenario_grid_idx', None) != scenario_idx:
            X = info.get("X")
            Y = info.get("Y")
            V = info.get("V")

            if (X is None or Y is None) and d_data:
                if d_data.get("X") is not None and d_data.get("Y") is not None:
                    if lx is not None and abs(d_data["X"].max() - float(lx)) < 0.05:
                        X = d_data["X"]
                        Y = d_data["Y"]

            if (X is None or Y is None) and lx is not None and ly is not None:
                target_nx = int(info.get("nx", d_data.get("nx", (V.shape[1] if V is not None else 100))))
                target_ny = int(info.get("ny", d_data.get("ny", (V.shape[0] if V is not None else 50))))
                x_pts = info.get("x_coords") if (info.get("x_coords") is not None and len(info["x_coords"]) == target_nx) else (
                    d_data.get("x_coords") if (d_data.get("x_coords") is not None and len(d_data["x_coords"]) == target_nx and abs(d_data["x_coords"][-1] - float(lx)) < 0.05) else np.linspace(0, float(lx), target_nx)
                )
                y_pts = info.get("y_coords") if (info.get("y_coords") is not None and len(info["y_coords"]) == target_ny) else (
                    d_data.get("y_coords") if (d_data.get("y_coords") is not None and len(d_data["y_coords"]) == target_ny) else np.linspace(0, float(ly), target_ny)
                )
                X, Y = np.meshgrid(x_pts, y_pts)

            if V is None or (X is not None and getattr(V, 'shape', None) != X.shape):
                if d_data.get("V") is not None and (X is not None and d_data["V"].shape == X.shape):
                    V = d_data["V"]

            if X is not None and Y is not None and V is not None and V.shape == X.shape and hasattr(self, 'ax_live'):
                if hasattr(self, 'contour_live') and self.contour_live is not None:
                    try:
                        self.contour_live.remove()
                    except Exception:
                        try:
                            for coll in getattr(self.contour_live, 'collections', []):
                                coll.remove()
                        except Exception:
                            pass
                    self.contour_live = None

                # Clean up any other lingering ContourSets on ax_live to prevent contour overlapping and yellow fog
                try:
                    import matplotlib.contour
                    for art in list(self.ax_live.collections):
                        if isinstance(art, matplotlib.contour.ContourSet):
                            art.remove()
                except Exception:
                    pass

                self.contour_live = self.ax_live.contourf(X, Y, V, 20, cmap="viridis", alpha=0.4, zorder=0)
                if hasattr(self, 'scat_bound') and self.scat_bound is not None:
                    self.scat_bound.set_zorder(2)
                if hasattr(self, 'scat_prim') and self.scat_prim is not None:
                    self.scat_prim.set_zorder(3)
                if hasattr(self, 'scat_cex') and self.scat_cex is not None:
                    self.scat_cex.set_zorder(4)
                if hasattr(self, 'scat_elec') and self.scat_elec is not None:
                    self.scat_elec.set_zorder(5)
                self._current_rendered_scenario_grid_idx = scenario_idx
                self._bg_live = None

        if bound_xy is not None and hasattr(self, 'scat_dmg_bound') and self.scat_dmg_bound is not None:
            self.scat_dmg_bound.set_offsets(bound_xy)
            self.scat_dmg_bound.set_zorder(2)

        # 1. Update primary ion particles, CEX ions, and electrons
        prim_xy = info.get("prim_xy")
        e_prim_eV = info.get("e_prim_eV")
        cex_xy = info.get("cex_xy")
        e_cex_eV = info.get("e_cex_eV")
        elec_xy = info.get("elec_xy")

        # Respect active scatter_limit (None = hide all, 0 = all, N = max N)
        scatter_lim = getattr(self, 'scatter_limit', getattr(getattr(self, '_sim_worker', None), '_scatter_limit', 2000))
        if scatter_lim is None:
            prim_xy = np.empty((0, 2), dtype=np.float32)
            cex_xy = np.empty((0, 2), dtype=np.float32)
            elec_xy = np.empty((0, 2), dtype=np.float32)
            e_prim_eV = np.empty(0, dtype=np.float32)
            e_cex_eV = np.empty(0, dtype=np.float32)
        elif scatter_lim > 0:
            if prim_xy is not None and len(prim_xy) > scatter_lim:
                sub_idx = np.random.choice(len(prim_xy), scatter_lim, replace=False)
                sub_idx.sort()
                prim_xy = prim_xy[sub_idx]
                if e_prim_eV is not None and len(e_prim_eV) >= len(sub_idx):
                    e_prim_eV = e_prim_eV[sub_idx]
            if cex_xy is not None and len(cex_xy) > scatter_lim:
                sub_idx = np.random.choice(len(cex_xy), scatter_lim, replace=False)
                sub_idx.sort()
                cex_xy = cex_xy[sub_idx]
                if e_cex_eV is not None and len(e_cex_eV) >= len(sub_idx):
                    e_cex_eV = e_cex_eV[sub_idx]
            if elec_xy is not None and len(elec_xy) > scatter_lim:
                sub_idx = np.random.choice(len(elec_xy), scatter_lim, replace=False)
                sub_idx.sort()
                elec_xy = elec_xy[sub_idx]

        energy_min = float('inf')
        energy_max = float('-inf')

        if prim_xy is not None and len(prim_xy) > 0 and hasattr(self, 'scat_prim') and self.scat_prim is not None:
            self.scat_prim.set_offsets(prim_xy)
            if e_prim_eV is not None and len(e_prim_eV) > 0:
                self.scat_prim.set_array(e_prim_eV)
                energy_min = min(energy_min, float(np.min(e_prim_eV)))
                energy_max = max(energy_max, float(np.max(e_prim_eV)))
        elif hasattr(self, 'scat_prim') and self.scat_prim is not None:
            self.scat_prim.set_offsets(np.empty((0, 2)))

        if cex_xy is not None and len(cex_xy) > 0 and hasattr(self, 'scat_cex') and self.scat_cex is not None:
            self.scat_cex.set_offsets(cex_xy)
            if e_cex_eV is not None and len(e_cex_eV) > 0:
                self.scat_cex.set_array(e_cex_eV)
                energy_min = min(energy_min, float(np.min(e_cex_eV)))
                energy_max = max(energy_max, float(np.max(e_cex_eV)))
        elif hasattr(self, 'scat_cex') and self.scat_cex is not None:
            self.scat_cex.set_offsets(np.empty((0, 2)))

        if energy_min != float('inf') and hasattr(self, 'scat_prim') and self.scat_prim is not None:
            if energy_max <= energy_min:
                energy_max = energy_min + 1.0
            self.scat_prim.set_clim(energy_min, energy_max)
            if hasattr(self, 'scat_cex') and self.scat_cex is not None:
                self.scat_cex.set_clim(energy_min, energy_max)

        if elec_xy is not None and len(elec_xy) > 0 and hasattr(self, 'scat_elec') and self.scat_elec is not None:
            self.scat_elec.set_offsets(elec_xy)
        elif hasattr(self, 'scat_elec') and self.scat_elec is not None:
            self.scat_elec.set_offsets(np.empty((0, 2)))

        # 2. Update Divergence and Saddle-Point Potential history graphs and sync telemetry/tracking
        self._sync_sweep_scenario_data_to_app(scenario_idx, info)

        iters = self.iter_history
        if iters and len(iters) > 0:
            x_data_div = self._get_plot_x_data('div') if hasattr(self, '_get_plot_x_data') else self.iter_history
            x_data_ebs = self._get_plot_x_data('ebs') if hasattr(self, '_get_plot_x_data') else self.iter_history
            marker_style = 'o' if len(iters) == 1 else 'None'
            if hasattr(self, 'line_div'):
                self.line_div.set_marker(marker_style)
                self.line_div.set_data(x_data_div, self.div_history)
            if hasattr(self, 'line_div_mid'):
                self.line_div_mid.set_marker(marker_style)
                self.line_div_mid.set_data(x_data_div, self.div_mid_history)
            if hasattr(self, 'line_ebs'):
                self.line_ebs.set_marker(marker_style)
                self.line_ebs.set_data(x_data_ebs, self.ebs_history)

            cur_iter = iters[-1]
            if hasattr(self, '_update_div_scroll_range'):
                self._update_div_scroll_range(cur_iter)
            elif hasattr(self, 'ax_div'):
                self.ax_div.set_xlim(max(0, cur_iter - 400), max(100, cur_iter))

            if hasattr(self, '_update_div_plot_visibility'):
                self._update_div_plot_visibility()
            elif hasattr(self, 'ax_div'):
                valid_divs = [d for d in self.div_history if np.isfinite(d)]
                if valid_divs:
                    self.ax_div.set_ylim(0, max(45, max(valid_divs) * 1.1))

            if hasattr(self, '_update_ebs_scroll_range'):
                self._update_ebs_scroll_range(cur_iter)
            elif hasattr(self, '_update_ebs_xlim'):
                self._update_ebs_xlim(cur_iter)
            elif hasattr(self, 'ax_ebs'):
                self.ax_ebs.set_xlim(max(0, cur_iter - 400), max(100, cur_iter))

            if hasattr(self, '_update_ebs_ylim'):
                self._update_ebs_ylim()
            elif hasattr(self, 'ax_ebs'):
                valid_ebs = [v for v in self.ebs_history if v is not None and np.isfinite(v)]
                if valid_ebs:
                    visible = valid_ebs[-400:] if len(valid_ebs) > 400 else valid_ebs
                    min_v = float(min(visible))
                    max_v = float(max(visible))
                    if max_v <= min_v:
                        max_v = min_v + 1.0
                    pad = max(15.0, abs(max_v - min_v) * 0.15)
                    self.ax_ebs.set_ylim(min_v - pad, max_v + pad)
        else:
            div_val = info.get("div", 0.0)
            if hasattr(self, 'line_div'):
                self.line_div.set_data([], [])
            if hasattr(self, 'line_div_mid'):
                self.line_div_mid.set_data([], [])
            if hasattr(self, 'line_ebs'):
                self.line_ebs.set_data([], [])
            if hasattr(self, 'ax_div'):
                self.ax_div.set_xlim(0, 100)
                self.ax_div.set_ylim(0, max(45, div_val * 1.1) if div_val > 0 else 45)
            if hasattr(self, 'ax_ebs'):
                self.ax_ebs.set_xlim(0, 100)
                self.ax_ebs.set_ylim(-100, 100)

        if d_data and not getattr(self, 'sim_isRunning', False):
            self._sync_sim_domain_from_scenario_data(d_data, include_fields=True)
            if info.get("Lx") is not None:
                self.sim.Lx = float(info["Lx"])
            if info.get("Ly") is not None:
                self.sim.Ly = float(info["Ly"])
            if info.get("nx") is not None:
                self.sim.nx = int(info["nx"])
            if info.get("ny") is not None:
                self.sim.ny = int(info["ny"])
            if info.get("V") is not None:
                self.sim.V = info["V"]
            if info.get("Tmap") is not None:
                self.sim.Tmap = info["Tmap"]

        if hasattr(self, '_update_div_plane_line'):
            self._update_div_plane_line()

        # 2.2 Update Erosion Groove Profiles (ax_grooves, line_grooves)
        if hasattr(self, 'line_grooves') and self.line_grooves and hasattr(self, 'sim') and self.sim is not None:
            if info.get("damage_map") is not None:
                self.sim.damage_map = info["damage_map"].copy()
            elif d_data and d_data.get("damage_map") is not None:
                self.sim.damage_map = d_data["damage_map"].copy()
            elif hasattr(self.sim, 'ny') and hasattr(self.sim, 'nx') and self.sim.ny > 0 and self.sim.nx > 0:
                self.sim.damage_map = np.zeros((self.sim.ny, self.sim.nx), dtype=np.float32)

            if info.get("eroded_depth") is not None:
                self.sim.eroded_depth = info["eroded_depth"].copy()
            elif hasattr(self.sim, 'damage_map') and self.sim.damage_map is not None:
                self.sim.eroded_depth = np.zeros_like(self.sim.damage_map, dtype=np.float32)

            groove_face = "downstream"
            thresh_val = self.inputs["Thresh"].value() if (hasattr(self, 'inputs') and "Thresh" in self.inputs) else 0.05
            n_grooves = len(self.line_grooves)
            title_fs = 8.5 if n_grooves > 2 else 10
            mask_grids_g = info.get("mask_grids") or getattr(self.sim, 'mask_grids', [])
            for groove_idx in range(len(mask_grids_g)):
                if groove_idx >= n_grooves:
                    break
                try:
                    y_mm, depth_um = self.sim.get_groove_profile(
                        groove_idx,
                        thresh=thresh_val,
                        face=groove_face
                    )
                    if hasattr(y_mm, 'size') and y_mm.size > 0:
                        self.line_grooves[groove_idx].set_data(y_mm, depth_um)
                        if hasattr(self, 'ax_grooves') and groove_idx < len(self.ax_grooves):
                            ax_g = self.ax_grooves[groove_idx]
                            ax_g.set_xlim(0, float(y_mm.max()))
                            dmax = float(depth_um.max()) if depth_um.size > 0 else 0.0
                            ax_g.set_ylim(max(dmax * 1.1, 1.0), 0.0)
                            ax_g.set_title(
                                f"Grid {groove_idx + 1} Erosion Profile — {groove_face} face",
                                fontsize=title_fs,
                                pad=3,
                            )
                except Exception:
                    pass

        # 2.5 Update Grid Temperature Map (ax_temp) and Grid Temps label (lbl_temp)
        Tmap = info.get("Tmap")
        if Tmap is None and d_data:
            Tmap = d_data.get("Tmap")
        mask_grids = info.get("mask_grids")
        if not mask_grids and d_data:
            mask_grids = d_data.get("mask_grids", [])
        is_bound_map = info.get("isBound")
        if is_bound_map is None and d_data:
            is_bound_map = d_data.get("isBound")
        X_map = info.get("X") if info.get("X") is not None else (d_data.get("X") if d_data else None)
        Y_map = info.get("Y") if info.get("Y") is not None else (d_data.get("Y") if d_data else None)

        if (
            hasattr(self, 'ax_temp')
            and Tmap is not None
            and X_map is not None
            and Y_map is not None
            and getattr(Tmap, 'shape', None) == getattr(X_map, 'shape', None)
        ):
            grid_mask = np.zeros(Tmap.shape, dtype=bool)
            if mask_grids:
                for mg in mask_grids:
                    if getattr(mg, 'shape', None) == grid_mask.shape:
                        grid_mask |= mg
            elif is_bound_map is not None and getattr(is_bound_map, 'shape', None) == grid_mask.shape:
                grid_mask |= is_bound_map
                grid_mask[:, 0] = False

            if np.any(grid_mask):
                import math
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                T_display_C = np.where(grid_mask, Tmap - 273.15, np.nan)

                if (
                    getattr(self, "tempmesh", None) is None
                    or self.tempmesh.get_array().size != T_display_C.size
                    or getattr(self, "_current_rendered_temp_scenario_idx", None) != scenario_idx
                ):
                    self.ax_temp.clear()
                    self.ax_temp.set_title("Grid Temp Map (°C)", fontsize=9, pad=3)
                    self.ax_temp.set_ylabel("r [mm]", fontsize=8, labelpad=2)
                    self.ax_temp.tick_params(axis="y", labelsize=7, pad=2)
                    self.ax_temp.set_facecolor("black")

                    self.tempmesh = self.ax_temp.pcolormesh(
                        X_map, Y_map, T_display_C,
                        cmap="inferno", shading="nearest"
                    )

                    if hasattr(self, 'cax_temp') and self.cax_temp in self.fig.axes:
                        self.cax_temp.set_axes_locator(None)
                        self.fig.delaxes(self.cax_temp)

                    divider_t = make_axes_locatable(self.ax_temp)
                    self.cax_temp = divider_t.append_axes("right", size="5%", pad=0.1)

                    self.cbar_temp = self.fig.colorbar(self.tempmesh, cax=self.cax_temp)
                    self.cbar_temp.set_label("Temperature (°C)")
                    self.cbar_temp.formatter.set_useOffset(False)
                    self.cbar_temp.formatter.set_scientific(False)
                    self.cbar_temp.update_ticks()
                    self._current_rendered_temp_scenario_idx = scenario_idx
                    self._bg_live = None
                else:
                    self.tempmesh.set_array(T_display_C.ravel())

                finite_vals = T_display_C[np.isfinite(T_display_C)]
                if finite_vals.size > 0:
                    vmin = float(np.min(finite_vals))
                    vmax = float(np.max(finite_vals))
                    vmin_plot = float(math.floor(vmin))
                    vmax_plot = max(vmin_plot + 5.0, float(math.ceil(vmax)))
                    self.tempmesh.set_clim(vmin_plot, vmax_plot)
                    if hasattr(self, "cbar_temp") and self.cbar_temp is not None:
                        self.cbar_temp.formatter.set_useOffset(False)
                        self.cbar_temp.formatter.set_scientific(False)
                        self.cbar_temp.update_ticks()

                if lx is not None and ly is not None:
                    self.ax_temp.set_xlim(0, float(lx))
                    self.ax_temp.set_ylim(0, float(ly))

        t_grids_scen = info.get("T_grids") or info.get("T_grids_K")
        if not t_grids_scen and hasattr(self, 'T_histories') and self.T_histories:
            t_grids_scen = [
                float(self.T_histories[k][-1]) if self.T_histories.get(k) else 300.0
                for k in sorted(self.T_histories.keys())
            ]
        if not t_grids_scen and d_data:
            t_grids_scen = d_data.get("T_grids", [])
        if t_grids_scen and hasattr(self, 'lbl_temp'):
            t_str = " | ".join([f"G{i+1}: {float(T) - 273.15:.1f}°C" for i, T in enumerate(t_grids_scen)])
            self.lbl_temp.setText("Grid Temps: " + t_str)

        # 3. Update status and metrics labels
        t_us = info.get("t_sim_us", 0.0)
        scenario_iter = info.get("iter", info.get("steady_iterations", 0))
        div = info.get("div", info.get("divergence_deg", 0.0))
        div_mid = info.get("div_mid", info.get("divergence_2nd_plane_deg", 0.0))
        i_exit = info.get("exit_ion_current_A", 0.0)
        trans = info.get("transparency", 0.0)
        status_tag = info.get("status", "unknown").upper()
        p_eff = info.get("perveance_effective", info.get("Peffective", 0.0))

        comp_s = info.get("computation_time_s")
        comp_str = f" | Wall Time: {format_comp_duration(comp_s)}" if (comp_s is not None and np.isfinite(comp_s) and comp_s > 0) else ""
        comp_diag_str = f"Computation Time: {format_comp_duration(comp_s)}\n" if (comp_s is not None and np.isfinite(comp_s) and comp_s > 0) else ""

        if hasattr(self, 'lblTime'):
            self.lblTime.setText(f"t_sim: {t_us:.2f} µs | Scenario {scenario_idx} [{status_tag}]")

        if hasattr(self, 'lblTransparency'):
            self.lblTransparency.setText(
                f"Transparency: {trans:.3f}\n"
                f"Divergence Exit: {div:.2f}°\n"
                f"Divergence Plane 2: {div_mid:.2f}°\n"
                f"Exit I_ion: {i_exit * 1e3:.3f} mA\n"
                f"Peffective: {p_eff:.3e}\n"
                f"{comp_diag_str}"
                f"Iteration: {scenario_iter:,}"
            )

        if hasattr(self, 'lbl_top_sim_time'):
            self.lbl_top_sim_time.setText(f"Plasma: {t_us:.2f} µs | Iter: {scenario_iter:,}{comp_str}")

        if hasattr(self, 'lblPerfPtcls'):
            num_pts = (len(prim_xy) if prim_xy is not None else 0) + (len(cex_xy) if cex_xy is not None else 0)
            num_elec = len(elec_xy) if elec_xy is not None else 0
            self.lblPerfPtcls.setText(f"Display Ptcls (i/e): {num_pts:,} / {num_elec:,} | State: {status_tag}")

        # Draw particles using blitting for animated scatter artists, with periodic full redraws for ax_temp / telemetry
        if hasattr(self, 'canvas') and hasattr(self, 'ax_live'):
            now_t = time.perf_counter()
            if not hasattr(self, '_last_full_redraw_t'):
                self._last_full_redraw_t = 0.0
            full_redraw_due = (
                getattr(self, '_bg_live', None) is None
                or (now_t - self._last_full_redraw_t >= 0.33)
            )
            if full_redraw_due:
                # Keep scatter artists animated so canvas.draw() renders a clean domain background without particles
                for a in (getattr(self, 'scat_prim', None), getattr(self, 'scat_cex', None), getattr(self, 'scat_elec', None)):
                    if a is not None:
                        a.set_animated(True)
                self.canvas.draw()
                if getattr(self, '_bg_live', None) is None:
                    try:
                        self._bg_live = self.canvas.copy_from_bbox(self.ax_live.bbox)
                    except Exception:
                        self._bg_live = None
                self._last_full_redraw_t = now_t

            bg = getattr(self, '_bg_live', None)
            if bg is not None:
                try:
                    self.canvas.restore_region(bg)
                    if hasattr(self, 'scat_prim') and self.scat_prim is not None:
                        self.ax_live.draw_artist(self.scat_prim)
                    if hasattr(self, 'scat_cex') and self.scat_cex is not None:
                        self.ax_live.draw_artist(self.scat_cex)
                    if hasattr(self, 'scat_elec') and self.scat_elec is not None:
                        self.ax_live.draw_artist(self.scat_elec)
                    self.canvas.blit(self.ax_live.bbox)
                except Exception:
                    self._bg_live = None
                    self.canvas.draw_idle()
            else:
                self.canvas.draw_idle()
            if not getattr(self, 'sim_isRunning', False):
                self.canvas.draw_idle()
        elif hasattr(self, 'canvas'):
            self.canvas.draw_idle()

        # 4. Update PPC Window for the selected scenario if open
        if getattr(self, 'ppc_window', None) and self.ppc_window.isVisible():
            try:
                self._update_ppc_window_for_scenario(scenario_idx, info)
            except Exception as exc:
                print(f"[PPC Snapshot Update Error] {exc}")

        # 5. Update Physical Constraints (Total Energy) Window for the selected scenario if open
        if getattr(self, 'phys_window', None) and self.phys_window.isVisible():
            try:
                self._update_phys_window_for_scenario(scenario_idx, info)
            except Exception as exc:
                print(f"[Phys Snapshot Update Error] {exc}")

        # 6. Update Computational Domain & Mesh Inspector Window for the selected scenario if open
        if (getattr(self, 'mesh_window', None) and self.mesh_window.isVisible()) or (getattr(self, 'substep_window', None) and self.substep_window.isVisible()):
            try:
                self._update_mesh_window_for_scenario(scenario_idx, info, d_data)
            except Exception as exc:
                print(f"[Mesh Snapshot Update Error] {exc}")

        # 7. Update Performance Monitor Window for the selected scenario if open
        if getattr(self, 'perf_window', None) and self.perf_window.isVisible():
            try:
                self._update_perf_window_for_scenario(scenario_idx, info)
            except Exception as exc:
                print(f"[Perf Snapshot Update Error] {exc}")

        # 8. Update Density Window for the selected scenario if open
        if getattr(self, 'density_window', None) and self.density_window.isVisible():
            try:
                self._update_density_window_for_scenario(scenario_idx, info, d_data)
            except Exception as exc:
                print(f"[Density Snapshot Update Error] {exc}")

        # 9. Update Total Charge Monitor Window for the selected scenario if open
        if getattr(self, 'charge_window', None) and self.charge_window.isVisible():
            try:
                self._update_charge_window_for_scenario(scenario_idx, info)
            except Exception as exc:
                print(f"[Charge Snapshot Update Error] {exc}")

        # 10. Update Energy Distribution (IEDF & EEDF) Window for the selected scenario if open
        if getattr(self, 'iedf_window', None) and self.iedf_window.isVisible():
            try:
                self._update_iedf_window_for_scenario(scenario_idx, info, d_data)
            except Exception as exc:
                print(f"[IEDF Snapshot Update Error] {exc}")

        # 11. Update Electric Field & Potential Diagnostics Window for the selected scenario if open
        if getattr(self, 'field_window', None) and self.field_window.isVisible():
            try:
                self._update_field_window_for_scenario(scenario_idx, info, d_data)
            except Exception as exc:
                print(f"[Field Snapshot Update Error] {exc}")

        # 12. Update Steady State Diagnostics Window for the selected scenario if open
        if getattr(self, 'steady_state_window', None) and self.steady_state_window.isVisible():
            try:
                self._update_steady_state_window_for_scenario(scenario_idx, info)
            except Exception as exc:
                print(f"[Steady State Snapshot Update Error] {exc}")

        QApplication.processEvents()

    def _update_ppc_window_for_scenario(self, scenario_idx, info):
        """Update PPC Window with instantaneous / accumulated PPC maps from scenario snapshot."""
        if not getattr(self, 'ppc_window', None) or not self.ppc_window.isVisible():
            return
        if not hasattr(self, 'sim') or self.sim is None:
            return

        try:
            # Sync domain geometry and mesh grids from cached scenario data to ensure dimensions match
            d_data = self._get_scenario_domain_data(scenario_idx)
            self._sync_sim_domain_from_scenario_data(d_data)

            cur_ny = getattr(self.sim, 'ny', 1) or 1
            cur_nx = getattr(self.sim, 'nx', 1) or 1

            ppc_map = info.get("current_ppc_map")
            accum_map = info.get("accum_ppc_map")
            ppc_steps = info.get("ppc_steps_count")
            scenario_iter = info.get("iter", 0)

            # If ppc_map is provided from worker/snapshot, its shape is the true scenario shape
            if ppc_map is not None and hasattr(ppc_map, 'shape') and len(ppc_map.shape) == 2:
                cur_ny, cur_nx = ppc_map.shape
                self.sim.ny = cur_ny
                self.sim.nx = cur_nx
                self.sim.current_ppc_map = ppc_map
            else:
                self.sim.current_ppc_map = np.zeros((cur_ny, cur_nx), dtype=int)

            if accum_map is not None and getattr(accum_map, 'shape', None) == (cur_ny, cur_nx):
                self.sim.accum_ppc_map = accum_map
            else:
                self.sim.accum_ppc_map = self.sim.current_ppc_map.astype(float)

            # Guarantee that X and Y match (cur_ny, cur_nx)
            if (
                not hasattr(self.sim, 'X')
                or self.sim.X is None
                or not hasattr(self.sim, 'Y')
                or self.sim.Y is None
                or self.sim.X.shape != (cur_ny, cur_nx)
            ):
                x_c = np.linspace(0, self.sim.Lx, cur_nx)
                y_c = np.linspace(0, self.sim.Ly, cur_ny)
                self.sim.X, self.sim.Y = np.meshgrid(x_c, y_c)

            self.sim.ppc_steps_count = int(ppc_steps) if ppc_steps is not None else 0
            self.sim.iteration = int(scenario_iter) if scenario_iter is not None else 0

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""
            self.ppc_window.setWindowTitle(f"Macroparticle Distribution per Cell (PPC) — Scenario {scenario_idx}{param_str}")
            self.ppc_window.update_plot(self.sim)
        except Exception as exc:
            print(f"[PPC Window Warning] Error updating PPC plot for scenario {scenario_idx}: {exc}")

    def _update_phys_window_for_scenario(self, scenario_idx, info):
        """Update Total Energy Conservation Monitor with energy history for scenario_idx."""
        if not getattr(self, 'phys_window', None) or not self.phys_window.isVisible():
            return
        if not hasattr(self, 'sim') or self.sim is None:
            return

        try:
            t_hist = info.get("energy_history_t")
            ke_i_hist = info.get("energy_history_ke_i")
            ke_e_hist = info.get("energy_history_ke_e")
            fe_hist = info.get("energy_history_fe")
            tot_hist = info.get("energy_history_tot")

            if tot_hist is not None and len(tot_hist) > 0:
                self.sim.energy_history_t = list(t_hist) if t_hist is not None else []
                self.sim.energy_history_ke_i = list(ke_i_hist) if ke_i_hist is not None else []
                self.sim.energy_history_ke_e = list(ke_e_hist) if ke_e_hist is not None else []
                self.sim.energy_history_fe = list(fe_hist) if fe_hist is not None else []
                self.sim.energy_history_tot = list(tot_hist)
            elif not getattr(self, 'is_sweep_active', False) or getattr(self, 'sweep_current_idx', 0) + 1 != scenario_idx:
                e_tot = info.get("total_energy_J")
                fe = info.get("field_energy_J", 0.0)
                t_us = info.get("t_sim_us", 0.0)
                if e_tot is not None and e_tot != 0.0:
                    t_s = float(t_us) * 1e-6
                    self.sim.energy_history_t = [t_s]
                    self.sim.energy_history_ke_i = [0.0]
                    self.sim.energy_history_ke_e = [0.0]
                    self.sim.energy_history_fe = [float(fe)]
                    self.sim.energy_history_tot = [float(e_tot)]
                else:
                    self.sim.energy_history_t = []
                    self.sim.energy_history_ke_i = []
                    self.sim.energy_history_ke_e = []
                    self.sim.energy_history_fe = []
                    self.sim.energy_history_tot = []

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""
            self.phys_window.setWindowTitle(f"Total Energy Conservation Monitor — Scenario {scenario_idx}{param_str}")
            self.phys_window.update_plot(self.sim)
        except Exception as exc:
            print(f"[Phys Window Warning] Error updating energy plot for scenario {scenario_idx}: {exc}")

    def _update_mesh_window_for_scenario(self, scenario_idx, info, d_data=None):
        """Update Computational Domain & Mesh Grid Inspector and Substeps graphs for scenario_idx."""
        has_mesh = getattr(self, 'mesh_window', None) and self.mesh_window.isVisible()
        has_substep = getattr(self, 'substep_window', None) and self.substep_window.isVisible()
        if not has_mesh and not has_substep:
            return
        if not hasattr(self, 'sim') or self.sim is None:
            return

        try:
            if d_data is None:
                d_data = self._get_scenario_domain_data(scenario_idx)

            self._sync_sim_domain_from_scenario_data(d_data)

            # Substep history
            subs_it = info.get("substep_history_iter")
            subs_i = info.get("substep_history_ion")
            subs_e = info.get("substep_history_e")
            if subs_it is not None and len(subs_it) > 0:
                self.sim.substep_history_iter = list(subs_it)
                self.sim.substep_history_ion = list(subs_i) if subs_i is not None else []
                self.sim.substep_history_e = list(subs_e) if subs_e is not None else []
            elif not getattr(self, 'is_sweep_active', False) or getattr(self, 'sweep_current_idx', 0) + 1 != scenario_idx:
                self.sim.substep_history_iter = []
                self.sim.substep_history_ion = []
                self.sim.substep_history_e = []

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""

            scen_params = d_data.get("params", self.get_params() if hasattr(self, 'get_params') else None) if d_data else None

            if has_mesh:
                self.mesh_window.setWindowTitle(f"Computational Domain & Mesh Grid Inspector — Scenario {scenario_idx}{param_str}")
                self.mesh_window.update_plot(self.sim, params=scen_params)

            if has_substep:
                self.substep_window.setWindowTitle(f"Displacement Substeps Monitor — Scenario {scenario_idx}{param_str}")
                self.substep_window.update_plot(self.sim)

        except Exception as exc:
            print(f"[Mesh Window Warning] Error updating mesh plot for scenario {scenario_idx}: {exc}")

    def _update_perf_window_for_scenario(self, scenario_idx, info):
        """Update Performance & Profiling Monitor for scenario_idx."""
        if not getattr(self, 'perf_window', None) or not self.perf_window.isVisible():
            return
        if not hasattr(self, 'sim') or self.sim is None:
            return

        try:
            from diagnostics.performance_monitor import PerformanceMonitor
            perf_hist = info.get("perf_history")
            if perf_hist is not None and len(perf_hist) > 0:
                if getattr(self.sim, '_perf_monitor', None) is None:
                    self.sim._perf_monitor = PerformanceMonitor()
                self.sim._perf_monitor.history = list(perf_hist)
            elif not getattr(self, 'is_sweep_active', False) or getattr(self, 'sweep_current_idx', 0) + 1 != scenario_idx:
                if getattr(self.sim, '_perf_monitor', None) is not None:
                    self.sim._perf_monitor.history = []

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""
            self.perf_window.setWindowTitle(f"Runtime Performance & Particle Diagnostics Monitor — Scenario {scenario_idx}{param_str}")
            self.perf_window.update_plot(self.sim)
        except Exception as exc:
            print(f"[Perf Window Warning] Error updating performance plot for scenario {scenario_idx}: {exc}")

    def _update_density_window_for_scenario(self, scenario_idx, info, d_data=None):
        """Update Plasma & Particle Density Diagnostics window for scenario_idx."""
        if not getattr(self, 'density_window', None) or not self.density_window.isVisible():
            return
        if not hasattr(self, 'sim') or self.sim is None:
            return

        try:
            if d_data is None:
                d_data = self._get_scenario_domain_data(scenario_idx)

            self._sync_sim_domain_from_scenario_data(d_data)

            cur_ny = getattr(self.sim, 'ny', 1) or 1
            cur_nx = getattr(self.sim, 'nx', 1) or 1

            # Sync particles from scenario snapshot
            prim_xy = info.get("prim_xy")
            cex_xy = info.get("cex_xy")
            elec_xy = info.get("elec_xy")

            n_prim = len(prim_xy) if prim_xy is not None else 0
            n_cex = len(cex_xy) if cex_xy is not None else 0
            n_tot = n_prim + n_cex

            if n_tot > 0:
                all_px = np.empty(n_tot, dtype=np.float32)
                all_py = np.empty(n_tot, dtype=np.float32)
                all_cex = np.zeros(n_tot, dtype=bool)
                if n_prim > 0:
                    all_px[:n_prim] = prim_xy[:, 0]
                    all_py[:n_prim] = prim_xy[:, 1]
                if n_cex > 0:
                    all_px[n_prim:] = cex_xy[:, 0]
                    all_py[n_prim:] = cex_xy[:, 1]
                    all_cex[n_prim:] = True
                self.sim.p_x = all_px
                self.sim.p_y = all_py
                self.sim.p_isCEX = all_cex
                self.sim.num_p = n_tot
            else:
                self.sim.num_p = 0
                self.sim.p_x = np.empty(0, dtype=np.float32)
                self.sim.p_y = np.empty(0, dtype=np.float32)
                self.sim.p_isCEX = np.empty(0, dtype=bool)

            n_elec = len(elec_xy) if elec_xy is not None else 0
            if n_elec > 0:
                self.sim.e_x = elec_xy[:, 0]
                self.sim.e_y = elec_xy[:, 1]
                self.sim.num_e = n_elec
            else:
                self.sim.num_e = 0
                self.sim.e_x = np.empty(0, dtype=np.float32)
                self.sim.e_y = np.empty(0, dtype=np.float32)

            self.sim.macro_weight = float(info.get("macro_weight", getattr(self.sim, 'macro_weight', 1e8)))
            self.sim.iteration = int(info.get("iter", 0))

            # Sync PPC spatial maps and plasma physics parameters
            self.sim.current_ppc_map = info.get("current_ppc_map")
            self.sim.accum_ppc_map = info.get("accum_ppc_map")
            self.sim.current_cex_ppc_map = info.get("current_cex_ppc_map")
            self.sim.accum_cex_ppc_map = info.get("accum_cex_ppc_map")
            self.sim.ppc_steps_count = int(info.get("ppc_steps_count", 0))
            if info.get("cell_vol_2d") is not None:
                self.sim.cell_vol_2d = info["cell_vol_2d"]
            if info.get("n0_plasma") is not None:
                self.sim.n0_plasma = float(info["n0_plasma"])
            if info.get("Te_up") is not None:
                self.sim.Te_up = float(info["Te_up"])
            if info.get("V") is not None:
                self.sim.V = info["V"]

            # Guarantee that X and Y match (cur_ny, cur_nx)
            if (
                not hasattr(self.sim, 'X')
                or self.sim.X is None
                or not hasattr(self.sim, 'Y')
                or self.sim.Y is None
                or self.sim.X.shape != (cur_ny, cur_nx)
            ):
                x_c = np.linspace(0, self.sim.Lx, cur_nx)
                y_c = np.linspace(0, self.sim.Ly, cur_ny)
                self.sim.X, self.sim.Y = np.meshgrid(x_c, y_c)

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""
            self.density_window.setWindowTitle(f"Plasma & Particle Density Diagnostics — Scenario {scenario_idx}{param_str}")
            self.density_window.update_plot(self.sim)
        except Exception as exc:
            print(f"[Density Window Warning] Error updating density plot for scenario {scenario_idx}: {exc}")

    def _update_charge_window_for_scenario(self, scenario_idx, info):
        """Update Total Charge Monitor Window with historical charge data of selected scenario."""
        if not getattr(self, 'charge_window', None) or not self.charge_window.isVisible():
            return
        try:
            # Sync charge history attributes to self.sim
            self.sim.charge_history_t = list(info.get("charge_history_t", []))
            self.sim.charge_history_q_i = list(info.get("charge_history_q_i", []))
            self.sim.charge_history_q_e = list(info.get("charge_history_q_e", []))
            self.sim.charge_history_q_b = list(info.get("charge_history_q_b", []))
            self.sim.charge_history_q_free = list(info.get("charge_history_q_free", []))
            self.sim.charge_history_q_net = list(info.get("charge_history_q_net", []))

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""
            self.charge_window.setWindowTitle(f"Total Charge Monitor — Scenario {scenario_idx}{param_str}")
            self.charge_window.update_plot(self.sim)
        except Exception as exc:
            print(f"[Charge Window Warning] Error updating charge monitor for scenario {scenario_idx}: {exc}")

    def _update_iedf_window_for_scenario(self, scenario_idx, info, d_data=None):
        """Update Energy Distribution (IEDF & EEDF) Window with particle data of selected scenario."""
        if not getattr(self, 'iedf_window', None) or not self.iedf_window.isVisible():
            return
        try:
            m_XE = getattr(self.sim, 'm_ion', 131.293 * 1.6605e-27)
            m_e  = getattr(self.sim, 'm_e', 9.1093837e-31)
            q    = getattr(self.sim, 'q_ion', getattr(self.sim, 'q', 1.602e-19))

            grids = info.get("grids", []) or (getattr(self.sim, 'grids', []) if hasattr(self, 'sim') else [])
            if grids:
                max_v = max([float(g.get("V", 0.0)) for g in grids])
            elif getattr(self, 'grid_widgets', None):
                max_v = max([g["V"].value() for g in self.grid_widgets])
            else:
                max_v = 1000.0
            max_v = max(max_v, 100.0)

            # Check if direct velocity arrays are available
            if d_data and 'p_vx' in d_data and len(d_data['p_vx']) > 0:
                p_vx = d_data['p_vx']
                p_vy = d_data.get('p_vy', np.zeros_like(p_vx))
                p_isCEX = d_data.get('p_isCEX', np.zeros(len(p_vx), dtype=bool))
            else:
                e_prim = d_data.get('e_prim_eV') if d_data and 'e_prim_eV' in d_data else info.get('e_prim_eV', np.empty(0))
                e_cex  = d_data.get('e_cex_eV') if d_data and 'e_cex_eV' in d_data else info.get('e_cex_eV', np.empty(0))
                e_prim = np.asarray(e_prim, dtype=np.float32) if e_prim is not None else np.empty(0, dtype=np.float32)
                e_cex  = np.asarray(e_cex, dtype=np.float32) if e_cex is not None else np.empty(0, dtype=np.float32)
                n_prim = len(e_prim)
                n_cex = len(e_cex)
                if n_prim + n_cex > 0:
                    p_isCEX = np.concatenate([np.zeros(n_prim, dtype=bool), np.ones(n_cex, dtype=bool)])
                    E_ions = np.concatenate([e_prim, e_cex])
                    p_vx = np.sqrt(2.0 * q * np.maximum(E_ions, 0.0) / m_XE)
                    p_vy = np.zeros_like(p_vx)
                else:
                    p_isCEX = np.empty(0, dtype=bool)
                    p_vx = np.empty(0, dtype=np.float32)
                    p_vy = np.empty(0, dtype=np.float32)

            if d_data and 'e_vx' in d_data and len(d_data['e_vx']) > 0:
                e_x = d_data.get('e_x', np.empty(0, dtype=np.float32))
                e_vx = d_data['e_vx']
                e_vy = d_data.get('e_vy', np.zeros_like(e_vx))
            else:
                elec_xy = d_data.get('elec_xy') if d_data and 'elec_xy' in d_data else info.get('elec_xy', np.empty((0, 2)))
                e_elec  = d_data.get('e_elec_eV') if d_data and 'e_elec_eV' in d_data else info.get('e_elec_eV', np.empty(0))
                elec_xy = np.asarray(elec_xy) if elec_xy is not None else np.empty((0, 2))
                e_elec  = np.asarray(e_elec, dtype=np.float32) if e_elec is not None else np.empty(0, dtype=np.float32)

                if len(elec_xy) > 0 and elec_xy.ndim == 2 and elec_xy.shape[1] >= 1:
                    e_x = elec_xy[:, 0]
                else:
                    e_x = np.empty(0, dtype=np.float32)

                n_e = min(len(e_x), len(e_elec)) if len(e_elec) > 0 else len(e_x)
                if n_e > 0:
                    e_x = e_x[:n_e]
                    if len(e_elec) >= n_e:
                        e_vx = np.sqrt(2.0 * q * np.maximum(e_elec[:n_e], 0.0) / m_e)
                    else:
                        e_vx = np.zeros(n_e, dtype=np.float32)
                    e_vy = np.zeros_like(e_vx)
                else:
                    e_x = np.empty(0, dtype=np.float32)
                    e_vx = np.empty(0, dtype=np.float32)
                    e_vy = np.empty(0, dtype=np.float32)

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""
            self.iedf_window.setWindowTitle(f"Energy Distribution Function (IEDF & EEDF) — Scenario {scenario_idx}{param_str}")
            self.iedf_window.update_histogram(p_vx, p_vy, p_isCEX, e_x, e_vx, e_vy, m_XE, m_e, q, max_v)
        except Exception as exc:
            print(f"[IEDF Window Warning] Error updating energy distribution for scenario {scenario_idx}: {exc}")

    def _update_field_window_for_scenario(self, scenario_idx, info, d_data=None):
        """Update Electric Field & Potential Diagnostics Window for the selected scenario."""
        if not getattr(self, 'field_window', None) or not self.field_window.isVisible():
            return
        try:
            self._sync_sim_domain_from_scenario_data(d_data, include_fields=True)

            # If dynamic potential is present in info, use it; otherwise fallback to domain cache V
            if info.get("V") is not None:
                self.sim.V = info["V"]
            elif d_data and d_data.get("V") is not None:
                self.sim.V = d_data["V"]

            # Compute self-consistent electric field gradients for the scenario's potential V
            if hasattr(self.sim, 'V') and self.sim.V is not None and getattr(self.sim.V, 'ndim', 0) == 2:
                y_m = (self.sim.Y[:, 0] if hasattr(self.sim, 'Y') and self.sim.Y is not None and self.sim.Y.shape == self.sim.V.shape else np.linspace(0, getattr(self.sim, 'Ly', 3.0), self.sim.V.shape[0])) * 1e-3
                x_m = (self.sim.X[0, :] if hasattr(self.sim, 'X') and self.sim.X is not None and self.sim.X.shape == self.sim.V.shape else np.linspace(0, getattr(self.sim, 'Lx', 3.0), self.sim.V.shape[1])) * 1e-3
                Ey, Ex = np.gradient(-self.sim.V, y_m, x_m)
                self.sim.Ex = Ex
                self.sim.Ey = Ey
            self.sim.iteration = int(info.get("iter", 0))

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""
            self.field_window.setWindowTitle(f"Electric Field & Potential Diagnostics — Scenario {scenario_idx}{param_str}")
            self.field_window.update_plot(self.sim)
            if hasattr(self.field_window, '_replot'):
                self.field_window._replot()
        except Exception as exc:
            print(f"[Field Window Warning] Error updating field diagnostics for scenario {scenario_idx}: {exc}")

    def _update_steady_state_window_for_scenario(self, scenario_idx, info):
        """Update Steady State Diagnostics Window with variation history of selected scenario."""
        if not getattr(self, 'steady_state_window', None) or not self.steady_state_window.isVisible():
            return
        try:
            # Sync histories to self.sim
            self.sim.energy_history_tot = list(info.get("energy_history_tot", []))
            self.sim.charge_history_q_net = list(info.get("charge_history_q_net", []))
            self.sim.energy_history_fe = list(info.get("energy_history_fe", []))
            self.sim.potential_history = list(info.get("ebs_history", []))
            self.sim.iteration = int(info.get("iter", 0))

            # Sync ion current history to self.sim so steady_state_window reads the SELECTED scenario's current!
            # Prefer smoothed cumulative running average (ion_current_exit_avg_history) matching the physics stationarity evaluator
            ion_curr_h = list(
                info.get("ion_current_exit_avg_history")
                or info.get("ion_current_history")
                or info.get("ion_current_exit_history")
                or []
            )
            self.sim.ion_current_history = ion_curr_h
            self.sim.exit_ion_current_history = ion_curr_h

            # Sync histories to app (self)
            self.ebs_history = list(info.get("ebs_history", []))
            self.div_history = list(info.get("div_history", []))
            self.ion_current_exit_history = list(ion_curr_h)
            self.ion_current_exit_avg_history = list(ion_curr_h)
            self.transmitted_ions_history = list(info.get("transmitted_ions_history", []))

            param_id = getattr(self, 'sweep_config', {}).get("param_id", "param")
            param_val = info.get("parameter_value", "")
            param_str = f" ({param_id} = {param_val})" if param_val != "" else ""
            w = self.steady_state_window.spin_window.value() if hasattr(self.steady_state_window, 'spin_window') else 100
            self.steady_state_window.setWindowTitle(f"Steady State Diagnostics ({w}-iter) — Scenario {scenario_idx}{param_str}")
            self.steady_state_window.update_plot(self.sim, self)
        except Exception as exc:
            print(f"[Steady State Window Warning] Error updating steady state monitor for scenario {scenario_idx}: {exc}")

    def _on_parallel_progress_updated(self, completed, total, status_text):
        is_throttled = ("Memory Guard" in status_text) or ("Throttled" in status_text)
        if hasattr(self, 'lbl_sweep_badge'):
            n_cores = getattr(self.parallel_sweep_worker, 'n_cores', '') if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker else ''
            if is_throttled:
                self.lbl_sweep_badge.setText(
                    f"⚡ Parallel Sweep [⚠️ RAM Guard: 1-2 workers | {completed}/{total} Done]"
                )
                self.lbl_sweep_badge.setStyleSheet(
                    "font-weight: bold; color: #b45309; background-color: #fffbeb; "
                    "border: 1px solid #f59e0b; border-radius: 4px; padding: 2px 6px;"
                )
            else:
                core_str = f"{n_cores} cores | " if n_cores else ""
                self.lbl_sweep_badge.setText(
                    f"⚡ Parallel Sweep [{core_str}{completed}/{total} Done]"
                )
                self.lbl_sweep_badge.setStyleSheet(
                    "font-weight: bold; color: #1e3a8a; background-color: #dbeafe; "
                    "border: 1px solid #93c5fd; border-radius: 4px; padding: 2px 6px;"
                )
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(status_text)
            if is_throttled:
                self.lbl_status.setStyleSheet("color: #b45309; font-weight: bold;")
            else:
                self.lbl_status.setStyleSheet("")

    def _on_parallel_scenario_completed(self, result_row, completed, total):
        self.sweep_results.append(result_row)
        if not hasattr(self, '_sweep_status_dict'):
            self._sweep_status_dict = {}
        if not hasattr(self, '_sweep_tracking_buffers'):
            self._sweep_tracking_buffers = {}
        s_idx = result_row.get("scenario_idx")
        if s_idx is not None:
            prev_info = self._sweep_status_dict.get(s_idx, {})
            t_grids_k = result_row.get("T_grids_K", prev_info.get("T_grids_K", prev_info.get("T_grids", [])))
            t_grids_c = result_row.get("T_grids_C", [float(t - 273.15) for t in t_grids_k] if t_grids_k else prev_info.get("T_grids_C", []))
            self._sweep_status_dict[s_idx] = {
                **prev_info,
                "status": "done",
                "iter": result_row.get("steady_iterations", 0),
                "steady_iterations": result_row.get("steady_iterations", 0),
                "parameter_value": result_row.get("parameter_value", 0.0),
                "steady_state_reached": result_row.get("steady_state_reached", True),
                "computation_time_s": result_row.get("computation_time_s", 0.0),
                "div": result_row.get("divergence_deg", result_row.get("div", 0.0)),
                "div_mid": result_row.get("divergence_2nd_plane_deg", result_row.get("div_mid", 0.0)),
                "saddle_potential_V": result_row.get("saddle_potential_V", prev_info.get("saddle_potential_V", 0.0)),
                "exit_ion_current_A": result_row.get("exit_ion_current_A", 0.0),
                "transparency": result_row.get("transparency", 0.0),
                "iter_history": list(result_row.get("iter_history") or prev_info.get("iter_history", [])),
                "time_history": list(result_row.get("time_history") or prev_info.get("time_history", [])),
                "div_history": list(result_row.get("div_history") or prev_info.get("div_history", [])),
                "div_mid_history": list(result_row.get("div_mid_history") or prev_info.get("div_mid_history", [])),
                "ebs_history": list(result_row.get("ebs_history") or prev_info.get("ebs_history", [])),
                "transparency_history": list(result_row.get("transparency_history") or prev_info.get("transparency_history", [])),
                "ion_current_exit_history": list(result_row.get("ion_current_exit_history") or prev_info.get("ion_current_exit_history", [])),
                "ion_current_exit_avg_history": list(result_row.get("ion_current_exit_avg_history") or prev_info.get("ion_current_exit_avg_history", [])),
                "transmitted_ions_history": list(result_row.get("transmitted_ions_history") or prev_info.get("transmitted_ions_history", [])),
                "active_cells_history": list(result_row.get("active_cells_history") or prev_info.get("active_cells_history", [])),
                "low_ppc_cells_history": list(result_row.get("low_ppc_cells_history") or prev_info.get("low_ppc_cells_history", [])),
                "T_histories": dict(result_row.get("T_histories") or prev_info.get("T_histories", {})),
                "T_grids": list(t_grids_k) if t_grids_k else prev_info.get("T_grids", []),
                "T_grids_K": list(t_grids_k) if t_grids_k else [],
                "T_grids_C": list(t_grids_c) if t_grids_c else [],
            }
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            w_status = self.parallel_sweep_worker.get_status_dict()
            for k_scen, v_dict in w_status.items():
                if k_scen not in self._sweep_status_dict:
                    self._sweep_status_dict[k_scen] = v_dict
                else:
                    cur = self._sweep_status_dict[k_scen]
                    for vk, vv in v_dict.items():
                        if "history" in vk and not vv and cur.get(vk):
                            continue
                        cur[vk] = vv
            if s_idx is not None and hasattr(self.parallel_sweep_worker, 'get_scenario_tracking_buffer'):
                tb = self.parallel_sweep_worker.get_scenario_tracking_buffer(s_idx)
                if tb:
                    self._sweep_tracking_buffers[s_idx] = list(tb)
        self._update_sweep_scenario_combo_items()

    def _on_parallel_sweep_finished(self, results):
        self.sweep_results = results
        if hasattr(self, '_sweep_total_timer') and self._sweep_total_timer is not None and self._sweep_total_timer.get_duration() > 0:
            total_time = self._sweep_total_timer.get_duration()
        elif hasattr(self, 'parallel_sweep_worker') and getattr(self.parallel_sweep_worker, 'total_computation_time_s', 0.0) > 0:
            total_time = self.parallel_sweep_worker.total_computation_time_s
        else:
            total_time = get_active_clock() - getattr(self, 'sweep_start_wall_time', get_active_clock())
        if self.sweep_config is not None:
            self.sweep_config["total_computation_time_s"] = float(total_time)
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            self.parallel_sweep_worker.total_computation_time_s = float(total_time)
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            if not hasattr(self, '_sweep_status_dict'):
                self._sweep_status_dict = {}
            self._sweep_status_dict.update(self.parallel_sweep_worker.get_status_dict())
            if not hasattr(self, '_sweep_tracking_buffers'):
                self._sweep_tracking_buffers = {}
            if hasattr(self.parallel_sweep_worker, 'get_all_tracking_buffers'):
                for k_scen, tb_list in self.parallel_sweep_worker.get_all_tracking_buffers().items():
                    if tb_list:
                        self._sweep_tracking_buffers[k_scen] = list(tb_list)
        if hasattr(self, 'sweep_monitor_timer') and self.sweep_monitor_timer is not None:
            self.sweep_monitor_timer.stop()
        if hasattr(self, '_pause_simulation_timer'):
            self._pause_simulation_timer()
        if hasattr(self, 'parallel_sweep_dlg') and self.parallel_sweep_dlg is not None:
            try:
                self.parallel_sweep_dlg.rejected.disconnect(self.stop_sweep)
            except Exception:
                pass
            self.parallel_sweep_dlg.accept()
            self.parallel_sweep_dlg = None
        self.finish_sweep()

    def _on_parallel_sweep_error(self, error_msg):
        self.stop_sweep()
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setText("⚡ Sweep Stopped (Error)")
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #fee2e2; color: #b91c1c; font-weight: bold; "
                "border: 1px solid #fca5a5; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        self._update_sweep_scenario_combo_items()

        num_done = len(getattr(self, 'sweep_results', []))
        total_scen = len(self.sweep_config.get("values", [])) if self.sweep_config else 0
        info_txt = (
            f"An error occurred during multi-core sweep:\n{error_msg}\n\n"
            f"{num_done} of {total_scen} scenario(s) completed successfully and their results are preserved.\n"
            f"You can review completed scenarios using the top dropdown, or click 'Resume Sweep' to continue."
        )
        QMessageBox.critical(self, "Parallel Sweep Error", info_txt)

    def _apply_sweep_value(self, param_id, value):
        """Applies a target parameter value to the GUI inputs and simulator."""
        self._suppress_auto_upstream_gap = True
        try:
            if hasattr(self, 'inputs') and param_id in self.inputs:
                try:
                    self.inputs[param_id].setValue(value)
                except Exception:
                    pass
            elif param_id.startswith("grid_") and hasattr(self, 'grid_widgets'):
                parts = param_id.split("_")
                if len(parts) >= 3:
                    try:
                        grid_idx = int(parts[1]) - 1
                        prop_key = parts[2]
                        if 0 <= grid_idx < len(self.grid_widgets):
                            if prop_key in self.grid_widgets[grid_idx]:
                                self.grid_widgets[grid_idx][prop_key].setValue(value)
                    except Exception:
                        pass
        finally:
            self._suppress_auto_upstream_gap = False

        cfg = getattr(self, 'sweep_config', None)
        if (
            isinstance(cfg, dict)
            and cfg.get("param_id") == param_id
            and hasattr(self, 'inputs')
            and "upstream_gap_mm" in self.inputs
        ):
            base_params = getattr(self, '_sweep_base_params', None)
            if not isinstance(base_params, dict):
                base_params = self.get_params() if hasattr(self, 'get_params') else {}
            if cfg.get("auto_update_upstream_gap", False):
                from physics.headless_sweep import apply_sweep_param_to_dict
                resolved = apply_sweep_param_to_dict(base_params, param_id, value, sweep_config=cfg)
                if "upstream_gap_mm" in resolved:
                    gap_val = float(resolved["upstream_gap_mm"])
                    self.inputs["upstream_gap_mm"].blockSignals(True)
                    self.inputs["upstream_gap_mm"].setValue(gap_val)
                    self.inputs["upstream_gap_mm"].blockSignals(False)
                    self._last_auto_gap = round(gap_val, 3)
            elif cfg.get("auto_update_upstream_gap") is False and "upstream_gap_mm" in base_params:
                gap_val = float(base_params["upstream_gap_mm"])
                self.inputs["upstream_gap_mm"].blockSignals(True)
                self.inputs["upstream_gap_mm"].setValue(gap_val)
                self.inputs["upstream_gap_mm"].blockSignals(False)

    def check_sweep_steady_state(self):
        """
        Called on each GUI render tick when sweep mode is active.
        Monitors whether the current scenario has reached steady state,
        records its physical snapshot, and advances to the next scenario.
        """
        if not getattr(self, 'is_sweep_active', False) or self.sweep_config is None:
            return

        if getattr(self, '_sweep_paused_current', False):
            return

        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            if self.parallel_sweep_worker.isRunning():
                return

        sim = getattr(self, 'sim', None)
        if sim is None:
            return

        cfg = self.sweep_config
        cur_idx = self.sweep_current_idx
        values = cfg["values"]
        total_scenarios = len(values)
        if cur_idx >= total_scenarios:
            self.finish_sweep()
            return

        cur_val = values[cur_idx]
        current_iter = getattr(sim, 'iteration', 0)
        scenario_start = getattr(self, 'sweep_scenario_start_iter', 0)
        scenario_iters = current_iter - scenario_start

        min_iters = cfg.get("min_iterations", 500)
        max_iters = cfg.get("max_iterations", 5000)
        window = cfg.get("window", 100)
        thresh = cfg.get("threshold", 1.0)

        # Track first grid entry iteration relative to scenario start
        entry_iter = getattr(self, 'sweep_scenario_entry_iter', None)
        if entry_iter is None:
            sim_entry = getattr(sim, 'first_grid_entry_iter', None)
            if sim_entry is not None:
                entry_iter = max(0, sim_entry - scenario_start)
                self.sweep_scenario_entry_iter = entry_iter
            elif getattr(sim, 'entered_optics', 0.0) > 0:
                entry_iter = scenario_iters
                self.sweep_scenario_entry_iter = entry_iter

        if entry_iter is not None:
            iters_since_entry = scenario_iters - entry_iter
        else:
            iters_since_entry = 0

        # Update live badge text
        if hasattr(self, 'lbl_sweep_badge'):
            cur_val_str = (
                format_sci_value(cur_val)
                if (cfg.get("param_id") in ("n0_plasma", "n0") or abs(cur_val) >= 1e4 or (0 < abs(cur_val) <= 1e-3))
                else f"{cur_val:.3g}"
            )
            entry_info = f" ({iters_since_entry}/{min_iters} post-entry)" if (entry_iter is not None and iters_since_entry < min_iters) else ""
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep [{cur_idx + 1}/{total_scenarios} | {cur_val_str} {cfg['param_unit']}{entry_info}]"
            )

        # Check steady-state condition across core physical observables
        steady_reached = False
        max_hit = scenario_iters >= max_iters

        if not max_hit:
            # Need at least min_iters evaluated from when particles entered the first grid, plus window steps
            if iters_since_entry < min_iters or scenario_iters < window:
                return

            histories_to_check = []
            if hasattr(sim, 'energy_history_tot') and len(sim.energy_history_tot) >= window:
                histories_to_check.append(sim.energy_history_tot[-scenario_iters:])
            if hasattr(sim, 'charge_history_q_net') and len(sim.charge_history_q_net) >= window:
                histories_to_check.append(sim.charge_history_q_net[-scenario_iters:])
            if hasattr(sim, 'potential_history') and len(sim.potential_history) >= window:
                histories_to_check.append(sim.potential_history[-scenario_iters:])
            elif hasattr(self, 'ebs_history') and len(self.ebs_history) >= window:
                histories_to_check.append(self.ebs_history[-scenario_iters:])
            if hasattr(sim, 'ion_current_history') and len(sim.ion_current_history) >= window:
                histories_to_check.append(sim.ion_current_history[-scenario_iters:])
            elif hasattr(self, 'ion_current_exit_history') and len(self.ion_current_exit_history) >= window:
                histories_to_check.append(self.ion_current_exit_history[-scenario_iters:])

            if histories_to_check:
                all_under_thresh = True
                for h in histories_to_check:
                    _, var = compute_rolling_variation(h, window=window, mode="rel_change")
                    if len(var) == 0 or abs(var[-1]) > thresh:
                        all_under_thresh = False
                        break
                steady_reached = all_under_thresh

        # If steady state reached OR safeguard limit hit, record snapshot and advance
        if steady_reached or max_hit:
            self._record_sweep_scenario_result(
                scenario_idx=cur_idx + 1,
                parameter_value=cur_val,
                steady_reached=steady_reached,
                scenario_iters=scenario_iters,
                max_iters_hit=max_iters if max_hit else None
            )
            self.advance_sweep_scenario()

    def _record_sweep_scenario_result(self, scenario_idx, parameter_value, steady_reached, scenario_iters, max_iters_hit=None, comp_time_s=None):
        """Records final steady-state physical observables for the completed scenario."""
        if comp_time_s is None:
            if hasattr(self, '_sweep_scenario_timer') and self._sweep_scenario_timer is not None:
                comp_time_s = self._sweep_scenario_timer.get_duration()
            else:
                comp_time_s = get_active_clock() - getattr(self, 'sweep_scenario_start_wall_time', get_active_clock())

        sim = self.sim
        t_sim_us = (sim.iteration * sim.dt * 1e6) if hasattr(sim, 'dt') else 0.0

        seq_h = getattr(self, '_seq_scenario_histories', {}).get(scenario_idx, {})
        rec_iters = list(seq_h.get("iter_history") or self.iter_history)
        rec_times = list(seq_h.get("time_history") or getattr(self, 'time_history', []))
        rec_divs = list(seq_h.get("div_history") or self.div_history)
        rec_divs_mid = list(seq_h.get("div_mid_history") or self.div_mid_history)
        rec_ebs = list(seq_h.get("ebs_history") or self.ebs_history)
        rec_trans = list(seq_h.get("transparency_history") or getattr(self, 'transparency_history', []))
        rec_trans3 = list(seq_h.get("transparency3_history") or getattr(self, 'transparency3_history', []))
        rec_i_exit = list(seq_h.get("ion_current_exit_history") or getattr(self, 'ion_current_exit_history', []))
        rec_i_avg = list(seq_h.get("ion_current_exit_avg_history") or getattr(self, 'ion_current_exit_avg_history', []))
        rec_trans_ions = list(seq_h.get("transmitted_ions_history") or getattr(self, 'transmitted_ions_history', []))
        rec_act_cells = list(seq_h.get("active_cells_history") or getattr(self, 'active_cells_history', []))
        rec_low_ppc = list(seq_h.get("low_ppc_cells_history") or getattr(self, 'low_ppc_cells_history', []))
        raw_t_hists = seq_h.get("T_histories") if seq_h.get("T_histories") else getattr(self, 'T_histories', {})
        rec_t_hists = {int(k): list(v) for k, v in raw_t_hists.items()} if isinstance(raw_t_hists, dict) else {}

        if not hasattr(self, '_sweep_tracking_buffers') or not isinstance(self._sweep_tracking_buffers, dict):
            self._sweep_tracking_buffers = {}
        if scenario_idx not in self._sweep_tracking_buffers and len(getattr(self, 'tracking_buffer', [])) > 0:
            self._sweep_tracking_buffers[scenario_idx] = list(self.tracking_buffer)
        rec_trk_buf = list(self._sweep_tracking_buffers.get(scenario_idx, []))

        e_tot = sim.energy_history_tot[-1] if hasattr(sim, 'energy_history_tot') and sim.energy_history_tot else 0.0
        q_net = sim.charge_history_q_net[-1] if hasattr(sim, 'charge_history_q_net') and sim.charge_history_q_net else 0.0
        fe = sim.energy_history_fe[-1] if hasattr(sim, 'energy_history_fe') and sim.energy_history_fe else 0.0
        saddle_v = sim.min_pot if hasattr(sim, 'min_pot') else (rec_ebs[-1] if rec_ebs else 0.0)
        div = 0.0
        if rec_divs:
            finite_div = [d for d in rec_divs if np.isfinite(d)]
            div = finite_div[-1] if finite_div else (0.0 if np.isnan(rec_divs[-1]) else rec_divs[-1])
        elif hasattr(sim, 'current_div') and np.isfinite(sim.current_div):
            div = sim.current_div

        div_plane2 = 0.0
        if rec_divs_mid:
            finite_div2 = [d for d in rec_divs_mid if np.isfinite(d)]
            div_plane2 = finite_div2[-1] if finite_div2 else (0.0 if np.isnan(rec_divs_mid[-1]) else rec_divs_mid[-1])
        elif hasattr(sim, 'current_div_mid') and np.isfinite(sim.current_div_mid):
            div_plane2 = sim.current_div_mid

        i_exit = rec_i_exit[-1] if rec_i_exit else 0.0
        trans = rec_trans[-1] if rec_trans else 0.0

        p_eff, p_max, perv_frac = self._compute_perveance_metrics(i_exit)

        t_grids_final_K = [float(t) for t in getattr(sim, 'T_grids', [])]
        if not t_grids_final_K and rec_t_hists:
            t_grids_final_K = [
                float(rec_t_hists[k][-1]) if rec_t_hists.get(k) else 300.0
                for k in sorted(rec_t_hists.keys())
            ]
        t_grids_final_C = [float(t - 273.15) for t in t_grids_final_K]
        grid_temps_dict = {}
        for g_idx, (tk, tc) in enumerate(zip(t_grids_final_K, t_grids_final_C), start=1):
            grid_temps_dict[f"grid_{g_idx}_temp_K"] = tk
            grid_temps_dict[f"grid_{g_idx}_temp_C"] = tc

        result_row = {
            "scenario_idx": scenario_idx,
            "parameter_value": parameter_value,
            "steady_state_reached": bool(steady_reached),
            "max_iters_hit": max_iters_hit,
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
            "T_grids_K": t_grids_final_K,
            "T_grids_C": t_grids_final_C,
            **grid_temps_dict,
        }
        self.sweep_results.append(result_row)
        if not hasattr(self, '_sweep_status_dict'):
            self._sweep_status_dict = {}
        self._sweep_status_dict[scenario_idx] = {
            "status": "done",
            "iter": int(scenario_iters),
            "max_iters": max_iters_hit,
            "parameter_value": parameter_value,
            "steady_state_reached": bool(steady_reached),
            "computation_time_s": float(comp_time_s),
            "t_sim_us": float(t_sim_us),
            "div": float(div),
            "div_mid": float(div_plane2),
            "exit_ion_current_A": float(i_exit),
            "perveance_effective": float(p_eff),
            "perveance_max": float(p_max),
            "perveance_fraction": float(perv_frac),
            "transparency": float(trans),
            "prim_xy": getattr(self, '_last_prim_xy', np.empty((0, 2))),
            "e_prim_eV": getattr(self, '_last_e_prim_eV', np.empty(0)),
            "cex_xy": getattr(self, '_last_cex_xy', np.empty((0, 2))),
            "e_cex_eV": getattr(self, '_last_e_cex_eV', np.empty(0)),
            "elec_xy": getattr(self, '_last_elec_xy', np.empty((0, 2))),
            "e_elec_eV": getattr(self, '_last_e_elec_eV', np.empty(0)),
            "iter_history": rec_iters,
            "time_history": rec_times,
            "div_history": rec_divs,
            "div_mid_history": rec_divs_mid,
            "ebs_history": rec_ebs,
            "transparency_history": rec_trans,
            "transparency3_history": rec_trans3,
            "ion_current_exit_history": rec_i_exit,
            "ion_current_exit_avg_history": rec_i_avg,
            "transmitted_ions_history": rec_trans_ions,
            "active_cells_history": rec_act_cells,
            "low_ppc_cells_history": rec_low_ppc,
            "T_histories": rec_t_hists,
            "T_grids": t_grids_final_K,
            "T_grids_K": t_grids_final_K,
            "T_grids_C": t_grids_final_C,
            "Tmap": getattr(sim, 'Tmap', None).copy() if getattr(sim, 'Tmap', None) is not None else None,
            "mask_grids": [m.copy() for m in getattr(sim, 'mask_grids', [])] if getattr(sim, 'mask_grids', None) else [],
            "isBound": getattr(sim, 'isBound', None).copy() if getattr(sim, 'isBound', None) is not None else None,
            "X": getattr(sim, 'X', None).copy() if getattr(sim, 'X', None) is not None else None,
            "Y": getattr(sim, 'Y', None).copy() if getattr(sim, 'Y', None) is not None else None,
            "V": getattr(sim, 'V', None).copy() if getattr(sim, 'V', None) is not None else None,
            "damage_map": getattr(sim, 'damage_map', None).copy() if getattr(sim, 'damage_map', None) is not None else None,
            "eroded_depth": getattr(sim, 'eroded_depth', None).copy() if getattr(sim, 'eroded_depth', None) is not None else None,
            "ion_current_history": list(getattr(sim, 'ion_current_history', [])),
            "tracking_buffer": rec_trk_buf,
            "energy_history_t": list(getattr(sim, 'energy_history_t', [])),
            "energy_history_ke_i": list(getattr(sim, 'energy_history_ke_i', [])),
            "energy_history_ke_e": list(getattr(sim, 'energy_history_ke_e', [])),
            "energy_history_fe": list(getattr(sim, 'energy_history_fe', [])),
            "energy_history_tot": list(getattr(sim, 'energy_history_tot', [])),
            "charge_history_t": list(getattr(sim, 'charge_history_t', [])),
            "charge_history_q_i": list(getattr(sim, 'charge_history_q_i', [])),
            "charge_history_q_e": list(getattr(sim, 'charge_history_q_e', [])),
            "charge_history_q_b": list(getattr(sim, 'charge_history_q_b', [])),
            "charge_history_q_free": list(getattr(sim, 'charge_history_q_free', [])),
            "charge_history_q_net": list(getattr(sim, 'charge_history_q_net', [])),
            "substep_history_iter": list(getattr(sim, 'substep_history_iter', [])),
            "substep_history_ion": list(getattr(sim, 'substep_history_ion', [])),
            "substep_history_e": list(getattr(sim, 'substep_history_e', [])),
            "Lx": float(getattr(sim, 'Lx', 3.0)),
            "Ly": float(getattr(sim, 'Ly', 3.0)),
            "nx": int(getattr(sim, 'nx', 0)),
            "ny": int(getattr(sim, 'ny', 0)),
            "dx": float(getattr(sim, 'dx', 0.05)),
            "dy": float(getattr(sim, 'dy', 0.05)),
            "current_ppc_map": getattr(sim, 'current_ppc_map', None).copy() if getattr(sim, 'current_ppc_map', None) is not None else None,
            "accum_ppc_map": getattr(sim, 'accum_ppc_map', None).copy() if getattr(sim, 'accum_ppc_map', None) is not None else None,
            "ppc_steps_count": int(getattr(sim, 'ppc_steps_count', 0)),
            "macro_weight": float(getattr(sim, 'macro_weight', 1e8)),
            "perf_history": list(sim._perf_monitor.history) if getattr(sim, '_perf_monitor', None) and sim._perf_monitor.history else [],
            "V": getattr(sim, 'V', None).copy() if getattr(sim, 'V', None) is not None else None,
            "result_row": result_row,
        }

    def _compute_perveance_metrics(self, i_exit):
        """
        Calculates effective perveance (from measured exit current and total acceleration voltage),
        theoretical maximum Child-Langmuir perveance (from physical constants and grid geometry),
        and perveance fraction (ratio of effective to theoretical maximum perveance).

        Returns:
            tuple: (p_effective, p_max, perveance_fraction)
        """
        if not np.isfinite(i_exit) or i_exit < 0.0:
            i_exit = 0.0

        sim = getattr(self, 'sim', None)
        grids = getattr(sim, 'grids', None)
        if not grids and hasattr(self, 'get_params'):
            try:
                grids = self.get_params().get('grids', [])
            except Exception:
                grids = []

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
        elif hasattr(self, 'inputs'):
            if 'Vs' in self.inputs:
                v_screen = float(self.inputs['Vs'].value())
            if 'Va' in self.inputs:
                v_accel = float(self.inputs['Va'].value())
            if 'gap' in self.inputs:
                gap_mm = float(self.inputs['gap'].value())
            if 'ts' in self.inputs:
                t_screen_mm = float(self.inputs['ts'].value())
            elif 't' in self.inputs:
                t_screen_mm = float(self.inputs['t'].value())
            if 'rs' in self.inputs:
                r_screen_mm = float(self.inputs['rs'].value())
            elif 'r' in self.inputs:
                r_screen_mm = float(self.inputs['r'].value())

        # Total acceleration voltage between screen and accelerator grids
        v_total = max(v_screen - v_accel, 0.0)

        # Measured effective perveance: Peffective = I_exit / (V_total^1.5)
        if v_total > 0.0 and i_exit > 0.0:
            p_eff = float(i_exit / (v_total ** 1.5))
        else:
            p_eff = 0.0

        # Physical constants
        # Child-Langmuir space-charge limited current perveance limit using Goebel & Katz
        # equivalent acceleration length: l_e = sqrt((l_g + t_s)^2 + r_s^2) = sqrt((l_g + t_s)^2 + d_s^2 / 4)
        # where l_g is grid separation, t_s is screen thickness, and r_s is screen hole radius.
        # P_max = (4 * eps_0 / 9) * sqrt(2 * q / m_i) * (A / l_e^2)
        eps_0 = 8.8541878128e-12  # F/m
        q = float(getattr(sim, 'q_ion', 1.602e-19))
        m_i = float(getattr(sim, 'm_ion', 131.293 * 1.6605e-27))
        if q <= 0.0 or m_i <= 0.0:
            q = 1.602e-19
            m_i = 131.293 * 1.6605e-27

        l_g = max(gap_mm * 1e-3, 1e-6)       # inter-grid separation in meters
        t_s = max(t_screen_mm * 1e-3, 0.0)    # screen grid thickness in meters
        r_s = max(r_screen_mm * 1e-3, 1e-6)   # screen aperture radius in meters
        l_e_sq = (l_g + t_s) ** 2 + (r_s ** 2)

        # Cross-sectional injection / extraction area in m^2
        last_area = getattr(sim, '_last_injection_area', None)
        if last_area is not None and last_area > 0.0:
            area = float(last_area)
        else:
            area = max(r_screen_mm * 1e-3 * 1e-3, 1e-8)

        p_max = float((4.0 * eps_0 / 9.0) * np.sqrt(2.0 * q / m_i) * (area / l_e_sq))

        if p_max > 0.0 and np.isfinite(p_eff):
            perv_frac = float(p_eff / p_max)
        else:
            perv_frac = 0.0

        return p_eff, p_max, perv_frac

    def advance_sweep_scenario(self):
        """Transitions the simulator to the next scenario in the sweep."""
        self.sweep_current_idx += 1
        cfg = self.sweep_config
        values = cfg["values"]

        if self.sweep_current_idx < len(values):
            next_scen_num = self.sweep_current_idx + 1
            if (
                next_scen_num in getattr(self, '_scenarios_pending_advisory', {})
                and next_scen_num not in getattr(self, '_scenario_advisory_decisions', {})
            ):
                self._prompt_scenario_plume_advisory(next_scen_num)
                if next_scen_num not in getattr(self, '_scenario_advisory_decisions', {}):
                    self._sweep_paused_current = True
                    self.sim_isRunning = False
                    if hasattr(self, 'lbl_status'):
                        self.lbl_status.setText(
                            f"Status: Scenario {next_scen_num} held awaiting plume decision. Select it in dropdown to choose."
                        )
                    if hasattr(self, 'btn_toggle'):
                        self.btn_toggle.setText("RESUME BEAM")
                    self._update_sweep_action_button_state()
                    return

            next_val = values[self.sweep_current_idx]
            if hasattr(self, 'combo_sweep_scenario') and self.combo_sweep_scenario is not None:
                if self.combo_sweep_scenario.currentIndex() == self.sweep_current_idx - 1:
                    self.combo_sweep_scenario.blockSignals(True)
                    self.combo_sweep_scenario.setCurrentIndex(self.sweep_current_idx)
                    self.combo_sweep_scenario.blockSignals(False)
            self._apply_sweep_value(cfg["param_id"], next_val)
            is_trunc = (getattr(self, '_scenario_advisory_decisions', {}).get(next_scen_num) == "truncate")
            self.truncate_downstream_plume = is_trunc
            if hasattr(self, 'sim') and self.sim is not None:
                self.sim.truncate_downstream_plume = is_trunc
            was_running = getattr(self, 'sim_isRunning', False)
            self.build_domain(force=True)
            self.sweep_scenario_start_iter = getattr(self.sim, 'iteration', 0)
            self.sweep_scenario_entry_iter = None
            self._get_sweep_scenario_timer().reset(start_running=was_running)
            self.sweep_scenario_start_wall_time = get_active_clock()
            self._sweep_paused_current = False
            if was_running:
                if not getattr(self, 'sim_isRunning', False):
                    self.sim.injection_enabled = True
                    self.sim_isRunning = True
                    self._resume_simulation_timer()
                    self._sim_worker.start_sim(self.get_params(), run_id=getattr(self, '_sim_run_id', 0))
                    self.update_sim_status_badge("RUNNING")
            if hasattr(self, 'btn_toggle'):
                self.btn_toggle.setText("PAUSE BEAM" if getattr(self, 'sim_isRunning', False) else "2. START BEAM")
            self._update_sweep_action_button_state()
            self._update_sweep_scenario_combo_items()
        else:
            self.finish_sweep()

    def stop_sweep(self):
        """User stops active sweep early (pauses all scenarios)."""
        self.is_sweep_active = False
        self.sweep_mode_armed = False
        self._sweep_stopped = True
        self._sweep_paused_current = True
        self.sim_isRunning = False

        if hasattr(self, 'sweep_monitor_timer') and self.sweep_monitor_timer is not None:
            self.sweep_monitor_timer.stop()
        if hasattr(self, '_pause_simulation_timer'):
            self._pause_simulation_timer()

        if hasattr(self, 'parallel_sweep_dlg') and self.parallel_sweep_dlg is not None:
            dlg = self.parallel_sweep_dlg
            self.parallel_sweep_dlg = None
            try:
                dlg.rejected.disconnect(self.stop_sweep)
            except Exception:
                pass
            try:
                dlg.reject()
            except Exception:
                pass

        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None and self.parallel_sweep_worker.isRunning():
            self.parallel_sweep_worker.pause_all_scenarios()
            if not hasattr(self, '_sweep_status_dict'):
                self._sweep_status_dict = {}
            self._sweep_status_dict.update(self.parallel_sweep_worker.get_status_dict())

        # If a sequential sweep was running the live simulator, pause it cleanly
        if hasattr(self, '_sim_worker') and self._sim_worker.isRunning():
            self._sim_worker.pause()
            if hasattr(self, 'update_sim_status_badge'):
                self.update_sim_status_badge("PAUSED")

        if hasattr(self, '_pause_simulation_timer'):
            self._pause_simulation_timer()

        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setText("Sweep mode stopped")
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #fef3c7; color: #b45309; font-weight: bold; "
                "border: 1px solid #fde68a; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(False)
        self._update_sweep_action_button_state()
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("RESUME BEAM")
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText("Parametric sweep stopped. Click 'Resume Sweep' or select a scenario to continue.")

        self._set_grid_edit_buttons_enabled(True)
        self._update_sweep_scenario_combo_items()

    def finish_sweep(self):
        """Called when all scenarios have completed."""
        self.is_sweep_active = False
        self.sweep_mode_armed = False
        self._sweep_paused_current = False
        self._set_grid_edit_buttons_enabled(True)

        if hasattr(self, '_sweep_total_timer') and self._sweep_total_timer is not None:
            total_time = self._sweep_total_timer.get_duration()
        else:
            total_time = get_active_clock() - getattr(self, 'sweep_start_wall_time', get_active_clock())
        if self.sweep_config is not None and "total_computation_time_s" not in self.sweep_config:
            self.sweep_config["total_computation_time_s"] = float(total_time)

        if hasattr(self, 'sweep_monitor_timer') and self.sweep_monitor_timer is not None:
            self.sweep_monitor_timer.stop()
        if hasattr(self, '_pause_simulation_timer'):
            self._pause_simulation_timer()

        if getattr(self, 'sim_isRunning', False):
            self.toggle_sim()

        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setText("✓ Sweep Completed")
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #dcfce7; color: #15803d; font-weight: bold; "
                "border: 1px solid #86efac; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(False)
        self._update_sweep_action_button_state()
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("2. START BEAM")

        self._update_sweep_scenario_combo_items()

        total_time_str = f" in {total_time:.1f}s" if total_time > 0 else ""
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(f"Parametric sweep completed ({len(self.sweep_results)} scenarios{total_time_str}). Use the dropdown to inspect any scenario.")

        # Open Results dialog
        self.sweep_results_dialog = SweepResultsDialog(self.sweep_config, self.sweep_results, self)
        self.sweep_results_dialog.show()
        self.sweep_results_dialog.raise_()

    def exit_sweep(self, confirm_choice=None):
        """Exit sweep mode and return to normal single simulation.

        If called while a sweep simulation is running or paused, prompts the user:
        'Attention: all data will be lost. Would you like to save data untill now?'
        with options 'Yes', 'No, exit', 'Cancel'.
        """
        if isinstance(confirm_choice, str):
            choice = confirm_choice.strip().lower()
            if choice not in ("yes", "no", "cancel"):
                choice = None
        else:
            choice = None

        is_active_or_paused = (
            getattr(self, 'sweep_config', None) is not None
            or getattr(self, 'is_sweep_active', False)
            or getattr(self, '_sweep_stopped', False)
            or getattr(self, 'sweep_mode_armed', False)
            or (hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None)
            or getattr(self, 'sim_isRunning', False)
            or getattr(self, '_sweep_paused_current', False)
            or bool(getattr(self, 'sweep_results', []))
            or bool(getattr(self, '_sweep_status_dict', {}))
        )

        if choice is None and is_active_or_paused:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Exit Sweep")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Attention: all data will be lost. Would you like to save data untill now?")
            btn_yes = msg_box.addButton("Yes", QMessageBox.YesRole)
            btn_no = msg_box.addButton("No, exit", QMessageBox.NoRole)
            btn_cancel = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            msg_box.setDefaultButton(btn_yes)
            msg_box.setEscapeButton(btn_cancel)
            msg_box.exec_()

            clicked = msg_box.clickedButton()
            if clicked == btn_cancel or clicked is None:
                choice = "cancel"
            elif clicked == btn_yes:
                choice = "yes"
            else:
                choice = "no"

        if choice == "cancel":
            return

        if choice == "yes":
            self._compile_and_show_interim_sweep_results()

        self._teardown_sweep_and_return_to_normal()

    def _compile_and_show_interim_sweep_results(self):
        """Compile available data until last iteration across all scenarios and open SweepResultsDialog."""
        completed_map = {r.get("scenario_idx"): r for r in getattr(self, 'sweep_results', []) if isinstance(r, dict)}

        status_dict = {}
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            try:
                status_dict = self.parallel_sweep_worker.get_status_dict()
            except Exception:
                status_dict = {}
        if hasattr(self, '_sweep_status_dict') and self._sweep_status_dict:
            status_dict.update(self._sweep_status_dict)

        cfg = getattr(self, 'sweep_config', {}) or {}
        param_id = cfg.get("param_id", "")
        values = cfg.get("values", [])

        # In sequential mode, include current scenario state if not already recorded
        if hasattr(self, 'sim') and self.sim is not None and getattr(self.sim, 'iteration', 0) > 0:
            cur_seq_idx = getattr(self, 'sweep_current_idx', 0)
            cur_scen = cur_seq_idx + 1
            if cur_scen not in completed_map and cur_scen not in status_dict:
                sim = self.sim
                cur_iter = getattr(sim, 'iteration', 0)
                scen_iter = cur_iter - getattr(self, 'sweep_scenario_start_iter', 0)
                div_val = getattr(sim, 'current_div', 0.0)
                div_m_val = getattr(sim, 'current_div_mid', 0.0)
                saddle_v = getattr(sim, 'min_pot', getattr(sim, 'saddle_point_potential', 0.0))
                i_exit = getattr(sim, 'exit_ion_current', 0.0)
                p_eff, p_max, perv_frac = self._compute_perveance_metrics(i_exit)
                status_dict[cur_scen] = {
                    "iter": scen_iter,
                    "parameter_value": values[cur_seq_idx] if cur_seq_idx < len(values) else 0.0,
                    "steady_state_reached": False,
                    "computation_time_s": (
                        self._sweep_scenario_timer.get_duration()
                        if hasattr(self, '_sweep_scenario_timer') and self._sweep_scenario_timer is not None
                        else (get_active_clock() - getattr(self, 'sweep_scenario_start_wall_time', get_active_clock()))
                    ),
                    "t_sim_us": float(getattr(sim, 'time_sim', 0.0)) * 1e6,
                    "min_pot": float(saddle_v) if np.isfinite(saddle_v) else 0.0,
                    "div": float(div_val) if np.isfinite(div_val) else 0.0,
                    "div_mid": float(div_m_val) if np.isfinite(div_m_val) else 0.0,
                    "exit_ion_current_A": float(i_exit) if np.isfinite(i_exit) else 0.0,
                    "perveance_fraction": float(perv_frac) if np.isfinite(perv_frac) else 0.0,
                    "transparency": float(getattr(sim, 'transparency', 0.0)),
                    "T_grids": [float(t) for t in getattr(sim, 'T_grids', [])],
                    "T_grids_K": [float(t) for t in getattr(sim, 'T_grids', [])],
                    "T_grids_C": [float(t - 273.15) for t in getattr(sim, 'T_grids', [])],
                }

        all_results = []
        for idx, val in enumerate(values):
            scen_num = idx + 1
            if scen_num in completed_map:
                all_results.append(completed_map[scen_num])
            elif scen_num in status_dict:
                info = status_dict[scen_num]
                t_grids_k_info = info.get("T_grids_K") or info.get("T_grids") or []
                if not t_grids_k_info and isinstance(info.get("T_histories"), dict) and info["T_histories"]:
                    t_grids_k_info = [
                        float(info["T_histories"][k][-1]) if info["T_histories"].get(k) else 300.0
                        for k in sorted(info["T_histories"].keys())
                    ]
                t_grids_c_info = info.get("T_grids_C") or [float(t - 273.15) for t in t_grids_k_info]
                grid_temps_info = {}
                for g_idx, (tk, tc) in enumerate(zip(t_grids_k_info, t_grids_c_info), start=1):
                    grid_temps_info[f"grid_{g_idx}_temp_K"] = tk
                    grid_temps_info[f"grid_{g_idx}_temp_C"] = tc
                row = {
                    "scenario_idx": scen_num,
                    "param_id": param_id,
                    "parameter_value": float(info.get("parameter_value", val)),
                    "steady_state_reached": bool(info.get("steady_state_reached", False)),
                    "max_iters_hit": None,
                    "steady_iterations": int(info.get("iter", 0)),
                    "computation_time_s": float(info.get("computation_time_s", 0.0)),
                    "t_sim_us": float(info.get("t_sim_us", 0.0)),
                    "saddle_potential_V": float(info.get("min_pot", 0.0)),
                    "divergence_deg": float(info.get("div", 0.0)),
                    "divergence_2nd_plane_deg": float(info.get("div_mid", 0.0)),
                    "exit_ion_current_A": float(info.get("exit_ion_current_A", 0.0)),
                    "perveance_fraction": float(info.get("perveance_fraction", 0.0)),
                    "transparency": float(info.get("transparency", 0.0)),
                    "total_energy_J": 0.0,
                    "net_charge_C": 0.0,
                    "T_grids_K": list(t_grids_k_info),
                    "T_grids_C": list(t_grids_c_info),
                    **grid_temps_info,
                }
                all_results.append(row)

        if not all_results and getattr(self, 'sweep_results', []):
            all_results = list(self.sweep_results)

        all_results.sort(key=lambda r: r.get("scenario_idx", 0))

        if all_results and cfg:
            self.sweep_results_dialog = SweepResultsDialog(cfg, all_results, self)
            self.sweep_results_dialog.show()
            self.sweep_results_dialog.raise_()

    def _teardown_sweep_and_return_to_normal(self):
        """Shutdown workers and restore the UI and state to normal single simulation."""
        # 1. Stop background workers
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            worker = self.parallel_sweep_worker
            self.parallel_sweep_worker = None
            try:
                worker.cancel()
                worker.wait(2000)
            except Exception:
                pass

        if hasattr(self, '_sim_worker') and self._sim_worker.isRunning():
            self._sim_worker.pause()
            self._sim_worker.wait(1000)

        # 2. Stop timers
        if hasattr(self, 'sweep_monitor_timer') and self.sweep_monitor_timer is not None:
            self.sweep_monitor_timer.stop()
        if hasattr(self, '_pause_simulation_timer'):
            self._pause_simulation_timer()

        # 3. Reset sweep state flags
        self.is_sweep_active = False
        self.sweep_mode_armed = False
        self._sweep_stopped = False
        self._sweep_paused_current = False
        self._sweep_domain_built = False
        self.sim_isRunning = False
        self.sweep_config = None
        self.sweep_results = []
        self._sweep_status_dict = {}
        self._sweep_tracking_buffers = {}
        self._seq_scenario_histories = {}

        # 4. Hide all sweep top-bar UI elements
        self._hide_sweep_top_bar_controls()
        self._set_grid_edit_buttons_enabled(True)
        self._bg_live = None

        # 5. Restore normal simulation controls and badge
        if getattr(self, 'ppc_window', None):
            self.ppc_window.setWindowTitle("Macroparticle Distribution per Cell (PPC)")
            if self.ppc_window.isVisible() and hasattr(self, 'sim'):
                self.ppc_window.update_plot(self.sim)

        if getattr(self, 'phys_window', None):
            self.phys_window.setWindowTitle("Total Energy Conservation Monitor")
            if self.phys_window.isVisible() and hasattr(self, 'sim'):
                self.phys_window.update_plot(self.sim)

        if getattr(self, 'mesh_window', None):
            self.mesh_window.setWindowTitle("Computational Domain & Mesh Grid Inspector")
            if self.mesh_window.isVisible() and hasattr(self, 'sim'):
                params = self.get_params() if hasattr(self, 'get_params') else None
                self.mesh_window.update_plot(self.sim, params=params)

        if getattr(self, 'substep_window', None):
            self.substep_window.setWindowTitle("Displacement Substeps Monitor — (Iterations vs Required Substeps)")
            if self.substep_window.isVisible() and hasattr(self, 'sim'):
                self.substep_window.update_plot(self.sim)

        if getattr(self, 'perf_window', None):
            self.perf_window.setWindowTitle("Runtime Performance & Particle Diagnostics Monitor")
            if self.perf_window.isVisible() and hasattr(self, 'sim'):
                self.perf_window.update_plot(self.sim)

        if getattr(self, 'density_window', None):
            self.density_window.setWindowTitle("Plasma & Particle Density Diagnostics")
            if self.density_window.isVisible() and hasattr(self, 'sim'):
                self.density_window.update_plot(self.sim)

        if getattr(self, 'charge_window', None):
            self.charge_window.setWindowTitle("Total Charge Monitor")
            if self.charge_window.isVisible() and hasattr(self, 'sim'):
                self.charge_window.update_plot(self.sim)

        if getattr(self, 'iedf_window', None):
            self.iedf_window.setWindowTitle("Energy Distribution Function (IEDF & EEDF)")

        if getattr(self, 'field_window', None):
            self.field_window.setWindowTitle("Electric Field & Potential Diagnostics")
            if self.field_window.isVisible() and hasattr(self, 'sim'):
                self.field_window.update_plot(self.sim)

        if getattr(self, 'steady_state_window', None):
            w = self.steady_state_window.spin_window.value() if hasattr(self.steady_state_window, 'spin_window') else 100
            self.steady_state_window.setWindowTitle(f"Steady State Diagnostics — {w}-Iteration Variation")
            if self.steady_state_window.isVisible() and hasattr(self, 'sim'):
                self.steady_state_window.update_plot(self.sim, self)

        if hasattr(self, 'sim') and hasattr(self.sim, 'V') and self.sim.V is not None:
            if getattr(self.sim, 'Ex', None) is None or getattr(self.sim, 'Ey', None) is None:
                try:
                    dy_m = float(getattr(self.sim, 'dy', 0.05)) * 1e-3
                    dx_m = float(getattr(self.sim, 'dx', 0.05)) * 1e-3
                    self.sim.Ey, self.sim.Ex = np.gradient(-self.sim.V, dy_m, dx_m)
                except Exception:
                    pass

        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("2. START BEAM")
        if hasattr(self, 'update_sim_status_badge'):
            self.update_sim_status_badge("READY")
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText("Exited sweep mode. Restored to normal single simulation.")
