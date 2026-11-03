"""
Parametric Sweep Mixin for DigitalTwinApp.
Coordinates automated execution across multiple physical scenarios,
steady-state condition checks, scenario transitions, and results collection.
"""

import numpy as np
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer

from ..dialogs import SweepSettingsDialog, SweepResultsDialog, ParallelSweepProgressDialog, format_sci_value
from ..workers import ParallelSweepWorker
from diagnostics.metrics import compute_rolling_variation


class SweepMixin:
    """Mixin managing the Parametric Sweep simulation workflow."""

    def open_sweep_dialog(self):
        """Open modal configuration dialog to start a parametric sweep."""
        if getattr(self, 'is_sweep_active', False):
            QMessageBox.information(
                self,
                "Sweep in Progress",
                "A parametric sweep is already active. Stop or wait for it to complete."
            )
            return

        dialog = SweepSettingsDialog(self)
        if dialog.exec_() == SweepSettingsDialog.Accepted:
            cfg = dialog.get_config()
            self.start_sweep(cfg)

    def start_sweep(self, config):
        """Initialize and launch the multi-scenario sweep."""
        self.sweep_config = config
        self.sweep_results = []
        self.sweep_current_idx = 0
        self.is_sweep_active = True

        total_scenarios = len(config.get("values", []))
        param_label = config.get("param_label", "Parameter")
        use_parallel = config.get("parallel", False) and total_scenarios > 0

        if use_parallel:
            n_cores = int(config.get("n_cores", 8))
            self._start_parallel_sweep(config, n_cores)
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
                f"⚡ Sweep mode active [1/{total_scenarios} | {first_val_str} {config['param_unit']}]"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(True)

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(
                f"Starting Parametric Sweep: {param_label} ({total_scenarios} scenarios)..."
            )

        # Apply initial scenario value
        self._apply_sweep_value(config["param_id"], first_val)

        # Build domain and start simulation
        self.build_domain()
        self.sweep_scenario_start_iter = getattr(self.sim, 'iteration', 0)

        if not getattr(self, 'sim_isRunning', False):
            self.toggle_sim()

    def _start_parallel_sweep(self, config, n_cores):
        """Launch parallel multi-core sweep using ProcessPoolExecutor and top-bar scenario selector."""
        total_scenarios = len(config.get("values", []))
        param_label = config.get("param_label", "Parameter")

        # Pause live simulator if currently running
        if getattr(self, 'sim_isRunning', False):
            self.toggle_sim()

        # Update Top Bar Badge & Button
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText(
                f"⚡ Parallel Sweep [{n_cores} cores | 0/{total_scenarios} Done]"
            )
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #f3e8ff; color: #7e22ce; font-weight: bold; "
                "border: 1px solid #d8b4fe; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )

        # Initialize and display top-bar scenario selector dropdown
        if hasattr(self, 'combo_sweep_scenario'):
            self.combo_sweep_scenario.blockSignals(True)
            self.combo_sweep_scenario.clear()
            for idx, val in enumerate(config.get("values", [])):
                val_str = (
                    format_sci_value(val)
                    if (config.get("param_id") in ("n0_plasma", "n0") or abs(val) >= 1e4 or (0 < abs(val) <= 1e-3))
                    else f"{val:.3g}"
                )
                unit_str = f" {config.get('param_unit', '')}" if config.get('param_unit') else ""
                self.combo_sweep_scenario.addItem(f"Scenario {idx + 1}: {config.get('param_id', 'param')} = {val_str}{unit_str}  [⏳ Queued]")
            self.combo_sweep_scenario.setCurrentIndex(0)
            self.combo_sweep_scenario.blockSignals(False)
            self.combo_sweep_scenario.setVisible(True)

        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(True)

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(
                f"Running Parallel Sweep: {param_label} ({total_scenarios} scenarios on {n_cores} cores). Switch active views via top dropdown."
            )

        base_params = self.get_params() if hasattr(self, 'get_params') else {}

        self.parallel_sweep_worker = ParallelSweepWorker(config, base_params, n_cores=n_cores, parent=self)

        self.parallel_sweep_worker.progress_updated.connect(self._on_parallel_progress_updated)
        self.parallel_sweep_worker.scenario_completed.connect(self._on_parallel_scenario_completed)
        self.parallel_sweep_worker.sweep_finished.connect(self._on_parallel_sweep_finished)
        self.parallel_sweep_worker.sweep_error.connect(self._on_parallel_sweep_error)

        # Start live GUI monitor timer to sync top-bar dropdown items and active canvas view
        if not hasattr(self, 'sweep_monitor_timer') or self.sweep_monitor_timer is None:
            self.sweep_monitor_timer = QTimer(self)
            self.sweep_monitor_timer.timeout.connect(self._on_sweep_monitor_tick)
        self.sweep_monitor_timer.start(150)

        self.parallel_sweep_worker.start()

    def _on_sweep_monitor_tick(self):
        """Periodic tick to update dropdown labels and render active scenario view."""
        self._update_sweep_scenario_combo_items()
        if hasattr(self, 'combo_sweep_scenario') and self.combo_sweep_scenario is not None:
            active_idx = self.combo_sweep_scenario.currentIndex()
            if active_idx >= 0:
                self._render_sweep_scenario_snapshot(active_idx + 1)

    def _on_sweep_scenario_selected(self, index):
        """User switched scenario in top dropdown: immediately refresh canvas."""
        if index < 0:
            return
        self._render_sweep_scenario_snapshot(index + 1)

    def _update_sweep_scenario_combo_items(self):
        """Update top-bar scenario dropdown items with real-time status and iteration count."""
        if not hasattr(self, 'combo_sweep_scenario') or self.sweep_config is None:
            return

        cfg = self.sweep_config
        values = cfg.get("values", [])
        param_id = cfg.get("param_id", "")
        param_unit = cfg.get("param_unit", "")

        status_dict = {}
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            status_dict = self.parallel_sweep_worker.get_status_dict()
        if hasattr(self, '_sweep_status_dict') and self._sweep_status_dict:
            status_dict.update(self._sweep_status_dict)

        current_idx = self.combo_sweep_scenario.currentIndex()
        self.combo_sweep_scenario.blockSignals(True)
        try:
            for idx, val in enumerate(values):
                scen_num = idx + 1
                info = status_dict.get(scen_num, {})
                status = info.get("status", "queued")
                scen_iter = info.get("iter", 0)

                val_str = (
                    format_sci_value(val)
                    if (param_id in ("n0_plasma", "n0") or abs(val) >= 1e4 or (0 < abs(val) <= 1e-3))
                    else f"{val:.3g}"
                )
                unit_str = f" {param_unit}" if param_unit else ""

                if status == "running":
                    tag = f"[▶ Running - {scen_iter} iters]"
                elif status == "done":
                    tag = f"[✔ Done - {scen_iter} iters]"
                elif status == "stopped":
                    tag = f"[⏹ Stopped]"
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

    def _render_sweep_scenario_snapshot(self, scenario_idx):
        """Render the chosen sweep scenario's particle data and physical observables on the canvas."""
        status_dict = {}
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            status_dict = self.parallel_sweep_worker.get_status_dict()
        if hasattr(self, '_sweep_status_dict') and self._sweep_status_dict:
            status_dict.update(self._sweep_status_dict)

        info = status_dict.get(scenario_idx)
        if not info:
            return

        # 1. Update primary ion particles
        prim_xy = info.get("prim_xy")
        e_prim_eV = info.get("e_prim_eV")
        if prim_xy is not None and len(prim_xy) > 0 and hasattr(self, 'scat_prim') and self.scat_prim is not None:
            self.scat_prim.set_offsets(prim_xy)
            if e_prim_eV is not None and len(e_prim_eV) > 0:
                self.scat_prim.set_array(e_prim_eV)
                e_min = float(np.min(e_prim_eV))
                e_max = float(np.max(e_prim_eV))
                if e_max <= e_min:
                    e_max = e_min + 1.0
                self.scat_prim.set_clim(e_min, e_max)

        # Clear CEX and electrons on sweep view if not tracked
        if hasattr(self, 'scat_cex') and self.scat_cex is not None:
            self.scat_cex.set_offsets(np.empty((0, 2)))
        if hasattr(self, 'scat_elec') and self.scat_elec is not None:
            self.scat_elec.set_offsets(np.empty((0, 2)))

        # 2. Update status and metrics labels
        t_us = info.get("t_sim_us", 0.0)
        scenario_iter = info.get("iter", 0)
        div = info.get("div", 0.0)
        div_mid = info.get("div_mid", 0.0)
        i_exit = info.get("exit_ion_current_A", 0.0)
        trans = info.get("transparency", 0.0)
        status_tag = info.get("status", "unknown").upper()
        p_eff = info.get("perveance_effective", 0.0)

        if hasattr(self, 'lblTime'):
            self.lblTime.setText(f"t_sim: {t_us:.2f} µs | Scenario {scenario_idx} [{status_tag}]")

        if hasattr(self, 'lblTransparency'):
            self.lblTransparency.setText(
                f"Transparency: {trans:.3f}\n"
                f"Divergence Exit: {div:.2f}°\n"
                f"Divergence Plane 2: {div_mid:.2f}°\n"
                f"Exit I_ion: {i_exit * 1e3:.3f} mA\n"
                f"Peffective: {p_eff:.3e}\n"
                f"Iteration: {scenario_iter:,}"
            )

        if hasattr(self, 'lbl_top_sim_time'):
            self.lbl_top_sim_time.setText(f"Scenario {scenario_idx} | Plasma: {t_us:.2f} µs | Iter: {scenario_iter:,}")

        if hasattr(self, 'lblPerfPtcls'):
            num_pts = len(prim_xy) if prim_xy is not None else 0
            self.lblPerfPtcls.setText(f"Display Ions: {num_pts:,} | State: {status_tag}")

        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _on_parallel_progress_updated(self, completed, total, status_text):
        if hasattr(self, 'lbl_sweep_badge'):
            n_cores = getattr(self.parallel_sweep_worker, 'n_cores', '') if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker else ''
            core_str = f"{n_cores} cores | " if n_cores else ""
            self.lbl_sweep_badge.setText(
                f"⚡ Parallel Sweep [{core_str}{completed}/{total} Done]"
            )
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(status_text)

    def _on_parallel_scenario_completed(self, result_row, completed, total):
        self.sweep_results.append(result_row)
        self._update_sweep_scenario_combo_items()

    def _on_parallel_sweep_finished(self, results):
        self.sweep_results = results
        if hasattr(self, 'sweep_monitor_timer') and self.sweep_monitor_timer is not None:
            self.sweep_monitor_timer.stop()
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
        QMessageBox.critical(self, "Parallel Sweep Error", f"An error occurred during multi-core sweep:\n{error_msg}")

    def _apply_sweep_value(self, param_id, value):
        """Applies a target parameter value to the GUI inputs and simulator."""
        if param_id == "Te_up" and "Te_up" in self.inputs:
            self.inputs["Te_up"].setValue(value)
        elif param_id == "n0_plasma" and "n0_plasma" in self.inputs:
            self.inputs["n0_plasma"].setValue(value)
        elif param_id == "Ti" and "Ti" in self.inputs:
            self.inputs["Ti"].setValue(value)
        elif param_id == "n0" and "n0" in self.inputs:
            self.inputs["n0"].setValue(value)
        elif param_id == "Tn" and "Tn" in self.inputs:
            self.inputs["Tn"].setValue(value)
        elif param_id == "Te" and "Te" in self.inputs:
            self.inputs["Te"].setValue(value)
        elif param_id == "grid_1_V" and len(getattr(self, 'grid_widgets', [])) >= 1:
            self.grid_widgets[0]["V"].setValue(value)
        elif param_id == "grid_2_V" and len(getattr(self, 'grid_widgets', [])) >= 2:
            self.grid_widgets[1]["V"].setValue(value)

    def check_sweep_steady_state(self):
        """
        Called on each GUI render tick when sweep mode is active.
        Monitors whether the current scenario has reached steady state,
        records its physical snapshot, and advances to the next scenario.
        """
        if not getattr(self, 'is_sweep_active', False) or self.sweep_config is None:
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
        scenario_iters = current_iter - getattr(self, 'sweep_scenario_start_iter', 0)

        min_iters = cfg.get("min_iterations", 50)
        max_iters = cfg.get("max_iterations", 5000)
        window = cfg.get("window", 50)
        thresh = cfg.get("threshold", 1.0)

        # Update live badge text
        if hasattr(self, 'lbl_sweep_badge'):
            cur_val_str = (
                format_sci_value(cur_val)
                if (cfg.get("param_id") in ("n0_plasma", "n0") or abs(cur_val) >= 1e4 or (0 < abs(cur_val) <= 1e-3))
                else f"{cur_val:.3g}"
            )
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep mode active [{cur_idx + 1}/{total_scenarios} | {cur_val_str} {cfg['param_unit']} ({scenario_iters:,} iters)]"
            )

        # Need at least min_iters and window steps
        if scenario_iters < max(min_iters, window):
            return

        # Check steady-state condition across core physical observables
        steady_reached = False
        max_hit = scenario_iters >= max_iters

        if not max_hit:
            histories_to_check = []
            if hasattr(sim, 'energy_history_tot') and len(sim.energy_history_tot) >= window:
                histories_to_check.append(sim.energy_history_tot[-scenario_iters:])
            if hasattr(sim, 'charge_history_q_net') and len(sim.charge_history_q_net) >= window:
                histories_to_check.append(sim.charge_history_q_net[-scenario_iters:])
            if hasattr(sim, 'potential_history') and len(sim.potential_history) >= window:
                histories_to_check.append(sim.potential_history[-scenario_iters:])
            elif hasattr(self, 'ebs_history') and len(self.ebs_history) >= window:
                histories_to_check.append(self.ebs_history[-scenario_iters:])

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

    def _record_sweep_scenario_result(self, scenario_idx, parameter_value, steady_reached, scenario_iters, max_iters_hit=None):
        """Records final steady-state physical observables for the completed scenario."""
        sim = self.sim
        t_sim_us = (sim.iteration * sim.dt * 1e6) if hasattr(sim, 'dt') else 0.0

        e_tot = sim.energy_history_tot[-1] if hasattr(sim, 'energy_history_tot') and sim.energy_history_tot else 0.0
        q_net = sim.charge_history_q_net[-1] if hasattr(sim, 'charge_history_q_net') and sim.charge_history_q_net else 0.0
        fe = sim.energy_history_fe[-1] if hasattr(sim, 'energy_history_fe') and sim.energy_history_fe else 0.0
        saddle_v = sim.min_pot if hasattr(sim, 'min_pot') else (self.ebs_history[-1] if self.ebs_history else 0.0)
        div = 0.0
        if self.div_history:
            finite_div = [d for d in self.div_history if np.isfinite(d)]
            div = finite_div[-1] if finite_div else (0.0 if np.isnan(self.div_history[-1]) else self.div_history[-1])
        elif hasattr(sim, 'current_div') and np.isfinite(sim.current_div):
            div = sim.current_div

        div_plane2 = 0.0
        div_mid_hist = getattr(self, 'div_mid_history', None)
        if div_mid_hist:
            finite_div2 = [d for d in div_mid_hist if np.isfinite(d)]
            div_plane2 = finite_div2[-1] if finite_div2 else (0.0 if np.isnan(div_mid_hist[-1]) else div_mid_hist[-1])
        elif hasattr(sim, 'current_div_mid') and np.isfinite(sim.current_div_mid):
            div_plane2 = sim.current_div_mid

        i_exit = self.ion_current_exit_history[-1] if self.ion_current_exit_history else 0.0
        trans = self.transparency_history[-1] if self.transparency_history else 0.0

        p_eff, p_max, perv_frac = self._compute_perveance_metrics(i_exit)

        result_row = {
            "scenario_idx": scenario_idx,
            "parameter_value": parameter_value,
            "steady_state_reached": bool(steady_reached),
            "max_iters_hit": max_iters_hit,
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
        self.sweep_results.append(result_row)

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
            next_val = values[self.sweep_current_idx]
            self._apply_sweep_value(cfg["param_id"], next_val)
            was_running = getattr(self, 'sim_isRunning', False)
            self.build_domain()
            self.sweep_scenario_start_iter = getattr(self.sim, 'iteration', 0)
            if was_running:
                self.toggle_sim()
        else:
            self.finish_sweep()

    def stop_sweep(self):
        """User cancels active sweep early."""
        self.is_sweep_active = False

        if hasattr(self, 'sweep_monitor_timer') and self.sweep_monitor_timer is not None:
            self.sweep_monitor_timer.stop()

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

        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            worker = self.parallel_sweep_worker
            self.parallel_sweep_worker = None
            try:
                worker.cancel()
                worker.wait(2000)
            except Exception:
                pass

        # If a sequential sweep was running the live simulator, pause it
        if getattr(self, 'sim_isRunning', False):
            self.toggle_sim()

        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setText("Sweep mode stopped")
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #f1f5f9; color: #64748b; font-weight: bold; "
                "border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(False)
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText("Parametric sweep cancelled by user.")

        self._update_sweep_scenario_combo_items()

    def finish_sweep(self):
        """Called when all scenarios have completed."""
        self.is_sweep_active = False

        if hasattr(self, 'sweep_monitor_timer') and self.sweep_monitor_timer is not None:
            self.sweep_monitor_timer.stop()

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

        self._update_sweep_scenario_combo_items()

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(f"Parametric sweep completed ({len(self.sweep_results)} scenarios). Use the dropdown to inspect any scenario.")

        # Open Results dialog
        self.sweep_results_dialog = SweepResultsDialog(self.sweep_config, self.sweep_results, self)
        self.sweep_results_dialog.show()
        self.sweep_results_dialog.raise_()
