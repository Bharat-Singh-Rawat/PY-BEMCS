"""
Parametric Sweep Mixin for DigitalTwinApp.
Coordinates automated execution across multiple physical scenarios,
steady-state condition checks, scenario transitions, and results collection.
"""

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from ..dialogs import SweepSettingsDialog, SweepResultsDialog
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

        total_scenarios = len(config["values"])
        param_label = config["param_label"]
        first_val = config["values"][0]

        # Update Top Bar Badge & Button
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep mode active [1/{total_scenarios} | {first_val:.3g} {config['param_unit']}]"
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
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep mode active [{cur_idx + 1}/{total_scenarios} | {cur_val:.3g} {cfg['param_unit']} ({scenario_iters:,} iters)]"
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
        div = self.div_history[-1] if self.div_history else 0.0
        i_exit = self.ion_current_exit_history[-1] if self.ion_current_exit_history else 0.0
        trans = self.transparency_history[-1] if self.transparency_history else 0.0

        result_row = {
            "scenario_idx": scenario_idx,
            "parameter_value": parameter_value,
            "steady_state_reached": bool(steady_reached),
            "max_iters_hit": max_iters_hit,
            "steady_iterations": int(scenario_iters),
            "t_sim_us": float(t_sim_us),
            "saddle_potential_V": float(saddle_v),
            "divergence_deg": float(div),
            "exit_ion_current_A": float(i_exit),
            "transparency": float(trans),
            "total_energy_J": float(e_tot),
            "total_charge_C": float(q_net),
            "field_energy_J": float(fe),
        }
        self.sweep_results.append(result_row)

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

    def finish_sweep(self):
        """Called when all scenarios have completed."""
        self.is_sweep_active = False

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

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(f"Parametric sweep completed ({len(self.sweep_results)} scenarios).")

        # Open Results dialog
        self.sweep_results_dialog = SweepResultsDialog(self.sweep_config, self.sweep_results, self)
        self.sweep_results_dialog.show()
        self.sweep_results_dialog.raise_()
