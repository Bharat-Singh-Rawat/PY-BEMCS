"""
Parametric Sweep Mixin for DigitalTwinApp.
Coordinates automated execution across multiple physical scenarios,
steady-state condition checks, scenario transitions, and results collection.
"""

import time
import copy
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer

from ..dialogs import SweepSettingsDialog, SweepResultsDialog, ParallelSweepProgressDialog, format_sci_value
from ..workers import ParallelSweepWorker
from diagnostics.metrics import compute_rolling_variation


class SweepMixin:
    """Mixin managing the Parametric Sweep simulation workflow."""

    def open_sweep_dialog(self):
        """Open modal configuration dialog to save sweep settings."""
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
            self.save_sweep_settings(cfg)

    def save_sweep_settings(self, config):
        """Save sweep configuration and arm sweep mode without immediately launching."""
        self.sweep_config = config
        self.sweep_results = []
        self.sweep_current_idx = 0
        self.sweep_mode_armed = True
        self.is_sweep_active = False
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
            self.combo_sweep_scenario.setVisible(True)

        # Update Top Bar Badge & Buttons
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep mode armed [1/{total_scenarios} | {first_val_str} {param_unit}]"
            )
            self.lbl_sweep_badge.setStyleSheet(
                "background-color: #f3e8ff; color: #7e22ce; font-weight: bold; "
                "border: 1px solid #c084fc; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(True)
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)
        if hasattr(self, 'btn_resume_sweep'):
            self.btn_resume_sweep.setVisible(True)
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("2. START BEAM")

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(
                f"Sweep settings saved for {param_label} ({total_scenarios} scenarios). "
                f"Build domain (1. BUILD DOMAIN) and start simulation to begin sweep."
            )

    def resume_sweep_with_selected(self):
        """Resume simulation / sweep with parameters from the selected scenario in combo_sweep_scenario."""
        if not hasattr(self, 'combo_sweep_scenario') or self.sweep_config is None:
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

        # Case A: Parallel sweep worker running/paused
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None and self.parallel_sweep_worker.isRunning():
            self.is_sweep_active = True
            self._sweep_stopped = False
            self._sweep_paused_current = False
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
                    f"⚡ Parallel Sweep [{scen_num}/{total_scenarios} | {val_str} {param_unit}]"
                )
                self.lbl_sweep_badge.setStyleSheet(
                    "background-color: #f3e8ff; color: #7e22ce; font-weight: bold; "
                    "border: 1px solid #d8b4fe; border-radius: 4px; padding: 2px 8px; font-size: 11px;"
                )
            if hasattr(self, 'btn_stop_sweep'):
                self.btn_stop_sweep.setVisible(True)
            if hasattr(self, 'btn_resume_sweep'):
                self.btn_resume_sweep.setVisible(False)
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

        # Case B: Sequential sweep mode
        domain_unbuilt = not hasattr(self.sim, 'Ex') or self.sim.Ex is None or not np.any(self.sim.Ex)
        needs_rebuild = domain_unbuilt

        if needs_rebuild:
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
        self.sweep_scenario_start_wall_time = time.perf_counter()
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
                f"⚡ Sweep mode active [{cur_idx + 1}/{total_scenarios} | {val_str} {param_unit}]"
            )
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(True)
        if hasattr(self, 'btn_resume_sweep'):
            self.btn_resume_sweep.setVisible(False)
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
        self.sweep_config = config
        self.sweep_results = []
        self.sweep_current_idx = 0
        self.is_sweep_active = True
        self.sweep_mode_armed = False
        self._sweep_paused_current = False
        self._sweep_base_params = copy.deepcopy(self.get_params()) if hasattr(self, 'get_params') else {}
        self._scenario_domain_cache = {}
        self._current_rendered_scenario_grid_idx = None
        self.sweep_start_wall_time = time.perf_counter()
        self.sweep_scenario_start_wall_time = time.perf_counter()
        if hasattr(self, '_resume_simulation_timer'):
            self._resume_simulation_timer()

        total_scenarios = len(config.get("values", []))
        param_label = config.get("param_label", "Parameter")
        use_parallel = config.get("parallel", False) and total_scenarios > 0

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

        if hasattr(self, 'btn_resume_sweep'):
            self.btn_resume_sweep.setVisible(True)

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
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)

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
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("PAUSE BEAM")

    def _start_parallel_sweep(self, config, n_cores):
        """Launch parallel multi-core sweep using ProcessPoolExecutor and top-bar scenario selector."""
        self.is_sweep_active = True
        self.sweep_mode_armed = False
        self._sweep_paused_current = False
        total_scenarios = len(config.get("values", []))
        param_label = config.get("param_label", "Parameter")

        # Pause live simulator if currently running
        if getattr(self, 'sim_isRunning', False):
            self.toggle_sim()
        if hasattr(self, '_resume_simulation_timer'):
            self._resume_simulation_timer()

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
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("PAUSE BEAM")

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(
                f"Running Parallel Sweep: {param_label} ({total_scenarios} scenarios on {n_cores} cores). Switch active views via top dropdown."
            )

        base_params = self.get_params() if hasattr(self, 'get_params') else {}
        self._sweep_base_params = copy.deepcopy(base_params)
        self._scenario_domain_cache = {}
        self._current_rendered_scenario_grid_idx = None
        scatter_limit = getattr(self, 'scatter_limit', getattr(getattr(self, '_sim_worker', None), '_scatter_limit', 100000))

        self.parallel_sweep_worker = ParallelSweepWorker(
            config, base_params, n_cores=n_cores, scatter_limit=scatter_limit, parent=self
        )

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
        self.parallel_sweep_worker.set_active_view_scenario(1)

    def _on_sweep_monitor_tick(self):
        """Periodic tick to update dropdown labels and render active scenario view."""
        self._update_sweep_scenario_combo_items()
        if hasattr(self, 'combo_sweep_scenario') and self.combo_sweep_scenario is not None:
            active_idx = self.combo_sweep_scenario.currentIndex()
            if active_idx >= 0:
                self._update_toggle_btn_for_scenario(active_idx + 1)
                self._render_sweep_scenario_snapshot(active_idx + 1)

    def _on_sweep_scenario_selected(self, index):
        """User switched scenario in top dropdown: immediately refresh canvas, inputs, and toggle button."""
        if index < 0:
            return
        scenario_idx = index + 1
        cfg = getattr(self, 'sweep_config', {})
        values = cfg.get("values", [])
        param_id = cfg.get("param_id", "")

        # 1. Update left panel parameter inputs
        if 0 <= index < len(values):
            val = values[index]
            self._apply_sweep_value(param_id, val)

        # 2. Sync grid spinboxes from cached scenario domain data
        d_data = self._get_scenario_domain_data(scenario_idx)
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
        current_idx = self.combo_sweep_scenario.currentIndex()
        self.combo_sweep_scenario.blockSignals(True)
        try:
            for idx, val in enumerate(values):
                scen_num = idx + 1
                info = status_dict.get(scen_num, {})
                status = info.get("status", "queued")
                scen_iter = info.get("iter", 0)

                if is_parallel:
                    if self.parallel_sweep_worker.is_scenario_paused(scen_num):
                        status = "paused"
                    elif status == "paused":
                        status = "running"
                elif status not in ("done",):
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

                val_str = (
                    format_sci_value(val)
                    if (param_id in ("n0_plasma", "n0") or abs(val) >= 1e4 or (0 < abs(val) <= 1e-3))
                    else f"{val:.3g}"
                )
                unit_str = f" {param_unit}" if param_unit else ""

                comp_s = info.get("computation_time_s")
                comp_tag = f", {comp_s:.1f}s" if (comp_s is not None and np.isfinite(comp_s) and comp_s > 0) else ""

                if status == "running":
                    tag = f"[▶ Running - {scen_iter} iters]"
                elif status == "paused":
                    tag = f"[⏸ Paused - {scen_iter} iters]"
                elif status == "done":
                    tag = f"[✔ Done - {scen_iter} iters{comp_tag}]"
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

    def _get_scenario_domain_data(self, scenario_idx):
        """Retrieve or compute domain geometry, grid electrode boundaries, and potential contour for scenario_idx."""
        if not hasattr(self, '_scenario_domain_cache'):
            self._scenario_domain_cache = {}

        if scenario_idx in self._scenario_domain_cache:
            return self._scenario_domain_cache[scenario_idx]

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

        from physics.headless_sweep import apply_sweep_param_to_dict
        scenario_params = apply_sweep_param_to_dict(base_params, param_id, val)

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
                "bound_xy": bound_xy,
                "X": sim_temp.X,
                "Y": sim_temp.Y,
                "V": sim_temp.V,
                "Lx": float(sim_temp.Lx),
                "Ly": float(sim_temp.Ly),
                "grids": scenario_params.get("grids", []),
                "params": scenario_params,
            }
            self._scenario_domain_cache[scenario_idx] = data
            return data
        except Exception:
            return {}

    def _update_toggle_btn_for_scenario(self, scenario_idx):
        """Update btn_toggle text ('PAUSE BEAM' or 'RESUME BEAM') based on selected scenario state."""
        if not hasattr(self, 'btn_toggle'):
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
                    self.btn_stop_sweep.setVisible(True)
                if hasattr(self, 'btn_resume_sweep'):
                    self.btn_resume_sweep.setVisible(False)
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

        info = status_dict.get(scenario_idx, {})

        # 0. Sync live axes limits, grid electrode boundaries, and potential contour
        d_data = self._get_scenario_domain_data(scenario_idx)
        lx = d_data.get("Lx", info.get("Lx"))
        ly = d_data.get("Ly", info.get("Ly"))
        if lx is not None and ly is not None and hasattr(self, 'ax_live'):
            self.ax_live.set_xlim(0, float(lx))
            self.ax_live.set_ylim(0, float(ly))

        bound_xy = d_data.get("bound_xy")
        if bound_xy is not None and hasattr(self, 'scat_bound') and self.scat_bound is not None:
            self.scat_bound.set_offsets(bound_xy)
            self.scat_bound.set_zorder(2)
        elif bound_xy is not None and hasattr(self, 'ax_live'):
            self.scat_bound = self.ax_live.scatter(bound_xy[:, 0], bound_xy[:, 1], s=4, c="k", alpha=0.8, zorder=2)

        if getattr(self, '_current_rendered_scenario_grid_idx', None) != scenario_idx:
            X = d_data.get("X")
            Y = d_data.get("Y")
            V = d_data.get("V")
            if X is not None and Y is not None and V is not None and hasattr(self, 'ax_live'):
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
        scatter_lim = getattr(self, 'scatter_limit', getattr(getattr(self, '_sim_worker', None), '_scatter_limit', 100000))
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

        # 2. Update Divergence and Saddle-Point Potential history graphs
        iters = info.get("iter_history")
        divs = info.get("div_history")
        divs_mid = info.get("div_mid_history")
        ebs = info.get("ebs_history")

        if not iters and info.get("iter", 0) > 0 and (info.get("div") is not None or info.get("div_mid") is not None):
            iters = [info.get("iter", 0)]
            divs = [info.get("div", 0.0)]
            divs_mid = [info.get("div_mid", 0.0)]
            ebs = [info.get("saddle_potential_V", 0.0)]

        if iters and len(iters) > 0:
            self.iter_history = list(iters)
            self.div_history = list(divs) if divs is not None else []
            self.div_mid_history = list(divs_mid) if divs_mid is not None else []
            self.ebs_history = list(ebs) if ebs is not None else []

            if hasattr(self, 'line_div'):
                self.line_div.set_data(self.iter_history, self.div_history)
            if hasattr(self, 'line_div_mid'):
                self.line_div_mid.set_data(self.iter_history, self.div_mid_history)
            if hasattr(self, 'line_ebs'):
                self.line_ebs.set_data(self.iter_history, self.ebs_history)

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

            if hasattr(self, 'ax_ebs'):
                self.ax_ebs.set_xlim(max(0, cur_iter - 400), max(100, cur_iter))
                valid_ebs = [v for v in self.ebs_history if np.isfinite(v)]
                if valid_ebs:
                    min_v = min(valid_ebs)
                    max_v = max(valid_ebs)
                    pad = max(20.0, abs(max_v - min_v) * 0.1)
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

        if hasattr(self, '_update_div_plane_line'):
            self._update_div_plane_line()

        # 3. Update status and metrics labels
        t_us = info.get("t_sim_us", 0.0)
        scenario_iter = info.get("iter", 0)
        div = info.get("div", 0.0)
        div_mid = info.get("div_mid", 0.0)
        i_exit = info.get("exit_ion_current_A", 0.0)
        trans = info.get("transparency", 0.0)
        status_tag = info.get("status", "unknown").upper()
        p_eff = info.get("perveance_effective", 0.0)

        comp_s = info.get("computation_time_s")
        comp_str = f" | Wall Time: {comp_s:.2f} s" if (comp_s is not None and np.isfinite(comp_s)) else ""
        comp_diag_str = f"Computation Time: {comp_s:.2f} s\n" if (comp_s is not None and np.isfinite(comp_s)) else ""

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
            self.lbl_top_sim_time.setText(f"Scenario {scenario_idx} | Plasma: {t_us:.2f} µs | Iter: {scenario_iter:,}{comp_str}")

        if hasattr(self, 'lblPerfPtcls'):
            num_pts = (len(prim_xy) if prim_xy is not None else 0) + (len(cex_xy) if cex_xy is not None else 0)
            num_elec = len(elec_xy) if elec_xy is not None else 0
            self.lblPerfPtcls.setText(f"Display Ptcls (i/e): {num_pts:,} / {num_elec:,} | State: {status_tag}")

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
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            if not hasattr(self, '_sweep_status_dict'):
                self._sweep_status_dict = {}
            self._sweep_status_dict.update(self.parallel_sweep_worker.get_status_dict())
        self._update_sweep_scenario_combo_items()

    def _on_parallel_sweep_finished(self, results):
        self.sweep_results = results
        total_time = time.perf_counter() - getattr(self, 'sweep_start_wall_time', time.perf_counter())
        if self.sweep_config is not None:
            self.sweep_config["total_computation_time_s"] = float(total_time)
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            if not hasattr(self, '_sweep_status_dict'):
                self._sweep_status_dict = {}
            self._sweep_status_dict.update(self.parallel_sweep_worker.get_status_dict())
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
        QMessageBox.critical(self, "Parallel Sweep Error", f"An error occurred during multi-core sweep:\n{error_msg}")

    def _apply_sweep_value(self, param_id, value):
        """Applies a target parameter value to the GUI inputs and simulator."""
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
        scenario_iters = current_iter - getattr(self, 'sweep_scenario_start_iter', 0)

        min_iters = cfg.get("min_iterations", 500)
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

    def _record_sweep_scenario_result(self, scenario_idx, parameter_value, steady_reached, scenario_iters, max_iters_hit=None, comp_time_s=None):
        """Records final steady-state physical observables for the completed scenario."""
        if comp_time_s is None:
            comp_time_s = time.perf_counter() - getattr(self, 'sweep_scenario_start_wall_time', time.perf_counter())

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
            "iter_history": list(self.iter_history),
            "div_history": list(self.div_history),
            "div_mid_history": list(self.div_mid_history),
            "ebs_history": list(self.ebs_history),
            "Lx": float(getattr(sim, 'Lx', 3.0)),
            "Ly": float(getattr(sim, 'Ly', 3.0)),
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
            next_val = values[self.sweep_current_idx]
            self._apply_sweep_value(cfg["param_id"], next_val)
            was_running = getattr(self, 'sim_isRunning', False)
            self.build_domain()
            self.sweep_scenario_start_iter = getattr(self.sim, 'iteration', 0)
            self.sweep_scenario_start_wall_time = time.perf_counter()
            self._sweep_paused_current = False
            if was_running:
                self.toggle_sim()
            if hasattr(self, 'btn_toggle'):
                self.btn_toggle.setText("PAUSE BEAM" if getattr(self, 'sim_isRunning', False) else "2. START BEAM")
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
        if hasattr(self, 'btn_resume_sweep'):
            self.btn_resume_sweep.setVisible(True)
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(True)
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("RESUME BEAM")
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText("Parametric sweep stopped. Select a scenario and click 'Resume with Selected Parameters' or 'RESUME BEAM' to continue.")

        self._update_sweep_scenario_combo_items()

    def finish_sweep(self):
        """Called when all scenarios have completed."""
        self.is_sweep_active = False
        self.sweep_mode_armed = False
        self._sweep_paused_current = False

        total_time = time.perf_counter() - getattr(self, 'sweep_start_wall_time', time.perf_counter())
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
        if hasattr(self, 'btn_resume_sweep'):
            self.btn_resume_sweep.setVisible(True)
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
                    "computation_time_s": time.perf_counter() - getattr(self, 'sweep_scenario_start_wall_time', time.perf_counter()),
                    "t_sim_us": float(getattr(sim, 'time_sim', 0.0)) * 1e6,
                    "min_pot": float(saddle_v) if np.isfinite(saddle_v) else 0.0,
                    "div": float(div_val) if np.isfinite(div_val) else 0.0,
                    "div_mid": float(div_m_val) if np.isfinite(div_m_val) else 0.0,
                    "exit_ion_current_A": float(i_exit) if np.isfinite(i_exit) else 0.0,
                    "perveance_fraction": float(perv_frac) if np.isfinite(perv_frac) else 0.0,
                    "transparency": float(getattr(sim, 'transparency', 0.0)),
                }

        all_results = []
        for idx, val in enumerate(values):
            scen_num = idx + 1
            if scen_num in completed_map:
                all_results.append(completed_map[scen_num])
            elif scen_num in status_dict:
                info = status_dict[scen_num]
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
        self.sim_isRunning = False
        self.sweep_config = None
        self.sweep_results = []
        self._sweep_status_dict = {}

        # 4. Hide all sweep top-bar UI elements
        if hasattr(self, 'lbl_sweep_badge'):
            self.lbl_sweep_badge.setVisible(False)
        if hasattr(self, 'combo_sweep_scenario'):
            self.combo_sweep_scenario.setVisible(False)
        if hasattr(self, 'btn_stop_sweep'):
            self.btn_stop_sweep.setVisible(False)
        if hasattr(self, 'btn_resume_sweep'):
            self.btn_resume_sweep.setVisible(False)
        if hasattr(self, 'btn_exit_sweep'):
            self.btn_exit_sweep.setVisible(False)

        # 5. Restore normal simulation controls and badge
        if hasattr(self, 'btn_toggle'):
            self.btn_toggle.setText("2. START BEAM")
        if hasattr(self, 'update_sim_status_badge'):
            self.update_sim_status_badge("READY")
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText("Exited sweep mode. Restored to normal single simulation.")
