"""
Simulation control loop, duration timing, domain build, and worker thread lifecycle.
"""
import copy
import time
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QApplication


class SimControlMixin:
    """Mixin providing simulation execution, pause/resume, and domain mesh initialization."""

    def get_simulation_duration(self):
        """Returns the total elapsed wall-clock duration of the simulation in seconds."""
        elapsed = getattr(self, 'sim_wall_elapsed', 0.0)
        is_running = (
            self.is_simulation_running()
            if hasattr(self, 'is_simulation_running')
            else getattr(self, 'sim_isRunning', False)
        )
        if is_running and getattr(self, 'sim_wall_start_time', None) is not None:
            elapsed += time.perf_counter() - self.sim_wall_start_time
        return elapsed

    def update_simulation_timer(self):
        """Refreshes the top simulation duration stopwatch and plasma timestamp."""
        if not hasattr(self, 'lbl_sim_timer'):
            return
        elapsed = self.get_simulation_duration()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        tenths = int((elapsed - int(elapsed)) * 10)
        self.lbl_sim_timer.setText(f"⏱ Simulation Duration: {hours:02d}:{minutes:02d}:{seconds:02d}.{tenths:01d}")

        is_sweep = (
            getattr(self, 'is_sweep_active', False)
            or (hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None and self.parallel_sweep_worker.isRunning())
            or getattr(self, 'sweep_config', None) is not None
        )
        if not is_sweep and hasattr(self, 'sim') and hasattr(self, 'lbl_top_sim_time'):
            t_sim_us = getattr(self.sim, 'iteration', 0) * getattr(self.sim, 'dt', 0.0) * 1e6
            it_count = getattr(self.sim, 'iteration', 0)
            self.lbl_top_sim_time.setText(f"Plasma Time: {t_sim_us:.2f} µs | Iter: {it_count}")

    def reset_simulation_timer(self):
        """Resets the simulation duration timer back to zero."""
        self.sim_wall_elapsed = 0.0
        if getattr(self, 'sim_isRunning', False):
            self.sim_wall_start_time = time.perf_counter()
        else:
            self.sim_wall_start_time = None
        self.update_simulation_timer()

    def update_sim_status_badge(self, status):
        """Updates the visual status indicator in the top header."""
        if not hasattr(self, 'lbl_sim_status_badge'):
            return
        st = status.upper()
        self.lbl_sim_status_badge.setText(st)
        if st == "RUNNING":
            self.lbl_sim_status_badge.setStyleSheet(
                "background-color: #dcfce7; color: #15803d; font-weight: bold; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        elif st == "PAUSED":
            self.lbl_sim_status_badge.setStyleSheet(
                "background-color: #fef3c7; color: #b45309; font-weight: bold; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        elif st == "READY":
            self.lbl_sim_status_badge.setStyleSheet(
                "background-color: #e0f2fe; color: #0369a1; font-weight: bold; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )
        else:
            self.lbl_sim_status_badge.setStyleSheet(
                "background-color: #e2e8f0; color: #475569; font-weight: bold; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            )

    def _resume_simulation_timer(self):
        if getattr(self, 'sim_wall_start_time', None) is None:
            self.sim_wall_start_time = time.perf_counter()
        self.update_sim_status_badge("RUNNING")

    def _pause_simulation_timer(self):
        if getattr(self, 'sim_wall_start_time', None) is not None:
            self.sim_wall_elapsed = getattr(self, 'sim_wall_elapsed', 0.0) + (time.perf_counter() - self.sim_wall_start_time)
            self.sim_wall_start_time = None
        self.update_sim_status_badge("PAUSED")

    def _extract_domain_signature(self):
        """Returns a signature of physical parameters affecting domain mesh/fields."""
        sig = []
        if hasattr(self, 'inputs'):
            for k in sorted(self.inputs.keys()):
                try:
                    sig.append((k, round(float(self.inputs[k].value()), 6)))
                except Exception:
                    pass
        if hasattr(self, 'grid_widgets'):
            for idx, gw in enumerate(self.grid_widgets):
                for k in ("V", "t", "gap", "r", "cham"):
                    if k in gw:
                        try:
                            sig.append((f"grid_{idx}_{k}", round(float(gw[k].value()), 6)))
                        except Exception:
                            pass
        if hasattr(self, 'adv_params') and isinstance(self.adv_params, dict):
            for k in ("Lx", "Ly"):
                if k in self.adv_params:
                    try:
                        sig.append((k, round(float(self.adv_params[k]), 6)))
                    except Exception:
                        pass
        return tuple(sig)

    def _is_domain_dirty(self):
        """Returns True if user-facing domain parameters have changed since last build_domain()."""
        if not hasattr(self, 'sim') or not hasattr(self.sim, 'Ex') or self.sim.Ex is None or not np.any(self.sim.Ex):
            return True
        last_sig = getattr(self, '_built_params_snapshot', None)
        if last_sig is None:
            return False
        current_sig = self._extract_domain_signature()
        return current_sig != last_sig

    def toggle_sim(self):
        # If sweep is active or parallel worker is running, toggle pause/resume for the scenario selected from tent menu
        if getattr(self, 'is_sweep_active', False) or (hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None and self.parallel_sweep_worker.isRunning()):
            if hasattr(self, 'toggle_sweep_scenario_pause'):
                self.toggle_sweep_scenario_pause()
                return

        if not np.any(self.sim.Ex):
            QMessageBox.warning(self, "Warning", "Build Domain first!")
            return

        self._validate_neut_x()

        if not self.sim_isRunning:
            self.sim.injection_enabled = True
            self.sim_isRunning = True
            self._sweep_paused_current = False
            self._resume_simulation_timer()
            self.btn_toggle.setText("PAUSE BEAM")
            self.update_sim_status_badge("RUNNING")
            self._sim_worker.start_sim(self.get_params(), run_id=getattr(self, '_sim_run_id', 0))
        else:
            self.sim_isRunning = False
            self._sweep_paused_current = True
            self.btn_toggle.setText("RESUME BEAM")
            self._sim_worker.pause()
            self._pause_simulation_timer()
            self.update_sim_status_badge("PAUSED")
        self.update_simulation_timer()
        if hasattr(self, '_update_sweep_scenario_combo_items') and (getattr(self, 'sweep_mode_armed', False) or getattr(self, 'is_sweep_active', False)):
            self._update_sweep_scenario_combo_items()

    def _validate_grids_geometry(self):
        """Return validation error strings for all active grids (1..N) before domain build."""
        grid_widgets = getattr(self, 'grid_widgets', [])
        n_grids = len(grid_widgets)
        if n_grids == 0:
            return ["At least one grid must be defined."]

        errors = []
        for idx, gw in enumerate(grid_widgets, start=1):
            t_val = float(gw["t"].value())
            gap_val = float(gw["gap"].value())
            r_val = float(gw["r"].value())
            cham_val = float(gw["cham"].value())

            if t_val <= 0.0:
                errors.append(f"Grid {idx}: Thickness (t = {t_val:.3g} mm) must be > 0 mm.")
            if r_val <= 0.0:
                errors.append(f"Grid {idx}: Aperture Radius (r = {r_val:.3g} mm) must be > 0 mm.")
            if idx < n_grids and gap_val <= 0.0:
                errors.append(
                    f"Grid {idx}: Gap to Grid {idx + 1} (gap = {gap_val:.3g} mm) must be > 0 mm."
                )
            elif gap_val < 0.0:
                errors.append(f"Grid {idx}: Downstream gap (gap = {gap_val:.3g} mm) cannot be negative.")
            if cham_val < 0.0 or cham_val > 45.0:
                errors.append(f"Grid {idx}: Chamfer ({cham_val:.3g}°) must be between 0° and 45°.")
        return errors

    def build_domain(self, force=False):
        if not force and self.is_simulation_running():
            msg = "Stop the sweep before building"
            print(f"[Build Domain] {msg}")
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(f"Status: {msg}")
            QMessageBox.warning(self, "Warning", msg)
            return
        if not force:
            grid_errors = self._validate_grids_geometry()
            if grid_errors:
                msg = (
                    "Cannot build domain due to invalid grid geometry:\n\n"
                    + "\n".join(f"• {e}" for e in grid_errors)
                )
                print(f"[Build Domain] Invalid grid geometry: {'; '.join(grid_errors)}")
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText("Status: Invalid grid geometry — check grid parameters.")
                QMessageBox.warning(self, "Invalid Grid Geometry", msg)
                return
        if hasattr(self, 'parallel_sweep_worker') and self.parallel_sweep_worker is not None:
            worker = self.parallel_sweep_worker
            self.parallel_sweep_worker = None
            try:
                worker.cancel()
                worker.wait(2000)
            except Exception:
                pass

        self._sim_worker.stop(wait_ms=3000)
        if hasattr(self, 'sim') and self.sim is not None:
            self.sim._stop_requested = False
        self._sim_run_id = getattr(self, '_sim_run_id', 0) + 1
        self._pending_snap = None
        self._sim_worker._last_damage_version = -1
        self._sim_worker._last_damage_copy = None
        self.sim_isRunning = False
        self.btn_toggle.setText("2. START BEAM")
        if not getattr(self, 'is_sweep_active', False):
            self.sim_wall_elapsed = 0.0
            self.sim_wall_start_time = None
        elif getattr(self, 'sim_wall_start_time', None) is None:
            self.sim_wall_start_time = time.perf_counter()
        self.update_sim_status_badge("READY")
        self.update_simulation_timer()
        self.iter_history.clear()
        self.ebs_history.clear()
        self.div_history.clear()
        self.div_mid_history.clear()
        self.time_history.clear()
        self.transparency_history.clear()
        self.transparency3_history.clear()
        self.ion_current_exit_history.clear()
        self.ion_current_exit_avg_history.clear()
        self.transmitted_ions_history.clear()
        self.active_cells_history.clear()
        self.low_ppc_cells_history.clear()
        self.lblTime.setText("t_sim:        — us")
        self.T_histories = {i: [] for i in range(len(self.grid_widgets))}
        self.tracking_buffer.clear()
        self.recorded_frames.clear()
        if hasattr(self, 'substep_history_iter'):
            self.substep_history_iter.clear()
            self.substep_history_ion.clear()
            self.substep_history_e.clear()
        if hasattr(self, 'chk_record'):
            self.chk_record.setChecked(False)
            self.chk_record.setText("Record Frames (0)")
        if hasattr(self, 'div_scrollbar'):
            self.div_scrollbar.blockSignals(True)
            self.div_scrollbar.setMaximum(0)
            self.div_scrollbar.setValue(0)
            self.div_scrollbar.blockSignals(False)
            self._div_live_mode = True

        self.lbl_status.setText("Building Multi-Grid Domain...")
        QApplication.processEvents()

        _n0   = self.inputs["n0_plasma"].value()
        _Te   = self.inputs["Te_up"].value()
        _eps0 = 8.854e-12
        _q    = 1.602e-19
        _lambda_D_m  = np.sqrt(_eps0 * _Te * _q / (_n0 * _q**2))
        _lambda_D_mm = _lambda_D_m * 1e3
        _ratio = float(self.adv_params.get(r"\deltax/debye_length", self.adv_params.get("dx_over_debye", 0.8)))
        _dxy_mm = _ratio * _lambda_D_mm
        self.sim.dx = _dxy_mm
        self.sim.dy = _dxy_mm

        auto_Lx, auto_Ly = self.compute_grid_domain_size()
        if not getattr(self, '_user_overrode_Lx', False):
            self.adv_params["Lx"] = auto_Lx
        if not getattr(self, '_user_overrode_Ly', False):
            self.adv_params["Ly"] = auto_Ly

        self.apply_advanced_settings_to_sim()
        try:
            self.sim.build_domain(self.get_params())
        except Exception as exc:
            self.lbl_status.setText(f"Status: Domain build error: {exc}")
            QMessageBox.critical(self, "Domain Build Error", f"Failed to build domain:\n\n{exc}")
            return
        self.sim.enable_perf_monitor(log_every=50)

        if not getattr(self, '_user_overrode_Lx', False):
            self.adv_params["Lx"] = round(self.sim.Lx, 3)
        if not getattr(self, '_user_overrode_Ly', False):
            self.adv_params["Ly"] = round(self.sim.Ly, 3)

        gap_used = self.sim.upstream_gap_mm
        self.inputs["upstream_gap_mm"].blockSignals(True)
        self.inputs["upstream_gap_mm"].setValue(round(gap_used, 3))
        self.inputs["upstream_gap_mm"].blockSignals(False)

        if hasattr(self.sim, 'hole_boundary_dist_mm') and "hole_boundary_dist_mm" in self.inputs:
            hdist_used = self.sim.hole_boundary_dist_mm
            self.inputs["hole_boundary_dist_mm"].blockSignals(True)
            self.inputs["hole_boundary_dist_mm"].setValue(round(hdist_used, 3))
            self.inputs["hole_boundary_dist_mm"].blockSignals(False)

        if hasattr(self.sim, 'grid_x_ends') and self.sim.grid_x_ends:
            _x_exit = self.sim.grid_x_ends[-1]
            _neut_auto = round(_x_exit + 0.9 * (self.sim.Lx - _x_exit), 3)
        else:
            _neut_auto = round(self.sim.Lx - 0.5, 3)
        _last = getattr(self, '_last_auto_neut_x', None)
        _current_neut = round(self.inputs["neut_x"].value(), 3)
        _is_auto_neut = (_last is None) or (_last >= 0.0 and abs(_current_neut - _last) <= 0.001)
        if _is_auto_neut:
            self._last_auto_neut_x = _neut_auto
            self.inputs["neut_x"].blockSignals(True)
            self.inputs["neut_x"].setValue(_neut_auto)
            self.inputs["neut_x"].blockSignals(False)
            print(f"[Build Domain] neut_x auto-set to {_neut_auto:.3f} mm  "
                  f"(90% into plume: x_exit={_x_exit if hasattr(self.sim, 'grid_x_ends') and self.sim.grid_x_ends else '?':.3f} mm, Lx={self.sim.Lx:.3f} mm)")
        elif _current_neut > self.sim.Lx or _current_neut < 0.0:
            if (
                not getattr(self, 'is_sweep_active', False)
                and not getattr(self, 'sweep_mode_armed', False)
                and getattr(self, 'isVisible', lambda: False)()
            ):
                QMessageBox.warning(
                    self,
                    "Neutralizer Warning",
                    f"Neutralizer axial position (x = {_current_neut:.3f} mm) is settled outside the x-domain [0, {self.sim.Lx:.3f} mm].\n\n"
                    f"Setting position to x = Lx ({self.sim.Lx:.3f} mm)."
                )
            else:
                print(f"[Domain Build] Auto-clamped neut_x from {_current_neut:.3f} mm to {self.sim.Lx:.3f} mm")
            self.inputs["neut_x"].blockSignals(True)
            self.inputs["neut_x"].setValue(round(self.sim.Lx, 3))
            self.inputs["neut_x"].blockSignals(False)
            self._last_auto_neut_x = -999.0

        self.draw_static_domain()

        cfg_name = getattr(self, "current_config_name", "config.json")
        dx_min = getattr(self.sim, 'dx_min', self.sim.dx)
        dx_max = getattr(self.sim, 'dx_max', self.sim.dx)
        mesh_mode_tag = "JSON 3-Zone" if self.adv_params.get("use_json_mesh_zones", False) else f"Auto {len(getattr(self.sim, 'zone_configs', []))}-Zone"
        if abs(dx_max - dx_min) > 1e-6:
            dx_str = f"dx=[{dx_min:.4f}..{dx_max:.4f}], dy={self.sim.dy:.4f} mm"
        else:
            dx_str = f"dx=dy={self.sim.dx:.4f} mm"
        self.lbl_status.setText(
            f"Domain Ready [{cfg_name}]\n{mesh_mode_tag} | {dx_str}"
        )
        self.lbl_temp.setText(
            "Grid Temps: " + " | ".join([f"G{i+1}: 26°C" for i in range(len(self.grid_widgets))])
        )
        if hasattr(self, 'lbl_material'):
            self.lbl_material.setText(f"Grid Material: {self.mat_name}")

        if getattr(self, 'mesh_window', None) is not None and self.mesh_window.isVisible():
            self.mesh_window.update_plot(self.sim, params=self.get_params())
        if getattr(self, 'substep_window', None) is not None and self.substep_window.isVisible():
            self.substep_window.update_plot(self.sim)

        self._built_params_snapshot = self._extract_domain_signature()
        self._grid_count_modified_since_build = False
        self._sweep_paused_current = False
        self._sweep_domain_built = True

        is_sweep = getattr(self, 'sweep_mode_armed', False) or getattr(self, 'is_sweep_active', False)
        if hasattr(self, 'lbl_build_domain_info'):
            self.lbl_build_domain_info.setVisible(is_sweep)
        if hasattr(self, 'lbl_beam_info'):
            self.lbl_beam_info.setVisible(is_sweep)

        if getattr(self, 'sweep_mode_armed', False) and hasattr(self, 'lbl_sweep_badge'):
            self._sweep_base_params = copy.deepcopy(self.get_params())
            self._scenario_domain_cache = {}
            cfg = getattr(self, 'sweep_config', {})
            total_scenarios = len(cfg.get("values", []))
            param_id = cfg.get("param_id", "")
            param_unit = cfg.get("param_unit", "")
            cur_idx = getattr(self, 'sweep_current_idx', 0)
            values = cfg.get("values", [])
            cur_val = values[cur_idx] if len(values) > cur_idx else 0.0
            try:
                from ..dialogs import format_sci_value
            except Exception:
                from gui.dialogs import format_sci_value
            cur_val_str = (
                format_sci_value(cur_val)
                if (param_id in ("n0_plasma", "n0") or abs(cur_val) >= 1e4 or (0 < abs(cur_val) <= 1e-3))
                else f"{cur_val:.3g}"
            )
            self.lbl_sweep_badge.setVisible(True)
            self.lbl_sweep_badge.setText(
                f"⚡ Sweep mode armed [{cur_idx + 1}/{total_scenarios} | {cur_val_str} {param_unit}]"
            )
            if hasattr(self, '_update_sweep_action_button_state'):
                self._update_sweep_action_button_state()

    def _on_worker_finished(self):
        """Called when the SimulationWorker thread exits its run loop."""
        if self.sim_isRunning:
            self.sim_isRunning = False
            self._pause_simulation_timer()
            if getattr(self, 'is_sweep_active', False):
                self._sweep_paused_current = True
                self.btn_toggle.setText("Resume")
            else:
                self.btn_toggle.setText("RESUME BEAM")
            self.update_sim_status_badge("STOPPED")
            self.lbl_status.setText("Status: Simulation worker stopped unexpectedly.")
            self.update_simulation_timer()
