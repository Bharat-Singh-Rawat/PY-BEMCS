"""
Simulation control loop, duration timing, domain build, and worker thread lifecycle.
"""
import time
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QApplication


class SimControlMixin:
    """Mixin providing simulation execution, pause/resume, and domain mesh initialization."""

    def get_simulation_duration(self):
        """Returns the total elapsed wall-clock duration of the simulation in seconds."""
        elapsed = getattr(self, 'sim_wall_elapsed', 0.0)
        if getattr(self, 'sim_isRunning', False) and getattr(self, 'sim_wall_start_time', None) is not None:
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

        if hasattr(self, 'sim') and hasattr(self, 'lbl_top_sim_time'):
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

    def toggle_sim(self):
        if not np.any(self.sim.Ex):
            QMessageBox.warning(self, "Warning", "Build Domain first!")
            return

        self._validate_neut_x()

        if not self.sim_isRunning:
            self.sim.injection_enabled = True
            self.sim_isRunning = True
            self._resume_simulation_timer()
            self.btn_toggle.setText("PAUSE BEAM")
            self._sim_worker.start_sim(self.get_params())
        else:
            self.sim_isRunning = False
            self._sim_worker.pause()
            self._pause_simulation_timer()
            self.btn_toggle.setText("RESUME BEAM")
        self.update_simulation_timer()

    def build_domain(self):
        self._sim_worker.pause()
        if self._sim_worker.isRunning():
            self._sim_worker.wait(5000)
        self._sim_worker._last_damage_version = -1
        self._sim_worker._last_damage_copy = None
        self.sim_isRunning = False
        self.btn_toggle.setText("2. START BEAM")
        self.sim_wall_elapsed = 0.0
        self.sim_wall_start_time = None
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
        if hasattr(self, 'chk_record'):
            self.chk_record.setChecked(False)
            self.chk_record.setText("Record Frames (0)")

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

        if hasattr(self.sim, 'grid_x_ends') and self.sim.grid_x_ends:
            _x_exit = self.sim.grid_x_ends[-1]
            _neut_auto = round(_x_exit + 0.9 * (self.sim.Lx - _x_exit), 3)
        else:
            _neut_auto = round(self.sim.Lx - 0.5, 3)
        _last = getattr(self, '_last_auto_neut_x', None)
        _current_neut = round(self.inputs["neut_x"].value(), 3)
        if _current_neut > self.sim.Lx or _current_neut < 0.0:
            QMessageBox.warning(
                self,
                "Neutralizer Warning",
                f"Neutralizer axial position (x = {_current_neut:.3f} mm) is settled outside the x-domain [0, {self.sim.Lx:.3f} mm].\n\n"
                f"Setting position to x = Lx ({self.sim.Lx:.3f} mm)."
            )
            self.inputs["neut_x"].blockSignals(True)
            self.inputs["neut_x"].setValue(round(self.sim.Lx, 3))
            self.inputs["neut_x"].blockSignals(False)
            self._last_auto_neut_x = round(self.sim.Lx, 3)
            print(f"[Warning] Neutralizer position x={_current_neut:.3f} mm is outside x-domain [0, {self.sim.Lx:.3f} mm]. Settled to x = Lx ({self.sim.Lx:.3f} mm).")
        elif _last is None or abs(_current_neut - _last) <= 0.001:
            self._last_auto_neut_x = _neut_auto
            self.inputs["neut_x"].blockSignals(True)
            self.inputs["neut_x"].setValue(_neut_auto)
            self.inputs["neut_x"].blockSignals(False)
            print(f"[Build Domain] neut_x auto-set to {_neut_auto:.3f} mm  "
                  f"(90% into plume: x_exit={_x_exit if hasattr(self.sim, 'grid_x_ends') and self.sim.grid_x_ends else '?':.3f} mm, Lx={self.sim.Lx:.3f} mm)")

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

    def _on_worker_finished(self):
        """Called when the SimulationWorker thread exits its run loop."""
        if self.sim_isRunning:
            self.sim_isRunning = False
            self._pause_simulation_timer()
            self.btn_toggle.setText("RESUME BEAM")
            self.update_sim_status_badge("STOPPED")
            self.lbl_status.setText("Status: Simulation worker stopped unexpectedly.")
            self.update_simulation_timer()
