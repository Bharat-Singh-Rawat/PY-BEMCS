"""
Canvas rendering, snapshot processing, and live plot update mixin.
"""
import time
import math
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtCore import QTimer


class RenderingMixin:
    """Mixin managing canvas plotting, static domain rendering, and live snapshot consumption."""

    def draw_static_domain(self, redraw=True):
        self.tempmesh = None
        self.dmg_mesh = None

        self.ax_live.clear()
        self.ax_live.set_title("Ion Beam Extraction & Particle Tracking", fontsize=10)
        self.ax_live.set_xlabel("Axial Position [mm]")
        self.ax_live.set_ylabel("Radial Position [mm]")
        self.ax_live.contourf(self.sim.X, self.sim.Y, self.sim.V, 20, cmap="viridis", alpha=0.4)

        gy, gx = np.where(self.sim.isBound)
        x_pts = getattr(self.sim, 'x_coords', getattr(self.sim, 'xpts', None))
        y_pts = getattr(self.sim, 'y_coords', getattr(self.sim, 'ypts', None))
        if x_pts is not None and y_pts is not None:
            self.ax_live.scatter(x_pts[gx], y_pts[gy], s=4, c="k", alpha=0.8)
        else:
            self.ax_live.scatter(gx * self.sim.dx, gy * self.sim.dy, s=4, c="k", alpha=0.8)

        max_v = max(gw['V'].value() for gw in self.grid_widgets) if self.grid_widgets else 1000.0
        self.scat_prim = self.ax_live.scatter([], [], c=[], s=2, cmap='turbo', vmin=0, vmax=max_v+300, alpha=0.8)
        self.scat_cex = self.ax_live.scatter([], [], c=[], s=3, cmap='turbo', vmin=0, vmax=max_v+300, alpha=1.0)
        self.scat_elec = self.ax_live.scatter([], [], s=1, c='#00FF00', alpha=0.8, zorder=5)

        if hasattr(self, 'cax_live') and self.cax_live in self.fig.axes:
            self.cax_live.set_axes_locator(None)
            self.fig.delaxes(self.cax_live)
        divider = make_axes_locatable(self.ax_live)
        self.cax_live = divider.append_axes("right", size="3%", pad=0.1)

        self.cbar_energy = self.fig.colorbar(self.scat_prim, cax=self.cax_live)
        self.cbar_energy.ax.set_title("[eV]", fontsize=8, pad=3)

        self.ax_live.set_xlim(0, self.sim.Lx)
        self.ax_live.set_ylim(0, self.sim.Ly)
        self.ax_live.set_title("Beam Extraction & Tracking")
        self._update_div_plane_line()

        if self.sim.Tmap is not None and len(self.sim.mask_grids) > 0:
            grid_mask = np.zeros_like(self.sim.isBound, dtype=bool)
            for mg in self.sim.mask_grids:
                if mg.shape == grid_mask.shape:
                    grid_mask |= mg
            TdisplayC = np.where(grid_mask, self.sim.Tmap - 273.15, np.nan)

            self.ax_temp.clear()
            self.ax_temp.set_title("Grid Temp Map (°C)")
            self.ax_temp.set_facecolor("black")

            finite_vals = TdisplayC[np.isfinite(TdisplayC)]
            if finite_vals.size > 0:
                vmin = float(np.min(finite_vals))
                vmax = float(np.max(finite_vals))
                vmin_plot = float(math.floor(vmin))
                vmax_plot = max(vmin_plot + 5.0, float(math.ceil(vmax)))
            else:
                vmin_plot, vmax_plot = 20.0, 35.0

            self.tempmesh = self.ax_temp.pcolormesh(
                self.sim.X, self.sim.Y, TdisplayC,
                cmap="inferno", shading="nearest"
            )
            self.tempmesh.set_clim(vmin_plot, vmax_plot)

            if hasattr(self, 'cax_temp') and self.cax_temp in self.fig.axes:
                self.cax_temp.set_axes_locator(None)
                self.fig.delaxes(self.cax_temp)
            dividert = make_axes_locatable(self.ax_temp)
            self.cax_temp = dividert.append_axes("right", size="5%", pad=0.1)
            self.cbar_temp = self.fig.colorbar(self.tempmesh, cax=self.cax_temp)
            self.cbar_temp.set_label("Temperature (°C)")
            self.cbar_temp.formatter.set_useOffset(False)
            self.cbar_temp.formatter.set_scientific(False)
            self.cbar_temp.update_ticks()
            self.ax_temp.set_xlim(0, self.sim.Lx)
            self.ax_temp.set_ylim(0, self.sim.Ly)

        if self.phys_window and self.phys_window.isVisible():
            self.phys_window.update_plot(self.sim)
        if self.charge_window and self.charge_window.isVisible():
            self.charge_window.update_plot(self.sim)
        if self.perf_window and self.perf_window.isVisible():
            self.perf_window.update_plot(self.sim)
        if getattr(self, 'field_window', None) and self.field_window.isVisible():
            self.field_window.update_plot(self.sim)
        if getattr(self, 'mesh_window', None) and self.mesh_window.isVisible():
            self.mesh_window.update_plot(self.sim, params=self.get_params())
        if getattr(self, 'substep_window', None) and self.substep_window.isVisible():
            self.substep_window.update_plot(self.sim)
        if getattr(self, 'steady_state_window', None) and self.steady_state_window.isVisible():
            self.steady_state_window.update_plot(self.sim, self)

        v_init_saddle = getattr(self.sim, 'saddle_point_potential', getattr(self.sim, 'min_pot', None))
        if v_init_saddle is not None:
            self.iter_history = [0]
            self.ebs_history = [v_init_saddle]
            self.div_history = [0.0]
            self.div_mid_history = [0.0]
            self.line_ebs.set_data(self.iter_history, self.ebs_history)
            self.line_div.set_data(self.iter_history, self.div_history)
            if hasattr(self, 'line_div_mid'):
                self.line_div_mid.set_data(self.iter_history, self.div_mid_history)
            pad = max(20.0, abs(v_init_saddle) * 0.1)
            self.ax_ebs.set_xlim(0, 100)
            self.ax_ebs.set_ylim(v_init_saddle - pad, v_init_saddle + pad)
            self.ax_div.set_xlim(0, 100)
            self.ax_div.set_ylim(0, 45)
        else:
            self.iter_history = []
            self.ebs_history = []
            self.div_history = []
            self.div_mid_history = []
            self.line_ebs.set_data([], [])
            self.line_div.set_data([], [])
            if hasattr(self, 'line_div_mid'):
                self.line_div_mid.set_data([], [])
            self.ax_ebs.set_xlim(0, 100)
            self.ax_div.set_xlim(0, 100)
            self.ax_div.set_ylim(0, 45)
        self._update_div_plot_visibility()

        self.line_groove.set_data([], [])
        self.ax_groove.set_xlim(0, max(1.0, float(getattr(self.sim, 'Ly', 1.0))))
        self.ax_groove.set_ylim(1.0, 0.0)
        self.ax_groove.set_title("Accel Grid Erosion Profile — Ready")

        self.ax_dmg.clear()
        self.ax_dmg.set_title("Sputter Damage Map", fontsize=10)
        self.ax_dmg.set_xlabel("Axial Position [mm]", fontsize=8)
        self.ax_dmg.set_ylabel("Radial Position [mm]", fontsize=8)
        gy, gx = np.where(self.sim.isBound)
        x_pts = getattr(self.sim, 'x_coords', getattr(self.sim, 'xpts', None))
        y_pts = getattr(self.sim, 'y_coords', getattr(self.sim, 'ypts', None))
        if x_pts is not None and y_pts is not None:
            self.ax_dmg.scatter(x_pts[gx], y_pts[gy], s=2, c="grey", alpha=0.5)
        else:
            self.ax_dmg.scatter(gx * self.sim.dx, gy * self.sim.dy, s=2, c="grey", alpha=0.5)
        self.ax_dmg.set_xlim(0, self.sim.Lx)
        self.ax_dmg.set_ylim(0, self.sim.Ly)

        self._reposition_div_menu_btn()
        if redraw:
            self.canvas.draw_idle()

    def _on_step_result(self, snap):
        """Process a single simulation step result on the GUI thread."""
        if self._gui_busy:
            self._pending_snap = snap
            return

        if self._pending_snap is not None and snap is not self._pending_snap:
            snap = self._pending_snap
        self._pending_snap = None
        self._gui_busy = True

        _t0 = time.perf_counter()
        try:
            self._process_snap(snap)
        finally:
            self._last_gui_ms = (time.perf_counter() - _t0) * 1000.0
            self._gui_busy = False

            if self._pending_snap is not None:
                QTimer.singleShot(0, lambda: self._on_step_result(self._pending_snap))

    def _process_snap(self, snap):
        """Inner body of _on_step_result."""
        params           = snap['params']
        remeshed         = snap['remeshed']
        min_pot          = snap['min_pot']
        current_div      = snap['current_div']
        current_div_mid  = snap.get('current_div_mid', np.nan)
        T_grids          = snap['T_grids']
        trans_last_frame = snap['trans_last_frame']
        transparency     = snap['transparency']

        inj_time    = params.get("inj_time", 0.0)
        inj_limited = inj_time > 0.0

        if inj_limited and (not snap['injection_enabled']) and (not snap['has_active_particles']):
            self._sim_worker.pause()
            self._pause_simulation_timer()
            self.sim_isRunning = False
            self.btn_toggle.setText("2. START BEAM")
            self.update_sim_status_badge("STOPPED")
            self.lbl_status.setText("Status: Injection completed and all particles removed. Simulation stopped.")
            self.update_simulation_timer()
            return

        t_sim = snap['iteration'] * snap['dt']

        self.lblTime.setText(f"t_sim: {t_sim * 1e6:.2f} us")
        self.lblTransparency.setText(
            f"Transparency tot: {transparency:.3f}\n"
            f"Transparency frame: {trans_last_frame:.3f}\n"
            f"Exit vx mean: {snap['exit_vx_mean']: .2e} m/s\n"
            f"Exit |v| mean: {snap['exit_v_mean']: .2e} m/s\n"
            f"Exit E mean: {snap['exit_energy_mean_eV']: .1f} eV\n"
            f"Exit count step: {snap['exit_count_step']}\n"
            f"Exit I_ion step: {snap['exit_ion_current_step'] * 1e3:.3f} mA"
        )
        pm = snap['perf_monitor']
        if pm and pm.history:
            last_diag = pm.history[-1]
            self.lblPerfPtcls.setText(f"Ptcls (i/e):  {last_diag.num_ions:,} / {last_diag.num_electrons:,}")
            self.lblPerfStep.setText(f"Step / RAM:   {last_diag.wall_time_ms:.0f} ms / {last_diag.memory_rss_mb:.0f} MB")
        else:
            self.lblPerfPtcls.setText(f"Ptcls (i/e):  {snap['num_p']:,} / {snap['num_e']:,}")
            self.lblPerfStep.setText("Step / RAM:   —")

        if remeshed:
            self.lbl_status.setText("Domain Remeshed (Thermal or Erosion)!")
            self.draw_static_domain(redraw=False)

        if self.scat_prim is not None and self.scat_cex is not None:
            self.scat_prim.set_offsets(snap['prim_xy'])
            self.scat_cex.set_offsets(snap['cex_xy'])
            self.scat_elec.set_offsets(snap['elec_xy'])

            e_prim_eV = snap['e_prim_eV']
            e_cex_eV  = snap['e_cex_eV']

            energy_min = float('inf')
            energy_max = float('-inf')

            if len(e_prim_eV) > 0:
                self.scat_prim.set_array(e_prim_eV)
                energy_min = min(energy_min, float(np.min(e_prim_eV)))
                energy_max = max(energy_max, float(np.max(e_prim_eV)))

            if len(e_cex_eV) > 0:
                self.scat_cex.set_array(e_cex_eV)
                energy_min = min(energy_min, float(np.min(e_cex_eV)))
                energy_max = max(energy_max, float(np.max(e_cex_eV)))

            if energy_min != float('inf'):
                if energy_max <= energy_min:
                    energy_max = energy_min + 1.0
                self.scat_prim.set_clim(energy_min, energy_max)
                self.scat_cex.set_clim(energy_min, energy_max)

            if self.iedf_window and self.iedf_window.isVisible():
                max_v = max([g["V"].value() for g in self.grid_widgets]) if self.grid_widgets else 1000.0
                self.iedf_window.update_histogram(
                    snap['p_vx'], snap['p_vy'], snap['p_isCEX'],
                    snap['e_x'],  snap['e_vx'], snap['e_vy'],
                    snap['m_ion'], snap['m_e'], snap['q'], max_v
                )

        now = time.perf_counter()
        diag_due = (now - self._last_diag_t) * 1000.0 >= self._DIAG_MIN_INTERVAL_MS
        if diag_due:
            self._last_diag_t = now
            if self.ppc_window and self.ppc_window.isVisible():
                self.ppc_window.update_plot(self.sim)
            if self.phys_window and self.phys_window.isVisible():
                self.phys_window.update_plot(self.sim)
            if self.charge_window and self.charge_window.isVisible():
                self.charge_window.update_plot(self.sim)
            if self.perf_window and self.perf_window.isVisible():
                self.perf_window.update_plot(self.sim)
            if getattr(self, 'field_window', None) and self.field_window.isVisible():
                self.field_window.update_plot(self.sim)
            if getattr(self, 'mesh_window', None) and self.mesh_window.isVisible():
                self.mesh_window.update_plot(self.sim, params=self.get_params())
            if getattr(self, 'substep_window', None) and self.substep_window.isVisible():
                self.substep_window.update_plot(self.sim)
            if getattr(self, 'steady_state_window', None) and self.steady_state_window.isVisible():
                self.steady_state_window.update_plot(self.sim, self)

        if self.chk_track_ptcls.isChecked():
            ptcl_data = self.sim.get_particle_kinematics()
            if ptcl_data.size > 0:
                self.tracking_buffer.append(ptcl_data)

        self.iter_history.append(snap['iteration'])
        if not hasattr(self, 'substep_history_iter'):
            self.substep_history_iter = []
            self.substep_history_ion = []
            self.substep_history_e = []
        self.substep_history_iter.append(snap['iteration'])
        self.substep_history_ion.append(snap.get('n_sub_ion', 1))
        self.substep_history_e.append(snap.get('n_sub_e', 1))
        self.ebs_history.append(min_pot)
        self.div_history.append(current_div)
        self.div_mid_history.append(current_div_mid)
        self.time_history.append(t_sim)
        self.transparency_history.append(transparency)
        self.transparency3_history.append(trans_last_frame)
        self.ion_current_exit_history.append(snap['exit_ion_current_step'])
        self.ion_current_exit_avg_history.append(snap['exit_ion_current_avg'])
        self.transmitted_ions_history.append(int(round(snap['transmitted_ions_step'])))
        self.active_cells_history.append(snap['total_active_cells'])
        self.low_ppc_cells_history.append(snap['low_ppc_cells'])

        for i, T in enumerate(T_grids):
            self.T_histories[i].append(T)

        self.line_ebs.set_data(self.iter_history, self.ebs_history)
        self.line_div.set_data(self.iter_history, self.div_history)
        if hasattr(self, 'line_div_mid'):
            self.line_div_mid.set_data(self.iter_history, self.div_mid_history)

        self.ax_ebs.set_xlim(max(0, snap['iteration'] - 400), max(100, snap['iteration']))
        self._update_div_scroll_range(snap['iteration'])

        if len(self.ebs_history) > 0:
            y_min = min(self.ebs_history)
            y_max = max(self.ebs_history)
            if y_max <= y_min:
                y_max = y_min + 1.0
            pad = max(5.0, 0.1 * (y_max - y_min))
            self.ax_ebs.set_ylim(y_min - pad, y_max + pad)

        self._update_div_plot_visibility()

        groove_idx = 1 if len(snap['mask_grids']) > 1 else 0
        groove_face = "downstream"
        y_mm, depth_um = self.sim.get_groove_profile(
            groove_idx,
            thresh=self.inputs["Thresh"].value(),
            face=groove_face
        )
        if y_mm.size > 0:
            self.line_groove.set_data(y_mm, depth_um)
            self.ax_groove.set_xlim(0, float(y_mm.max()))
            dmax = float(depth_um.max()) if depth_um.size > 0 else 0.0
            self.ax_groove.set_ylim(max(dmax * 1.1, 1.0), 0.0)
            self.ax_groove.set_title(
                f"Accel Grid Erosion Profile — {groove_face} face (Grid {groove_idx + 1})"
            )

        Tmap = snap['Tmap']
        mask_grids = snap['mask_grids']
        if Tmap is not None and len(mask_grids) > 0 and Tmap.shape == snap['X'].shape:
            grid_mask = np.zeros_like(snap['isBound'], dtype=bool)
            for mg in mask_grids:
                if mg.shape == grid_mask.shape:
                    grid_mask |= mg

            T_display_C = np.where(grid_mask, Tmap - 273.15, np.nan)

            if getattr(self, "tempmesh", None) is None or self.tempmesh.get_array().size != T_display_C.size:
                self.ax_temp.clear()
                self.ax_temp.set_title("Grid Temp Map (°C)")
                self.ax_temp.set_facecolor("black")

                self.tempmesh = self.ax_temp.pcolormesh(
                    snap['X'], snap['Y'], T_display_C,
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

            self.ax_temp.set_xlim(0, snap['Lx'])
            self.ax_temp.set_ylim(0, snap['Ly'])

        damage_map = snap['damage_map']
        if damage_map is not None and damage_map.shape == snap['X'].shape:
            if getattr(self, "dmg_mesh", None) is None or self.dmg_mesh.get_array().size != damage_map.size:
                self.ax_dmg.clear()
                self.ax_dmg.set_title("Sputter Damage Map", fontsize=10)
                self.ax_dmg.set_xlabel("Axial Position [mm]", fontsize=8)
                self.ax_dmg.set_ylabel("r [mm]", fontsize=8)
                self.dmg_mesh = self.ax_dmg.pcolormesh(
                    snap['X'], snap['Y'], damage_map,
                    cmap="hot", shading="nearest"
                )
                gy, gx = np.where(snap['isBound'])
                x_pts = snap['x_coords']
                y_pts = snap['y_coords']
                if x_pts is not None and y_pts is not None:
                    self.ax_dmg.scatter(x_pts[gx], y_pts[gy], s=2, c="grey", alpha=0.5)
                else:
                    self.ax_dmg.scatter(gx * snap['dx'], gy * snap['dy'], s=2, c="grey", alpha=0.5)
                self.ax_dmg.set_xlim(0, snap['Lx'])
                self.ax_dmg.set_ylim(0, snap['Ly'])
            else:
                self.dmg_mesh.set_array(damage_map.ravel())
                dmg_max = float(np.max(damage_map))
                if dmg_max > 0:
                    self.dmg_mesh.set_clim(0, dmg_max)

        self.lbl_status.setText(
            f"Ions {snap['num_p']} e- {snap['num_e']} Iter {snap['iteration']}"
        )

        t_str = " | ".join([f"G{i+1}: {T - 273.15:.1f}°C" for i, T in enumerate(T_grids)])
        self.lbl_temp.setText("Grid Temps: " + t_str)

        self.canvas.draw_idle()

        if self.chk_record.isChecked():
            self.recorded_frames.append(self.canvas.grab())
            self.chk_record.setText(f"Record Frames ({len(self.recorded_frames)})")

        if self.sim_isRunning:
            p = self.get_params()
            p['_last_gui_ms'] = self._last_gui_ms
            self._sim_worker.update_params(p)

        if getattr(self, 'is_sweep_active', False):
            self.check_sweep_steady_state()

    def _update_div_legend(self):
        """Build/update interactive clickable legend with [x] / [ ] checkmarks for ax_div."""
        if not hasattr(self, 'ax_div'):
            return
        if not hasattr(self, '_div_legend_map'):
            self._div_legend_map = {}
        if not hasattr(self, '_div_curve_visible'):
            self._div_curve_visible = {'grid': True, 'mid': True}

        self._div_legend_map.clear()
        method_str = getattr(self, 'div_method', '95%')
        tag_map = {
            "95%": "θ95",
            "90%": "θ90",
            "RMS (Twiss)": "θ_Twiss",
            "RMS (twiss)": "θ_Twiss",
            "Direct RMS": "θ_RMS",
            "FWHM": "θ_FWHM/2",
        }
        tag = tag_map.get(method_str, f"θ{method_str.replace('%', '')}")
        plane_x = getattr(self, 'div_plane_x', None)
        if plane_x is not None:
            plane_label = f"Plane x={plane_x:.2f}mm ({tag})"
        else:
            plane_label = f"Mid Plume ({tag})"

        quantities = [
            ('grid', f"Grid Exit ({tag})", '#1f77b4', '-', 1.8),
            ('mid', plane_label, '#e67e22', '-', 1.8),
        ]
        legend_handles = []
        legend_labels = []

        for key, name, color, ls, lw in quantities:
            active = self._div_curve_visible.get(key, True)
            marker_str = "[x]" if active else "[ ]"
            alpha = 1.0 if active else 0.3
            label = f"{marker_str} {name}"

            handle = Line2D(
                [0], [0], color=color, linestyle=ls, lw=lw,
                alpha=alpha
            )
            legend_handles.append(handle)
            legend_labels.append(label)

        leg = self.ax_div.legend(
            legend_handles, legend_labels,
            fontsize=7.5, loc='upper right', framealpha=0.85, labelspacing=0.25
        )

        if leg is not None:
            for leg_line, leg_text, (key, _, _, _, _) in zip(leg.get_lines(), leg.get_texts(), quantities):
                if not self._div_curve_visible.get(key, True):
                    leg_text.set_alpha(0.35)
                leg_line.set_picker(True)
                leg_line.set_pickradius(6)
                leg_text.set_picker(True)

                self._div_legend_map[leg_line] = key
                self._div_legend_map[leg_text] = key

    def _on_legend_pick(self, event):
        """Handle click on divergence graph legend items to toggle curve visibility."""
        artist = event.artist
        key = getattr(self, '_div_legend_map', {}).get(artist)
        if key is not None and key in getattr(self, '_div_curve_visible', {}):
            self._div_curve_visible[key] = not self._div_curve_visible[key]
            self._update_div_plot_visibility()
            if hasattr(self, 'canvas'):
                self.canvas.draw_idle()

    def _update_div_plot_visibility(self):
        """Apply visibility states and recompute y-limits for ax_div."""
        if not hasattr(self, '_div_curve_visible'):
            self._div_curve_visible = {'grid': True, 'mid': True}

        show_grid = self._div_curve_visible.get('grid', True)
        show_mid = self._div_curve_visible.get('mid', True)

        if hasattr(self, 'line_div'):
            self.line_div.set_visible(show_grid)
        if hasattr(self, 'line_div_mid'):
            self.line_div_mid.set_visible(show_mid)
        if hasattr(self, 'line_div_plane') and self.line_div_plane is not None:
            self.line_div_plane.set_visible(show_mid)

        visible_divs = []
        if show_grid and hasattr(self, 'div_history') and len(self.div_history) > 0:
            visible_divs.extend([d for d in self.div_history if np.isfinite(d)])
        if show_mid and hasattr(self, 'div_mid_history') and len(self.div_mid_history) > 0:
            visible_divs.extend([d for d in self.div_mid_history if np.isfinite(d)])

        if hasattr(self, 'ax_div'):
            if len(visible_divs) > 0:
                div_max = max(visible_divs)
                self.ax_div.set_ylim(0, max(45, div_max * 1.1))
            else:
                self.ax_div.set_ylim(0, 45)

            self._update_div_legend()

    def set_divergence_method(self, method_name: str):
        """Set the active divergence evaluation method (e.g. '95%', '90%', 'RMS (Twiss)', 'Direct RMS', 'FWHM')."""
        self.div_method = method_name
        methods = getattr(self, 'DIVERGENCE_METHODS', {
            "95%": 95.0,
            "90%": 90.0,
            "RMS (Twiss)": "RMS (Twiss)",
            "Direct RMS": "Direct RMS",
            "FWHM": "FWHM",
        })
        val = methods.get(method_name)
        if isinstance(val, (int, float)):
            self.div_percentile = float(val)
        elif method_name == "90%":
            self.div_percentile = 90.0
        elif method_name == "95%":
            self.div_percentile = 95.0

        if hasattr(self, 'sim'):
            self.sim.div_method = self.div_method
            self.sim.div_percentile = getattr(self, 'div_percentile', 95.0)

        # Update action checkmarks in menu
        if hasattr(self, '_div_method_actions'):
            for name, act in self._div_method_actions.items():
                act.setChecked(name.strip().lower() == method_name.strip().lower())

        # Notify active simulation worker if running
        if getattr(self, 'sim_isRunning', False) and hasattr(self, '_sim_worker'):
            p = self.get_params()
            self._sim_worker.update_params(p)

        # Re-render legend to update label tag (e.g. θ95 vs θ90 vs θ_Twiss)
        self._update_div_legend()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(f"Divergence evaluation method set to {method_name}")

    def set_divergence_plane(self, plane_x: float):
        """Set the axial position (x) in mm for beam divergence evaluation."""
        self.div_plane_x = float(plane_x)
        if hasattr(self, 'sim'):
            self.sim.div_plane_x = self.div_plane_x

        # Notify active simulation worker if running
        if getattr(self, 'sim_isRunning', False) and hasattr(self, '_sim_worker'):
            p = self.get_params()
            self._sim_worker.update_params(p)

        # Update dotted plane indicator in ax_live
        self._update_div_plane_line()

        # Re-render legend to update label with new plane position
        self._update_div_legend()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(f"Divergence evaluation plane set to x = {self.div_plane_x:.3f} mm")

    def _update_div_plane_line(self):
        """Draw or update the dotted vertical line showing the divergence evaluation plane in ax_live."""
        if not hasattr(self, 'ax_live'):
            return

        plane_x = getattr(self, 'div_plane_x', None)
        if plane_x is None:
            if hasattr(self, 'get_divergence_plane_limits'):
                x_min, x_max = self.get_divergence_plane_limits()
                plane_x = round(x_min + 0.5 * (x_max - x_min), 3)
            elif hasattr(self, 'sim') and hasattr(self.sim, 'Lx') and self.sim.Lx > 0:
                plane_x = self.sim.Lx * 0.75
            else:
                plane_x = None

        if plane_x is None:
            return

        show_plane = True
        if hasattr(self, '_div_curve_visible'):
            show_plane = self._div_curve_visible.get('mid', True)

        if getattr(self, 'line_div_plane', None) is not None and self.line_div_plane in self.ax_live.lines:
            self.line_div_plane.set_xdata([plane_x, plane_x])
            self.line_div_plane.set_visible(show_plane)
        else:
            self.line_div_plane = self.ax_live.axvline(
                x=plane_x,
                color="#e67e22",
                linestyle=":",
                linewidth=1.8,
                alpha=0.85,
                zorder=6
            )
            self.line_div_plane.set_visible(show_plane)

    def _reposition_div_menu_btn(self):
        """Position the dropdown arrow button immediately aside the 'Beam Divergence' title."""
        if not hasattr(self, 'btn_div_menu') or not hasattr(self, 'ax_div'):
            return

        try:
            renderer = self.canvas.get_renderer()
            bbox = self.ax_div.title.get_window_extent(renderer)
            if bbox.width > 0 and bbox.height > 0:
                canvas_h = self.canvas.height()
                btn_w = self.btn_div_menu.width()
                btn_h = self.btn_div_menu.height()
                x = int(bbox.x1 + 6)
                y = int(canvas_h - bbox.y1 + (bbox.height - btn_h) / 2)
                x = max(0, min(x, self.canvas.width() - btn_w - 2))
                y = max(0, min(y, canvas_h - btn_h - 2))
                self.btn_div_menu.move(x, y)
                self.btn_div_menu.show()
                self.btn_div_menu.raise_()
                return
        except Exception:
            pass

        # Fallback using normalized axes position
        try:
            pos = self.ax_div.get_position()
            canvas_w = self.canvas.width()
            canvas_h = self.canvas.height()
            center_x = (pos.x0 + pos.x1) * 0.5 * canvas_w
            x = int(center_x + 55)
            y = int((1.0 - pos.y1) * canvas_h - 18)
            self.btn_div_menu.move(x, y)
            self.btn_div_menu.show()
            self.btn_div_menu.raise_()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Divergence history scrollbar helpers
    # ------------------------------------------------------------------

    def _reposition_div_scrollbar(self):
        """Position the floating scrollbar and Live button below the 'Iteration' xlabel of ax_div."""
        if not hasattr(self, 'div_scrollbar') or not hasattr(self, 'ax_div'):
            return

        try:
            renderer = self.canvas.get_renderer()
            ax_bbox = self.ax_div.get_window_extent(renderer)   # in display pixels (y from bottom)

            canvas_h = self.canvas.height()
            canvas_w = self.canvas.width()

            # ax_div left/right edges in Qt coords (y flipped: Qt y=0 is top)
            x_left  = int(ax_bbox.x0)
            x_right = int(ax_bbox.x1)

            # Find the bottom of the "Iteration" xlabel in display coordinates (so we don't cover tick values)
            y_bottom_disp = None
            try:
                xlabel_bbox = self.ax_div.xaxis.label.get_window_extent(renderer)
                if xlabel_bbox is not None and xlabel_bbox.height > 0:
                    y_bottom_disp = xlabel_bbox.y0
            except Exception:
                pass

            if y_bottom_disp is None:
                try:
                    tight_bbox = self.ax_div.get_tightbbox(renderer)
                    if tight_bbox is not None:
                        y_bottom_disp = tight_bbox.y0
                except Exception:
                    pass

            if y_bottom_disp is None:
                # Fallback: estimate 25px below ax_div bottom to clear tick values and xlabel
                y_bottom_disp = ax_bbox.y0 - 25

            # Bottom of "Iteration" xlabel in Qt coords (Qt y=0 is top)
            y_bottom_qt = int(canvas_h - y_bottom_disp)

            bar_h   = self.div_scrollbar.height()   # 12 px fixed
            btn_w   = self.btn_div_live.width()      # 48 px fixed
            gap     = 2                              # px gap below xlabel

            total_w = x_right - x_left
            scroll_w = max(20, total_w - btn_w - gap)

            y_pos = y_bottom_qt + gap

            # Clamp to canvas bounds
            y_pos  = max(0, min(y_pos, canvas_h - bar_h))
            x_left = max(0, min(x_left, canvas_w - total_w))

            self.div_scrollbar.setGeometry(x_left, y_pos, scroll_w, bar_h)
            self.btn_div_live.setGeometry(x_left + scroll_w + gap, y_pos, btn_w, bar_h)

            self.div_scrollbar.show()
            self.btn_div_live.show()
            self.div_scrollbar.raise_()
            self.btn_div_live.raise_()
        except Exception:
            pass


    _DIV_WINDOW = 400   # number of iterations visible in the plot window

    def _update_div_scroll_range(self, current_iter: int):
        """Update scrollbar range and, if in live mode, auto-follow the latest data."""
        if not hasattr(self, 'div_scrollbar'):
            return

        n = len(self.iter_history)
        window = self._DIV_WINDOW

        # Maximum value the scrollbar can reach: the leftmost start of the live window
        # Each unit of scroll corresponds to 1 iteration offset from the beginning.
        max_scroll = max(0, n - window)

        # Block signals to prevent recursive _on_div_scroll while we update the range
        self.div_scrollbar.blockSignals(True)
        self.div_scrollbar.setMaximum(max_scroll)
        self.div_scrollbar.setPageStep(max(1, window // 4))

        live_mode = getattr(self, '_div_live_mode', True)
        if live_mode:
            # Snap to rightmost (latest) position
            self.div_scrollbar.setValue(max_scroll)
            x_end = max(100, current_iter)
            x_start = max(0, x_end - window)
            self.ax_div.set_xlim(x_start, x_end)
        # else: leave xlim as-is (user is scrolling)

        self.div_scrollbar.blockSignals(False)

    def _on_div_scroll(self, value: int):
        """Called when the user drags the history scrollbar."""
        if not hasattr(self, 'iter_history') or len(self.iter_history) == 0:
            return

        # Moving away from the rightmost position disables live mode
        live_mode = getattr(self, '_div_live_mode', True)
        max_val = self.div_scrollbar.maximum()

        if value < max_val:
            # User scrolled back in history — switch to history mode
            self._div_live_mode = False
            if hasattr(self, 'btn_div_live'):
                self.btn_div_live.blockSignals(True)
                self.btn_div_live.setChecked(False)
                self.btn_div_live.setText("▶ Live")
                self.btn_div_live.blockSignals(False)
        else:
            # Scrolled back to the end — restore live mode
            self._div_live_mode = True
            if hasattr(self, 'btn_div_live'):
                self.btn_div_live.blockSignals(True)
                self.btn_div_live.setChecked(True)
                self.btn_div_live.setText("📌 Live")
                self.btn_div_live.blockSignals(False)

        window = self._DIV_WINDOW
        iters = self.iter_history

        if len(iters) == 0:
            return

        # Map scroll position to iteration range
        # value = index of the first data-point visible (0-based)
        idx_start = value
        idx_end = min(len(iters) - 1, idx_start + window)

        x_start = iters[idx_start] if idx_start < len(iters) else 0
        x_end = iters[idx_end] if idx_end < len(iters) else iters[-1]
        if x_end <= x_start:
            x_end = x_start + window

        self.ax_div.set_xlim(x_start, x_end)
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _on_div_live_toggled(self, checked: bool):
        """Called when the user clicks the Live/History toggle button."""
        self._div_live_mode = checked
        if hasattr(self, 'btn_div_live'):
            if checked:
                self.btn_div_live.setText("📌 Live")
                # Jump scrollbar to the end immediately
                if hasattr(self, 'div_scrollbar'):
                    self.div_scrollbar.setValue(self.div_scrollbar.maximum())
            else:
                self.btn_div_live.setText("▶ Live")
