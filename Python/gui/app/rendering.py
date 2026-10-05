"""
Canvas rendering, snapshot processing, and live plot update mixin.
"""
import time
import math
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtCore import QTimer


class RenderingMixin:
    """Mixin managing canvas plotting, static domain rendering, and live snapshot consumption."""

    def draw_static_domain(self):
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
        self.canvas.draw_idle()

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
            self.cbartemp = self.cbar_temp
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

        v_init_saddle = getattr(self.sim, 'saddle_point_potential', getattr(self.sim, 'min_pot', None))
        if v_init_saddle is not None:
            self.iter_history = [0]
            self.ebs_history = [v_init_saddle]
            self.div_history = [0.0]
            self.line_ebs.set_data(self.iter_history, self.ebs_history)
            self.line_div.set_data(self.iter_history, self.div_history)
            pad = max(20.0, abs(v_init_saddle) * 0.1)
            self.ax_ebs.set_xlim(0, 100)
            self.ax_ebs.set_ylim(v_init_saddle - pad, v_init_saddle + pad)
            self.ax_div.set_xlim(0, 100)
            self.ax_div.set_ylim(0, 45)
        else:
            self.iter_history = []
            self.ebs_history = []
            self.div_history = []
            self.line_ebs.set_data([], [])
            self.line_div.set_data([], [])
            self.ax_ebs.set_xlim(0, 100)
            self.ax_div.set_xlim(0, 100)
            self.ax_div.set_ylim(0, 45)

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
            self.draw_static_domain()

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

        if self.chk_track_ptcls.isChecked():
            ptcl_data = self.sim.get_particle_kinematics()
            if ptcl_data.size > 0:
                self.tracking_buffer.append(ptcl_data)

        self.iter_history.append(snap['iteration'])
        self.ebs_history.append(min_pot)
        self.div_history.append(current_div)
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

        self.ax_ebs.set_xlim(max(0, snap['iteration'] - 400), max(100, snap['iteration']))
        self.ax_div.set_xlim(max(0, snap['iteration'] - 400), max(100, snap['iteration']))

        if len(self.ebs_history) > 0:
            y_min = min(self.ebs_history)
            y_max = max(self.ebs_history)
            if y_max <= y_min:
                y_max = y_min + 1.0
            pad = max(5.0, 0.1 * (y_max - y_min))
            self.ax_ebs.set_ylim(y_min - pad, y_max + pad)

        finite_div = [d for d in self.div_history if np.isfinite(d)]
        if len(finite_div) > 0:
            div_max = max(finite_div)
            self.ax_div.set_ylim(0, max(45, div_max * 1.1))
        else:
            self.ax_div.set_ylim(0, 45)

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
                self.cbartemp = self.cbar_temp
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
