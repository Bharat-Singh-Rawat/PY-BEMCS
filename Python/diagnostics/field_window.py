"""
Electric Field & Potential Diagnostics Window for PY-BEMCS.

Visualizes and analyzes the 2D electrostatic field topology in the multi-grid accelerator:
  - Total electric field magnitude |E| = sqrt(Ex^2 + Ey^2)
  - Axial electric field Ex(x, y) (beam acceleration and deceleration barriers)
  - Transverse electric field Ey(x, y) (aperture beamlet focusing and defocusing)
  - Vector quiver field (subsampled directional force arrows)
  - Electric field streamlines (force flux lines)
  - Self-consistent Poisson potential V(x, y) with space charge
  - Static (Laplace) vacuum grid potential V_dc(x, y) and static field |E_static|
  - Space-charge induced potential deformation (V - V_dc)
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    NavigationToolbar = None

from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QGroupBox, QGridLayout, QFrame, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt


class FieldDiagnosticWindow(QDialog):
    """
    Pop-up window for comprehensive electric field and electrostatic potential diagnostics.
    """

    MODES = [
        "|E| Electric Field Magnitude",
        "Ex(x,y) Axial Field (Accel / Decel)",
        "Ey(x,y) Transverse Field (Focusing)",
        "E Vector Field (Quiver Arrows)",
        "E Field Streamlines (Flux Lines)",
        "V(x,y) Total Potential (Self-Consistent)",
        "V_dc(x,y) Static Vacuum Potential",
        "|E_static| Static Vacuum Field",
        "Space-Charge Potential (V - V_dc)",
    ]

    UNITS_E = ["kV/cm", "V/m", "kV/mm"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Electric Field & Potential Diagnostics")
        self.resize(960, 740)
        self.setMinimumSize(800, 580)

        self._cax = None
        self._cbar = None
        self._current_sim = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ── Controls toolbar ──────────────────────────────────────────
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(10)

        ctrl_layout.addWidget(QLabel("Field Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(self.MODES)
        self.combo_mode.currentIndexChanged.connect(self._replot)
        ctrl_layout.addWidget(self.combo_mode)

        ctrl_layout.addWidget(QLabel("E Unit:"))
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(self.UNITS_E)
        self.combo_unit.currentIndexChanged.connect(self._replot)
        ctrl_layout.addWidget(self.combo_unit)

        self.chk_overlay_arrows = QCheckBox("Overlay Arrows")
        self.chk_overlay_arrows.setChecked(False)
        self.chk_overlay_arrows.setToolTip("Superimpose vector arrows on top of the scalar map (in Ex/Ey modes, arrows indicate component direction)")
        self.chk_overlay_arrows.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_overlay_arrows)

        ctrl_layout.addWidget(QLabel("Arrow Step:"))
        self.spin_density = QSpinBox()
        self.spin_density.setRange(2, 30)
        self.spin_density.setValue(6)
        self.spin_density.setSuffix(" cells")
        self.spin_density.setToolTip("Subsampling step size for vector quiver arrows")
        self.spin_density.valueChanged.connect(self._replot)
        ctrl_layout.addWidget(self.spin_density)

        self.chk_contours = QCheckBox("Equipotentials")
        self.chk_contours.setChecked(True)
        self.chk_contours.setToolTip("Overlay subtle equipotential contour lines")
        self.chk_contours.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_contours)

        self.chk_grids = QCheckBox("Grids")
        self.chk_grids.setChecked(True)
        self.chk_grids.setToolTip("Show grid electrode physical boundaries")
        self.chk_grids.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_grids)

        ctrl_layout.addStretch()

        self.chk_live = QCheckBox("Live Update")
        self.chk_live.setChecked(True)
        self.chk_live.setToolTip("Automatically refresh display when simulation steps advance")
        ctrl_layout.addWidget(self.chk_live)

        self.btn_refresh = QPushButton("⟳ Refresh")
        self.btn_refresh.setToolTip("Re-read simulation fields and redraw")
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        ctrl_layout.addWidget(self.btn_refresh)

        main_layout.addLayout(ctrl_layout)

        # ── Matplotlib Canvas ─────────────────────────────────────────
        self.fig = Figure(figsize=(9.0, 4.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(30)
            main_layout.addWidget(self.toolbar)

        main_layout.addWidget(self.canvas, stretch=1)

        # Mouse hover position and field tracker label
        self.lbl_hover = QLabel("Hover over domain to inspect local field values: x = — mm | y = — mm | V = — V | |E| = — kV/cm")
        self.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11px; color: #475569; padding: 2px 4px;")
        main_layout.addWidget(self.lbl_hover)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # ── Analytics Dashboard ───────────────────────────────────────
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.setSpacing(12)

        # Card 1: Peak Electric Fields
        grp_peak = QGroupBox("Peak Electric Fields")
        grp_peak.setStyleSheet(self._group_style())
        lay_peak = QVBoxLayout(grp_peak)
        self.lbl_e_max = QLabel("Peak |E| (total): —")
        self.lbl_e_loc = QLabel("Peak |E| Location: —")
        self.lbl_estat_max = QLabel("Peak |E| (vacuum): —")
        self.lbl_egap_mean = QLabel("Inter-Grid Mean Gap Field: —")
        for lbl in (self.lbl_e_max, self.lbl_e_loc, self.lbl_estat_max, self.lbl_egap_mean):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_peak.addWidget(lbl)
        stats_layout.addWidget(grp_peak)

        # Card 2: Directional Components (Ex, Ey)
        grp_comp = QGroupBox("Axial & Transverse Extrema")
        grp_comp.setStyleSheet(self._group_style())
        lay_comp = QVBoxLayout(grp_comp)
        self.lbl_ex_accel = QLabel("Max Ex (Acceleration): —")
        self.lbl_ex_decel = QLabel("Min Ex (Decel / Barrier): —")
        self.lbl_ey_focus = QLabel("Max |Ey| (Aperture Focus): —")
        self.lbl_e_breakdown = QLabel("Dielectric Margin: —")
        for lbl in (self.lbl_ex_accel, self.lbl_ex_decel, self.lbl_ey_focus, self.lbl_e_breakdown):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_comp.addWidget(lbl)
        stats_layout.addWidget(grp_comp)

        # Card 3: Electrostatic Potential & Space Charge
        grp_pot = QGroupBox("Potential & Space Charge")
        grp_pot.setStyleSheet(self._group_style())
        lay_pot = QVBoxLayout(grp_pot)
        self.lbl_v_plasma = QLabel("Upstream V_plasma: —")
        self.lbl_v_saddle = QLabel("Saddle-Point V_sp: —")
        self.lbl_v_sc_max = QLabel("Max |V - V_dc| (Space Charge): —")
        self.lbl_grid_volts = QLabel("Applied Grid Voltages: —")
        for lbl in (self.lbl_v_plasma, self.lbl_v_saddle, self.lbl_v_sc_max, self.lbl_grid_volts):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_pot.addWidget(lbl)
        stats_layout.addWidget(grp_pot)

        main_layout.addWidget(stats_frame)

    @staticmethod
    def _group_style():
        return """
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 8px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: #ffffff;
            }
        """

    def _on_refresh_clicked(self):
        if self.parent_app and hasattr(self.parent_app, "sim"):
            self.update_plot(self.parent_app.sim)

    def update_plot(self, sim):
        """Update fields from the simulator instance."""
        self._current_sim = sim
        if not self.chk_live.isChecked() and self.isVisible():
            return
        self._replot()

    def _replot(self):
        sim = self._current_sim
        if sim is None or not hasattr(sim, "V") or sim.V is None:
            self.ax.clear()
            self.ax.text(
                0.5, 0.5,
                "No simulation fields initialized yet.\nClick '1. BUILD DOMAIN' to generate fields.",
                ha="center", va="center", transform=self.ax.transAxes,
                color="#64748b", fontsize=12
            )
            self.canvas.draw_idle()
            return

        mode = self.combo_mode.currentText()
        unit_e = self.combo_unit.currentText()
        scale_e = 1.0  # V/m
        unit_label = "V/m"
        if unit_e == "kV/cm":
            scale_e = 1e-5
            unit_label = "kV/cm"
        elif unit_e == "kV/mm":
            scale_e = 1e-6
            unit_label = "kV/mm"

        # Obtain coordinates in mm
        X = getattr(sim, "X", None)
        Y = getattr(sim, "Y", None)
        if X is None or Y is None:
            x_coords = getattr(sim, "x_coords", getattr(sim, "xpts", None))
            y_coords = getattr(sim, "y_coords", getattr(sim, "ypts", None))
            if x_coords is None:
                x_coords = np.linspace(0, sim.Lx, sim.nx)
            if y_coords is None:
                y_coords = np.linspace(0, sim.Ly, sim.ny)
            X, Y = np.meshgrid(x_coords, y_coords)

        # Dynamic Potential V
        V = sim.V.copy()

        # Vacuum Potential Vdc
        Vdc = getattr(sim, "Vdc", None)
        if Vdc is None or Vdc.shape != V.shape:
            Vdc = getattr(sim, "V_fixed", V).copy()

        # Compute dynamic Electric Field
        Ex = getattr(sim, "Ex", None)
        Ey = getattr(sim, "Ey", None)
        if Ex is None or Ey is None or Ex.shape != V.shape:
            y_m = Y[:, 0] * 1e-3
            x_m = X[0, :] * 1e-3
            Ey, Ex = np.gradient(-V, y_m, x_m)

        # Compute static Electric Field
        y_m = Y[:, 0] * 1e-3
        x_m = X[0, :] * 1e-3
        Ey_stat, Ex_stat = np.gradient(-Vdc, y_m, x_m)

        E_mag = np.sqrt(Ex ** 2 + Ey ** 2)
        E_stat_mag = np.sqrt(Ex_stat ** 2 + Ey_stat ** 2)
        V_sc = V - Vdc

        # Clear axes
        self.ax.clear()
        if hasattr(self, "_cax") and self._cax in self.fig.axes:
            self._cax.set_axes_locator(None)
            self.fig.delaxes(self._cax)
            self._cax = None
            self._cbar = None

        data_to_plot = None
        cmap_name = "viridis"
        cbar_title = ""
        is_quiver_only = False
        is_stream_only = False

        if "|E| Electric Field Magnitude" in mode:
            data_to_plot = E_mag * scale_e
            cmap_name = "turbo"
            cbar_title = f"|E| [{unit_label}]"
            self.ax.set_title(f"Total Electric Field Magnitude |E| [{unit_label}]", fontsize=10, fontweight="bold")

        elif "Ex(x,y)" in mode:
            data_to_plot = Ex * scale_e
            cmap_name = "coolwarm"
            cbar_title = f"Ex [{unit_label}]"
            self.ax.set_title(f"Axial Electric Field Ex(x, y) [{unit_label}] (+: Accel, -: Barrier)", fontsize=10, fontweight="bold")

        elif "Ey(x,y)" in mode:
            data_to_plot = Ey * scale_e
            cmap_name = "coolwarm"
            cbar_title = f"Ey [{unit_label}]"
            self.ax.set_title(f"Transverse Electric Field Ey(x, y) [{unit_label}] (+: Outward, -: Focusing)", fontsize=10, fontweight="bold")

        elif "E Vector Field (Quiver Arrows)" in mode:
            data_to_plot = E_mag * scale_e
            cmap_name = "viridis"
            cbar_title = f"|E| [{unit_label}]"
            is_quiver_only = True
            self.ax.set_title(f"Electric Field Vector Topology E(x, y) [{unit_label}]", fontsize=10, fontweight="bold")

        elif "E Field Streamlines" in mode:
            is_stream_only = True
            cmap_name = "turbo"
            self.ax.set_title(f"Electric Field Streamlines (Force Flux Lines)", fontsize=10, fontweight="bold")

        elif "V(x,y) Total Potential" in mode:
            data_to_plot = V
            cmap_name = "plasma"
            cbar_title = "V [V]"
            self.ax.set_title("Self-Consistent Electrostatic Potential V(x, y) [V]", fontsize=10, fontweight="bold")

        elif "V_dc(x,y) Static" in mode:
            data_to_plot = Vdc
            cmap_name = "plasma"
            cbar_title = "V_dc [V]"
            self.ax.set_title("Static (Laplace) Vacuum Potential V_dc(x, y) [V]", fontsize=10, fontweight="bold")

        elif "|E_static|" in mode:
            data_to_plot = E_stat_mag * scale_e
            cmap_name = "turbo"
            cbar_title = f"|E_stat| [{unit_label}]"
            self.ax.set_title(f"Static (Laplace) Vacuum Field Magnitude |E_static| [{unit_label}]", fontsize=10, fontweight="bold")

        elif "Space-Charge Potential" in mode:
            data_to_plot = V_sc
            cmap_name = "bwr"
            cbar_title = "V - V_dc [V]"
            self.ax.set_title("Space-Charge Induced Potential Deformation (V - V_dc) [V]", fontsize=10, fontweight="bold")

        # ── Render scalar color map ──
        if data_to_plot is not None and not is_stream_only:
            # Mask grid interiors for clean aesthetics
            is_bound = getattr(sim, "isBound", None)
            display_data = data_to_plot.copy()
            if is_bound is not None and is_bound.shape == display_data.shape:
                # Mask inside grid electrodes with nan to emphasize vacuum domain
                # except for boundary column x=0
                mask_grids_only = is_bound.copy()
                mask_grids_only[:, 0] = False
                display_data[mask_grids_only] = np.nan

            if cmap_name in ("coolwarm", "bwr"):
                vabs = float(np.nanmax(np.abs(display_data)))
                vmin, vmax = -vabs, vabs
            else:
                vmin = float(np.nanmin(display_data))
                vmax = float(np.nanmax(display_data))
                if vmax <= vmin:
                    vmax = vmin + 1.0

            mesh = self.ax.pcolormesh(
                X, Y, display_data,
                cmap=cmap_name, shading="auto",
                vmin=vmin, vmax=vmax, alpha=0.92
            )

            # Equipotential contour lines
            if self.chk_contours.isChecked():
                try:
                    self.ax.contour(X, Y, V, levels=14, colors="#1e293b", linewidths=0.6, alpha=0.45)
                except Exception:
                    pass

            divider = make_axes_locatable(self.ax)
            self._cax = divider.append_axes("right", size="3%", pad=0.1)
            self._cbar = self.fig.colorbar(mesh, cax=self._cax)
            self._cbar.ax.set_title(cbar_title, fontsize=8, pad=3)

        # ── Streamlines Mode ──
        if is_stream_only:
            try:
                x_1d = X[0, :]
                y_1d = Y[:, 0]

                # Ensure strictly ascending unique coordinates for non-uniform mesh
                unique_x_mask = np.concatenate(([True], np.diff(x_1d) > 1e-12))
                x_clean = x_1d[unique_x_mask]
                Ex_clean = Ex[:, unique_x_mask]
                Ey_clean = Ey[:, unique_x_mask]

                dx_diff = np.diff(x_clean)
                is_uniform = np.allclose(dx_diff, dx_diff[0], rtol=1e-3, atol=1e-6)

                if is_uniform:
                    x_plot, y_plot = x_clean, y_1d
                    Ex_plot, Ey_plot = Ex_clean, Ey_clean
                    Emag_plot = np.sqrt(Ex_plot ** 2 + Ey_plot ** 2) * scale_e
                else:
                    from scipy.interpolate import RegularGridInterpolator
                    nx_uni = max(60, min(240, len(x_clean)))
                    ny_uni = max(40, min(180, len(y_1d)))
                    x_plot = np.linspace(x_clean[0], x_clean[-1], nx_uni)
                    y_plot = np.linspace(y_1d[0], y_1d[-1], ny_uni)

                    interp_ex = RegularGridInterpolator((y_1d, x_clean), Ex_clean, bounds_error=False, fill_value=0.0)
                    interp_ey = RegularGridInterpolator((y_1d, x_clean), Ey_clean, bounds_error=False, fill_value=0.0)

                    YY_uni, XX_uni = np.meshgrid(y_plot, x_plot, indexing="ij")
                    pts = np.stack([YY_uni, XX_uni], axis=-1)
                    Ex_plot = interp_ex(pts)
                    Ey_plot = interp_ey(pts)
                    Emag_plot = np.sqrt(Ex_plot ** 2 + Ey_plot ** 2) * scale_e

                strm = self.ax.streamplot(
                    x_plot, y_plot, Ex_plot, Ey_plot,
                    color=Emag_plot, cmap=cmap_name,
                    density=1.4, linewidth=1.1, arrowsize=1.0
                )
                divider = make_axes_locatable(self.ax)
                self._cax = divider.append_axes("right", size="3%", pad=0.1)
                norm = plt.Normalize(vmin=float(np.nanmin(Emag_plot)), vmax=float(np.nanmax(Emag_plot)))
                sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
                sm.set_array([])
                self._cbar = self.fig.colorbar(sm, cax=self._cax)
                self._cbar.ax.set_title(f"|E| [{unit_label}]", fontsize=8, pad=3)
            except Exception as exc:
                self.ax.text(0.5, 0.5, f"Streamplot error: {exc}", ha="center", va="center", transform=self.ax.transAxes)

        # ── Vector Quiver Arrows (Standalone or Overlay) ──
        if is_quiver_only or (self.chk_overlay_arrows.isChecked() and not is_stream_only):
            step = max(2, int(self.spin_density.value()))
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            Ex_sub = Ex[::step, ::step]
            Ey_sub = Ey[::step, ::step]

            if "Ex(x,y)" in mode:
                # In Ex view, show pure axial component vectors matching Ex color zones
                u_comp = Ex_sub
                v_comp = np.zeros_like(Ey_sub)
                mag_comp = np.abs(u_comp)
            elif "Ey(x,y)" in mode:
                # In Ey view, show pure transverse component vectors matching Ey color zones
                u_comp = np.zeros_like(Ex_sub)
                v_comp = Ey_sub
                mag_comp = np.abs(v_comp)
            else:
                # Full 2D vector field
                u_comp = Ex_sub
                v_comp = Ey_sub
                mag_comp = np.sqrt(u_comp ** 2 + v_comp ** 2)

            norm_mag = np.where(mag_comp > 1e-12, mag_comp, 1.0)
            u = u_comp / norm_mag
            v = v_comp / norm_mag

            arrow_color = "#0f172a" if not is_quiver_only else "#1e293b"
            alpha_val = 0.65 if not is_quiver_only else 0.85
            self.ax.quiver(
                X_sub, Y_sub, u, v,
                color=arrow_color, alpha=alpha_val,
                scale=32, width=0.0022, headwidth=3.5, headlength=4.0
            )

        # ── Draw Grid Electrodes ──
        if self.chk_grids.isChecked() and hasattr(sim, "isBound") and sim.isBound is not None:
            is_bound = sim.isBound.copy()
            is_bound[:, 0] = False  # skip inlet boundary line for electrode plotting
            gy, gx = np.where(is_bound)
            if len(gx) > 0:
                self.ax.scatter(X[gy, gx], Y[gy, gx], s=6, c="#0f172a", alpha=0.95, zorder=5)

        self.ax.set_xlim(0, sim.Lx)
        self.ax.set_ylim(0, sim.Ly)
        self.ax.set_xlabel("Axial Position [mm]")
        self.ax.set_ylabel("Radial Position [mm]")

        self.fig.tight_layout(rect=(0, 0, 0.98, 1))
        self.canvas.draw_idle()

        # ── Update Analytics Dashboard Labels ──
        self._update_analytics(sim, E_mag, Ex, Ey, V, Vdc, scale_e, unit_label)

    def _update_analytics(self, sim, E_mag, Ex, Ey, V, Vdc, scale_e, unit_label):
        max_idx = np.unravel_index(np.argmax(E_mag), E_mag.shape)
        max_e_val = E_mag[max_idx] * scale_e
        x_at_max = sim.X[max_idx] if hasattr(sim, "X") else max_idx[1] * sim.dx
        y_at_max = sim.Y[max_idx] if hasattr(sim, "Y") else max_idx[0] * sim.dy

        # Vacuum field max
        y_m = (sim.Y[:, 0] if hasattr(sim, "Y") else np.linspace(0, sim.Ly, sim.ny)) * 1e-3
        x_m = (sim.X[0, :] if hasattr(sim, "X") else np.linspace(0, sim.Lx, sim.nx)) * 1e-3
        Ey_stat, Ex_stat = np.gradient(-Vdc, y_m, x_m)
        estat_max = np.max(np.sqrt(Ex_stat ** 2 + Ey_stat ** 2)) * scale_e

        self.lbl_e_max.setText(f"Peak |E| (total): {max_e_val:.2f} {unit_label}")
        self.lbl_e_loc.setText(f"Peak |E| Location: ({x_at_max:.2f}, {y_at_max:.2f}) mm")
        self.lbl_estat_max.setText(f"Peak |E| (vacuum): {estat_max:.2f} {unit_label}")

        # Inter-grid gap mean field
        grids = getattr(sim, "grids", [])
        if grids and len(grids) >= 2:
            v_g1 = float(grids[0].get("V", 1000.0))
            v_g2 = float(grids[1].get("V", -200.0))
            gap_mm = float(grids[0].get("gap", 1.0))
            if gap_mm > 0:
                egap_val = (abs(v_g1 - v_g2) / (gap_mm * 1e-3)) * scale_e
                self.lbl_egap_mean.setText(f"Grid 1–2 Gap Mean: {egap_val:.2f} {unit_label} (ΔV={abs(v_g1 - v_g2):.0f}V)")
            else:
                self.lbl_egap_mean.setText("Inter-Grid Mean Gap Field: —")
        else:
            self.lbl_egap_mean.setText("Inter-Grid Mean Gap Field: —")

        # Ex Accel & Decel
        ex_max = np.max(Ex) * scale_e
        ex_min = np.min(Ex) * scale_e
        ey_focus_max = np.max(np.abs(Ey)) * scale_e

        self.lbl_ex_accel.setText(f"Max Ex (Forward Accel): {ex_max:.2f} {unit_label}")
        self.lbl_ex_decel.setText(f"Min Ex (Decel / Barrier): {ex_min:.2f} {unit_label}")
        self.lbl_ey_focus.setText(f"Max |Ey| (Aperture Focus): {ey_focus_max:.2f} {unit_label}")

        # Kilpatrick breakdown threshold check (typically ~100-150 kV/cm in vacuum for mm gaps)
        # 100 kV/cm in selected unit
        breakdown_kv_cm = 100.0
        peak_kv_cm = np.max(E_mag) * 1e-5
        margin_pct = (breakdown_kv_cm - peak_kv_cm) / breakdown_kv_cm * 100.0
        if margin_pct > 25:
            self.lbl_e_breakdown.setText(f"Vacuum Spark Margin: Safe (+{margin_pct:.0f}% to 100 kV/cm)")
            self.lbl_e_breakdown.setStyleSheet("font-size: 11px; color: #16a34a; font-weight: bold;")
        elif margin_pct > 0:
            self.lbl_e_breakdown.setText(f"Vacuum Spark Margin: Moderate (+{margin_pct:.0f}% to 100 kV/cm)")
            self.lbl_e_breakdown.setStyleSheet("font-size: 11px; color: #d97706; font-weight: bold;")
        else:
            self.lbl_e_breakdown.setText(f"Vacuum Spark Margin: EXCEEDED ({peak_kv_cm:.1f} > 100 kV/cm)")
            self.lbl_e_breakdown.setStyleSheet("font-size: 11px; color: #dc2626; font-weight: bold;")

        # Potential card
        v_plasma = float(np.max(V[:, 0])) if V.shape[1] > 0 else 0.0
        v_min_center = float(np.min(V[0, :])) if V.shape[0] > 0 else 0.0
        max_sc = float(np.max(np.abs(V - Vdc)))

        self.lbl_v_plasma.setText(f"Upstream V_plasma: {v_plasma:.1f} V")
        self.lbl_v_saddle.setText(f"Centerline V_min (Barrier): {v_min_center:.1f} V")
        self.lbl_v_sc_max.setText(f"Max |V - V_dc| Space Charge: {max_sc:.2f} V")

        if grids:
            volts_str = " | ".join([f"G{i+1}: {g.get('V', 0.0):.0f}V" for i, g in enumerate(grids)])
            self.lbl_grid_volts.setText(f"Grids: {volts_str}")
        else:
            self.lbl_grid_volts.setText("Grids: —")

    def _on_mouse_move(self, event):
        """Show instantaneous coordinates and field values under mouse cursor."""
        sim = self._current_sim
        if event.inaxes != self.ax or sim is None or not hasattr(sim, "V"):
            return
        x_mm, y_mm = event.xdata, event.ydata
        if x_mm is None or y_mm is None or x_mm < 0 or x_mm > sim.Lx or y_mm < 0 or y_mm > sim.Ly:
            return

        unit_e = self.combo_unit.currentText()
        scale_e = 1e-5 if unit_e == "kV/cm" else (1e-6 if unit_e == "kV/mm" else 1.0)

        # Lookup nearest index
        ix = int(np.clip(round(x_mm / sim.dx), 0, sim.nx - 1))
        iy = int(np.clip(round(y_mm / sim.dy), 0, sim.ny - 1))

        v_val = sim.V[iy, ix] if hasattr(sim, "V") else np.nan
        ex_val = sim.Ex[iy, ix] * scale_e if hasattr(sim, "Ex") else np.nan
        ey_val = sim.Ey[iy, ix] * scale_e if hasattr(sim, "Ey") else np.nan
        emag_val = math.sqrt(ex_val ** 2 + ey_val ** 2) if np.isfinite(ex_val) and np.isfinite(ey_val) else np.nan

        self.lbl_hover.setText(
            f"Cursor: x = {x_mm:.3f} mm | y = {y_mm:.3f} mm  ⟶  "
            f"V = {v_val:.1f} V | Ex = {ex_val:+.2f} {unit_e} | Ey = {ey_val:+.2f} {unit_e} | |E| = {emag_val:.2f} {unit_e}"
        )
