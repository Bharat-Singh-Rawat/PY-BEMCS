"""
Density Diagnostic Window for PY-BEMCS.

Visualizes 2D spatial particle number densities with a logarithmic color scale
and 1D radial density profiles at the two divergence evaluation planes:
  - Plane 1: Accelerator Grid Exit face
  - Plane 2: Second Divergence Plane (User-selected / Mid-Plume)

Supports species selection:
  - Total Ion Density (Primary + CEX)
  - Primary Ion Density
  - CEX Ion Density
  - Electron Density (Kinetic + Boltzmann fluid)
"""

import numpy as np
import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton
    )


class DensityWindow(QWidget):
    """
    Diagnostic window displaying:
      1. 2D spatial number density map on a logarithmic color scale.
      2. 1D radial density profile at Divergence Plane 1 (Grid Exit).
      3. 1D radial density profile at Divergence Plane 2 (2nd Plane / Mid-Plume).
    """

    SPECIES_OPTIONS = [
        ("Total Ion Density", "ions"),
        ("Primary Ion Density", "primary"),
        ("CEX Ion Density", "cex"),
        ("Electron Density", "electrons"),
    ]

    COLORMAPS = ["turbo", "viridis", "inferno", "plasma", "magma"]

    def __init__(self, parent=None):
        super().__init__()
        self.parent_app = parent
        self.setWindowTitle("Plasma & Particle Density Diagnostics")
        self.setGeometry(100, 100, 1020, 760)
        self.setMinimumSize(860, 600)

        # Accumulator state for time-averaged mode
        self._accum_maps = {}
        self._accum_count = 0
        self._last_sim_step = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Controls toolbar ──────────────────────────────────────────
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(10)

        ctrl_layout.addWidget(QLabel("<b>Species:</b>"))
        self.combo_species = QComboBox()
        for label, _ in self.SPECIES_OPTIONS:
            self.combo_species.addItem(label)
        self.combo_species.currentIndexChanged.connect(self._on_species_change)
        ctrl_layout.addWidget(self.combo_species)

        ctrl_layout.addWidget(QLabel("<b>Mode:</b>"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Instantaneous Density (Current Step)",
            "Time-Averaged Density (Accumulator)",
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_change)
        ctrl_layout.addWidget(self.combo_mode)

        self.btn_reset_accum = QPushButton("Reset Accumulator")
        self.btn_reset_accum.clicked.connect(self._reset_accum)
        ctrl_layout.addWidget(self.btn_reset_accum)

        ctrl_layout.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(self.COLORMAPS)
        self.combo_cmap.setCurrentText("turbo")
        self.combo_cmap.currentIndexChanged.connect(self._replot_from_cache)
        ctrl_layout.addWidget(self.combo_cmap)

        self.chk_planes = QCheckBox("Show Divergence Planes")
        self.chk_planes.setChecked(True)
        self.chk_planes.stateChanged.connect(self._replot_from_cache)
        ctrl_layout.addWidget(self.chk_planes)

        self.chk_overlay_all = QCheckBox("Overlay All Species in 1D")
        self.chk_overlay_all.setChecked(False)
        self.chk_overlay_all.setToolTip(
            "When checked, radial profile plots show Total Ions, CEX, and Electrons simultaneously."
        )
        self.chk_overlay_all.stateChanged.connect(self._replot_from_cache)
        ctrl_layout.addWidget(self.chk_overlay_all)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # ── Summary Statistics Banner ─────────────────────────────────
        self.lbl_stats = QLabel("Density Diagnostic: Ready.")
        self.lbl_stats.setStyleSheet(
            "font-family: monospace; font-size: 11px; background-color: #f4f6f9; "
            "padding: 6px; border: 1px solid #d0d7de; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_stats)

        # ── Matplotlib Figure Layout ──────────────────────────────────
        # Top: 2D Spatial Map + Colorbar
        # Bottom: Two 1D Radial Profile plots side-by-side
        self.fig = plt.figure(figsize=(10.0, 7.2))
        gs = self.fig.add_gridspec(
            2, 2,
            height_ratios=[1.25, 1.0],
            width_ratios=[1.0, 1.0],
            hspace=0.32,
            wspace=0.22,
            left=0.08,
            right=0.92,
            top=0.94,
            bottom=0.08
        )

        # Row 0: 2D contour plot spanning both columns with dedicated colorbar on right
        self.ax_2d = self.fig.add_subplot(gs[0, :])
        # Position colorbar explicitly to avoid axes divider issues
        self.cax = self.fig.add_axes([0.93, 0.56, 0.015, 0.38])

        # Row 1: Two 1D radial plots
        self.ax_p1 = self.fig.add_subplot(gs[1, 0])
        self.ax_p2 = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self._cached_sim = None

    def _get_active_species_key(self) -> str:
        idx = self.combo_species.currentIndex()
        if 0 <= idx < len(self.SPECIES_OPTIONS):
            return self.SPECIES_OPTIONS[idx][1]
        return "ions"

    def _on_species_change(self):
        self._replot_from_cache()

    def _on_mode_change(self):
        self._replot_from_cache()

    def reset_accumulator(self):
        """Resets the time-averaged density spatial accumulator."""
        self._accum_maps.clear()
        self._accum_count = 0
        self._last_sim_step = -1
        self._last_mesh_shape = None

    def _reset_accum(self):
        self.reset_accumulator()
        self._replot_from_cache()

    def _replot_from_cache(self):
        if self._cached_sim is not None:
            self.update_plot(self._cached_sim, is_new_step=False)

    def update_plot(self, sim, is_new_step: bool = True):
        """Update and render the 2D density map and the two 1D radial profile graphs."""
        if sim is None or not hasattr(sim, 'nx') or sim.nx <= 0:
            return

        self._cached_sim = sim
        current_step = getattr(sim, 'iteration', -1)
        cur_shape = (int(getattr(sim, 'ny', 0)), int(getattr(sim, 'nx', 0)))

        # Automatically reset accumulator if mesh dimensions change (e.g. sweep scenario)
        # or if simulation iteration jumped backwards / restarted
        if getattr(self, '_last_mesh_shape', None) != cur_shape or current_step < self._last_sim_step:
            self._accum_maps.clear()
            self._accum_count = 0
            self._last_mesh_shape = cur_shape

        # Check if step advanced for accumulator
        if is_new_step and current_step != self._last_sim_step:
            self._last_sim_step = current_step
            self._accum_count += 1
            for _, sp_key in self.SPECIES_OPTIONS:
                sp_map = sim.get_density_map(species=sp_key)
                if sp_key not in self._accum_maps or self._accum_maps[sp_key].shape != sp_map.shape:
                    self._accum_maps[sp_key] = np.zeros_like(sp_map, dtype=np.float64)
                self._accum_maps[sp_key] += sp_map

        # Selected mode and species
        mode_text = self.combo_mode.currentText()
        is_accum = "Time-Averaged" in mode_text
        active_key = self._get_active_species_key()
        active_label = self.combo_species.currentText()

        # Retrieve 2D density map for active species
        if (
            is_accum
            and self._accum_count > 0
            and active_key in self._accum_maps
            and self._accum_maps[active_key].shape == cur_shape
        ):
            density_2d = self._accum_maps[active_key] / float(self._accum_count)
            step_info = f"Time-Averaged ({self._accum_count} steps)"
        else:
            density_2d = sim.get_density_map(species=active_key)
            step_info = f"Instantaneous (Step {getattr(sim, 'iteration', 0)})"

        # Divergence planes
        if hasattr(sim, 'get_divergence_plane_locations'):
            x1, x2 = sim.get_divergence_plane_locations()
        else:
            x1 = float(sim.grid_x_ends[-1]) if hasattr(sim, 'grid_x_ends') and sim.grid_x_ends else sim.Lx * 0.5
            x2 = getattr(sim, 'div_plane_x', None)
            if x2 is None:
                x2 = x1 + 0.5 * (sim.Lx - x1)

        # Statistics computation
        valid_mask = (density_2d > 0.0)
        n_pos = int(np.count_nonzero(valid_mask))
        if n_pos > 0:
            max_dens = float(np.max(density_2d))
            mean_dens = float(np.mean(density_2d[valid_mask]))
        else:
            max_dens = 0.0
            mean_dens = 0.0

        # Plane profiles
        y_pts, prof_p1 = sim.get_radial_density_profile(x1, species=active_key, density_map=density_2d)
        _, prof_p2 = sim.get_radial_density_profile(x2, species=active_key, density_map=density_2d)

        p1_max = float(np.max(prof_p1)) if len(prof_p1) > 0 else 0.0
        p2_max = float(np.max(prof_p2)) if len(prof_p2) > 0 else 0.0

        self.lbl_stats.setText(
            f"Species: {active_label} [{step_info}]  |  "
            f"Peak: {max_dens:.2e} m⁻³  |  Mean Active: {mean_dens:.2e} m⁻³  |  "
            f"Exit (x={x1:.2f}mm) Peak: {p1_max:.2e} m⁻³  |  "
            f"2nd Plane (x={x2:.2f}mm) Peak: {p2_max:.2e} m⁻³"
        )

        # ── 1. Clear All Axes ─────────────────────────────────────────
        self.ax_2d.clear()
        self.cax.clear()
        self.ax_p1.clear()
        self.ax_p2.clear()

        # ── 2. Render 2D Density Map ──────────────────────────────────
        cmap_name = self.combo_cmap.currentText()
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color='#0d1117')

        # Logarithmic normalization bounds (e.g. 5 orders of magnitude below peak)
        vmax = max_dens if max_dens > 0.0 else 1e16
        vmin = max(1e11, vmax * 1e-5)
        if vmax <= vmin:
            vmax = vmin * 10.0

        norm = LogNorm(vmin=vmin, vmax=vmax)
        density_masked = np.ma.masked_less_equal(density_2d, 0.0)

        X = getattr(sim, 'X', None)
        Y = getattr(sim, 'Y', None)
        if X is not None and Y is not None and X.shape == density_masked.shape:
            im = self.ax_2d.pcolormesh(X, Y, density_masked, cmap=cmap, norm=norm, shading='nearest')
        else:
            extent = [0, sim.Lx, 0, sim.Ly]
            im = self.ax_2d.imshow(
                density_masked, origin='lower', extent=extent,
                cmap=cmap, norm=norm, aspect='auto'
            )

        self.ax_2d.set_title(
            f"2D {active_label} [m⁻³] — {step_info}",
            fontsize=10, fontweight='bold', pad=6
        )
        self.ax_2d.set_xlabel("Axial Position x [mm]", fontsize=9)
        self.ax_2d.set_ylabel("Radial Position y [mm]", fontsize=9)
        self.ax_2d.set_xlim(0, sim.Lx)
        self.ax_2d.set_ylim(0, sim.Ly)

        # Grid electrode bodies overlay
        if hasattr(sim, 'isBound') and sim.isBound is not None and np.any(sim.isBound):
            gy, gx = np.where(sim.isBound)
            x_pts = getattr(sim, 'x_coords', getattr(sim, 'xpts', None))
            y_pts_grid = getattr(sim, 'y_coords', getattr(sim, 'ypts', None))
            if x_pts is not None and y_pts_grid is not None:
                valid_m = (gx < len(x_pts)) & (gy < len(y_pts_grid))
                if np.any(valid_m):
                    self.ax_2d.scatter(x_pts[gx[valid_m]], y_pts_grid[gy[valid_m]], s=2.5, c='silver', alpha=0.85)
            else:
                self.ax_2d.scatter(gx * sim.dx, gy * sim.dy, s=2.5, c='silver', alpha=0.85)

        # Divergence Planes vertical markers
        if self.chk_planes.isChecked():
            self.ax_2d.axvline(x=x1, color='#00d2d3', linestyle='--', linewidth=1.5, alpha=0.9)
            self.ax_2d.text(
                x1, sim.Ly * 0.92, f" Plane 1 (Exit)\n x={x1:.2f}mm",
                color='#00d2d3', fontsize=7.5, fontweight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#1e272e', alpha=0.75, edgecolor='none')
            )

            self.ax_2d.axvline(x=x2, color='#ff9f43', linestyle='--', linewidth=1.5, alpha=0.9)
            self.ax_2d.text(
                x2, sim.Ly * 0.92, f" Plane 2 (Div)\n x={x2:.2f}mm",
                color='#ff9f43', fontsize=7.5, fontweight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#1e272e', alpha=0.75, edgecolor='none')
            )

        # Colorbar
        cbar = self.fig.colorbar(im, cax=self.cax)
        cbar.set_label("Number Density [m⁻³]", fontsize=8.5)

        # ── 3. Render 1D Radial Profile Graphs ────────────────────────
        species_styles = {
            "ions": ("Total Ions", "#1f77b4", "-"),
            "primary": ("Primary Ions", "#3498db", "--"),
            "cex": ("CEX Ions", "#e74c3c", "-"),
            "electrons": ("Electrons", "#2ecc71", "-."),
        }

        overlay_all = self.chk_overlay_all.isChecked()
        species_to_plot = list(species_styles.keys()) if overlay_all else [active_key]

        floor_density = 1e10
        y_max_plot = max(p1_max, p2_max, 1e16) * 2.0
        y_min_plot = max(floor_density, y_max_plot * 1e-6)

        # --- Subplot 1: Grid Exit Plane (x1) ---
        for sp_k in species_to_plot:
            label, color, ls = species_styles.get(sp_k, (sp_k, "blue", "-"))
            if is_accum and self._accum_count > 0 and sp_k in self._accum_maps and self._accum_maps[sp_k].shape == cur_shape:
                sp_dens = self._accum_maps[sp_k] / float(self._accum_count)
            else:
                sp_dens = None
            _, prof = sim.get_radial_density_profile(x1, species=sp_k, density_map=sp_dens)
            prof_clipped = np.maximum(prof, floor_density)
            self.ax_p1.plot(y_pts, prof_clipped, label=label, color=color, linestyle=ls, linewidth=1.7)

        self.ax_p1.set_yscale('log')
        self.ax_p1.set_ylim(bottom=y_min_plot, top=y_max_plot)
        self.ax_p1.set_xlim(0, sim.Ly)
        self.ax_p1.set_title(f"Plane 1 (Grid Exit): x = {x1:.2f} mm", fontsize=9.5, fontweight='bold')
        self.ax_p1.set_xlabel("Radial Coordinate y [mm]", fontsize=9)
        self.ax_p1.set_ylabel("Density n [m⁻³]", fontsize=9)
        self.ax_p1.grid(True, which='both', linestyle=':', alpha=0.55)
        self.ax_p1.legend(fontsize=8, loc='upper right', framealpha=0.85)

        # --- Subplot 2: Second Divergence Plane (x2) ---
        for sp_k in species_to_plot:
            label, color, ls = species_styles.get(sp_k, (sp_k, "blue", "-"))
            if is_accum and self._accum_count > 0 and sp_k in self._accum_maps and self._accum_maps[sp_k].shape == cur_shape:
                sp_dens = self._accum_maps[sp_k] / float(self._accum_count)
            else:
                sp_dens = None
            _, prof = sim.get_radial_density_profile(x2, species=sp_k, density_map=sp_dens)
            prof_clipped = np.maximum(prof, floor_density)
            self.ax_p2.plot(y_pts, prof_clipped, label=label, color=color, linestyle=ls, linewidth=1.7)

        self.ax_p2.set_yscale('log')
        self.ax_p2.set_ylim(bottom=y_min_plot, top=y_max_plot)
        self.ax_p2.set_xlim(0, sim.Ly)
        self.ax_p2.set_title(f"Plane 2 (Divergence / Plume): x = {x2:.2f} mm", fontsize=9.5, fontweight='bold')
        self.ax_p2.set_xlabel("Radial Coordinate y [mm]", fontsize=9)
        self.ax_p2.set_ylabel("Density n [m⁻³]", fontsize=9)
        self.ax_p2.grid(True, which='both', linestyle=':', alpha=0.55)
        self.ax_p2.legend(fontsize=8, loc='upper right', framealpha=0.85)

        self.canvas.draw_idle()

