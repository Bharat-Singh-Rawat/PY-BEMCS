"""
Computational Domain and Mesh Grid Diagnostic Window for PY-BEMCS.

Displays the physical 2D domain geometry with the computational mesh grid overlaid,
potential contours, grid electrodes, and physical zone partitions.
Provides numerical readouts for:
- Delta x / Delta y (base, min, max, ratio over Debye length)
- Electron Debye length and Presheath Debye length
- Domain dimensions and various mesh sizes (total nodes, cells, zone breakdown)
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

try:
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    NavigationToolbar = None

from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QGroupBox, QGridLayout, QFrame, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class MeshWindow(QDialog):
    """
    Popup window displaying the computational domain and mesh grid with
    spatial resolution, Debye length, and mesh dimension analytics.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Computational Domain & Mesh Grid Inspector")
        self.resize(920, 720)
        self.setMinimumSize(780, 560)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ── Controls toolbar ──────────────────────────────────────────
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(14)

        self.chk_mesh = QCheckBox("Show Mesh Lines")
        self.chk_mesh.setChecked(True)
        self.chk_mesh.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_mesh)

        self.chk_potential = QCheckBox("Show Potential Contours")
        self.chk_potential.setChecked(True)
        self.chk_potential.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_potential)

        self.chk_electrodes = QCheckBox("Show Grid Electrodes")
        self.chk_electrodes.setChecked(True)
        self.chk_electrodes.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_electrodes)

        self.chk_zones = QCheckBox("Show Physical Zones")
        self.chk_zones.setChecked(True)
        self.chk_zones.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_zones)

        ctrl_layout.addStretch()

        self.btn_refresh = QPushButton("⟳ Refresh")
        self.btn_refresh.setToolTip("Re-read simulation state and redraw mesh")
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        ctrl_layout.addWidget(self.btn_refresh)

        main_layout.addLayout(ctrl_layout)

        # ── Matplotlib Canvas ─────────────────────────────────────────
        self.fig = Figure(figsize=(8.5, 4.0), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(30)
            main_layout.addWidget(self.toolbar)

        main_layout.addWidget(self.canvas, stretch=1)

        # ── Under-the-graph Analytics Dashboard ───────────────────────
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

        # Card 1: Grid Spacing (Δx, Δy)
        grp_dx = QGroupBox("Grid Spacing (Δx, Δy)")
        grp_dx.setStyleSheet(self._group_style())
        lay_dx = QVBoxLayout(grp_dx)
        self.lbl_dx_nominal = QLabel("Δx (base): —")
        self.lbl_dy_nominal = QLabel("Δy (base): —")
        self.lbl_dx_range   = QLabel("Δx min..max: —")
        self.lbl_ratio      = QLabel("Δx / λ_D: —")
        for lbl in (self.lbl_dx_nominal, self.lbl_dy_nominal, self.lbl_dx_range, self.lbl_ratio):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_dx.addWidget(lbl)
        stats_layout.addWidget(grp_dx)

        # Card 2: Debye Length
        grp_debye = QGroupBox("Debye Length (λ_D)")
        grp_debye.setStyleSheet(self._group_style())
        lay_debye = QVBoxLayout(grp_debye)
        self.lbl_debye_std   = QLabel("λ_D (unscreened): —")
        self.lbl_debye_bohm  = QLabel("λ_D (Bohm presheath): —")
        self.lbl_plasma_n0   = QLabel("Plasma n₀: —")
        self.lbl_plasma_te   = QLabel("Electron Te: —")
        for lbl in (self.lbl_debye_std, self.lbl_debye_bohm, self.lbl_plasma_n0, self.lbl_plasma_te):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_debye.addWidget(lbl)
        stats_layout.addWidget(grp_debye)

        # Card 3: Mesh Sizes & Dimensions
        grp_sizes = QGroupBox("Mesh Dimensions & Sizes")
        grp_sizes.setStyleSheet(self._group_style())
        lay_sizes = QVBoxLayout(grp_sizes)
        self.lbl_domain_size = QLabel("Domain (Lx × Ly): —")
        self.lbl_grid_nodes  = QLabel("Grid Nodes (Nx × Ny): —")
        self.lbl_grid_cells  = QLabel("Total Cells: —")
        self.lbl_mesh_mode   = QLabel("Mesh Structure: —")
        for lbl in (self.lbl_domain_size, self.lbl_grid_nodes, self.lbl_grid_cells, self.lbl_mesh_mode):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_sizes.addWidget(lbl)
        stats_layout.addWidget(grp_sizes)

        main_layout.addWidget(stats_frame)

        # ── Zone Breakdown (Collapsible / Bottom Bar) ──────────────────
        self.lbl_zones_summary = QLabel("Zone Breakdown: —")
        self.lbl_zones_summary.setStyleSheet(
            "font-family: monospace; font-size: 10px; color: #475569; "
            "background-color: #f1f5f9; padding: 4px 8px; border-radius: 4px; border: 1px solid #cbd5e1;"
        )
        main_layout.addWidget(self.lbl_zones_summary)

        # Cached state
        self._cached_sim = None
        self._cached_params = None

    def _group_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                margin-top: 8px;
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
        sim = getattr(self.parent_app, 'sim', None)
        params = self.parent_app.get_params() if hasattr(self.parent_app, 'get_params') else None
        if sim is not None:
            self.update_plot(sim, params=params)

    def _replot(self):
        if self._cached_sim is not None:
            self.update_plot(self._cached_sim, params=self._cached_params)

    def update_plot(self, sim, params=None):
        """Update domain visualization and mesh analytics."""
        self._cached_sim = sim
        self._cached_params = params

        self.ax.clear()

        # Check if domain has been built
        is_built = (
            sim is not None and
            getattr(sim, 'X', None) is not None and
            getattr(sim, 'isBound', None) is not None and
            getattr(sim, 'nx', 0) > 0
        )
        if not is_built:
            self.ax.text(
                0.5, 0.5,
                "Domain is not built yet.\nClick '1. BUILD DOMAIN' in main window to initialize mesh.",
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=11, color="#64748b"
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.lbl_dx_nominal.setText("Δx (base): —")
            self.lbl_dy_nominal.setText("Δy (base): —")
            self.lbl_dx_range.setText("Δx min..max: —")
            self.lbl_ratio.setText("Δx / λ_D ratio: —")
            self.lbl_debye_std.setText("λ_D (unscreened): —")
            self.lbl_debye_bohm.setText("λ_D (Bohm presheath): —")
            self.lbl_plasma_n0.setText("Plasma n₀: —")
            self.lbl_plasma_te.setText("Electron Te: —")
            self.lbl_domain_size.setText("Domain (Lx × Ly): —")
            self.lbl_grid_nodes.setText("Grid Nodes (Nx × Ny): —")
            self.lbl_grid_cells.setText("Total Mesh Cells: —")
            self.lbl_mesh_mode.setText("Mesh Structure: —")
            self.lbl_zones_summary.setText("Domain is not initialized.")
            self.canvas.draw_idle()
            return

        Lx = getattr(sim, 'Lx', 10.0)
        Ly = getattr(sim, 'Ly', 3.0)
        dx = getattr(sim, 'dx', 0.04)
        dy = getattr(sim, 'dy', 0.04)
        nx = getattr(sim, 'nx', len(getattr(sim, 'x_coords', [])))
        ny = getattr(sim, 'ny', len(getattr(sim, 'y_coords', [])))

        x_coords = getattr(sim, 'x_coords', getattr(sim, 'xpts', None))
        y_coords = getattr(sim, 'y_coords', getattr(sim, 'ypts', None))
        if x_coords is None:
            x_coords = np.linspace(0, Lx, nx)
        if y_coords is None:
            y_coords = np.linspace(0, Ly, ny)

        # ── 1. Potential field background ─────────────────────────────
        if self.chk_potential.isChecked() and hasattr(sim, 'V') and sim.V is not None:
            try:
                V = sim.V
                if V.shape == (len(y_coords), len(x_coords)):
                    self.ax.contourf(
                        sim.X if hasattr(sim, 'X') else x_coords,
                        sim.Y if hasattr(sim, 'Y') else y_coords,
                        V, levels=25, cmap="viridis", alpha=0.35, zorder=1
                    )
            except Exception:
                pass

        # ── 2. Mesh Grid Lines ────────────────────────────────────────
        if self.chk_mesh.isChecked():
            # Vertical mesh lines at x_coords
            self.ax.vlines(
                x_coords, ymin=0, ymax=Ly,
                colors="#334155", linestyles="-", linewidths=0.45, alpha=0.35, zorder=2
            )
            # Horizontal mesh lines at y_coords
            self.ax.hlines(
                y_coords, xmin=0, xmax=Lx,
                colors="#334155", linestyles="-", linewidths=0.45, alpha=0.35, zorder=2
            )

        # ── 3. Grid Electrodes Overlay ────────────────────────────────
        if self.chk_electrodes.isChecked() and hasattr(sim, 'isBound') and np.any(sim.isBound):
            gy, gx = np.where(sim.isBound)
            if len(gx) > 0 and len(gy) > 0:
                self.ax.scatter(
                    x_coords[gx], y_coords[gy],
                    s=7, c="#0f172a", marker="s", alpha=0.85, zorder=4, label="Grid Electrodes"
                )

        # ── 4. Physical Zones Delimiters ──────────────────────────────
        zone_configs = getattr(sim, 'zone_configs', None)
        if self.chk_zones.isChecked() and zone_configs:
            y_top = Ly * 0.92
            for z in zone_configs:
                xs = z.get('x_start', 0.0)
                xe = z.get('x_end', Lx)
                name = z.get('name', '')
                factor = z.get('factor', 1.0)
                if xs > 0.0:
                    self.ax.axvline(x=xs, color="#e11d48", linestyle="--", linewidth=1.0, alpha=0.75, zorder=5)
                xc = 0.5 * (xs + xe)
                short_name = name.replace("Grid_", "G").replace("_barrel", "").replace("Gap_", "Gap ").replace("_Plume", " Plm")
                tag = f"{short_name}\n({factor:.1f}×)"
                self.ax.text(
                    xc, y_top, tag, color="#0f172a", fontsize=7, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.85),
                    zorder=6
                )
        elif self.chk_zones.isChecked():
            up_gap = getattr(sim, 'upstream_gap_mm', 0.0)
            if up_gap > 0.0:
                self.ax.axvline(x=up_gap, color="#e11d48", linestyle="--", linewidth=1.0, alpha=0.75, zorder=5)
                self.ax.text(
                    up_gap * 0.5, Ly * 0.92, "Presheath",
                    color="#0f172a", fontsize=7.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.85)
                )

        self.ax.set_xlim(0, Lx)
        self.ax.set_ylim(0, Ly)
        self.ax.set_xlabel("Axial Position x [mm]", fontsize=9, fontweight="bold")
        self.ax.set_ylabel("Radial Position y [mm]", fontsize=9, fontweight="bold")
        self.ax.set_title(
            f"Domain Mesh Grid: {nx} × {ny} Nodes (Total Cells: {(nx - 1) * (ny - 1):,})",
            fontsize=10, fontweight="bold", pad=8
        )
        self.ax.grid(False)

        # ── 5. Compute & Update Analytical Readouts ───────────────────
        # Physical constants
        eps0 = getattr(sim, 'eps0', 8.854187817e-12)
        q    = getattr(sim, 'q', 1.602176634e-19)

        # Plasma parameters
        n0_plasma = 1e17
        Te_up = 3.0
        if params is not None:
            n0_plasma = float(params.get("n0_plasma", 1e17))
            Te_up = float(params.get("Te_up", 3.0))
        elif hasattr(self.parent_app, 'inputs'):
            inp = self.parent_app.inputs
            if "n0_plasma" in inp:
                n0_plasma = inp["n0_plasma"].value()
            if "Te_up" in inp:
                Te_up = inp["Te_up"].value()

        # Debye lengths
        lambda_D_m = math.sqrt(eps0 * Te_up * q / (n0_plasma * q**2)) if n0_plasma > 0 else 0.0
        lambda_D_mm = lambda_D_m * 1e3
        lambda_D_um = lambda_D_m * 1e6

        lambda_D_bohm_m = math.sqrt(eps0 * Te_up / (q * n0_plasma * 0.61)) if n0_plasma > 0 else 0.0
        lambda_D_bohm_mm = lambda_D_bohm_m * 1e3
        lambda_D_bohm_um = lambda_D_bohm_m * 1e6

        # Delta x and Delta y
        dx_cells = getattr(sim, 'dx_cells', np.diff(x_coords) if len(x_coords) > 1 else np.array([dx]))
        dx_min = float(np.min(dx_cells)) if len(dx_cells) > 0 else dx
        dx_max = float(np.max(dx_cells)) if len(dx_cells) > 0 else dx
        dy_cells = getattr(sim, 'dy_cells', np.diff(y_coords) if len(y_coords) > 1 else np.array([dy]))
        dy_val = float(np.mean(dy_cells)) if len(dy_cells) > 0 else dy

        ratio_dx_debye = (dx * 1e-3) / lambda_D_m if lambda_D_m > 0 else 0.0

        # Update Card 1: Δx, Δy
        self.lbl_dx_nominal.setText(f"Δx (base): <b>{dx:.4f} mm</b> ({dx * 1e3:.1f} µm)")
        self.lbl_dy_nominal.setText(f"Δy (base): <b>{dy_val:.4f} mm</b> ({dy_val * 1e3:.1f} µm)")
        if abs(dx_max - dx_min) > 1e-6:
            self.lbl_dx_range.setText(f"Δx min..max: <b>{dx_min:.4f}..{dx_max:.4f} mm</b>")
        else:
            self.lbl_dx_range.setText(f"Δx min..max: uniform ({dx:.4f} mm)")
        self.lbl_ratio.setText(f"Δx / λ_D ratio: <b>{ratio_dx_debye:.3f}</b>")

        # Update Card 2: Debye length
        self.lbl_debye_std.setText(f"λ_D (unscreened): <b>{lambda_D_mm:.4f} mm</b> ({lambda_D_um:.1f} µm)")
        self.lbl_debye_bohm.setText(f"λ_D (Bohm presheath): <b>{lambda_D_bohm_mm:.4f} mm</b> ({lambda_D_bohm_um:.1f} µm)")
        self.lbl_plasma_n0.setText(f"Plasma n₀: <b>{n0_plasma:.2e} m⁻³</b>")
        self.lbl_plasma_te.setText(f"Electron Te: <b>{Te_up:.2f} eV</b>")

        # Update Card 3: Mesh dimensions
        total_nodes = nx * ny
        total_cells = (nx - 1) * (ny - 1)
        self.lbl_domain_size.setText(f"Domain (Lx × Ly): <b>{Lx:.3f} × {Ly:.3f} mm</b>")
        self.lbl_grid_nodes.setText(f"Grid Nodes (Nx × Ny): <b>{nx} × {ny}</b> ({total_nodes:,})")
        self.lbl_grid_cells.setText(f"Total Mesh Cells: <b>{total_cells:,}</b>")
        if zone_configs and len(zone_configs) > 1:
            mode_str = f"Multi-Zone Adaptive ({len(zone_configs)} zones)"
        else:
            mode_str = "Single Zone / Uniform"
        self.lbl_mesh_mode.setText(f"Mesh Structure: <b>{mode_str}</b>")

        # Zone Breakdown text
        if zone_configs:
            parts = []
            for z in zone_configs:
                z_name = z.get('name', 'Zone')
                z_xs = z.get('x_start', 0.0)
                z_xe = z.get('x_end', 0.0)
                z_fac = z.get('factor', 1.0)
                z_dx = dx * z_fac
                parts.append(f"[{z_name}: x={z_xs:.2f}..{z_xe:.2f} mm | fac={z_fac:.1f}× | Δx={z_dx:.4f} mm]")
            self.lbl_zones_summary.setText("  •  ".join(parts))
        else:
            self.lbl_zones_summary.setText(f"Uniform Mesh: Δx={dx:.4f} mm, Δy={dy_val:.4f} mm across full domain [0, {Lx:.2f} mm].")

        self.canvas.draw_idle()
