"""
PPC (Particles Per Cell) Distribution Diagnostic Window.

Displays the 2D spatial distribution of Particles Per Cell across the simulation domain,
supporting Time-Averaged Accumulator, Instantaneous PPC, and Low PPC Warning mask (< 3 ptcls/cell).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap, BoundaryNorm
try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton
    )


class PPCWindow(QWidget):
    """
    Window displaying the 2D spatial distribution of Particles Per Cell (PPC).
    Supports Time-Averaged Accumulator, Instantaneous PPC,
    and Low PPC Warning (< 3 ptcls/cell) mask across Presheath, Optics, and Plume zones.

    Uses GridSpec (not make_axes_locatable) so that ax.clear() / cax.clear() never
    corrupts matplotlib's _shared_axes siblings graph, which would trigger a RecursionError.
    """
    def __init__(self, parent=None):
        super().__init__()
        self.parent_app = parent
        self.setWindowTitle("Macroparticle Distribution per Cell (PPC)")
        self.setGeometry(120, 120, 780, 540)

        layout = QVBoxLayout(self)

        # Controls row
        ctrl_layout = QHBoxLayout()

        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Time-Averaged PPC (Accumulator)",
            "Instantaneous PPC (Current Step)",
            "Low PPC Warning Mask (< 3 ptcls/cell)"
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_change)
        ctrl_layout.addWidget(QLabel("<b>PPC Mode:</b>"))
        ctrl_layout.addWidget(self.combo_mode)

        self.chk_zones = QCheckBox("Show Physical Zones")
        self.chk_zones.setChecked(True)
        self.chk_zones.stateChanged.connect(self._on_mode_change)
        ctrl_layout.addWidget(self.chk_zones)

        self.btn_reset_accum = QPushButton("Reset Accumulator")
        self.btn_reset_accum.clicked.connect(self._reset_accum)
        ctrl_layout.addWidget(self.btn_reset_accum)

        layout.addLayout(ctrl_layout)

        # Statistics Summary Panel
        self.lbl_stats = QLabel("Active Cells: — | Low PPC (< 3): — (—%) | Mean Active PPC: —")
        self.lbl_stats.setStyleSheet(
            "font-family: monospace; font-size: 11px; background-color: #f4f6f9; "
            "padding: 6px; border: 1px solid #d0d7de; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_stats)

        # Matplotlib Figure with pre-allocated GridSpec axes
        self.fig = plt.figure(figsize=(7.5, 4.2))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[28, 1], wspace=0.05)
        self.ax = self.fig.add_subplot(gs[0])
        self.cax = self.fig.add_subplot(gs[1])
        self.cbar = None
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def _reset_accum(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            self.parent_app.sim.reset_ppc_accumulator()
            self.update_plot(self.parent_app.sim)

    def _on_mode_change(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            self.update_plot(self.parent_app.sim)

    def update_plot(self, sim):
        if sim is None or not hasattr(sim, 'nx') or sim.nx <= 0 or sim.ny <= 0:
            return

        mode = self.combo_mode.currentText()
        if "Time-Averaged" in mode:
            raw_ppc = sim.get_avg_ppc_map()
            title = f"Time-Averaged PPC — {getattr(sim, 'ppc_steps_count', 0)} steps"
        elif "Instantaneous" in mode:
            raw_ppc = getattr(sim, 'current_ppc_map', np.zeros((sim.ny, sim.nx))).astype(float)
            title = f"Instantaneous PPC — Step {sim.iteration}"
        else:
            raw_ppc = getattr(sim, 'current_ppc_map', np.zeros((sim.ny, sim.nx))).astype(float)
            title = f"Low PPC Warning Mask (< 3 particles/cell) — Step {sim.iteration}"

        # True active and low PPC masks based on actual particle numbers
        active_mask = (raw_ppc > 0)
        low_mask = (raw_ppc > 0) & (raw_ppc < 3)
        total_active = int(np.count_nonzero(active_mask))
        low_count = int(np.count_nonzero(low_mask))
        pct_low = (low_count / total_active * 100.0) if total_active > 0 else 0.0
        mean_active = float(np.mean(raw_ppc[active_mask])) if total_active > 0 else 0.0

        # Physical zone boundaries
        x_up = getattr(sim, 'upstream_gap_mm', 0.8)
        x_last = sim.grid_x_ends[-1] if hasattr(sim, 'grid_x_ends') and sim.grid_x_ends else sim.Lx * 0.5
        ix_up = int(np.clip(sim._x_to_ix(x_up) if hasattr(sim, '_x_to_ix') else round(x_up / sim.dx), 0, sim.nx))
        ix_last = int(np.clip(sim._x_to_ix(x_last) if hasattr(sim, '_x_to_ix') else round(x_last / sim.dx), 0, sim.nx))

        def _zone_stats(cols):
            a = int(np.count_nonzero(active_mask[:, cols]))
            l = int(np.count_nonzero(low_mask[:, cols]))
            p = (l / a * 100.0) if a > 0 else 0.0
            return a, l, p

        act_up, low_up, pct_up = _zone_stats(slice(None, ix_up))
        act_opt, low_opt, pct_opt = _zone_stats(slice(ix_up, ix_last))
        act_plm, low_plm, pct_plm = _zone_stats(slice(ix_last, None))

        self.lbl_stats.setText(
            f"Active Cells: {total_active} | Low PPC (< 3): {low_count} ({pct_low:.1f}%) | Mean PPC: {mean_active:.1f}\n"
            f"Zones Low%:  Presheath {pct_up:.1f}% ({low_up}/{act_up})  |  "
            f"Optics {pct_opt:.1f}% ({low_opt}/{act_opt})  |  "
            f"Plume {pct_plm:.1f}% ({low_plm}/{act_plm})"
        )
        if low_count > 0:
            self.lbl_stats.setStyleSheet(
                "font-family: monospace; font-size: 11px; background-color: #fff8e6; "
                "padding: 6px; border: 1px solid #e3a008; border-radius: 4px;"
            )
        else:
            self.lbl_stats.setStyleSheet(
                "font-family: monospace; font-size: 11px; background-color: #f0fdf4; "
                "padding: 6px; border: 1px solid #86efac; border-radius: 4px;"
            )

        # Clear axes safely — reset axes locator
        self.ax.clear()
        self.cax.set_axes_locator(None)
        self.cax.clear()
        self.cbar = None

        self.ax.set_title(title, fontsize=10, pad=8)
        self.ax.set_xlabel("Axial Position [mm]")
        self.ax.set_ylabel("Radial Position [mm]")

        X = getattr(sim, 'X', None)
        Y = getattr(sim, 'Y', None)

        if "Low PPC" in mode:
            # 3 categories: 0 = Empty / Vacuum, 1 = OK (>= 3), 2 = LOW (< 3)
            cat_map = np.zeros((sim.ny, sim.nx), dtype=float)
            cat_map[raw_ppc >= 3] = 1.0
            cat_map[low_mask] = 2.0

            cmap = ListedColormap(['#eef1f6', '#2ca02c', '#d62728'])
            norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
            if X is not None and Y is not None:
                im = self.ax.pcolormesh(X, Y, cat_map, cmap=cmap, norm=norm, shading='nearest')
            else:
                extent = [0, sim.Lx, 0, sim.Ly]
                im = self.ax.imshow(cat_map, origin='lower', extent=extent,
                                    cmap=cmap, norm=norm, aspect='auto')
        else:
            vmax = max(10.0, float(np.percentile(raw_ppc[raw_ppc > 0], 98))) if np.any(raw_ppc > 0) else 10.0
            if X is not None and Y is not None:
                im = self.ax.pcolormesh(X, Y, raw_ppc, cmap='turbo', vmin=0, vmax=vmax, shading='nearest')
            else:
                extent = [0, sim.Lx, 0, sim.Ly]
                im = self.ax.imshow(raw_ppc, origin='lower', extent=extent,
                                    cmap='turbo', vmin=0, vmax=vmax, aspect='auto')

        self.ax.set_xlim(0, sim.Lx)
        self.ax.set_ylim(0, sim.Ly)

        # Grid boundary overlay
        if hasattr(sim, 'isBound') and np.any(sim.isBound):
            gy, gx = np.where(sim.isBound)
            x_pts = getattr(sim, 'x_coords', getattr(sim, 'xpts', None))
            y_pts = getattr(sim, 'y_coords', getattr(sim, 'ypts', None))
            if x_pts is not None and y_pts is not None:
                self.ax.scatter(x_pts[gx], y_pts[gy], s=3, c='black', alpha=0.7)
            else:
                self.ax.scatter(gx * sim.dx, gy * sim.dy, s=3, c='black', alpha=0.7)

        # Zone delimiter lines and labels
        if self.chk_zones.isChecked():
            self.ax.axvline(x=x_up, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
            self.ax.axvline(x=x_last, color='cyan', linestyle='--', linewidth=1.2, alpha=0.8)
            y_pos = sim.Ly * 0.90
            for xc, label in [
                (x_up * 0.5, "Presheath"),
                ((x_up + x_last) * 0.5, "Optics"),
                ((x_last + sim.Lx) * 0.5, "Plume"),
            ]:
                self.ax.text(xc, y_pos, label, color='white', fontsize=8, ha='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

        self.ax.set_xlim(0, sim.Lx)
        self.ax.set_ylim(0, sim.Ly)

        # Colorbar drawn into pre-allocated cax
        self.cbar = self.fig.colorbar(im, cax=self.cax)
        if "Low PPC" in mode:
            self.cbar.set_ticks([0, 1, 2])
            self.cbar.set_ticklabels(["Empty", "OK (>= 3)", "LOW (< 3)"])
        else:
            self.cbar.set_label("Particles / Cell (PPC)", fontsize=8)

        self.canvas.draw_idle()
