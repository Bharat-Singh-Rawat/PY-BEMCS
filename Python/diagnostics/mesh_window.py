"""
Computational Domain and Mesh Grid Diagnostic Window for PY-BEMCS.

Displays the physical 2D domain geometry with the computational mesh grid overlaid,
potential contours, grid electrodes, and physical zone partitions.
Provides numerical readouts for:
- Delta x / Delta y (base, min, max, ratio over Debye length)
- Electron Debye length and Presheath Debye length
- Domain dimensions and various mesh sizes (total nodes, cells, zone breakdown)

Also features a dedicated Displacement Substeps Monitor (Tab and Pop-out Second Window)
showing simulation iterations on the x-axis and required Boris pusher sub-steps on the y-axis,
with highlight green dots for electrons and red dots for ions.
"""

import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

try:
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    try:
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    except Exception:
        NavigationToolbar = None

from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QGroupBox, QFrame,
    QTabWidget, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt


class SubstepGraphWidget(QWidget):
    """
    Graph widget showing the history of required Boris pusher sub-steps
    per iteration for ions (red dots) and electrons (highlight green dots).
    Tracks Courant/Boris displacement violations.
    """
    def __init__(self, parent=None, is_standalone=False):
        super().__init__(parent)
        self.parent_app = parent
        self.is_standalone = is_standalone
        self._cached_sim = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Controls toolbar ──────────────────────────────────────────
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(12)

        self.chk_elec = QCheckBox("Electrons (Green Dots)")
        self.chk_elec.setChecked(True)
        self.chk_elec.setToolTip("Show required sub-steps for electrons as highlight green dots")
        self.chk_elec.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_elec)

        self.chk_ions = QCheckBox("Ions (Red Dots)")
        self.chk_ions.setChecked(True)
        self.chk_ions.setToolTip("Show required sub-steps for ions as red dots")
        self.chk_ions.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_ions)

        self.chk_violations = QCheckBox("Violations Only (N_sub > 1)")
        self.chk_violations.setChecked(False)
        self.chk_violations.setToolTip("Display only iterations where displacement criterion was violated")
        self.chk_violations.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_violations)

        self.chk_lines = QCheckBox("Connect Dots")
        self.chk_lines.setChecked(False)
        self.chk_lines.setToolTip("Draw light connecting lines between sequential dots")
        self.chk_lines.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_lines)

        ctrl_layout.addStretch()

        self.btn_clear = QPushButton("Clear History")
        self.btn_clear.setToolTip("Clear iteration substep history")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        ctrl_layout.addWidget(self.btn_clear)

        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.setToolTip("Export iteration substeps history to CSV")
        self.btn_export.clicked.connect(self._on_export_clicked)
        ctrl_layout.addWidget(self.btn_export)

        layout.addLayout(ctrl_layout)

        # ── Matplotlib Canvas ─────────────────────────────────────────
        self.fig = Figure(figsize=(8.5, 4.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.14)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(30)
            layout.addWidget(self.toolbar)

        layout.addWidget(self.canvas, stretch=1)

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

        # Card 1: Current Step
        grp_curr = QGroupBox("Current Step Substeps")
        grp_curr.setStyleSheet(self._group_style())
        lay_curr = QVBoxLayout(grp_curr)
        self.lbl_curr_iter = QLabel("Iteration: —")
        self.lbl_curr_ion  = QLabel("Ion Substeps: —")
        self.lbl_curr_elec = QLabel("Electron Substeps: —")
        for lbl in (self.lbl_curr_iter, self.lbl_curr_ion, self.lbl_curr_elec):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_curr.addWidget(lbl)
        stats_layout.addWidget(grp_curr)

        # Card 2: Violations Summary
        grp_viol = QGroupBox("Displacement Violations (N_sub > 1)")
        grp_viol.setStyleSheet(self._group_style())
        lay_viol = QVBoxLayout(grp_viol)
        self.lbl_viol_ion  = QLabel("Ion Violations: 0 (0.0%)")
        self.lbl_viol_elec = QLabel("Electron Violations: 0 (0.0%)")
        self.lbl_total_steps = QLabel("Total Steps Recorded: 0")
        for lbl in (self.lbl_viol_ion, self.lbl_viol_elec, self.lbl_total_steps):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_viol.addWidget(lbl)
        stats_layout.addWidget(grp_viol)

        # Card 3: Peak Substeps
        grp_peak = QGroupBox("Peak Substeps Record")
        grp_peak.setStyleSheet(self._group_style())
        lay_peak = QVBoxLayout(grp_peak)
        self.lbl_peak_ion  = QLabel("Max Ion Substeps: 1")
        self.lbl_peak_elec = QLabel("Max Electron Substeps: 1")
        self.lbl_criterion = QLabel("Criterion: Δs ≤ 0.25·min(Δx, Δy)")
        for lbl in (self.lbl_peak_ion, self.lbl_peak_elec, self.lbl_criterion):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            lay_peak.addWidget(lbl)
        stats_layout.addWidget(grp_peak)

        layout.addWidget(stats_frame)

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

    def _replot(self):
        self.update_plot()

    def _on_clear_clicked(self):
        sim = self._cached_sim
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
        if sim is not None:
            if hasattr(sim, 'substep_history_iter'):
                sim.substep_history_iter.clear()
                sim.substep_history_ion.clear()
                sim.substep_history_e.clear()
        if self.parent_app and hasattr(self.parent_app, 'substep_history_iter'):
            self.parent_app.substep_history_iter.clear()
            self.parent_app.substep_history_ion.clear()
            self.parent_app.substep_history_e.clear()
        self.update_plot()

    def _on_export_clicked(self):
        sim = self._cached_sim
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
        iters = getattr(sim, 'substep_history_iter', [])
        subs_i = getattr(sim, 'substep_history_ion', [])
        subs_e = getattr(sim, 'substep_history_e', [])
        if not iters and self.parent_app and hasattr(self.parent_app, 'substep_history_iter'):
            iters = self.parent_app.substep_history_iter
            subs_i = self.parent_app.substep_history_ion
            subs_e = self.parent_app.substep_history_e
        if not iters:
            QMessageBox.information(self, "No Data", "No substep history to export.")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self, "Export Substeps History", "substeps_history.csv", "CSV Files (*.csv)"
        )
        if not fpath:
            return

        try:
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Required_Substeps_Ion", "Required_Substeps_Electron"])
                for it, ni, ne in zip(iters, subs_i, subs_e):
                    writer.writerow([it, ni, ne])
            QMessageBox.information(self, "Export Complete", f"Saved {len(iters)} steps to:\n{fpath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save CSV:\n{e}")

    def update_plot(self, sim=None):
        """Update the iterations vs required substeps graph."""
        if sim is not None:
            self._cached_sim = sim
        else:
            sim = self._cached_sim
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim

        iters = getattr(sim, 'substep_history_iter', [])
        subs_i = getattr(sim, 'substep_history_ion', [])
        subs_e = getattr(sim, 'substep_history_e', [])

        if (not iters) and self.parent_app and hasattr(self.parent_app, 'substep_history_iter'):
            iters = self.parent_app.substep_history_iter
            subs_i = self.parent_app.substep_history_ion
            subs_e = self.parent_app.substep_history_e

        self.ax.clear()

        n_pts = min(len(iters or []), len(subs_i or []), len(subs_e or []))
        if not iters or n_pts == 0:
            self.ax.text(
                0.5, 0.5,
                "No simulation steps recorded yet.\nClick '2. START BEAM' in main window to record sub-step diagnostics.",
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=11, color="#64748b"
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.lbl_curr_iter.setText("Iteration: —")
            self.lbl_curr_ion.setText("Ion Substeps: —")
            self.lbl_curr_elec.setText("Electron Substeps: —")
            self.lbl_viol_ion.setText("Ion Violations: 0 (0.0%)")
            self.lbl_viol_elec.setText("Electron Violations: 0 (0.0%)")
            self.lbl_total_steps.setText("Total Steps Recorded: 0")
            self.lbl_peak_ion.setText("Max Ion Substeps: 1")
            self.lbl_peak_elec.setText("Max Electron Substeps: 1")
            self.canvas.draw_idle()
            return

        x = np.asarray(iters, dtype=int)
        y_i = np.asarray(subs_i, dtype=int)
        y_e = np.asarray(subs_e, dtype=int)

        x = x[:n_pts]
        y_i = y_i[:n_pts]
        y_e = y_e[:n_pts]

        total_steps = len(x)
        viol_i_count = int(np.sum(y_i > 1))
        viol_e_count = int(np.sum(y_e > 1))
        pct_i = (viol_i_count / total_steps * 100.0) if total_steps > 0 else 0.0
        pct_e = (viol_e_count / total_steps * 100.0) if total_steps > 0 else 0.0
        max_i = int(np.max(y_i)) if total_steps > 0 else 1
        max_e = int(np.max(y_e)) if total_steps > 0 else 1

        self.lbl_curr_iter.setText(f"Iteration: <b>{x[-1]}</b>")
        self.lbl_curr_ion.setText(f"Ion Substeps: <b>{y_i[-1]}</b>")
        self.lbl_curr_elec.setText(f"Electron Substeps: <b>{y_e[-1]}</b>")

        tag_i = f"<span style='color: {'#dc2626' if viol_i_count > 0 else '#16a34a'}; font-weight: bold;'>{viol_i_count} ({pct_i:.1f}%)</span>"
        tag_e = f"<span style='color: {'#dc2626' if viol_e_count > 0 else '#16a34a'}; font-weight: bold;'>{viol_e_count} ({pct_e:.1f}%)</span>"
        self.lbl_viol_ion.setText(f"Ion Violations: {tag_i}")
        self.lbl_viol_elec.setText(f"Electron Violations: {tag_e}")
        self.lbl_total_steps.setText(f"Total Steps Recorded: <b>{total_steps:,}</b>")
        self.lbl_peak_ion.setText(f"Max Ion Substeps: <b>{max_i}</b>")
        self.lbl_peak_elec.setText(f"Max Electron Substeps: <b>{max_e}</b>")

        # Baseline reference line at N_sub = 1
        self.ax.axhline(1, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.75, zorder=2, label="Nominal (N_sub = 1)")

        # Filter if violations only
        if self.chk_violations.isChecked():
            mask_viol = (y_i > 1) | (y_e > 1)
            x_plot = x[mask_viol]
            yi_plot = y_i[mask_viol]
            ye_plot = y_e[mask_viol]
            if len(x_plot) == 0:
                self.ax.text(
                    0.5, 0.5,
                    "No displacement violations (N_sub > 1) recorded so far.\nUncheck 'Violations Only' to view all steps.",
                    ha="center", va="center", transform=self.ax.transAxes,
                    fontsize=10.5, color="#16a34a"
                )
        else:
            x_plot = x
            yi_plot = y_i
            ye_plot = y_e

        if len(x_plot) > 0:
            offset = 0.12 if len(x_plot) < 800 else 0.0

            # Connecting lines if requested
            if self.chk_lines.isChecked():
                if self.chk_elec.isChecked():
                    self.ax.plot(x_plot, ye_plot, color="#00e676", alpha=0.35, linewidth=1.0, zorder=3)
                if self.chk_ions.isChecked():
                    self.ax.plot(x_plot, yi_plot, color="#ef4444", alpha=0.35, linewidth=1.0, zorder=4)

            # Highlight green dots for electrons
            if self.chk_elec.isChecked():
                self.ax.scatter(
                    x_plot - offset, ye_plot,
                    s=34, c="#00e676", edgecolors="#004d40", linewidths=0.7,
                    alpha=0.9, zorder=6, label=f"Electrons (highlight green dots, peak={max_e})"
                )

            # Red dots for ions
            if self.chk_ions.isChecked():
                self.ax.scatter(
                    x_plot + offset, yi_plot,
                    s=34, c="#ef4444", edgecolors="#7f1d1d", linewidths=0.7,
                    alpha=0.9, zorder=7, label=f"Ions (red dots, peak={max_i})"
                )

            x_min = np.min(x_plot)
            x_max = np.max(x_plot)
            pad_x = max(1, int((x_max - x_min) * 0.03))
            self.ax.set_xlim(max(0, x_min - pad_x), x_max + pad_x)

            peak_y = max(max_i, max_e, 5)
            self.ax.set_ylim(0.5, peak_y + 0.5)

        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_xlabel("Iteration", fontsize=9.5, fontweight="bold")
        self.ax.set_ylabel("Required Substeps", fontsize=9.5, fontweight="bold")
        self.ax.set_title("Required Boris Pusher Substeps per Iteration (Displacement Criterion)", fontsize=10.5, fontweight="bold", pad=28)
        self.ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
        self.ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            framealpha=0.95,
            fontsize=8.5
        )

        self.canvas.draw_idle()


class DisplacementSubstepsWindow(QDialog):
    """
    Dedicated second window displaying the displacement substeps diagnostic graph
    (Iterations on x-axis, Required Substeps on y-axis, highlight green dots for electrons, red dots for ions).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Displacement Substeps Monitor — (Iterations vs Required Substeps)")
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.resize(880, 580)
        self.setMinimumSize(680, 440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.substep_widget = SubstepGraphWidget(parent=self.parent_app, is_standalone=True)
        layout.addWidget(self.substep_widget)

    def update_plot(self, sim=None):
        self.substep_widget.update_plot(sim)


class PlumeMeshSizeGraphWidget(QWidget):
    """
    Graph widget showing the evolution of downstream plume mesh cell size
    Delta x [mm] vs. simulation iterations as automatic adaptive remeshing executes.
    """
    def __init__(self, parent=None, is_standalone=False):
        super().__init__(parent)
        self.parent_app = parent
        self.is_standalone = is_standalone
        self._cached_sim = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Controls toolbar ──────────────────────────────────────────
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(12)

        self.chk_points = QCheckBox("Show Data Points")
        self.chk_points.setChecked(True)
        self.chk_points.setToolTip("Show individual remesh event markers on the graph")
        self.chk_points.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_points)

        self.chk_lines = QCheckBox("Connect Step Line")
        self.chk_lines.setChecked(True)
        self.chk_lines.setToolTip("Draw step-wise lines showing plume mesh cell size across iterations")
        self.chk_lines.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_lines)

        self.chk_baseline = QCheckBox("Show Baseline Nominal Δx₀")
        self.chk_baseline.setChecked(True)
        self.chk_baseline.setToolTip("Show nominal uncoarsened baseline cell size line")
        self.chk_baseline.stateChanged.connect(self._replot)
        ctrl_layout.addWidget(self.chk_baseline)

        ctrl_layout.addStretch()

        self.btn_clear = QPushButton("Clear History")
        self.btn_clear.setToolTip("Clear plume cell size history")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        ctrl_layout.addWidget(self.btn_clear)

        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.setToolTip("Export plume cell size history to CSV")
        self.btn_export.clicked.connect(self._on_export_clicked)
        ctrl_layout.addWidget(self.btn_export)

        layout.addLayout(ctrl_layout)

        # ── Matplotlib Canvas ─────────────────────────────────────────
        self.fig = Figure(figsize=(8.5, 4.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.14)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(30)
            layout.addWidget(self.toolbar)

        layout.addWidget(self.canvas, stretch=1)

        # ── Readouts bar at the bottom ────────────────────────────────
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
        stats_layout.setSpacing(16)

        self.lbl_curr_iter = QLabel("Current Iteration: —")
        self.lbl_curr_dx = QLabel("Plume Cell Size Δx: —")
        self.lbl_curr_factor = QLabel("Plume Factor: —")
        self.lbl_base_dx = QLabel("Nominal Δx₀: —")
        self.lbl_total_remeshes = QLabel("Remesh Events: 0")

        for lbl in (self.lbl_curr_iter, self.lbl_curr_dx, self.lbl_curr_factor, self.lbl_base_dx, self.lbl_total_remeshes):
            lbl.setStyleSheet("font-size: 11px; color: #1e293b;")
            stats_layout.addWidget(lbl)

        layout.addWidget(stats_frame)

    def _replot(self):
        if self._cached_sim is not None:
            self.update_plot(self._cached_sim)

    def _on_clear_clicked(self):
        if self._cached_sim is not None:
            if hasattr(self._cached_sim, 'plume_history_iterations'):
                self._cached_sim.plume_history_iterations.clear()
            if hasattr(self._cached_sim, 'plume_history_dx'):
                self._cached_sim.plume_history_dx.clear()
            if hasattr(self._cached_sim, 'plume_dx_history'):
                self._cached_sim.plume_dx_history.clear()
        self.update_plot(self._cached_sim)

    def _on_export_clicked(self):
        sim = self._cached_sim
        iters = getattr(sim, 'plume_history_iterations', [])
        dxs = getattr(sim, 'plume_history_dx', [])
        if not iters or len(iters) == 0:
            QMessageBox.information(self, "Export CSV", "No plume cell size history to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Plume Cell Size History", "plume_cell_size_history.csv", "CSV Files (*.csv)")
        if not path:
            return

        dx0 = getattr(sim, 'dx', 0.04)
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Plume_dx_mm", "Plume_dx_um", "Nominal_dx0_mm", "Coarsening_Factor"])
                for it, val in zip(iters, dxs):
                    fac = val / dx0 if dx0 > 0 else 1.0
                    writer.writerow([it, f"{val:.6f}", f"{val * 1e3:.3f}", f"{dx0:.6f}", f"{fac:.2f}"])
            QMessageBox.information(self, "Export CSV", f"Successfully exported {len(iters)} records to {path}.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    def update_plot(self, sim=None):
        self._cached_sim = sim
        if sim is None:
            return

        from physics.mesh_utils import compute_plume_mean_dx
        dx0 = float(getattr(sim, 'dx', 0.04))
        iters = getattr(sim, 'plume_history_iterations', [])
        dxs = getattr(sim, 'plume_history_dx', [])

        # Auto-seed initial baseline if history is empty
        if len(iters) == 0:
            init_dx = compute_plume_mean_dx(sim)
            if not hasattr(sim, 'plume_history_iterations'):
                sim.plume_history_iterations = []
                sim.plume_history_dx = []
                sim.plume_dx_history = []
            sim.plume_history_iterations.append(0)
            sim.plume_history_dx.append(init_dx)
            sim.plume_dx_history.append((0, init_dx))
            iters = sim.plume_history_iterations
            dxs = sim.plume_history_dx

        self.ax.clear()

        n_pts = min(len(iters), len(dxs))
        if n_pts == 0:
            self.ax.text(
                0.5, 0.5,
                "No plume remesh history recorded yet.\nStart simulation to view adaptive plume cell size evolution.",
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=11, color="#64748b"
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.lbl_curr_iter.setText("Current Iteration: —")
            self.lbl_curr_dx.setText("Plume Cell Size Δx: —")
            self.lbl_curr_factor.setText("Plume Factor: —")
            self.lbl_base_dx.setText(f"Nominal Δx₀: <b>{dx0:.4f} mm</b>")
            self.lbl_total_remeshes.setText("Remesh Events: 0")
            self.canvas.draw_idle()
            return

        x = np.asarray(iters[:n_pts], dtype=int)
        y = np.asarray(dxs[:n_pts], dtype=float)

        curr_it = getattr(sim, 'iteration', x[-1])
        curr_val = y[-1]
        curr_fac = curr_val / dx0 if dx0 > 0 else 1.0

        self.lbl_curr_iter.setText(f"Current Iteration: <b>{curr_it}</b>")
        self.lbl_curr_dx.setText(f"Plume Cell Size Δx: <b>{curr_val:.4f} mm</b> ({curr_val * 1e3:.1f} µm)")
        self.lbl_curr_factor.setText(f"Plume Factor: <b>{curr_fac:.2f}×</b>")
        self.lbl_base_dx.setText(f"Nominal Δx₀: <b>{dx0:.4f} mm</b> ({dx0 * 1e3:.1f} µm)")
        self.lbl_total_remeshes.setText(f"Remesh Events: <b>{n_pts:,}</b>")

        # Baseline reference line at nominal dx0
        if self.chk_baseline.isChecked():
            self.ax.axhline(dx0, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.85, zorder=2, label=f"Baseline Nominal Δx₀ ({dx0:.4f} mm)")

        # Step line showing evolution across iterations
        if len(x) > 0 and x[0] > 0:
            x_line_start = np.insert(x, 0, 0)
            y_line_start = np.insert(y, 0, dx0)
        else:
            x_line_start = x
            y_line_start = y

        if self.chk_lines.isChecked():
            if curr_it > x_line_start[-1]:
                x_line = np.append(x_line_start, curr_it)
                y_line = np.append(y_line_start, y_line_start[-1])
            else:
                x_line = x_line_start
                y_line = y_line_start
            self.ax.step(x_line, y_line, where='post', color="#2563eb", linewidth=1.8, alpha=0.9, zorder=4, label="Plume Δx(t)")

        # Data point markers at each remesh event
        if self.chk_points.isChecked():
            self.ax.scatter(
                x, y,
                s=40, c="#2563eb", edgecolors="#1e3a8a", linewidths=0.9,
                alpha=0.95, zorder=6, label=f"Remesh Milestones ({n_pts})"
            )

        x_max = max(np.max(x), curr_it)
        pad_x = max(10, int(x_max * 0.05)) if x_max > 0 else 150
        self.ax.set_xlim(0, x_max + pad_x)

        y_min = min(dx0, np.min(y)) * 0.85
        y_max = max(dx0, np.max(y)) * 1.15
        self.ax.set_ylim(max(0.0, y_min), y_max)

        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_xlabel("Simulation Iteration", fontsize=9.5, fontweight="bold")
        self.ax.set_ylabel("Plume Mesh Cell Size Δx [mm]", fontsize=9.5, fontweight="bold")
        self.ax.set_title("Adaptive Plume Mesh Cell Size vs. Simulation Iteration", fontsize=10.5, fontweight="bold", pad=28)
        self.ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
        self.ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            framealpha=0.95,
            fontsize=8.5
        )

        self.canvas.draw_idle()


class PlumeMeshSizeWindow(QDialog):
    """
    Dedicated second window displaying the Plume Mesh Cell Size diagnostic graph
    (Iterations on x-axis, Plume Cell Size Delta x [mm] on y-axis).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Plume Mesh Cell Size Monitor — (Iterations vs Plume Δx)")
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.resize(880, 580)
        self.setMinimumSize(680, 440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.plume_widget = PlumeMeshSizeGraphWidget(parent=self.parent_app, is_standalone=True)
        layout.addWidget(self.plume_widget)

    def update_plot(self, sim=None):
        self.plume_widget.update_plot(sim)


class MeshWindow(QDialog):
    """
    Popup window displaying the computational domain and mesh grid with
    spatial resolution, Debye length, and mesh dimension analytics,
    along with a second tab/window for Courant/Boris displacement substeps.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Computational Domain & Mesh Grid Inspector")
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.resize(940, 740)
        self.setMinimumSize(780, 560)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── Tab widget ────────────────────────────────────────────────
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cbd5e1;
                background: #ffffff;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #f1f5f9;
                color: #475569;
                padding: 7px 18px;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #cbd5e1;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0f172a;
                border-bottom: 2px solid #2563eb;
            }
            QTabBar::tab:hover {
                background: #e2e8f0;
            }
        """)

        # ── Tab 1: Computational Domain & Mesh Grid ───────────────────
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(8, 8, 8, 8)
        tab1_layout.setSpacing(8)

        # Controls toolbar
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

        self.btn_open_substeps = QPushButton("🗗 Open Substeps in Second Window")
        self.btn_open_substeps.setToolTip("Open Required Substeps graph in a separate floating window")
        self.btn_open_substeps.clicked.connect(self.open_substeps_window)
        ctrl_layout.addWidget(self.btn_open_substeps)

        self.btn_open_plume = QPushButton("🗗 Open Plume Δx in Second Window")
        self.btn_open_plume.setToolTip("Open Plume Cell Size vs. Iterations graph in a separate floating window")
        self.btn_open_plume.clicked.connect(self.open_plume_window)
        ctrl_layout.addWidget(self.btn_open_plume)

        self.btn_refresh = QPushButton("⟳ Refresh")
        self.btn_refresh.setToolTip("Re-read simulation state and redraw mesh")
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        ctrl_layout.addWidget(self.btn_refresh)

        tab1_layout.addLayout(ctrl_layout)

        # Matplotlib Canvas for Mesh
        self.fig = Figure(figsize=(8.5, 4.0), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(30)
            tab1_layout.addWidget(self.toolbar)

        tab1_layout.addWidget(self.canvas, stretch=1)

        # Under-the-graph Analytics Dashboard (Cards 1, 2, 3)
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

        # Card 1: Grid Spacing
        grp_dx = QGroupBox("Grid Spacing (Δx, Δy)")
        grp_dx.setStyleSheet(self._group_style())
        lay_dx = QVBoxLayout(grp_dx)
        self.lbl_dx_nominal = QLabel("Δx (base): —")
        self.lbl_dy_nominal = QLabel("Δy (base): —")
        self.lbl_dx_range   = QLabel("Δx min..max: —")
        self.lbl_ratio      = QLabel("Δx / λ_D: —")
        self.lbl_dt         = QLabel("Time Step dt: —")
        for lbl in (self.lbl_dx_nominal, self.lbl_dy_nominal, self.lbl_dx_range, self.lbl_ratio, self.lbl_dt):
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

        tab1_layout.addWidget(stats_frame)

        self.lbl_zones_summary = QLabel("Zone Breakdown: —")
        self.lbl_zones_summary.setStyleSheet(
            "font-family: monospace; font-size: 10px; color: #475569; "
            "background-color: #f1f5f9; padding: 4px 8px; border-radius: 4px; border: 1px solid #cbd5e1;"
        )
        tab1_layout.addWidget(self.lbl_zones_summary)

        self.tabs.addTab(tab1, "🌐 Computational Domain & Mesh Grid")

        # ── Tab 2: Displacement Substeps Monitor ──────────────────────
        self.substep_widget = SubstepGraphWidget(parent=self.parent_app, is_standalone=False)
        self.tabs.addTab(self.substep_widget, "⚡ Displacement Substeps (Iterations vs Substeps)")

        # ── Tab 3: Plume Mesh Cell Size Monitor ───────────────────────
        self.plume_mesh_widget = PlumeMeshSizeGraphWidget(parent=self.parent_app, is_standalone=False)
        self.tabs.addTab(self.plume_mesh_widget, "📈 Plume Cell Size History (Δx vs Iterations)")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self.tabs)

        # Cached state
        self._cached_sim = None
        self._cached_params = None
        self._substeps_dialog = None
        self._plume_size_dialog = None

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

    def open_substeps_window(self):
        """Open or focus the separate Displacement Substeps Monitor window."""
        if getattr(self, '_substeps_dialog', None) is None:
            self._substeps_dialog = DisplacementSubstepsWindow(parent=self.parent_app or self)
        self._substeps_dialog.show()
        self._substeps_dialog.raise_()
        self._substeps_dialog.activateWindow()
        sim = getattr(self, '_cached_sim', None)
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
        if sim is not None:
            self._substeps_dialog.update_plot(sim)

    def open_plume_window(self):
        """Open or focus the separate Plume Mesh Cell Size Monitor window."""
        if getattr(self, '_plume_size_dialog', None) is None:
            self._plume_size_dialog = PlumeMeshSizeWindow(parent=self.parent_app or self)
        self._plume_size_dialog.show()
        self._plume_size_dialog.raise_()
        self._plume_size_dialog.activateWindow()
        sim = getattr(self, '_cached_sim', None)
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
        if sim is not None:
            self._plume_size_dialog.update_plot(sim)

    def _on_tab_changed(self, idx):
        sim = getattr(self, '_cached_sim', None)
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
        params = getattr(self, '_cached_params', None)
        if params is None and self.parent_app and hasattr(self.parent_app, 'get_params'):
            params = self.parent_app.get_params()

        if idx == 0:
            if sim is not None:
                self.update_plot(sim, params=params)
        elif idx == 1:
            if hasattr(self, 'substep_widget'):
                self.substep_widget.update_plot(sim)
        elif idx == 2:
            if hasattr(self, 'plume_mesh_widget'):
                self.plume_mesh_widget.update_plot(sim)

    def _on_refresh_clicked(self):
        sim = getattr(self.parent_app, 'sim', None)
        params = self.parent_app.get_params() if hasattr(self.parent_app, 'get_params') else None
        if sim is not None:
            self.update_plot(sim, params=params)

    def _replot(self):
        if self._cached_sim is not None:
            self.update_plot(self._cached_sim, params=self._cached_params)

    def update_plot(self, sim, params=None):
        """Update domain visualization, mesh analytics, substeps graph, and plume cell size graph."""
        self._cached_sim = sim
        self._cached_params = params

        # Always update substeps and plume size graphs (and pop-out windows if active)
        if hasattr(self, 'substep_widget'):
            self.substep_widget.update_plot(sim)
        if getattr(self, '_substeps_dialog', None) is not None and self._substeps_dialog.isVisible():
            self._substeps_dialog.update_plot(sim)

        if hasattr(self, 'plume_mesh_widget'):
            self.plume_mesh_widget.update_plot(sim)
        if getattr(self, '_plume_size_dialog', None) is not None and self._plume_size_dialog.isVisible():
            self._plume_size_dialog.update_plot(sim)

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
            self.ax.vlines(
                x_coords, ymin=0, ymax=Ly,
                colors="#334155", linestyles="-", linewidths=0.45, alpha=0.35, zorder=2
            )
            self.ax.hlines(
                y_coords, xmin=0, xmax=Lx,
                colors="#334155", linestyles="-", linewidths=0.45, alpha=0.35, zorder=2
            )

        # ── 3. Grid Electrodes Overlay ────────────────────────────────
        if self.chk_electrodes.isChecked() and hasattr(sim, 'isBound') and sim.isBound is not None and np.any(sim.isBound):
            gy, gx = np.where(sim.isBound)
            if len(gx) > 0 and len(gy) > 0:
                valid_m = (gx < len(x_coords)) & (gy < len(y_coords))
                if np.any(valid_m):
                    self.ax.scatter(
                        x_coords[gx[valid_m]], y_coords[gy[valid_m]],
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
        eps0 = getattr(sim, 'eps0', 8.854187817e-12)
        q    = getattr(sim, 'q', 1.602176634e-19)

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

        lambda_D_m = math.sqrt(eps0 * Te_up * q / (n0_plasma * q**2)) if n0_plasma > 0 else 0.0
        lambda_D_mm = lambda_D_m * 1e3
        lambda_D_um = lambda_D_m * 1e6

        lambda_D_bohm_m = math.sqrt(eps0 * Te_up / (q * n0_plasma * 0.61)) if n0_plasma > 0 else 0.0
        lambda_D_bohm_mm = lambda_D_bohm_m * 1e3
        lambda_D_bohm_um = lambda_D_bohm_m * 1e6

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
        self.lbl_ratio.setText(f"Δx / λ_D ratio: <b>{ratio_dx_debye:.3f}</b>")
        sim_dt = getattr(sim, 'dt', 5e-10)
        cfl_v = (dx * 1e-3) / sim_dt if sim_dt > 0 else 0.0
        self.lbl_dt.setText(f"Time Step dt: <b>{sim_dt * 1e9:.3f} ns</b> (CFL {cfl_v:.1e} m/s)")

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
