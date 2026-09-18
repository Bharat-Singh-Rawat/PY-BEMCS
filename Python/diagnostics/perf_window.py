"""
Runtime Performance & Particle Diagnostics Monitor Window.

Pop-up window displaying real-time simulation performance diagnostics:
    - Macroparticle populations (Ions and Electrons) vs time
    - Step flux balance: Injected vs. Losses (Grid, OOB, Transmitted)
    - Spatial PPC quality: Mean PPC in Upstream and Plume zones
    - Computational footprint: Step wall-clock latency (ms) and RAM RSS (MB)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFileDialog
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFileDialog
    )


class PerformanceMonitorWindow(QWidget):
    """
    Pop-up window displaying real-time simulation performance diagnostics:
        - Macroparticle populations (Ions and Electrons) vs time
        - Step flux balance: Injected vs. Losses (Grid, OOB, Transmitted)
        - Spatial PPC quality: Mean PPC in Upstream and Plume zones
        - Computational footprint: Step wall-clock latency (ms) and RAM RSS (MB)

    Also provides one-click export of complete runtime performance metrics to CSV.
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent_app = parent
        self.setWindowTitle("Runtime Performance & Particle Diagnostics Monitor")
        self.setGeometry(160, 140, 960, 660)

        layout = QVBoxLayout(self)

        # Controls row
        ctrl = QHBoxLayout()
        btn_export = QPushButton("Export Performance Log (.csv)")
        btn_export.clicked.connect(self._export_csv)
        ctrl.addWidget(btn_export)

        btn_clear = QPushButton("Clear History")
        btn_clear.clicked.connect(self._clear_history)
        ctrl.addWidget(btn_clear)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh)
        ctrl.addWidget(btn_refresh)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Stats header label
        self.lbl_stats = QLabel("Ions: — | Electrons: — | Net Balance: — | PPC Upstream: — | Step: — ms | RSS: — MB")
        self.lbl_stats.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #f4f6f9; "
            "padding: 6px; border: 1px solid #d0d7de; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_stats)

        # Matplotlib figure (2x2 subplots)
        self.fig, self.axs = plt.subplots(2, 2, figsize=(9.2, 5.6))
        self.fig.subplots_adjust(hspace=0.36, wspace=0.28, left=0.08, right=0.96, top=0.93, bottom=0.09)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def _refresh(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            self.update_plot(self.parent_app.sim)

    def _clear_history(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            mon = getattr(self.parent_app.sim, '_perf_monitor', None)
            if mon is not None:
                mon.history.clear()
        self._refresh()

    def _export_csv(self):
        if not (self.parent_app and hasattr(self.parent_app, 'sim')):
            return
        mon = getattr(self.parent_app.sim, '_perf_monitor', None)
        if mon is None or not mon.history:
            QMessageBox.warning(self, "No Data", "No performance history data to export yet.")
            return

        suggested = os.path.join(os.path.expanduser("~"), time.strftime("pybemcs_perf_%Y%m%d%H%M%S.csv"))
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Performance CSV", suggested, "CSV Files (*.csv)")
        if file_name:
            mon.export_csv(file_name)
            QMessageBox.information(self, "Export Successful", f"Performance log saved to:\n{file_name}")

    def update_plot(self, sim):
        mon = getattr(sim, '_perf_monitor', None) if sim is not None else None
        if mon is None or not mon.history:
            for ax in self.axs.flat:
                ax.clear()
            self.axs[0, 0].set_title("No performance data yet — start simulation", fontsize=9)
            self.canvas.draw_idle()
            return

        history = mon.history
        # Subsample if history is very large to keep GUI responsive
        n = len(history)
        if n > 800:
            stride = max(1, n // 500)
            data = history[::stride]
            if data[-1] is not history[-1]:
                data.append(history[-1])
        else:
            data = history

        iters = np.array([d.iteration for d in data])
        t_us = np.array([d.sim_time_s * 1e6 for d in data])
        x_axis = t_us if (t_us[-1] > 0) else iters
        x_label = "Sim Time [µs]" if (t_us[-1] > 0) else "Iteration"

        num_ions = np.array([d.num_ions for d in data])
        num_elec = np.array([d.num_electrons for d in data])
        injected = np.array([d.injected_this_step for d in data])
        lost_g = np.array([d.lost_to_grid for d in data])
        lost_o = np.array([d.lost_to_oob for d in data])
        tx = np.array([d.transmitted for d in data])
        net = np.array([d.net_balance for d in data])
        ppc_up = np.array([d.mean_ppc_upstream for d in data])
        ppc_pl = np.array([d.mean_ppc_plume for d in data])
        dt_ms = np.array([d.wall_time_ms for d in data])
        rss_mb = np.array([d.memory_rss_mb for d in data])

        last = history[-1]
        warn_text = ""
        is_warn = False
        if last.num_ions > mon.warn_particles or last.num_electrons > mon.warn_particles:
            warn_text += f" | ⚠️ Particle Limit Exceeded (>{mon.warn_particles:,})"
            is_warn = True
        if last.memory_rss_mb > mon.warn_memory_mb:
            warn_text += f" | ⚠️ RAM Limit Exceeded (>{mon.warn_memory_mb:.0f} MB)"
            is_warn = True
        p_status = getattr(last, 'poisson_status', 'converged')
        if p_status == 'diverged':
            warn_text += f" | ⚠️ Poisson DIVERGED (ΔV={last.poisson_delta_V:.2f}V)"
            is_warn = True
        elif p_status == 'stagnated':
            warn_text += f" | ⚠️ Poisson STAGNATED (ΔV={last.poisson_delta_V:.2f}V)"
            is_warn = True
        elif last.poisson_delta_V > 1.0:
            warn_text += f" | ⚠️ Poisson Under-converged (ΔV={last.poisson_delta_V:.2f}V)"
            is_warn = True

        status_tag = f" [{p_status}]" if p_status != 'converged' else ""
        rms_str = f", rms={last.poisson_rms*1000:.1f}mV" if hasattr(last, 'poisson_rms') and last.poisson_rms > 0 else ""
        self.lbl_stats.setText(
            f"Iter: {last.iteration} | "
            f"Ions: {last.num_ions:,} | e-: {last.num_electrons:,} | "
            f"Inj/step: {last.injected_this_step} | Lost(g+o): {last.lost_to_grid}+{last.lost_to_oob} | "
            f"Net: {last.net_balance:+d} | "
            f"PPC Up: {last.mean_ppc_upstream:.1f} | "
            f"Poisson: {last.poisson_iters} iters{status_tag} (ΔV={last.poisson_delta_V:.3f}V{rms_str}) | "
            f"Step: {last.wall_time_ms:.1f} ms | RSS: {last.memory_rss_mb:.0f} MB"
            + warn_text
        )
        if is_warn:
            self.lbl_stats.setStyleSheet(
                "font-family: monospace; font-size: 11px; background: #fff3cd; "
                "padding: 6px; border: 1px solid #ffc107; border-radius: 4px;"
            )
        else:
            self.lbl_stats.setStyleSheet(
                "font-family: monospace; font-size: 11px; background: #f4f6f9; "
                "padding: 6px; border: 1px solid #d0d7de; border-radius: 4px;"
            )

        # Plot 1: Particle Populations
        ax1 = self.axs[0, 0]
        ax1.clear()
        ax1.plot(x_axis, num_ions, color='#1f77b4', lw=1.8, label="Ions")
        ax1.plot(x_axis, num_elec, color='#ff7f0e', lw=1.4, label="Electrons")
        if mon.warn_particles < np.max(num_ions) * 1.3:
            ax1.axhline(mon.warn_particles, color='red', linestyle=':', alpha=0.7, label="Warn Limit")
        ax1.set_title("Particle Population", fontsize=9, fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=8)
        ax1.set_ylabel("Macroparticle Count", fontsize=8)
        ax1.legend(fontsize=7, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Flux & Balance
        ax2 = self.axs[0, 1]
        ax2.clear()
        ax2.plot(x_axis, injected, color='#2ca02c', lw=1.5, label="Injected / step")
        ax2.plot(x_axis, lost_g, color='#d62728', lw=1.2, label="Grid Loss")
        ax2.plot(x_axis, lost_o, color='#e377c2', lw=1.2, label="OOB Loss")
        ax2.plot(x_axis, tx, color='#9467bd', lw=1.2, label="Transmitted")
        ax2.plot(x_axis, net, color='#333333', lw=1.0, linestyle='--', label="Net Balance")
        ax2.set_title("Particle Flux & Loss Balance", fontsize=9, fontweight='bold')
        ax2.set_xlabel(x_label, fontsize=8)
        ax2.set_ylabel("Particles / step", fontsize=8)
        ax2.legend(fontsize=7, loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Plot 3: PPC Upstream vs Plume
        ax3 = self.axs[1, 0]
        ax3.clear()
        ax3.plot(x_axis, ppc_up, color='#1f77b4', lw=1.6, label="Upstream PPC")
        ax3.plot(x_axis, ppc_pl, color='#2ca02c', lw=1.4, label="Plume PPC")
        ax3.axhline(3.0, color='red', linestyle='--', alpha=0.6, label="PPC Target (3)")
        ax3.set_title("Spatial PPC Quality", fontsize=9, fontweight='bold')
        ax3.set_xlabel(x_label, fontsize=8)
        ax3.set_ylabel("Mean Particles / Cell", fontsize=8)
        ax3.legend(fontsize=7, loc='upper left')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Step Latency & Memory
        ax4 = self.axs[1, 1]
        ax4.clear()
        ax4.plot(x_axis, dt_ms, color='#8c564b', lw=1.4, label="Step Time [ms]")
        if np.any(rss_mb > 0):
            ax4.plot(x_axis, rss_mb, color='#17becf', lw=1.3, linestyle='--', label="RAM RSS [MB]")
        ax4.set_title("Execution Time & RAM Usage", fontsize=9, fontweight='bold')
        ax4.set_xlabel(x_label, fontsize=8)
        ax4.set_ylabel("Latency [ms] / RAM [MB]", fontsize=8)
        ax4.legend(fontsize=7, loc='upper right')
        ax4.grid(True, alpha=0.3)

        self.canvas.draw_idle()
