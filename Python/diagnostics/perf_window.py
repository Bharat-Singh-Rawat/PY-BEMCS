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
from matplotlib.lines import Line2D
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
    Features clickable checkmarks ([x] / [ ]) embedded directly within each individual
    graph legend to toggle curve visibility dynamically.
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent_app = parent
        self.setWindowTitle("Runtime Performance & Particle Diagnostics Monitor")
        self.setGeometry(160, 140, 960, 660)

        layout = QVBoxLayout(self)

        # Controls row (no common top legend bar as per design specifications)
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

        # Curve visibility state per subplot (checkmarks directly inside each graph legend)
        self.curve_visibility = {
            'p1_ions': True,
            'p1_elec': True,
            'p1_warn': True,
            'p2_inj': True,
            'p2_lost_g': True,
            'p2_lost_o': True,
            'p2_tx': True,
            'p2_net': True,
            'p3_ppc_up': True,
            'p3_ppc_pl': True,
            'p3_target': True,
            'p4_dt': True,
            'p4_rss': True,
        }
        self._legend_pick_map = {}
        self._last_sim = None

        # Connect pick events on all plot legends to toggle visibility
        self.canvas.mpl_connect('pick_event', self._on_legend_pick)

    def _on_legend_pick(self, event):
        """Handle mouse clicks directly on legend text/lines in any graph to toggle visibility."""
        artist = event.artist
        key = self._legend_pick_map.get(artist)
        if key is not None and key in self.curve_visibility:
            self.curve_visibility[key] = not self.curve_visibility[key]
            sim = self._last_sim
            if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
                sim = self.parent_app.sim
            if sim is not None:
                self.update_plot(sim)
            else:
                self.canvas.draw_idle()

    def _build_interactive_legend(self, ax, quantities, loc='best'):
        """Build a clickable interactive legend with [x] / [ ] checkmarks for the given axis."""
        legend_handles = []
        legend_labels = []

        for key, name, color, ls, lw in quantities:
            active = self.curve_visibility.get(key, True)
            marker_str = "[x]" if active else "[ ]"
            alpha = 1.0 if active else 0.3
            label = f"{marker_str} {name}"

            handle = Line2D(
                [0], [0], color=color, linestyle=ls, lw=lw,
                alpha=alpha
            )
            legend_handles.append(handle)
            legend_labels.append(label)

        leg = ax.legend(
            legend_handles, legend_labels,
            fontsize=7, loc=loc, framealpha=0.85, labelspacing=0.25
        )

        if leg is not None:
            for leg_line, leg_text, (key, _, _, _, _) in zip(leg.get_lines(), leg.get_texts(), quantities):
                if not self.curve_visibility.get(key, True):
                    leg_text.set_alpha(0.35)
                leg_line.set_picker(True)
                leg_line.set_pickradius(6)
                leg_text.set_picker(True)

                self._legend_pick_map[leg_line] = key
                self._legend_pick_map[leg_text] = key

    def _refresh(self):
        sim = self._last_sim
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
        if sim is not None:
            self.update_plot(sim)

    def _clear_history(self):
        sim = self._last_sim
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
        if sim is not None:
            mon = getattr(sim, '_perf_monitor', None)
            if mon is not None:
                mon.history.clear()
        self._refresh()

    def _export_csv(self):
        sim = self._last_sim
        if sim is None and self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
        if sim is None:
            return
        mon = getattr(sim, '_perf_monitor', None)
        if mon is None or not mon.history:
            QMessageBox.warning(self, "No Data", "No performance history data to export yet.")
            return

        suggested = os.path.join(os.path.expanduser("~"), time.strftime("pybemcs_perf_%Y%m%d%H%M%S.csv"))
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Performance CSV", suggested, "CSV Files (*.csv)")
        if file_name:
            mon.export_csv(file_name)
            QMessageBox.information(self, "Export Successful", f"Performance log saved to:\n{file_name}")

    def update_plot(self, sim):
        self._last_sim = sim
        mon = getattr(sim, '_perf_monitor', None) if sim is not None else None
        if mon is None or not mon.history:
            for ax in self.axs.flat:
                ax.clear()
            self.axs[0, 0].set_title("No performance data yet — start simulation", fontsize=9)
            self.canvas.draw_idle()
            return

        self._legend_pick_map.clear()
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
        has_warn = (mon.warn_particles < np.max(num_ions) * 1.3)
        has_any_1 = False

        if self.curve_visibility.get('p1_ions', True):
            ax1.plot(x_axis, num_ions, color='#1f77b4', lw=1.8, zorder=3)
            has_any_1 = True
        if self.curve_visibility.get('p1_elec', True):
            ax1.plot(x_axis, num_elec, color='#ff7f0e', lw=1.4, zorder=3)
            has_any_1 = True
        if has_warn and self.curve_visibility.get('p1_warn', True):
            ax1.axhline(mon.warn_particles, color='red', linestyle=':', alpha=0.7, lw=1.2, zorder=2)
            has_any_1 = True

        if not has_any_1:
            if len(x_axis) > 1:
                ax1.set_xlim(x_axis[0], x_axis[-1])
            ax1.text(0.5, 0.5, "No quantities selected in legend",
                     ha='center', va='center', transform=ax1.transAxes,
                     color='gray', fontsize=8, fontstyle='italic')

        ax1.set_title("Particle Population", fontsize=9, fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=8)
        ax1.set_ylabel("Macroparticle Count", fontsize=8)
        ax1.grid(True, alpha=0.3)

        p1_quantities = [
            ('p1_ions', "Ions", '#1f77b4', '-', 1.8),
            ('p1_elec', "Electrons", '#ff7f0e', '-', 1.4),
        ]
        if has_warn:
            p1_quantities.append(('p1_warn', "Warn Limit", 'red', ':', 1.2))
        self._build_interactive_legend(ax1, p1_quantities, loc='upper left')

        # Plot 2: Flux & Balance
        ax2 = self.axs[0, 1]
        ax2.clear()
        has_any_2 = False

        if self.curve_visibility.get('p2_inj', True):
            ax2.plot(x_axis, injected, color='#2ca02c', lw=1.5, zorder=3)
            has_any_2 = True
        if self.curve_visibility.get('p2_lost_g', True):
            ax2.plot(x_axis, lost_g, color='#d62728', lw=1.2, zorder=3)
            has_any_2 = True
        if self.curve_visibility.get('p2_lost_o', True):
            ax2.plot(x_axis, lost_o, color='#e377c2', lw=1.2, zorder=3)
            has_any_2 = True
        if self.curve_visibility.get('p2_tx', True):
            ax2.plot(x_axis, tx, color='#9467bd', lw=1.2, zorder=3)
            has_any_2 = True
        if self.curve_visibility.get('p2_net', True):
            ax2.plot(x_axis, net, color='#333333', lw=1.0, linestyle='--', zorder=3)
            has_any_2 = True

        if not has_any_2:
            if len(x_axis) > 1:
                ax2.set_xlim(x_axis[0], x_axis[-1])
            ax2.text(0.5, 0.5, "No quantities selected in legend",
                     ha='center', va='center', transform=ax2.transAxes,
                     color='gray', fontsize=8, fontstyle='italic')

        ax2.set_title("Particle Flux & Loss Balance", fontsize=9, fontweight='bold')
        ax2.set_xlabel(x_label, fontsize=8)
        ax2.set_ylabel("Particles / step", fontsize=8)
        ax2.grid(True, alpha=0.3)

        p2_quantities = [
            ('p2_inj', "Injected / step", '#2ca02c', '-', 1.5),
            ('p2_lost_g', "Grid Loss", '#d62728', '-', 1.2),
            ('p2_lost_o', "OOB Loss", '#e377c2', '-', 1.2),
            ('p2_tx', "Transmitted", '#9467bd', '-', 1.2),
            ('p2_net', "Net Balance", '#333333', '--', 1.0),
        ]
        self._build_interactive_legend(ax2, p2_quantities, loc='upper right')

        # Plot 3: PPC Upstream vs Plume
        ax3 = self.axs[1, 0]
        ax3.clear()
        has_any_3 = False

        if self.curve_visibility.get('p3_ppc_up', True):
            ax3.plot(x_axis, ppc_up, color='#1f77b4', lw=1.6, zorder=3)
            has_any_3 = True
        if self.curve_visibility.get('p3_ppc_pl', True):
            ax3.plot(x_axis, ppc_pl, color='#2ca02c', lw=1.4, zorder=3)
            has_any_3 = True
        if self.curve_visibility.get('p3_target', True):
            ax3.axhline(3.0, color='red', linestyle='--', alpha=0.6, lw=1.2, zorder=2)
            has_any_3 = True

        if not has_any_3:
            if len(x_axis) > 1:
                ax3.set_xlim(x_axis[0], x_axis[-1])
            ax3.text(0.5, 0.5, "No quantities selected in legend",
                     ha='center', va='center', transform=ax3.transAxes,
                     color='gray', fontsize=8, fontstyle='italic')

        ax3.set_title("Spatial PPC Quality", fontsize=9, fontweight='bold')
        ax3.set_xlabel(x_label, fontsize=8)
        ax3.set_ylabel("Mean Particles / Cell", fontsize=8)
        ax3.grid(True, alpha=0.3)

        p3_quantities = [
            ('p3_ppc_up', "Upstream PPC", '#1f77b4', '-', 1.6),
            ('p3_ppc_pl', "Plume PPC", '#2ca02c', '-', 1.4),
            ('p3_target', "PPC Target (3)", 'red', '--', 1.2),
        ]
        self._build_interactive_legend(ax3, p3_quantities, loc='upper left')

        # Plot 4: Step Latency & Memory
        ax4 = self.axs[1, 1]
        ax4.clear()
        has_any_4 = False
        has_rss = np.any(rss_mb > 0)

        if self.curve_visibility.get('p4_dt', True):
            ax4.plot(x_axis, dt_ms, color='#8c564b', lw=1.4, zorder=3)
            has_any_4 = True
        if has_rss and self.curve_visibility.get('p4_rss', True):
            ax4.plot(x_axis, rss_mb, color='#17becf', lw=1.3, linestyle='--', zorder=3)
            has_any_4 = True

        if not has_any_4:
            if len(x_axis) > 1:
                ax4.set_xlim(x_axis[0], x_axis[-1])
            ax4.text(0.5, 0.5, "No quantities selected in legend",
                     ha='center', va='center', transform=ax4.transAxes,
                     color='gray', fontsize=8, fontstyle='italic')

        ax4.set_title("Execution Time & RAM Usage", fontsize=9, fontweight='bold')
        ax4.set_xlabel(x_label, fontsize=8)
        ax4.set_ylabel("Latency [ms] / RAM [MB]", fontsize=8)
        ax4.grid(True, alpha=0.3)

        p4_quantities = [
            ('p4_dt', "Step Time [ms]", '#8c564b', '-', 1.4),
        ]
        if has_rss:
            p4_quantities.append(('p4_rss', "RAM RSS [MB]", '#17becf', '--', 1.3))
        self._build_interactive_legend(ax4, p4_quantities, loc='upper right')

        self.canvas.draw_idle()
