"""
Total Charge Monitor Diagnostic Window.

Tracks the time evolution of the PIC system's total charge budget:
    - Ion space charge (Q_ions > 0)
    - Kinetic electron space charge (Q_elec <= 0, e.g. neutralizer)
    - Fluid Boltzmann electron continuum (Q_boltzmann <= 0, upstream presheath)
    - Net particle charge (Q_free = Q_ions + Q_elec)
    - Total net space charge (Q_net = Q_free + Q_boltzmann)

Features an interactive checkmark legend allowing the user to selectively display/hide
any individual charge component in the graph, with synchronized controls in both
the UI legend bar and the plot's clickable legend.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QFrame
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QFrame
    )

try:
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    try:
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    except Exception:
        NavigationToolbar = None


class TotalChargeMonitorWindow(QWidget):
    """
    Pop-up window showing the time evolution of the PIC system's total charge budget:
        - Ion space charge (Q_ions > 0)
        - Kinetic electron space charge (Q_elec <= 0, e.g. neutralizer)
        - Fluid Boltzmann electron continuum (Q_boltzmann <= 0, upstream presheath)
        - Net particle charge (Q_free = Q_ions + Q_elec)
        - Total net space charge (Q_net = Q_free + Q_boltzmann)

    Displays real-time charge balance to monitor beam neutralization, space-charge
    accumulation, and steady-state charge equilibrium.
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent_app = parent
        self.setWindowTitle("Total Charge Monitor")
        self.setGeometry(160, 160, 860, 560)

        layout = QVBoxLayout(self)

        # --- Row 1: General Controls ---
        ctrl = QHBoxLayout()
        self.chk_normalize = QCheckBox("Normalize to Peak")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_normalize)

        btn_clear = QPushButton("Clear History")
        btn_clear.clicked.connect(self._clear_history)
        ctrl.addWidget(btn_clear)

        btn_all = QPushButton("Show All")
        btn_all.setToolTip("Enable all charge curves")
        btn_all.clicked.connect(lambda: self._set_all_visible(True))
        ctrl.addWidget(btn_all)

        btn_none = QPushButton("Hide All")
        btn_none.setToolTip("Disable all charge curves")
        btn_none.clicked.connect(lambda: self._set_all_visible(False))
        ctrl.addWidget(btn_none)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Checkbox states for individual curves (toggled via in-graph clickable legend)
        self.chk_net = QCheckBox("Total Net Charge")
        self.chk_net.setChecked(True)
        self.chk_net.stateChanged.connect(self._refresh)

        self.chk_q_i = QCheckBox("Ions (+)")
        self.chk_q_i.setChecked(True)
        self.chk_q_i.stateChanged.connect(self._refresh)

        self.chk_q_e = QCheckBox("Kinetic e⁻ (-)")
        self.chk_q_e.setChecked(True)
        self.chk_q_e.stateChanged.connect(self._refresh)

        self.chk_q_b = QCheckBox("Boltzmann e⁻ (-)")
        self.chk_q_b.setChecked(True)
        self.chk_q_b.stateChanged.connect(self._refresh)

        self.chk_q_free = QCheckBox("Free Ptcls (i + e)")
        self.chk_q_free.setChecked(False)
        self.chk_q_free.stateChanged.connect(self._refresh)

        self.chk_neutrality = QCheckBox("Neutrality (Q=0)")
        self.chk_neutrality.setChecked(True)
        self.chk_neutrality.stateChanged.connect(self._refresh)

        # --- Matplotlib Canvas and Navigation Toolbar ---
        self.fig = Figure(figsize=(8.0, 4.4), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # Toolbar placed under controls ("the house, left/right arrows and so on")
        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(30)
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None

        # --- Stats label ---
        self.lbl_stats = QLabel("Net Charge: -  |  Q_ions: -  |  Q_elec: -  |  Delta(last step): -")
        self.lbl_stats.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #f4f6f9; "
            "padding: 5px; border: 1px solid #d0d7de; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_stats)

        layout.addWidget(self.canvas, stretch=1)

        # Connect pick events on the plot's legend to toggle the checkboxes
        self.canvas.mpl_connect('pick_event', self._on_legend_pick)
        self._legend_pick_map = {}

    @property
    def chk_components(self):
        """Backward compatibility helper for component checkboxes."""
        class _ComponentsProxy:
            def __init__(self, parent_win):
                self._p = parent_win
            def isChecked(self):
                return (self._p.chk_q_i.isChecked() or
                        self._p.chk_q_e.isChecked() or
                        self._p.chk_q_b.isChecked() or
                        self._p.chk_q_free.isChecked())
            def setChecked(self, val):
                self._p.chk_q_i.setChecked(bool(val))
                self._p.chk_q_e.setChecked(bool(val))
                self._p.chk_q_b.setChecked(bool(val))
        return _ComponentsProxy(self)

    @property
    def chk_boltzmann(self):
        """Backward compatibility proxy for Boltzmann continuum."""
        return self.chk_q_b

    def _set_all_visible(self, visible: bool):
        """Enable or disable all curve checkboxes at once."""
        self.chk_net.blockSignals(True)
        self.chk_q_i.blockSignals(True)
        self.chk_q_e.blockSignals(True)
        self.chk_q_b.blockSignals(True)
        self.chk_q_free.blockSignals(True)
        self.chk_neutrality.blockSignals(True)

        self.chk_net.setChecked(visible)
        self.chk_q_i.setChecked(visible)
        self.chk_q_e.setChecked(visible)
        self.chk_q_b.setChecked(visible)
        self.chk_q_free.setChecked(visible)
        self.chk_neutrality.setChecked(visible)

        self.chk_net.blockSignals(False)
        self.chk_q_i.blockSignals(False)
        self.chk_q_e.blockSignals(False)
        self.chk_q_b.blockSignals(False)
        self.chk_q_free.blockSignals(False)
        self.chk_neutrality.blockSignals(False)

        self._refresh()

    def _on_legend_pick(self, event):
        """Handle mouse clicks directly on legend text/lines in the plot to toggle visibility."""
        artist = event.artist
        chk = self._legend_pick_map.get(artist)
        if chk is not None:
            chk.setChecked(not chk.isChecked())

    def _clear_history(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            sim = self.parent_app.sim
            if hasattr(sim, 'charge_history_t'):
                sim.charge_history_t.clear()
                sim.charge_history_q_i.clear()
                sim.charge_history_q_e.clear()
                sim.charge_history_q_b.clear()
                sim.charge_history_q_free.clear()
                sim.charge_history_q_net.clear()
                sim._prev_total_charge = None
        self._refresh()

    def _refresh(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            self.update_plot(self.parent_app.sim)

    @staticmethod
    def _format_charge(q):
        """Format charge with SI prefixes: pC, nC, uC, or C."""
        abs_q = abs(q)
        if abs_q < 1e-15:
            return "0.00 C"
        elif abs_q < 1e-9:
            return f"{q*1e12:+.2f} pC"
        elif abs_q < 1e-6:
            return f"{q*1e9:+.3f} nC"
        elif abs_q < 1e-3:
            return f"{q*1e6:+.3f} uC"
        else:
            return f"{q:+.4e} C"

    def update_plot(self, sim):
        """Redraw the charge-vs-time graph from sim.charge_history_*."""
        if sim is None or not hasattr(sim, 'charge_history_t'):
            return

        t = np.asarray(sim.charge_history_t, dtype=float)
        q_i = np.asarray(sim.charge_history_q_i, dtype=float)
        q_e = np.asarray(sim.charge_history_q_e, dtype=float)
        q_b = np.asarray(sim.charge_history_q_b, dtype=float)
        q_free = np.asarray(sim.charge_history_q_free, dtype=float)
        q_net = np.asarray(sim.charge_history_q_net, dtype=float)

        min_n = min(len(t), len(q_i), len(q_e), len(q_b), len(q_free), len(q_net))
        if min_n == 0:
            self.ax.clear()
            self.ax.set_title("No charge data yet — start simulation")
            self.canvas.draw_idle()
            return

        t = t[:min_n]
        q_i = q_i[:min_n]
        q_e = q_e[:min_n]
        q_b = q_b[:min_n]
        q_free = q_free[:min_n]
        q_net = q_net[:min_n]

        t_us = t * 1e6   # convert s -> us for display

        # Visible arrays for scale computation
        active_series = []
        if self.chk_net.isChecked():
            active_series.append(q_net)
        if self.chk_q_i.isChecked():
            active_series.append(q_i)
        if self.chk_q_e.isChecked():
            active_series.append(q_e)
        if self.chk_q_b.isChecked():
            active_series.append(q_b)
        if self.chk_q_free.isChecked():
            active_series.append(q_free)

        # Auto-scaling unit prefix for the plot axis based on active series
        if active_series:
            max_val = max(np.max(np.abs(s)) if len(s) > 0 else 0.0 for s in active_series)
        else:
            max_val = np.max(np.abs(q_net)) if len(q_net) > 0 else 0.0

        normalize = self.chk_normalize.isChecked()
        if normalize:
            ref = max_val if max_val > 0 else 1.0
            scale = 1.0 / ref
            ylabel = "Charge (normalized to peak)"
        elif max_val < 1e-9:
            scale = 1e12
            ylabel = "Charge [pC]"
        elif max_val < 1e-6:
            scale = 1e9
            ylabel = "Charge [nC]"
        elif max_val < 1e-3:
            scale = 1e6
            ylabel = "Charge [uC]"
        else:
            scale = 1.0
            ylabel = "Charge [C]"

        # Stats label update
        last_net = float(q_net[-1]) if len(q_net) > 0 else 0.0
        last_qi = float(q_i[-1]) if len(q_i) > 0 else 0.0
        last_qe = float(q_e[-1]) if len(q_e) > 0 else 0.0
        last_qb = float(q_b[-1]) if len(q_b) > 0 else 0.0

        if len(q_net) > 1 and abs(q_net[-2]) > 1e-15:
            delta_rel = (q_net[-1] - q_net[-2]) / abs(q_net[-2]) * 100.0
            delta_str = f"{delta_rel:+.2f}%"
        else:
            delta_str = "-"

        stats_text = (
            f"Net Charge: {self._format_charge(last_net)}  |  "
            f"Q_ions: {self._format_charge(last_qi)}  |  "
            f"Q_elec: {self._format_charge(last_qe)}  |  "
            f"Q_boltz: {self._format_charge(last_qb)}  |  "
            f"Delta(last step): {delta_str}"
        )
        self.lbl_stats.setText(stats_text)

        # Check if user zoomed/panned via the toolbar
        user_zoomed = False
        cur_xlim = None
        cur_ylim = None
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            nav_stack_len = len(getattr(self.toolbar, '_nav_stack', []))
            is_active_tool = getattr(self.toolbar, 'mode', '') != ''
            if nav_stack_len > 1 or is_active_tool:
                user_zoomed = True
                cur_xlim = self.ax.get_xlim()
                cur_ylim = self.ax.get_ylim()

        # Plotting
        self.ax.clear()

        # Zero reference line
        if self.chk_neutrality.isChecked():
            self.ax.axhline(0, color='#64748b', linestyle=':', lw=1.0, alpha=0.7, zorder=2)

        has_any_curve = False

        if self.chk_net.isChecked():
            self.ax.plot(t_us, q_net * scale, color='#1f77b4', lw=2.2, zorder=4)
            has_any_curve = True

        if self.chk_q_i.isChecked():
            self.ax.plot(t_us, q_i * scale, color='#ff7f0e', lw=1.2,
                         linestyle='--', zorder=3)
            has_any_curve = True

        if self.chk_q_e.isChecked():
            self.ax.plot(t_us, q_e * scale, color='#2ca02c', lw=1.2,
                         linestyle='--', zorder=3)
            has_any_curve = True

        if self.chk_q_b.isChecked():
            self.ax.plot(t_us, q_b * scale, color='#9467bd', lw=1.2,
                         linestyle='--', zorder=3)
            has_any_curve = True

        if self.chk_q_free.isChecked():
            self.ax.plot(t_us, q_free * scale, color='#17becf', lw=1.0,
                         linestyle=':', zorder=3)
            has_any_curve = True

        if not has_any_curve and not self.chk_neutrality.isChecked():
            self.ax.text(
                0.5, 0.5, "No quantities selected in legend",
                ha='center', va='center', transform=self.ax.transAxes,
                color='gray', fontsize=10, fontstyle='italic'
            )

        self.ax.set_xlabel("Simulation Time [µs]")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title("PIC Total Charge Monitor", fontsize=10)
        self.ax.grid(True, alpha=0.3)

        if user_zoomed and cur_xlim is not None and cur_ylim is not None:
            self.ax.set_xlim(cur_xlim)
            self.ax.set_ylim(cur_ylim)

        # --- Interactive Legend with Checkmarks [x] / [ ] ---
        self._legend_pick_map.clear()

        quantities = [
            (self.chk_net, "Total Net Charge", '#1f77b4', '-', 2.2),
            (self.chk_q_i, "Ions (+)", '#ff7f0e', '--', 1.2),
            (self.chk_q_e, "Kinetic e- (-)", '#2ca02c', '--', 1.2),
            (self.chk_q_b, "Boltzmann e- (-)", '#9467bd', '--', 1.2),
            (self.chk_q_free, "Free Ptcls (i + e)", '#17becf', ':', 1.0),
            (self.chk_neutrality, "Neutrality (Q=0)", '#64748b', ':', 1.0),
        ]

        legend_handles = []
        legend_labels = []

        for chk, name, color, ls, lw in quantities:
            active = chk.isChecked()
            marker_str = "[x]" if active else "[ ]"
            alpha = 1.0 if active else 0.3
            label = f"{marker_str} {name}"

            handle = Line2D(
                [0], [0], color=color, linestyle=ls, lw=lw,
                alpha=alpha
            )
            legend_handles.append(handle)
            legend_labels.append(label)

        leg = self.ax.legend(
            legend_handles, legend_labels,
            fontsize=8, loc='best', framealpha=0.85
        )

        if leg is not None:
            for leg_line, leg_text, (chk, _, _, _, _) in zip(leg.get_lines(), leg.get_texts(), quantities):
                if not chk.isChecked():
                    leg_text.set_alpha(0.35)
                leg_line.set_picker(True)
                leg_line.set_pickradius(6)
                leg_text.set_picker(True)

                self._legend_pick_map[leg_line] = chk
                self._legend_pick_map[leg_text] = chk

        self.fig.tight_layout()
        self.canvas.draw_idle()


# Alias for convenience
ChargeMonitorWindow = TotalChargeMonitorWindow
