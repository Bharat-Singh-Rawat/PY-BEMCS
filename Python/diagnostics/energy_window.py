"""
Total Energy Conservation Monitor Diagnostic Window.

Tracks the time evolution of the PIC system's energy budget:
    - Ion kinetic energy (KE_ions)
    - Electron kinetic energy (KE_elec)
    - Electrostatic field energy (E_field)
    - Total energy = KE_ions + KE_elec + E_field

Features an interactive checkmark legend allowing the user to selectively display/hide
any individual energy component or warning markers in the graph, with synchronized
controls in both the UI legend bar and the plot's clickable legend.
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


class PhysicalConstraintsWindow(QWidget):
    """
    Pop-up window showing the time evolution of the PIC system's energy budget:
        - Ion kinetic energy (KE_ions)
        - Electron kinetic energy (KE_elec)
        - Electrostatic field energy (E_field)
        - Total energy = KE_ions + KE_elec + E_field

    In an ideal PIC simulation total energy should be approximately conserved.
    Abrupt changes (>5% per step) are flagged by red vertical markers AND
    by a terminal warning (printed regardless of whether this window is open).
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent_app = parent
        self.setWindowTitle("Total Energy Conservation Monitor")
        self.setGeometry(150, 150, 840, 560)

        layout = QVBoxLayout(self)

        # --- Row 1: General Controls ---
        ctrl = QHBoxLayout()
        self.chk_normalize = QCheckBox("Normalize to Initial Energy (E/E₀)")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_normalize)

        btn_clear = QPushButton("Clear History")
        btn_clear.clicked.connect(self._clear_history)
        ctrl.addWidget(btn_clear)

        btn_all = QPushButton("Show All")
        btn_all.setToolTip("Enable all energy components")
        btn_all.clicked.connect(lambda: self._set_all_visible(True))
        ctrl.addWidget(btn_all)

        btn_none = QPushButton("Hide All")
        btn_none.setToolTip("Disable all components")
        btn_none.clicked.connect(lambda: self._set_all_visible(False))
        ctrl.addWidget(btn_none)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # --- Row 2: Checkmark Legend Bar ---
        leg_bar = QFrame()
        leg_bar.setStyleSheet(
            "QFrame { background-color: #f8fafc; border: 1px solid #e2e8f0; "
            "border-radius: 6px; padding: 4px; }"
        )
        leg_layout = QHBoxLayout(leg_bar)
        leg_layout.setContentsMargins(6, 2, 6, 2)

        lbl_leg = QLabel("<b>Legend & Curves:</b>")
        lbl_leg.setStyleSheet("color: #334155; font-size: 11px;")
        leg_layout.addWidget(lbl_leg)

        # Individual Checkmarks for each quantity
        self.chk_total = QCheckBox("Total (KE + Field)")
        self.chk_total.setChecked(True)
        self.chk_total.setStyleSheet("QCheckBox { font-weight: bold; color: #1f77b4; font-size: 11px; }")
        self.chk_total.stateChanged.connect(self._refresh)
        leg_layout.addWidget(self.chk_total)

        self.chk_ke_i = QCheckBox("KE ions")
        self.chk_ke_i.setChecked(True)
        self.chk_ke_i.setStyleSheet("QCheckBox { font-weight: bold; color: #ff7f0e; font-size: 11px; }")
        self.chk_ke_i.stateChanged.connect(self._refresh)
        leg_layout.addWidget(self.chk_ke_i)

        self.chk_ke_e = QCheckBox("KE electrons")
        self.chk_ke_e.setChecked(True)
        self.chk_ke_e.setStyleSheet("QCheckBox { font-weight: bold; color: #2ca02c; font-size: 11px; }")
        self.chk_ke_e.stateChanged.connect(self._refresh)
        leg_layout.addWidget(self.chk_ke_e)

        self.chk_fe = QCheckBox("E field")
        self.chk_fe.setChecked(True)
        self.chk_fe.setStyleSheet("QCheckBox { font-weight: bold; color: #9467bd; font-size: 11px; }")
        self.chk_fe.stateChanged.connect(self._refresh)
        leg_layout.addWidget(self.chk_fe)

        self.chk_warnings = QCheckBox("Spikes (>5%)")
        self.chk_warnings.setChecked(True)
        self.chk_warnings.setStyleSheet("QCheckBox { font-weight: bold; color: #d62728; font-size: 11px; }")
        self.chk_warnings.stateChanged.connect(self._refresh)
        leg_layout.addWidget(self.chk_warnings)

        leg_layout.addStretch()
        layout.addWidget(leg_bar)

        # --- Stats label ---
        self.lbl_stats = QLabel("Total Energy: — J  |  Delta(last step): —%  |  Warnings: 0")
        self.lbl_stats.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #f4f6f9; "
            "padding: 5px; border: 1px solid #d0d7de; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_stats)

        # --- Matplotlib figure ---
        self.fig = plt.figure(figsize=(8.0, 4.4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Connect pick events on the plot's legend to toggle the checkboxes
        self.canvas.mpl_connect('pick_event', self._on_legend_pick)
        self._legend_pick_map = {}

        self._warn_count = 0

    @property
    def chk_components(self):
        """Backward compatibility helper representing the individual component checkboxes."""
        class _ComponentsProxy:
            def __init__(self, parent_win):
                self._p = parent_win
            def isChecked(self):
                return (self._p.chk_ke_i.isChecked() or
                        self._p.chk_ke_e.isChecked() or
                        self._p.chk_fe.isChecked())
            def setChecked(self, val):
                self._p.chk_ke_i.setChecked(bool(val))
                self._p.chk_ke_e.setChecked(bool(val))
                self._p.chk_fe.setChecked(bool(val))
        return _ComponentsProxy(self)

    def _set_all_visible(self, visible: bool):
        """Enable or disable all curve checkboxes at once."""
        self.chk_total.blockSignals(True)
        self.chk_ke_i.blockSignals(True)
        self.chk_ke_e.blockSignals(True)
        self.chk_fe.blockSignals(True)
        self.chk_warnings.blockSignals(True)

        self.chk_total.setChecked(visible)
        self.chk_ke_i.setChecked(visible)
        self.chk_ke_e.setChecked(visible)
        self.chk_fe.setChecked(visible)
        self.chk_warnings.setChecked(visible)

        self.chk_total.blockSignals(False)
        self.chk_ke_i.blockSignals(False)
        self.chk_ke_e.blockSignals(False)
        self.chk_fe.blockSignals(False)
        self.chk_warnings.blockSignals(False)

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
            sim.energy_history_t.clear()
            sim.energy_history_ke_i.clear()
            sim.energy_history_ke_e.clear()
            sim.energy_history_fe.clear()
            sim.energy_history_tot.clear()
            sim._prev_total_energy = None
            self._warn_count = 0
        self._refresh()

    def _refresh(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            self.update_plot(self.parent_app.sim)

    def update_plot(self, sim):
        """Redraw the energy-vs-time graph from sim.energy_history_*."""
        if sim is None or not hasattr(sim, 'energy_history_t'):
            return

        t = np.asarray(sim.energy_history_t, dtype=float)
        ke_i = np.asarray(sim.energy_history_ke_i, dtype=float)
        ke_e = np.asarray(sim.energy_history_ke_e, dtype=float)
        fe = np.asarray(sim.energy_history_fe, dtype=float)
        tot = np.asarray(sim.energy_history_tot, dtype=float)

        if len(t) == 0:
            self.ax.clear()
            self.ax.set_title("No energy data yet — start simulation")
            self.canvas.draw_idle()
            return

        t_us = t * 1e6   # convert s -> us for display

        normalize = self.chk_normalize.isChecked()
        ref = tot[0] if (normalize and tot[0] != 0) else 1.0

        # Identify warning steps (>5% relative change)
        warn_steps = []
        threshold = getattr(sim, 'energy_warning_threshold', 0.05)
        if len(tot) > 1:
            diffs = np.abs(np.diff(tot))
            refs = np.abs(tot[:-1])
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(refs > 0, diffs / refs, 0.0)
            warn_steps = np.where(rel > threshold)[0] + 1  # index of the offending point
        self._warn_count = len(warn_steps)

        # Update stats label
        last_tot = float(tot[-1])
        if len(tot) > 1 and tot[-2] != 0:
            last_rel = (tot[-1] - tot[-2]) / abs(tot[-2]) * 100.0
            delta_str = f"{last_rel:+.2f}%"
        else:
            delta_str = "—"
        self.lbl_stats.setText(
            f"Total Energy: {last_tot:.4e} J  |  "
            f"Delta(last step): {delta_str}  |  "
            f"Warnings (>{threshold*100:.0f}%): {self._warn_count}"
        )
        if self._warn_count > 0:
            self.lbl_stats.setStyleSheet(
                "font-family: monospace; font-size: 11px; background: #fff3cd; "
                "padding: 5px; border: 1px solid #ffc107; border-radius: 4px;"
            )
        else:
            self.lbl_stats.setStyleSheet(
                "font-family: monospace; font-size: 11px; background: #f4f6f9; "
                "padding: 5px; border: 1px solid #d0d7de; border-radius: 4px;"
            )

        # Plot
        self.ax.clear()

        if normalize:
            plot_tot = tot / ref
            plot_ki = ke_i / ref
            plot_ke = ke_e / ref
            plot_fe = fe / ref
            ylabel = "Energy (normalized to E₀)"
        else:
            plot_tot = tot
            plot_ki = ke_i
            plot_ke = ke_e
            plot_fe = fe
            ylabel = "Energy [J]"

        # --- Draw Curves based on checkmarks ---
        has_any_curve = False

        if self.chk_total.isChecked():
            self.ax.plot(t_us, plot_tot, color='#1f77b4', lw=2.2, zorder=4)
            has_any_curve = True

        if self.chk_ke_i.isChecked():
            self.ax.plot(t_us, plot_ki, color='#ff7f0e', lw=1.2,
                         linestyle='--', zorder=3)
            has_any_curve = True

        if self.chk_ke_e.isChecked():
            self.ax.plot(t_us, plot_ke, color='#2ca02c', lw=1.2,
                         linestyle='--', zorder=3)
            has_any_curve = True

        if self.chk_fe.isChecked():
            self.ax.plot(t_us, plot_fe, color='#9467bd', lw=1.2,
                         linestyle='--', zorder=3)
            has_any_curve = True

        if self.chk_warnings.isChecked():
            for ws in warn_steps:
                if ws < len(t_us):
                    self.ax.axvline(x=t_us[ws], color='red', lw=1.0, alpha=0.55, zorder=2)

        if not has_any_curve and not (self.chk_warnings.isChecked() and len(warn_steps) > 0):
            self.ax.text(
                0.5, 0.5, "No quantities selected in legend",
                ha='center', va='center', transform=self.ax.transAxes,
                color='gray', fontsize=10, fontstyle='italic'
            )

        self.ax.set_xlabel("Simulation Time [µs]")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title("PIC Energy Conservation Monitor", fontsize=10)
        self.ax.grid(True, alpha=0.3)

        # --- Interactive Legend with Checkmarks [x] / [ ] ---
        self._legend_pick_map.clear()

        # Build comprehensive legend handles and labels reflecting checked state
        quantities = [
            (self.chk_total, "Total (KE + Field)", '#1f77b4', '-', 2.2),
            (self.chk_ke_i, "KE ions", '#ff7f0e', '--', 1.2),
            (self.chk_ke_e, "KE electrons", '#2ca02c', '--', 1.2),
            (self.chk_fe, "E field", '#9467bd', '--', 1.2),
            (self.chk_warnings, f"Spikes (>{threshold*100:.0f}%)", '#d62728', ':', 1.0)
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
            fontsize=8, loc='upper left', framealpha=0.85
        )

        if leg is not None:
            for leg_line, leg_text, (chk, _, _, _, _) in zip(leg.get_lines(), leg.get_texts(), quantities):
                # Dim inactive text
                if not chk.isChecked():
                    leg_text.set_alpha(0.35)
                # Make pickable
                leg_line.set_picker(True)
                leg_line.set_pickradius(6)
                leg_text.set_picker(True)

                self._legend_pick_map[leg_line] = chk
                self._legend_pick_map[leg_text] = chk

        self.fig.tight_layout()
        self.canvas.draw_idle()


# Alias for intuitive import
TotalEnergyWindow = PhysicalConstraintsWindow
