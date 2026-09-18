"""
Total Energy Conservation Monitor Diagnostic Window.

Tracks the time evolution of the PIC system's energy budget:
    - Ion kinetic energy (KE_ions)
    - Electron kinetic energy (KE_elec)
    - Electrostatic field energy (E_field)
    - Total energy = KE_ions + KE_elec + E_field
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton
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
        self.setGeometry(150, 150, 820, 520)

        layout = QVBoxLayout(self)

        # Controls row
        ctrl = QHBoxLayout()
        self.chk_components = QCheckBox("Show Energy Components")
        self.chk_components.setChecked(True)
        self.chk_components.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_components)

        self.chk_normalize = QCheckBox("Normalize to First Value")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_normalize)

        btn_clear = QPushButton("Clear History")
        btn_clear.clicked.connect(self._clear_history)
        ctrl.addWidget(btn_clear)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Stats label
        self.lbl_stats = QLabel("Total Energy: — J  |  Delta(last step): —%  |  Warnings: 0")
        self.lbl_stats.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #f4f6f9; "
            "padding: 5px; border: 1px solid #d0d7de; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_stats)

        # Matplotlib figure
        self.fig = plt.figure(figsize=(8.0, 4.4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self._warn_count = 0

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
            ylabel = "Energy (normalized to E_0)"
        else:
            plot_tot = tot
            plot_ki = ke_i
            plot_ke = ke_e
            plot_fe = fe
            ylabel = "Energy [J]"

        # Total energy (always shown — thick line)
        self.ax.plot(t_us, plot_tot, color='#1f77b4', lw=2.2,
                     label="Total (KE + E_field)", zorder=4)

        if self.chk_components.isChecked():
            self.ax.plot(t_us, plot_ki, color='#ff7f0e', lw=1.2,
                         linestyle='--', label="KE ions", zorder=3)
            self.ax.plot(t_us, plot_ke, color='#2ca02c', lw=1.2,
                         linestyle='--', label="KE electrons", zorder=3)
            self.ax.plot(t_us, plot_fe, color='#9467bd', lw=1.2,
                         linestyle='--', label="E field", zorder=3)

        # Warning markers — red vertical lines at warning steps
        for ws in warn_steps:
            if ws < len(t_us):
                self.ax.axvline(x=t_us[ws], color='red', lw=1.0, alpha=0.55, zorder=2)
        if len(warn_steps) > 0:
            self.ax.axvline(x=np.nan, color='red', lw=1.0, alpha=0.55,
                            label=f"Energy spike (>{threshold*100:.0f}%)")

        self.ax.set_xlabel("Simulation Time [µs]")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title("PIC Energy Conservation Monitor", fontsize=10)
        self.ax.legend(fontsize=8, loc='upper left')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()


# Alias for intuitive import
TotalEnergyWindow = PhysicalConstraintsWindow
