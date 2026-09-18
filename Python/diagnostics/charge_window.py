"""
Total Charge Monitor Diagnostic Window.

Tracks the time evolution of the PIC system's total charge budget:
    - Ion space charge (Q_ions > 0)
    - Kinetic electron space charge (Q_elec <= 0, e.g. neutralizer)
    - Fluid Boltzmann electron continuum (Q_boltzmann <= 0, upstream presheath)
    - Net particle charge (Q_free = Q_ions + Q_elec)
    - Total net space charge (Q_net = Q_free + Q_boltzmann)
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
        self.setGeometry(160, 160, 820, 520)

        layout = QVBoxLayout(self)

        # Controls row
        ctrl = QHBoxLayout()
        self.chk_components = QCheckBox("Show Charge Components")
        self.chk_components.setChecked(True)
        self.chk_components.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_components)

        self.chk_boltzmann = QCheckBox("Include Boltzmann Continuum")
        self.chk_boltzmann.setChecked(True)
        self.chk_boltzmann.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_boltzmann)

        self.chk_normalize = QCheckBox("Normalize to Peak")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_normalize)

        btn_clear = QPushButton("Clear History")
        btn_clear.clicked.connect(self._clear_history)
        ctrl.addWidget(btn_clear)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Stats label
        self.lbl_stats = QLabel("Net Charge: -  |  Q_ions: -  |  Q_elec: -  |  Delta(last step): -")
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

        if len(t) == 0:
            self.ax.clear()
            self.ax.set_title("No charge data yet — start simulation")
            self.canvas.draw_idle()
            return

        t_us = t * 1e6   # convert s -> us for display

        include_boltzmann = self.chk_boltzmann.isChecked()
        target_q = q_net if include_boltzmann else q_free

        # Auto-scaling unit prefix for the plot axis:
        max_val = np.max(np.abs(target_q)) if len(target_q) > 0 else 0.0
        if self.chk_components.isChecked():
            comp_max = max(
                np.max(np.abs(q_i)) if len(q_i) > 0 else 0.0,
                np.max(np.abs(q_e)) if len(q_e) > 0 else 0.0
            )
            max_val = max(max_val, comp_max)

        normalize = self.chk_normalize.isChecked()
        if normalize:
            ref = np.max(np.abs(target_q)) if np.max(np.abs(target_q)) > 0 else 1.0
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
        last_net = float(target_q[-1])
        last_qi = float(q_i[-1])
        last_qe = float(q_e[-1])
        last_qb = float(q_b[-1])

        if len(target_q) > 1 and abs(target_q[-2]) > 1e-15:
            delta_rel = (target_q[-1] - target_q[-2]) / abs(target_q[-2]) * 100.0
            delta_str = f"{delta_rel:+.2f}%"
        else:
            delta_str = "-"

        net_name = "Net Charge" if include_boltzmann else "Particle Net"
        stats_text = (
            f"{net_name}: {self._format_charge(last_net)}  |  "
            f"Q_ions: {self._format_charge(last_qi)}  |  "
            f"Q_elec: {self._format_charge(last_qe)}  |  "
        )
        if include_boltzmann:
            stats_text += f"Q_boltz: {self._format_charge(last_qb)}  |  "
        stats_text += f"Delta(last step): {delta_str}"

        self.lbl_stats.setText(stats_text)

        # Plotting
        self.ax.clear()

        # Zero reference line
        self.ax.axhline(0, color='gray', linestyle=':', lw=1.0, alpha=0.7, label="Neutrality (Q=0)")

        # Main curve (thick blue)
        main_label = "Total Net Charge (Ions + Electrons)" if include_boltzmann else "Net Particle Charge (Ions + e-)"
        self.ax.plot(t_us, target_q * scale, color='#1f77b4', lw=2.2, label=main_label, zorder=4)

        if self.chk_components.isChecked():
            self.ax.plot(t_us, q_i * scale, color='#ff7f0e', lw=1.2,
                         linestyle='--', label="Ions (+)", zorder=3)
            self.ax.plot(t_us, q_e * scale, color='#2ca02c', lw=1.2,
                         linestyle='--', label="Kinetic e- (-)", zorder=3)
            if include_boltzmann:
                self.ax.plot(t_us, q_b * scale, color='#9467bd', lw=1.2,
                             linestyle='--', label="Boltzmann e- (-)", zorder=3)
            if include_boltzmann and len(q_e) > 0 and np.any(q_e != 0):
                self.ax.plot(t_us, q_free * scale, color='#17becf', lw=1.0,
                             linestyle=':', label="Free Ptcls (i + e)", zorder=3)

        self.ax.set_xlabel("Simulation Time [µs]")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title("PIC Total Charge Monitor", fontsize=10)
        self.ax.legend(fontsize=8, loc='best')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()


# Alias for convenience
ChargeMonitorWindow = TotalChargeMonitorWindow
