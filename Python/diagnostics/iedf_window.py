"""
Energy Distribution Functions (IEDF & EEDF) Window for PY-BEMCS.
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox


class IEDFWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Energy Distribution Function (IEDF & EEDF)")
        self.setGeometry(100, 100, 600, 450)

        layout = QVBoxLayout(self)

        self.combo_type = QComboBox()
        self.combo_type.addItems([
            "All Ions", "Primary Ions Only", "CEX Ions Only",
            "All Electrons", "Grid Secondary Electrons (SEE) [x <= 4mm]",
            "Neutralizer Electrons (Neut) [x > 4mm]"
        ])
        layout.addWidget(QLabel("Select Particle Population:"))
        layout.addWidget(self.combo_type)
        self.combo_type.currentIndexChanged.connect(self._replot)

        self.fig = plt.figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)
        self._last_args = None

    def _replot(self):
        if getattr(self, '_last_args', None) is not None:
            self.update_histogram(*self._last_args)

    def update_histogram(self, p_vx, p_vy, p_isCEX, e_x, e_vx, e_vy, m_XE, m_e, q, Vs_max):
        self._last_args = (p_vx, p_vy, p_isCEX, e_x, e_vx, e_vy, m_XE, m_e, q, Vs_max)
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)

        mode = self.combo_type.currentText()
        data = np.array([])
        color = "gray"
        Vs_max_val = float(Vs_max) if (Vs_max is not None and np.isfinite(Vs_max) and Vs_max > 0) else 1000.0
        x_max = Vs_max_val + 100.0
        title = "Energy Distribution"

        p_vx = np.asarray(p_vx) if p_vx is not None else np.empty(0)
        p_vy = np.asarray(p_vy) if p_vy is not None else np.empty(0)
        p_isCEX = np.asarray(p_isCEX, dtype=bool) if p_isCEX is not None else np.empty(0, dtype=bool)
        min_ion = min(len(p_vx), len(p_vy), len(p_isCEX))
        p_vx = p_vx[:min_ion]
        p_vy = p_vy[:min_ion]
        p_isCEX = p_isCEX[:min_ion]

        e_x = np.asarray(e_x) if e_x is not None else np.empty(0)
        e_vx = np.asarray(e_vx) if e_vx is not None else np.empty(0)
        e_vy = np.asarray(e_vy) if e_vy is not None else np.empty(0)
        min_e = min(len(e_x), len(e_vx), len(e_vy))
        e_x = e_x[:min_e]
        e_vx = e_vx[:min_e]
        e_vy = e_vy[:min_e]

        if "Ions" in mode:
            self.ax.set_xlabel("Energy (eV)")
            self.ax.set_ylabel("Ion Count")
            if len(p_vx) > 0:
                v_sq = p_vx**2 + p_vy**2
                E_all = (0.5 * m_XE * v_sq) / q

                if mode == "All Ions":
                    data, color, title = E_all, "purple", "Ion Energy Distribution (All Ions)"
                elif mode == "Primary Ions Only":
                    data, color, title = E_all[~p_isCEX], "blue", "Ion Energy Distribution (Primary Beam)"
                else:
                    data, color, title = E_all[p_isCEX], "red", "Ion Energy Distribution (CEX Only)"

                x_max = max(Vs_max_val * 0.3, np.max(data) + 50) if len(data) > 0 else 500

        else:
            self.ax.set_xlabel("Electron Kinetic Energy (eV)")
            self.ax.set_ylabel("Electron Count")
            if len(e_vx) > 0:
                v_sq_e = e_vx**2 + e_vy**2
                E_elec = (0.5 * m_e * v_sq_e) / q

                if mode == "All Electrons":
                    data, color, title = E_elec, "#2ECC71", "Electron Energy Distribution (All)"
                elif "SEE" in mode:
                    data, color, title = E_elec[e_x <= 4.0], "#E67E22", "Electron Energy Distribution (Grid/SEE Zone)"
                else:
                    data, color, title = E_elec[e_x > 4.0], "#1ABC9C", "Electron Energy Distribution (Plume/Neut Zone)"

                x_max = max(20.0, np.percentile(data, 99) * 1.2) if len(data) > 0 else 20.0

        self.ax.set_title(title)
        if len(data) > 0:
            self.ax.hist(data, bins=50, range=(0, x_max), color=color, alpha=0.8, edgecolor="black")
        self.ax.set_xlim(0, x_max)
        self.canvas.draw_idle()
