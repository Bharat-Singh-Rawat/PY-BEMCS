"""
Steady State Diagnostics Monitor Window for PY-BEMCS.

Monitors the approach to steady-state conditions by plotting percentage variations
of key physical quantities (Total Energy, Total Space Charge, Electric Field Energy,
Saddle-point Potential, Ion Current, Divergence) over a rolling window (default: 50 iterations).
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
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
    QPushButton, QGroupBox, QGridLayout, QFrame, QComboBox, QSpinBox,
    QDoubleSpinBox, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .metrics import compute_rolling_variation


class SteadyStateWindow(QDialog):
    """
    Pop-up window analyzing whether the simulation has reached stationary (steady-state) conditions.

    Plots the relative or peak-to-peak percentage variation of key observables across
    a configurable window of iterations (default: 50 iterations).
    """

    MODES = [
        ("Relative Change (Q[k] - Q[k-W]) / |Q[k-W]|", "rel_change"),
        ("Absolute Change |Q[k] - Q[k-W]| / |Q[k-W]|", "abs_change"),
        ("Peak-to-Peak Fluctuation (max - min) / |mean|", "p2p_spread"),
    ]

    QUANTITIES = [
        ("energy_tot", "Total Energy (E_tot)", "#1f77b4", True),
        ("charge_net", "Total Charge (Q_net)", "#d62728", True),
        ("field_energy", "Electric Field Energy (E_field)", "#9467bd", True),
        ("potential", "Saddle Potential (V_saddle)", "#2ca02c", True),
        ("ion_current", "Exit Ion Current (I_exit)", "#ff7f0e", True),
        ("divergence", "Beam Divergence (θ_div)", "#17becf", False),
        ("particles", "Ion Particle Count (N_ions)", "#8c564b", False),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Steady State Diagnostics — 50-Iteration Variation")
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.resize(1020, 720)
        self.setMinimumSize(850, 560)

        self._current_sim = None
        self._current_app = None
        self._lines = {}
        self._band_span = None
        self._thresh_pos_line = None
        self._thresh_neg_line = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        # ── 1. Controls Bar ──────────────────────────────────────────
        ctrl_box = QGroupBox("Analysis Parameters")
        ctrl_layout = QHBoxLayout(ctrl_box)
        ctrl_layout.setContentsMargins(8, 6, 8, 6)
        ctrl_layout.setSpacing(12)

        ctrl_layout.addWidget(QLabel("Metric:"))
        self.combo_mode = QComboBox()
        for label, mode_key in self.MODES:
            self.combo_mode.addItem(label, mode_key)
        self.combo_mode.currentIndexChanged.connect(self._replot)
        ctrl_layout.addWidget(self.combo_mode)

        ctrl_layout.addWidget(QLabel("Window (iters):"))
        self.spin_window = QSpinBox()
        self.spin_window.setRange(5, 1000)
        self.spin_window.setValue(50)
        self.spin_window.setSingleStep(10)
        self.spin_window.valueChanged.connect(self._on_window_changed)
        ctrl_layout.addWidget(self.spin_window)

        ctrl_layout.addWidget(QLabel("Tolerance (%):"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.01, 50.0)
        self.spin_thresh.setValue(1.0)
        self.spin_thresh.setSingleStep(0.2)
        self.spin_thresh.setDecimals(2)
        self.spin_thresh.setSuffix(" %")
        self.spin_thresh.valueChanged.connect(self._on_threshold_changed)
        ctrl_layout.addWidget(self.spin_thresh)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._replot)
        ctrl_layout.addWidget(btn_refresh)

        ctrl_layout.addStretch()
        main_layout.addWidget(ctrl_box)

        # ── 2. Stationarity Status Banner ────────────────────────────
        banner_frame = QFrame()
        banner_frame.setFrameShape(QFrame.StyledPanel)
        banner_layout = QHBoxLayout(banner_frame)
        banner_layout.setContentsMargins(10, 6, 10, 6)

        self.lbl_status_icon = QLabel("⚪")
        self.lbl_status_icon.setStyleSheet("font-size: 16px;")
        banner_layout.addWidget(self.lbl_status_icon)

        self.lbl_status_text = QLabel("Awaiting simulation data (needs ≥ window iterations)...")
        self.lbl_status_text.setStyleSheet("font-weight: bold; font-size: 12px; color: #495057;")
        banner_layout.addWidget(self.lbl_status_text, 1)

        self.lbl_summary_stats = QLabel("Iter: —")
        self.lbl_summary_stats.setStyleSheet("font-family: monospace; font-size: 11px; color: #212529;")
        banner_layout.addWidget(self.lbl_summary_stats)

        main_layout.addWidget(banner_frame)

        # ── 3. Observables Toggles Bar ───────────────────────────────
        obs_box = QGroupBox("Monitored Quantities")
        obs_layout = QHBoxLayout(obs_box)
        obs_layout.setContentsMargins(8, 4, 8, 4)
        obs_layout.setSpacing(10)

        self.chk_quantities = {}
        for q_id, q_name, q_color, q_default in self.QUANTITIES:
            chk = QCheckBox(q_name)
            chk.setChecked(q_default)
            chk.setStyleSheet(f"QCheckBox {{ color: {q_color}; font-weight: bold; }}")
            chk.stateChanged.connect(self._replot)
            obs_layout.addWidget(chk)
            self.chk_quantities[q_id] = chk

        obs_layout.addStretch()

        btn_sel_all = QPushButton("All")
        btn_sel_all.setMaximumWidth(40)
        btn_sel_all.clicked.connect(lambda: self._set_all_quantities(True))
        obs_layout.addWidget(btn_sel_all)

        btn_sel_none = QPushButton("None")
        btn_sel_none.setMaximumWidth(48)
        btn_sel_none.clicked.connect(lambda: self._set_all_quantities(False))
        obs_layout.addWidget(btn_sel_none)

        main_layout.addWidget(obs_box)

        # ── 4. Matplotlib Graph ──────────────────────────────────────
        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #ffffff; border: 1px solid #ced4da; border-radius: 4px;")

        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=2.2)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            main_layout.addWidget(self.toolbar)

        main_layout.addWidget(self.canvas, 1)

        # ── 5. Instantaneous Readout Table ───────────────────────────
        self.lbl_table = QLabel()
        self.lbl_table.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; background: #f8f9fa; "
            "padding: 6px 10px; border: 1px solid #dee2e6; border-radius: 4px;"
        )
        self.lbl_table.setText(
            "Values | E_tot: — | Q_net: — | E_field: — | V_saddle: — | I_exit: — | θ_div: —"
        )
        main_layout.addWidget(self.lbl_table)

        self._init_plot()

    def _init_plot(self):
        """Set up initial axes labels, grid, and reference lines."""
        self.ax.clear()
        self.ax.set_title("Steady State Variation Analysis", fontsize=11, fontweight="bold", pad=8)
        self.ax.set_xlabel("Iteration", fontsize=10)
        self.ax.set_ylabel("Variation (%)", fontsize=10)
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.axhline(0.0, color="#6c757d", linestyle="-", linewidth=0.8, alpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _set_all_quantities(self, checked: bool):
        for chk in self.chk_quantities.values():
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)
        self._replot()

    def _on_window_changed(self):
        w = self.spin_window.value()
        self.setWindowTitle(f"Steady State Diagnostics — {w}-Iteration Variation")
        self._replot()

    def _on_threshold_changed(self):
        self._replot()

    def _get_raw_series(self, sim, app=None):
        """Extract historical raw series for each physical observable."""
        series = {}
        if sim is None:
            return series

        # 1. Total Energy
        if hasattr(sim, 'energy_history_tot') and len(sim.energy_history_tot) > 0:
            series["energy_tot"] = np.asarray(sim.energy_history_tot, dtype=np.float64)

        # 2. Total Charge
        if hasattr(sim, 'charge_history_q_net') and len(sim.charge_history_q_net) > 0:
            series["charge_net"] = np.asarray(sim.charge_history_q_net, dtype=np.float64)

        # 3. Electric Field Energy
        if hasattr(sim, 'energy_history_fe') and len(sim.energy_history_fe) > 0:
            series["field_energy"] = np.asarray(sim.energy_history_fe, dtype=np.float64)

        # 4. Saddle-point Potential
        if hasattr(sim, 'potential_history') and len(sim.potential_history) > 0:
            series["potential"] = np.asarray(sim.potential_history, dtype=np.float64)
        elif app is not None and hasattr(app, 'ebs_history') and len(app.ebs_history) > 0:
            series["potential"] = np.asarray(app.ebs_history, dtype=np.float64)

        # 5. Exit Ion Current
        if app is not None and hasattr(app, 'ion_current_exit_history') and len(app.ion_current_exit_history) > 0:
            series["ion_current"] = np.asarray(app.ion_current_exit_history, dtype=np.float64)
        elif app is not None and hasattr(app, 'ion_current_exit_avg_history') and len(app.ion_current_exit_avg_history) > 0:
            series["ion_current"] = np.asarray(app.ion_current_exit_avg_history, dtype=np.float64)

        # 6. Beam Divergence
        if app is not None and hasattr(app, 'div_history') and len(app.div_history) > 0:
            series["divergence"] = np.asarray(app.div_history, dtype=np.float64)

        # 7. Active Particles (Ions)
        if app is not None and hasattr(app, 'transmitted_ions_history') and len(app.transmitted_ions_history) > 0:
            series["particles"] = np.asarray(app.transmitted_ions_history, dtype=np.float64)

        return series

    def update_plot(self, sim, app=None):
        """Update simulation reference and replot with latest history."""
        self._current_sim = sim
        self._current_app = app if app is not None else self.parent_app
        self._replot()

    def _replot(self):
        sim = self._current_sim
        app = self._current_app
        if sim is None:
            return

        window = self.spin_window.value()
        mode_idx = self.combo_mode.currentIndex()
        mode_key = self.combo_mode.itemData(mode_idx) or "rel_change"
        thresh = self.spin_thresh.value()

        raw_data = self._get_raw_series(sim, app)
        if not raw_data:
            return

        self.ax.clear()
        self.ax.grid(True, linestyle="--", alpha=0.45)

        # Threshold band
        if mode_key in ("abs_change", "p2p_spread"):
            self.ax.axhspan(0.0, thresh, color="#28a745", alpha=0.15, label=f"Stationary Zone (≤ {thresh:.2f}%)")
            self.ax.axhline(thresh, color="#28a745", linestyle=":", linewidth=1.5)
        else:
            self.ax.axhspan(-thresh, thresh, color="#28a745", alpha=0.15, label=f"Stationary Zone (±{thresh:.2f}%)")
            self.ax.axhline(thresh, color="#28a745", linestyle=":", linewidth=1.2)
            self.ax.axhline(-thresh, color="#28a745", linestyle=":", linewidth=1.2)
            self.ax.axhline(0.0, color="#6c757d", linestyle="-", linewidth=0.8, alpha=0.6)

        max_iter = 0
        min_iter = 0
        latest_variations = {}
        all_stationary = True
        active_count = 0

        for q_id, q_label, q_color, _ in self.QUANTITIES:
            chk = self.chk_quantities.get(q_id)
            if not chk or not chk.isChecked():
                continue

            series = raw_data.get(q_id)
            if series is None or len(series) <= 1:
                continue

            iters, var = compute_rolling_variation(series, window=window, mode=mode_key)
            if len(var) == 0:
                continue

            active_count += 1
            cur_var = var[-1]
            latest_variations[q_id] = cur_var
            abs_var = abs(cur_var)

            if abs_var > thresh:
                all_stationary = False

            min_iter = min(min_iter, iters[0]) if max_iter > 0 else iters[0]
            max_iter = max(max_iter, iters[-1])

            label_str = f"{q_label} [Δ={cur_var:+.2f}%]"
            self.ax.plot(iters, var, label=label_str, color=q_color, linewidth=1.6, alpha=0.9)

        # Visual layout adjustments
        mode_desc = self.combo_mode.currentText()
        self.ax.set_title(f"Steady-State Variation across {window} Iterations — {mode_desc}", fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Simulation Iteration", fontsize=9)
        self.ax.set_ylabel("Variation (%)", fontsize=9)

        if active_count > 0:
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            if max_iter > min_iter:
                self.ax.set_xlim(min_iter, max_iter + 1)
        else:
            self.ax.text(
                0.5, 0.5, "No quantities selected or awaiting simulation data...",
                ha="center", va="center", transform=self.ax.transAxes, color="#6c757d", fontsize=11
            )

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

        # Update Status Banner & Readout Table
        current_iter = getattr(sim, 'iteration', max_iter)
        if active_count == 0 or max_iter < window:
            self.lbl_status_icon.setText("⏳")
            self.lbl_status_text.setText(f"Gathering history... (iter {current_iter}/{window} needed for full window)")
            self.lbl_status_text.setStyleSheet("font-weight: bold; font-size: 12px; color: #6c757d;")
        elif all_stationary:
            self.lbl_status_icon.setText("🟢")
            self.lbl_status_text.setText(f"STEADY STATE ACHIEVED — All active quantities variation ≤ {thresh:.2f}% across last {window} iterations.")
            self.lbl_status_text.setStyleSheet("font-weight: bold; font-size: 12px; color: #28a745;")
        else:
            self.lbl_status_icon.setText("🟠")
            max_drift_q = max(latest_variations.items(), key=lambda kv: abs(kv[1])) if latest_variations else (None, 0.0)
            self.lbl_status_text.setText(f"TRANSIENT / DRIFTING — Max variation: {max_drift_q[0]} ({max_drift_q[1]:+.2f}%), tolerance is ±{thresh:.2f}%.")
            self.lbl_status_text.setStyleSheet("font-weight: bold; font-size: 12px; color: #fd7e14;")

        self.lbl_summary_stats.setText(f"Iter: {current_iter:,} | Window: {window} | Tol: ±{thresh:.1f}%")

        # Table summary text
        parts = []
        for q_id, q_name, _, _ in self.QUANTITIES:
            if q_id in latest_variations:
                v = latest_variations[q_id]
                stat = "✓" if abs(v) <= thresh else "✗"
                parts.append(f"{q_id}: {v:+.2f}% ({stat})")
        if parts:
            self.lbl_table.setText(" | ".join(parts))
        else:
            self.lbl_table.setText("No active variation metrics available yet.")
