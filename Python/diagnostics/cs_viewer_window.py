"""
Cross-Section Data Viewer and Spline Fitting Window for PY-BEMCS.
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import UnivariateSpline

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt


class CrossSectionViewerWindow(QWidget):
    REACTION_TYPES = ["Charge Exchange (CX)", "Secondary Electron Yield (SEE)", "Custom Reaction"]

    def __init__(self, cs_store, parent=None):
        super().__init__()
        self.cs_store = cs_store
        self.setWindowTitle("Cross-Section Data Manager")
        self.setMinimumSize(950, 600)
        self.resize(950, 600)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left control channel
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(8, 8, 8, 8)

        left.addWidget(QLabel("Reaction Type:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(self.REACTION_TYPES)
        left.addWidget(self.combo_type)

        self.btn_import = QPushButton("Import CSV...")
        self.btn_import.clicked.connect(self._import_csv)
        left.addWidget(self.btn_import)

        left.addWidget(QLabel("Loaded Datasets:"))
        self.combo_datasets = QComboBox()
        self.combo_datasets.currentIndexChanged.connect(self._on_dataset_selected)
        left.addWidget(self.combo_datasets)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._remove_dataset)
        left.addWidget(self.btn_remove)

        left.addWidget(QLabel("Spline Smoothing:"))
        self.spin_smooth = QDoubleSpinBox()
        self.spin_smooth.setRange(0.0, 1e6)
        self.spin_smooth.setValue(0.0)
        self.spin_smooth.setDecimals(2)
        self.spin_smooth.setToolTip('0 = interpolating spline (passes through all points)')
        left.addWidget(self.spin_smooth)

        self.btn_fit = QPushButton("Fit Spline")
        self.btn_fit.clicked.connect(self._fit_spline)
        left.addWidget(self.btn_fit)

        self.lbl_info = QLabel("No data loaded.")
        self.lbl_info.setWordWrap(True)
        left.addWidget(self.lbl_info)

        left.addWidget(QLabel("Data Preview:"))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Energy (eV)", "Cross-Section (m²)"])
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.Stretch)
        left.addWidget(self.table)

        left_widget.setMinimumWidth(280)
        left_widget.setMaximumWidth(350)

        # Right: matplotlib plot
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumWidth(400)

        splitter.addWidget(left_widget)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        self._refresh_combo()

    def _refresh_combo(self):
        self.combo_datasets.blockSignals(True)
        self.combo_datasets.clear()
        for label in self.cs_store:
            self.combo_datasets.addItem(label)
        self.combo_datasets.blockSignals(False)
        if self.combo_datasets.count() > 0:
            self.combo_datasets.setCurrentIndex(self.combo_datasets.count() - 1)
            self._on_dataset_selected(self.combo_datasets.currentIndex())
        else:
            self._plot_current()
            self._update_table()

    def _import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Import Cross-Section CSV", "",
            "CSV Files (*.csv);;Text Files (*.txt *.dat);;All Files (*)"
        )
        if not fname:
            return

        try:
            raw = np.loadtxt(fname, delimiter=None, comments="#", skiprows=0)
            if raw.ndim == 1:
                raise ValueError("File must have at least two columns.")
        except Exception:
            try:
                raw = np.loadtxt(fname, delimiter=",", comments="#", skiprows=1)
            except Exception as e2:
                QMessageBox.critical(self, "Import Error", f"Could not parse CSV:\n{e2}")
                return

        if raw.ndim != 2 or raw.shape[1] < 2:
            QMessageBox.critical(self, "Import Error", "File must have at least 2 columns.")
            return

        energy = raw[:, 0]
        cs = raw[:, 1]

        # Sort by energy
        order = np.argsort(energy)
        energy = energy[order]
        cs = cs[order]

        rtype = self.combo_type.currentText()
        base = rtype.split("(")[-1].replace(")", "").strip()
        count = sum(1 for k in self.cs_store if k.startswith(base))
        label = f"{base}_{count + 1}" if count > 0 else base

        self.cs_store[label] = {
            "energy": energy,
            "cs": cs,
            "spline": None,
            "type": rtype
        }

        self._refresh_combo()
        self._plot_current()
        self.lbl_info.setText(
            f'Loaded "{label}": {len(energy)} points, E range [{energy[0]:.1f}, {energy[-1]:.1f}] eV'
        )

    def _remove_dataset(self):
        label = self.combo_datasets.currentText()
        if label and label in self.cs_store:
            del self.cs_store[label]
        self._refresh_combo()

    def _on_dataset_selected(self, idx):
        self._plot_current()
        self._update_table()

    def _update_table(self):
        label = self.combo_datasets.currentText()
        if not label or label not in self.cs_store:
            self.table.setRowCount(0)
            return

        ds = self.cs_store[label]
        n = min(len(ds["energy"]), 50)
        self.table.setRowCount(n)
        for i in range(n):
            self.table.setItem(i, 0, QTableWidgetItem(f"{ds['energy'][i]:.4e}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{ds['cs'][i]:.4e}"))

    def _fit_spline(self):
        label = self.combo_datasets.currentText()
        if not label or label not in self.cs_store:
            QMessageBox.warning(self, "No Data", "Select a dataset first.")
            return

        ds = self.cs_store[label]
        energy = ds["energy"]
        cs = ds["cs"]

        if len(energy) < 4:
            QMessageBox.warning(self, "Too Few Points", "Need at least 4 data points for spline fitting.")
            return

        try:
            s_val = self.spin_smooth.value()
            log_e = np.log10(np.maximum(energy, 1e-30))
            log_cs = np.log10(np.maximum(cs, 1e-50))
            spline = UnivariateSpline(log_e, log_cs, s=s_val, k=3)
            ds["spline"] = spline
            self.lbl_info.setText(f'Spline fitted for "{label}" (smoothing={s_val}).')
            self._plot_current()
        except Exception as e:
            QMessageBox.critical(self, "Spline Error", f"Failed to fit spline:\n{e}")

    def _plot_current(self):
        self.ax.clear()
        label = self.combo_datasets.currentText()

        if label and label in self.cs_store:
            ds = self.cs_store[label]
            energy = ds["energy"]
            cs = ds["cs"]

            self.ax.loglog(energy, cs, "o", ms=4, label="Data", color="#2980B9")

            if ds["spline"] is not None:
                e_fine = np.logspace(np.log10(max(energy[0], 1e-30)), np.log10(energy[-1]), 500)
                log_cs_fine = ds["spline"](np.log10(e_fine))
                cs_fine = 10.0 ** log_cs_fine
                self.ax.loglog(e_fine, cs_fine, "-", lw=2, label="Spline Fit", color="#E74C3C")

            self.ax.set_title(f"{label} — {ds.get('type', '')}")
            self.ax.legend()
            self.ax.grid(True, which="both", alpha=0.3)
        else:
            self.ax.set_title("No data loaded")

        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Cross-Section (m²)")
        self.fig.tight_layout()
        self.canvas.draw_idle()
