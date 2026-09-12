"""
PY-BEMCS GUI
All default parameters are loaded from config.json.
No simulation defaults are imposed in this file.
"""

import os
import sys
import json
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QPushButton, QCheckBox,
    QMessageBox, QFileDialog, QApplication, QComboBox,
    QScrollArea, QGroupBox, QAction, QDialog, QFormLayout,
    QSpinBox, QTableWidget, QTableWidgetItem, QMenuBar,
    QHeaderView, QSplitter, QProgressDialog
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtCore import Qt as QtCore_Qt
from PyQt5.QtGui import QImage
from scipy.interpolate import UnivariateSpline

from physics_engine import DigitalTwinSimulator, compute_debye_upstream_gap


class PyInstallerWorker(QThread):
    """Runs PyInstaller in a background thread and reports progress via signals."""
    progress_line = pyqtSignal(str)   # each new stdout/stderr line
    progress_pct  = pyqtSignal(int)   # 0-100 estimated percentage
    finished      = pyqtSignal(bool, str)  # (success, message)

    # PyInstaller log lines that signal meaningful milestones (order matters)
    _MILESTONES = [
        ("checking python",         5),
        ("running analysis",        10),
        ("processing module hooks", 20),
        ("looking for implied",     30),
        ("copying dependencies",    40),
        ("building pyz",            50),
        ("building pkg",            60),
        ("building exe",            70),
        ("appending archive",       80),
        ("building exe from",       85),
        ("exe successfully",        95),
    ]

    def __init__(self, script_path: str, dest_dir: str, parent=None):
        super().__init__(parent)
        self._script_path = script_path
        self._dest_dir    = dest_dir

    def run(self):
        import subprocess, shutil

        script_dir  = os.path.dirname(os.path.abspath(self._script_path))
        script_name = os.path.splitext(os.path.basename(self._script_path))[0]

        # Build inside a temporary work directory so we don't pollute the source tree
        build_dir = os.path.join(script_dir, "_exe_build_tmp")
        dist_dir  = os.path.join(build_dir,  "dist")
        work_dir  = os.path.join(build_dir,  "work")

        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--name",    script_name,
            "--distpath", dist_dir,
            "--workpath", work_dir,
            "--specpath", build_dir,
            "--noconfirm",
            self._script_path,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=script_dir,
            )

            self.progress_pct.emit(2)
            current_pct = 2

            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                self.progress_line.emit(line)
                lower = line.lower()
                for keyword, pct in self._MILESTONES:
                    if keyword in lower and pct > current_pct:
                        current_pct = pct
                        self.progress_pct.emit(current_pct)
                        break

            proc.wait()

            if proc.returncode != 0:
                self.finished.emit(False, "PyInstaller exited with errors. See log above.")
                return

            # Locate the produced executable
            exe_name = script_name + (".exe" if sys.platform == "win32" else "")
            src_exe  = os.path.join(dist_dir, exe_name)
            if not os.path.isfile(src_exe):
                self.finished.emit(False, f"Build succeeded but executable not found:\n{src_exe}")
                return

            # Copy to user-selected destination
            dst_exe = os.path.join(self._dest_dir, exe_name)
            shutil.copy2(src_exe, dst_exe)

            # Also copy config.json next to the exe if it exists
            cfg_src = os.path.join(script_dir, "config.json")
            if os.path.isfile(cfg_src):
                shutil.copy2(cfg_src, os.path.join(self._dest_dir, "config.json"))

            self.progress_pct.emit(100)
            self.finished.emit(True, dst_exe)

        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            # Clean up temporary build artefacts
            try:
                import shutil as _sh
                if os.path.isdir(build_dir):
                    _sh.rmtree(build_dir, ignore_errors=True)
            except Exception:
                pass


def _collect_extra_lib_paths():
    """Return library dirs that must be on LD_LIBRARY_PATH before we load
    PyQt5 or CuPy. Qt5 ships inside the PyQt5 wheel; CUDA 12 runtime libs
    ship as separate nvidia-*-cu12 wheels that CuPy 13 doesn't auto-load."""
    paths = []

    try:
        import PyQt5  # noqa: F401
        qt_lib = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "lib")
        if os.path.isdir(qt_lib):
            paths.append(qt_lib)
    except Exception:
        pass

    try:
        import sysconfig
        site_pkgs = sysconfig.get_paths()["purelib"]
        nvidia_root = os.path.join(site_pkgs, "nvidia")
        if os.path.isdir(nvidia_root):
            for sub in sorted(os.listdir(nvidia_root)):
                lib = os.path.join(nvidia_root, sub, "lib")
                if os.path.isdir(lib):
                    paths.append(lib)
    except Exception:
        pass

    return paths


def _ensure_lib_paths():
    if os.environ.get("PYBEMCS_LD_PATCHED") == "1":
        return

    extras = _collect_extra_lib_paths()
    if not extras:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = current.split(":") if current else []
    missing = [p for p in extras if p not in current_parts[:len(extras)]]
    if not missing:
        return

    merged = ":".join(extras + [p for p in current_parts if p and p not in extras])
    os.environ["LD_LIBRARY_PATH"] = merged
    os.environ["PYBEMCS_LD_PATCHED"] = "1"
    # Re-exec so the dynamic linker picks up the prepended paths
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_lib_paths()

if not os.environ.get("QT_QPA_PLATFORM"):
    if sys.platform == "win32":
        os.environ["QT_QPA_PLATFORM"] = "windows"
    else:
        os.environ["QT_QPA_PLATFORM"] = "xcb;wayland;wayland-egl"


def _safe_start_dir():
    try:
        return os.getcwd()
    except Exception:
        return os.path.expanduser("~")


def _config_path():
    """
    Returns the path to config.json located next to the .exe (or next to
    main.py when running from source). Works on any PC, no hardcoding.
    """
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller .exe — use the folder containing the .exe
        base = os.path.dirname(sys.executable)
    else:
        # Running as normal Python — use the folder containing main.py
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "config.json")

def load_json_config(config_path=None):
    """
    Loads and parses a JSON configuration file for the simulation.
    If no path is provided, it defaults to the 'config.json' located next to the
    executable or main.py. Returns a dictionary with configuration parameters if
    the file exists, or None if the file is not found (allowing the caller to fall
    back to default values).
    """
    if config_path is None:
        config_path = _config_path()

    if not os.path.isfile(config_path):
        return None # no config present — caller will use defaults

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cross_sections_from_config(config):
    cs_store = {}
    cs_cfg = config.get("cross_sections", {})
    smoothing = cs_cfg.get("spline_smoothing", 0.0)

    cs_files = {
        "CX": cs_cfg.get("cx_file", "").strip(),
        "SEE": cs_cfg.get("see_file", "").strip(),
        "Custom": cs_cfg.get("custom_file", "").strip(),
    }

    for label, fpath in cs_files.items():
        if not fpath:
            continue
        if not os.path.isfile(fpath):
            print(f"Warning: cross-section file not found for {label}: {fpath}")
            continue

        try:
            try:
                raw = np.loadtxt(fpath, delimiter=None, comments="#")
            except Exception:
                raw = np.loadtxt(fpath, delimiter=",", comments="#", skiprows=1)

            if raw.ndim != 2 or raw.shape[1] < 2:
                print(f"Warning: {fpath} must have at least 2 columns. Skipping.")
                continue

            energy = raw[:, 0]
            cs = raw[:, 1]
            order  = np.argsort(energy)
            energy = energy[order]
            cs = cs[order]

            log_e  = np.log10(np.maximum(energy, 1e-30))
            log_cs = np.log10(np.maximum(cs, 1e-50))
            spline = UnivariateSpline(log_e, log_cs, s=smoothing, k=3)

            cs_store[label] = {
                "energy": energy,
                "cs": cs,
                "spline": spline,
                "type": label
            }
        except Exception as e:
            print(f"Warning: failed to load {label} cross-section from {fpath}: {e}")

    return cs_store

class ScientificSpinBox(QDoubleSpinBox):
    def textFromValue(self, value):
        return f"{value:.3e}"

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return 0.0

    def validate(self, text, pos):
        from PyQt5.QtGui import QValidator
        try:
            float(text.replace("E", "e"))
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            if text in ("", "-", "+", ".", "e", "E", "-e", "+e"):
                return (QValidator.Intermediate, text, pos)
            return (QValidator.Invalid, text, pos)
        
class BeamSpeciesDialog(QDialog):
    PRESETS = [
        ("Custom", 0, 1),
        ("Xenon (Xe)", 131.293, 1),
        ("Krypton (Kr)", 83.798, 1),
        ("Argon (Ar)", 39.948, 1),
        ("Nitrogen (N₂)", 28.014, 1),
        ("Oxygen (O₂)", 31.998, 1),
        ("Hydrogen (H₂)", 2.016, 1),
        ("Helium (He)", 4.0026, 1),
        ("Mercury (Hg)", 200.59, 1),
        ("Cesium (Cs)", 132.905, 1),
    ]

    def __init__(self, current_mass_amu, current_charge, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ion Beam Species")
        self.setMinimumWidth(350)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select a preset or enter custom values:"))

        self.combo_preset = QComboBox()
        for name, _, _ in self.PRESETS:
            self.combo_preset.addItem(name)
        self.combo_preset.currentIndexChanged.connect(self._on_preset)
        layout.addWidget(self.combo_preset)

        form = QFormLayout()

        self.spin_mass = QDoubleSpinBox()
        self.spin_mass.setRange(0.5, 500.0)
        self.spin_mass.setDecimals(3)
        self.spin_mass.setSingleStep(0.1)
        self.spin_mass.setValue(current_mass_amu)
        self.spin_mass.setSuffix(" amu")
        form.addRow("Atomic / Molecular Mass:", self.spin_mass)

        self.spin_charge = QSpinBox()
        self.spin_charge.setRange(1, 10)
        self.spin_charge.setValue(current_charge)
        self.spin_charge.setPrefix("+")
        form.addRow("Charge State:", self.spin_charge)

        layout.addLayout(form)

        btn_box = QHBoxLayout()
        save_btn = QPushButton("Apply")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)
        self._sync_preset_from_values(current_mass_amu)

    def _sync_preset_from_values(self, mass_amu):
        for i, (_, m, _) in enumerate(self.PRESETS):
            if abs(m - mass_amu) < 0.01:
                self.combo_preset.blockSignals(True)
                self.combo_preset.setCurrentIndex(i)
                self.combo_preset.blockSignals(False)
                return
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentIndex(0)
        self.combo_preset.blockSignals(False)

    def _on_preset(self, idx):
        if idx > 0:
            _, mass, charge = self.PRESETS[idx]
            self.spin_mass.setValue(mass)
            self.spin_charge.setValue(charge)

    def get_values(self):
        return self.spin_mass.value(), self.spin_charge.value()

# Grid material properties dialog with presets and custom input fields
class GridMaterialDialog(QDialog):
    PRESETS = {
        "Molybdenum": {
            "k": 138.0, "rho": 10280.0, "cp": 250.0,
            "emissivity": 0.80, "alpha": 4.8e-6, "E_mod": 329e9,
            "Y_coeff": 1.05e-4, "E_th": 30.0
        },
        "Steel (SS316)": {
            "k": 16.3, "rho": 8000.0, "cp": 500.0,
            "emissivity": 0.60, "alpha": 16.0e-6, "E_mod": 193e9,
            "Y_coeff": 2.8e-4, "E_th": 25.0
        },
        "Titanium": {
            "k": 21.9, "rho": 4507.0, "cp": 520.0,
            "emissivity": 0.50, "alpha": 8.6e-6, "E_mod": 116e9,
            "Y_coeff": 1.8e-4, "E_th": 20.0
        },
        "Graphite": {
            "k": 120.0, "rho": 2200.0, "cp": 710.0,
            "emissivity": 0.85, "alpha": 3.0e-6, "E_mod": 11e9,
            "Y_coeff": 3.5e-4, "E_th": 15.0
        },
        "Custom": None,
    }

    FIELD_DEFS = [
        ("k", "Thermal Conductivity (W/m/K):", 0.1, 5000, 138.0, 1),
        ("rho", "Density (kg/m³):", 100, 25000, 10280.0, 0),
        ("cp", "Specific Heat (J/kg/K):", 50, 5000, 250.0, 0),
        ("emissivity", "Emissivity (0-1):", 0.01, 1.0, 0.8, 2),
        ("alpha", "Thermal Expansion (1/K):", 0, 1e-3, 4.8e-6, 7),
        ("E_mod", "Young's Modulus (Pa):", 1e8, 1e12, 329e9, 0),
        ("Y_coeff", "Sputter Yield Coeff:", 0, 1e-2, 1.05e-4, 6),
        ("E_th", "Sputter Threshold (eV):", 0, 500, 30.0, 1),
    ]

    def __init__(self, current_mat_name, current_props, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grid Material Properties")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select a preset or enter custom values:"))

        self.combo = QComboBox()
        self.combo.addItems(list(self.PRESETS.keys()))
        self.combo.currentTextChanged.connect(self._on_preset)
        layout.addWidget(self.combo)

        self.form = QFormLayout()
        self.spins = {}
        for key, label, mn, mx, default, decimals in self.FIELD_DEFS:
            spin = QDoubleSpinBox()
            spin.setRange(mn, mx)
            spin.setDecimals(decimals)
            spin.setValue(current_props.get(key, default))
            spin.setSingleStep(10 ** (-decimals) if decimals > 0 else max(1, mx / 100))
            self.form.addRow(label, spin)
            self.spins[key] = spin
        layout.addLayout(self.form)

        btn_box = QHBoxLayout()
        save_btn = QPushButton("Apply")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

        # Sync combo to current selection
        if current_mat_name in self.PRESETS:
            self.combo.setCurrentText(current_mat_name)
        else:
            self.combo.setCurrentText("Custom")

    def _on_preset(self, name):
        props = self.PRESETS.get(name)
        if props is not None:
            for key, spin in self.spins.items():
                spin.setValue(props[key])

    def get_values(self):
        name = self.combo.currentText()
        props = {k: s.value() for k, s in self.spins.items()}
        return name, props


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
            # Some PyQt/PySide stubs/type-checkers mark this as Optional; guard at runtime
            header.setSectionResizeMode(QHeaderView.Stretch)
        left.addWidget(self.table)

        left_widget.setMinimumWidth(280)
        left_widget.setMaximumWidth(350)

        # —- Right: matplotlib plot —-
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
            # Fit in log-log space for better behavior across decades
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


# —- ADVANCED SETTINGS DIALOG —-
class AdvancedSettingsDialog(QDialog):
    def __init__(self, current_params, parent=None, default_Lx=None, default_Ly=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Simulation Parameters")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.form = QFormLayout()
        self.inputs = {}

        def add_spin(key, label, min_v, max_v, default_v, decimals=1, step=1.0):
            spin = QDoubleSpinBox()
            spin.setRange(min_v, max_v)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setValue(default_v)
            self.form.addRow(label, spin)
            self.inputs[key] = spin

        add_spin("neut_x", "Neutralizer Axial Dist (x, mm):", 0, 100, current_params.get("neut_x", 10.0), decimals=3, step=0.5)
        add_spin("neut_r", "Neutralizer Radius (y, mm):", 0.1, 50, current_params.get("neut_r", 1.5), decimals=2, step=0.1)
        add_spin("V_plasma_offset", "Plasma Potential Offset (V):", 0, 500, current_params.get("V_plasma_offset", 20.0))
        add_spin("m_e_ratio", "Electron Mass Ratio (m_Xe / X):", 1, 100000, current_params.get("m_e_ratio", 1000.0), 0, 100)

        val_Lx = current_params.get("Lx", default_Lx if default_Lx is not None else 20.0)
        val_Ly = current_params.get("Ly", default_Ly if default_Ly is not None else 3.0)
        add_spin("Lx", "Domain Length (Lx, mm):", 0.1, 500, val_Lx, decimals=3, step=0.5)
        add_spin("Ly", "Domain Height (Ly, mm):", 0.1, 100, val_Ly, decimals=3, step=0.1)

        if default_Lx is not None and default_Ly is not None:
            btn_reset_domain = QPushButton(f"Reset Lx/Ly to Grid Defaults ({default_Lx:.3f} × {default_Ly:.3f} mm)")
            btn_reset_domain.clicked.connect(lambda: (
                self.inputs["Lx"].setValue(default_Lx),
                self.inputs["Ly"].setValue(default_Ly)
            ))
            self.form.addRow(btn_reset_domain)

        # —- Entire Bulk Plasma option —-
        self.chk_bulk = QCheckBox("Entire Bulk Plasma")
        self.chk_bulk.setChecked(bool(current_params.get("entire_bulk_plasma", False)))
        self.chk_bulk.setToolTip(
            "When checked: simulates the entire bulk plasma region.\n"
            "  • Upstream gap is set by Debye length (80/40/30 × λ_D).\n"
            "  • No Bohm (0.61) factor applied to the ion current.\n"
            "When unchecked (default): presheath mode.\n"
            "  • Upstream gap = 0.75 × screen radius.\n"
            "  • Ion current includes the Bohm factor (0.61).\n"
            "In both modes the Upstream Gap spinbox can be manually overridden."
        )
        layout.addWidget(self.chk_bulk)

        layout.addLayout(self.form)

        btn_box = QHBoxLayout()
        save_btn = QPushButton("Save & Apply")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

    def get_values(self):
        result = {k: v.value() for k, v in self.inputs.items()}
        result["entire_bulk_plasma"] = self.chk_bulk.isChecked()
        return result


class IEDFWindow(QWidget):
    def __init__(self):
        super().__init__()
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

        self.fig = plt.figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

    def update_histogram(self, p_vx, p_vy, p_isCEX, e_x, e_vx, e_vy, m_XE, m_e, q, Vs_max):
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)

        mode = self.combo_type.currentText()
        data = np.array([])
        color = "gray"
        x_max = Vs_max + 100
        title = "Energy Distribution"

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

                x_max = max(Vs_max * 0.3, np.max(data) + 50) if len(data) > 0 else 500

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


class PPCWindow(QWidget):
    """
    Window displaying the 2D spatial distribution of Particles Per Cell (PPC).
    Supports Time-Averaged Accumulator (Solution 2), Instantaneous PPC,
    and Low PPC Warning (< 3 ptcls/cell) mask across Presheath, Optics, and Plume zones.

    Uses GridSpec (not make_axes_locatable) so that ax.clear() / cax.clear() never
    corrupts matplotlib's _shared_axes siblings graph, which would trigger a RecursionError.
    """
    def __init__(self, parent=None):
        super().__init__()
        self.parent_app = parent
        self.setWindowTitle("Macroparticle Distribution per Cell (PPC)")
        self.setGeometry(120, 120, 780, 540)

        layout = QVBoxLayout(self)

        # Controls row
        ctrl_layout = QHBoxLayout()

        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Time-Averaged PPC (Accumulator)",
            "Instantaneous PPC (Current Step)",
            "Low PPC Warning Mask (< 3 ptcls/cell)"
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_change)
        ctrl_layout.addWidget(QLabel("<b>PPC Mode:</b>"))
        ctrl_layout.addWidget(self.combo_mode)

        self.chk_zones = QCheckBox("Show Physical Zones")
        self.chk_zones.setChecked(True)
        self.chk_zones.stateChanged.connect(self._on_mode_change)
        ctrl_layout.addWidget(self.chk_zones)

        self.btn_reset_accum = QPushButton("Reset Accumulator")
        self.btn_reset_accum.clicked.connect(self._reset_accum)
        ctrl_layout.addWidget(self.btn_reset_accum)

        layout.addLayout(ctrl_layout)

        # Statistics Summary Panel
        self.lbl_stats = QLabel("Active Cells: — | Low PPC (< 3): — (—%) | Mean Active PPC: —")
        self.lbl_stats.setStyleSheet(
            "font-family: monospace; font-size: 11px; background-color: #f4f6f9; "
            "padding: 6px; border: 1px solid #d0d7de; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_stats)

        # —- Matplotlib Figure with pre-allocated GridSpec axes —-
        # Using GridSpec avoids make_axes_locatable, which corrupts _shared_axes
        # on repeated ax.clear() calls and causes RecursionError in matplotlib.
        self.fig = plt.figure(figsize=(7.5, 4.2))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[28, 1], wspace=0.05)
        self.ax  = self.fig.add_subplot(gs[0])
        self.cax = self.fig.add_subplot(gs[1])
        self.cbar = None
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def _reset_accum(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            self.parent_app.sim.reset_ppc_accumulator()
            self.update_plot(self.parent_app.sim)

    def _on_mode_change(self):
        if self.parent_app and hasattr(self.parent_app, 'sim'):
            self.update_plot(self.parent_app.sim)

    def update_plot(self, sim):
        if sim is None or not hasattr(sim, 'nx') or sim.nx <= 0 or sim.ny <= 0:
            return

        mode = self.combo_mode.currentText()
        if "Time-Averaged" in mode:
            data = sim.get_avg_ppc_map()
            title = f"Time-Averaged PPC — {getattr(sim, 'ppc_steps_count', 0)} steps"
        elif "Instantaneous" in mode:
            data = getattr(sim, 'current_ppc_map', np.zeros((sim.ny, sim.nx))).astype(float)
            title = f"Instantaneous PPC — Step {sim.iteration}"
        else:
            cur = getattr(sim, 'current_ppc_map', np.zeros((sim.ny, sim.nx)))
            data = np.where((cur > 0) & (cur < 3), 1.0, 0.0)
            title = f"Low PPC Warning (< 3 particles/cell) — Step {sim.iteration}"

        # Statistics
        active_mask  = data > 0
        total_active = int(np.count_nonzero(active_mask))
        if "Low PPC" in mode:
            low_count = int(np.count_nonzero(data == 1.0))
        else:
            low_count = int(np.count_nonzero((data > 0) & (data < 3)))
        pct_low     = (low_count / total_active * 100.0) if total_active > 0 else 0.0
        mean_active = float(np.mean(data[active_mask])) if total_active > 0 else 0.0

        # Physical zone boundaries
        x_up   = getattr(sim, 'upstream_gap_mm', 0.8)
        x_last = sim.grid_x_ends[-1] if hasattr(sim, 'grid_x_ends') and sim.grid_x_ends else sim.Lx * 0.5
        ix_up   = int(np.clip(round(x_up   / sim.dx), 0, sim.nx))
        ix_last = int(np.clip(round(x_last / sim.dx), 0, sim.nx))

        def _zone_stats(cols):
            a = int(np.count_nonzero(active_mask[:, cols]))
            l = int(np.count_nonzero((data[:, cols] > 0) & (data[:, cols] < 3)))
            p = (l / a * 100.0) if a > 0 else 0.0
            return a, l, p

        act_up,  low_up,  pct_up  = _zone_stats(slice(None, ix_up))
        act_opt, low_opt, pct_opt = _zone_stats(slice(ix_up, ix_last))
        act_plm, low_plm, pct_plm = _zone_stats(slice(ix_last, None))

        self.lbl_stats.setText(
            f"Active Cells: {total_active} | Low PPC (< 3): {low_count} ({pct_low:.1f}%) | Mean PPC: {mean_active:.1f}\n"
            f"Zones Low%:  Presheath {pct_up:.1f}% ({low_up}/{act_up})  |  "
            f"Optics {pct_opt:.1f}% ({low_opt}/{act_opt})  |  "
            f"Plume {pct_plm:.1f}% ({low_plm}/{act_plm})"
        )

        # Clear axes safely — reset axes locator to prevent matplotlib from
        # nesting ColorbarLocator wrappers on every iteration (which triggers RecursionError after ~1000 steps).
        self.ax.clear()
        self.cax.set_axes_locator(None)
        self.cax.clear()
        self.cbar = None

        self.ax.set_title(title, fontsize=10, pad=8)
        self.ax.set_xlabel("Axial Position [mm]")
        self.ax.set_ylabel("Radial Position [mm]")

        extent = [0, sim.Lx, 0, sim.Ly]

        if "Low PPC" in mode:
            cmap = matplotlib.colormaps['Reds'].resampled(2)
            im = self.ax.imshow(data, origin='lower', extent=extent,
                                cmap=cmap, vmin=0, vmax=1, aspect='auto')
        else:
            vmax = max(10.0, float(np.percentile(data[data > 0], 98))) if np.any(data > 0) else 10.0
            im = self.ax.imshow(data, origin='lower', extent=extent,
                                cmap='turbo', vmin=0, vmax=vmax, aspect='auto')

        # Grid boundary overlay
        if hasattr(sim, 'isBound') and np.any(sim.isBound):
            gy, gx = np.where(sim.isBound)
            self.ax.scatter(gx * sim.dx, gy * sim.dy, s=3, c='black', alpha=0.7)

        # Zone delimiter lines and labels
        if self.chk_zones.isChecked():
            self.ax.axvline(x=x_up,   color='white', linestyle='--', linewidth=1.2, alpha=0.8)
            self.ax.axvline(x=x_last, color='cyan',  linestyle='--', linewidth=1.2, alpha=0.8)
            y_pos = sim.Ly * 0.90
            for xc, label in [
                (x_up * 0.5,             "Presheath"),
                ((x_up + x_last) * 0.5,  "Optics"),
                ((x_last + sim.Lx) * 0.5, "Plume"),
            ]:
                self.ax.text(xc, y_pos, label, color='white', fontsize=8, ha='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

        self.ax.set_xlim(0, sim.Lx)
        self.ax.set_ylim(0, sim.Ly)

        # Colorbar drawn into the pre-allocated cax (no shared-axis corruption)
        self.cbar = self.fig.colorbar(im, cax=self.cax)
        if "Low PPC" in mode:
            self.cbar.set_ticks([0.25, 0.75])
            self.cbar.set_ticklabels(["OK (>= 3)", "LOW (< 3)"])
        else:
            self.cbar.set_label("Particles / Cell (PPC)", fontsize=8)

        self.canvas.draw_idle()

        self.canvas.draw_idle()


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
        self.setWindowTitle("Physical Constraints — Energy Conservation Monitor")
        self.setGeometry(150, 150, 820, 520)

        layout = QVBoxLayout(self)

        # --- Controls row ---
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

        # --- Stats label ---
        self.lbl_stats = QLabel("Total Energy: — J  |  Δ(last step): —%  |  Warnings: 0")
        self.lbl_stats.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #f4f6f9; "
            "padding: 5px; border: 1px solid #d0d7de; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_stats)

        # --- Matplotlib figure ---
        self.fig = plt.figure(figsize=(8.0, 4.4))
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self._warn_count = 0

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def update_plot(self, sim):
        """Redraw the energy-vs-time graph from sim.energy_history_*."""
        if sim is None or not hasattr(sim, 'energy_history_t'):
            return

        t  = np.asarray(sim.energy_history_t,    dtype=float)
        ke_i = np.asarray(sim.energy_history_ke_i, dtype=float)
        ke_e = np.asarray(sim.energy_history_ke_e, dtype=float)
        fe   = np.asarray(sim.energy_history_fe,   dtype=float)
        tot  = np.asarray(sim.energy_history_tot,  dtype=float)

        if len(t) == 0:
            self.ax.clear()
            self.ax.set_title("No energy data yet — start simulation")
            self.canvas.draw_idle()
            return

        t_us = t * 1e6   # convert s → µs for display

        normalize = self.chk_normalize.isChecked()
        ref = tot[0] if (normalize and tot[0] != 0) else 1.0

        # --- Identify warning steps (>5% relative change) ---
        warn_steps = []
        threshold = getattr(sim, 'energy_warning_threshold', 0.05)
        if len(tot) > 1:
            diffs = np.abs(np.diff(tot))
            refs  = np.abs(tot[:-1])
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(refs > 0, diffs / refs, 0.0)
            warn_steps = np.where(rel > threshold)[0] + 1  # index of the offending point
        self._warn_count = len(warn_steps)

        # --- Update stats label ---
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

        # --- Plot ---
        self.ax.clear()

        if normalize:
            plot_tot  = tot  / ref
            plot_ki   = ke_i / ref
            plot_ke   = ke_e / ref
            plot_fe   = fe   / ref
            ylabel = "Energy (normalized to E₀)"
        else:
            plot_tot  = tot
            plot_ki   = ke_i
            plot_ke   = ke_e
            plot_fe   = fe
            ylabel = "Energy [J]"

        # Total energy (always shown — thick line)
        self.ax.plot(t_us, plot_tot, color='#1f77b4', lw=2.2,
                     label="Total (KE + E_field)", zorder=4)

        if self.chk_components.isChecked():
            self.ax.plot(t_us, plot_ki, color='#ff7f0e', lw=1.2,
                         linestyle='--', label="KE ions",      zorder=3)
            self.ax.plot(t_us, plot_ke, color='#2ca02c', lw=1.2,
                         linestyle='--', label="KE electrons",  zorder=3)
            self.ax.plot(t_us, plot_fe, color='#9467bd', lw=1.2,
                         linestyle='--', label="E field",       zorder=3)

        # Warning markers — red vertical lines at warning steps
        for ws in warn_steps:
            if ws < len(t_us):
                self.ax.axvline(x=t_us[ws], color='red', lw=1.0, alpha=0.55, zorder=2)
        if len(warn_steps) > 0:
            # Invisible proxy for legend
            self.ax.axvline(x=np.nan, color='red', lw=1.0, alpha=0.55,
                            label=f"Energy spike (>{threshold*100:.0f}%)")

        self.ax.set_xlabel("Simulation Time [µs]")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title("PIC Energy Conservation Monitor", fontsize=10)
        self.ax.legend(fontsize=8, loc='upper left')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()


class DigitalTwinApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PY-BEMCS (Multi-Grid & Co-Extraction)")
        self.setGeometry(20, 30, 1500, 800)

        self.current_config_path = _config_path()
        self.current_config_name = "config.json"
        self.config = load_json_config(self.current_config_path)
        self.sim = DigitalTwinSimulator()
        self.sim_isRunning = False

        self.iter_history = []
        self.ebs_history  = []
        self.div_history  = []
        self.time_history = []   # simulated time [s]
        self.transparency_history  = []
        self.transparency3_history = []
        self.active_cells_history  = []
        self.low_ppc_cells_history = []
        self.T_histories = {}
        self.recorded_frames = []
        self.tracking_buffer = []
        self.iedf_window = None
        self.ppc_window  = None
        self.phys_window = None
        self.cs_viewer_window = None

        self.cbar_temp = None
        self.cbar_energy = None

        self.grid_widgets = []

        self.beam_mass_amu     = None
        self.beam_charge_state = None
        self.cs_store   = {}
        self.mat_name   = None
        self.mat_props  = {}
        self.adv_params = {}

        self.inputs = {}

        self.setup_menu_bar()
        self.setup_ui()
        if self.config is not None:
            self.apply_config(self.config, config_name=self.current_config_name)
        else:
            # No config.json found — load hardcoded defaults directly into the UI
            self._apply_defaults()

        self.timer = QTimer()
        self.timer.timeout.connect(self.run_sim_step)
        self.timer.start(33)

    def setup_menu_bar(self):
        menubar = self.menuBar()
        # On some platforms/menu styles menuBar() may return None; create one if needed
        if menubar is None:
            menubar = QMenuBar(self)
            self.setMenuBar(menubar)
        # make type checkers aware menubar is not None
        assert menubar is not None

        settings_menu = menubar.addMenu("Settings")  # type: ignore[attr-defined]
        adv_action = QAction("Advanced Parameters...", self)
        adv_action.triggered.connect(self.open_advanced_settings)
        settings_menu.addAction(adv_action)

        self.reload_action = QAction(f"Reload {self.current_config_name}", self)
        self.reload_action.triggered.connect(self.reload_config)
        settings_menu.addAction(self.reload_action)

        beam_menu = menubar.addMenu("Beam")
        assert beam_menu is not None # make type checkers aware beam menu is not None
        species_action = QAction("Ion Species...", self)
        species_action.triggered.connect(self.open_beam_species)
        beam_menu.addAction(species_action)

        beam_menu.addSeparator()

        cs_action = QAction("Cross-Section Manager...", self)
        cs_action.triggered.connect(self.open_cs_viewer)
        beam_menu.addAction(cs_action)

        mat_menu = menubar.addMenu("Materials")
        mat_action = QAction("Grid Material...", self)
        mat_action.triggered.connect(self.open_grid_material)
        mat_menu.addAction(mat_action)
        open_action = QAction("Open Config JSON...", self)
        open_action.triggered.connect(self.open_config_json)
        settings_menu.addAction(open_action)

        settings_menu.addSeparator()
        build_exe_action = QAction("Build .exe...", self)
        build_exe_action.triggered.connect(self.build_exe)
        settings_menu.addAction(build_exe_action)

    def open_config_json(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Config JSON",
            _safe_start_dir(),
            "JSON Files (*.json)"
        )
        if not file_name:
            return

        try:
            cfg = load_json_config(file_name)
            self.current_config_path = file_name
            self.current_config_name = os.path.basename(file_name)
            self.apply_config(cfg, config_name=self.current_config_name)
            if hasattr(self, 'reload_action'):
                self.reload_action.setText(f"Reload {self.current_config_name}")
        except Exception as e:
            QMessageBox.critical(self, "Config Error", f"Failed to load config:\n{e}")

    def create_input(self, labeltext, minv, maxv, step, decimals=1, scientific=False):
        row = QHBoxLayout()
        lbl = QLabel(labeltext)
        lbl.setFixedWidth(130)
        spin = ScientificSpinBox() if scientific else QDoubleSpinBox()
        spin.setRange(minv, maxv)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        row.addWidget(lbl)
        row.addWidget(spin)
        return row, spin

    def add_grid_ui(self, v_val, t_val, gap_val, r_val, cham_val):
        idx = len(self.grid_widgets) + 1
        gb = QGroupBox(f"Grid {idx}")
        lay = QVBoxLayout()

        row1, spin_v = self.create_input("DC Voltage (V):", -5000, 15000, 100, 3)
        row2, spin_t = self.create_input("Thickness (mm):", 0.0, 10.0, 0.01, 4)
        row3, spin_gap = self.create_input("Gap to Next (mm):", 0.0, 10.0, 0.01, 4)
        row4, spin_r = self.create_input("Hole Radius (mm):", 0.0, 10.0, 0.01, 4)
        row5, spin_cham = self.create_input("Chamfer (°):", 0.0, 45.0, 0.1, 3)

        spin_v.setValue(v_val)
        spin_t.setValue(t_val)
        spin_gap.setValue(gap_val)
        spin_r.setValue(r_val)
        spin_cham.setValue(cham_val)

        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addLayout(row3)
        lay.addLayout(row4)
        lay.addLayout(row5)
        gb.setLayout(lay)

        self.grids_layout.insertWidget(self.grids_layout.count() - 1, gb)

        self.grid_widgets.append({
            "gb": gb, "V": spin_v, "t": spin_t, "gap": spin_gap,
            "r": spin_r, "cham": spin_cham
        })
        self.update_rf_combo()

    def clear_grid_ui(self):
        while self.grid_widgets:
            gw = self.grid_widgets.pop()
            gw["gb"].deleteLater()
        self.update_rf_combo()

    def remove_grid_ui(self):
        if len(self.grid_widgets) > 1:
            gw = self.grid_widgets.pop()
            gw["gb"].deleteLater()
            self.update_rf_combo()

    def update_rf_combo(self):
        if hasattr(self, "combo_rf_grid"):
            current = self.combo_rf_grid.currentIndex()
            self.combo_rf_grid.clear()
            self.combo_rf_grid.addItems([f"Grid {i + 1}" for i in range(len(self.grid_widgets))])
            if self.combo_rf_grid.count() > 0:
                self.combo_rf_grid.setCurrentIndex(max(0, min(current, self.combo_rf_grid.count() - 1)))

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # SCROLLABLE CONTROL PANEL
        scroll_area = QScrollArea()
        scroll_area.setFixedWidth(330)
        scroll_area.setWidgetResizable(True)

        control_panel  = QWidget()
        control_layout = QVBoxLayout(control_panel)

        self.combo_rf_grid = QComboBox()

        control_layout.addWidget(QLabel("1. MULTI-GRID OPTICS"))
        self.grids_layout = QVBoxLayout()

        # Upstream gap: distance from injection wall to left face of screen grid (mm)
        row, self.inputs["upstream_gap_mm"] = self.create_input(
            "Upstream Gap (mm):", 0.0, 20.0, 0.1, 2
        )
        control_layout.addLayout(row)
        self.inputs["upstream_gap_mm"].setToolTip(
            "Distance from the injection wall to the left face of the screen grid [mm].\n"
            "\n"
            "Presheath mode (default): auto = 0.75 × screen radius.\n"
            "Entire Bulk Plasma mode: auto = Debye-based (80/40/30 × λ_D).\n"
            "\n"
            "Set to 0 to always use the automatic value for the active mode.\n"
            "Any non-zero value manually entered here overrides the auto-computed gap.\n"
            "After 'Build Domain' the value actually used is displayed here."
        )
        
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("+ Add Grid")
        btn_rem = QPushButton("- Remove Grid")
        btn_add.clicked.connect(lambda: self.add_grid_ui(0.0, 0.0, 0.0, 0.0, 0.0))
        btn_rem.clicked.connect(self.remove_grid_ui)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_rem)

        self.grids_layout.addLayout(btn_layout)
        control_layout.addLayout(self.grids_layout)

        control_layout.addSpacing(15)

        control_layout.addWidget(QLabel("2. RF CO-EXTRACTION"))
        self.chk_rf = QCheckBox("Enable RF Modulated Potential")
        control_layout.addWidget(self.chk_rf)

        rf_row = QHBoxLayout()
        rf_row.addWidget(QLabel("Apply RF to:"))
        rf_row.addWidget(self.combo_rf_grid)
        control_layout.addLayout(rf_row)

        row_freq, self.spin_rf_freq = self.create_input("Frequency (MHz):", 0.0, 100.0, 0.1, 4)
        row_amp, self.spin_rf_amp   = self.create_input("Amplitude (V):", 0.0, 5000.0, 1.0, 4)
        control_layout.addLayout(row_freq)
        control_layout.addLayout(row_amp)

        control_layout.addSpacing(15)

        control_layout.addWidget(QLabel("3. PLASMA & SPUTTERING"))
        row, self.inputs["n0_plasma"] = self.create_input("Plasma Dens (m-3):", 0.0, 1e25, 1e16, 4, scientific=True)
        control_layout.addLayout(row)
        row, self.inputs["Te_up"] = self.create_input("Upstream Te (eV):", 0.0, 1000.0, 0.1, 4)
        control_layout.addLayout(row)
        # Auto-refresh Upstream Gap whenever n0 or Te changes
        self.inputs["n0_plasma"].valueChanged.connect(self._update_debye_gap)
        self.inputs["Te_up"].valueChanged.connect(self._update_debye_gap)

        row, self.inputs["Ti"] = self.create_input("Ion Temp (eV):", 0.0, 1000.0, 0.1, 4)
        control_layout.addLayout(row)
        row, self.inputs["Tn"] = self.create_input("Neutral Temp (K):", 0.0, 10000.0, 1.0, 4)
        control_layout.addLayout(row)
        row, self.inputs["n0"] = self.create_input("Neutral Dens (m-3):", 0.0, 1e25, 1e18, 4, scientific=True)
        control_layout.addLayout(row)
        row, self.inputs["Accel"] = self.create_input("Accel. Factor (X):", 0.0, 1e20, 0.1, 6)
        control_layout.addLayout(row)
        row, self.inputs["Thresh"] = self.create_input("Cell Fail Thresh:", 0.0, 1e12, 1.0, 4)
        control_layout.addLayout(row)

        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("4. SIMULATION MODE"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Both", "Thermal", "Erosion", "Trajectories"])
        control_layout.addWidget(self.combo_mode)

        # Geometry mode selector
        control_layout.addWidget(QLabel("Geometry Mode:"))
        self.combo_geometry = QComboBox()
        self.combo_geometry.addItems(["half_hole", "one_hole", "two_holes"])
        self.combo_geometry.setToolTip(
            "half_hole  — half-pitch symmetry, hole at y=0 (fastest, default)\n"
            "one_hole   — full single-aperture domain, hole centred at y=1.5*r\n"
            "two_holes  — full dual-aperture domain, two holes one pitch apart"
        )
        control_layout.addWidget(self.combo_geometry)

        # Injection time (us) — 0 means "inject indefinitely" (current behaviour)
        row, self.inputs["inj_time_µs"] = self.create_input(
            "Injection Time (us):", 0.0, 1000.0, 0.01, 4
        )
        control_layout.addLayout(row)
        self.inputs["inj_time_µs"].setValue(0.0)  # default: unlimited injection

        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("5. NEUTRALIZER"))
        row, self.inputs["neut_rate"] = self.create_input("e- Inject Rate:", 0.0, 1e9, 1.0, 4)
        control_layout.addLayout(row)
        row, self.inputs["Te"] = self.create_input("e- Temp (eV):", 0.0, 1000.0, 0.1, 4)
        control_layout.addLayout(row)

        control_layout.addSpacing(15)

        self.btn_build = QPushButton("1. BUILD DOMAIN")
        self.btn_build.clicked.connect(self.build_domain)
        self.btn_toggle = QPushButton("2. START BEAM")
        self.btn_toggle.clicked.connect(self.toggle_sim)

        self.btn_csv = QPushButton("Export Data (.csv)")
        self.btn_csv.clicked.connect(self.export_csv)

        self.chk_track_ptcls = QCheckBox("Record Kinematics")
        self.btn_export_trk  = QPushButton("Export Particle Data (.csv)")
        self.btn_export_trk.clicked.connect(self.exporttrackingdata)

        self.btn_iedf = QPushButton("Show Energy Dist. (IEDF/EEDF)")
        self.btn_iedf.clicked.connect(self.open_iedf_window)

        self.btn_ppc = QPushButton("Show PPC Distribution")
        self.btn_ppc.clicked.connect(self.open_ppc_window)

        self.btn_phys = QPushButton("Physical Constraints")
        self.btn_phys.clicked.connect(self.open_phys_window)
        self.btn_phys.setToolTip(
            "Opens the Energy Conservation Monitor.\n"
            "Shows KE_ions + KE_electrons + E_field vs simulation time.\n"
            "Energy spikes (>5% per step) are flagged in red here AND in the terminal "
            "even when this window is closed."
        )
        self.chk_record = QCheckBox("Record Frames (0)")
        self.btn_save   = QPushButton("Save GIF Animation")
        self.btn_save.clicked.connect(self.save_gif)

        self.lbl_status   = QLabel("Status: Ready.")
        self.lbl_temp     = QLabel("Grid Temps: Ready")
        self.lbl_material = QLabel(f"Grid Material: {self.mat_name or 'Molybdenum'}")

        for w in [
            self.btn_build, self.btn_toggle, self.btn_csv, self.chk_track_ptcls,
            self.btn_export_trk, self.btn_iedf, self.btn_ppc, self.btn_phys,
            self.chk_record, self.btn_save,
            self.lbl_status, self.lbl_temp, self.lbl_material
        ]:
            control_layout.addWidget(w)

        # —- Performance Panel —-
        perfbox    = QGroupBox("⚡ Beam Diagnostics ")
        perflayout = QVBoxLayout()
        perflayout.setSpacing(2)
        perflayout.setContentsMargins(6, 4, 6, 4)

        self.lblTime         = QLabel("t_sim:        — us")
        self.lblTransparency = QLabel("Transparency: —")

        for lbl in [self.lblTime, self.lblTransparency]:
            lbl.setStyleSheet("font-family: monospace; font-size: 11px;")
            perflayout.addWidget(lbl)

        perfbox.setLayout(perflayout)
        control_layout.addWidget(perfbox)

        scroll_area.setWidget(control_panel)
        main_layout.addWidget(scroll_area)

        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        grid = plt.GridSpec(3, 3, height_ratios=[1.2, 1, 0.9])

        # —- Row 0: wide beam plot + narrow temperature map —-
        self.ax_live = self.fig.add_subplot(grid[0, 0:2])
        self.ax_live.set_title("Ion Beam Extraction & Particle Tracking", fontsize=10)
        self.ax_live.set_xlabel("Axial Position [mm]")
        self.ax_live.set_ylabel("Radial Position [mm]")

        # Narrow column: use short labels to avoid colorbar/neighbour overlap
        self.ax_temp = self.fig.add_subplot(grid[0, 2])
        self.ax_temp.set_title("Grid Temperature", fontsize=10)
        self.ax_temp.set_xlabel("Axial Position [mm]", fontsize=8)
        self.ax_temp.set_ylabel("Radial Position [mm]", fontsize=8)   # units only — avoids crowding the colorbar

        # —- Row 1: damage map + diagnostics —-
        self.ax_dmg = self.fig.add_subplot(grid[1, 0])
        self.ax_dmg.set_title("Sputter Damage Map", fontsize=10)
        self.ax_dmg.set_xlabel("Axial Position [mm]", fontsize=8)
        self.ax_dmg.set_ylabel("Radial Position [mm]", fontsize=8)

        self.ax_ebs = self.fig.add_subplot(grid[1, 1])
        self.ax_ebs.set_title("Saddle-Point Potential", fontsize=10)
        self.ax_ebs.set_xlabel("Iteration", fontsize=8)
        self.ax_ebs.set_ylabel("[V]", fontsize=8)

        self.ax_div = self.fig.add_subplot(grid[1, 2])
        self.ax_div.set_title("Beam Divergence", fontsize=10)
        self.ax_div.set_xlabel("Iteration", fontsize=8)
        self.ax_div.set_ylabel("[°]", fontsize=8)

        # —- Row 2: full-width erosion profile —-
        self.ax_groove = self.fig.add_subplot(grid[2, :])
        self.ax_groove.set_title("Accel Grid Erosion Profile", fontsize=10)
        self.ax_groove.set_xlabel("Radial Position [mm]")
        self.ax_groove.set_ylabel("Erosion Depth [um]")
        self.ax_groove.grid(True, alpha=0.3)
        self.ax_groove.invert_yaxis()

        self.line_ebs, = self.ax_ebs.plot([], [], "m-", lw=2)
        self.line_div, = self.ax_div.plot([], [], "b-", lw=2)
        self.line_groove, = self.ax_groove.plot([], [], "r-", lw=1.5)

        self.scat_prim = None
        self.scat_cex = None
        self.scat_elec = self.ax_live.scatter([], [], s=1, c="#00FF00", alpha=0.5)

        # h_pad / w_pad give breathing room between rows and columns
        self.fig.tight_layout(rect=(0, 0, 0.97, 1), h_pad=1.5, w_pad=0.8)

    def _apply_defaults(self):
        """Populate the UI with safe hardcoded defaults when no config.json is present."""
        # Beam species
        self.beam_mass_amu  = 131.293   # Xenon
        self.beam_charge_state = 1

        # Material
        self.mat_name  = "Molybdenum"
        self.mat_props = GridMaterialDialog.PRESETS["Molybdenum"].copy()
        if hasattr(self, 'lbl_material'):
            self.lbl_material.setText(f"Grid Material: {self.mat_name}")

        # Advanced params
        self.adv_params = {
            "neut_x":           10.0,
            "neut_r":            2.0,
            "V_plasma_offset":  20.0,
            "m_e_ratio":      1000.0,
            "entire_bulk_plasma": False,  # default: presheath mode
        }

        # Plasma / sputtering inputs
        self.inputs["n0_plasma"].setValue(1e17)
        self.inputs["Te_up"].setValue(3.0)
        self.inputs["Ti"].setValue(0.5)
        self.inputs["Tn"].setValue(300.0)
        self.inputs["n0"].setValue(1e18)
        self.inputs["Accel"].setValue(1.0)
        self.inputs["Thresh"].setValue(1e6)
        self.inputs["inj_time_µs"].setValue(0.0)
        # Default (presheath) gap = 0.75 x default screen radius (0.80 mm)
        _default_screen_r = 0.80
        _auto_gap = round(0.75 * _default_screen_r, 3)
        self._last_auto_gap = _auto_gap
        self.inputs["upstream_gap_mm"].setValue(_auto_gap)

        # Neutralizer
        self.inputs["neut_rate"].setValue(30.0)
        self.inputs["Te"].setValue(2.0)

        # RF off
        self.chk_rf.setChecked(False)
        self.spin_rf_freq.setValue(13.56)
        self.spin_rf_amp.setValue(0.0)

        # Geometry default
        self.combo_geometry.setCurrentText("half_hole")

        # Add 3 default grids: Screen / Accel / Deccel (typical 3-grid ion optics)
        self.clear_grid_ui()
        self.add_grid_ui(1100.0, 0.38, 0.64, 0.80, 0.0)   # Screen grid
        self.add_grid_ui(-200.0, 0.38, 0.64, 0.70, 0.0)   # Accel grid
        self.add_grid_ui(  0.0,  0.38, 2.00, 0.75, 0.0)   # Deccel grid

        auto_Lx, auto_Ly = self.compute_grid_domain_size()
        self.adv_params["Lx"] = auto_Lx
        self.adv_params["Ly"] = auto_Ly
        self._user_overrode_Lx = False
        self._user_overrode_Ly = False

        self.lbl_status.setText("Status: No config.json found — using default values.")

    def compute_grid_domain_size(self):
        """Compute default Lx and Ly directly from grid geometry and simulation settings.

        Lx = upstream_gap + sum(t + gap for each grid)
        Ly depends on geometry mode:
          - 'two_holes': screen_r + 2*screen_r + pitch
          - 'one_hole':  3 * screen_r
          - 'half_hole': 0.5 * screen_r + 0.30 * pitch
        """
        if hasattr(self, 'grid_widgets') and self.grid_widgets:
            grids = [{
                "t": gw["t"].value(),
                "gap": gw["gap"].value(),
                "r": gw["r"].value()
            } for gw in self.grid_widgets]
        elif hasattr(self, 'config') and self.config and "grids" in self.config:
            grids = self.config["grids"]
        else:
            grids = []

        if grids:
            total_grid_thickness = sum(float(g["t"]) + float(g["gap"]) for g in grids)
            screen_r = float(grids[0]["r"])
        else:
            total_grid_thickness = 0.0
            screen_r = 0.80

        # Upstream gap
        upstream_gap = 0.0
        if hasattr(self, 'inputs') and "upstream_gap_mm" in self.inputs:
            upstream_gap = self.inputs["upstream_gap_mm"].value()
        if upstream_gap <= 0.0:
            bulk = self.adv_params.get("entire_bulk_plasma", False) if hasattr(self, 'adv_params') else False
            if bulk and hasattr(self, 'inputs') and "n0_plasma" in self.inputs and "Te_up" in self.inputs:
                upstream_gap = compute_debye_upstream_gap(
                    self.inputs["n0_plasma"].value(),
                    self.inputs["Te_up"].value()
                )
            else:
                upstream_gap = 0.75 * screen_r

        auto_Lx = round(upstream_gap + total_grid_thickness, 3)

        # Geometry for Ly
        geom = "half_hole"
        if hasattr(self, 'combo_geometry'):
            geom = self.combo_geometry.currentText()
        elif hasattr(self, 'config') and self.config and "simulation" in self.config:
            geom = self.config["simulation"].get("geometry", "half_hole")

        pitch = getattr(self, "pitch_mm", 3.0)

        if geom == "two_holes":
            auto_Ly = screen_r + 2.0 * screen_r + pitch
        elif geom == "one_hole":
            auto_Ly = 3.0 * screen_r
        else:  # half_hole
            auto_Ly = 0.5 * screen_r + 0.30 * pitch

        auto_Ly = round(auto_Ly, 3)
        return auto_Lx, auto_Ly

    def _update_debye_gap(self):
        """Auto-update the Upstream Gap spinbox when n0 or Te_up change.

        Only updates if the current spinbox value matches the last auto-computed
        value — i.e. the user has not manually overridden it.

        In presheath mode (default) the gap depends on the screen radius, not
        on n0/Te_up, so the spinbox is not changed by this handler.
        In Entire Bulk Plasma mode the Debye-length-based gap is recomputed.
        """
        if not hasattr(self, '_last_auto_gap'):
            return
        current = round(self.inputs["upstream_gap_mm"].value(), 3)
        if abs(current - self._last_auto_gap) > 0.001:
            # User has manually set a different value — do not overwrite
            return
        bulk = self.adv_params.get("entire_bulk_plasma", False) if hasattr(self, 'adv_params') else False
        if bulk:
            # Bulk plasma: recompute from Debye length
            n0    = self.inputs["n0_plasma"].value()
            Te_up = self.inputs["Te_up"].value()
            if n0 <= 0 or Te_up <= 0:
                return
            new_gap = round(compute_debye_upstream_gap(n0, Te_up), 3)
        else:
            # Presheath: gap = 0.75 × screen radius — read from first grid widget
            if self.grid_widgets:
                screen_r = self.grid_widgets[0]["r"].value()
            else:
                screen_r = 0.80  # safe fallback
            new_gap = round(0.75 * screen_r, 3)
        self._last_auto_gap = new_gap
        self.inputs["upstream_gap_mm"].blockSignals(True)
        self.inputs["upstream_gap_mm"].setValue(new_gap)
        self.inputs["upstream_gap_mm"].blockSignals(False)

    def apply_config(self, config, config_name=None):
        if config_name:
            self.current_config_name = config_name
        cfg_name = getattr(self, "current_config_name", "config.json")
        if hasattr(self, 'reload_action'):
            self.reload_action.setText(f"Reload {cfg_name}")

        self.config = config
        discharge_chamber = config.get("discharge_chamber", {})
        self.pitch_mm = discharge_chamber["pitch_mm"]

        beam = config.get("beam_species", {})
        self.beam_mass_amu = beam["mass_amu"]
        self.beam_charge_state = beam["charge_state"]

        mat_cfg = config.get("grid_material", {})
        preset = mat_cfg.get("preset", "Custom")
        if preset in GridMaterialDialog.PRESETS and GridMaterialDialog.PRESETS[preset] is not None:
            self.mat_name = preset
            self.mat_props = GridMaterialDialog.PRESETS[preset].copy()
        else:
            self.mat_name = "Custom"
            self.mat_props = {
                "k": mat_cfg["k"],
                "rho": mat_cfg["rho"],
                "cp": mat_cfg["cp"],
                "emissivity": mat_cfg["emissivity"],
                "alpha": mat_cfg["alpha"],
                "E_mod": mat_cfg["E_mod"],
                "Y_coeff": mat_cfg["Y_coeff"],
                "E_th": mat_cfg["E_th"],
            }

        adv = config.get("advanced_settings", {})
        self.adv_params = {
            "neut_x": adv.get("neut_x", 10.0),
            "neut_r": adv.get("neut_r", 2.0),
            "V_plasma_offset": adv.get("V_plasma_offset", 20.0),
            "m_e_ratio": adv.get("m_e_ratio", 1000.0),
            "entire_bulk_plasma": bool(adv.get("entire_bulk_plasma", False)),
        }

        sim = config.get("simulation", {})
        # Block signals while loading to avoid premature _update_debye_gap firings
        self.inputs["n0_plasma"].blockSignals(True)
        self.inputs["Te_up"].blockSignals(True)
        self.inputs["n0_plasma"].setValue(sim["n0_plasma"])
        self.inputs["Te_up"].setValue(sim["Te_up"])
        self.inputs["n0_plasma"].blockSignals(False)
        self.inputs["Te_up"].blockSignals(False)

        self.inputs["Ti"].setValue(sim["Ti"])
        self.inputs["Tn"].setValue(sim["Tn"])
        self.inputs["n0"].setValue(sim["n0"])
        self.inputs["Accel"].setValue(sim["Accel"])
        self.inputs["Thresh"].setValue(sim["Thresh"])

        mode = sim["sim_mode"]
        idx = self.combo_mode.findText(mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)

        geom = sim.get("geometry", "half_hole")
        idx_geom = self.combo_geometry.findText(geom)
        if idx_geom >= 0:
            self.combo_geometry.setCurrentIndex(idx_geom)

        # Compute the auto upstream gap based on the active mode.
        # Do NOT read upstream_gap_mm from the config — it is a derived quantity.
        grids_cfg = config.get("grids", [])
        if self.adv_params["entire_bulk_plasma"]:
            # Bulk plasma: Debye-length-based gap
            _gap_val = round(compute_debye_upstream_gap(sim["n0_plasma"], sim["Te_up"]), 3)
        else:
            # Presheath (default): 0.75 × screen radius
            _screen_r = grids_cfg[0]["r"] if grids_cfg else 0.80
            _gap_val = round(0.75 * _screen_r, 3)
        self._last_auto_gap = _gap_val
        self.inputs["upstream_gap_mm"].setValue(_gap_val)


        rf = config.get("rf_co_extraction", {})
        self.chk_rf.setChecked(rf["rf_enable"])
        self.spin_rf_freq.setValue(rf["rf_freq"])
        self.spin_rf_amp.setValue(rf["rf_amp"])

        neut = config.get("neutralizer", {})
        self.inputs["neut_rate"].setValue(neut["neut_rate"])
        self.inputs["Te"].setValue(neut["Te"])

        self.clear_grid_ui()
        grids = config.get("grids", [])
        if len(grids) == 0:
            self._apply_defaults()
            return

        for g in grids:
            self.add_grid_ui(g["V"], g["t"], g["gap"], g["r"], g["cham"])

        rf_idx = rf["rf_grid_idx"]
        if 0 <= rf_idx < self.combo_rf_grid.count():
            self.combo_rf_grid.setCurrentIndex(rf_idx)

        self.cs_store = load_cross_sections_from_config(config)
        self.pitch_mm = config.get("discharge_chamber", {}).get("pitch_mm", 3.0)

        auto_Lx, auto_Ly = self.compute_grid_domain_size()
        if "Lx" in adv:
            self.adv_params["Lx"] = adv["Lx"]
            self._user_overrode_Lx = True
        else:
            self.adv_params["Lx"] = auto_Lx
            self._user_overrode_Lx = False

        if "Ly" in adv:
            self.adv_params["Ly"] = adv["Ly"]
            self._user_overrode_Ly = True
        else:
            self.adv_params["Ly"] = auto_Ly
            self._user_overrode_Ly = False

        cfg_name = getattr(self, "current_config_name", "config.json")
        self.lbl_status.setText(f"Loaded {cfg_name} | {len(grids)} grids")
        self.lbl_temp.setText("Grid Temps: " + " | ".join([f"G{i+1}: ready" for i in range(len(grids))]))
        if hasattr(self, 'lbl_material'):
            self.lbl_material.setText(f"Grid Material: {self.mat_name}")

    def reload_config(self):
        cfg_path = getattr(self, "current_config_path", None) or _config_path()
        cfg_name = getattr(self, "current_config_name", "config.json")
        cfg = load_json_config(cfg_path)
        if cfg is None:
            QMessageBox.warning(self, "No Config Found",
                                f"{cfg_name} not found. Using current defaults.")
            return
        try:
            self.apply_config(cfg, config_name=cfg_name)
            QMessageBox.information(self, "Config Reloaded",
                                    f"{cfg_name} was reloaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Config Error", f"Failed to reload config:\n{e}")
    def open_advanced_settings(self):
        auto_Lx, auto_Ly = self.compute_grid_domain_size()
        if not getattr(self, '_user_overrode_Lx', False):
            self.adv_params["Lx"] = auto_Lx
        if not getattr(self, '_user_overrode_Ly', False):
            self.adv_params["Ly"] = auto_Ly

        dialog = AdvancedSettingsDialog(self.adv_params, self, default_Lx=auto_Lx, default_Ly=auto_Ly)
        if dialog.exec_() == QDialog.Accepted:
            new_vals = dialog.get_values()
            if abs(new_vals.get("Lx", auto_Lx) - auto_Lx) > 1e-4:
                self._user_overrode_Lx = True
            else:
                self._user_overrode_Lx = False

            if abs(new_vals.get("Ly", auto_Ly) - auto_Ly) > 1e-4:
                self._user_overrode_Ly = True
            else:
                self._user_overrode_Ly = False

            self.adv_params.update(new_vals)
            # Refresh the upstream-gap spinbox auto-value to match the new mode,
            # but only if the user has not manually overridden it.
            if hasattr(self, '_last_auto_gap'):
                current = round(self.inputs["upstream_gap_mm"].value(), 3)
                if abs(current - self._last_auto_gap) <= 0.001:
                    if self.adv_params.get("entire_bulk_plasma", False):
                        n0    = self.inputs["n0_plasma"].value()
                        Te_up = self.inputs["Te_up"].value()
                        if n0 > 0 and Te_up > 0:
                            new_gap = round(compute_debye_upstream_gap(n0, Te_up), 3)
                        else:
                            new_gap = self._last_auto_gap
                    else:
                        screen_r = self.grid_widgets[0]["r"].value() if self.grid_widgets else 0.80
                        new_gap = round(0.75 * screen_r, 3)
                    self._last_auto_gap = new_gap
                    self.inputs["upstream_gap_mm"].blockSignals(True)
                    self.inputs["upstream_gap_mm"].setValue(new_gap)
                    self.inputs["upstream_gap_mm"].blockSignals(False)
            QMessageBox.information(
                self, "Settings Updated",
                "Advanced settings updated in memory. Click '1. BUILD DOMAIN' to apply."
            )

    def open_beam_species(self):
        dialog = BeamSpeciesDialog(self.beam_mass_amu, self.beam_charge_state, self)
        if dialog.exec_() == QDialog.Accepted:
            self.beam_mass_amu, self.beam_charge_state = dialog.get_values()
            QMessageBox.information(
                self, "Species Updated",
                f"Beam ion: {self.beam_mass_amu:.3f} amu, charge +{self.beam_charge_state}.\n"
                f"Click '1. BUILD DOMAIN' to apply."
            )

    def open_cs_viewer(self):
        if self.cs_viewer_window is None:
            self.cs_viewer_window = CrossSectionViewerWindow(self.cs_store)
        self.cs_viewer_window.show()
        self.cs_viewer_window.raise_()
        self.cs_viewer_window.activateWindow()

    def open_grid_material(self):
        dialog = GridMaterialDialog(self.mat_name, self.mat_props, self)
        if dialog.exec_() == QDialog.Accepted:
            self.mat_name, self.mat_props = dialog.get_values()
            if hasattr(self, 'lbl_material'):
                self.lbl_material.setText(f"Grid Material: {self.mat_name}")
            QMessageBox.information(
                self, "Material Updated",
                f"Grid material: {self.mat_name}\n"
                f"Click '1. BUILD DOMAIN' to apply."
            )

    def apply_advanced_settings_to_sim(self):
        self.sim.Lx = self.adv_params["Lx"]
        self.sim.Ly = self.adv_params["Ly"]
        self.sim.nx = int(self.sim.Lx / self.sim.dx) + 1
        self.sim.ny = int(self.sim.Ly / self.sim.dy) + 1

        self.sim.m_ion = self.beam_mass_amu * 1.6605e-27
        self.sim.m_XE = self.sim.m_ion
        self.sim.Z_ion = self.beam_charge_state
        self.sim.q_ion = self.beam_charge_state * self.sim.q
        self.sim.m_e = self.sim.m_ion / self.adv_params["m_e_ratio"]

        self.sim.set_material(props=self.mat_props)

        self.sim.user_cs = {}
        for label, ds in self.cs_store.items():
            if ds.get("spline") is not None:
                self.sim.user_cs[label] = ds

        self.sim.x_pts = np.linspace(0, self.sim.Lx, self.sim.nx)
        self.sim.y_pts = np.linspace(0, self.sim.Ly, self.sim.ny)
        self.sim.X, self.sim.Y = np.meshgrid(self.sim.x_pts, self.sim.y_pts)

        # Do not create T_map / T_map_new here.
        # physics_engine.build_domain() creates all field arrays again
        # after computing the final geometry-dependent Ly.

    def get_params(self):
        params = {k: v.value() for k, v in self.inputs.items()}
        params.update(self.adv_params)
        params["sim_mode"] = self.combo_mode.currentText()
        params["geometry"] = self.combo_geometry.currentText()
        params["rf_enable"] = self.chk_rf.isChecked()
        params["rf_grid_idx"] = self.combo_rf_grid.currentIndex()
        params["rf_freq"] = self.spin_rf_freq.value()
        params["rf_amp"] = self.spin_rf_amp.value()
        params["pitch_mm"] = getattr(self, "pitch_mm", 0.0)

        grids = []
        for gw in self.grid_widgets:
            grids.append({
                "V": gw["V"].value(),
                "t": gw["t"].value(),
                "gap": gw["gap"].value(),
                "r": gw["r"].value(),
                "cham": gw["cham"].value(),
            })
        params["grids"] = grids

        inj_time_µs = self.inputs.get("inj_time_µs", None)
        if inj_time_µs is not None:
            params["inj_time"] = inj_time_µs.value() * 1e-6
        else:
            params["inj_time"] = 0.0

        params['macro_weight'] = self.sim.macro_weight   # <- ADD THIS
        return params

    def toggle_sim(self):
        if not np.any(self.sim.Ex):
            QMessageBox.warning(self, "Warning", "Build Domain first!")
            return

        # If we are starting (not pausing), ensure injection is enabled again
        if not self.sim_isRunning:
            self.sim.injection_enabled = True

        self.sim_isRunning = not self.sim_isRunning
        self.btn_toggle.setText("PAUSE BEAM" if self.sim_isRunning else "RESUME BEAM")

    def open_iedf_window(self):
        if self.iedf_window is None:
            self.iedf_window = IEDFWindow()
        self.iedf_window.show()

    def open_ppc_window(self):
        if self.ppc_window is None:
            self.ppc_window = PPCWindow(self)
        self.ppc_window.show()
        self.ppc_window.raise_()
        self.ppc_window.activateWindow()
        if hasattr(self, 'sim'):
            self.ppc_window.update_plot(self.sim)

    def open_phys_window(self):
        if self.phys_window is None:
            self.phys_window = PhysicalConstraintsWindow(self)
        self.phys_window.show()
        self.phys_window.raise_()
        self.phys_window.activateWindow()
        if hasattr(self, 'sim'):
            self.phys_window.update_plot(self.sim)

    def build_domain(self):
        self.sim_isRunning = False
        self.btn_toggle.setText("2. START BEAM")
        self.iter_history.clear()
        self.ebs_history.clear()
        self.div_history.clear()
        self.time_history.clear()
        self.transparency_history.clear()
        self.transparency3_history.clear()
        self.active_cells_history.clear()
        self.low_ppc_cells_history.clear()
        self.lblTime.setText("t_sim:        — us")
        self.T_histories = {i: [] for i in range(len(self.grid_widgets))}
        self.tracking_buffer.clear()

        self.lbl_status.setText("Building Multi-Grid Domain...")
        QApplication.processEvents()

        # ————————————————————————————————-
        # Compute grid spacing from Debye length: dx = dy = 0.8 * lambda_D
        # This ensures that the grid always resolves the Debye sheath with a
        # safety margin (0.8 < 1), as required by PIC theory.
        # ————————————————————————————————-
        _n0   = self.inputs["n0_plasma"].value()
        _Te   = self.inputs["Te_up"].value()
        _eps0 = 8.854e-12
        _q    = 1.602e-19
        _lambda_D_m  = np.sqrt(_eps0 * _Te * _q / (_n0 * _q**2))  # [m]
        _lambda_D_mm = _lambda_D_m * 1e3                           # [mm]
        _dxy_mm = 0.8 * _lambda_D_mm
        self.sim.dx = _dxy_mm
        self.sim.dy = _dxy_mm
        # Ensure Lx and Ly reflect current grid geometry if not manually overridden
        auto_Lx, auto_Ly = self.compute_grid_domain_size()
        if not getattr(self, '_user_overrode_Lx', False):
            self.adv_params["Lx"] = auto_Lx
        if not getattr(self, '_user_overrode_Ly', False):
            self.adv_params["Ly"] = auto_Ly

        self.apply_advanced_settings_to_sim()
        self.sim.build_domain(self.get_params())

        # Keep adv_params synchronized with the domain built
        if not getattr(self, '_user_overrode_Lx', False):
            self.adv_params["Lx"] = round(self.sim.Lx, 3)
        if not getattr(self, '_user_overrode_Ly', False):
            self.adv_params["Ly"] = round(self.sim.Ly, 3)

        # Sync the spinbox to the actual gap used (auto Debye or user override)
        gap_used = self.sim.upstream_gap_mm
        self.inputs["upstream_gap_mm"].blockSignals(True)
        self.inputs["upstream_gap_mm"].setValue(round(gap_used, 3))
        self.inputs["upstream_gap_mm"].blockSignals(False)

        # Auto-sync neutralizer x position to the computed default:
        # 9/10 of the distance between last grid exit and domain end.
        # Only updates if user has not manually changed the value from the
        # previously auto-set default (tracked via _last_auto_neut_x).
        if hasattr(self.sim, 'grid_x_ends') and self.sim.grid_x_ends:
            _x_exit = self.sim.grid_x_ends[-1]
            _neut_auto = round(_x_exit + 0.9 * (self.sim.Lx - _x_exit), 3)
        else:
            _neut_auto = round(self.sim.Lx - 0.5, 3)
        _last = getattr(self, '_last_auto_neut_x', None)
        _current_neut = round(self.adv_params.get("neut_x", _neut_auto), 3)
        if _last is None or abs(_current_neut - _last) <= 0.001:
            # User hasn't overridden — apply the new auto value
            self._last_auto_neut_x = _neut_auto
            self.adv_params["neut_x"] = _neut_auto
            print(f"[Build Domain] neut_x auto-set to {_neut_auto:.3f} mm  "
                  f"(90% into plume: x_exit={_x_exit if hasattr(self.sim, 'grid_x_ends') and self.sim.grid_x_ends else '?':.3f} mm, Lx={self.sim.Lx:.3f} mm)")

        self.draw_static_domain()

        cfg_name = getattr(self, "current_config_name", "config.json")
        self.lbl_status.setText(
            f"Domain Ready [{cfg_name}] | dx=dy={self.sim.dx:.4f} mm"
        )
        self.lbl_temp.setText(
            "Grid Temps: " + " | ".join([f"G{i+1}: 26°C" for i in range(len(self.grid_widgets))])
        )
        if hasattr(self, 'lbl_material'):
            self.lbl_material.setText(f"Grid Material: {self.mat_name}")

    def draw_static_domain(self):
        self.tempmesh = None
        self.dmg_mesh = None

        self.ax_live.clear()
        self.ax_live.set_title("Ion Beam Extraction & Particle Tracking", fontsize=10)
        self.ax_live.set_xlabel("Axial Position [mm]")
        self.ax_live.set_ylabel("Radial Position [mm]")
        self.ax_live.contourf(self.sim.X, self.sim.Y, self.sim.V, 20, cmap="viridis", alpha=0.4)

        gy, gx = np.where(self.sim.isBound)
        self.ax_live.scatter(gx * self.sim.dx, gy * self.sim.dy, s=4, c="k", alpha=0.8)

        max_v = max(gw['V'].value() for gw in self.grid_widgets) if self.grid_widgets else 1000.0
        self.scat_prim = self.ax_live.scatter([], [], c=[], s=2, cmap='turbo', vmin=0, vmax=max_v+300, alpha=0.8)
        self.scat_cex = self.ax_live.scatter([], [], c=[], s=3, cmap='turbo', vmin=0, vmax=max_v+300, alpha=1.0)
        self.scat_elec = self.ax_live.scatter([], [], s=1, c='#00FF00', alpha=0.8, zorder=5)

        # Always destroy and recreate cax_live: reusing an axis whose divider
        # relationship was broken by ax_live.clear() causes the colorbar to fill
        # the entire figure on the next rebuild after a simulation has been run.
        if hasattr(self, 'cax_live') and self.cax_live in self.fig.axes:
            self.cax_live.set_axes_locator(None)
            self.fig.delaxes(self.cax_live)
        divider = make_axes_locatable(self.ax_live)
        self.cax_live = divider.append_axes("right", size="3%", pad=0.1)

        self.cbar_energy = self.fig.colorbar(self.scat_prim, cax=self.cax_live)
        self.cbar_energy.ax.set_title("[eV]", fontsize=8, pad=3)

        self.ax_live.set_xlim(0, self.sim.Lx)
        self.ax_live.set_ylim(0, self.sim.Ly)
        self.ax_live.set_title("Beam Extraction & Tracking")
        self.canvas.draw_idle()
        if self.sim.Tmap is not None and len(self.sim.mask_grids) > 0:
            grid_mask = np.zeros_like(self.sim.isBound, dtype=bool)
            for mg in self.sim.mask_grids:
                if mg.shape == grid_mask.shape:
                    grid_mask |= mg
            TdisplayC = np.where(grid_mask, self.sim.Tmap - 273.15, np.nan)

            self.ax_temp.clear()
            self.ax_temp.set_title("Grid Temp Map °C")
            self.ax_temp.set_facecolor("black")
            self.tempmesh = self.ax_temp.pcolormesh(
                self.sim.X, self.sim.Y, TdisplayC,
                cmap="inferno", shading="nearest"
            )
            # Always destroy and recreate cax_temp for same reason as cax_live.
            if hasattr(self, 'cax_temp') and self.cax_temp in self.fig.axes:
                self.cax_temp.set_axes_locator(None)
                self.fig.delaxes(self.cax_temp)
            dividert = make_axes_locatable(self.ax_temp)
            self.cax_temp = dividert.append_axes("right", size="5%", pad=0.1)
            self.cbartemp = self.fig.colorbar(self.tempmesh, cax=self.cax_temp)
            self.cbartemp.set_label("Temperature °C")
            self.ax_temp.set_xlim(0, self.sim.Lx)
            self.ax_temp.set_ylim(0, self.sim.Ly)

    def run_sim_step(self):
        if not self.sim_isRunning:
            return

        params = self.get_params()
        step_out = self.sim.step(params)
        if len(step_out) == 5:
            remeshed, min_pot, current_div, T_grids, trans_last_frame = step_out
        else:
            remeshed, min_pot, current_div, T_grids = step_out
            trans_last_frame = 0.0

        inj_time = params.get("inj_time", 0.0)
        inj_limited = inj_time > 0.0

        if inj_limited and (not self.sim.injection_enabled) and (not self.sim.has_active_particles()):
            self.sim_isRunning = False
            self.btn_toggle.setText("2. START BEAM")
            self.lbl_status.setText("Status: Injection completed and all particles removed. Simulation stopped.")
            return

        t_sim = self.sim.iteration * self.sim.dt
        transparency = self.sim.get_transparency()

        self.lblTime.setText(f"t_sim: {t_sim * 1e6:.2f} us")
        self.lblTransparency.setText(
            f"Transparency tot: {transparency:.3f}\n"
            f"Transparency frame: {trans_last_frame:.3f}\n"
            f"Exit vx mean: {self.sim.exit_vx_mean: .2e} m/s\n"
            f"Exit |v| mean: {self.sim.exit_v_mean: .2e} m/s\n"
            f"Exit E mean: {self.sim.exit_energy_mean_eV: .1f} eV\n"
            f"Exit count step: {self.sim.exit_count_step}"
        )
        QApplication.processEvents()

        if remeshed:
            self.lbl_status.setText("Domain Remeshed (Thermal or Erosion)!")
            self.draw_static_domain()

        if self.sim.iteration % 1 == 0 and self.scat_prim is not None and self.scat_cex is not None:
            p_x = self.sim.p_x[:self.sim.num_p]
            p_y = self.sim.p_y[:self.sim.num_p]
            p_vx = self.sim.p_vx[:self.sim.num_p]
            p_vy = self.sim.p_vy[:self.sim.num_p]
            p_vz = self.sim.p_vz[:self.sim.num_p]
            p_is_cex = self.sim.p_isCEX[:self.sim.num_p]

            e_x = self.sim.e_x[:self.sim.num_e]
            e_y = self.sim.e_y[:self.sim.num_e]
            e_vx = self.sim.e_vx[:self.sim.num_e]
            e_vy = self.sim.e_vy[:self.sim.num_e]

            prim_mask = ~p_is_cex
            cex_mask = p_is_cex

            self.scat_prim.set_offsets(
                np.column_stack((p_x[prim_mask], p_y[prim_mask]))
                if np.any(prim_mask) else np.empty((0, 2))
            )
            self.scat_cex.set_offsets(
                np.column_stack((p_x[cex_mask], p_y[cex_mask]))
                if np.any(cex_mask) else np.empty((0, 2))
            )
            self.scat_elec.set_offsets(
                np.column_stack((e_x, e_y))
                if len(e_x) > 0 else np.empty((0, 2))
            )

            energy_min = float('inf')
            energy_max = float('-inf')

            if np.any(prim_mask):
                v_sq_prim = p_vx[prim_mask]**2 + p_vy[prim_mask]**2 + p_vz[prim_mask]**2
                e_prim = (0.5 * self.sim.m_ion * v_sq_prim) / self.sim.q
                self.scat_prim.set_array(e_prim)
                energy_min = min(energy_min, np.min(e_prim))
                energy_max = max(energy_max, np.max(e_prim))

            if np.any(cex_mask):
                v_sq_cex = p_vx[cex_mask]**2 + p_vy[cex_mask]**2 + p_vz[cex_mask]**2
                e_cex = (0.5 * self.sim.m_ion * v_sq_cex) / self.sim.q
                self.scat_cex.set_array(e_cex)
                energy_min = min(energy_min, np.min(e_cex))
                energy_max = max(energy_max, np.max(e_cex))
            
            if energy_min != float('inf'):
                if energy_max <= energy_min:
                    energy_max = energy_min + 1.0
                self.scat_prim.set_clim(energy_min, energy_max)
                self.scat_cex.set_clim(energy_min, energy_max)

            if self.iedf_window and self.iedf_window.isVisible():
                max_v = max([g["V"].value() for g in self.grid_widgets]) if self.grid_widgets else 1000.0
                self.iedf_window.update_histogram(
                    p_vx, p_vy, p_is_cex,
                    e_x, e_vx, e_vy,
                    self.sim.m_ion, self.sim.m_e, self.sim.q, max_v
                )

            if self.ppc_window and self.ppc_window.isVisible():
                self.ppc_window.update_plot(self.sim)

        # --- Energy conservation: always computed (for terminal warnings) ---
        self.sim.get_total_energy()
        if self.phys_window and self.phys_window.isVisible():
            self.phys_window.update_plot(self.sim)

        if self.chk_track_ptcls.isChecked():
            ptcl_data = self.sim.get_particle_kinematics()
            if ptcl_data.size > 0:
                self.tracking_buffer.append(ptcl_data)

        self.iter_history.append(self.sim.iteration)
        self.ebs_history.append(min_pot)
        self.div_history.append(current_div)
        self.time_history.append(t_sim)
        self.transparency_history.append(transparency)
        self.transparency3_history.append(trans_last_frame)
        self.active_cells_history.append(self.sim.total_active_cells)
        self.low_ppc_cells_history.append(self.sim.low_ppc_cells)

        for i, T in enumerate(T_grids):
            self.T_histories[i].append(T)

        self.line_ebs.set_data(self.iter_history, self.ebs_history)
        self.line_div.set_data(self.iter_history, self.div_history)

        self.ax_ebs.set_xlim(max(0, self.sim.iteration - 400), max(100, self.sim.iteration))
        self.ax_div.set_xlim(max(0, self.sim.iteration - 400), max(100, self.sim.iteration))

        if len(self.ebs_history) > 0:
            y_min = min(self.ebs_history)
            y_max = max(self.ebs_history)
            if y_max <= y_min:
                y_max = y_min + 1.0
            pad = max(5.0, 0.1 * (y_max - y_min))
            self.ax_ebs.set_ylim(y_min - pad, y_max + pad)

        finite_div = [d for d in self.div_history if np.isfinite(d)]
        if len(finite_div) > 0:
            div_max = max(finite_div)
            self.ax_div.set_ylim(0, max(45, div_max * 1.1))
        else:
            self.ax_div.set_ylim(0, 45)

        groove_idx = 1 if len(self.sim.mask_grids) > 1 else 0
        groove_face = "downstream"
        y_mm, depth_um = self.sim.get_groove_profile(
            groove_idx,
            thresh=self.inputs["Thresh"].value(),
            face=groove_face
        )
        if y_mm.size > 0:
            self.line_groove.set_data(y_mm, depth_um)
            self.ax_groove.set_xlim(0, float(y_mm.max()))
            dmax = float(depth_um.max()) if depth_um.size > 0 else 0.0
            self.ax_groove.set_ylim(max(dmax * 1.1, 1.0), 0.0)
            self.ax_groove.set_title(
                f"Accel Grid Erosion Profile — {groove_face} face (Grid {groove_idx + 1})"
            )

        if self.sim.Tmap is not None and len(self.sim.mask_grids) > 0:
            grid_mask = np.zeros_like(self.sim.isBound, dtype=bool)
            for mg in self.sim.mask_grids:
                if mg.shape == grid_mask.shape:
                    grid_mask |= mg

            T_display_C = np.where(grid_mask, self.sim.Tmap - 273.15, np.nan)

            if getattr(self, "tempmesh", None) is None or self.tempmesh.get_array().size != T_display_C.size:
                self.ax_temp.clear()
                self.ax_temp.set_title("Grid Temp Map (°C)")
                self.ax_temp.set_facecolor("black")

                self.tempmesh = self.ax_temp.pcolormesh(
                    self.sim.X, self.sim.Y, T_display_C,
                    cmap="inferno", shading="nearest"
                )

                if not hasattr(self, "cax_temp"):
                    divider_t = make_axes_locatable(self.ax_temp)
                    self.cax_temp = divider_t.append_axes("right", size="5%", pad=0.1)
                else:
                    self.cax_temp.set_axes_locator(None)
                    self.cax_temp.clear()

                self.cbar_temp = self.fig.colorbar(self.tempmesh, cax=self.cax_temp)
                self.cbar_temp.set_label("Temperature (°C)")
            else:
                self.tempmesh.set_array(T_display_C.ravel())
                finite_vals = T_display_C[np.isfinite(T_display_C)]
                if finite_vals.size > 0:
                    vmin = float(np.min(finite_vals))
                    vmax = float(np.max(finite_vals))
                    if vmax <= vmin:
                        vmax = vmin + 1.0
                    self.tempmesh.set_clim(vmin, vmax)

            self.ax_temp.set_xlim(0, self.sim.Lx)
            self.ax_temp.set_ylim(0, self.sim.Ly)

        if getattr(self, "dmg_mesh", None) is None or self.dmg_mesh.get_array().size != self.sim.damage_map.size:
            self.ax_dmg.clear()
            self.ax_dmg.set_title("Sputter Damage Map", fontsize=10)
            self.ax_dmg.set_xlabel("Axial Position [mm]", fontsize=8)
            self.ax_dmg.set_ylabel("r [mm]", fontsize=8)
            self.dmg_mesh = self.ax_dmg.pcolormesh(
                self.sim.X, self.sim.Y, self.sim.damage_map,
                cmap="hot", shading="nearest"
            )
            gy, gx = np.where(self.sim.isBound)
            self.ax_dmg.scatter(gx * self.sim.dx, gy * self.sim.dy, s=2, c="grey", alpha=0.5)
            self.ax_dmg.set_xlim(0, self.sim.Lx)
            self.ax_dmg.set_ylim(0, self.sim.Ly)
        else:
            self.dmg_mesh.set_array(self.sim.damage_map.ravel())
            dmg_max = float(np.max(self.sim.damage_map))
            if dmg_max > 0:
                self.dmg_mesh.set_clim(0, dmg_max)

        self.lbl_status.setText(
            f"Ions {self.sim.num_p} e- {self.sim.num_e} Iter {self.sim.iteration}"
        )

        t_str = " | ".join([f"G{i+1}: {T - 273.15:.1f}°C" for i, T in enumerate(T_grids)])
        self.lbl_temp.setText("Grid Temps: " + t_str)

        self.canvas.draw_idle()

        if self.chk_record.isChecked():
            self.recorded_frames.append(self.canvas.grab())
            self.chk_record.setText(f"Record Frames ({len(self.recorded_frames)})")

    def _create_progress_dialog(self, title, label, total):
        progress = QProgressDialog(label, "Cancel", 0, max(1, total), self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        return progress

    def export_csv(self):
        n = len(self.iter_history)
        if n == 0:
            QMessageBox.warning(self, "No Data", "No simulation history data to export.")
            return

        startdir = _safe_start_dir()
        if not os.access(startdir, os.W_OK):
            startdir = os.path.expanduser("~")
        suggested = os.path.join(startdir, time.strftime("pybemcs_%Y%m%d%H%M%S.csv"))

        file_name, _ = QFileDialog.getSaveFileName(self, "Export Data", suggested, "CSV Files (*.csv)")
        if not file_name:
            return

        was_running = self.sim_isRunning
        self.sim_isRunning = False

        progress = self._create_progress_dialog("Exporting CSV Data", "Preparing CSV export...", n)
        cancelled = False

        try:
            with open(file_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = [
                    'iteration', 't_sim_s',
                    'minpotential',
                    'beamdivergence°',
                    'transparency',
                    'total_active_cells',
                    'cells_less_than_3_macroparticles'
                ]
                for i in range(len(self.T_histories)):
                    header.append(f"grid_{i+1}_temp_K")
                writer.writerow(header)

                chunk_size = max(1, n // 100)
                for j in range(n):
                    if progress.wasCanceled():
                        cancelled = True
                        break

                    row = [
                        self.iter_history[j],
                        self.time_history[j] if j < len(self.time_history) else '',
                        self.ebs_history[j],
                        self.div_history[j],
                        self.transparency_history[j] if j < len(self.transparency_history) else '',
                        self.active_cells_history[j] if j < len(self.active_cells_history) else '',
                        self.low_ppc_cells_history[j] if j < len(self.low_ppc_cells_history) else ''
                    ]
                    for i in range(len(self.T_histories)):
                        row.append(self.T_histories[i][j] if j < len(self.T_histories[i]) else "")
                    writer.writerow(row)

                    if (j + 1) % chunk_size == 0 or j == n - 1:
                        pct = int(((j + 1) / n) * 100)
                        progress.setValue(j + 1)
                        progress.setLabelText(f"Writing row {j+1} of {n} ({pct}%)...")
                        self.lbl_status.setText(f"Exporting CSV: {j+1}/{n} ({pct}%)...")
                        QApplication.processEvents()

            if cancelled:
                if os.path.exists(file_name):
                    try:
                        os.remove(file_name)
                    except Exception:
                        pass
                self.lbl_status.setText("CSV export cancelled.")
                QMessageBox.information(self, "Cancelled", "CSV export was cancelled.")
            else:
                progress.setValue(n)
                self.lbl_status.setText(f"CSV saved: {file_name}")
                QMessageBox.information(self, "Success", f"CSV exported:\n{file_name}")
        except Exception as e:
            self.lbl_status.setText("CSV export failed.")
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{e}")
        finally:
            progress.close()
            self.sim_isRunning = was_running

    def exporttrackingdata(self):
        if len(self.tracking_buffer) == 0:
            QMessageBox.warning(self, "No Data", "No particle tracking data recorded.")
            return

        startdir = _safe_start_dir()
        if not os.access(startdir, os.W_OK):
            startdir = os.path.expanduser("~")

        suggested = os.path.join(startdir, time.strftime("pybemcs_particles_%Y%m%d%H%M%S.csv"))
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Particle Data", suggested, "CSV Files (*.csv)"
        )
        if not filename:
            return

        was_running = self.sim_isRunning
        self.sim_isRunning = False

        try:
            data = np.vstack(self.tracking_buffer)
            n_rows = len(data)
            ncols = data.shape[1] if data.ndim > 1 else 0

            if ncols == 8:
                header = ['time_s', 'x_mm', 'y_mm', 'vx_ms', 'vy_ms', 'vz_ms', 'energy_eV', 'type']
            elif ncols == 6:
                header = ['x_mm', 'y_mm', 'vx_ms', 'vy_ms', 'vz_ms', 'isCEX']
            else:
                header = [f"col_{c}" for c in range(ncols)]

            progress = self._create_progress_dialog("Exporting Particle Data", "Writing particle tracking CSV...", n_rows)
            cancelled = False

            chunk_size = max(500, n_rows // 100)
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for idx in range(0, n_rows, chunk_size):
                    if progress.wasCanceled():
                        cancelled = True
                        break

                    chunk = data[idx:idx + chunk_size]
                    writer.writerows(chunk.tolist())

                    written = min(idx + chunk_size, n_rows)
                    pct = int((written / n_rows) * 100)
                    progress.setValue(written)
                    progress.setLabelText(f"Writing particle {written} of {n_rows} ({pct}%)...")
                    self.lbl_status.setText(f"Exporting particles: {written}/{n_rows} ({pct}%)...")
                    QApplication.processEvents()

            if cancelled:
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except Exception:
                        pass
                self.lbl_status.setText("Particle export cancelled.")
                QMessageBox.information(self, "Cancelled", "Particle data export was cancelled.")
            else:
                progress.setValue(n_rows)
                self.lbl_status.setText(f"Particle data exported: {filename}")
                QMessageBox.information(self, "Success", f"Particle data exported:\n{filename}")
        except PermissionError:
            alt = os.path.join(os.path.expanduser("~"), os.path.basename(filename))
            try:
                data = np.vstack(self.tracking_buffer)
                with open(alt, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(data.tolist())
                QMessageBox.information(self, "Saved to Home",
                    f"Permission denied at original path.\nSaved to:\n{alt}")
            except Exception as e2:
                QMessageBox.critical(self, "Export Error", f"Failed to export particle data:\n{e2}")
        except Exception as e:
            self.lbl_status.setText("Particle export failed.")
            QMessageBox.critical(self, "Export Error", f"Failed to export particle data:\n{e}")
        finally:
            if 'progress' in locals():
                progress.close()
            self.sim_isRunning = was_running

    def save_gif(self):
        if len(self.recorded_frames) == 0:
            QMessageBox.warning(self, "No Frames", "No recorded frames to save.")
            return

        suggested = os.path.join(_safe_start_dir(), time.strftime("pybemcs_%Y%m%d_%H%M%S.gif"))
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Animation", suggested, "GIF Files (*.gif)")
        if not file_name:
            return

        was_running = self.sim_isRunning
        self.sim_isRunning = False

        total = len(self.recorded_frames)
        # Total steps: total frames + 1 finalize/write step
        progress = self._create_progress_dialog("Saving GIF Animation", "Preparing GIF export...", total + 1)
        cancelled = False

        try:
            try:
                from PIL import Image
            except ImportError:
                progress.close()
                QMessageBox.warning(self, "Error", "Install Pillow: pip install Pillow")
                return

            palettized_frames = []
            for i, qpix in enumerate(self.recorded_frames):
                if progress.wasCanceled():
                    cancelled = True
                    break

                qimg = qpix.toImage().convertToFormat(QImage.Format_RGBA8888)
                width = qimg.width()
                height = qimg.height()
                ptr = qimg.bits()
                ptr.setsize(qimg.byteCount())
                arr = np.array(ptr, dtype=np.uint8).reshape(height, width, 4)
                im = Image.fromarray(arr[:, :, :3])
                # Pre-quantize frame to 256-color palette for 20x faster, smooth export
                palettized_frames.append(im.quantize(colors=256, method=Image.Quantize.FASTOCTREE))

                pct = int(((i + 1) / (total + 1)) * 100)
                progress.setValue(i + 1)
                progress.setLabelText(f"Processing frame {i+1} of {total} ({pct}%)...")
                self.lbl_status.setText(f"Saving GIF: {i+1}/{total} ({pct}%)...")
                QApplication.processEvents()

            if not cancelled and palettized_frames:
                progress.setLabelText("Finalizing & writing GIF to disk (98%)...")
                QApplication.processEvents()

                palettized_frames[0].save(
                    file_name,
                    save_all=True,
                    append_images=palettized_frames[1:],
                    duration=50,
                    loop=0,
                    optimize=False,
                )

                progress.setValue(total + 1)
                progress.setLabelText("Export completed (100%)")
                QApplication.processEvents()

            if cancelled:
                if os.path.exists(file_name):
                    try:
                        os.remove(file_name)
                    except Exception:
                        pass
                self.lbl_status.setText("GIF export cancelled.")
                QMessageBox.information(self, "Cancelled", "GIF export was cancelled.")
            else:
                self.recorded_frames.clear()
                self.chk_record.setChecked(False)
                self.chk_record.setText("Record Frames (0)")
                self.lbl_status.setText(f"GIF saved: {file_name}")
                QMessageBox.information(self, "Success", f"GIF saved to:\n{file_name}")
        except Exception as e:
            self.lbl_status.setText("GIF save failed.")
            QMessageBox.critical(self, "Error", f"GIF save failed:\n{e}")
        finally:
            progress.close()
            self.sim_isRunning = was_running

    # —————————————————————————————————
    # Build standalone .exe via PyInstaller
    # —————————————————————————————————
    def build_exe(self):
        """Compile main.py into a standalone executable using PyInstaller.

        The user selects the destination folder; a progress dialog with live
        log output is shown while the build runs in a background thread.
        """
        # 1. Confirm PyInstaller is reachable
        try:
            import PyInstaller  # noqa: F401  (just a presence check)
        except ImportError:
            QMessageBox.critical(
                self, "PyInstaller Not Found",
                "PyInstaller is not installed in the current Python environment.\n"
                "Install it with:\n\n    pip install pyinstaller"
            )
            return

        # 2. Pick destination folder
        dest_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder for .exe",
            _safe_start_dir(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not dest_dir:
            return

        # 3. Resolve script path
        script_path = os.path.abspath(
            getattr(sys, "frozen", None) and sys.executable
            or __file__
        )
        if not os.path.isfile(script_path) or not script_path.endswith(".py"):
            # Fallback: look for main.py next to the current script
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        if not os.path.isfile(script_path):
            QMessageBox.critical(self, "Error", f"Cannot find main.py:\n{script_path}")
            return

        # 4. Build progress dialog
        progress = QProgressDialog(
            "Initialising PyInstaller...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Building .exe")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(550)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        # Keep a log for post-mortem inspection
        self._build_log_lines = []

        # 5. Start worker thread
        worker = PyInstallerWorker(script_path, dest_dir, parent=self)
        self._build_worker = worker   # keep alive while thread runs

        def _on_line(line):
            self._build_log_lines.append(line)
            # Show last meaningful (non-empty) line in the dialog label
            stripped = line.strip()
            if stripped:
                # Truncate very long lines so they fit in the dialog
                display = stripped if len(stripped) <= 80 else stripped[:77] + "..."
                progress.setLabelText(display)
            QApplication.processEvents()

        def _on_pct(pct):
            if not progress.wasCanceled():
                progress.setValue(pct)
            QApplication.processEvents()

        def _on_finished(success, message):
            progress.setValue(100)
            progress.close()
            if success:
                self.lbl_status.setText(f"Build complete -> {message}")
                QMessageBox.information(
                    self, "Build Successful",
                    f"Executable created successfully:\n\n{message}\n\n"
                    "config.json has been copied alongside the .exe (if present)."
                )
            else:
                self.lbl_status.setText("Build failed.")
                log_snippet = "\n".join(self._build_log_lines[-20:])
                QMessageBox.critical(
                    self, "Build Failed",
                    f"{message}\n\n—- Last build output —-\n{log_snippet}"
                )

        def _on_cancel():
            if worker.isRunning():
                worker.terminate()
                worker.wait(2000)
                self.lbl_status.setText("Build cancelled.")

        worker.progress_line.connect(_on_line)
        worker.progress_pct.connect(_on_pct)
        worker.finished.connect(_on_finished)
        progress.canceled.connect(_on_cancel)

        worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DigitalTwinApp()

    if len(sys.argv) > 1:
        cfg = load_json_config(sys.argv[1])
        if cfg is not None:
            try:
                window.current_config_path = sys.argv[1]
                window.current_config_name = os.path.basename(sys.argv[1])
                window.apply_config(cfg, config_name=window.current_config_name)
            except Exception as e:
                QMessageBox.critical(window, "Config Error", f"Failed to load config:\n{e}")
        else:
            QMessageBox.warning(window, "File Not Found",
                                f"Config file not found:\n{sys.argv[1]}")

    window.show()
    sys.exit(app.exec_())