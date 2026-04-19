"""
This code handles the primary for the GUI/UI design. One can use this
section to add or modify any new features to the screen.
"""
import os
import sys


def _ensure_qt_lib_path():
    if os.environ.get("PYBEMCS_QT_PATCHED") == "1":
        return
    try:
        import PyQt5  # noqa: F401
        qt_lib = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "lib")
    except Exception:
        return
    if not os.path.isdir(qt_lib):
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current.split(":")[0] == qt_lib:
        return  # already first in the search order
    os.environ["LD_LIBRARY_PATH"] = qt_lib + (":" + current if current else "")
    os.environ["PYBEMCS_QT_PATCHED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_qt_lib_path()

# Qt platform fall-back chain for Ubuntu 24.04+ / Wayland / headless CI.
if not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb;wayland;wayland-egl"

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QDoubleSpinBox, QPushButton, QCheckBox,
                             QFrame, QMessageBox, QFileDialog, QApplication, QComboBox,
                             QScrollArea, QGroupBox, QAction, QDialog, QFormLayout,
                             QSpinBox, QMenu, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSplitter, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt
from scipy.interpolate import UnivariateSpline
import csv
from physics_engine import DigitalTwinSimulator

# --- BEAM SPECIES DIALOG ---
class BeamSpeciesDialog(QDialog):
    # Common ion species presets: (name, mass_amu, default_charge)
    PRESETS = [
        ('Custom', 0, 1),
        ('Xenon (Xe)', 131.293, 1),
        ('Krypton (Kr)', 83.798, 1),
        ('Argon (Ar)', 39.948, 1),
        ('Nitrogen (N₂)', 28.014, 1),
        ('Oxygen (O₂)', 31.998, 1),
        ('Hydrogen (H₂)', 2.016, 1),
        ('Helium (He)', 4.0026, 1),
        ('Mercury (Hg)', 200.59, 1),
        ('Cesium (Cs)', 132.905, 1),
    ]

    def __init__(self, current_mass_amu, current_charge, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ion Beam Species")
        self.setMinimumWidth(350)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Select a preset or enter custom values:</b>"))

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
        self.spin_mass.setSuffix(' amu')
        form.addRow('Atomic / Molecular Mass:', self.spin_mass)

        self.spin_charge = QSpinBox()
        self.spin_charge.setRange(1, 10)
        self.spin_charge.setValue(current_charge)
        self.spin_charge.setPrefix('+')
        form.addRow('Charge State:', self.spin_charge)

        layout.addLayout(form)

        btn_box = QHBoxLayout()
        save_btn = QPushButton("Apply")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

        # Try to auto-select the matching preset
        self._sync_preset_from_values(current_mass_amu)

    def _sync_preset_from_values(self, mass_amu):
        for i, (_, m, _) in enumerate(self.PRESETS):
            if abs(m - mass_amu) < 0.01:
                self.combo_preset.blockSignals(True)
                self.combo_preset.setCurrentIndex(i)
                self.combo_preset.blockSignals(False)
                return
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentIndex(0)  # Custom
        self.combo_preset.blockSignals(False)

    def _on_preset(self, idx):
        if idx > 0:
            _, mass, charge = self.PRESETS[idx]
            self.spin_mass.setValue(mass)
            self.spin_charge.setValue(charge)

    def get_values(self):
        return self.spin_mass.value(), self.spin_charge.value()


# --- GRID MATERIAL DIALOG ---
class GridMaterialDialog(QDialog):
    PRESETS = {
        'Molybdenum':   {'k': 138.0, 'rho': 10280.0, 'cp': 250.0,
                         'emissivity': 0.80, 'alpha': 4.8e-6, 'E_mod': 329e9,
                         'Y_coeff': 1.05e-4, 'E_th': 30.0},
        'Steel (SS316)':{'k': 16.3,  'rho': 8000.0,  'cp': 500.0,
                         'emissivity': 0.60, 'alpha': 16.0e-6, 'E_mod': 193e9,
                         'Y_coeff': 2.8e-4,  'E_th': 25.0},
        'Titanium':     {'k': 21.9,  'rho': 4507.0,  'cp': 520.0,
                         'emissivity': 0.50, 'alpha': 8.6e-6, 'E_mod': 116e9,
                         'Y_coeff': 1.8e-4,  'E_th': 20.0},
        'Graphite':     {'k': 120.0, 'rho': 2200.0,  'cp': 710.0,
                         'emissivity': 0.85, 'alpha': 3.0e-6, 'E_mod': 11e9,
                         'Y_coeff': 3.5e-4,  'E_th': 15.0},
        'Custom':       None,
    }

    FIELD_DEFS = [
        ('k',          'Thermal Conductivity (W/m/K):',  0.1, 5000, 138.0,  1),
        ('rho',        'Density (kg/m³):',               100, 25000, 10280.0, 0),
        ('cp',         'Specific Heat (J/kg/K):',        50, 5000, 250.0,   0),
        ('emissivity', 'Emissivity (0-1):',              0.01, 1.0, 0.8,    2),
        ('alpha',      'Thermal Expansion (1/K):',       0, 1e-3, 4.8e-6,  7),
        ('E_mod',      "Young's Modulus (Pa):",           1e8, 1e12, 329e9,  0),
        ('Y_coeff',    'Sputter Yield Coeff:',           0, 1e-2, 1.05e-4, 6),
        ('E_th',       'Sputter Threshold (eV):',        0, 500,  30.0,    1),
    ]

    def __init__(self, current_mat_name, current_props, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grid Material Properties")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Select a preset or enter custom values:</b>"))

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
            self.combo.setCurrentText('Custom')

    def _on_preset(self, name):
        props = self.PRESETS.get(name)
        if props is not None:
            for key, spin in self.spins.items():
                spin.setValue(props[key])

    def get_values(self):
        name = self.combo.currentText()
        props = {k: s.value() for k, s in self.spins.items()}
        return name, props


# --- CROSS-SECTION VIEWER / MANAGER WINDOW ---
class CrossSectionViewerWindow(QWidget):
    """
    Allows the user to import CSV cross-section tables (Energy [eV] vs σ [m²]),
    visualize them, and fit a cubic spline for runtime interpolation.
    Supports multiple reaction channels: CX, SEE yield, and custom.
    """
    REACTION_TYPES = ['Charge Exchange (CX)', 'Secondary Electron Yield (SEE)', 'Custom Reaction']

    def __init__(self, cs_store, parent=None):
        super().__init__()  # No parent — opens as independent top-level window
        self.cs_store = cs_store
        self.setWindowTitle('Cross-Section Data Manager')
        self.setMinimumSize(950, 600)
        self.resize(950, 600)

        # Use a QSplitter so user can resize the control panel vs plot
        splitter = QSplitter(Qt.Horizontal, self)

        # --- Left: controls panel ---
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(8, 8, 8, 8)

        left.addWidget(QLabel('<b>Reaction Type:</b>'))
        self.combo_type = QComboBox()
        self.combo_type.addItems(self.REACTION_TYPES)
        left.addWidget(self.combo_type)

        self.btn_import = QPushButton('Import CSV...')
        self.btn_import.setStyleSheet("background-color: #AED6F1; font-weight: bold;")
        self.btn_import.clicked.connect(self._import_csv)
        left.addWidget(self.btn_import)

        left.addWidget(QLabel('<b>Loaded Datasets:</b>'))
        self.combo_datasets = QComboBox()
        self.combo_datasets.currentIndexChanged.connect(self._on_dataset_selected)
        left.addWidget(self.combo_datasets)

        self.btn_remove = QPushButton('Remove Selected')
        self.btn_remove.clicked.connect(self._remove_dataset)
        left.addWidget(self.btn_remove)

        left.addWidget(QLabel('<b>Spline Smoothing:</b>'))
        self.spin_smooth = QDoubleSpinBox()
        self.spin_smooth.setRange(0.0, 1e6)
        self.spin_smooth.setValue(0.0)
        self.spin_smooth.setDecimals(2)
        self.spin_smooth.setToolTip('0 = interpolating spline (passes through all points)')
        left.addWidget(self.spin_smooth)

        self.btn_fit = QPushButton('Fit Spline')
        self.btn_fit.setStyleSheet("background-color: #ABEBC6; font-weight: bold;")
        self.btn_fit.clicked.connect(self._fit_spline)
        left.addWidget(self.btn_fit)

        self.lbl_info = QLabel('No data loaded.')
        self.lbl_info.setWordWrap(True)
        left.addWidget(self.lbl_info)

        # Data preview table
        left.addWidget(QLabel('<b>Data Preview:</b>'))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Energy (eV)', 'Cross-Section (m²)'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left.addWidget(self.table)

        left_widget.setMinimumWidth(280)
        left_widget.setMaximumWidth(350)

        # --- Right: matplotlib plot ---
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

    def _import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Import Cross-Section CSV', '',
            'CSV Files (*.csv);;Text Files (*.txt *.dat);;All Files (*)')
        if not fname:
            return

        try:
            raw = np.loadtxt(fname, delimiter=None, comments='#', skiprows=0)
            if raw.ndim == 1:
                raise ValueError("File must have at least two columns.")
            # Try comma-delimited if auto didn't work
        except Exception:
            try:
                raw = np.loadtxt(fname, delimiter=',', comments='#', skiprows=1)
            except Exception as e2:
                QMessageBox.critical(self, 'Import Error', f'Could not parse CSV:\n{e2}')
                return

        if raw.ndim != 2 or raw.shape[1] < 2:
            QMessageBox.critical(self, 'Import Error',
                                 'File must have at least 2 columns (Energy, Cross-Section).')
            return

        energy = raw[:, 0]
        cs = raw[:, 1]

        # Sort by energy
        order = np.argsort(energy)
        energy = energy[order]
        cs = cs[order]

        rtype = self.combo_type.currentText()
        # Build a unique label
        base = rtype.split('(')[-1].replace(')', '').strip()
        count = sum(1 for k in self.cs_store if k.startswith(base))
        label = f"{base}_{count+1}" if count > 0 else base

        self.cs_store[label] = {
            'energy': energy,
            'cs': cs,
            'spline': None,
            'type': rtype
        }

        self._refresh_combo()
        self._plot_current()
        self.lbl_info.setText(f'Loaded "{label}": {len(energy)} points, '
                              f'E range [{energy[0]:.1f}, {energy[-1]:.1f}] eV')

    def _remove_dataset(self):
        label = self.combo_datasets.currentText()
        if label and label in self.cs_store:
            del self.cs_store[label]
            self._refresh_combo()
            self._plot_current()
            self.lbl_info.setText(f'Removed "{label}".')

    def _on_dataset_selected(self, idx):
        self._plot_current()
        self._update_table()

    def _update_table(self):
        label = self.combo_datasets.currentText()
        if not label or label not in self.cs_store:
            self.table.setRowCount(0)
            return
        ds = self.cs_store[label]
        n = min(len(ds['energy']), 50)  # Show first 50 rows
        self.table.setRowCount(n)
        for i in range(n):
            self.table.setItem(i, 0, QTableWidgetItem(f"{ds['energy'][i]:.4e}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{ds['cs'][i]:.4e}"))

    def _fit_spline(self):
        label = self.combo_datasets.currentText()
        if not label or label not in self.cs_store:
            QMessageBox.warning(self, 'No Data', 'Select a dataset first.')
            return

        ds = self.cs_store[label]
        energy = ds['energy']
        cs = ds['cs']

        if len(energy) < 4:
            QMessageBox.warning(self, 'Too Few Points',
                                'Need at least 4 data points for spline fitting.')
            return

        try:
            s_val = self.spin_smooth.value()
            # Fit in log-log space for better behavior across decades
            log_e = np.log10(np.maximum(energy, 1e-30))
            log_cs = np.log10(np.maximum(cs, 1e-50))
            spline = UnivariateSpline(log_e, log_cs, s=s_val, k=3)
            ds['spline'] = spline
            self.lbl_info.setText(f'Spline fitted for "{label}" (smoothing={s_val}).\n'
                                  f'This data will be used in the simulation.')
            self._plot_current()
        except Exception as e:
            QMessageBox.critical(self, 'Spline Error', f'Failed to fit spline:\n{e}')

    def _plot_current(self):
        self.ax.clear()
        label = self.combo_datasets.currentText()

        if label and label in self.cs_store:
            ds = self.cs_store[label]
            energy = ds['energy']
            cs = ds['cs']

            self.ax.loglog(energy, cs, 'o', ms=4, label='Data', color='#2980B9')

            if ds['spline'] is not None:
                e_fine = np.logspace(np.log10(max(energy[0], 1e-30)),
                                     np.log10(energy[-1]), 500)
                log_cs_fine = ds['spline'](np.log10(e_fine))
                cs_fine = 10.0 ** log_cs_fine
                self.ax.loglog(e_fine, cs_fine, '-', lw=2, label='Spline Fit',
                               color='#E74C3C')

            self.ax.set_xlabel('Energy (eV)')
            self.ax.set_ylabel('Cross-Section (m²)')
            self.ax.set_title(f'{label} — {ds.get("type", "")}')
            self.ax.legend()
            self.ax.grid(True, which='both', alpha=0.3)
        else:
            self.ax.set_title('No data loaded')
            self.ax.set_xlabel('Energy (eV)')
            self.ax.set_ylabel('Cross-Section (m²)')

        self.fig.tight_layout()
        self.canvas.draw()


# --- ADVANCED SETTINGS DIALOG ---
class AdvancedSettingsDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Simulation Parameters")
        self.setMinimumWidth(350)
        self.layout = QVBoxLayout(self)
        
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

        add_spin('neut_x', 'Neutralizer Axial Dist (x, mm):', 0, 100, current_params.get('neut_x', 19.9))
        add_spin('neut_r', 'Neutralizer Radius (y, mm):', 0.1, 50, current_params.get('neut_r', 3.0))
        add_spin('V_plasma_offset', 'Plasma Potential Offset (V):', 0, 500, current_params.get('V_plasma_offset', 20.0))
        add_spin('m_e_ratio', 'Electron Mass Ratio (m_Xe / X):', 1, 100000, current_params.get('m_e_ratio', 1000.0), 0, 100)
        add_spin('Lx', 'Domain Length (Lx, mm):', 5, 200, current_params.get('Lx', 20.0))
        add_spin('Ly', 'Domain Height (Ly, mm):', 1, 50, current_params.get('Ly', 3.0))
        
        self.layout.addLayout(self.form)
        
        btn_box = QHBoxLayout()
        save_btn = QPushButton("Save & Apply")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        self.layout.addLayout(btn_box)

    def get_values(self):
        return {k: v.value() for k, v in self.inputs.items()}

# --- IEDF & EEDF SUB-WINDOW CLASS ---
class IEDFWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Energy Distribution Function (IEDF & EEDF)')
        self.setGeometry(100, 100, 600, 450)
        
        layout = QVBoxLayout(self)
        
        self.combo_type = QComboBox()
        self.combo_type.addItems([
            'All Ions', 'Primary Ions Only', 'CEX Ions Only',
            'All Electrons', 'Grid Secondary Electrons (SEE) [x <= 4mm]',
            'Neutralizer Electrons (Neut) [x > 4mm]'
        ])
        layout.addWidget(QLabel("<b>Select Particle Population:</b>"))
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
        color = 'gray'
        x_max = Vs_max + 100 
        title = 'Energy Distribution'

        if 'Ions' in mode:
            self.ax.set_xlabel('Energy (eV)')
            self.ax.set_ylabel('Ion Count')
            if len(p_vx) > 0:
                v_sq = p_vx**2 + p_vy**2
                E_all = (0.5 * m_XE * v_sq) / q
                
                if mode == 'All Ions':
                    data, color, title = E_all, 'purple', 'Ion Energy Distribution (All Ions)'
                elif mode == 'Primary Ions Only':
                    data, color, title = E_all[~p_isCEX], 'blue', 'Ion Energy Distribution (Primary Beam)'
                else:
                    data, color, title = E_all[p_isCEX], 'red', 'Ion Energy Distribution (CEX Only)'
                    x_max = max(Vs_max * 0.3, np.max(data) + 50) if len(data) > 0 else 500 

        else:
            self.ax.set_xlabel('Electron Kinetic Energy (eV)')
            self.ax.set_ylabel('Electron Count')
            if len(e_vx) > 0:
                v_sq_e = e_vx**2 + e_vy**2
                E_elec = (0.5 * m_e * v_sq_e) / q
                
                if mode == 'All Electrons':
                    data, color, title = E_elec, '#2ECC71', 'Electron Energy Distribution (All)'
                elif 'SEE' in mode:
                    data, color, title = E_elec[e_x <= 4.0], '#E67E22', 'Electron Energy Distribution (Grid/SEE Zone)'
                else:
                    data, color, title = E_elec[e_x > 4.0], '#1ABC9C', 'Electron Energy Distribution (Plume/Neut Zone)'
                
                if len(data) > 0:
                    hist_max = np.percentile(data, 99) 
                    x_max = max(20.0, hist_max * 1.2) 
                else:
                    x_max = 20.0

        self.ax.set_title(title)
        if len(data) > 0:
            self.ax.hist(data, bins=50, range=(0, x_max), color=color, alpha=0.8, edgecolor='black')
            self.ax.set_xlim(0, x_max)
        self.canvas.draw()


class DigitalTwinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PY-BEMCS (Multi-Grid & Co-Extraction)')
        self.setGeometry(20, 30, 1500, 800) 

        self.sim = DigitalTwinSimulator()
        self.sim_isRunning = False

        self.iter_history = []
        self.ebs_history = []
        self.div_history = []
        self.T_histories = {}
        self.recorded_frames = []
        self.tracking_buffer = []
        self.iedf_window = None
        self.cs_viewer_window = None

        self.cbar_temp = None
        self.cbar_energy = None

        self.grid_widgets = []

        # Beam species configuration
        self.beam_mass_amu = 131.293  # Xenon default
        self.beam_charge_state = 1

        # Cross-section data store: {label: {energy, cs, spline, type}}
        self.cs_store = {}

        # Grid material configuration
        self.mat_name = 'Molybdenum'
        self.mat_props = GridMaterialDialog.PRESETS['Molybdenum'].copy()
        
        # Advanced settings defaults
        self.adv_params = {
            'neut_x': 19.9,
            'neut_r': 3.0,
            'V_plasma_offset': 20.0,
            'm_e_ratio': 1000.0,
            'Lx': 20.0,
            'Ly': 3.0
        }
        
        self.setup_menu_bar()
        self.setup_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_sim_step)
        self.timer.start(10)

    def setup_menu_bar(self):
        menubar = self.menuBar()

        # --- Settings Menu ---
        settings_menu = menubar.addMenu('Settings')
        adv_action = QAction('Advanced Parameters...', self)
        adv_action.triggered.connect(self.open_advanced_settings)
        settings_menu.addAction(adv_action)

        # --- Beam Menu ---
        beam_menu = menubar.addMenu('Beam')

        species_action = QAction('Ion Species...', self)
        species_action.triggered.connect(self.open_beam_species)
        beam_menu.addAction(species_action)

        beam_menu.addSeparator()

        cs_action = QAction('Cross-Section Manager...', self)
        cs_action.triggered.connect(self.open_cs_viewer)
        beam_menu.addAction(cs_action)

        # --- Materials Menu ---
        mat_menu = menubar.addMenu('Materials')

        mat_action = QAction('Grid Material...', self)
        mat_action.triggered.connect(self.open_grid_material)
        mat_menu.addAction(mat_action)

    def open_advanced_settings(self):
        dialog = AdvancedSettingsDialog(self.adv_params, self)
        if dialog.exec_() == QDialog.Accepted:
            self.adv_params.update(dialog.get_values())
            QMessageBox.information(self, "Settings Updated",
                                    "Settings saved. Click '1. BUILD DOMAIN' to apply geometry or mass changes.")

    def open_beam_species(self):
        dialog = BeamSpeciesDialog(self.beam_mass_amu, self.beam_charge_state, self)
        if dialog.exec_() == QDialog.Accepted:
            self.beam_mass_amu, self.beam_charge_state = dialog.get_values()
            QMessageBox.information(
                self, "Species Updated",
                f"Beam ion: {self.beam_mass_amu:.3f} amu, charge +{self.beam_charge_state}.\n"
                f"Click '1. BUILD DOMAIN' to apply.")

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
            QMessageBox.information(
                self, "Material Updated",
                f"Grid material: {self.mat_name}\n"
                f"k={self.mat_props['k']:.1f} W/m/K, "
                f"rho={self.mat_props['rho']:.0f} kg/m³, "
                f"cp={self.mat_props['cp']:.0f} J/kg/K\n"
                f"Click '1. BUILD DOMAIN' to apply.")

    def apply_advanced_settings_to_sim(self):
        """Injects advanced parameters into the sim engine before building"""
        self.sim.Lx = self.adv_params['Lx']
        self.sim.Ly = self.adv_params['Ly']
        self.sim.nx = int(self.sim.Lx / self.sim.dx) + 1
        self.sim.ny = int(self.sim.Ly / self.sim.dy) + 1

        # Apply beam species: mass and charge
        self.sim.m_ion = self.beam_mass_amu * 1.6605e-27
        self.sim.m_XE = self.sim.m_ion  # backward compat alias
        self.sim.Z_ion = self.beam_charge_state
        self.sim.q_ion = self.beam_charge_state * self.sim.q

        self.sim.m_e = self.sim.m_ion / self.adv_params['m_e_ratio']

        # Apply grid material properties
        self.sim.set_material(props=self.mat_props)

        # Pass user cross-section splines to the engine
        self.sim.user_cs = {}
        for label, ds in self.cs_store.items():
            if ds.get('spline') is not None:
                self.sim.user_cs[label] = ds

        self.sim.x_pts = np.linspace(0, self.sim.Lx, self.sim.nx)
        self.sim.y_pts = np.linspace(0, self.sim.Ly, self.sim.ny)
        self.sim.X, self.sim.Y = np.meshgrid(self.sim.x_pts, self.sim.y_pts)

        self.sim.T_map = np.full((self.sim.ny, self.sim.nx), 300.0, dtype=np.float32)
        self.sim.T_map_new = np.full((self.sim.ny, self.sim.nx), 300.0, dtype=np.float32)
        
        # We don't need to call reset_arrays here because sim.build_domain calls it natively

    def create_input(self, label_text, default_val, min_v, max_v, step, decimals=1):
        row = QHBoxLayout()
        lbl = QLabel(label_text); lbl.setFixedWidth(130)
        spin = QDoubleSpinBox(); spin.setRange(min_v, max_v); spin.setValue(default_val)
        spin.setSingleStep(step); spin.setDecimals(decimals)
        row.addWidget(lbl); row.addWidget(spin)
        return row, spin

    def add_grid_ui(self, v_def, t_def, gap_def, r_def, cham_def):
        idx = len(self.grid_widgets) + 1
        gb = QGroupBox(f"Grid {idx}")
        lay = QVBoxLayout()
        
        row1, spin_v = self.create_input('DC Voltage (V):', v_def, -5000, 15000, 100, 0)
        row2, spin_t = self.create_input('Thickness (mm):', t_def, 0.1, 10.0, 0.1)
        row3, spin_gap = self.create_input('Gap to Next (mm):', gap_def, 0.1, 10.0, 0.1)
        row4, spin_r = self.create_input('Hole Radius (mm):', r_def, 0.1, 10.0, 0.1)
        row5, spin_cham = self.create_input('Chamfer (°):', cham_def, 0, 45, 1)
        
        lay.addLayout(row1); lay.addLayout(row2); lay.addLayout(row3)
        lay.addLayout(row4); lay.addLayout(row5)
        gb.setLayout(lay)
        self.grids_layout.insertWidget(self.grids_layout.count() - 1, gb)
        
        self.grid_widgets.append({
            'gb': gb, 'V': spin_v, 't': spin_t, 'gap': spin_gap, 
            'r': spin_r, 'cham': spin_cham
        })
        self.update_rf_combo()

    def remove_grid_ui(self):
        if len(self.grid_widgets) > 1:
            gw = self.grid_widgets.pop()
            gw['gb'].deleteLater()
            self.update_rf_combo()

    def update_rf_combo(self):
        # Check if the combo box has been initialized yet before trying to update it
        if hasattr(self, 'combo_rf_grid'):
            self.combo_rf_grid.clear()
            self.combo_rf_grid.addItems([f"Grid {i+1}" for i in range(len(self.grid_widgets))])

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # SCROLLABLE CONTROL PANEL
        scroll_area = QScrollArea()
        scroll_area.setFixedWidth(350)
        scroll_area.setWidgetResizable(True)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        self.inputs = {}

        # Pre-initialize the RF Combo Box so add_grid_ui can safely update it
        self.combo_rf_grid = QComboBox()

        # 1. DYNAMIC GRIDS
        control_layout.addWidget(QLabel('<b>1. MULTI-GRID OPTICS</b>'))
        self.grids_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        btn_add = QPushButton('+ Add Grid')
        btn_rem = QPushButton('- Remove Grid')
        btn_add.clicked.connect(lambda: self.add_grid_ui(0, 1.0, 1.0, 1.0, 0))
        btn_rem.clicked.connect(self.remove_grid_ui)
        btn_layout.addWidget(btn_add); btn_layout.addWidget(btn_rem)
        
        self.grids_layout.addLayout(btn_layout)
        control_layout.addLayout(self.grids_layout)
        
        # Initialize default 2 grids
        self.add_grid_ui(1650, 1.0, 1.0, 1.0, 0)
        self.add_grid_ui(-350, 1.0, 1.0, 0.6, 0)

        control_layout.addSpacing(15)
        
        
        # 2. RF CO-EXTRACTION
        control_layout.addWidget(QLabel('<b>2. RF CO-EXTRACTION</b>'))
        self.chk_rf = QCheckBox('Enable RF Modulated Potential')
        control_layout.addWidget(self.chk_rf)
        
        rf_row = QHBoxLayout()
        rf_row.addWidget(QLabel('Apply RF to:'))
        rf_row.addWidget(self.combo_rf_grid)
        control_layout.addLayout(rf_row)
        
        # FIX: Use distinct variables for the layout rows so they don't overwrite each other
        row_freq, self.spin_rf_freq = self.create_input('Frequency (MHz):', 13.56, 0.1, 100, 0.1)
        row_amp, self.spin_rf_amp = self.create_input('Amplitude (V):', 500, 0, 5000, 50, 0)
        
        control_layout.addLayout(row_freq)
        control_layout.addLayout(row_amp)

        control_layout.addSpacing(15)

        # 3. PLASMA & SPUTTERING
        control_layout.addWidget(QLabel('<b>3. PLASMA & SPUTTERING</b>'))
        _, self.inputs['n0_plasma'] = self.create_input('Plasma Dens (m-3):', 1e17, 1e15, 1e19, 1e16, 0)
        control_layout.addLayout(_)
        _, self.inputs['Te_up'] = self.create_input('Upstream Te (eV):', 3.0, 0.1, 20.0, 0.5, 1)
        control_layout.addLayout(_)
        _, self.inputs['Ti'] = self.create_input('Ion Temp (eV):', 2.0, 0.1, 10, 0.5)
        control_layout.addLayout(_)
        _, self.inputs['Tn'] = self.create_input('Neutral Temp (K):', 300, 100, 2000, 100, 0)
        control_layout.addLayout(_)
        _, self.inputs['n0'] = self.create_input('Neutral Dens (m-3):', 1e20, 1e18, 1e22, 1e19, 0)
        control_layout.addLayout(_)
        _, self.inputs['Accel'] = self.create_input('Accel. Factor (X):', 1, 10, 1e16, 1e12, 0)
        control_layout.addLayout(_)
        _, self.inputs['Thresh'] = self.create_input('Cell Fail Thresh:', 10000.0, 0.1, 100000.0, 0.1)
        control_layout.addLayout(_)

        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel('<b>4. SIMULATION MODE</b>'))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(['Both', 'Thermal', 'Erosion'])
        control_layout.addWidget(self.combo_mode)
        
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel('<b>5. NEUTRALIZER</b>'))
        _, self.inputs['neut_rate'] = self.create_input('e- Inject Rate:', 0, 0, 1e6, 10, 0)
        control_layout.addLayout(_)
        _, self.inputs['Te'] = self.create_input('e- Temp (eV):', 5.0, 0.1, 20.0, 0.5, 1)
        control_layout.addLayout(_)
        
        control_layout.addSpacing(15)

        # ACTION BUTTONS
        self.btn_build = QPushButton('1. BUILD DOMAIN')
        self.btn_build.clicked.connect(self.build_domain)
        self.btn_toggle = QPushButton('2. START BEAM')
        self.btn_toggle.clicked.connect(self.toggle_sim)
        
        self.btn_csv = QPushButton('Export Data (.csv)')
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_csv.setStyleSheet("background-color: #E6E6FA; font-weight: bold;")
        
        self.chk_track_ptcls = QCheckBox('Record Kinematics')
        self.btn_export_trk = QPushButton('Export Particle Data (.csv)')
        self.btn_export_trk.clicked.connect(self.export_tracking_data)
        self.btn_export_trk.setStyleSheet("background-color: #FFDAB9; font-weight: bold;")
        
        self.btn_iedf = QPushButton('Show Energy Dist. (IEDF/EEDF)')
        self.btn_iedf.clicked.connect(self.open_iedf_window)
        self.btn_iedf.setStyleSheet("background-color: #D1F2EB; font-weight: bold;")
        
        self.chk_record = QCheckBox('Record Frames (0)')
        self.btn_save = QPushButton('Save GIF Animation')
        self.btn_save.clicked.connect(self.save_gif)
        
        self.lbl_status = QLabel('Status: Ready.')
        self.lbl_temp = QLabel('Grid Temps: Ready')
        self.lbl_temp.setStyleSheet("color: #D35400; font-weight: bold;") 

        for w in [self.btn_build, self.btn_toggle, self.btn_csv, self.chk_track_ptcls, 
                  self.btn_export_trk, self.btn_iedf, self.chk_record, self.btn_save, 
                  self.lbl_status, self.lbl_temp]:
            control_layout.addWidget(w)
            
        scroll_area.setWidget(control_panel)
        main_layout.addWidget(scroll_area)

        # PLOTTING
        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        grid = plt.GridSpec(2, 3, height_ratios=[1.2, 1])
        self.ax_live = self.fig.add_subplot(grid[0, 0:2]); self.ax_live.set_title('Beam trajectory')
        self.ax_temp = self.fig.add_subplot(grid[0, 2]); self.ax_temp.set_title('Grid Temp Map (°C)')
        self.ax_dmg = self.fig.add_subplot(grid[1, 0]); self.ax_dmg.set_title('Damage Map')
        self.ax_ebs = self.fig.add_subplot(grid[1, 1]); self.ax_ebs.set_title('Centerline Min Potential (V)')
        self.ax_div = self.fig.add_subplot(grid[1, 2]); self.ax_div.set_title('Beam Divergence')
        
        self.line_ebs, = self.ax_ebs.plot([], [], 'm-', lw=2)
        self.line_div, = self.ax_div.plot([], [], 'b-', lw=2)
        
        self.scat_prim = None
        self.scat_cex = None
        self.scat_elec = self.ax_live.scatter([], [], s=1, c='#00FF00', alpha=0.5) 
        self.fig.tight_layout()

    def get_params(self):
        params = {k: v.value() for k, v in self.inputs.items()}
        
        # Inject the advanced dialog settings directly into the params dictionary 
        # so the physics engine can receive them without modification
        params.update(self.adv_params)
        
        params['sim_mode'] = self.combo_mode.currentText()
        params['rf_enable'] = self.chk_rf.isChecked()
        params['rf_grid_idx'] = self.combo_rf_grid.currentIndex()
        params['rf_freq'] = self.spin_rf_freq.value()
        params['rf_amp'] = self.spin_rf_amp.value()
        
        grids = []
        for gw in self.grid_widgets:
            grids.append({
                'V': gw['V'].value(),
                't': gw['t'].value(),
                'gap': gw['gap'].value(),
                'r': gw['r'].value(),
                'cham': gw['cham'].value()
            })
        params['grids'] = grids
        return params

    def toggle_sim(self):
        if not np.any(self.sim.Ex):
            QMessageBox.warning(self, "Warning", "Build Domain first!")
            return
        self.sim_isRunning = not self.sim_isRunning
        self.btn_toggle.setText('PAUSE BEAM' if self.sim_isRunning else 'RESUME BEAM')

    def open_iedf_window(self):
        if self.iedf_window is None:
            self.iedf_window = IEDFWindow()
        self.iedf_window.show()

    def build_domain(self):
        self.sim_isRunning = False
        self.btn_toggle.setText('2. START BEAM')
        self.iter_history.clear(); self.ebs_history.clear(); self.div_history.clear()
        
        self.T_histories = {i: [] for i in range(len(self.grid_widgets))}
        self.tracking_buffer.clear() 
        
        self.lbl_status.setText('Building Multi-Grid Domain...')
        QApplication.processEvents() 
        
        # Apply the advanced properties BEFORE generating the domain mesh
        self.apply_advanced_settings_to_sim()
        
        self.sim.build_domain(self.get_params())
        self.draw_static_domain()
        
        species_str = f'{self.beam_mass_amu:.1f} amu, +{self.beam_charge_state}'
        cs_count = sum(1 for ds in self.cs_store.values() if ds.get('spline') is not None)
        cs_str = f' | {cs_count} CS loaded' if cs_count > 0 else ''
        self.lbl_status.setText(f'Domain Ready. Wt: {self.sim.macro_weight:.1e} | {species_str} | {self.mat_name}{cs_str}')
        self.lbl_temp.setText('Grid Temps: ' + ' | '.join([f'G{i+1}: 26°C' for i in range(len(self.grid_widgets))]))

    def draw_static_domain(self):
        self.ax_live.clear()
        self.ax_live.contourf(self.sim.X, self.sim.Y, self.sim.V, 20, cmap='viridis', alpha=0.4) 
        
        gy, gx = np.where(self.sim.isBound)
        self.ax_live.scatter(gx * self.sim.dx, gy * self.sim.dy, s=12, c='k', alpha=0.8)
        
        max_v = max([g['V'].value() for g in self.grid_widgets]) if self.grid_widgets else 1000
        
        self.scat_prim = self.ax_live.scatter([], [], c=[], s=2, cmap='turbo', vmin=0, vmax=max_v+50, alpha=0.8)
        self.scat_cex = self.ax_live.scatter([], [], c=[], s=7, cmap='turbo', vmin=0, vmax=max_v+50, alpha=1.0)
        self.scat_elec = self.ax_live.scatter([], [], s=1, c='#00FF00', alpha=0.5)
        
        if not hasattr(self, 'cax_live'):
            divider = make_axes_locatable(self.ax_live)
            self.cax_live = divider.append_axes("right", size="3%", pad=0.1) 
        else:
            self.cax_live.clear()
            
        self.cbar_energy = self.fig.colorbar(self.scat_prim, cax=self.cax_live)
        self.cbar_energy.set_label('Kinetic Energy (eV)')
        
        self.ax_live.set_xlim(0, self.sim.Lx)
        self.ax_live.set_ylim(0, self.sim.Ly)
        self.ax_live.set_title('Beam Extraction & Tracking')
        self.canvas.draw()

    def run_sim_step(self):
        if not self.sim_isRunning: return

        remeshed, min_pot, current_div, T_grids = self.sim.step(self.get_params())
        
        if remeshed:
            self.lbl_status.setText('Domain Remeshed (Thermal or Erosion)!')
            self.draw_static_domain()

        if self.sim.iteration % 5 == 0:
            prim_mask = ~self.sim.p_isCEX
            cex_mask = self.sim.p_isCEX
            
            self.scat_prim.set_offsets(np.column_stack((self.sim.p_x[prim_mask], self.sim.p_y[prim_mask])) if np.any(prim_mask) else np.empty((0, 2)))
            self.scat_cex.set_offsets(np.column_stack((self.sim.p_x[cex_mask], self.sim.p_y[cex_mask])) if np.any(cex_mask) else np.empty((0, 2)))
            self.scat_elec.set_offsets(np.column_stack((self.sim.e_x, self.sim.e_y)) if len(self.sim.e_x) > 0 else np.empty((0, 2)))

            if np.any(prim_mask):
                v_sq_prim = self.sim.p_vx[prim_mask]**2 + self.sim.p_vy[prim_mask]**2
                self.scat_prim.set_array((0.5 * self.sim.m_ion * v_sq_prim) / self.sim.q)

            if np.any(cex_mask):
                v_sq_cex = self.sim.p_vx[cex_mask]**2 + self.sim.p_vy[cex_mask]**2
                self.scat_cex.set_array((0.5 * self.sim.m_ion * v_sq_cex) / self.sim.q)

            if self.iedf_window and self.iedf_window.isVisible():
                max_v = max([g['V'].value() for g in self.grid_widgets]) if self.grid_widgets else 1000
                self.iedf_window.update_histogram(
                    self.sim.p_vx, self.sim.p_vy, self.sim.p_isCEX,
                    self.sim.e_x, self.sim.e_vx, self.sim.e_vy,
                    self.sim.m_ion, self.sim.m_e, self.sim.q, max_v
                )

            if self.chk_track_ptcls.isChecked():
                ptcl_data = self.sim.get_particle_kinematics()
                if ptcl_data.size > 0: self.tracking_buffer.append(ptcl_data)

            self.iter_history.append(self.sim.iteration)
            self.ebs_history.append(min_pot)
            self.div_history.append(current_div)
            for i, T in enumerate(T_grids):
                self.T_histories[i].append(T)

            self.line_ebs.set_data(self.iter_history, self.ebs_history)
            self.line_div.set_data(self.iter_history, self.div_history)
            
            self.ax_ebs.set_xlim(max(0, self.sim.iteration - 400), max(100, self.sim.iteration))
            self.ax_ebs.set_ylim(min(-30, min(self.ebs_history)), max(10, max(self.ebs_history)))
            self.ax_div.set_xlim(max(0, self.sim.iteration - 400), max(100, self.sim.iteration))
            self.ax_div.set_ylim(0, 45)

            self.ax_temp.clear()
            self.ax_temp.set_title('Grid Temp Map (°C)')
            self.ax_temp.set_facecolor('black')
            
            if np.any(self.sim.isBound):
                T_display_C = np.copy(self.sim.T_map) - 273.15
                T_display_C[~self.sim.isBound] = np.nan 
                
                contour = self.ax_temp.contourf(self.sim.X, self.sim.Y, T_display_C, 15, cmap='inferno')
                
                if not hasattr(self, 'cax_temp'):
                    divider_t = make_axes_locatable(self.ax_temp)
                    self.cax_temp = divider_t.append_axes("right", size="5%", pad=0.1)
                else:
                    self.cax_temp.clear()
                    
                self.cbar_temp = self.fig.colorbar(contour, cax=self.cax_temp)
                self.cbar_temp.set_label('Temperature (°C)')
                
                total_len = 1.0 + sum([g['t'].value() + g['gap'].value() for g in self.grid_widgets])
                self.ax_temp.set_xlim(0.8, total_len + 0.2)

            self.ax_dmg.clear()
            self.ax_dmg.set_title('Damage Map')
            self.ax_dmg.contourf(self.sim.X, self.sim.Y, self.sim.damage_map, 15, cmap='hot')
            gy, gx = np.where(self.sim.isBound)
            self.ax_dmg.scatter(gx * self.sim.dx, gy * self.sim.dy, s=2, c='grey', alpha=0.5)
            self.ax_dmg.set_xlim(0, self.sim.Lx)
            self.ax_dmg.set_ylim(0, self.sim.Ly)

            self.lbl_status.setText(f'Ions: {len(self.sim.p_x)} | e-: {len(self.sim.e_x)} | Iter: {self.sim.iteration}')
            t_str = ' | '.join([f'G{i+1}: {int(T-273.15)}°C' for i, T in enumerate(T_grids)])
            self.lbl_temp.setText('Grid Temps: ' + t_str)
            self.canvas.draw()

            if self.chk_record.isChecked():
                self.recorded_frames.append(self.canvas.grab().toImage())
                self.chk_record.setText(f'Record Frames ({len(self.recorded_frames)})')

    def export_csv(self):
        if not self.iter_history:
            QMessageBox.warning(self, "Error", "No telemetry data to export! Run the simulation first.")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Telemetry", "", "CSV Files (*.csv)")
        if file_name:
            try:
                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    
                    headers = ['Iteration', 'Min_Potential_V', 'Divergence_Deg']
                    headers.extend([f'Grid_{i+1}_Temp_C' for i in range(len(self.T_histories))])
                    writer.writerow(headers)
                    
                    for i in range(len(self.iter_history)):
                        row = [
                            self.iter_history[i], 
                            round(self.ebs_history[i], 2), 
                            round(self.div_history[i], 2) if not np.isnan(self.div_history[i]) else ""
                        ]
                        for g in range(len(self.T_histories)):
                            row.append(round(self.T_histories[g][i] - 273.15, 1))
                        writer.writerow(row)
                        
                QMessageBox.information(self, "Success", f"Telemetry Exported to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to write file:\n{str(e)}")

    def export_tracking_data(self):
        if not self.tracking_buffer:
            QMessageBox.warning(self, "Error", "No tracking data recorded! Check 'Record Kinematics' and run beam.")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Particle Kinematics", "", "CSV Files (*.csv)")
        if file_name:
            try:
                self.lbl_status.setText('Exporting huge data file. Please wait...')
                QApplication.processEvents() 
                
                master_array = np.vstack(self.tracking_buffer)
                header_str = "Time_s,Z_mm,R_mm,Vz_m_s,Vr_m_s,Energy_eV,Particle_Type_ID"
                np.savetxt(file_name, master_array, delimiter=",", header=header_str, comments='', fmt='%.5e')
                
                self.lbl_status.setText(f'Export Complete! ({len(master_array)} rows)')
                QMessageBox.information(self, "Success", f"Exported {len(master_array)} particle states to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to write file:\n{str(e)}")

    def save_gif(self):
        if not self.recorded_frames:
            QMessageBox.warning(self, "Error", "No frames recorded!")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Animation", "", "GIF Files (*.gif)")
        if file_name:
            try:
                from PIL import Image
                pil_images = []
                for qimg in self.recorded_frames:
                    qimg = qimg.convertToFormat(4) 
                    ptr = qimg.bits(); ptr.setsize(qimg.byteCount())
                    arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)
                    pil_images.append(Image.fromarray(arr[..., [2, 1, 0]], 'RGB'))
                
                pil_images[0].save(file_name, save_all=True, append_images=pil_images[1:], duration=50, loop=0)
                self.recorded_frames.clear(); self.chk_record.setChecked(False)
                QMessageBox.information(self, "Success", "GIF Saved Successfully!")
            except ImportError:
                QMessageBox.warning(self, "Error", "Install Pillow (`pip install Pillow`)")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion') 
    window = DigitalTwinApp()
    window.show()
    sys.exit(app.exec_())