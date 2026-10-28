"""
Modal configuration dialogs for PY-BEMCS.
Includes Beam Species, Grid Material, Advanced Physics/Mesh, and GUI Settings dialogs.
"""
import math
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton,
    QMessageBox, QScrollArea, QWidget, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog
)
from .workers.simulation_worker import SimulationWorker


class BeamSpeciesDialog(QDialog):
    """Dialog for selecting preset ion beam species (Xe, Kr, Ar, etc.) or defining custom mass and charge state."""
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


class GridMaterialDialog(QDialog):
    """Dialog for configuring grid material properties (thermal, mechanical, and sputtering) from presets or custom values."""
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


class AdvancedSettingsDialog(QDialog):
    """Dialog for tuning advanced physical and numerical parameters (plasma offset, mass ratio, mesh domains)."""
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

        add_spin("V_plasma_offset", "Plasma Potential Offset (V):", 0, 500, current_params.get("V_plasma_offset", 20.0))
        add_spin("m_e_ratio", "Electron Mass Ratio (m_Xe / X):", 1, 100000, current_params.get("m_e_ratio", 1000.0), 0, 100)

        val_dx_debye = current_params.get(r"\deltax/debye_length", current_params.get("dx_over_debye", 0.8))
        spin_dx = QDoubleSpinBox()
        spin_dx.setRange(0.01, 10.0)
        spin_dx.setDecimals(3)
        spin_dx.setSingleStep(0.05)
        spin_dx.setValue(float(val_dx_debye))
        spin_dx.setToolTip(
            "Ratio between spatial grid cell size (Δx, Δy) and electron Debye length (λ_D).\n"
            "Default: 0.8 (Δx = 0.8 · λ_D). Smaller values provide finer spatial resolution."
        )
        self.form.addRow(r"\deltax/debye_length =", spin_dx)
        self.inputs["dx_over_debye"] = spin_dx

        val_dt_ns = current_params.get("dt_ns", current_params.get("dt", 5e-10) * 1e9)
        add_spin("dt_ns", "Time Step (dt, ns):", 0.001, 100.0, float(val_dt_ns), decimals=3, step=0.05)

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

        # —- Use .json Mesh Zones Configuration option —-
        self.chk_json_zones = QCheckBox("Use .json Mesh Zones Configuration")
        self.chk_json_zones.setChecked(bool(current_params.get("use_json_mesh_zones", False)))
        self.chk_json_zones.setToolTip(
            "When checked: uses the fixed 3-zone coarsening factors specified in the .json file\n"
            "  (presheath_factor, optics_factor, plume_factor).\n"
            "When unchecked (default): uses the automated multi-zone estimator that physically partitions\n"
            "  the domain across apertures, gaps, and plume with adaptive coarsening factors."
        )
        layout.addWidget(self.chk_json_zones)

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
        result["use_json_mesh_zones"] = self.chk_json_zones.isChecked()
        if "dx_over_debye" in result:
            result[r"\deltax/debye_length"] = result["dx_over_debye"]
            result["deltax_debye_length"] = result["dx_over_debye"]
        if "dt_ns" in result:
            result["dt"] = result["dt_ns"] * 1e-9
        return result


class GUISettingsDialog(QDialog):
    """Dialog for configuring GUI rendering and performance parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GUI Settings")
        self.setMinimumWidth(420)
        self._app = parent  # DigitalTwinApp reference

        layout = QVBoxLayout(self)

        # ── Scatter Particle Limit ────────────────────────────────────────────
        grp_scatter = QGroupBox("Scatter Plot — Particle Visualisation Limit")
        grp_scatter.setToolTip(
            "Limits how many ion/electron positions are drawn on the main canvas "
            "each simulation step.  Reducing this number can significantly improve "
            "GUI responsiveness when particle counts are high."
        )
        scatter_layout = QVBoxLayout(grp_scatter)

        self._scatter_group = QButtonGroup(self)

        self._rb_none   = QRadioButton("None  — hide all particles (best performance)")
        self._rb_all    = QRadioButton("All   — show every particle (may freeze GUI at high N)")
        self._rb_custom = QRadioButton("Custom limit:")

        for rb in (self._rb_none, self._rb_all, self._rb_custom):
            scatter_layout.addWidget(rb)
            self._scatter_group.addButton(rb)

        # Custom spinner row
        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(22, 0, 0, 0)
        self._spin_custom = QSpinBox()
        self._spin_custom.setRange(1, 10_000_000)
        self._spin_custom.setSingleStep(10_000)
        self._spin_custom.setValue(100_000)
        self._spin_custom.setSuffix("  particles")
        self._spin_custom.setToolTip("Maximum particles shown per species (ions / electrons)")
        custom_row.addWidget(self._spin_custom)
        custom_row.addStretch()
        scatter_layout.addLayout(custom_row)

        # Enable / disable spinner based on selection
        self._rb_custom.toggled.connect(self._spin_custom.setEnabled)
        self._spin_custom.setEnabled(False)

        layout.addWidget(grp_scatter)

        # ── Diagnostic Window Refresh Rate ───────────────────────────────────
        grp_diag = QGroupBox("Diagnostic Windows — Refresh Interval")
        diag_layout = QFormLayout(grp_diag)

        self._spin_diag_ms = QSpinBox()
        self._spin_diag_ms.setRange(100, 10_000)
        self._spin_diag_ms.setSingleStep(100)
        self._spin_diag_ms.setSuffix(" ms")
        self._spin_diag_ms.setToolTip(
            "Minimum time between redraws of the PPC, Energy, Charge and Performance "
            "diagnostic windows.  Lower values give smoother updates but cost more CPU "
            "on the GUI thread.  Default: 500 ms (~2 fps)."
        )
        diag_layout.addRow("Minimum interval:", self._spin_diag_ms)
        layout.addWidget(grp_diag)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Apply")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

        # ── Populate current values ───────────────────────────────────────────
        self._load_current()

    def _load_current(self):
        """Read current values from the app and pre-fill the dialog."""
        app = self._app
        if app is None:
            self._rb_custom.setChecked(True)
            self._spin_diag_ms.setValue(500)
            return

        # Scatter limit
        lim = getattr(app._sim_worker, '_scatter_limit', SimulationWorker.MAX_SCATTER)
        if lim is None:
            self._rb_none.setChecked(True)
        elif lim == 0:
            self._rb_all.setChecked(True)
        else:
            self._rb_custom.setChecked(True)
            self._spin_custom.setValue(int(lim))
        self._spin_custom.setEnabled(self._rb_custom.isChecked())

        # Diagnostic interval
        diag_ms = int(getattr(app, '_DIAG_MIN_INTERVAL_MS', 500))
        self._spin_diag_ms.setValue(diag_ms)

    def get_scatter_limit(self):
        """Return the chosen scatter limit: None, 0 (all), or a positive int."""
        if self._rb_none.isChecked():
            return None
        if self._rb_all.isChecked():
            return 0
        return int(self._spin_custom.value())

    def get_diag_interval_ms(self):
        """Return the chosen diagnostic refresh interval in milliseconds."""
        return int(self._spin_diag_ms.value())


class DivergencePlaneDialog(QDialog):
    """Dialog for setting the axial plane position where beam divergence is evaluated."""
    def __init__(self, current_x, x_min, x_max, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Divergence Evaluation Plane")
        self.setMinimumWidth(380)

        self.x_min = float(x_min)
        self.x_max = float(x_max)

        # Ensure current_x is within bounds
        if current_x is None or not (self.x_min <= current_x <= self.x_max):
            current_x = self.x_min + 0.5 * (self.x_max - self.x_min)
        self.current_x = float(current_x)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        info_lbl = QLabel(
            "Configure the axial position (x) downstream of the grid system\n"
            "where beam divergence is evaluated.\n\n"
            f"• Last Grid Exit:  x = {self.x_min:.3f} mm\n"
            f"• Domain Boundary: x = {self.x_max:.3f} mm"
        )
        info_lbl.setStyleSheet("color: #495057; font-size: 11px; margin-bottom: 6px;")
        layout.addWidget(info_lbl)

        form = QFormLayout()
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setDecimals(3)
        self.spin_x.setSingleStep(0.05)
        # Strict control on range
        self.spin_x.setRange(self.x_min, self.x_max)
        self.spin_x.setValue(self.current_x)
        self.spin_x.setSuffix(" mm")
        self.spin_x.setToolTip(
            f"Plane position in mm. Must be between {self.x_min:.3f} mm and {self.x_max:.3f} mm."
        )
        form.addRow("Evaluation Plane (x):", self.spin_x)
        layout.addLayout(form)

        # Preset shortcuts
        btn_layout = QHBoxLayout()
        btn_grid_exit = QPushButton("Last Grid Exit")
        btn_grid_exit.setToolTip("Set to exit face of last grid")
        btn_grid_exit.clicked.connect(lambda: self.spin_x.setValue(self.x_min))

        btn_mid = QPushButton("Mid-Plume")
        btn_mid.setToolTip("Set to halfway between last grid and boundary")
        mid_val = round(self.x_min + 0.5 * (self.x_max - self.x_min), 3)
        btn_mid.clicked.connect(lambda: self.spin_x.setValue(mid_val))

        btn_boundary = QPushButton("End Boundary")
        btn_boundary.setToolTip("Set to domain exit boundary")
        btn_boundary.clicked.connect(lambda: self.spin_x.setValue(self.x_max))

        btn_layout.addWidget(btn_grid_exit)
        btn_layout.addWidget(btn_mid)
        btn_layout.addWidget(btn_boundary)
        layout.addLayout(btn_layout)

        layout.addSpacing(10)

        btn_box = QHBoxLayout()
        ok_btn = QPushButton("Apply")
        ok_btn.setStyleSheet("font-weight: bold;")
        ok_btn.clicked.connect(self.validate_and_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

    def validate_and_accept(self):
        val = self.spin_x.value()
        if val < self.x_min or val > self.x_max:
            QMessageBox.warning(
                self,
                "Invalid Plane Position",
                f"Selected plane (x = {val:.3f} mm) is outside the valid domain range.\n\n"
                f"It must be between the last grid (x = {self.x_min:.3f} mm) "
                f"and the domain boundary (x = {self.x_max:.3f} mm)."
            )
            return
        self.accept()

    def get_plane_x(self):
        return round(self.spin_x.value(), 3)


EXPORT_PARAMETER_DEFS = [
    {
        "id": "iteration",
        "label": "Iteration Number",
        "header": "iteration",
        "tooltip": "Simulation iteration index",
        "default": True,
        "getter": lambda app, j: app.iter_history[j] if j < len(app.iter_history) else '',
    },
    {
        "id": "t_sim_s",
        "label": "Simulation Time (t_sim_s) [s]",
        "header": "t_sim_s",
        "tooltip": "Physical plasma simulation time in seconds",
        "default": True,
        "getter": lambda app, j: app.time_history[j] if j < len(app.time_history) else '',
    },
    {
        "id": "minpotential",
        "label": "Saddle-Point Potential (minpotential) [V]",
        "header": "minpotential",
        "tooltip": "Centerline minimum / saddle-point potential in volts",
        "default": True,
        "getter": lambda app, j: app.ebs_history[j] if j < len(app.ebs_history) else '',
    },
    {
        "id": "beamdivergence",
        "label": "Beam Divergence (Grid Exit) [°]",
        "header": "beamdivergence°",
        "tooltip": "Beam divergence angle measured at the exit face of the accelerator grid",
        "default": True,
        "getter": lambda app, j: app.div_history[j] if j < len(app.div_history) else '',
    },
    {
        "id": "beamdivergence_mid",
        "label": "Beam Divergence (Mid-Plume) [°]",
        "header": "beamdivergence_mid_plume°",
        "tooltip": "Beam divergence angle measured at mid-plume location",
        "default": True,
        "getter": lambda app, j: app.div_mid_history[j] if j < len(app.div_mid_history) else '',
    },
    {
        "id": "transparency",
        "label": "Grid Transparency",
        "header": "transparency",
        "tooltip": "Grid optical / ion particle transmission efficiency",
        "default": True,
        "getter": lambda app, j: app.transparency_history[j] if j < len(app.transparency_history) else '',
    },
    {
        "id": "ion_current_exit",
        "label": "Exit Ion Current (Instantaneous) [A]",
        "header": "ion_current_exit_A",
        "tooltip": "Instantaneous ion beam current crossing downstream boundary in Amperes",
        "default": True,
        "getter": lambda app, j: app.ion_current_exit_history[j] if j < len(app.ion_current_exit_history) else '',
    },
    {
        "id": "ion_current_exit_avg",
        "label": "Exit Ion Current (Running Avg) [A]",
        "header": "ion_current_exit_avg_A",
        "tooltip": "Time-averaged exit ion current in Amperes",
        "default": True,
        "getter": lambda app, j: app.ion_current_exit_avg_history[j] if j < len(app.ion_current_exit_avg_history) else '',
    },
    {
        "id": "transmitted_ions",
        "label": "Transmitted Ions per Step",
        "header": "transmitted_ions_step",
        "tooltip": "Number of macroparticles crossing the exit plane in current iteration",
        "default": True,
        "getter": lambda app, j: app.transmitted_ions_history[j] if j < len(app.transmitted_ions_history) else '',
    },
    {
        "id": "total_active_cells",
        "label": "Total Active PIC Cells",
        "header": "total_active_cells",
        "tooltip": "Number of grid cells containing active particles",
        "default": True,
        "getter": lambda app, j: app.active_cells_history[j] if j < len(app.active_cells_history) else '',
    },
    {
        "id": "low_ppc_cells",
        "label": "Low-PPC Cells (< threshold)",
        "header_fn": lambda app: f'cells_less_than_{getattr(app.sim, "min_ppc_threshold", 3)}_macroparticles',
        "tooltip": "Count of cells with particle count below statistical threshold",
        "default": True,
        "getter": lambda app, j: app.low_ppc_cells_history[j] if j < len(app.low_ppc_cells_history) else '',
    },
    {
        "id": "grid_temps",
        "label": "Grid Temperatures [K]",
        "is_multi": True,
        "tooltip": "Temperatures of individual grid electrodes in Kelvin",
        "default": True,
    },
]


class ExportParametersDialog(QDialog):
    """
    Dialog allowing the user to select which simulation parameters to export
    into the telemetry CSV / TXT file via an interactive checklist.
    """
    def __init__(self, current_selection=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Parameters")
        self.setMinimumWidth(380)
        self.setMinimumHeight(440)

        self._current_selection = dict(current_selection) if current_selection else {}
        self._checkboxes = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header description
        info_lbl = QLabel("Select data parameters to include in the exported (.csv / .txt) file:")
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("font-weight: bold; color: #212529;")
        layout.addWidget(info_lbl)

        # Quick action buttons row
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(self._deselect_all)
        btn_reset = QPushButton("Reset Defaults")
        btn_reset.clicked.connect(self._reset_defaults)

        for b in [btn_all, btn_none, btn_reset]:
            b.setStyleSheet("font-size: 11px; padding: 2px 6px;")
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Checklist inside a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #ced4da; border-radius: 4px; background: #ffffff; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(6)
        scroll_layout.setContentsMargins(10, 8, 10, 8)

        for pdef in EXPORT_PARAMETER_DEFS:
            pid = pdef["id"]
            chk = QCheckBox(pdef["label"])
            chk.setToolTip(pdef.get("tooltip", ""))
            is_checked = self._current_selection.get(pid, pdef.get("default", True))
            chk.setChecked(bool(is_checked))
            chk.setStyleSheet("QCheckBox { font-size: 12px; } QCheckBox:hover { color: #0d6efd; }")
            self._checkboxes[pid] = chk
            scroll_layout.addWidget(chk)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)

        # Bottom buttons (Apply / Cancel)
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("Apply")
        ok_btn.setStyleSheet("font-weight: bold; min-width: 75px;")
        ok_btn.clicked.connect(self.validate_and_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("min-width: 75px;")
        cancel_btn.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

    def _select_all(self):
        for chk in self._checkboxes.values():
            chk.setChecked(True)

    def _deselect_all(self):
        for chk in self._checkboxes.values():
            chk.setChecked(False)

    def _reset_defaults(self):
        for pdef in EXPORT_PARAMETER_DEFS:
            pid = pdef["id"]
            if pid in self._checkboxes:
                self._checkboxes[pid].setChecked(pdef.get("default", True))

    def validate_and_accept(self):
        selected_any = any(chk.isChecked() for chk in self._checkboxes.values())
        if not selected_any:
            QMessageBox.warning(
                self,
                "No Parameters Selected",
                "Please select at least one parameter to export."
            )
            return
        self.accept()

    def get_selected_params(self):
        return {pid: chk.isChecked() for pid, chk in self._checkboxes.items()}


EXPORT_PARTICLE_PARAMETER_DEFS = [
    {
        "id": "time_s",
        "label": "Time (time_s) [s]",
        "header": "time_s",
        "index_8": 0,
        "index_6": None,
        "tooltip": "Simulation physical timestamp in seconds",
        "default": True,
    },
    {
        "id": "x_mm",
        "label": "Axial Position (x_mm) [mm]",
        "header": "x_mm",
        "index_8": 1,
        "index_6": 0,
        "tooltip": "Axial position along thruster extraction axis in mm",
        "default": True,
    },
    {
        "id": "y_mm",
        "label": "Radial Position (y_mm) [mm]",
        "header": "y_mm",
        "index_8": 2,
        "index_6": 1,
        "tooltip": "Radial position in mm",
        "default": True,
    },
    {
        "id": "vx_ms",
        "label": "Axial Velocity (vx_ms) [m/s]",
        "header": "vx_ms",
        "index_8": 3,
        "index_6": 2,
        "tooltip": "Velocity along the axial direction in m/s",
        "default": True,
    },
    {
        "id": "vy_ms",
        "label": "Radial Velocity (vy_ms) [m/s]",
        "header": "vy_ms",
        "index_8": 4,
        "index_6": 3,
        "tooltip": "Velocity along the radial direction in m/s",
        "default": True,
    },
    {
        "id": "vz_ms",
        "label": "Azimuthal Velocity (vz_ms) [m/s]",
        "header": "vz_ms",
        "index_8": 5,
        "index_6": 4,
        "tooltip": "Velocity along the azimuthal direction in m/s",
        "default": True,
    },
    {
        "id": "energy_eV",
        "label": "Kinetic Energy (energy_eV) [eV]",
        "header": "energy_eV",
        "index_8": 6,
        "index_6": None,
        "tooltip": "Kinetic energy of the particle in electron-volts",
        "default": True,
    },
    {
        "id": "type",
        "label": "Particle Type / Species (type)",
        "header": "type",
        "index_8": 7,
        "index_6": 5,
        "tooltip": "Particle classification: 0: Primary Ion, 1: CEX Ion, 2/3: Electron",
        "default": True,
    },
]


class ExportParticleParametersDialog(QDialog):
    """
    Dialog allowing the user to select which particle kinematic parameters to export
    into the particle CSV / TXT file via an interactive checklist.
    """
    def __init__(self, current_selection=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Particle Parameters")
        self.setMinimumWidth(380)
        self.setMinimumHeight(400)

        self._current_selection = dict(current_selection) if current_selection else {}
        self._checkboxes = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info_lbl = QLabel("Select particle data parameters to include in the exported (.csv / .txt) file:")
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("font-weight: bold; color: #212529;")
        layout.addWidget(info_lbl)

        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(self._deselect_all)
        btn_reset = QPushButton("Reset Defaults")
        btn_reset.clicked.connect(self._reset_defaults)

        for b in [btn_all, btn_none, btn_reset]:
            b.setStyleSheet("font-size: 11px; padding: 2px 6px;")
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #ced4da; border-radius: 4px; background: #ffffff; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(6)
        scroll_layout.setContentsMargins(10, 8, 10, 8)

        for pdef in EXPORT_PARTICLE_PARAMETER_DEFS:
            pid = pdef["id"]
            chk = QCheckBox(pdef["label"])
            chk.setToolTip(pdef.get("tooltip", ""))
            is_checked = self._current_selection.get(pid, pdef.get("default", True))
            chk.setChecked(bool(is_checked))
            chk.setStyleSheet("QCheckBox { font-size: 12px; } QCheckBox:hover { color: #0d6efd; }")
            self._checkboxes[pid] = chk
            scroll_layout.addWidget(chk)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)

        btn_box = QHBoxLayout()
        ok_btn = QPushButton("Apply")
        ok_btn.setStyleSheet("font-weight: bold; min-width: 75px;")
        ok_btn.clicked.connect(self.validate_and_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("min-width: 75px;")
        cancel_btn.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

    def _select_all(self):
        for chk in self._checkboxes.values():
            chk.setChecked(True)

    def _deselect_all(self):
        for chk in self._checkboxes.values():
            chk.setChecked(False)

    def _reset_defaults(self):
        for pdef in EXPORT_PARTICLE_PARAMETER_DEFS:
            pid = pdef["id"]
            if pid in self._checkboxes:
                self._checkboxes[pid].setChecked(pdef.get("default", True))

    def validate_and_accept(self):
        selected_any = any(chk.isChecked() for chk in self._checkboxes.values())
        if not selected_any:
            QMessageBox.warning(
                self,
                "No Parameters Selected",
                "Please select at least one parameter to export."
            )
            return
        self.accept()

    def get_selected_params(self):
        return {pid: chk.isChecked() for pid, chk in self._checkboxes.items()}


class UpstreamGapDialog(QDialog):
    """
    Dialog allowing the user to view and modify the ratio used to calculate
    the upstream gap, or define a custom gap value directly in millimeters.
    """

    def __init__(self, current_gap, screen_r, n0, Te_up, is_bulk=False, saved_settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upstream Gap Settings")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(500)

        self.screen_r = max(float(screen_r), 0.01)
        self.n0 = max(float(n0), 1e12)
        self.Te_up = max(float(Te_up), 0.01)
        self.is_bulk = bool(is_bulk)
        self.current_gap = float(current_gap)

        # Compute physics Debye length in mm: λ_D = sqrt(eps0 * Te_up / (q * n0 * 0.61)) * 1e3
        eps0 = 8.854e-12
        q = 1.6e-19
        self.debye_mm = math.sqrt(eps0 * self.Te_up / (q * self.n0 * 0.61)) * 1e3

        # Default literature ratios
        self.def_screen_ratio = 0.75
        if self.n0 <= 1e17:
            self.def_debye_ratio = 80.0
        elif self.n0 <= 4e17:
            self.def_debye_ratio = 40.0
        else:
            self.def_debye_ratio = 30.0

        # Determine initial selection & values
        if saved_settings:
            initial_method = saved_settings.get("method", "ratio")
            initial_base = saved_settings.get("ratio_base", "debye" if self.is_bulk else "screen_r")
            initial_ratio = float(saved_settings.get("ratio", self.def_debye_ratio if initial_base == "debye" else self.def_screen_ratio))
            initial_custom = float(saved_settings.get("custom_value", self.current_gap))
        else:
            initial_base = "debye" if self.is_bulk else "screen_r"
            default_base_ratio = self.def_debye_ratio if initial_base == "debye" else self.def_screen_ratio
            default_gap = default_base_ratio * (self.debye_mm if initial_base == "debye" else self.screen_r)
            if self.current_gap > 0 and abs(self.current_gap - default_gap) > 0.005:
                initial_method = "custom"
                initial_custom = self.current_gap
                initial_ratio = default_base_ratio
            else:
                initial_method = "ratio"
                initial_custom = self.current_gap if self.current_gap > 0 else default_gap
                initial_ratio = default_base_ratio

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 1. Physics Context Box
        grp_ctx = QGroupBox("Physical Parameters & Reference State")
        form_ctx = QFormLayout(grp_ctx)
        form_ctx.setLabelAlignment(Qt.AlignLeft)

        mode_str = "Entire Bulk Plasma (Debye-based)" if self.is_bulk else "Presheath Mode (Screen Aperture-based)"
        lbl_mode = QLabel(f"<b>{mode_str}</b>")
        form_ctx.addRow("Active Simulation Mode:", lbl_mode)

        lbl_screen_r = QLabel(f"{self.screen_r:.3f} mm")
        form_ctx.addRow("Screen Grid Hole Radius (r_screen):", lbl_screen_r)

        lbl_n0 = QLabel(f"{self.n0:.2e} m⁻³")
        form_ctx.addRow("Plasma Density (n₀):", lbl_n0)

        lbl_te = QLabel(f"{self.Te_up:.2f} eV")
        form_ctx.addRow("Electron Temperature (T_e,up):", lbl_te)

        lbl_debye = QLabel(f"{self.debye_mm:.4f} mm")
        form_ctx.addRow("Debye Length (λ_D):", lbl_debye)

        layout.addWidget(grp_ctx)

        # 2. Upstream Gap Method Configuration
        grp_method = QGroupBox("Upstream Gap Configuration")
        vbox_method = QVBoxLayout(grp_method)
        vbox_method.setSpacing(10)

        self.btn_group = QButtonGroup(self)

        # Radio 1: Ratio-based
        self.radio_ratio = QRadioButton("Ratio-Based Calculation")
        self.btn_group.addButton(self.radio_ratio)
        vbox_method.addWidget(self.radio_ratio)

        self.frame_ratio = QWidget()
        layout_ratio = QVBoxLayout(self.frame_ratio)
        layout_ratio.setContentsMargins(20, 0, 0, 6)
        layout_ratio.setSpacing(6)

        form_ratio = QFormLayout()
        self.combo_base = QComboBox()
        self.combo_base.addItem("Screen Grid Radius (r_screen)", "screen_r")
        self.combo_base.addItem("Debye Length (λ_D)", "debye")
        idx = self.combo_base.findData(initial_base)
        if idx >= 0:
            self.combo_base.setCurrentIndex(idx)
        form_ratio.addRow("Reference Base:", self.combo_base)

        ratio_row = QHBoxLayout()
        self.spin_ratio = QDoubleSpinBox()
        self.btn_reset_ratio = QPushButton("Reset to Default Ratio")
        self.btn_reset_ratio.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        ratio_row.addWidget(self.spin_ratio, 1)
        ratio_row.addWidget(self.btn_reset_ratio, 0)
        form_ratio.addRow("Ratio Factor:", ratio_row)

        layout_ratio.addLayout(form_ratio)

        self.lbl_ratio_formula = QLabel()
        self.lbl_ratio_formula.setStyleSheet("color: #444; font-size: 11px;")
        layout_ratio.addWidget(self.lbl_ratio_formula)

        vbox_method.addWidget(self.frame_ratio)

        # Radio 2: Custom Fixed Value
        self.radio_custom = QRadioButton("Custom Fixed Value (without ratio)")
        self.btn_group.addButton(self.radio_custom)
        vbox_method.addWidget(self.radio_custom)

        self.frame_custom = QWidget()
        layout_custom = QVBoxLayout(self.frame_custom)
        layout_custom.setContentsMargins(20, 0, 0, 4)
        layout_custom.setSpacing(4)

        form_custom = QFormLayout()
        self.spin_custom = QDoubleSpinBox()
        self.spin_custom.setRange(0.001, 50.0)
        self.spin_custom.setDecimals(3)
        self.spin_custom.setSingleStep(0.05)
        self.spin_custom.setSuffix(" mm")
        self.spin_custom.setValue(initial_custom)
        form_custom.addRow("Custom Gap (mm):", self.spin_custom)
        layout_custom.addLayout(form_custom)

        lbl_custom_hint = QLabel("Enter a direct physical gap [mm]. Density and temperature changes will not overwrite this value.")
        lbl_custom_hint.setStyleSheet("color: #666; font-size: 11px;")
        lbl_custom_hint.setWordWrap(True)
        layout_custom.addWidget(lbl_custom_hint)

        vbox_method.addWidget(self.frame_custom)
        layout.addWidget(grp_method)

        # 3. Live Preview Card
        self.card_preview = QFrame()
        self.card_preview.setFrameShape(QFrame.StyledPanel)
        self.card_preview.setStyleSheet(
            "QFrame { background-color: #f0f4f8; border: 1px solid #c0d0e0; border-radius: 6px; padding: 8px; }"
        )
        lay_prev = QVBoxLayout(self.card_preview)
        lay_prev.setContentsMargins(8, 8, 8, 8)
        self.lbl_preview_title = QLabel("Resulting Upstream Gap:")
        self.lbl_preview_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #1a365d;")
        self.lbl_preview_math = QLabel()
        self.lbl_preview_math.setStyleSheet("font-size: 12px; color: #2d3748;")
        lay_prev.addWidget(self.lbl_preview_title)
        lay_prev.addWidget(self.lbl_preview_math)
        layout.addWidget(self.card_preview)

        # 4. Buttons (Apply / Cancel)
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_apply = QPushButton("Apply")
        btn_apply.setStyleSheet("font-weight: bold; min-width: 80px; padding: 5px 12px;")
        btn_apply.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("min-width: 80px; padding: 5px 12px;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_apply)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

        # Setup initial ratio ranges & values
        self._update_ratio_spin_config()
        self.spin_ratio.setValue(initial_ratio)

        if initial_method == "custom":
            self.radio_custom.setChecked(True)
        else:
            self.radio_ratio.setChecked(True)

        # Signals
        self.radio_ratio.toggled.connect(self._on_method_toggled)
        self.radio_custom.toggled.connect(self._on_method_toggled)
        self.combo_base.currentIndexChanged.connect(self._on_base_changed)
        self.spin_ratio.valueChanged.connect(self._update_preview)
        self.spin_custom.valueChanged.connect(self._update_preview)
        self.btn_reset_ratio.clicked.connect(self._reset_default_ratio)

        self._on_method_toggled()
        self._update_preview()

    def _update_ratio_spin_config(self):
        base = self.combo_base.currentData()
        self.spin_ratio.blockSignals(True)
        if base == "screen_r":
            self.spin_ratio.setRange(0.01, 50.0)
            self.spin_ratio.setDecimals(3)
            self.spin_ratio.setSingleStep(0.05)
            self.spin_ratio.setSuffix(" × r_screen")
        else:
            self.spin_ratio.setRange(0.1, 1000.0)
            self.spin_ratio.setDecimals(2)
            self.spin_ratio.setSingleStep(1.0)
            self.spin_ratio.setSuffix(" × λ_D")
        self.spin_ratio.blockSignals(False)

    def _on_base_changed(self):
        self._update_ratio_spin_config()
        base = self.combo_base.currentData()
        if base == "screen_r":
            self.spin_ratio.setValue(self.def_screen_ratio)
        else:
            self.spin_ratio.setValue(self.def_debye_ratio)
        self._update_preview()

    def _reset_default_ratio(self):
        base = self.combo_base.currentData()
        if base == "screen_r":
            self.spin_ratio.setValue(self.def_screen_ratio)
        else:
            self.spin_ratio.setValue(self.def_debye_ratio)
        self._update_preview()

    def _on_method_toggled(self):
        is_ratio = self.radio_ratio.isChecked()
        self.frame_ratio.setEnabled(is_ratio)
        self.frame_custom.setEnabled(not is_ratio)
        self._update_preview()

    def _update_preview(self):
        if self.radio_ratio.isChecked():
            base = self.combo_base.currentData()
            ratio = self.spin_ratio.value()
            if base == "screen_r":
                base_val = self.screen_r
                base_name = "Screen Hole Radius (r_screen)"
                formula_text = (
                    f"Calculation: {ratio:.3f} × {base_val:.3f} mm = {ratio * base_val:.3f} mm "
                    f"(Default recommended: {self.def_screen_ratio:.2f})"
                )
            else:
                base_val = self.debye_mm
                base_name = "Debye Length (λ_D)"
                formula_text = (
                    f"Calculation: {ratio:.2f} × {base_val:.4f} mm = {ratio * base_val:.3f} mm "
                    f"(Literature recommended: {self.def_debye_ratio:.0f})"
                )
            final_gap = round(ratio * base_val, 3)
            self.lbl_ratio_formula.setText(formula_text)
            self.lbl_preview_title.setText(f"Resulting Upstream Gap: <b>{final_gap:.3f} mm</b>")
            self.lbl_preview_math.setText(f"Determined via ratio over {base_name}: <i>{ratio:.3f} × {base_val:.3f} mm</i>")
        else:
            final_gap = round(self.spin_custom.value(), 3)
            self.lbl_preview_title.setText(f"Resulting Upstream Gap: <b>{final_gap:.3f} mm</b>")
            self.lbl_preview_math.setText("Direct custom value (no ratio applied)")

    def get_result(self):
        if self.radio_ratio.isChecked():
            base = self.combo_base.currentData()
            ratio = self.spin_ratio.value()
            base_val = self.screen_r if base == "screen_r" else self.debye_mm
            gap = round(ratio * base_val, 3)
            settings = {
                "method": "ratio",
                "ratio_base": base,
                "ratio": ratio,
                "custom_value": self.spin_custom.value(),
                "final_gap": gap,
            }
            return gap, settings
        else:
            gap = round(self.spin_custom.value(), 3)
            settings = {
                "method": "custom",
                "ratio_base": self.combo_base.currentData(),
                "ratio": self.spin_ratio.value(),
                "custom_value": gap,
                "final_gap": gap,
            }
            return gap, settings


SWEEP_PARAMETERS = [
    {
        "id": "Te_up",
        "label": "Upstream Electron Temp Te_up (eV)",
        "unit": "eV",
        "default_start": 2.0,
        "default_end": 10.0,
        "default_n": 5,
        "min": 0.01,
        "max": 1000.0,
        "step": 0.5,
        "decimals": 2,
        "scientific": False,
    },
    {
        "id": "n0_plasma",
        "label": "Plasma Density n0_plasma (m⁻³)",
        "unit": "m⁻³",
        "default_start": 1e16,
        "default_end": 1e18,
        "default_n": 5,
        "min": 1e12,
        "max": 1e25,
        "step": 1e16,
        "decimals": 4,
        "scientific": True,
    },
    {
        "id": "Ti",
        "label": "Ion Temperature Ti (eV)",
        "unit": "eV",
        "default_start": 0.1,
        "default_end": 2.0,
        "default_n": 5,
        "min": 0.01,
        "max": 1000.0,
        "step": 0.1,
        "decimals": 2,
        "scientific": False,
    },
    {
        "id": "n0",
        "label": "Neutral Gas Density n0 (m⁻³)",
        "unit": "m⁻³",
        "default_start": 1e17,
        "default_end": 1e19,
        "default_n": 5,
        "min": 1e12,
        "max": 1e25,
        "step": 1e17,
        "decimals": 4,
        "scientific": True,
    },
    {
        "id": "Tn",
        "label": "Neutral Gas Temp Tn (K)",
        "unit": "K",
        "default_start": 300.0,
        "default_end": 800.0,
        "default_n": 5,
        "min": 1.0,
        "max": 10000.0,
        "step": 50.0,
        "decimals": 1,
        "scientific": False,
    },
    {
        "id": "grid_1_V",
        "label": "Grid 1 DC Voltage (V)",
        "unit": "V",
        "default_start": 800.0,
        "default_end": 1600.0,
        "default_n": 5,
        "min": -5000.0,
        "max": 15000.0,
        "step": 100.0,
        "decimals": 1,
        "scientific": False,
    },
    {
        "id": "grid_2_V",
        "label": "Grid 2 DC Voltage (V)",
        "unit": "V",
        "default_start": -400.0,
        "default_end": -100.0,
        "default_n": 5,
        "min": -5000.0,
        "max": 5000.0,
        "step": 50.0,
        "decimals": 1,
        "scientific": False,
    },
    {
        "id": "Te",
        "label": "Neutralizer Electron Temp Te (eV)",
        "unit": "eV",
        "default_start": 1.0,
        "default_end": 5.0,
        "default_n": 5,
        "min": 0.1,
        "max": 100.0,
        "step": 0.5,
        "decimals": 2,
        "scientific": False,
    },
]


def format_sci_value(val):
    """Format physical numerical values cleanly in scientific notation (e.g., 1e16) without redundant zeros."""
    if not isinstance(val, (int, float)):
        return str(val)
    s = f"{val:.4g}"
    if "e+" in s:
        return s.replace("e+", "e")
    elif "e-" in s:
        parts = s.split("e-")
        return f"{parts[0]}e-{int(parts[1])}"
    elif "e" not in s and (abs(val) >= 1e4 or (0 < abs(val) <= 1e-3)):
        return f"{val:.3e}".replace("e+0", "e").replace("e+", "e")
    return s


class SweepParamSpinBox(QDoubleSpinBox):
    """
    QDoubleSpinBox supporting both standard decimal display and clean
    scientific notation (e.g. 1e16 instead of 10000000000000000.0000)
    for high-magnitude physical quantities like plasma and neutral gas densities.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scientific = False

    def textFromValue(self, value):
        if self.scientific:
            return format_sci_value(value)
        return super().textFromValue(value)

    def valueFromText(self, text):
        if self.scientific:
            clean = text.replace(self.suffix(), "").strip()
            try:
                return float(clean)
            except Exception:
                return 0.0
        return super().valueFromText(text)

    def validate(self, text, pos):
        if self.scientific:
            clean = text.replace(self.suffix(), "").strip()
            if clean in ("", "-", "+", ".", "e", "E", "-e", "+e"):
                return (QValidator.Intermediate, text, pos)
            low = clean.lower()
            if low.endswith("e") or low.endswith("e+") or low.endswith("e-"):
                try:
                    prefix = low.rstrip("e+-")
                    if prefix:
                        float(prefix)
                    return (QValidator.Intermediate, text, pos)
                except ValueError:
                    return (QValidator.Invalid, text, pos)
            try:
                float(clean.replace("E", "e"))
                return (QValidator.Acceptable, text, pos)
            except ValueError:
                return (QValidator.Invalid, text, pos)
        return super().validate(text, pos)


class SweepSettingsDialog(QDialog):
    """
    Dialog allowing the user to configure a parametric sweep across multiple scenarios.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parametric Sweep Configuration")
        self.setMinimumWidth(480)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info_lbl = QLabel(
            "<b>Parametric Sweep Simulation</b><br>"
            "Automatically simulate multiple scenarios sequentially without manually editing parameters. "
            "Scenarios transition automatically once <b>Steady State</b> is reached."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #1e293b; background: #f8fafc; padding: 8px; border: 1px solid #e2e8f0; border-radius: 4px;")
        layout.addWidget(info_lbl)

        # ── Group 1: Parameter & Range ────────────────────────────────
        gb_param = QGroupBox("1. Parameter & Sweep Range")
        form_param = QFormLayout(gb_param)
        form_param.setLabelAlignment(Qt.AlignRight)

        self.combo_param = QComboBox()
        for pdef in SWEEP_PARAMETERS:
            self.combo_param.addItem(pdef["label"], pdef["id"])
        self.combo_param.currentIndexChanged.connect(self._on_param_changed)
        form_param.addRow("Quantity to Sweep:", self.combo_param)

        self.spin_start = SweepParamSpinBox()
        self.spin_start.setRange(-1e9, 1e25)
        self.spin_start.setDecimals(4)
        self.spin_start.valueChanged.connect(self._update_preview)
        form_param.addRow("Start Value:", self.spin_start)

        self.spin_end = SweepParamSpinBox()
        self.spin_end.setRange(-1e9, 1e25)
        self.spin_end.setDecimals(4)
        self.spin_end.valueChanged.connect(self._update_preview)
        form_param.addRow("End Value:", self.spin_end)

        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 100)
        self.spin_n.setValue(5)
        self.spin_n.valueChanged.connect(self._update_preview)
        form_param.addRow("Number of Simulations:", self.spin_n)

        self.combo_scale = QComboBox()
        self.combo_scale.addItem("Linear Spacing", "linear")
        self.combo_scale.addItem("Logarithmic Spacing", "log")
        self.combo_scale.currentIndexChanged.connect(self._update_preview)
        form_param.addRow("Spacing Mode:", self.combo_scale)

        self.lbl_preview = QLabel()
        self.lbl_preview.setStyleSheet("color: #475569; font-style: italic; font-size: 11px;")
        form_param.addRow("Values Preview:", self.lbl_preview)

        layout.addWidget(gb_param)

        # ── Group 2: Steady-State Transition Criteria ─────────────────
        gb_steady = QGroupBox("2. Steady-State Transition Criteria")
        form_steady = QFormLayout(gb_steady)
        form_steady.setLabelAlignment(Qt.AlignRight)

        self.spin_window = QSpinBox()
        self.spin_window.setRange(10, 500)
        self.spin_window.setValue(50)
        self.spin_window.setSuffix(" iterations")
        form_steady.addRow("Rolling Window:", self.spin_window)

        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.05, 20.0)
        self.spin_thresh.setValue(1.0)
        self.spin_thresh.setSingleStep(0.2)
        self.spin_thresh.setSuffix(" %")
        form_steady.addRow("Stationary Tolerance:", self.spin_thresh)

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 100000)
        self.spin_max_iters.setValue(5000)
        self.spin_max_iters.setSingleStep(500)
        self.spin_max_iters.setSuffix(" iterations")
        self.spin_max_iters.setToolTip("Safeguard upper bound: if steady state is not reached within this count, a warning is recorded and the sweep proceeds.")
        form_steady.addRow("Safeguard Max Iterations:", self.spin_max_iters)

        self.spin_min_iters = QSpinBox()
        self.spin_min_iters.setRange(10, 2000)
        self.spin_min_iters.setValue(50)
        self.spin_min_iters.setSuffix(" iterations")
        form_steady.addRow("Min Iterations per Point:", self.spin_min_iters)

        layout.addWidget(gb_steady)

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_start = QPushButton("Start Sweep")
        btn_start.setStyleSheet("font-weight: bold; background-color: #7e22ce; color: white; padding: 6px 14px; border-radius: 4px;")
        btn_start.clicked.connect(self.accept)
        btn_box.addWidget(btn_start)

        layout.addLayout(btn_box)

        # Initialize with first parameter
        self._on_param_changed()

    def _on_param_changed(self):
        idx = self.combo_param.currentIndex()
        if idx < 0 or idx >= len(SWEEP_PARAMETERS):
            return
        pdef = SWEEP_PARAMETERS[idx]
        unit = pdef.get("unit", "")
        self.spin_start.setSuffix(f" {unit}" if unit else "")
        self.spin_end.setSuffix(f" {unit}" if unit else "")

        is_sci = pdef.get("scientific", False)
        self.spin_start.scientific = is_sci
        self.spin_end.scientific = is_sci

        self.spin_start.setSingleStep(pdef.get("step", 0.5))
        self.spin_end.setSingleStep(pdef.get("step", 0.5))

        if is_sci:
            self.spin_start.setDecimals(4)
            self.spin_end.setDecimals(4)
            if self.combo_scale.findData("log") >= 0:
                self.combo_scale.setCurrentIndex(self.combo_scale.findData("log"))
        else:
            self.spin_start.setDecimals(pdef.get("decimals", 2))
            self.spin_end.setDecimals(pdef.get("decimals", 2))
            self.combo_scale.setCurrentIndex(0)

        self.spin_start.setValue(pdef.get("default_start", 2.0))
        self.spin_end.setValue(pdef.get("default_end", 10.0))
        self.spin_n.setValue(pdef.get("default_n", 5))
        self._update_preview()

    def _update_preview(self):
        vals = self.get_values_array()
        if len(vals) == 0:
            self.lbl_preview.setText("—")
            return
        idx = self.combo_param.currentIndex()
        pdef = SWEEP_PARAMETERS[idx] if 0 <= idx < len(SWEEP_PARAMETERS) else {}
        is_sci = pdef.get("scientific", False)

        def fmt_v(v):
            if is_sci or abs(v) >= 1e4 or (0 < abs(v) <= 1e-3):
                return format_sci_value(v)
            return f"{v:.3g}"

        if len(vals) <= 6:
            txt = ", ".join(fmt_v(v) for v in vals)
        else:
            txt = f"{fmt_v(vals[0])}, {fmt_v(vals[1])}, ... {fmt_v(vals[-1])} ({len(vals)} points)"
        self.lbl_preview.setText(f"[{txt}]")

    def get_values_array(self):
        start = self.spin_start.value()
        end = self.spin_end.value()
        n = self.spin_n.value()
        scale = self.combo_scale.currentData()
        if scale == "log" and start > 0 and end > 0:
            import numpy as np
            return np.logspace(np.log10(start), np.log10(end), n)
        else:
            import numpy as np
            return np.linspace(start, end, n)

    def get_config(self):
        idx = self.combo_param.currentIndex()
        pdef = SWEEP_PARAMETERS[idx]
        return {
            "param_id": pdef["id"],
            "param_label": pdef["label"],
            "param_unit": pdef.get("unit", ""),
            "start": self.spin_start.value(),
            "end": self.spin_end.value(),
            "n_sims": self.spin_n.value(),
            "scale": self.combo_scale.currentData(),
            "values": self.get_values_array().tolist(),
            "window": self.spin_window.value(),
            "threshold": self.spin_thresh.value(),
            "max_iterations": self.spin_max_iters.value(),
            "min_iterations": self.spin_min_iters.value(),
        }


class SweepResultsDialog(QDialog):
    """
    Pop-up dialog displaying results of a completed parametric sweep,
    including clear flags/warnings if steady state was not reached,
    and one-click CSV / JSON export.
    """
    def __init__(self, sweep_config, results, parent=None):
        super().__init__(parent)
        self.sweep_config = sweep_config
        self.results = results
        self.setWindowTitle("Parametric Sweep Results")
        self.resize(1020, 500)
        self.setMinimumSize(800, 400)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header with warning if any point failed steady state
        param_label = sweep_config.get("param_label", "Parameter")
        unconverged_count = sum(1 for r in results if not r.get("steady_state_reached", True))

        header_box = QFrame()
        header_box.setFrameShape(QFrame.StyledPanel)
        h_layout = QVBoxLayout(header_box)
        h_layout.setContentsMargins(10, 8, 10, 8)

        title_lbl = QLabel(f"<b>Sweep Completed: {len(results)} Scenarios ({param_label})</b>")
        title_lbl.setStyleSheet("font-size: 13px; color: #1e293b;")
        h_layout.addWidget(title_lbl)

        if unconverged_count > 0:
            warn_lbl = QLabel(
                f"⚠️ <b>WARNING: {unconverged_count} of {len(results)} scenarios did not reach steady state "
                f"within the safeguard limit of {sweep_config.get('max_iterations', 5000)} iterations!</b>"
            )
            warn_lbl.setStyleSheet("color: #b91c1c; font-weight: bold; background: #fee2e2; padding: 4px 8px; border-radius: 4px;")
            h_layout.addWidget(warn_lbl)
        else:
            ok_lbl = QLabel("✓ All scenarios successfully achieved steady-state condition.")
            ok_lbl.setStyleSheet("color: #15803d; font-weight: bold; background: #dcfce7; padding: 4px 8px; border-radius: 4px;")
            h_layout.addWidget(ok_lbl)

        layout.addWidget(header_box)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Scenario",
            f"{param_label}",
            "Steady State?",
            "Iterations",
            "Saddle V (V)",
            "Divergence Grid Exit (°)",
            "Divergence 2nd Plane (°)",
            "Exit Current (A)",
            "Perveance Fraction",
            "Transparency",
            "Total Energy (J)",
        ])
        self.table.setRowCount(len(results))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        param_id = sweep_config.get("param_id", "")
        is_sci = param_id in ("n0_plasma", "n0") or any(
            p.get("scientific", False) for p in SWEEP_PARAMETERS if p["id"] == param_id
        )

        for row, res in enumerate(results):
            scen_idx = res.get("scenario_idx", row + 1)
            p_val = res.get("parameter_value", 0.0)
            ss_reached = res.get("steady_state_reached", True)
            iters = res.get("steady_iterations", 0)
            saddle_v = res.get("saddle_potential_V", 0.0)
            div = res.get("divergence_deg", res.get("divergence_grid_deg", 0.0))
            div_2nd = res.get("divergence_2nd_plane_deg")
            if div_2nd is None:
                div_2nd = res.get("divergence_plane2_deg")
            if div_2nd is None:
                div_2nd = res.get("divergence_mid_deg", 0.0)
            i_exit = res.get("exit_ion_current_A", 0.0)
            perv_frac = res.get("perveance_fraction")
            if perv_frac is None:
                perv_frac = res.get("P_fraction")
            trans = res.get("transparency", 0.0)
            e_tot = res.get("total_energy_J", 0.0)

            ss_text = "✓ YES" if ss_reached else f"⚠️ NO (Max {res.get('max_iters_hit', 5000)})"

            self.table.setItem(row, 0, QTableWidgetItem(f"Scenario {scen_idx}"))
            p_text = format_sci_value(p_val) if (is_sci or abs(p_val) >= 1e4 or (0 < abs(p_val) <= 1e-3)) else f"{p_val:.4g}"
            self.table.setItem(row, 1, QTableWidgetItem(p_text))

            item_ss = QTableWidgetItem(ss_text)
            if not ss_reached:
                item_ss.setForeground(Qt.red)
            else:
                item_ss.setForeground(Qt.darkGreen)
            self.table.setItem(row, 2, item_ss)

            self.table.setItem(row, 3, QTableWidgetItem(f"{iters:,}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{saddle_v:.2f}"))
            div_str = f"{div:.2f}" if (isinstance(div, (int, float)) and np.isfinite(div)) else str(div)
            self.table.setItem(row, 5, QTableWidgetItem(div_str))
            div_2nd_str = f"{div_2nd:.2f}" if (isinstance(div_2nd, (int, float)) and np.isfinite(div_2nd)) else str(div_2nd)
            self.table.setItem(row, 6, QTableWidgetItem(div_2nd_str))
            self.table.setItem(row, 7, QTableWidgetItem(f"{i_exit:.3e}"))
            perv_frac_str = f"{perv_frac:.4f}" if (isinstance(perv_frac, (int, float)) and np.isfinite(perv_frac)) else ("N/A" if perv_frac is None else str(perv_frac))
            self.table.setItem(row, 8, QTableWidgetItem(perv_frac_str))
            self.table.setItem(row, 9, QTableWidgetItem(f"{trans:.3f}"))
            self.table.setItem(row, 10, QTableWidgetItem(f"{e_tot:.3e}"))

        layout.addWidget(self.table, 1)

        # Action Buttons
        btn_row = QHBoxLayout()

        btn_export = QPushButton("Export to...")
        btn_export.setStyleSheet("font-weight: bold; padding: 6px 16px; background-color: #2563eb; color: white; border-radius: 4px;")
        btn_export.setToolTip("Export sweep results to CSV, TXT, or JSON")
        btn_export.clicked.connect(self._export_dialog)
        btn_row.addWidget(btn_export)

        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)

        layout.addLayout(btn_row)

    def _export_dialog(self):
        """Single export entry point allowing the user to select format (.csv, .txt, .json) in the file dialog."""
        fpath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Sweep Results",
            "sweep_results.csv",
            "CSV Files (*.csv);;Text Files (*.txt);;JSON Files (*.json);;All Files (*.*)"
        )
        if not fpath:
            return

        import os
        if "(*.json)" in selected_filter:
            if fpath.lower().endswith(".csv") or fpath.lower().endswith(".txt"):
                fpath = os.path.splitext(fpath)[0] + ".json"
            elif not fpath.lower().endswith(".json"):
                fpath += ".json"
            self._write_json(fpath, notify=True)
        elif "(*.txt)" in selected_filter:
            if fpath.lower().endswith(".csv") or fpath.lower().endswith(".json"):
                fpath = os.path.splitext(fpath)[0] + ".txt"
            elif not fpath.lower().endswith(".txt"):
                fpath += ".txt"
            self._write_tabular(fpath, is_txt=True, notify=True)
        else:
            if fpath.lower().endswith(".json"):
                self._write_json(fpath, notify=True)
                return
            is_txt = fpath.lower().endswith(".txt")
            if not is_txt and not fpath.lower().endswith(".csv"):
                fpath += ".csv"
            self._write_tabular(fpath, is_txt=is_txt, notify=True)

    def _write_tabular(self, fpath, is_txt=False, notify=True):
        import csv

        delim = "\t" if is_txt else ","
        unconverged = [r for r in self.results if not r.get("steady_state_reached", True)]
        param_label = self.sweep_config.get("param_label", "Parameter")

        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delim)
            writer.writerow(["# PY-BEMCS Parametric Sweep Results"])
            writer.writerow([f"# Swept Parameter: {param_label}"])
            writer.writerow([f"# Total Scenarios: {len(self.results)}"])

            if unconverged:
                writer.writerow([
                    f"# WARNING: {len(unconverged)} of {len(self.results)} scenarios DID NOT reach steady state "
                    f"within the safeguard limit of {self.sweep_config.get('max_iterations', 5000)} iterations!"
                ])
                for r in unconverged:
                    writer.writerow([f"# WARNING Scenario {r.get('scenario_idx')}: stopped at {r.get('steady_iterations')} iters without convergence."])
            else:
                writer.writerow(["# Status: All scenarios achieved steady state."])

            # Header row
            writer.writerow([
                "scenario_idx",
                "parameter_id",
                "parameter_label",
                "parameter_value",
                "steady_state_reached",
                "warning_note",
                "iterations",
                "t_sim_us",
                "saddle_potential_V",
                "beam_divergence_deg",
                "beam_divergence_2nd_plane_deg",
                "exit_ion_current_A",
                "perveance_effective",
                "perveance_max",
                "perveance_fraction",
                "grid_transparency",
                "total_energy_J",
                "total_charge_C",
                "field_energy_J",
            ])

            param_id = self.sweep_config.get("param_id", "")
            is_sci = param_id in ("n0_plasma", "n0") or any(
                p.get("scientific", False) for p in SWEEP_PARAMETERS if p["id"] == param_id
            )

            for r in self.results:
                reached = r.get("steady_state_reached", True)
                warn = "" if reached else f"WARNING: Max iterations ({r.get('max_iters_hit', 5000)}) reached without steady state"
                val = r.get("parameter_value", "")
                if is_sci and isinstance(val, (int, float)):
                    val_str = format_sci_value(val)
                else:
                    val_str = val
                div_val = r.get("divergence_deg", r.get("divergence_grid_deg", ""))
                div_2nd_val = r.get("divergence_2nd_plane_deg")
                if div_2nd_val is None:
                    div_2nd_val = r.get("divergence_plane2_deg")
                if div_2nd_val is None:
                    div_2nd_val = r.get("divergence_mid_deg", "")

                p_eff = r.get("perveance_effective", r.get("Peffective", ""))
                p_max = r.get("perveance_max", r.get("Pmax", ""))
                p_frac = r.get("perveance_fraction", r.get("P_fraction", ""))
                p_eff_str = f"{p_eff:.4e}" if (isinstance(p_eff, (int, float)) and np.isfinite(p_eff)) else str(p_eff)
                p_max_str = f"{p_max:.4e}" if (isinstance(p_max, (int, float)) and np.isfinite(p_max)) else str(p_max)
                p_frac_str = f"{p_frac:.4f}" if (isinstance(p_frac, (int, float)) and np.isfinite(p_frac)) else str(p_frac)

                writer.writerow([
                    r.get("scenario_idx", ""),
                    self.sweep_config.get("param_id", ""),
                    param_label,
                    val_str,
                    "YES" if reached else "NO",
                    warn,
                    r.get("steady_iterations", ""),
                    r.get("t_sim_us", ""),
                    r.get("saddle_potential_V", ""),
                    div_val,
                    div_2nd_val,
                    r.get("exit_ion_current_A", ""),
                    p_eff_str,
                    p_max_str,
                    p_frac_str,
                    r.get("transparency", ""),
                    r.get("total_energy_J", ""),
                    r.get("total_charge_C", ""),
                    r.get("field_energy_J", ""),
                ])

        if notify:
            fmt_name = "Text file (.txt)" if is_txt else "CSV file (.csv)"
            QMessageBox.information(self, "Export Complete", f"Sweep results successfully saved as {fmt_name} to:\n{fpath}")

    def _write_json(self, fpath, notify=True):
        import json

        unconverged_count = sum(1 for r in self.results if not r.get("steady_state_reached", True))
        data = {
            "metadata": {
                "software": "PY-BEMCS",
                "sweep_config": self.sweep_config,
                "total_scenarios": len(self.results),
                "unconverged_count": unconverged_count,
                "warning": (
                    f"WARNING: {unconverged_count} scenarios did not reach steady state before hitting the safeguard limit of {self.sweep_config.get('max_iterations', 5000)} iterations!"
                    if unconverged_count > 0 else None
                ),
            },
            "results": self.results,
        }
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if notify:
            QMessageBox.information(self, "Export Complete", f"Sweep results successfully saved to:\n{fpath}")



