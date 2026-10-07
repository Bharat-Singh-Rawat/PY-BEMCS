"""
Modal configuration dialogs for PY-BEMCS.
Includes Beam Species, Grid Material, Advanced Physics/Mesh, and GUI Settings dialogs.
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton,
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton
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
