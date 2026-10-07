"""
UI Builder mixin for DigitalTwinApp.
Builds the top status bar, menu bar, scrollable control panel, and matplotlib canvas.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QCheckBox, QMessageBox, QFileDialog, QComboBox,
    QScrollArea, QGroupBox, QAction, QMenuBar, QSplitter,
    QFrame, QSizePolicy, QToolButton, QMenu, QActionGroup
)
from PyQt5.QtCore import Qt

from ..widgets import ScientificSpinBox
from ..bootstrap import _safe_start_dir
from ..config_io import load_json_config
from .figure_layout import setup_figure_layout


class UIBuilderMixin:
    """Mixin providing layout construction and widget setup for DigitalTwinApp."""

    def setup_menu_bar(self):
        menubar = self.menuBar()
        if menubar is None:
            menubar = QMenuBar(self)
            self.setMenuBar(menubar)
        assert menubar is not None

        settings_menu = menubar.addMenu("Settings")
        adv_action = QAction("Advanced Parameters...", self)
        adv_action.triggered.connect(self.open_advanced_settings)
        settings_menu.addAction(adv_action)

        gui_action = QAction("GUI Settings...", self)
        gui_action.triggered.connect(self.open_gui_settings)
        settings_menu.addAction(gui_action)

        self.reload_action = QAction(f"Reload {self.current_config_name}", self)
        self.reload_action.triggered.connect(self.reload_config)
        settings_menu.addAction(self.reload_action)

        beam_menu = menubar.addMenu("Beam")
        assert beam_menu is not None
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

        diag_menu = menubar.addMenu("Diagnostics")
        assert diag_menu is not None

        mesh_action = QAction("Mesh", self)
        mesh_action.triggered.connect(self.open_mesh_window)
        diag_menu.addAction(mesh_action)

        perf_action = QAction("⚡ Performance Monitor...", self)
        perf_action.triggered.connect(self.open_perf_window)
        diag_menu.addAction(perf_action)

        ppc_action = QAction("PPC Distribution...", self)
        ppc_action.triggered.connect(self.open_ppc_window)
        diag_menu.addAction(ppc_action)

        phys_action = QAction("Total Energy ...", self)
        phys_action.triggered.connect(self.open_phys_window)
        diag_menu.addAction(phys_action)

        charge_action = QAction("Total Charge Monitor...", self)
        charge_action.triggered.connect(self.open_charge_window)
        diag_menu.addAction(charge_action)

        iedf_action = QAction("Energy Dist. (IEDF/EEDF)...", self)
        iedf_action.triggered.connect(self.open_iedf_window)
        diag_menu.addAction(iedf_action)

        self.lbl_active_config = QLabel(f"  Config: {self.current_config_name}  ")
        self.lbl_active_config.setStyleSheet(
            "font-weight: bold; color: #1a5276; font-size: 11px; padding-right: 12px;"
        )
        menubar.setCornerWidget(self.lbl_active_config, Qt.TopRightCorner)

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
        row3, spin_gap = self.create_input("Gap to Next (mm):", 0.0, 1000.0, 0.01, 4)
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
        root_layout = QVBoxLayout(main_widget)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # Top Bar: Simulation Duration Timer & Status
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar.setStyleSheet("""
            QFrame#topBar {
                background-color: #f8fafc;
                border-bottom: 1px solid #cbd5e1;
            }
        """)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(14, 5, 14, 5)
        top_bar_layout.setSpacing(12)

        self.lbl_sim_status_badge = QLabel("IDLE")
        self.lbl_sim_status_badge.setStyleSheet(
            "background-color: #e2e8f0; color: #475569; font-weight: bold; "
            "border-radius: 4px; padding: 2px 8px; font-size: 11px;"
        )
        top_bar_layout.addWidget(self.lbl_sim_status_badge)

        self.lbl_sim_timer = QLabel("⏱ Simulation Duration: 00:00:00.0")
        self.lbl_sim_timer.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; font-weight: bold; "
            "color: #0f172a; padding: 3px 10px; background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px;"
        )
        top_bar_layout.addWidget(self.lbl_sim_timer)

        self.btn_reset_timer = QPushButton("Reset Timer")
        self.btn_reset_timer.setToolTip("Reset the simulation duration timer to 00:00:00.0")
        self.btn_reset_timer.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        self.btn_reset_timer.clicked.connect(self.reset_simulation_timer)
        top_bar_layout.addWidget(self.btn_reset_timer)

        top_bar_layout.addStretch()

        self.lbl_top_sim_time = QLabel("Plasma Time: 0.00 µs | Iter: 0")
        self.lbl_top_sim_time.setStyleSheet("font-family: monospace; font-size: 11px; color: #64748b;")
        top_bar_layout.addWidget(self.lbl_top_sim_time)

        root_layout.addWidget(top_bar)

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(0)
        content_splitter.setChildrenCollapsible(False)

        # SCROLLABLE CONTROL PANEL
        scroll_area = QScrollArea()
        scroll_area.setFixedWidth(330)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

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
            "Distance from the injection wall to the left face of the screen grid [mm].\n\n"
            "Presheath mode (default): auto = 0.75 × screen radius.\n"
            "Entire Bulk Plasma mode: auto = Debye-based (80/40/30 × λ_D).\n\n"
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
        row, self.inputs["target_ppc"] = self.create_input("Target PPC:", 1.0, 100000.0, 40.0, 0)
        control_layout.addLayout(row)

        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("4. SIMULATION MODE"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Both", "Thermal", "Erosion", "Trajectories"])
        control_layout.addWidget(self.combo_mode)

        control_layout.addWidget(QLabel("Geometry Mode:"))
        self.combo_geometry = QComboBox()
        self.combo_geometry.addItems(["half_hole", "one_hole", "two_holes"])
        self.combo_geometry.setToolTip(
            "half_hole  — half-pitch symmetry, hole at y=0 (fastest, default)\n"
            "one_hole   — full single-aperture domain, hole centred at y=1.5*r\n"
            "two_holes  — full dual-aperture domain, two holes one pitch apart"
        )
        control_layout.addWidget(self.combo_geometry)

        row, self.inputs["inj_time_µs"] = self.create_input(
            "Injection Time (us):", 0.0, 1000.0, 0.01, 4
        )
        control_layout.addLayout(row)
        self.inputs["inj_time_µs"].setValue(0.0)

        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("5. NEUTRALIZER"))
        self.chk_neut_match_ion = QCheckBox("el. rate = ion rate")
        self.chk_neut_match_ion.setToolTip(
            "When checked, the neutralizer automatically emits electron macroparticles\n"
            "matching the ion beam current exiting the last grid each step.\n"
            "The 'e- Inject Rate' input is ignored in this mode."
        )
        self.chk_neut_match_ion.setChecked(False)
        self.chk_neut_match_ion.stateChanged.connect(self._on_neut_match_toggled)
        control_layout.addWidget(self.chk_neut_match_ion)

        row, self.inputs["neut_rate"] = self.create_input("e- Inject Rate(macro):", 0.0, 1e9, 1.0, 4)
        control_layout.addLayout(row)
        row, self.inputs["Te"] = self.create_input("e- Temp (eV):", 0.0, 1000.0, 0.1, 4)
        control_layout.addLayout(row)
        row, self.inputs["neut_x"] = self.create_input("Axial Dist (x, mm):", 0.0, 500.0, 0.1, 3)
        self.inputs["neut_x"].setToolTip("Neutralizer axial distance along x (mm). Auto-set to 90% into downstream plume if not manually edited.")
        self.inputs["neut_x"].editingFinished.connect(self._validate_neut_x)
        control_layout.addLayout(row)
        row, self.inputs["neut_r"] = self.create_input("Radius (y, mm):", 0.01, 100.0, 0.1, 2)
        self.inputs["neut_r"].setToolTip("Neutralizer radius (y, mm). Radial boundary for injected electrons.")
        control_layout.addLayout(row)

        control_layout.addSpacing(15)

        self.btn_build = QPushButton("1. BUILD DOMAIN")
        self.btn_build.clicked.connect(self.build_domain)
        self.btn_toggle = QPushButton("2. START BEAM")
        self.btn_toggle.clicked.connect(self.toggle_sim)

        self.btn_csv = QPushButton("Export Data (.csv / .txt)")
        self.btn_csv.clicked.connect(self.export_csv)

        self.chk_track_ptcls = QCheckBox("Record Kinematics")
        self.btn_export_trk  = QPushButton("Export Particle Data (.csv / .txt)")
        self.btn_export_trk.clicked.connect(self.exporttrackingdata)

        self.chk_record = QCheckBox("Record Frames (0)")
        self.btn_save   = QPushButton("Save GIF Animation")
        self.btn_save.clicked.connect(self.save_gif)

        self.lbl_status   = QLabel("Status: Ready.")
        self.lbl_temp     = QLabel("Grid Temps: Ready")
        self.lbl_material = QLabel(f"Grid Material: {self.mat_name or 'Molybdenum'}")
        self.lbl_status.setWordWrap(True)
        self.lbl_temp.setWordWrap(True)
        self.lbl_material.setWordWrap(True)

        for w in [
            self.btn_build, self.btn_toggle, self.btn_csv, self.chk_track_ptcls,
            self.btn_export_trk,
            self.chk_record, self.btn_save,
            self.lbl_status, self.lbl_temp, self.lbl_material
        ]:
            control_layout.addWidget(w)

        perfbox    = QGroupBox("⚡ Beam Diagnostics ")
        perflayout = QVBoxLayout()
        perflayout.setSpacing(2)
        perflayout.setContentsMargins(6, 4, 6, 4)

        self.lblTime         = QLabel("t_sim:        — us")
        self.lblTransparency = QLabel("Transparency: —")
        self.lblPerfPtcls    = QLabel("Ptcls (i/e):  —")
        self.lblPerfStep     = QLabel("Step / RAM:   —")

        for lbl in [self.lblTime, self.lblTransparency, self.lblPerfPtcls, self.lblPerfStep]:
            lbl.setStyleSheet("font-family: monospace; font-size: 11px;")
            perflayout.addWidget(lbl)

        perfbox.setLayout(perflayout)
        control_layout.addWidget(perfbox)

        scroll_area.setWidget(control_panel)
        content_splitter.addWidget(scroll_area)

        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_splitter.addWidget(self.canvas)

        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)

        root_layout.addWidget(content_splitter)

        setup_figure_layout(self)
        self.canvas.mpl_connect('pick_event', self._on_legend_pick)
        self._update_div_legend()
        self.setup_divergence_menu()

    def setup_divergence_menu(self):
        """Build the downstream arrow dropdown button and cascading tent menus for divergence options."""
        self.DIVERGENCE_METHODS = {
            "95%": 95.0,
            "90%": 90.0,
        }

        # Small downstream arrow button overlaid directly on the canvas
        self.btn_div_menu = QToolButton(self.canvas)
        self.btn_div_menu.setText("▼")
        self.btn_div_menu.setToolTip("Divergence Options")
        self.btn_div_menu.setFixedSize(18, 16)
        self.btn_div_menu.setCursor(Qt.PointingHandCursor)
        self.btn_div_menu.setFocusPolicy(Qt.NoFocus)
        self.btn_div_menu.setStyleSheet("""
            QToolButton {
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 3px;
                color: #495057;
                font-size: 8px;
                font-weight: bold;
                padding: 0px;
            }
            QToolButton:hover {
                background-color: #e2e6ea;
                border-color: #adb5bd;
                color: #212529;
            }
            QToolButton:pressed {
                background-color: #dae0e5;
            }
            QToolButton::menu-indicator {
                image: none;
                width: 0px;
            }
        """)

        # Main tent menu
        self.menu_div = QMenu(self)
        self.menu_div.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                border: 1px solid #ced4da;
                padding: 4px 0px;
                font-size: 11px;
            }
            QMenu::item {
                padding: 5px 22px 5px 20px;
            }
            QMenu::item:selected {
                background-color: #e7f1ff;
                color: #0d6efd;
            }
        """)

        # Submenu: "Divergence evaluation" (extensible for future submenus / buttons)
        self.menu_div_eval = self.menu_div.addMenu("Divergence evaluation")
        self.menu_div_eval.setStyleSheet(self.menu_div.styleSheet())

        # Submenu items: 95% and 90% (extensible for future evaluation methods)
        self._div_method_actions = {}
        self._div_method_group = QActionGroup(self)
        self._div_method_group.setExclusive(True)

        current_method = getattr(self, 'div_method', '95%')
        for method_name in self.DIVERGENCE_METHODS.keys():
            act = self.menu_div_eval.addAction(method_name)
            act.setCheckable(True)
            act.setChecked(method_name == current_method)
            act.triggered.connect(lambda checked, m=method_name: self.set_divergence_method(m))
            self._div_method_group.addAction(act)
            self._div_method_actions[method_name] = act

        self.btn_div_menu.setMenu(self.menu_div)
        self.btn_div_menu.setPopupMode(QToolButton.InstantPopup)

        # Reposition button on canvas draw and resize
        self.canvas.mpl_connect('draw_event', lambda event: self._reposition_div_menu_btn())
        self.canvas.mpl_connect('resize_event', lambda event: self._reposition_div_menu_btn())
        self._reposition_div_menu_btn()
