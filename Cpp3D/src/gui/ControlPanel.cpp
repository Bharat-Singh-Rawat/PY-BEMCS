#include "gui/ControlPanel.h"
#include <QScrollArea>
#include <QHBoxLayout>

namespace BEMCS {

ControlPanel::ControlPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
}

QHBoxLayout* ControlPanel::createSpinRow(const QString& label,
                                          QDoubleSpinBox*& spin,
                                          double min, double max, double val,
                                          double step, int decimals) {
    auto row = new QHBoxLayout();
    auto lbl = new QLabel(label);
    lbl->setFixedWidth(140);
    spin = new QDoubleSpinBox();
    spin->setRange(min, max);
    spin->setValue(val);
    spin->setSingleStep(step);
    spin->setDecimals(decimals);
    row->addWidget(lbl);
    row->addWidget(spin);
    return row;
}

void ControlPanel::addGridUI(double v, double t, double gap, double r,
                              double cham) {
    int idx = static_cast<int>(gridWidgets_.size()) + 1;
    GridWidget gw;
    gw.group = new QGroupBox(QString("Grid %1").arg(idx));
    auto lay = new QVBoxLayout();

    QHBoxLayout* row;
    row = createSpinRow("DC Voltage (V):", gw.voltage, -5000, 15000, v, 100, 0);
    lay->addLayout(row);
    row = createSpinRow("Thickness (mm):", gw.thickness, 0.1, 10, t, 0.1, 1);
    lay->addLayout(row);
    row = createSpinRow("Gap to Next (mm):", gw.gap, 0.1, 10, gap, 0.1, 1);
    lay->addLayout(row);
    row = createSpinRow("Hole Radius (mm):", gw.radius, 0.1, 10, r, 0.1, 1);
    lay->addLayout(row);
    row = createSpinRow("Chamfer (deg):", gw.chamfer, 0, 45, cham, 1, 0);
    lay->addLayout(row);

    gw.group->setLayout(lay);
    gridsLayout_->insertWidget(gridsLayout_->count() - 1, gw.group);
    gridWidgets_.push_back(gw);

    // Update RF grid combo
    comboRFGrid_->clear();
    for (size_t i = 0; i < gridWidgets_.size(); i++) {
        comboRFGrid_->addItem(QString("Grid %1").arg(i + 1));
    }
}

void ControlPanel::removeGridUI() {
    if (gridWidgets_.size() > 1) {
        auto gw = gridWidgets_.back();
        gw.group->deleteLater();
        gridWidgets_.pop_back();

        comboRFGrid_->clear();
        for (size_t i = 0; i < gridWidgets_.size(); i++) {
            comboRFGrid_->addItem(QString("Grid %1").arg(i + 1));
        }
    }
}

void ControlPanel::setupUI() {
    auto scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    scrollArea->setFixedWidth(380);

    auto panel = new QWidget();
    auto mainLay = new QVBoxLayout(panel);

    // ── 1. MULTI-GRID OPTICS ───────────────────────────────────────────
    mainLay->addWidget(new QLabel("<b>1. MULTI-GRID OPTICS</b>"));
    gridsLayout_ = new QVBoxLayout();

    auto btnRow = new QHBoxLayout();
    auto btnAdd = new QPushButton("+ Add Grid");
    auto btnRem = new QPushButton("- Remove Grid");
    connect(btnAdd, &QPushButton::clicked, [this]() {
        addGridUI(0, 1.0, 1.0, 1.0, 0);
    });
    connect(btnRem, &QPushButton::clicked, [this]() { removeGridUI(); });
    btnRow->addWidget(btnAdd);
    btnRow->addWidget(btnRem);
    gridsLayout_->addLayout(btnRow);

    mainLay->addLayout(gridsLayout_);

    // Initialize RF combo before adding grids
    comboRFGrid_ = new QComboBox();
    addGridUI(1650, 1.0, 1.0, 1.0, 0);
    addGridUI(-350, 1.0, 1.0, 0.6, 0);

    mainLay->addSpacing(10);

    // ── 2. RF CO-EXTRACTION ────────────────────────────────────────────
    mainLay->addWidget(new QLabel("<b>2. RF CO-EXTRACTION</b>"));
    chkRF_ = new QCheckBox("Enable RF Modulated Potential");
    mainLay->addWidget(chkRF_);

    auto rfRow = new QHBoxLayout();
    rfRow->addWidget(new QLabel("Apply RF to:"));
    rfRow->addWidget(comboRFGrid_);
    mainLay->addLayout(rfRow);

    QHBoxLayout* row;
    row = createSpinRow("Frequency (MHz):", spinRFFreq_, 0.1, 100, 13.56, 0.1, 2);
    mainLay->addLayout(row);
    row = createSpinRow("Amplitude (V):", spinRFAmp_, 0, 5000, 500, 50, 0);
    mainLay->addLayout(row);

    mainLay->addSpacing(10);

    // ── 3. PLASMA & SPUTTERING ─────────────────────────────────────────
    mainLay->addWidget(new QLabel("<b>3. PLASMA & SPUTTERING</b>"));
    row = createSpinRow("Plasma Dens (m-3):", spinPlasmaDens_, 1e15, 1e19, 1e17, 1e16, 0);
    mainLay->addLayout(row);
    row = createSpinRow("Upstream Te (eV):", spinTeUp_, 0.1, 20, 3.0, 0.5, 1);
    mainLay->addLayout(row);
    row = createSpinRow("Ion Temp (eV):", spinTi_, 0.1, 10, 2.0, 0.5, 1);
    mainLay->addLayout(row);
    row = createSpinRow("Neutral Temp (K):", spinTn_, 100, 2000, 300, 100, 0);
    mainLay->addLayout(row);
    row = createSpinRow("Neutral Dens (m-3):", spinNeutralDens_, 1e18, 1e22, 1e20, 1e19, 0);
    mainLay->addLayout(row);
    row = createSpinRow("Accel. Factor:", spinAccel_, 1, 1e16, 1, 1e12, 0);
    mainLay->addLayout(row);
    row = createSpinRow("Cell Fail Thresh:", spinThresh_, 0.1, 1e5, 10000, 0.1, 1);
    mainLay->addLayout(row);

    mainLay->addSpacing(10);

    // ── 4. DOMAIN SETTINGS ─────────────────────────────────────────────
    mainLay->addWidget(new QLabel("<b>4. 3D DOMAIN</b>"));
    row = createSpinRow("Lx transv. (mm):", spinLx_, 0.1, 200, 3, 0.5, 1);
    mainLay->addLayout(row);
    row = createSpinRow("Ly transv. (mm):", spinLy_, 0.1, 100, 3, 0.5, 1);
    mainLay->addLayout(row);
    row = createSpinRow("Lz beam (mm):", spinLz_, 0.1, 200, 10, 0.5, 1);
    mainLay->addLayout(row);
    row = createSpinRow("Cell Size (mm):", spinDx_, 0.001, 1.0, 0.05, 0.001, 4);
    mainLay->addLayout(row);

    // dt and batch controls
    row = createSpinRow("Timestep dt (ns):", spinDt_, 0.1, 100.0, 1.0, 0.5, 1);
    spinDt_->setToolTip("Simulation timestep. Keep below ~1 ns for stability at 1650 V / 0.05 mm cells.\n"
                        "Increase to see extraction in fewer steps (may reduce accuracy).");
    mainLay->addLayout(row);

    auto batchRow = new QHBoxLayout();
    auto lblBatch = new QLabel("Steps/Frame:"); lblBatch->setFixedWidth(140);
    spinBatchSteps_ = new QSpinBox();
    spinBatchSteps_->setRange(1, 500);
    spinBatchSteps_->setValue(10);
    spinBatchSteps_->setToolTip("Physics steps computed per render frame.\n"
                                "Increase to advance simulation faster visually.");
    batchRow->addWidget(lblBatch);
    batchRow->addWidget(spinBatchSteps_);
    mainLay->addLayout(batchRow);

    // Dimensional scaling for Debye length resolution
    auto scaleRow = new QHBoxLayout();
    auto lblScale = new QLabel("Dim. Scaling:"); lblScale->setFixedWidth(140);
    comboDimScale_ = new QComboBox();
    comboDimScale_->addItems({"1x (Physical)", "10x (Resolve Debye)", "100x (Fine Debye)"});
    comboDimScale_->setToolTip(
        "Self-similar PIC scaling to make the Debye length resolvable.\n"
        "10x:  divide all lengths and voltages by 10; plasma density * 100;\n"
        "      timestep / 10. Preserves E-field on particles.\n"
        "100x: same with factor 100.\n"
        "Affects domain, cells, grid optics, neutralizer, voltages, dt,\n"
        "and plasma/neutral densities (temperatures stay physical).");
    scaleRow->addWidget(lblScale);
    scaleRow->addWidget(comboDimScale_);
    mainLay->addLayout(scaleRow);

    mainLay->addSpacing(10);

    // ── 5. NEUTRALIZER ─────────────────────────────────────────────────
    mainLay->addWidget(new QLabel("<b>5. NEUTRALIZER</b>"));
    auto neutRow1 = new QHBoxLayout();
    auto lblNR = new QLabel("Inject Rate:"); lblNR->setFixedWidth(140);
    spinNeutRate_ = new QSpinBox();
    spinNeutRate_->setRange(0, 1000);
    spinNeutRate_->setValue(0);
    neutRow1->addWidget(lblNR);
    neutRow1->addWidget(spinNeutRate_);
    mainLay->addLayout(neutRow1);

    row = createSpinRow("Electron Temp (eV):", spinNeutTemp_, 0.1, 50, 5.0, 0.5, 1);
    mainLay->addLayout(row);

    mainLay->addSpacing(10);

    // ── 6. SIMULATION MODE ─────────────────────────────────────────────
    mainLay->addWidget(new QLabel("<b>6. SIMULATION MODE</b>"));
    comboSimMode_ = new QComboBox();
    comboSimMode_->addItems({"Both (Thermal + Erosion)", "Thermal Only", "Erosion Only"});
    mainLay->addWidget(comboSimMode_);

    mainLay->addSpacing(15);

    // ── CONTROL BUTTONS ────────────────────────────────────────────────
    auto btnBuild = new QPushButton("1. BUILD DOMAIN");
    btnBuild->setStyleSheet("QPushButton { background-color: #2196F3; color: white; "
                            "font-weight: bold; padding: 8px; }");
    connect(btnBuild, &QPushButton::clicked, this, &ControlPanel::buildDomainRequested);
    mainLay->addWidget(btnBuild);

    btnStartStop_ = new QPushButton("2. START SIMULATION");
    btnStartStop_->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; "
                                  "font-weight: bold; padding: 8px; }");
    connect(btnStartStop_, &QPushButton::clicked, [this]() {
        isRunning_ = !isRunning_;
        btnStartStop_->setText(isRunning_ ? "STOP SIMULATION" : "2. START SIMULATION");
        btnStartStop_->setStyleSheet(isRunning_
            ? "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }"
            : "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }");
        emit startStopToggled(isRunning_);
    });
    mainLay->addWidget(btnStartStop_);

    auto btnImport = new QPushButton("Import STEP Geometry...");
    btnImport->setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 6px; }");
    connect(btnImport, &QPushButton::clicked, this, &ControlPanel::importGeometryRequested);
    mainLay->addWidget(btnImport);

    auto btnMesh = new QPushButton("Meshing Options...");
    connect(btnMesh, &QPushButton::clicked, this, &ControlPanel::meshSettingsRequested);
    mainLay->addWidget(btnMesh);

    auto utilRow = new QHBoxLayout();
    auto btnReset = new QPushButton("Reset");
    connect(btnReset, &QPushButton::clicked, this, &ControlPanel::resetRequested);
    auto btnExport = new QPushButton("Export Data");
    connect(btnExport, &QPushButton::clicked, this, &ControlPanel::exportDataRequested);
    utilRow->addWidget(btnReset);
    utilRow->addWidget(btnExport);
    mainLay->addLayout(utilRow);

    mainLay->addSpacing(10);

    // ── VIEW CONTROLS ──────────────────────────────────────────────────
    mainLay->addWidget(new QLabel("<b>VIEW CONTROLS</b>"));

    auto viewRow = new QHBoxLayout();
    auto btnXY = new QPushButton("XY");
    auto btnXZ = new QPushButton("XZ");
    auto btnIso = new QPushButton("ISO");
    connect(btnXY, &QPushButton::clicked, this, &ControlPanel::viewXYRequested);
    connect(btnXZ, &QPushButton::clicked, this, &ControlPanel::viewXZRequested);
    connect(btnIso, &QPushButton::clicked, this, &ControlPanel::viewIsoRequested);
    viewRow->addWidget(btnXY);
    viewRow->addWidget(btnXZ);
    viewRow->addWidget(btnIso);
    mainLay->addLayout(viewRow);

    auto chkParticles = new QCheckBox("Show Particles");
    chkParticles->setChecked(true);
    connect(chkParticles, &QCheckBox::toggled, this, &ControlPanel::showParticlesChanged);
    mainLay->addWidget(chkParticles);

    auto chkPotential = new QCheckBox("Show Potential Field");
    chkPotential->setChecked(true);
    connect(chkPotential, &QCheckBox::toggled, this, &ControlPanel::showPotentialChanged);
    mainLay->addWidget(chkPotential);

    auto chkTemp = new QCheckBox("Show Temperature");
    connect(chkTemp, &QCheckBox::toggled, this, &ControlPanel::showTemperatureChanged);
    mainLay->addWidget(chkTemp);

    auto chkDamage = new QCheckBox("Show Damage Map");
    connect(chkDamage, &QCheckBox::toggled, this, &ControlPanel::showDamageChanged);
    mainLay->addWidget(chkDamage);

    auto sliceRow = new QHBoxLayout();
    sliceRow->addWidget(new QLabel("Slice:"));
    auto comboSlice = new QComboBox();
    comboSlice->addItems({"XY (Z-slice)", "XZ (Y-slice)", "YZ (X-slice)"});
    connect(comboSlice, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ControlPanel::slicePlaneChanged);
    sliceRow->addWidget(comboSlice);
    mainLay->addLayout(sliceRow);

    QDoubleSpinBox* spinSlice;
    row = createSpinRow("Slice Position:", spinSlice, 0, 1, 0.5, 0.05, 2);
    connect(spinSlice, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ControlPanel::slicePositionChanged);
    mainLay->addLayout(row);

    // Cut plane controls
    auto chkCut = new QCheckBox("Cut Plane (Half Section)");
    connect(chkCut, &QCheckBox::toggled, this, &ControlPanel::cutPlaneToggled);
    mainLay->addWidget(chkCut);

    auto cutAxisRow = new QHBoxLayout();
    cutAxisRow->addWidget(new QLabel("Cut Axis:"));
    auto comboCutAxis = new QComboBox();
    comboCutAxis->addItems({"X", "Y", "Z"});
    connect(comboCutAxis, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ControlPanel::cutAxisChanged);
    cutAxisRow->addWidget(comboCutAxis);
    mainLay->addLayout(cutAxisRow);

    mainLay->addSpacing(10);

    // ── DIAGNOSTICS ────────────────────────────────────────────────────
    mainLay->addWidget(new QLabel("<b>DIAGNOSTICS</b>"));
    lblIteration_  = new QLabel("Iteration: 0");
    lblIonCount_   = new QLabel("Ions: 0");
    lblElecCount_  = new QLabel("Electrons: 0");
    lblDivergence_ = new QLabel("Beam Div: --");
    lblSaddlePot_  = new QLabel("Saddle Pot: --");
    lblMeanEnergy_ = new QLabel("Mean E: --");
    lblGridTemps_  = new QLabel("Grid Temps: --");

    mainLay->addWidget(lblIteration_);
    mainLay->addWidget(lblIonCount_);
    mainLay->addWidget(lblElecCount_);
    mainLay->addWidget(lblDivergence_);
    mainLay->addWidget(lblSaddlePot_);
    mainLay->addWidget(lblMeanEnergy_);
    mainLay->addWidget(lblGridTemps_);

    mainLay->addStretch();

    scrollArea->setWidget(panel);

    auto outerLay = new QVBoxLayout(this);
    outerLay->setContentsMargins(0, 0, 0, 0);
    outerLay->addWidget(scrollArea);
}

SimParams ControlPanel::getParams() const {
    SimParams p;

    // Grid optics
    for (const auto& gw : gridWidgets_) {
        GridOptic g;
        g.voltage_V     = gw.voltage->value();
        g.thickness_mm  = gw.thickness->value();
        g.gap_mm        = gw.gap->value();
        g.hole_radius_mm = gw.radius->value();
        g.chamfer_deg   = gw.chamfer->value();
        p.grids.push_back(g);
    }

    // Plasma
    p.plasmaDensity   = spinPlasmaDens_->value();
    p.electronTemp_eV = spinTeUp_->value();
    p.ionTemp_eV      = spinTi_->value();
    p.neutralTemp_K   = spinTn_->value();
    p.neutralDensity  = spinNeutralDens_->value();
    p.accelFactor     = spinAccel_->value();
    p.cellFailThreshold = spinThresh_->value();

    // RF
    p.rfEnable      = chkRF_->isChecked();
    p.rfGridIndex   = comboRFGrid_->currentIndex();
    p.rfFreqMHz     = spinRFFreq_->value();
    p.rfAmplitudeV  = spinRFAmp_->value();

    // Neutralizer
    p.neutRate         = spinNeutRate_->value();
    p.neutElectronTemp = spinNeutTemp_->value();

    // Domain (physical values — scaling applied below)
    p.Lx = spinLx_->value();
    p.Ly = spinLy_->value();
    p.Lz = spinLz_->value();
    p.dx = spinDx_->value();
    p.dy = spinDx_->value();
    p.dz = spinDx_->value();

    // Sim mode
    int modeIdx = comboSimMode_->currentIndex();
    p.simMode = (modeIdx == 0) ? SimParams::Both
              : (modeIdx == 1) ? SimParams::Thermal
              : SimParams::Erosion;

    p.dt = spinDt_->value() * 1e-9; // convert ns → s

    // ── Dimensional scaling for Debye-length resolution ─────────────────
    // Self-similar PIC scaling: divide all lengths by s, divide all potentials
    // by s to preserve E = -∇V on particles, divide dt by s for CFL, and
    // multiply plasma/neutral densities by s² so λ_D ∝ 1/sqrt(n) also shrinks
    // with the geometry. Temperatures stay at their physical values.
    double s = 1.0;
    if (comboDimScale_->currentIndex() == 1) s = 10.0;
    else if (comboDimScale_->currentIndex() == 2) s = 100.0;

    if (s != 1.0) {
        const double invS = 1.0 / s;
        const double s2   = s * s;

        // Domain & cells
        p.Lx *= invS; p.Ly *= invS; p.Lz *= invS;
        p.dx *= invS; p.dy *= invS; p.dz *= invS;

        // Grid optics (lengths + voltages)
        for (auto& g : p.grids) {
            g.thickness_mm   *= invS;
            g.gap_mm         *= invS;
            g.hole_radius_mm *= invS;
            g.voltage_V      *= invS;
            // chamfer_deg is an angle — unchanged
        }

        // Other potentials
        p.plasmaOffset_V *= invS;
        p.rfAmplitudeV   *= invS;

        // Neutralizer position and radius
        p.neutX_mm *= invS;
        p.neutR_mm *= invS;

        // Densities: λ_D ∝ 1/sqrt(n), so n × s² scales λ_D by 1/s
        p.plasmaDensity  *= s2;
        p.neutralDensity *= s2;

        // CFL: smaller cells need proportionally smaller dt
        p.dt *= invS;
    }

    return p;
}

int ControlPanel::stepsPerFrame() const {
    return spinBatchSteps_->value();
}

void ControlPanel::updateDiagnostics(int iteration, int ionCount, int electronCount,
                                      double divergence, double saddlePot,
                                      double meanEnergy,
                                      const std::vector<double>& gridTemps) {
    lblIteration_->setText(QString("Iteration: %1").arg(iteration));
    lblIonCount_->setText(QString("Ions: %1").arg(ionCount));
    lblElecCount_->setText(QString("Electrons: %1").arg(electronCount));
    lblDivergence_->setText(QString("Beam Div: %1 deg").arg(divergence, 0, 'f', 2));
    lblSaddlePot_->setText(QString("Saddle Pot: %1 V").arg(saddlePot, 0, 'f', 1));
    lblMeanEnergy_->setText(QString("Mean E: %1 eV").arg(meanEnergy, 0, 'f', 1));

    QString tempStr = "Grid Temps: ";
    for (size_t i = 0; i < gridTemps.size(); i++) {
        if (i > 0) tempStr += ", ";
        tempStr += QString("G%1=%2K").arg(i + 1).arg(gridTemps[i], 0, 'f', 0);
    }
    lblGridTemps_->setText(tempStr);
}

} // namespace BEMCS
