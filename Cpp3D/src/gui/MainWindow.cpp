#include "gui/MainWindow.h"
#include "gui/MeshingDialog.h"
#include "gui/GeometryImportDialog.h"

#include <QHBoxLayout>
#include <QMenuBar>
#include <QStatusBar>
#include <QMessageBox>
#include <QFileDialog>
#include <QApplication>
#include <QAction>
#include <QTime>
#include <fstream>
#include <cmath>
#include "core/Constants.h"

#ifdef USE_VTK
#include <vtkWindowToImageFilter.h>
#include <vtkRenderWindow.h>
#endif

namespace BEMCS {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("PYBEMCS-3D  —  3D Beam Extraction & Monte Carlo Simulation");
    setGeometry(20, 30, 1600, 900);

    // ── Central layout: [ControlPanel | 3D View] ──────────────────────
    auto central = new QWidget();
    setCentralWidget(central);
    auto mainLay = new QHBoxLayout(central);

    controlPanel_ = new ControlPanel();
    view3D_ = new SimulationView3D();

    mainLay->addWidget(controlPanel_);
    mainLay->addWidget(view3D_, 1); // Stretch factor for 3D view

    // ── Connect signals ────────────────────────────────────────────────
    connect(controlPanel_, &ControlPanel::buildDomainRequested,
            this, &MainWindow::onBuildDomain);
    connect(controlPanel_, &ControlPanel::startStopToggled,
            this, &MainWindow::onStartStop);
    connect(controlPanel_, &ControlPanel::importGeometryRequested,
            this, &MainWindow::onImportGeometry);
    connect(controlPanel_, &ControlPanel::meshSettingsRequested,
            this, &MainWindow::onMeshSettings);
    connect(controlPanel_, &ControlPanel::resetRequested,
            this, &MainWindow::onReset);
    connect(controlPanel_, &ControlPanel::exportDataRequested,
            this, &MainWindow::onExportData);

    // View controls
    connect(controlPanel_, &ControlPanel::viewXYRequested,
            view3D_, &SimulationView3D::setViewXY);
    connect(controlPanel_, &ControlPanel::viewXZRequested,
            view3D_, &SimulationView3D::setViewXZ);
    connect(controlPanel_, &ControlPanel::viewIsoRequested,
            view3D_, &SimulationView3D::setViewIso);
    connect(controlPanel_, &ControlPanel::showParticlesChanged,
            view3D_, &SimulationView3D::setShowParticles);
    connect(controlPanel_, &ControlPanel::showPotentialChanged,
            view3D_, &SimulationView3D::setShowPotential);
    connect(controlPanel_, &ControlPanel::showTemperatureChanged,
            view3D_, &SimulationView3D::setShowTemperature);
    connect(controlPanel_, &ControlPanel::showDamageChanged,
            view3D_, &SimulationView3D::setShowDamage);

    connect(controlPanel_, &ControlPanel::slicePlaneChanged,
            [this](int idx) {
                view3D_->setSlicePlane(
                    static_cast<SimulationView3D::SlicePlane>(idx));
            });
    connect(controlPanel_, &ControlPanel::slicePositionChanged,
            view3D_, &SimulationView3D::setSlicePosition);
    connect(controlPanel_, &ControlPanel::cutPlaneToggled,
            [this](bool on) {
                view3D_->setCutPlaneEnabled(on);
                sputterView_->setCutPlaneEnabled(on);
                thermalView_->setCutPlaneEnabled(on);
            });
    connect(controlPanel_, &ControlPanel::cutAxisChanged,
            [this](int idx) {
                auto axis = static_cast<SimulationView3D::CutAxis>(idx);
                view3D_->setCutAxis(axis);
                sputterView_->setCutAxis(axis);
                thermalView_->setCutAxis(axis);
            });

    // ── Simulation timer ───────────────────────────────────────────────
    simTimer_ = new QTimer(this);
    connect(simTimer_, &QTimer::timeout, this, &MainWindow::onSimStep);

    setupMenuBar();
    createStatusBar();
    setupLogWindow();
    setupContourDocks();

    logMessage("PYBEMCS-3D initialized. Ready.");
}

void MainWindow::setupMenuBar() {
    auto menubar = menuBar();

    auto fileMenu = menubar->addMenu("&File");
    fileMenu->addAction("Import STEP...", this, &MainWindow::onImportGeometry);
    fileMenu->addAction("Export Data...", this, &MainWindow::onExportData);
    fileMenu->addSeparator();
    gifRecordAction_ = fileMenu->addAction("Record GIF", this,
                                            &MainWindow::onToggleGifRecording,
                                            QKeySequence("Ctrl+G"));
    gifRecordAction_->setCheckable(true);
    fileMenu->addAction("Save GIF...", this, &MainWindow::onSaveGif,
                         QKeySequence("Ctrl+Shift+G"));
    fileMenu->addSeparator();
    fileMenu->addAction("E&xit", qApp, &QApplication::quit, QKeySequence::Quit);

    auto simMenu = menubar->addMenu("&Simulation");
    simMenu->addAction("Build Domain", this, &MainWindow::onBuildDomain,
                        QKeySequence("Ctrl+B"));
    simMenu->addAction("Reset", this, &MainWindow::onReset, QKeySequence("Ctrl+R"));

    auto meshMenu = menubar->addMenu("&Mesh");
    meshMenu->addAction("Meshing Options...", this, &MainWindow::onMeshSettings);

    auto viewMenu = menubar->addMenu("&View");
    auto cutAction = viewMenu->addAction("Cut Plane (Half Section)");
    cutAction->setCheckable(true);
    connect(cutAction, &QAction::toggled, [this](bool on) {
        view3D_->setCutPlaneEnabled(on);
        if (sputterView_) sputterView_->setCutPlaneEnabled(on);
        if (thermalView_) thermalView_->setCutPlaneEnabled(on);
        logMessage(on ? "Cut plane enabled" : "Cut plane disabled");
    });
    auto cutXAction = viewMenu->addAction("Cut Along X");
    connect(cutXAction, &QAction::triggered, [this]() {
        view3D_->setCutAxis(SimulationView3D::CutAxis::X);
    });
    auto cutYAction = viewMenu->addAction("Cut Along Y");
    connect(cutYAction, &QAction::triggered, [this]() {
        view3D_->setCutAxis(SimulationView3D::CutAxis::Y);
    });
    auto cutZAction = viewMenu->addAction("Cut Along Z");
    connect(cutZAction, &QAction::triggered, [this]() {
        view3D_->setCutAxis(SimulationView3D::CutAxis::Z);
    });
    viewMenu->addSeparator();
    viewMenu->addAction("Show Log Window", [this]() { logDock_->show(); });
    viewMenu->addAction("Show Sputtering Map", [this]() { sputterDock_->show(); });
    viewMenu->addAction("Show Thermal Map", [this]() { thermalDock_->show(); });

    auto helpMenu = menubar->addMenu("&Help");
    helpMenu->addAction("About PYBEMCS-3D", [this]() {
        QMessageBox::about(this, "About PYBEMCS-3D",
            "PYBEMCS-3D v1.0\n\n"
            "3D Particle-In-Cell simulation for\n"
            "beam extraction & sputtering/erosion physics.\n\n"
            "Developed by Dr. Bharat Singh Rawat.\n\n"
            "Features:\n"
            "- Full 3D PIC solver with Boris pusher\n"
            "- Conjugate Gradient Poisson solver\n"
            "- CEX collisions & sputtering model\n"
            "- STEP file geometry import (OpenCASCADE)\n"
            "- VTK-based 3D visualization\n"
            "- OpenMP parallel acceleration");
    });
}

void MainWindow::createStatusBar() {
    statusBar()->showMessage("Ready. Build domain to begin.");
}

void MainWindow::setupLogWindow() {
    logDock_ = new QDockWidget("Process Log", this);
    logDock_->setAllowedAreas(Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);

    logWindow_ = new QTextEdit();
    logWindow_->setReadOnly(true);
    logWindow_->setMaximumHeight(180);
    logWindow_->setStyleSheet(
        "QTextEdit { background: #1a1a2e; color: #e0e0e0; "
        "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; }");

    logDock_->setWidget(logWindow_);
    addDockWidget(Qt::BottomDockWidgetArea, logDock_);
}

void MainWindow::setupContourDocks() {
    // Sputtering contour map dock
    sputterDock_ = new QDockWidget("Sputtering / Damage Map", this);
    sputterView_ = new SimulationView3D();
    sputterView_->setShowGrid(false);
    sputterView_->setShowParticles(false);
    sputterView_->setShowDamage(true);
    sputterView_->setShowPotential(false);
    sputterDock_->setWidget(sputterView_);
    addDockWidget(Qt::RightDockWidgetArea, sputterDock_);
    sputterDock_->setMinimumWidth(350);

    // Thermal contour map dock
    thermalDock_ = new QDockWidget("Temperature Map", this);
    thermalView_ = new SimulationView3D();
    thermalView_->setShowGrid(false);
    thermalView_->setShowParticles(false);
    thermalView_->setShowTemperature(true);
    thermalView_->setShowPotential(false);
    thermalDock_->setWidget(thermalView_);
    addDockWidget(Qt::RightDockWidgetArea, thermalDock_);
    thermalDock_->setMinimumWidth(350);

    // Stack the two contour docks
    tabifyDockWidget(sputterDock_, thermalDock_);
    sputterDock_->raise();
}

void MainWindow::logMessage(const QString& msg) {
    if (!logWindow_) return;
    QString timestamp = QTime::currentTime().toString("HH:mm:ss");
    logWindow_->append(QString("[%1] %2").arg(timestamp, msg));
}

void MainWindow::onBuildDomain() {
    currentParams_ = controlPanel_->getParams();

    logMessage("Building 3D domain...");
    statusBar()->showMessage("Building 3D domain...");
    QApplication::processEvents();

    if (!importedGeometry_.empty()) {
        // Build from imported STEP geometry
        bool ok = meshGen_.generateFromSurface(importedGeometry_, meshSettings_,
                                                simulator_.getGrid(), currentParams_);
        if (!ok) {
            QMessageBox::warning(this, "Mesh Error",
                QString::fromStdString(meshGen_.getError()));
            statusBar()->showMessage("Mesh generation failed.");
            return;
        }
        // Still need to solve fields
        PoissonSolver3D poisson;
        poisson.solveWithBoltzmann(simulator_.getGrid(), currentParams_, 30, 500);
    } else {
        // Build from built-in grid parameters
        simulator_.buildDomain(currentParams_);
    }

    view3D_->updateFromSimulator(simulator_, currentParams_);
    view3D_->resetCamera();

    // Update contour views
    sputterView_->updateFromSimulator(simulator_, currentParams_);
    sputterView_->resetCamera();
    thermalView_->updateFromSimulator(simulator_, currentParams_);
    thermalView_->resetCamera();

    auto stats = meshGen_.getStats(simulator_.getGrid());
    QString msg = QString("Domain built: %1x%2x%3 cells (%4 total, %5 boundary)")
            .arg(simulator_.getGrid().nx)
            .arg(simulator_.getGrid().ny)
            .arg(simulator_.getGrid().nz)
            .arg(stats.numVertices)
            .arg(stats.numTriangles);
    statusBar()->showMessage(msg);
    logMessage(msg);
}

void MainWindow::onStartStop(bool running) {
    if (running) {
        if (simulator_.getGrid().totalCells == 0) {
            QMessageBox::warning(this, "No Domain",
                                  "Build the domain first!");
            return;
        }
        currentParams_ = controlPanel_->getParams();
        simTimer_->start(0); // As fast as possible, batched steps below
        statusBar()->showMessage("Simulation running...");
        logMessage("Simulation started.");
    } else {
        simTimer_->stop();
        statusBar()->showMessage("Simulation paused.");
        logMessage("Simulation paused.");
    }
}

void MainWindow::onSimStep() {
    currentParams_ = controlPanel_->getParams();

    // Batch multiple physics steps per render frame to reduce GUI lag
    for (int batch = 0; batch < 10; batch++) {
        simulator_.step(currentParams_);
    }

    // Update visualization every render call
    {
        view3D_->updateFromSimulator(simulator_, currentParams_);

        double div = simulator_.getBeamDivergence(currentParams_);
        double saddle = simulator_.getSaddlePointPotential(currentParams_);
        double meanE = simulator_.getMeanIonEnergy();

        controlPanel_->updateDiagnostics(
            simulator_.getIteration(),
            simulator_.getIonCount(),
            simulator_.getElectronCount(),
            div, saddle, meanE,
            simulator_.getGrid().gridTemps);

        // Update contour map views
        sputterView_->updateFromSimulator(simulator_, currentParams_);
        thermalView_->updateFromSimulator(simulator_, currentParams_);

        // Capture GIF frame every render
        if (gifRecording_) {
            captureGifFrame();
        }

        // Log diagnostics every 100 iterations
        if (simulator_.getIteration() % 100 == 0) {
            logMessage(QString("Step %1: %2 ions, %3 electrons, div=%4 deg, E=%5 eV")
                .arg(simulator_.getIteration())
                .arg(simulator_.getIonCount())
                .arg(simulator_.getElectronCount())
                .arg(div, 0, 'f', 2)
                .arg(meanE, 0, 'f', 1));
        }
    }
}

void MainWindow::onImportGeometry() {
    GeometryImportDialog dialog(this);
    if (dialog.exec() == QDialog::Accepted && dialog.hasValidMesh()) {
        importedGeometry_ = dialog.getMesh();
        statusBar()->showMessage(
            QString("Geometry imported: %1 vertices, %2 triangles. "
                    "Click BUILD DOMAIN to apply.")
                .arg(importedGeometry_.vertices.size())
                .arg(importedGeometry_.triangles.size()));
    }
}

void MainWindow::onMeshSettings() {
    MeshingDialog dialog(meshSettings_, this);

    // Show current stats if we have a grid
    if (simulator_.getGrid().totalCells > 0) {
        dialog.showStats(meshGen_.getStats(simulator_.getGrid()));
    }

    if (dialog.exec() == QDialog::Accepted) {
        meshSettings_ = dialog.getSettings();
        statusBar()->showMessage("Mesh settings updated. Rebuild domain to apply.");
    }
}

void MainWindow::onReset() {
    simTimer_->stop();
    simulator_.reset();
    importedGeometry_.clear();
    view3D_->updateFromSimulator(simulator_, currentParams_);
    controlPanel_->updateDiagnostics(0, 0, 0, 0, 0, 0, {});
    statusBar()->showMessage("Simulation reset.");
    logMessage("Simulation reset. All particles and fields cleared.");
}

void MainWindow::onExportData() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export Particle Data", "particles.csv",
        "CSV Files (*.csv);;All Files (*)");

    if (path.isEmpty()) return;

    std::ofstream file(path.toStdString());
    if (!file.is_open()) {
        QMessageBox::warning(this, "Export Error", "Could not open file for writing.");
        return;
    }

    file << "type,x_mm,y_mm,z_mm,vx_m/s,vy_m/s,vz_m/s,energy_eV\n";

    const auto& ions = simulator_.getIons();
    for (size_t i = 0; i < ions.count; i++) {
        if (!ions.alive[i]) continue;
        double v2 = ions.vx[i]*ions.vx[i] + ions.vy[i]*ions.vy[i] + ions.vz[i]*ions.vz[i];
        double E = 0.5 * M_XE * v2 / Q_E;
        const char* type = (ions.species[i] == Species::CEX_Ion) ? "CEX" : "ION";
        file << type << ","
             << ions.x[i] << "," << ions.y[i] << "," << ions.z[i] << ","
             << ions.vx[i] << "," << ions.vy[i] << "," << ions.vz[i] << ","
             << E << "\n";
    }

    const auto& elecs = simulator_.getElectrons();
    for (size_t i = 0; i < elecs.count; i++) {
        if (!elecs.alive[i]) continue;
        double m_e = M_ELECTRON;
        double v2 = elecs.vx[i]*elecs.vx[i] + elecs.vy[i]*elecs.vy[i] + elecs.vz[i]*elecs.vz[i];
        double E = 0.5 * m_e * v2 / Q_E;
        file << "ELEC,"
             << elecs.x[i] << "," << elecs.y[i] << "," << elecs.z[i] << ","
             << elecs.vx[i] << "," << elecs.vy[i] << "," << elecs.vz[i] << ","
             << E << "\n";
    }

    file.close();
    statusBar()->showMessage("Data exported to " + path);
}

void MainWindow::onToggleGifRecording() {
    gifRecording_ = !gifRecording_;

    if (gifRecording_) {
        // Start recording: initialize the GIF writer with the view size
        int w = view3D_->width();
        int h = view3D_->height();
        // Use 10 centiseconds (100ms) delay between frames
        gifWriter_.init(w, h, 10);
        gifRecordAction_->setText("Stop Recording GIF");
        statusBar()->showMessage("GIF recording started...");
        logMessage("GIF recording started.");
    } else {
        gifRecordAction_->setText("Record GIF");
        statusBar()->showMessage(
            QString("GIF recording stopped. %1 frames captured. Use Save GIF to export.")
                .arg(gifWriter_.frameCount()));
    }
}

void MainWindow::onSaveGif() {
    if (gifWriter_.frameCount() == 0) {
        QMessageBox::information(this, "No Frames",
            "No GIF frames recorded. Start recording (Ctrl+G) during simulation first.");
        return;
    }

    // Stop recording if still active
    if (gifRecording_) {
        gifRecording_ = false;
        gifRecordAction_->setChecked(false);
        gifRecordAction_->setText("Record GIF");
    }

    QString path = QFileDialog::getSaveFileName(
        this, "Save Animated GIF", "simulation.gif",
        "GIF Files (*.gif);;All Files (*)");

    if (path.isEmpty()) return;

    int numFrames = gifWriter_.frameCount();
    statusBar()->showMessage("Saving GIF...");
    logMessage(QString("Saving GIF with %1 frames...").arg(numFrames));
    QApplication::processEvents();

    bool ok = gifWriter_.finish(path.toStdString());

    if (ok) {
        QString msg = QString("GIF saved to %1 (%2 frames)").arg(path).arg(numFrames);
        statusBar()->showMessage(msg);
        logMessage(msg);
    } else {
        QMessageBox::warning(this, "Save Error", "Failed to write GIF file.");
        statusBar()->showMessage("GIF save failed.");
        logMessage("GIF save failed.");
    }
}

void MainWindow::captureGifFrame() {
#ifdef USE_VTK
    auto renderWindow = view3D_->renderWindow();
    if (!renderWindow) return;

    renderWindow->Render();

    auto windowToImage = vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImage->SetInput(renderWindow);
    windowToImage->SetInputBufferTypeToRGB();
    windowToImage->ReadFrontBufferOff();
    windowToImage->Update();

    auto* imageData = windowToImage->GetOutput();
    if (!imageData) return;

    int* dims = imageData->GetDimensions();
    int w = dims[0];
    int h = dims[1];
    if (w <= 0 || h <= 0) return;

    // Reinitialize GIF writer if capture size differs from init size
    // (happens due to DPI scaling: VTK render size != Qt widget size)
    if (gifWriter_.frameCount() == 0) {
        gifWriter_.init(w, h, 10);
    }

    auto* pixels = static_cast<uint8_t*>(
        imageData->GetScalarPointer());
    if (!pixels) return;

    // VTK gives bottom-up rows; flip vertically for GIF (top-down)
    std::vector<uint8_t> flipped(w * h * 3);
    for (int y = 0; y < h; y++) {
        memcpy(&flipped[y * w * 3],
               &pixels[(h - 1 - y) * w * 3],
               w * 3);
    }

    gifWriter_.addFrameRGB(flipped.data());
#else
    // Fallback: capture from QWidget
    QPixmap pixmap = view3D_->grab();
    QImage img = pixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    if (gifWriter_.frameCount() == 0) {
        gifWriter_.init(img.width(), img.height(), 10);
    }

    gifWriter_.addFrame(img.constBits());
#endif
}

} // namespace BEMCS
