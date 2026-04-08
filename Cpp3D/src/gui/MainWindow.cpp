#include "gui/MainWindow.h"
#include "gui/MeshingDialog.h"
#include "gui/GeometryImportDialog.h"

#include <QHBoxLayout>
#include <QMenuBar>
#include <QStatusBar>
#include <QMessageBox>
#include <QFileDialog>
#include <QApplication>
#include <fstream>
#include <cmath>

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

    // ── Simulation timer ───────────────────────────────────────────────
    simTimer_ = new QTimer(this);
    connect(simTimer_, &QTimer::timeout, this, &MainWindow::onSimStep);

    setupMenuBar();
    createStatusBar();
}

void MainWindow::setupMenuBar() {
    auto menubar = menuBar();

    auto fileMenu = menubar->addMenu("&File");
    fileMenu->addAction("Import STEP...", this, &MainWindow::onImportGeometry);
    fileMenu->addAction("Export Data...", this, &MainWindow::onExportData);
    fileMenu->addSeparator();
    fileMenu->addAction("E&xit", qApp, &QApplication::quit, QKeySequence::Quit);

    auto simMenu = menubar->addMenu("&Simulation");
    simMenu->addAction("Build Domain", this, &MainWindow::onBuildDomain,
                        QKeySequence("Ctrl+B"));
    simMenu->addAction("Reset", this, &MainWindow::onReset, QKeySequence("Ctrl+R"));

    auto meshMenu = menubar->addMenu("&Mesh");
    meshMenu->addAction("Meshing Options...", this, &MainWindow::onMeshSettings);

    auto helpMenu = menubar->addMenu("&Help");
    helpMenu->addAction("About PYBEMCS-3D", [this]() {
        QMessageBox::about(this, "About PYBEMCS-3D",
            "PYBEMCS-3D v1.0\n\n"
            "3D Particle-In-Cell simulation for\n"
            "beam extraction & sputtering/erosion physics.\n\n"
            "Based on PYBEMCS by University of Liverpool.\n\n"
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

void MainWindow::onBuildDomain() {
    currentParams_ = controlPanel_->getParams();

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

    auto stats = meshGen_.getStats(simulator_.getGrid());
    statusBar()->showMessage(
        QString("Domain built: %1x%2x%3 cells (%4 total, %5 boundary)")
            .arg(simulator_.getGrid().nx)
            .arg(simulator_.getGrid().ny)
            .arg(simulator_.getGrid().nz)
            .arg(stats.numVertices)
            .arg(stats.numTriangles));
}

void MainWindow::onStartStop(bool running) {
    if (running) {
        if (simulator_.getGrid().totalCells == 0) {
            QMessageBox::warning(this, "No Domain",
                                  "Build the domain first!");
            return;
        }
        currentParams_ = controlPanel_->getParams();
        simTimer_->start(1); // As fast as possible
        statusBar()->showMessage("Simulation running...");
    } else {
        simTimer_->stop();
        statusBar()->showMessage("Simulation paused.");
    }
}

void MainWindow::onSimStep() {
    currentParams_ = controlPanel_->getParams();
    simulator_.step(currentParams_);

    // Update visualization every 5 iterations for performance
    if (simulator_.getIteration() % 5 == 0) {
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
        double m_e = M_XE / 1000.0;
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

} // namespace BEMCS
