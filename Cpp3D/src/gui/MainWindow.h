#pragma once
#include "core/Simulator3D.h"
#include "geometry/MeshGenerator.h"
#include "geometry/Mesh3D.h"
#include "gui/SimulationView3D.h"
#include "gui/ControlPanel.h"

#include <QMainWindow>
#include <QTimer>
#include <memory>

namespace BEMCS {

// ============================================================================
// Main Application Window
//
// Combines the 3D visualization, control panel, and connects all
// GUI signals to the simulator.
// ============================================================================
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() = default;

private slots:
    void onBuildDomain();
    void onStartStop(bool running);
    void onImportGeometry();
    void onMeshSettings();
    void onReset();
    void onExportData();
    void onSimStep();

private:
    void setupMenuBar();
    void createStatusBar();

    // Core simulation
    Simulator3D simulator_;
    SimParams currentParams_;
    MeshGenerator meshGen_;
    MeshGenerator::MeshSettings meshSettings_;

    // Imported geometry
    SurfaceMesh importedGeometry_;

    // GUI components
    SimulationView3D* view3D_;
    ControlPanel* controlPanel_;
    QTimer* simTimer_;
};

} // namespace BEMCS
