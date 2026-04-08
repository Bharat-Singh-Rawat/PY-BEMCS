// ============================================================================
// PYBEMCS-3D: 3D Particle-In-Cell Beam Extraction & Monte Carlo Simulation
//
// A full 3D extension of the PYBEMCS code for studying beam extraction
// and sputtering/erosion physics in ion thruster grids.
//
// Features:
//   - 3D PIC solver with Boris particle pusher
//   - Conjugate Gradient Poisson solver with Boltzmann electrons
//   - CEX collisions, sputtering/erosion, thermal conduction
//   - STEP file geometry import via OpenCASCADE
//   - Interactive Qt6 GUI with VTK 3D visualization
//   - OpenMP parallel acceleration
//
// University of Liverpool
// ============================================================================

#include "gui/MainWindow.h"
#include <QApplication>
#include <QSurfaceFormat>

#ifdef USE_VTK
#include <QVTKOpenGLNativeWidget.h>
#include <vtkOpenGLRenderWindow.h>
#endif

int main(int argc, char* argv[]) {
#ifdef USE_VTK
    // VTK requires this before QApplication
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    vtkOpenGLRenderWindow::SetGlobalMaximumNumberOfMultiSamples(0);
#endif

    QApplication app(argc, argv);
    app.setApplicationName("PYBEMCS-3D");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("University of Liverpool");

    // Apply dark theme stylesheet
    app.setStyleSheet(R"(
        QMainWindow { background-color: #2b2b2b; }
        QWidget { color: #e0e0e0; background-color: #2b2b2b; }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 12px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        QLabel { background-color: transparent; }
        QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
            color: #e0e0e0;
        }
        QPushButton {
            background-color: #404040;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px 12px;
            color: #e0e0e0;
        }
        QPushButton:hover { background-color: #505050; }
        QPushButton:pressed { background-color: #606060; }
        QCheckBox { background-color: transparent; }
        QScrollArea { border: none; }
        QScrollBar:vertical {
            background: #2b2b2b;
            width: 12px;
        }
        QScrollBar::handle:vertical {
            background: #555;
            border-radius: 4px;
            min-height: 20px;
        }
        QMenuBar { background-color: #333; color: #e0e0e0; }
        QMenuBar::item:selected { background-color: #505050; }
        QMenu { background-color: #333; color: #e0e0e0; }
        QMenu::item:selected { background-color: #2196F3; }
        QStatusBar { background-color: #333; color: #aaa; }
    )");

    BEMCS::MainWindow window;
    window.show();

    return app.exec();
}
