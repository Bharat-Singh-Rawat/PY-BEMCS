#pragma once

#include "core/Simulator3D.h"

#ifdef USE_VTK
#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkActor.h>
#include <vtkScalarBarActor.h>
#include <vtkLookupTable.h>
#else
#include <QOpenGLWidget>
#endif

#include <QWidget>

namespace BEMCS {

// ============================================================================
// 3D Visualization Widget using VTK (or fallback OpenGL)
//
// Renders:
// - Grid geometry (boundary cells as transparent surface)
// - Particles (ions = blue, CEX = red, electrons = green)
// - Potential field (color-mapped slice or isosurface)
// - Temperature field on grid surfaces
// - Damage/erosion map
// ============================================================================

#ifdef USE_VTK

class SimulationView3D : public QVTKOpenGLNativeWidget {
    Q_OBJECT
public:
    explicit SimulationView3D(QWidget* parent = nullptr);

    // Update visualization from current sim state
    void updateFromSimulator(const Simulator3D& sim, const SimParams& params);

    // View controls
    void resetCamera();
    void setViewXY();  // Top view
    void setViewXZ();  // Side view
    void setViewIso(); // Isometric

    // Display toggles
    void setShowParticles(bool show) { showParticles_ = show; }
    void setShowPotential(bool show) { showPotential_ = show; }
    void setShowTemperature(bool show) { showTemperature_ = show; }
    void setShowDamage(bool show) { showDamage_ = show; }
    void setShowGrid(bool show) { showGrid_ = show; }

    enum class SlicePlane { XY, XZ, YZ };
    void setSlicePlane(SlicePlane plane) { slicePlane_ = plane; }
    void setSlicePosition(double pos) { slicePos_ = pos; }

private:
    void setupPipeline();

    vtkSmartPointer<vtkRenderer> renderer_;

    // Geometry actors
    vtkSmartPointer<vtkActor> gridActor_;
    vtkSmartPointer<vtkActor> particleActor_;
    vtkSmartPointer<vtkActor> sliceActor_;
    vtkSmartPointer<vtkScalarBarActor> colorBar_;
    vtkSmartPointer<vtkLookupTable> lut_;

    bool showParticles_ = true;
    bool showPotential_ = true;
    bool showTemperature_ = false;
    bool showDamage_ = false;
    bool showGrid_ = true;

    SlicePlane slicePlane_ = SlicePlane::XY;
    double slicePos_ = 0.5; // Normalized [0,1]
};

#else

// Fallback: simple OpenGL widget
class SimulationView3D : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit SimulationView3D(QWidget* parent = nullptr);
    void updateFromSimulator(const Simulator3D& sim, const SimParams& params);
    void resetCamera();
    void setViewXY();
    void setViewXZ();
    void setViewIso();
    void setShowParticles(bool) {}
    void setShowPotential(bool) {}
    void setShowTemperature(bool) {}
    void setShowDamage(bool) {}
    void setShowGrid(bool) {}

    enum class SlicePlane { XY, XZ, YZ };
    void setSlicePlane(SlicePlane) {}
    void setSlicePosition(double) {}

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

private:
    const Simulator3D* lastSim_ = nullptr;
};

#endif

} // namespace BEMCS
