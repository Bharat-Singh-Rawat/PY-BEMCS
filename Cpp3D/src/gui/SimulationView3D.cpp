#include "gui/SimulationView3D.h"

#ifdef USE_VTK
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkStructuredGrid.h>
#include <vtkDataSetMapper.h>
#include <vtkPlane.h>
#include <vtkCutter.h>
#include <vtkThreshold.h>
#include <vtkGeometryFilter.h>
#include <vtkCamera.h>
#include <vtkTextProperty.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>
#endif

namespace BEMCS {

#ifdef USE_VTK

SimulationView3D::SimulationView3D(QWidget* parent)
    : QVTKOpenGLNativeWidget(parent) {
    setupPipeline();
}

void SimulationView3D::setupPipeline() {
    auto renWin = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    setRenderWindow(renWin);

    renderer_ = vtkSmartPointer<vtkRenderer>::New();
    renderer_->SetBackground(0.1, 0.1, 0.15);
    renderer_->SetBackground2(0.2, 0.2, 0.3);
    renderer_->GradientBackgroundOn();
    renWin->AddRenderer(renderer_);

    // Lookup table for potential / temperature coloring
    lut_ = vtkSmartPointer<vtkLookupTable>::New();
    lut_->SetHueRange(0.667, 0.0); // Blue to red
    lut_->SetNumberOfTableValues(256);
    lut_->Build();

    // Color bar
    colorBar_ = vtkSmartPointer<vtkScalarBarActor>::New();
    colorBar_->SetLookupTable(lut_);
    colorBar_->SetTitle("Potential (V)");
    colorBar_->GetTitleTextProperty()->SetFontSize(12);
    colorBar_->SetNumberOfLabels(5);
    colorBar_->SetPosition(0.85, 0.1);
    colorBar_->SetWidth(0.1);
    colorBar_->SetHeight(0.8);
    renderer_->AddActor2D(colorBar_);

    // Initialize empty actors
    gridActor_ = vtkSmartPointer<vtkActor>::New();
    particleActor_ = vtkSmartPointer<vtkActor>::New();
    sliceActor_ = vtkSmartPointer<vtkActor>::New();

    renderer_->AddActor(gridActor_);
    renderer_->AddActor(particleActor_);
    renderer_->AddActor(sliceActor_);

    resetCamera();
}

void SimulationView3D::updateFromSimulator(const Simulator3D& sim,
                                            const SimParams& params) {
    const Grid3D& grid = sim.getGrid();
    if (grid.totalCells == 0) return;

    // ── Update grid geometry (boundary surfaces) ───────────────────────
    if (showGrid_) {
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto polyData = vtkSmartPointer<vtkPolyData>::New();

        // Extract boundary cell centers as points (simplified surface)
        for (int iz = 0; iz < grid.nz; iz++) {
            for (int iy = 0; iy < grid.ny; iy++) {
                for (int ix = 0; ix < grid.nx; ix++) {
                    size_t id = grid.idx(ix, iy, iz);
                    if (grid.isBound[id]) {
                        // Only show surface cells (cells with at least one non-bound neighbor)
                        bool isSurface = false;
                        if (ix > 0 && !grid.isBound[grid.idx(ix-1,iy,iz)]) isSurface = true;
                        if (ix < grid.nx-1 && !grid.isBound[grid.idx(ix+1,iy,iz)]) isSurface = true;
                        if (iy > 0 && !grid.isBound[grid.idx(ix,iy-1,iz)]) isSurface = true;
                        if (iy < grid.ny-1 && !grid.isBound[grid.idx(ix,iy+1,iz)]) isSurface = true;
                        if (iz > 0 && !grid.isBound[grid.idx(ix,iy,iz-1)]) isSurface = true;
                        if (iz < grid.nz-1 && !grid.isBound[grid.idx(ix,iy,iz+1)]) isSurface = true;

                        if (isSurface) {
                            points->InsertNextPoint(ix * grid.dx, iy * grid.dy,
                                                    iz * grid.dz);
                        }
                    }
                }
            }
        }

        polyData->SetPoints(points);

        auto glyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        glyphFilter->SetInputData(polyData);
        glyphFilter->Update();

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(glyphFilter->GetOutputPort());

        gridActor_->SetMapper(mapper);
        gridActor_->GetProperty()->SetColor(0.7, 0.7, 0.7);
        gridActor_->GetProperty()->SetPointSize(3);
        gridActor_->GetProperty()->SetOpacity(0.5);
        gridActor_->SetVisibility(true);
    } else {
        gridActor_->SetVisibility(false);
    }

    // ── Update particles ───────────────────────────────────────────────
    if (showParticles_) {
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors->SetNumberOfComponents(3);
        colors->SetName("ParticleColors");

        const auto& ions = sim.getIons();
        const auto& elecs = sim.getElectrons();

        // Sample particles if too many (for rendering performance)
        size_t maxRender = 50000;
        size_t ionStep = std::max((size_t)1, ions.count / maxRender);
        size_t elecStep = std::max((size_t)1, elecs.count / maxRender);

        for (size_t i = 0; i < ions.count; i += ionStep) {
            if (!ions.alive[i]) continue;
            points->InsertNextPoint(ions.x[i], ions.y[i], ions.z[i]);

            if (ions.species[i] == Species::CEX_Ion) {
                unsigned char red[3] = {220, 50, 50};
                colors->InsertNextTypedTuple(red);
            } else {
                unsigned char blue[3] = {50, 100, 220};
                colors->InsertNextTypedTuple(blue);
            }
        }

        for (size_t i = 0; i < elecs.count; i += elecStep) {
            if (!elecs.alive[i]) continue;
            points->InsertNextPoint(elecs.x[i], elecs.y[i], elecs.z[i]);
            unsigned char green[3] = {50, 200, 100};
            colors->InsertNextTypedTuple(green);
        }

        auto polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData->SetPoints(points);
        polyData->GetPointData()->SetScalars(colors);

        auto glyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        glyphFilter->SetInputData(polyData);
        glyphFilter->Update();

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(glyphFilter->GetOutputPort());

        particleActor_->SetMapper(mapper);
        particleActor_->GetProperty()->SetPointSize(2);
        particleActor_->SetVisibility(true);
    } else {
        particleActor_->SetVisibility(false);
    }

    // ── Update potential/temperature slice ──────────────────────────────
    if (showPotential_ || showTemperature_) {
        const auto& field = showTemperature_ ? grid.T_map : grid.V;
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto scalars = vtkSmartPointer<vtkFloatArray>::New();
        scalars->SetName(showTemperature_ ? "Temperature" : "Potential");

        // Extract a 2D slice through the domain
        int sliceIdx = 0;
        if (slicePlane_ == SlicePlane::XY) {
            sliceIdx = static_cast<int>(slicePos_ * (grid.nz - 1));
            for (int iy = 0; iy < grid.ny; iy++) {
                for (int ix = 0; ix < grid.nx; ix++) {
                    points->InsertNextPoint(ix * grid.dx, iy * grid.dy,
                                            sliceIdx * grid.dz);
                    scalars->InsertNextValue(
                        static_cast<float>(field[grid.idx(ix, iy, sliceIdx)]));
                }
            }
        } else if (slicePlane_ == SlicePlane::XZ) {
            sliceIdx = static_cast<int>(slicePos_ * (grid.ny - 1));
            for (int iz = 0; iz < grid.nz; iz++) {
                for (int ix = 0; ix < grid.nx; ix++) {
                    points->InsertNextPoint(ix * grid.dx, sliceIdx * grid.dy,
                                            iz * grid.dz);
                    scalars->InsertNextValue(
                        static_cast<float>(field[grid.idx(ix, sliceIdx, iz)]));
                }
            }
        } else { // YZ
            sliceIdx = static_cast<int>(slicePos_ * (grid.nx - 1));
            for (int iz = 0; iz < grid.nz; iz++) {
                for (int iy = 0; iy < grid.ny; iy++) {
                    points->InsertNextPoint(sliceIdx * grid.dx, iy * grid.dy,
                                            iz * grid.dz);
                    scalars->InsertNextValue(
                        static_cast<float>(field[grid.idx(sliceIdx, iy, iz)]));
                }
            }
        }

        auto polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData->SetPoints(points);
        polyData->GetPointData()->SetScalars(scalars);

        auto glyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        glyphFilter->SetInputData(polyData);
        glyphFilter->Update();

        double range[2];
        scalars->GetRange(range);
        lut_->SetRange(range[0], range[1]);

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(glyphFilter->GetOutputPort());
        mapper->SetLookupTable(lut_);
        mapper->SetScalarRange(range[0], range[1]);

        sliceActor_->SetMapper(mapper);
        sliceActor_->GetProperty()->SetPointSize(4);
        sliceActor_->SetVisibility(true);

        colorBar_->SetTitle(showTemperature_ ? "Temperature (K)" : "Potential (V)");
        colorBar_->SetVisibility(true);
    } else {
        sliceActor_->SetVisibility(false);
        colorBar_->SetVisibility(false);
    }

    renderWindow()->Render();
}

void SimulationView3D::resetCamera() {
    renderer_->ResetCamera();
    renderWindow()->Render();
}

void SimulationView3D::setViewXY() {
    auto cam = renderer_->GetActiveCamera();
    cam->SetPosition(0, 0, 100);
    cam->SetFocalPoint(0, 0, 0);
    cam->SetViewUp(0, 1, 0);
    renderer_->ResetCamera();
    renderWindow()->Render();
}

void SimulationView3D::setViewXZ() {
    auto cam = renderer_->GetActiveCamera();
    cam->SetPosition(0, 100, 0);
    cam->SetFocalPoint(0, 0, 0);
    cam->SetViewUp(0, 0, 1);
    renderer_->ResetCamera();
    renderWindow()->Render();
}

void SimulationView3D::setViewIso() {
    auto cam = renderer_->GetActiveCamera();
    cam->SetPosition(50, 30, 40);
    cam->SetFocalPoint(0, 0, 0);
    cam->SetViewUp(0, 1, 0);
    renderer_->ResetCamera();
    renderWindow()->Render();
}

#else // No VTK fallback

SimulationView3D::SimulationView3D(QWidget* parent)
    : QOpenGLWidget(parent) {}

void SimulationView3D::updateFromSimulator(const Simulator3D& sim,
                                            const SimParams&) {
    lastSim_ = &sim;
    update();
}

void SimulationView3D::resetCamera() { update(); }
void SimulationView3D::setViewXY() { update(); }
void SimulationView3D::setViewXZ() { update(); }
void SimulationView3D::setViewIso() { update(); }

void SimulationView3D::initializeGL() {
    // Basic OpenGL setup
}

void SimulationView3D::paintGL() {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Minimal fallback rendering - displays message
}

void SimulationView3D::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

#endif

} // namespace BEMCS
