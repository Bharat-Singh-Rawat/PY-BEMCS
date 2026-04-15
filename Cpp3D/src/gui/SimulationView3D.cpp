#include "gui/SimulationView3D.h"

#ifdef USE_VTK
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredPoints.h>
#include <vtkDataSetMapper.h>
#include <vtkPlane.h>
#include <vtkCutter.h>
#include <vtkClipPolyData.h>
#include <vtkThreshold.h>
#include <vtkGeometryFilter.h>
#include <vtkCamera.h>
#include <vtkTextProperty.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCellArray.h>
#include <vtkTriangle.h>
#include <vtkOutlineSource.h>
#include <vtkImageData.h>
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

    domainBoxActor_ = vtkSmartPointer<vtkActor>::New();

    renderer_->AddActor(domainBoxActor_);
    renderer_->AddActor(gridActor_);
    renderer_->AddActor(particleActor_);
    renderer_->AddActor(sliceActor_);

    // Cut plane (default: clip at midpoint along Y)
    cutPlane_ = vtkSmartPointer<vtkPlane>::New();

    resetCamera();
}

// Build solid surface mesh from boundary voxels using quad faces
void SimulationView3D::buildSolidGridSurface(const Grid3D& grid) {
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto cells = vtkSmartPointer<vtkCellArray>::New();
    auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("GridColors");

    // For each boundary cell, check each of its 6 faces.
    // If the neighbor on that face is NOT a boundary cell, emit a quad.
    double dx = grid.dx, dy = grid.dy, dz = grid.dz;

    // Direction offsets and quad vertex offsets for each face
    // Face normals: -X, +X, -Y, +Y, -Z, +Z
    int dix[6] = {-1, 1, 0, 0, 0, 0};
    int diy[6] = {0, 0, -1, 1, 0, 0};
    int diz[6] = {0, 0, 0, 0, -1, 1};

    // Quad corner offsets (relative to cell origin at ix,iy,iz)
    // For face pointing in -X direction: quad on the x=ix plane
    double quadVerts[6][4][3] = {
        // -X face (x = ix*dx)
        {{0,0,0}, {0,1,0}, {0,1,1}, {0,0,1}},
        // +X face (x = (ix+1)*dx)
        {{1,0,0}, {1,0,1}, {1,1,1}, {1,1,0}},
        // -Y face
        {{0,0,0}, {0,0,1}, {1,0,1}, {1,0,0}},
        // +Y face
        {{0,1,0}, {1,1,0}, {1,1,1}, {0,1,1}},
        // -Z face
        {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}},
        // +Z face
        {{0,0,1}, {0,1,1}, {1,1,1}, {1,0,1}},
    };

    unsigned char solidColor[3] = {180, 180, 190};
    unsigned char damageColor[3] = {220, 80, 50};

    for (int iz = 0; iz < grid.nz; iz++) {
        for (int iy = 0; iy < grid.ny; iy++) {
            for (int ix = 0; ix < grid.nx; ix++) {
                size_t id = grid.idx(ix, iy, iz);
                if (!grid.isBound[id]) continue;

                for (int face = 0; face < 6; face++) {
                    int nx_ = ix + dix[face];
                    int ny_ = iy + diy[face];
                    int nz_ = iz + diz[face];

                    // Emit face if neighbor is outside domain or not boundary
                    bool emitFace = false;
                    if (nx_ < 0 || nx_ >= grid.nx ||
                        ny_ < 0 || ny_ >= grid.ny ||
                        nz_ < 0 || nz_ >= grid.nz) {
                        emitFace = true;
                    } else if (!grid.isBound[grid.idx(nx_, ny_, nz_)]) {
                        emitFace = true;
                    }

                    if (emitFace) {
                        vtkIdType pts[4];
                        for (int v = 0; v < 4; v++) {
                            double px = (ix + quadVerts[face][v][0]) * dx;
                            double py = (iy + quadVerts[face][v][1]) * dy;
                            double pz = (iz + quadVerts[face][v][2]) * dz;
                            pts[v] = points->InsertNextPoint(px, py, pz);

                            // Color by damage level
                            double dmg = grid.damage[id];
                            if (dmg > 0.01) {
                                double t = std::min(dmg / 50.0, 1.0);
                                unsigned char c[3] = {
                                    static_cast<unsigned char>(solidColor[0] + t * (damageColor[0] - solidColor[0])),
                                    static_cast<unsigned char>(solidColor[1] + t * (damageColor[1] - solidColor[1])),
                                    static_cast<unsigned char>(solidColor[2] + t * (damageColor[2] - solidColor[2]))
                                };
                                colors->InsertNextTypedTuple(c);
                            } else {
                                colors->InsertNextTypedTuple(solidColor);
                            }
                        }
                        cells->InsertNextCell(4, pts);
                    }
                }
            }
        }
    }

    auto polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->SetPolys(cells);
    polyData->GetPointData()->SetScalars(colors);

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

    // Apply cut plane clipping if enabled
    if (cutPlaneEnabled_) {
        double cx = grid.Lx * 0.5;
        double cy = grid.Ly * 0.5;
        double cz = grid.Lz * 0.5;

        if (cutAxis_ == CutAxis::X) {
            cutPlane_->SetOrigin(cx, 0, 0);
            cutPlane_->SetNormal(-1, 0, 0);
        } else if (cutAxis_ == CutAxis::Y) {
            cutPlane_->SetOrigin(0, cy, 0);
            cutPlane_->SetNormal(0, -1, 0);
        } else {
            cutPlane_->SetOrigin(0, 0, cz);
            cutPlane_->SetNormal(0, 0, -1);
        }

        auto clipper = vtkSmartPointer<vtkClipPolyData>::New();
        clipper->SetInputData(polyData);
        clipper->SetClipFunction(cutPlane_);
        clipper->Update();
        mapper->SetInputConnection(clipper->GetOutputPort());
    } else {
        mapper->SetInputData(polyData);
    }

    gridActor_->SetMapper(mapper);
    gridActor_->GetProperty()->SetColor(0.7, 0.7, 0.75);
    gridActor_->GetProperty()->SetOpacity(0.6);
    gridActor_->GetProperty()->SetEdgeVisibility(true);
    gridActor_->GetProperty()->SetEdgeColor(0.3, 0.3, 0.35);
    gridActor_->GetProperty()->SetLineWidth(0.5);
    gridActor_->GetProperty()->SetAmbient(0.4);
    gridActor_->GetProperty()->SetDiffuse(0.6);
    gridActor_->GetProperty()->SetSpecular(0.1);
}

void SimulationView3D::updateFromSimulator(const Simulator3D& sim,
                                            const SimParams& params) {
    const Grid3D& grid = sim.getGrid();
    if (grid.totalCells == 0) return;

    // ── Domain bounding box (transparent wireframe) ─────────────────
    {
        auto outline = vtkSmartPointer<vtkOutlineSource>::New();
        outline->SetBounds(0, grid.Lx, 0, grid.Ly, 0, grid.Lz);
        outline->Update();

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(outline->GetOutputPort());
        domainBoxActor_->SetMapper(mapper);
        domainBoxActor_->GetProperty()->SetColor(0.4, 0.6, 0.8);
        domainBoxActor_->GetProperty()->SetOpacity(0.5);
        domainBoxActor_->GetProperty()->SetLineWidth(1.5);
        domainBoxActor_->SetVisibility(true);
    }

    // ── Update grid geometry (solid boundary surfaces) ────────────────
    if (showGrid_) {
        buildSolidGridSurface(grid);
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

    // ── Update potential/temperature/damage slice ──────────────────────
    if (showPotential_ || showTemperature_ || showDamage_) {
        const std::vector<double>* fieldPtr = &grid.V;
        const char* fieldName = "Potential";
        const char* barTitle = "Potential (V)";

        if (showDamage_) {
            fieldPtr = &grid.damage;
            fieldName = "Damage";
            barTitle = "Sputtering Damage";
        } else if (showTemperature_) {
            fieldPtr = &grid.T_map;
            fieldName = "Temperature";
            barTitle = "Temperature (K)";
        }
        const auto& field = *fieldPtr;

        auto points = vtkSmartPointer<vtkPoints>::New();
        auto scalars = vtkSmartPointer<vtkDoubleArray>::New();
        scalars->SetName(fieldName);
        auto cells = vtkSmartPointer<vtkCellArray>::New();

        // Build a structured 2D quad mesh for the slice (smooth contour)
        int sliceIdx = 0;
        int rows = 0, cols = 0;

        if (slicePlane_ == SlicePlane::XY) {
            sliceIdx = static_cast<int>(slicePos_ * (grid.nz - 1));
            rows = grid.ny; cols = grid.nx;
            for (int iy = 0; iy < grid.ny; iy++) {
                for (int ix = 0; ix < grid.nx; ix++) {
                    points->InsertNextPoint(ix * grid.dx, iy * grid.dy,
                                            sliceIdx * grid.dz);
                    scalars->InsertNextValue(
                        static_cast<double>(field[grid.idx(ix, iy, sliceIdx)]));
                }
            }
        } else if (slicePlane_ == SlicePlane::XZ) {
            sliceIdx = static_cast<int>(slicePos_ * (grid.ny - 1));
            rows = grid.nz; cols = grid.nx;
            for (int iz = 0; iz < grid.nz; iz++) {
                for (int ix = 0; ix < grid.nx; ix++) {
                    points->InsertNextPoint(ix * grid.dx, sliceIdx * grid.dy,
                                            iz * grid.dz);
                    scalars->InsertNextValue(
                        static_cast<double>(field[grid.idx(ix, sliceIdx, iz)]));
                }
            }
        } else { // YZ
            sliceIdx = static_cast<int>(slicePos_ * (grid.nx - 1));
            rows = grid.nz; cols = grid.ny;
            for (int iz = 0; iz < grid.nz; iz++) {
                for (int iy = 0; iy < grid.ny; iy++) {
                    points->InsertNextPoint(sliceIdx * grid.dx, iy * grid.dy,
                                            iz * grid.dz);
                    scalars->InsertNextValue(
                        static_cast<double>(field[grid.idx(sliceIdx, iy, iz)]));
                }
            }
        }

        // Build quad cells for smooth rendering
        for (int r = 0; r < rows - 1; r++) {
            for (int c = 0; c < cols - 1; c++) {
                vtkIdType quad[4] = {
                    r * cols + c,
                    r * cols + c + 1,
                    (r + 1) * cols + c + 1,
                    (r + 1) * cols + c
                };
                cells->InsertNextCell(4, quad);
            }
        }

        auto polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData->SetPoints(points);
        polyData->SetPolys(cells);
        polyData->GetPointData()->SetScalars(scalars);

        double range[2];
        scalars->GetRange(range);
        if (range[0] == range[1]) range[1] = range[0] + 1.0;
        lut_->SetRange(range[0], range[1]);

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(polyData);
        mapper->SetLookupTable(lut_);
        mapper->SetScalarRange(range[0], range[1]);

        sliceActor_->SetMapper(mapper);
        sliceActor_->SetVisibility(true);

        colorBar_->SetTitle(barTitle);
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
    // Top-down view: looking along Z (beam axis), X-Y transverse plane
    auto cam = renderer_->GetActiveCamera();
    cam->SetPosition(0, 0, -100);
    cam->SetFocalPoint(0, 0, 0);
    cam->SetViewUp(0, 1, 0);
    renderer_->ResetCamera();
    renderWindow()->Render();
}

void SimulationView3D::setViewXZ() {
    // Side view: Z (beam) horizontal, X transverse vertical
    auto cam = renderer_->GetActiveCamera();
    cam->SetPosition(0, -100, 0);
    cam->SetFocalPoint(0, 0, 0);
    cam->SetViewUp(0, 0, 1); // Z (beam) points right
    renderer_->ResetCamera();
    renderWindow()->Render();
}

void SimulationView3D::setViewIso() {
    // Isometric: Z (beam) runs into the scene
    auto cam = renderer_->GetActiveCamera();
    cam->SetPosition(30, -40, -20);
    cam->SetFocalPoint(0, 0, 10);
    cam->SetViewUp(0, 0, 1);
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
}

void SimulationView3D::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

#endif

} // namespace BEMCS
