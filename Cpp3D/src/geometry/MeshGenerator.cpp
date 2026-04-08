#include "geometry/MeshGenerator.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace BEMCS {

bool MeshGenerator::generateFromSurface(const SurfaceMesh& surface,
                                         const MeshSettings& settings,
                                         Grid3D& outGrid,
                                         SimParams& params) {
    if (surface.empty()) {
        lastError_ = "Empty surface mesh";
        return false;
    }

    // Compute bounding box
    Vec3 minPt, maxPt;
    surface.getBoundingBox(minPt, maxPt);

    // Add padding
    double pad = settings.padding_mm;
    params.Lx = (maxPt.x - minPt.x) + 2 * pad;
    params.Ly = (maxPt.y - minPt.y) + 2 * pad;
    params.Lz = (maxPt.z - minPt.z) + 2 * pad;

    double cellSize = settings.cellSize_mm;
    if (settings.autoSize) {
        // Auto-size based on smallest geometry dimension
        double minDim = std::min({params.Lx, params.Ly, params.Lz});
        cellSize = minDim / 100.0; // ~100 cells across smallest dimension
        cellSize = std::max(cellSize, 0.01); // Floor at 10 microns
    }

    params.dx = cellSize;
    params.dy = cellSize;
    params.dz = cellSize;

    int nx = static_cast<int>(params.Lx / cellSize) + 1;
    int ny = static_cast<int>(params.Ly / cellSize) + 1;
    int nz = static_cast<int>(params.Lz / cellSize) + 1;

    if (static_cast<long long>(nx) * ny * nz > settings.maxCells) {
        lastError_ = "Mesh would exceed maximum cell count (" +
                     std::to_string(settings.maxCells) + "). Increase cell size.";
        return false;
    }

    // Initialize grid
    outGrid.initialize(params);

    // Offset: shift geometry so bounding box starts at (pad, pad, pad)
    Vec3 offset(pad - minPt.x, pad - minPt.y, pad - minPt.z);

    // Create shifted surface for voxelization
    SurfaceMesh shifted = surface;
    for (auto& v : shifted.vertices) {
        v.x += offset.x;
        v.y += offset.y;
        v.z += offset.z;
    }

    // Voxelize
    voxelizeSurface(shifted, outGrid);

    // Mark interior cells
    #pragma omp parallel for collapse(3)
    for (int iz = 0; iz < outGrid.nz; iz++) {
        for (int iy = 0; iy < outGrid.ny; iy++) {
            for (int ix = 0; ix < outGrid.nx; ix++) {
                size_t id = outGrid.idx(ix, iy, iz);
                if (!outGrid.isBound[id] &&
                    ix > 0 && ix < outGrid.nx - 1 &&
                    iy > 0 && iy < outGrid.ny - 1 &&
                    iz > 0 && iz < outGrid.nz - 1) {
                    outGrid.isInterior[id] = 1;
                }
            }
        }
    }

    return true;
}

void MeshGenerator::voxelizeSurface(const SurfaceMesh& surface, Grid3D& grid) {
    // For each grid cell center, test if it's inside the closed surface
    // This is the simplest (but O(N*M)) approach. For production, use
    // a spatial acceleration structure (octree, BVH).

    #pragma omp parallel for collapse(3) schedule(dynamic, 16)
    for (int iz = 0; iz < grid.nz; iz++) {
        for (int iy = 0; iy < grid.ny; iy++) {
            for (int ix = 0; ix < grid.nx; ix++) {
                Vec3 pt(ix * grid.dx, iy * grid.dy, iz * grid.dz);

                if (surface.isPointInside(pt)) {
                    size_t id = grid.idx(ix, iy, iz);
                    grid.isBound[id] = 1;
                }
            }
        }
    }
}

bool MeshGenerator::generateBuiltinGrids(const SimParams& params,
                                           Grid3D& outGrid) {
    // Delegate to Grid3D::buildDomain which handles built-in multi-aperture grids
    outGrid.buildDomain(params);
    return true;
}

MeshStats MeshGenerator::getStats(const Grid3D& grid) const {
    MeshStats stats;
    stats.numVertices = grid.nx * grid.ny * grid.nz;

    int boundCount = 0;
    for (size_t i = 0; i < grid.totalCells; i++) {
        if (grid.isBound[i]) boundCount++;
    }
    stats.numTriangles = boundCount; // Approximate: boundary cells
    stats.numTetrahedra = 0; // Structured grid, no tets

    stats.minEdgeLength = std::min({grid.dx, grid.dy, grid.dz});
    stats.maxEdgeLength = std::max({grid.dx, grid.dy, grid.dz});
    stats.avgEdgeLength = (grid.dx + grid.dy + grid.dz) / 3.0;

    stats.bboxMin = Vec3(0, 0, 0);
    stats.bboxMax = Vec3(grid.Lx, grid.Ly, grid.Lz);

    return stats;
}

} // namespace BEMCS
