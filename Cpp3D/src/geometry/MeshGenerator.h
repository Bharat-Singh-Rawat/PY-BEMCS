#pragma once
#include "Mesh3D.h"
#include "core/Grid3D.h"
#include "core/Particle.h"

namespace BEMCS {

// ============================================================================
// 3D Mesh Generator
//
// Converts imported CAD geometry (SurfaceMesh) into the structured Grid3D
// used by the PIC solver. Also provides standalone meshing utilities.
// ============================================================================
class MeshGenerator {
public:
    // Meshing parameters
    struct MeshSettings {
        double cellSize_mm = 0.05;   // Target cell size
        double refineFactor = 1.0;   // Refinement near geometry (1.0 = uniform)
        int    maxCells = 10000000;   // Safety limit
        bool   autoSize = true;      // Auto-compute cell size from geometry
        double padding_mm = 2.0;     // Padding around geometry for domain
    };

    // Generate structured Grid3D from imported surface mesh
    // The surface mesh is voxelized onto the grid to set boundary cells.
    bool generateFromSurface(const SurfaceMesh& surface,
                              const MeshSettings& settings,
                              Grid3D& outGrid,
                              SimParams& params);

    // Generate a simple multi-aperture grid geometry (built-in)
    // This is the equivalent of the Python code's build_domain
    bool generateBuiltinGrids(const SimParams& params,
                               Grid3D& outGrid);

    // Get mesh statistics
    MeshStats getStats(const Grid3D& grid) const;

    // Get last error
    const std::string& getError() const { return lastError_; }

private:
    // Voxelize: mark grid cells as boundary if they intersect the surface
    void voxelizeSurface(const SurfaceMesh& surface, Grid3D& grid);

    std::string lastError_;
};

} // namespace BEMCS
