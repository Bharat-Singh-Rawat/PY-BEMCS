#pragma once
#include "Vec3.h"
#include "Particle.h"
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cassert>

namespace BEMCS {

// ============================================================================
// 3D Structured Grid for PIC fields
//
// Index convention: flat index = iz * (ny * nx) + iy * nx + ix
// All field arrays are stored flat for cache-friendly iteration.
// ============================================================================
class Grid3D {
public:
    int nx = 0, ny = 0, nz = 0;
    double dx, dy, dz;           // Cell size [mm]
    double Lx, Ly, Lz;          // Domain size [mm]
    size_t totalCells = 0;

    // Scalar fields
    std::vector<double> V;        // Electrostatic potential [V]
    std::vector<double> rho;      // Charge density [C/m³]
    std::vector<double> V_fixed;  // Fixed boundary potential
    std::vector<double> T_map;    // Temperature [K]
    std::vector<double> T_map_new;// Temperature buffer
    std::vector<double> damage;   // Sputtering damage accumulator

    // Vector electric field
    std::vector<double> Ex, Ey, Ez;

    // Vector magnetic field
    std::vector<double> Bx, By, Bz;

    // Boundary / geometry mask
    std::vector<uint8_t> isBound;    // 1 = solid boundary, 0 = vacuum
    std::vector<uint8_t> isInterior; // 1 = interior (solvable), 0 = boundary/edge

    // Grid mask per optic (for thermal tracking)
    std::vector<std::vector<uint8_t>> gridMasks;
    // Immutable snapshot of gridMasks taken at buildDomain; used for erosion diagnostics
    // so we can tell which voxels were original grid material even after cells are removed.
    std::vector<std::vector<uint8_t>> originalGridMasks;
    std::vector<double> gridTemps;

    void initialize(const SimParams& params);
    void buildDomain(const SimParams& params);
    void rebuildInteriorMask();   // lightweight: only recomputes isInterior after erosion
    void computeEField();
    void clearRho() { std::fill(rho.begin(), rho.end(), 0.0); }

    // Flat index from (ix, iy, iz)
    inline size_t idx(int ix, int iy, int iz) const {
        return static_cast<size_t>(iz) * ny * nx +
               static_cast<size_t>(iy) * nx + ix;
    }

    // Clamp grid indices
    inline int clampX(int ix) const { return std::clamp(ix, 0, nx - 1); }
    inline int clampY(int iy) const { return std::clamp(iy, 0, ny - 1); }
    inline int clampZ(int iz) const { return std::clamp(iz, 0, nz - 1); }

    // Trilinear interpolation of a scalar field at position (px, py, pz) in mm
    double interpolate(const std::vector<double>& field,
                       double px, double py, double pz) const;

    // Accumulate charge at a position (atomic-safe with OpenMP)
    void accumulateCharge(double px, double py, double pz, double chargeVal);

    // Check if position is inside a boundary cell
    bool isInsideBoundary(double px, double py, double pz) const;
};

} // namespace BEMCS
