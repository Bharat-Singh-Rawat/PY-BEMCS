#include "core/Grid3D.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace BEMCS {

void Grid3D::initialize(const SimParams& params) {
    Lx = params.Lx; Ly = params.Ly; Lz = params.Lz;
    dx = params.dx; dy = params.dy; dz = params.dz;

    nx = static_cast<int>(Lx / dx) + 1;
    ny = static_cast<int>(Ly / dy) + 1;
    nz = static_cast<int>(Lz / dz) + 1;
    totalCells = static_cast<size_t>(nx) * ny * nz;

    V.assign(totalCells, 0.0);
    rho.assign(totalCells, 0.0);
    V_fixed.assign(totalCells, 0.0);
    T_map.assign(totalCells, 300.0);
    T_map_new.assign(totalCells, 300.0);
    damage.assign(totalCells, 0.0);

    Ex.assign(totalCells, 0.0);
    Ey.assign(totalCells, 0.0);
    Ez.assign(totalCells, 0.0);

    Bx.assign(totalCells, params.B_external.x);
    By.assign(totalCells, params.B_external.y);
    Bz.assign(totalCells, params.B_external.z);

    isBound.assign(totalCells, 0);
    isInterior.assign(totalCells, 0);

    gridMasks.clear();
    originalGridMasks.clear();
    gridTemps.clear();
}

void Grid3D::buildDomain(const SimParams& params) {
    initialize(params);

    // Build multi-grid optics geometry in 3D
    // The beam axis is along Z. Grids are planar structures with circular apertures.
    // In 3D: hole is a cylinder along Z through each grid slab.
    double currentZ = params.firstGridZ_mm; // Upstream face of the first grid [mm]

    for (size_t gi = 0; gi < params.grids.size(); gi++) {
        const auto& g = params.grids[gi];
        double gStart = currentZ;
        double gEnd   = gStart + g.thickness_mm;

        std::vector<uint8_t> mask(totalCells, 0);

        #pragma omp parallel for collapse(3)
        for (int iz = 0; iz < nz; iz++) {
            for (int iy = 0; iy < ny; iy++) {
                for (int ix = 0; ix < nx; ix++) {
                    double px = ix * dx;
                    double py = iy * dy;
                    double pz = iz * dz;

                    // Check if within grid slab (along Z = beam axis)
                    if (pz >= gStart && pz <= gEnd) {
                        // Radial distance from beam axis (center of X-Y plane)
                        double cx = Lx / 2.0;
                        double cy = Ly / 2.0;
                        double r = std::sqrt((px - cx) * (px - cx) +
                                             (py - cy) * (py - cy));

                        // Aperture radius with chamfer
                        double localZ = pz - gStart;
                        double R_aperture = g.hole_radius_mm +
                            localZ * std::tan(g.chamfer_deg * PI / 180.0);

                        // Outside the hole = solid grid material
                        if (r >= R_aperture) {
                            size_t id = idx(ix, iy, iz);
                            isBound[id] = 1;
                            V_fixed[id] = g.voltage_V;
                            mask[id] = 1;
                        }
                    }
                }
            }
        }

        gridMasks.push_back(std::move(mask));
        gridTemps.push_back(300.0);
        currentZ = gEnd + g.gap_mm;
    }

    originalGridMasks = gridMasks;

    // Z=0 boundary (plasma source, upstream)
    double vPlasmaBound = params.grids.empty() ? 1000.0 + params.plasmaOffset_V
                          : params.grids[0].voltage_V + params.plasmaOffset_V;
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            size_t id = idx(ix, iy, 0);
            isBound[id] = 1;
            V_fixed[id] = vPlasmaBound;
        }
    }

    // Mark interior cells (not boundary, not domain edge)
    #pragma omp parallel for collapse(3)
    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                size_t id = idx(ix, iy, iz);
                if (!isBound[id] &&
                    ix > 0 && ix < nx - 1 &&
                    iy > 0 && iy < ny - 1 &&
                    iz > 0 && iz < nz - 1) {
                    isInterior[id] = 1;
                }
            }
        }
    }
}

void Grid3D::rebuildInteriorMask() {
    #pragma omp parallel for collapse(3)
    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                size_t id = idx(ix, iy, iz);
                isInterior[id] = (!isBound[id] &&
                                  ix > 0 && ix < nx - 1 &&
                                  iy > 0 && iy < ny - 1 &&
                                  iz > 0 && iz < nz - 1) ? 1 : 0;
            }
        }
    }
}

void Grid3D::computeEField() {
    double dx_m = dx * 1e-3;
    double dy_m = dy * 1e-3;
    double dz_m = dz * 1e-3;

    #pragma omp parallel for collapse(3)
    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                size_t id = idx(ix, iy, iz);

                int ixp = std::min(ix + 1, nx - 1);
                int ixm = std::max(ix - 1, 0);
                int iyp = std::min(iy + 1, ny - 1);
                int iym = std::max(iy - 1, 0);
                int izp = std::min(iz + 1, nz - 1);
                int izm = std::max(iz - 1, 0);

                double hx = (ixp - ixm) * dx_m;
                double hy = (iyp - iym) * dy_m;
                double hz = (izp - izm) * dz_m;

                Ex[id] = -(V[idx(ixp, iy, iz)] - V[idx(ixm, iy, iz)]) / hx;
                Ey[id] = -(V[idx(ix, iyp, iz)] - V[idx(ix, iym, iz)]) / hy;
                Ez[id] = -(V[idx(ix, iy, izp)] - V[idx(ix, iy, izm)]) / hz;
            }
        }
    }
}

double Grid3D::interpolate(const std::vector<double>& field,
                            double px, double py, double pz) const {
    double fx = px / dx;
    double fy = py / dy;
    double fz = pz / dz;

    int ix0 = std::clamp(static_cast<int>(std::floor(fx)), 0, nx - 2);
    int iy0 = std::clamp(static_cast<int>(std::floor(fy)), 0, ny - 2);
    int iz0 = std::clamp(static_cast<int>(std::floor(fz)), 0, nz - 2);

    double wx = fx - ix0;
    double wy = fy - iy0;
    double wz = fz - iz0;

    double c000 = field[idx(ix0,     iy0,     iz0)];
    double c100 = field[idx(ix0 + 1, iy0,     iz0)];
    double c010 = field[idx(ix0,     iy0 + 1, iz0)];
    double c110 = field[idx(ix0 + 1, iy0 + 1, iz0)];
    double c001 = field[idx(ix0,     iy0,     iz0 + 1)];
    double c101 = field[idx(ix0 + 1, iy0,     iz0 + 1)];
    double c011 = field[idx(ix0,     iy0 + 1, iz0 + 1)];
    double c111 = field[idx(ix0 + 1, iy0 + 1, iz0 + 1)];

    double c00 = c000 * (1 - wx) + c100 * wx;
    double c01 = c001 * (1 - wx) + c101 * wx;
    double c10 = c010 * (1 - wx) + c110 * wx;
    double c11 = c011 * (1 - wx) + c111 * wx;

    double c0 = c00 * (1 - wy) + c10 * wy;
    double c1 = c01 * (1 - wy) + c11 * wy;

    return c0 * (1 - wz) + c1 * wz;
}

void Grid3D::accumulateCharge(double px, double py, double pz, double chargeVal) {
    int ix = std::clamp(static_cast<int>(std::round(px / dx)), 1, nx - 2);
    int iy = std::clamp(static_cast<int>(std::round(py / dy)), 1, ny - 2);
    int iz = std::clamp(static_cast<int>(std::round(pz / dz)), 1, nz - 2);

    size_t id = idx(ix, iy, iz);
    #pragma omp atomic
    rho[id] += chargeVal;
}

bool Grid3D::isInsideBoundary(double px, double py, double pz) const {
    int ix = std::clamp(static_cast<int>(std::round(px / dx)), 0, nx - 1);
    int iy = std::clamp(static_cast<int>(std::round(py / dy)), 0, ny - 1);
    int iz = std::clamp(static_cast<int>(std::round(pz / dz)), 0, nz - 1);
    return isBound[idx(ix, iy, iz)] != 0;
}

} // namespace BEMCS
