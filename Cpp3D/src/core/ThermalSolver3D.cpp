#include "core/ThermalSolver3D.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace BEMCS {

void ThermalSolver3D::applyImpactHeating(Grid3D& grid,
                                          const ParticleManager::HitInfo& hits,
                                          double mass, double macroWeight) {
    double dx_m = grid.dx * 1e-3;
    double dy_m = grid.dy * 1e-3;
    double dz_m = grid.dz * 1e-3;
    double cellVol = dx_m * dy_m * dz_m;
    double C_cell = MOLY_DENSITY * cellVol * MOLY_CP;

    for (size_t h = 0; h < hits.hitIndices.size(); h++) {
        double E_J = hits.hitEnergies_eV[h] * Q_E * macroWeight;
        double dT = E_J / C_cell;

        size_t id = grid.idx(hits.hitIx[h], hits.hitIy[h], hits.hitIz[h]);
        grid.T_map[id] += dT;
    }
}

void ThermalSolver3D::applyRadiativeCooling(Grid3D& grid,
                                             const SimParams& params) {
    double dx_m = grid.dx * 1e-3;
    double dy_m = grid.dy * 1e-3;
    double A_cell = 2.0 * dx_m * dy_m; // Approximate surface area
    double cellVol = dx_m * dy_m * (grid.dz * 1e-3);
    double C_cell = MOLY_DENSITY * cellVol * MOLY_CP;

    double factor = (MOLY_EMISSIVITY * SIGMA_SB * A_cell * params.dt *
                     params.accelFactor) / C_cell;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(grid.totalCells); i++) {
        if (grid.isBound[i]) {
            double T = grid.T_map[i];
            double dT = factor * (T * T * T * T - 300.0 * 300.0 * 300.0 * 300.0);
            grid.T_map[i] = std::max(T - dT, 300.0);
        }
    }
}

void ThermalSolver3D::conductionStep(Grid3D& grid, const SimParams& params) {
    double alpha = MOLY_K / (MOLY_DENSITY * MOLY_CP);
    double dt_thermal = params.dt * params.accelFactor;

    double dx_m = grid.dx * 1e-3;
    double dy_m = grid.dy * 1e-3;
    double dz_m = grid.dz * 1e-3;

    double Fo_x = alpha * dt_thermal / (dx_m * dx_m);
    double Fo_y = alpha * dt_thermal / (dy_m * dy_m);
    double Fo_z = alpha * dt_thermal / (dz_m * dz_m);

    // Stability limit
    double maxFo = 0.15; // Slightly conservative for 3D
    double maxCurrent = std::max({Fo_x, Fo_y, Fo_z});
    if (maxCurrent > maxFo) {
        double scale = maxFo / maxCurrent;
        Fo_x *= scale;
        Fo_y *= scale;
        Fo_z *= scale;
    }

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                size_t id = grid.idx(ix, iy, iz);

                if (grid.isBound[id]) {
                    double T_c = grid.T_map[id];

                    // Neighbor temps (clamped to self if neighbor not solid)
                    auto safeT = [&](int jx, int jy, int jz) -> double {
                        jx = std::clamp(jx, 0, nx - 1);
                        jy = std::clamp(jy, 0, ny - 1);
                        jz = std::clamp(jz, 0, nz - 1);
                        size_t jid = grid.idx(jx, jy, jz);
                        return grid.isBound[jid] ? grid.T_map[jid] : T_c;
                    };

                    double T_xp = safeT(ix + 1, iy, iz);
                    double T_xm = safeT(ix - 1, iy, iz);
                    double T_yp = safeT(ix, iy + 1, iz);
                    double T_ym = safeT(ix, iy - 1, iz);
                    double T_zp = safeT(ix, iy, iz + 1);
                    double T_zm = safeT(ix, iy, iz - 1);

                    double dT = Fo_x * (T_xm - 2.0 * T_c + T_xp)
                              + Fo_y * (T_ym - 2.0 * T_c + T_yp)
                              + Fo_z * (T_zm - 2.0 * T_c + T_zp);

                    grid.T_map_new[id] = std::max(T_c + dT, 300.0);
                } else {
                    grid.T_map_new[id] = grid.T_map[id];
                }
            }
        }
    }

    // Swap buffers
    std::swap(grid.T_map, grid.T_map_new);
}

void ThermalSolver3D::updateGridTemps(Grid3D& grid) {
    for (size_t gi = 0; gi < grid.gridMasks.size(); gi++) {
        double sum = 0.0;
        int count = 0;
        for (size_t i = 0; i < grid.totalCells; i++) {
            if (grid.gridMasks[gi][i]) {
                sum += grid.T_map[i];
                count++;
            }
        }
        grid.gridTemps[gi] = (count > 0) ? sum / count : 300.0;
    }
}

} // namespace BEMCS
