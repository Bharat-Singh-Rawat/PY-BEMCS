#include "core/SputteringModel.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

namespace BEMCS {

double SputteringModel::sputterYield(double energy_eV) {
    // Empirical Xe+ on Mo yield: Y = 1.05e-4 * max(E - 30, 0)^1.5
    // Capped at 1.0
    double above_threshold = std::max(energy_eV - 30.0, 0.0);
    return std::min(1.05e-4 * std::pow(above_threshold, 1.5), 1.0);
}

bool SputteringModel::accumulateDamage(Grid3D& grid,
                                        const ParticleManager::HitInfo& hits,
                                        const SimParams& params) {
    if (hits.hitIndices.empty()) return false;

    for (size_t h = 0; h < hits.hitIndices.size(); h++) {
        double E_eV = hits.hitEnergies_eV[h];
        double Y = sputterYield(E_eV);

        int ix = hits.hitIx[h];
        int iy = hits.hitIy[h];
        int iz = hits.hitIz[h];
        size_t id = grid.idx(ix, iy, iz);

        grid.damage[id] += Y * params.accelFactor;
    }

    // Check for broken cells
    bool anyRemoved = false;
    for (size_t i = 0; i < grid.totalCells; i++) {
        if (grid.damage[i] > params.cellFailThreshold && grid.isBound[i]) {
            grid.isBound[i] = 0;
            grid.damage[i] = 0.0;

            // Update grid masks
            for (auto& mask : grid.gridMasks) {
                if (i < mask.size()) mask[i] = 0;
            }
            anyRemoved = true;
        }
    }

    return anyRemoved;
}

double SputteringModel::maxDamage(const Grid3D& grid) const {
    double maxD = 0.0;
    for (size_t i = 0; i < grid.totalCells; i++) {
        if (grid.damage[i] > maxD) maxD = grid.damage[i];
    }
    return maxD;
}

} // namespace BEMCS
