#pragma once
#include "Grid3D.h"
#include "ParticleManager.h"

namespace BEMCS {

// ============================================================================
// Sputtering / Erosion Model
//
// Computes sputter yield from ion impact energy and accumulates damage
// on grid cells. Removes cells that exceed the failure threshold.
// ============================================================================
class SputteringModel {
public:
    // Accumulate erosion damage from ion impacts
    // Returns true if any cells were removed (requires re-meshing)
    bool accumulateDamage(Grid3D& grid,
                          const ParticleManager::HitInfo& hits,
                          const SimParams& params);

    // Compute sputter yield Y(E) for Xe+ on Mo (empirical fit).
    // Energy must be supplied in *physical* eV — scaled simulations should
    // multiply the scaled ion energy by dimScaleFactor before calling this.
    static double sputterYield(double energy_eV);

    // Get current max damage value (for visualization)
    double maxDamage(const Grid3D& grid) const;
};

} // namespace BEMCS
