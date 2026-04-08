#pragma once
#include "Grid3D.h"
#include "ParticleManager.h"

namespace BEMCS {

// ============================================================================
// 3D Thermal Conduction & Radiative Cooling Solver
//
// Finite-difference heat equation on boundary cells with:
// - Particle impact heating
// - Radiative cooling (Stefan-Boltzmann)
// - 3D thermal conduction (7-point stencil)
// ============================================================================
class ThermalSolver3D {
public:
    // Apply heating from particle impacts
    void applyImpactHeating(Grid3D& grid,
                            const ParticleManager::HitInfo& hits,
                            double mass, double macroWeight);

    // Apply radiative cooling on boundary cells
    void applyRadiativeCooling(Grid3D& grid, const SimParams& params);

    // Run 3D thermal conduction step (explicit finite difference)
    void conductionStep(Grid3D& grid, const SimParams& params);

    // Update per-grid average temperatures
    void updateGridTemps(Grid3D& grid);
};

} // namespace BEMCS
