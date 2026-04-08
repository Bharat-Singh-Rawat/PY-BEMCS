#pragma once
#include "Particle.h"
#include "Grid3D.h"
#include <random>

namespace BEMCS {

// ============================================================================
// Charge Exchange (CEX) and Secondary Electron Emission (SEE) handler
// ============================================================================
class CollisionHandler {
public:
    CollisionHandler();

    // CEX collisions: primary ions → slow CEX ions
    void processCEX(ParticleArray& ions, const Grid3D& grid,
                    const SimParams& params);

    // Secondary electron emission from grid impacts
    void processSecondaryEmission(ParticleArray& electrons,
                                  const ParticleManager::HitInfo& ionHits,
                                  const Grid3D& grid,
                                  const SimParams& params);

private:
    std::mt19937 rng_;
    std::normal_distribution<double> normal_;
    std::uniform_real_distribution<double> uniform_;
};

} // namespace BEMCS
