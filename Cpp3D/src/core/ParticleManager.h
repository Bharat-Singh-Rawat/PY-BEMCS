#pragma once
#include "Particle.h"
#include "Grid3D.h"
#include <random>

namespace BEMCS {

// ============================================================================
// Manages particle injection, boundary removal, charge accumulation
// ============================================================================
class ParticleManager {
public:
    ParticleManager();

    // Inject beam ions from the plasma source (left boundary)
    void injectIons(ParticleArray& ions, const Grid3D& grid,
                    const SimParams& params);

    // Inject source electrons for RF co-extraction
    void injectSourceElectrons(ParticleArray& electrons, const Grid3D& grid,
                               const SimParams& params);

    // Inject neutralizer electrons
    void injectNeutralizerElectrons(ParticleArray& electrons, const Grid3D& grid,
                                     const SimParams& params);

    // Accumulate charge density on grid from particles
    void accumulateCharge(ParticleArray& particles, Grid3D& grid,
                          double chargeSign, double macroWeight) const;

    // Remove particles that hit boundaries or leave domain
    // Returns vectors of hit positions + energies for thermal/erosion
    struct HitInfo {
        std::vector<size_t> hitIndices;
        std::vector<double> hitEnergies_eV;
        std::vector<int> hitIx, hitIy, hitIz;  // Grid cell of impact
    };

    HitInfo removeDeadParticles(ParticleArray& particles, const Grid3D& grid,
                                 double mass, bool trackHits = true);

private:
    std::mt19937 rng_;
    std::normal_distribution<double> normal_;
    std::uniform_real_distribution<double> uniform_;
};

} // namespace BEMCS
