#pragma once
#include "Grid3D.h"
#include "Particle.h"
#include "PoissonSolver3D.h"
#include "BorisPusher3D.h"
#include "ParticleManager.h"
#include "CollisionHandler.h"
#include "SputteringModel.h"
#include "ThermalSolver3D.h"
#include <atomic>

namespace BEMCS {

// ============================================================================
// 3D PIC Simulator Orchestrator
//
// Manages the full simulation loop: inject → accumulate → solve fields →
// push → collisions → sputtering → thermal → cleanup
// ============================================================================
class Simulator3D {
public:
    Simulator3D();

    // Build the simulation domain from parameters
    void buildDomain(const SimParams& params);

    // Execute one simulation timestep. Returns true if mesh was modified.
    bool step(const SimParams& params);

    // Reset everything
    void reset();

    // Accessors
    const Grid3D& getGrid() const { return grid_; }
    Grid3D& getGrid() { return grid_; }
    const ParticleArray& getIons() const { return ions_; }
    const ParticleArray& getElectrons() const { return electrons_; }
    int getIteration() const { return iteration_; }
    double getTime() const { return iteration_ * dt_; }

    // Diagnostics
    double getBeamDivergence(const SimParams& params) const;
    double getSaddlePointPotential(const SimParams& params) const;
    double getMeanIonEnergy() const;
    int getIonCount() const { return static_cast<int>(ions_.count); }
    int getElectronCount() const { return static_cast<int>(electrons_.count); }

    // Thread-safe running flag
    std::atomic<bool> isRunning{false};

private:
    Grid3D grid_;
    ParticleArray ions_;
    ParticleArray electrons_;

    PoissonSolver3D poisson_;
    BorisPusher3D pusher_;
    ParticleManager particleMgr_;
    CollisionHandler collisions_;
    SputteringModel sputtering_;
    ThermalSolver3D thermal_;

    int iteration_ = 0;
    double dt_ = 1e-9;
};

} // namespace BEMCS
