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

    // Erosion profile diagnostics: 1D line plot of cumulative groove depth
    // at the downstream face of the acceleration grid.
    struct ErosionProfile {
        std::vector<double> coord_mm;  // transverse coordinate (x or y), mm
        std::vector<double> depth_um;  // erosion depth at that coordinate, microns
    };
    enum class ProfileAxis { X, Y };

    // Returns the index of the grid with the most negative voltage
    // (conventionally the accel grid). Returns -1 if no grids are defined.
    int getAccelGridIndex(const SimParams& params) const;

    // Computes the groove-depth profile along the requested axis,
    // sliced at the opposite axis centre (X -> y=Ly/2 slice, Y -> x=Lx/2 slice).
    ErosionProfile getErosionProfile(const SimParams& params, ProfileAxis axis) const;

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
    double macroWeight_ = 1.0;  // computed at buildDomain, reused every step
};

} // namespace BEMCS
