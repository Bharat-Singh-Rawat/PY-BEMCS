#pragma once
#include "Grid3D.h"
#include "Particle.h"

namespace BEMCS {

// ============================================================================
// 3D Boris Particle Pusher
//
// Full 3D Boris algorithm with trilinear field interpolation.
// Supports E and B fields. OpenMP parallelized.
// ============================================================================
class BorisPusher3D {
public:
    // Push all particles of a given species for one timestep
    void push(ParticleArray& particles, const Grid3D& grid,
              double dt, double qm) const;
};

} // namespace BEMCS
