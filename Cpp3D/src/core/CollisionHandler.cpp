#include "core/CollisionHandler.h"
#include "core/ParticleManager.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

namespace BEMCS {

CollisionHandler::CollisionHandler()
    : rng_(std::random_device{}()),
      normal_(0.0, 1.0),
      uniform_(0.0, 1.0) {}

void CollisionHandler::processCEX(ParticleArray& ions, const Grid3D& grid,
                                   const SimParams& params) {
    if (ions.count == 0) return;

    double neutVth = std::sqrt(2.0 * K_B * params.neutralTemp_K / M_XE);

    for (size_t i = 0; i < ions.count; i++) {
        if (!ions.alive[i]) continue;
        if (ions.species[i] == Species::CEX_Ion) continue;

        double px = ions.x[i];
        // Only collide within grid region
        if (px < 1.0 || px > grid.Lx) continue;

        double vMag = std::sqrt(ions.vx[i] * ions.vx[i] +
                                ions.vy[i] * ions.vy[i] +
                                ions.vz[i] * ions.vz[i]);
        double g = std::max(vMag, 1.0);

        // Xenon CEX cross section (Rapp-Francis type)
        double logG = std::log(g);
        double sigmaVal = (-0.8821 * logG + 15.1262);
        double sigma = sigmaVal * sigmaVal * 1e-20; // [m²]

        double prob = 1.0 - std::exp(-params.neutralDensity * sigma * g * params.dt);

        if (uniform_(rng_) < prob) {
            // Replace velocity with thermal neutral velocity
            ions.vx[i] = normal_(rng_) * neutVth;
            ions.vy[i] = normal_(rng_) * neutVth;
            ions.vz[i] = normal_(rng_) * neutVth;
            ions.species[i] = Species::CEX_Ion;
        }
    }
}

void CollisionHandler::processSecondaryEmission(
    ParticleArray& electrons,
    const ParticleManager::HitInfo& ionHits,
    const Grid3D& grid,
    const SimParams& params) {

    if (ionHits.hitIndices.empty()) return;

    double T_see = 2.0; // SEE electron temperature [eV]
    double v_see_th = std::sqrt(2.0 * Q_E * T_see / M_ELECTRON);

    for (size_t h = 0; h < ionHits.hitIndices.size(); h++) {
        double E_eV = ionHits.hitEnergies_eV[h];

        // SEE yield model: gamma = 0.05 + 1e-4 * E_eV, capped at 1.0
        double gamma = std::clamp(0.05 + 1e-4 * E_eV, 0.0, 1.0);

        if (uniform_(rng_) < gamma) {
            // Spawn secondary electron near the impact site
            double px = ionHits.hitIx[h] * grid.dx;
            double py = ionHits.hitIy[h] * grid.dy;
            double pz = ionHits.hitIz[h] * grid.dz;

            // Push slightly away from surface
            px -= 0.5 * grid.dx;

            double pvx = normal_(rng_) * v_see_th;
            double pvy = normal_(rng_) * v_see_th;
            double pvz = normal_(rng_) * v_see_th;

            electrons.add(px, py, pz, pvx, pvy, pvz,
                          Species::Electron, params.macroWeight);
        }
    }
}

} // namespace BEMCS
