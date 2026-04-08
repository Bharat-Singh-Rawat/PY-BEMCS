#include "core/ParticleManager.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

namespace BEMCS {

ParticleManager::ParticleManager()
    : rng_(std::random_device{}()),
      normal_(0.0, 1.0),
      uniform_(0.0, 1.0) {}

void ParticleManager::injectIons(ParticleArray& ions, const Grid3D& grid,
                                  const SimParams& params) {
    if (params.grids.empty()) return;

    double Te = params.electronTemp_eV;
    double v_bohm = std::sqrt(Q_E * Te / M_XE);
    double rMax = params.grids[0].hole_radius_mm - 0.05;

    // Injection area: circular aperture (3D)
    double rMax_m = rMax * 1e-3;
    double injectionArea = PI * rMax_m * rMax_m;

    double I_ion = Q_E * 0.61 * params.plasmaDensity * v_bohm * injectionArea;
    double chargePerMacro = Q_E * params.macroWeight;
    double numInjectFloat = (I_ion * params.dt) / chargePerMacro;

    int numInject = static_cast<int>(numInjectFloat);
    if (uniform_(rng_) < (numInjectFloat - numInject)) numInject++;

    if (numInject <= 0) return;

    double v_spread = std::sqrt(Q_E * params.ionTemp_eV / M_XE);
    double cy = grid.Ly / 2.0;
    double cz = grid.Lz / 2.0;

    for (int i = 0; i < numInject; i++) {
        // Uniform random position within circular aperture
        double r = rMax * std::sqrt(uniform_(rng_));
        double theta = 2.0 * PI * uniform_(rng_);
        double py = cy + r * std::cos(theta);
        double pz = cz + r * std::sin(theta);
        double px = 0.1; // Near left boundary

        double pvx = v_bohm + normal_(rng_) * v_spread;
        double pvy = normal_(rng_) * v_spread;
        double pvz = normal_(rng_) * v_spread;

        ions.add(px, py, pz, pvx, pvy, pvz, Species::Ion, params.macroWeight);
    }
}

void ParticleManager::injectSourceElectrons(ParticleArray& electrons,
                                             const Grid3D& grid,
                                             const SimParams& params) {
    if (!params.rfEnable || params.grids.empty()) return;

    double Te = params.electronTemp_eV;
    double v_e_th = std::sqrt(2.0 * Q_E * Te / M_ELECTRON);
    double rMax = params.grids[0].hole_radius_mm - 0.05;
    double rMax_m = rMax * 1e-3;
    double injArea = PI * rMax_m * rMax_m;

    double I_e = Q_E * 0.25 * params.plasmaDensity * v_e_th * injArea;
    double chargePerMacro = Q_E * params.macroWeight;
    double numFloat = (I_e * params.dt) / chargePerMacro;

    int numInject = static_cast<int>(numFloat);
    if (uniform_(rng_) < (numFloat - numInject)) numInject++;

    if (numInject <= 0) return;

    double v_bohm = std::sqrt(Q_E * Te / M_XE);
    double cy = grid.Ly / 2.0;
    double cz = grid.Lz / 2.0;

    for (int i = 0; i < numInject; i++) {
        double r = rMax * std::sqrt(uniform_(rng_));
        double theta = 2.0 * PI * uniform_(rng_);
        double py = cy + r * std::cos(theta);
        double pz = cz + r * std::sin(theta);
        double px = 0.1;

        double pvx = std::abs(normal_(rng_)) * v_e_th + v_bohm;
        double pvy = normal_(rng_) * v_e_th;
        double pvz = normal_(rng_) * v_e_th;

        electrons.add(px, py, pz, pvx, pvy, pvz, Species::Electron,
                      params.macroWeight);
    }
}

void ParticleManager::injectNeutralizerElectrons(ParticleArray& electrons,
                                                   const Grid3D& grid,
                                                   const SimParams& params) {
    int numInject = params.neutRate;
    if (numInject <= 0) return;

    double Te = params.neutElectronTemp;
    double v_e_th = std::sqrt(2.0 * Q_E * Te / M_ELECTRON);
    double cy = grid.Ly / 2.0;
    double cz = grid.Lz / 2.0;
    double neutR = params.neutR_mm;

    for (int i = 0; i < numInject; i++) {
        double r = neutR * std::sqrt(uniform_(rng_));
        double theta = 2.0 * PI * uniform_(rng_);
        double py = cy + r * std::cos(theta);
        double pz = cz + r * std::sin(theta);
        double px = params.neutX_mm;

        double pvx = normal_(rng_) * v_e_th;
        double pvy = normal_(rng_) * v_e_th;
        double pvz = normal_(rng_) * v_e_th;

        electrons.add(px, py, pz, pvx, pvy, pvz, Species::Electron,
                      params.macroWeight);
    }
}

void ParticleManager::accumulateCharge(ParticleArray& particles, Grid3D& grid,
                                        double chargeSign,
                                        double macroWeight) const {
    double cellVol = (grid.dx * 1e-3) * (grid.dy * 1e-3) * (grid.dz * 1e-3);
    double chargeDensity = chargeSign * Q_E * macroWeight / cellVol;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(particles.count); i++) {
        if (!particles.alive[i]) continue;
        grid.accumulateCharge(particles.x[i], particles.y[i], particles.z[i],
                              chargeDensity);
    }
}

ParticleManager::HitInfo
ParticleManager::removeDeadParticles(ParticleArray& particles,
                                      const Grid3D& grid,
                                      double mass, bool trackHits) {
    HitInfo info;

    for (size_t i = 0; i < particles.count; i++) {
        if (!particles.alive[i]) continue;

        double px = particles.x[i];
        double py = particles.y[i];
        double pz = particles.z[i];

        bool outOfBounds = (px < 0 || px > grid.Lx ||
                            py < 0 || py > grid.Ly ||
                            pz < 0 || pz > grid.Lz ||
                            std::isnan(px));

        bool hitBound = !outOfBounds && grid.isInsideBoundary(px, py, pz);

        if (outOfBounds || hitBound) {
            particles.alive[i] = false;

            if (trackHits && hitBound && px > 0.5) {
                double v2 = particles.vx[i] * particles.vx[i] +
                            particles.vy[i] * particles.vy[i] +
                            particles.vz[i] * particles.vz[i];
                double E_eV = (0.5 * mass * v2) / Q_E;

                int ix = std::clamp(static_cast<int>(std::round(px / grid.dx)),
                                    0, grid.nx - 1);
                int iy = std::clamp(static_cast<int>(std::round(py / grid.dy)),
                                    0, grid.ny - 1);
                int iz = std::clamp(static_cast<int>(std::round(pz / grid.dz)),
                                    0, grid.nz - 1);

                info.hitIndices.push_back(i);
                info.hitEnergies_eV.push_back(E_eV);
                info.hitIx.push_back(ix);
                info.hitIy.push_back(iy);
                info.hitIz.push_back(iz);
            }
        }
    }

    particles.compact();
    return info;
}

} // namespace BEMCS
