#include "core/Simulator3D.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace BEMCS {

Simulator3D::Simulator3D() {
    ions_.reserve(100000);
    electrons_.reserve(100000);
}

void Simulator3D::buildDomain(const SimParams& params) {
    dt_ = params.dt;
    iteration_ = 0;

    // Compute macro weight from target particles per cell
    double cellVol = (params.dx * 1e-3) * (params.dy * 1e-3) * (params.dz * 1e-3);
    double targetPPC = 20.0;
    SimParams p = params;
    p.macroWeight = std::max((params.plasmaDensity * cellVol) / targetPPC, 1e3);

    grid_.buildDomain(p);
    ions_.clear();
    electrons_.clear();

    // Initial Poisson solve (many iterations for convergence)
    poisson_.solveWithBoltzmann(grid_, p, 30, 500);
}

bool Simulator3D::step(const SimParams& params) {
    iteration_++;
    bool remeshed = false;

    // ── RF co-extraction: modulate grid voltage ────────────────────────
    if (params.rfEnable && !params.grids.empty()) {
        int rfIdx = params.rfGridIndex;
        if (rfIdx < static_cast<int>(grid_.gridMasks.size())) {
            double t = iteration_ * dt_;
            double f_hz = params.rfFreqMHz * 1e6;
            double v_rf = params.rfAmplitudeV * std::sin(2.0 * PI * f_hz * t);

            for (size_t i = 0; i < grid_.totalCells; i++) {
                if (grid_.gridMasks[rfIdx][i]) {
                    grid_.V_fixed[i] = params.grids[rfIdx].voltage_V + v_rf;
                }
            }
            poisson_.solveWithBoltzmann(grid_, params, 2, 100);
        }
    }

    // ── A. Inject particles ────────────────────────────────────────────
    particleMgr_.injectIons(ions_, grid_, params);
    particleMgr_.injectSourceElectrons(electrons_, grid_, params);
    particleMgr_.injectNeutralizerElectrons(electrons_, grid_, params);

    // ── B. Accumulate charge density ───────────────────────────────────
    grid_.clearRho();
    particleMgr_.accumulateCharge(ions_, grid_, +1.0, params.macroWeight);
    particleMgr_.accumulateCharge(electrons_, grid_, -1.0, params.macroWeight);

    // ── C. Poisson solve (every 2nd iteration for speed) ───────────────
    if (iteration_ % 2 == 0) {
        poisson_.solveWithBoltzmann(grid_, params, 5, 200);
    }

    // ── D. Boris push ──────────────────────────────────────────────────
    if (ions_.count > 0) {
        pusher_.push(ions_, grid_, dt_, Q_E / M_XE);
    }
    if (electrons_.count > 0) {
        // Use scaled electron mass (same as 2D code)
        double m_e_scaled = M_XE / 1000.0;
        pusher_.push(electrons_, grid_, dt_, -Q_E / m_e_scaled);
    }

    // ── E. Hit detection & removal ─────────────────────────────────────
    auto ionHits = particleMgr_.removeDeadParticles(ions_, grid_, M_XE, true);
    particleMgr_.removeDeadParticles(electrons_, grid_, M_XE / 1000.0, false);

    // ── F. Thermal effects ─────────────────────────────────────────────
    if (params.simMode == SimParams::Both || params.simMode == SimParams::Thermal) {
        thermal_.applyImpactHeating(grid_, ionHits, M_XE, params.macroWeight);
        thermal_.applyRadiativeCooling(grid_, params);
        thermal_.conductionStep(grid_, params);
        thermal_.updateGridTemps(grid_);
    }

    // ── G. Secondary electron emission ─────────────────────────────────
    collisions_.processSecondaryEmission(electrons_, ionHits, grid_, params);

    // ── H. Sputtering / erosion ────────────────────────────────────────
    if (params.simMode == SimParams::Both || params.simMode == SimParams::Erosion) {
        if (sputtering_.accumulateDamage(grid_, ionHits, params)) {
            // Cells removed — need to rebuild interior mask and re-solve
            grid_.buildDomain(params);
            poisson_.solveWithBoltzmann(grid_, params, 10, 300);
            remeshed = true;
        }
    }

    // ── I. CEX collisions ──────────────────────────────────────────────
    collisions_.processCEX(ions_, grid_, params);

    return remeshed;
}

void Simulator3D::reset() {
    iteration_ = 0;
    ions_.clear();
    electrons_.clear();
    grid_ = Grid3D();
}

double Simulator3D::getBeamDivergence(const SimParams& params) const {
    if (ions_.count < 5) return std::nan("");

    // Post-grid region
    double maxGridX = 1.0;
    for (const auto& g : params.grids) {
        maxGridX += g.thickness_mm + g.gap_mm;
    }

    std::vector<double> angles;
    for (size_t i = 0; i < ions_.count; i++) {
        if (!ions_.alive[i]) continue;
        if (ions_.species[i] == Species::CEX_Ion) continue;
        if (ions_.x[i] <= maxGridX) continue;

        double vPerp = std::sqrt(ions_.vy[i] * ions_.vy[i] +
                                 ions_.vz[i] * ions_.vz[i]);
        double angle = std::abs(std::atan2(vPerp, ions_.vx[i])) * 180.0 / PI;
        angles.push_back(angle);
    }

    if (angles.size() < 5) return std::nan("");

    std::sort(angles.begin(), angles.end());
    size_t idx95 = static_cast<size_t>(0.95 * angles.size());
    return angles[std::min(idx95, angles.size() - 1)];
}

double Simulator3D::getSaddlePointPotential(const SimParams& params) const {
    if (params.grids.size() < 2) return 0.0;

    double g2Start = 1.0 + params.grids[0].thickness_mm + params.grids[0].gap_mm;
    double g2Center = g2Start + params.grids[1].thickness_mm / 2.0;

    int ix = std::clamp(static_cast<int>(g2Center / grid_.dx), 0, grid_.nx - 1);
    int iy = grid_.ny / 2;
    int iz = grid_.nz / 2;

    return grid_.V[grid_.idx(ix, iy, iz)];
}

double Simulator3D::getMeanIonEnergy() const {
    if (ions_.count == 0) return 0.0;

    double sumE = 0.0;
    int count = 0;
    for (size_t i = 0; i < ions_.count; i++) {
        if (!ions_.alive[i]) continue;
        double v2 = ions_.vx[i] * ions_.vx[i] +
                    ions_.vy[i] * ions_.vy[i] +
                    ions_.vz[i] * ions_.vz[i];
        sumE += 0.5 * M_XE * v2 / Q_E;
        count++;
    }
    return count > 0 ? sumE / count : 0.0;
}

} // namespace BEMCS
