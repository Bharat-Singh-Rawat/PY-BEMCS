#pragma once
#include "Vec3.h"
#include <vector>
#include <cstdint>

namespace BEMCS {

// ============================================================================
// Particle species enumeration
// ============================================================================
enum class Species : uint8_t {
    Ion         = 0,  // Primary beam ion
    CEX_Ion     = 1,  // Charge-exchange ion
    Electron    = 2,  // Plasma / SEE electron
    Neutral     = 3   // Background neutral (for DSMC if needed)
};

// ============================================================================
// Structure-of-Arrays (SoA) particle storage for cache-friendly access
// ============================================================================
struct ParticleArray {
    std::vector<double> x, y, z;       // Position [m] (in mesh coords: mm)
    std::vector<double> vx, vy, vz;    // Velocity [m/s]
    std::vector<Species> species;       // Particle type
    std::vector<double>  weight;        // Macro-particle weight
    std::vector<bool>    alive;         // Active flag
    size_t count = 0;                   // Number of active particles
    size_t capacity = 0;

    void reserve(size_t n) {
        x.reserve(n); y.reserve(n); z.reserve(n);
        vx.reserve(n); vy.reserve(n); vz.reserve(n);
        species.reserve(n); weight.reserve(n); alive.reserve(n);
        capacity = n;
    }

    void resize(size_t n) {
        x.resize(n, 0.0); y.resize(n, 0.0); z.resize(n, 0.0);
        vx.resize(n, 0.0); vy.resize(n, 0.0); vz.resize(n, 0.0);
        species.resize(n, Species::Ion);
        weight.resize(n, 1.0);
        // Preserve alive status of existing particles; only new slots get false
        size_t oldSize = alive.size();
        alive.resize(n);
        for (size_t i = oldSize; i < n; i++) alive[i] = false;
        capacity = n;
    }

    // Add a single particle, growing buffer if needed
    size_t add(double px, double py, double pz,
               double pvx, double pvy, double pvz,
               Species sp, double w) {
        if (count >= capacity) {
            size_t newCap = std::max(capacity * 2, (size_t)1024);
            resize(newCap);
        }
        size_t idx = count;
        x[idx] = px; y[idx] = py; z[idx] = pz;
        vx[idx] = pvx; vy[idx] = pvy; vz[idx] = pvz;
        species[idx] = sp;
        weight[idx] = w;
        alive[idx] = true;
        count++;
        return idx;
    }

    // Compact: remove dead particles by shifting alive ones forward
    void compact() {
        size_t writeIdx = 0;
        for (size_t i = 0; i < count; i++) {
            if (alive[i]) {
                if (writeIdx != i) {
                    x[writeIdx] = x[i]; y[writeIdx] = y[i]; z[writeIdx] = z[i];
                    vx[writeIdx] = vx[i]; vy[writeIdx] = vy[i]; vz[writeIdx] = vz[i];
                    species[writeIdx] = species[i];
                    weight[writeIdx] = weight[i];
                    alive[writeIdx] = true;
                }
                writeIdx++;
            }
        }
        count = writeIdx;
    }

    void clear() {
        count = 0;
        // Don't deallocate, just reset count
    }
};

// ============================================================================
// Grid optic definition (maps from config.ini style)
// ============================================================================
struct GridOptic {
    double voltage_V    = 0.0;
    double thickness_mm = 1.0;
    double gap_mm       = 1.0;
    double hole_radius_mm = 1.0;
    double chamfer_deg  = 0.0;
};

// ============================================================================
// Simulation parameters (passed through the system)
// ============================================================================
struct SimParams {
    // Plasma
    double plasmaDensity    = 1e17;   // [m^-3]
    double electronTemp_eV  = 3.0;
    double ionTemp_eV       = 2.0;
    double neutralTemp_K    = 300.0;
    double neutralDensity   = 1e20;   // [m^-3]

    // Domain (mm)
    double Lx = 3.0, Ly = 3.0, Lz = 10.0;
    double dx = 0.05, dy = 0.05, dz = 0.05;
    // Z position [mm] of the first grid's upstream face (plasma-source gap)
    double firstGridZ_mm = 1.0;

    // Acceleration / erosion
    double accelFactor      = 1.0;
    double cellFailThreshold = 10000.0;
    double macroWeight      = 3e5;

    // RF co-extraction
    bool   rfEnable         = false;
    int    rfGridIndex      = 0;
    double rfFreqMHz        = 13.56;
    double rfAmplitudeV     = 500.0;

    // Neutralizer
    int    neutRate         = 0;
    double neutElectronTemp = 5.0;
    double neutX_mm        = 9.0; // Neutralizer Z position along beam
    double neutR_mm        = 3.0;

    // Advanced
    double plasmaOffset_V  = 20.0;
    double dt              = 1e-9;   // Timestep [s]

    // Simulation mode
    enum Mode { Both, Thermal, Erosion };
    Mode simMode = Both;

    // Grid optics
    std::vector<GridOptic> grids;

    // Magnetic field (uniform for now, can be extended to field map)
    Vec3 B_external = {0.0, 0.0, 0.0}; // [T]
};

} // namespace BEMCS
