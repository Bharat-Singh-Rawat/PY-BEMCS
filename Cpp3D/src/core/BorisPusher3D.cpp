#include "core/BorisPusher3D.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace BEMCS {

void BorisPusher3D::push(ParticleArray& particles, const Grid3D& grid,
                          double dt, double qm) const {
    const size_t np = particles.count;
    if (np == 0) return;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(np); i++) {
        if (!particles.alive[i]) continue;

        double px = particles.x[i];
        double py = particles.y[i];
        double pz = particles.z[i];

        // ── Trilinear field interpolation ──────────────────────────────
        double Ex_p = grid.interpolate(grid.Ex, px, py, pz);
        double Ey_p = grid.interpolate(grid.Ey, px, py, pz);
        double Ez_p = grid.interpolate(grid.Ez, px, py, pz);

        double Bx_p = grid.interpolate(grid.Bx, px, py, pz);
        double By_p = grid.interpolate(grid.By, px, py, pz);
        double Bz_p = grid.interpolate(grid.Bz, px, py, pz);

        // ── BORIS ALGORITHM ────────────────────────────────────────────
        // Step 1: Half E-field acceleration (v_minus)
        double half_qm_dt = qm * dt * 0.5;

        double v_minus_x = particles.vx[i] + half_qm_dt * Ex_p;
        double v_minus_y = particles.vy[i] + half_qm_dt * Ey_p;
        double v_minus_z = particles.vz[i] + half_qm_dt * Ez_p;

        // Step 2: Magnetic rotation
        double tx = half_qm_dt * Bx_p;
        double ty = half_qm_dt * By_p;
        double tz = half_qm_dt * Bz_p;
        double t_mag_sq = tx * tx + ty * ty + tz * tz;

        double sx = 2.0 * tx / (1.0 + t_mag_sq);
        double sy = 2.0 * ty / (1.0 + t_mag_sq);
        double sz = 2.0 * tz / (1.0 + t_mag_sq);

        // v_prime = v_minus + (v_minus x t)
        double vpx = v_minus_x + (v_minus_y * tz - v_minus_z * ty);
        double vpy = v_minus_y + (v_minus_z * tx - v_minus_x * tz);
        double vpz = v_minus_z + (v_minus_x * ty - v_minus_y * tx);

        // v_plus = v_minus + (v_prime x s)
        double v_plus_x = v_minus_x + (vpy * sz - vpz * sy);
        double v_plus_y = v_minus_y + (vpz * sx - vpx * sz);
        double v_plus_z = v_minus_z + (vpx * sy - vpy * sx);

        // Step 3: Second half E-field acceleration
        particles.vx[i] = v_plus_x + half_qm_dt * Ex_p;
        particles.vy[i] = v_plus_y + half_qm_dt * Ey_p;
        particles.vz[i] = v_plus_z + half_qm_dt * Ez_p;

        // ── Kinematic update (positions in mm) ─────────────────────────
        particles.x[i] += particles.vx[i] * dt * 1000.0;
        particles.y[i] += particles.vy[i] * dt * 1000.0;
        particles.z[i] += particles.vz[i] * dt * 1000.0;
    }
}

} // namespace BEMCS
