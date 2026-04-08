#include "core/PoissonSolver3D.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace BEMCS {

void PoissonSolver3D::applyLaplacian(const Grid3D& grid,
                                      const std::vector<double>& x,
                                      std::vector<double>& result) const {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const double invDx2 = 1.0 / ((grid.dx * 1e-3) * (grid.dx * 1e-3));
    const double invDy2 = 1.0 / ((grid.dy * 1e-3) * (grid.dy * 1e-3));
    const double invDz2 = 1.0 / ((grid.dz * 1e-3) * (grid.dz * 1e-3));
    const double diag   = -2.0 * (invDx2 + invDy2 + invDz2);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                size_t id = grid.idx(ix, iy, iz);

                if (grid.isBound[id]) {
                    // Boundary: identity
                    result[id] = x[id];
                } else if (!grid.isInterior[id]) {
                    // Domain edge: Neumann (zero gradient)
                    result[id] = x[id];
                } else {
                    // Interior: 7-point stencil Laplacian
                    result[id] = diag * x[id]
                        + invDx2 * (x[grid.idx(ix+1, iy, iz)] + x[grid.idx(ix-1, iy, iz)])
                        + invDy2 * (x[grid.idx(ix, iy+1, iz)] + x[grid.idx(ix, iy-1, iz)])
                        + invDz2 * (x[grid.idx(ix, iy, iz+1)] + x[grid.idx(ix, iy, iz-1)]);
                }
            }
        }
    }
}

void PoissonSolver3D::conjugateGradient(Grid3D& grid,
                                         const std::vector<double>& rhs,
                                         std::vector<double>& solution,
                                         int maxIter, double tol) const {
    const size_t N = grid.totalCells;

    // Resize workspace if needed
    if (r_.size() != N) {
        r_.resize(N); p_.resize(N); Ap_.resize(N); z_.resize(N);
    }

    // Compute r = rhs - A*solution
    applyLaplacian(grid, solution, Ap_);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); i++) {
        r_[i] = rhs[i] - Ap_[i];
    }

    // Jacobi preconditioner: z = M^{-1} * r
    const double invDx2 = 1.0 / ((grid.dx * 1e-3) * (grid.dx * 1e-3));
    const double invDy2 = 1.0 / ((grid.dy * 1e-3) * (grid.dy * 1e-3));
    const double invDz2 = 1.0 / ((grid.dz * 1e-3) * (grid.dz * 1e-3));
    const double diagVal = -2.0 * (invDx2 + invDy2 + invDz2);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); i++) {
        if (grid.isInterior[i]) {
            z_[i] = r_[i] / diagVal;
        } else {
            z_[i] = r_[i]; // Identity on boundaries
        }
    }

    std::copy(z_.begin(), z_.end(), p_.begin());

    double rz_old = 0.0;
    #pragma omp parallel for reduction(+:rz_old)
    for (int i = 0; i < static_cast<int>(N); i++) {
        rz_old += r_[i] * z_[i];
    }

    for (int iter = 0; iter < maxIter; iter++) {
        applyLaplacian(grid, p_, Ap_);

        double pAp = 0.0;
        #pragma omp parallel for reduction(+:pAp)
        for (int i = 0; i < static_cast<int>(N); i++) {
            pAp += p_[i] * Ap_[i];
        }

        if (std::abs(pAp) < 1e-30) break;
        double alpha = rz_old / pAp;

        double rz_new = 0.0;
        #pragma omp parallel for reduction(+:rz_new)
        for (int i = 0; i < static_cast<int>(N); i++) {
            solution[i] += alpha * p_[i];
            r_[i] -= alpha * Ap_[i];

            // Preconditioner
            if (grid.isInterior[i]) {
                z_[i] = r_[i] / diagVal;
            } else {
                z_[i] = r_[i];
            }
            rz_new += r_[i] * z_[i];
        }

        // Check convergence
        double rnorm = 0.0;
        #pragma omp parallel for reduction(+:rnorm)
        for (int i = 0; i < static_cast<int>(N); i++) {
            rnorm += r_[i] * r_[i];
        }
        if (std::sqrt(rnorm) < tol) break;

        double beta = rz_new / rz_old;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); i++) {
            p_[i] = z_[i] + beta * p_[i];
        }
        rz_old = rz_new;
    }
}

void PoissonSolver3D::solve(Grid3D& grid, const SimParams& params,
                             int maxIter, double tol) {
    const size_t N = grid.totalCells;
    std::vector<double> rhs(N, 0.0);

    const double coeff = -1.0 / EPS_0;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); i++) {
        if (grid.isBound[i]) {
            rhs[i] = grid.V_fixed[i];
        } else if (grid.isInterior[i]) {
            rhs[i] = coeff * grid.rho[i];
        } else {
            rhs[i] = 0.0; // Neumann edges
        }
    }

    conjugateGradient(grid, rhs, grid.V, maxIter, tol);
    grid.computeEField();
}

void PoissonSolver3D::solveWithBoltzmann(Grid3D& grid, const SimParams& params,
                                          int outerIter, int cgIter) {
    const size_t N = grid.totalCells;
    const double Te = params.electronTemp_eV;
    const double n0 = params.plasmaDensity;

    double vPlasma = params.grids.empty()
        ? 1000.0 + params.plasmaOffset_V
        : params.grids[0].voltage_V + params.plasmaOffset_V;

    const double coeff = -1.0 / EPS_0;
    const double omega = 0.2; // Under-relaxation

    std::vector<double> rhs(N, 0.0);

    for (int outer = 0; outer < outerIter; outer++) {
        // Compute Boltzmann electron density
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); i++) {
            if (grid.isBound[i]) {
                rhs[i] = grid.V_fixed[i];
            } else if (grid.isInterior[i]) {
                double Ve = std::min(grid.V[i], vPlasma);
                double rho_e = -Q_E * n0 * std::exp((Ve - vPlasma) / Te);
                double rho_total = grid.rho[i] + rho_e;
                rhs[i] = coeff * rho_total;
            } else {
                rhs[i] = 0.0;
            }
        }

        std::vector<double> V_new = grid.V;
        conjugateGradient(grid, rhs, V_new, cgIter, 1e-5);

        // Under-relax
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); i++) {
            grid.V[i] = (1.0 - omega) * grid.V[i] + omega * V_new[i];
        }
    }

    grid.computeEField();
}

} // namespace BEMCS
