#pragma once
#include "Grid3D.h"
#include "Particle.h"

namespace BEMCS {

// ============================================================================
// 3D Poisson Solver using Conjugate Gradient with Jacobi preconditioner
//
// Solves: ∇²V = -ρ/ε₀  on interior cells
// With Dirichlet BCs on boundary cells and Neumann on domain edges.
//
// Also includes Boltzmann electron fluid model for plasma sheath.
// ============================================================================
class PoissonSolver3D {
public:
    void solve(Grid3D& grid, const SimParams& params, int maxIter = 500,
               double tol = 1e-6);

    // Iterative Poisson with Boltzmann electrons (like the 2D code)
    void solveWithBoltzmann(Grid3D& grid, const SimParams& params,
                           int outerIter = 5, int cgIter = 200);

private:
    // Apply Laplacian operator: result = A * x
    void applyLaplacian(const Grid3D& grid, const std::vector<double>& x,
                        std::vector<double>& result) const;

    // Conjugate Gradient solver
    void conjugateGradient(Grid3D& grid, const std::vector<double>& rhs,
                          std::vector<double>& solution,
                          int maxIter, double tol) const;

    // Workspace vectors (reused across calls)
    mutable std::vector<double> r_, p_, Ap_, z_;
};

} // namespace BEMCS
