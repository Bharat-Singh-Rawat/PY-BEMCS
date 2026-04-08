#pragma once
// ============================================================================
// PYBEMCS-3D  Physical & Simulation Constants
// ============================================================================

namespace BEMCS {

// Fundamental constants (SI)
constexpr double Q_E      = 1.602176634e-19;   // Elementary charge [C]
constexpr double EPS_0    = 8.8541878128e-12;   // Vacuum permittivity [F/m]
constexpr double K_B      = 1.380649e-23;       // Boltzmann constant [J/K]
constexpr double AMU      = 1.66053906660e-27;  // Atomic mass unit [kg]
constexpr double M_XE     = 131.293 * AMU;      // Xenon ion mass [kg]
constexpr double M_E      = 9.1093837015e-31;   // Electron mass [kg]
constexpr double PI       = 3.14159265358979323846;

// Material: Molybdenum (default grid material)
constexpr double MOLY_DENSITY   = 10280.0;  // [kg/m³]
constexpr double MOLY_CP        = 250.0;    // [J/(kg·K)]
constexpr double MOLY_K         = 138.0;    // [W/(m·K)]
constexpr double MOLY_ALPHA_CTE = 4.8e-6;   // Coeff. thermal expansion [1/K]
constexpr double MOLY_EMISSIVITY = 0.8;

// Stefan-Boltzmann constant
constexpr double SIGMA_SB = 5.670374419e-8; // [W/(m²·K⁴)]

} // namespace BEMCS
