"""
Physical constants and material property presets.
"""

Q_ELEM = 1.602e-19       # elementary charge [C]
M_XE = 131.293 * 1.6605e-27 # mass of Xenon ion [kg]
K_B = 1.380649e-23       # Boltzmann constant [J/K]
EPS_0 = 8.854e-12        # permittivity of free space [F/m]
SIGMA_SB = 5.67e-8       # Stefan-Boltzmann constant [W/(m^2 K^4)]

MATERIAL_PRESETS = {
    'Molybdenum': {
        'k': 138.0, 'rho': 10280.0, 'cp': 250.0,
        'emissivity': 0.80, 'alpha': 4.8e-6,
        'E_mod': 329e9, 'Y_coeff': 1.05e-4, 'E_th': 30.0
    },
    'Steel (SS316)': {
        'k': 16.3, 'rho': 8000.0, 'cp': 500.0,
        'emissivity': 0.60, 'alpha': 16.0e-6,
        'E_mod': 193e9, 'Y_coeff': 2.8e-4, 'E_th': 25.0
    },
    'Titanium': {
        'k': 21.9, 'rho': 4507.0, 'cp': 520.0,
        'emissivity': 0.50, 'alpha': 8.6e-6,
        'E_mod': 116e9, 'Y_coeff': 1.8e-4, 'E_th': 20.0
    },
    'Graphite': {
        'k': 120.0, 'rho': 2200.0, 'cp': 710.0,
        'emissivity': 0.85, 'alpha': 3.0e-6,
        'E_mod': 11e9, 'Y_coeff': 3.5e-4, 'E_th': 15.0
    },
}
