
"""
This script runs the PY-BEMCS simulation without a GUI.
All simulation parameters are loaded from a configuration file (config.json).
The script will print selected simulation parameters to the terminal during the run.

Instructions:
1.  Make sure you have all the required libraries installed:
    pip install numpy scipy taichi

2.  Create a file named 'config.ini' in the same directory as this script.
    Copy the example configuration from the comments at the end of this file into 'config.ini'.

3.  Modify the 'config.ini' file to set your desired simulation parameters.
    You can choose which parameters to print to the terminal by editing the 'terminal_output' section.

4.  Run this script from your terminal:
    python run_simulation_from_config.py
"""
#import configparser
import json
from logging import config
import re
import time
import os
import numpy as np
from scipy.interpolate import UnivariateSpline
from physics_engine import DigitalTwinSimulator

def parse_config(config_path='config.json'):
    """
    Parses the config.json file and returns:
    - params: dictionary for DigitalTwinSimulator
    - output_selection: dictionary controlling terminal prints
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    params = {}

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
        }
    }

    # --- Beam species ---
    beam = config.get('beam_species', {})
    params['beam_mass_amu'] = beam.get('mass_amu', 131.293)
    params['beam_charge_state'] = beam.get('charge_state', 1)

    # --- Grid material ---
    mat_cfg = config.get('grid_material', {})
    preset_name = mat_cfg.get('preset', 'Molybdenum').strip()

    if preset_name in MATERIAL_PRESETS:
        params['_material'] = (preset_name, MATERIAL_PRESETS[preset_name])
    else:
        params['_material'] = ('Custom', {
            'k': mat_cfg.get('k', 138.0),
            'rho': mat_cfg.get('rho', 10280.0),
            'cp': mat_cfg.get('cp', 250.0),
            'emissivity': mat_cfg.get('emissivity', 0.80),
            'alpha': mat_cfg.get('alpha', 4.8e-6),
            'E_mod': mat_cfg.get('E_mod', 329e9),
            'Y_coeff': mat_cfg.get('Y_coeff', 1.05e-4),
            'E_th': mat_cfg.get('E_th', 30.0)
        })

    # --- Advanced settings ---
    adv = config.get('advanced_settings', {})
    params['neut_x'] = adv.get('neut_x', 19.9)
    params['neut_r'] = adv.get('neut_r', 3.0)
    params['V_plasma_offset'] = adv.get('V_plasma_offset', 20.0)
    params['m_e_ratio'] = adv.get('m_e_ratio', 1000.0)
    if 'Lx' in adv:
        params['Lx'] = adv['Lx']
    if 'Ly' in adv:
        params['Ly'] = adv['Ly']

    # --- Cross-section files ---
    cs_store = {}
    cs_cfg = config.get('cross_sections', {})
    smoothing = cs_cfg.get('spline_smoothing', 0.0)

    cs_files = {
        'CX': cs_cfg.get('cx_file', '').strip(),
        'SEE': cs_cfg.get('see_file', '').strip(),
        'Custom': cs_cfg.get('custom_file', '').strip(),
    }

    for label, fpath in cs_files.items():
        if not fpath:
            continue
        if not os.path.isfile(fpath):
            print(f"Warning: cross-section file not found for {label}: {fpath}")
            continue
        try:
            try:
                raw = np.loadtxt(fpath, delimiter=None, comments='#')
            except Exception:
                raw = np.loadtxt(fpath, delimiter=',', comments='#', skiprows=1)

            if raw.ndim != 2 or raw.shape[1] < 2:
                print(f"Warning: {fpath} must have at least 2 columns. Skipping.")
                continue

            energy = raw[:, 0]
            cs = raw[:, 1]
            order = np.argsort(energy)
            energy, cs = energy[order], cs[order]

            log_e = np.log10(np.maximum(energy, 1e-30))
            log_cs = np.log10(np.maximum(cs, 1e-50))
            spline = UnivariateSpline(log_e, log_cs, s=smoothing, k=3)

            cs_store[label] = {
                'energy': energy,
                'cs': cs,
                'spline': spline,
                'type': label
            }
            print(f"Loaded {label} cross-section: {len(energy)} points from {fpath}")
        except Exception as e:
            print(f"Warning: failed to load {label} CS from {fpath}: {e}")

    params['_cs_store'] = cs_store

    # --- Simulation ---
    sim_cfg = config.get('simulation', {})
    params['n0_plasma'] = sim_cfg.get('n0_plasma', 1e17)
    params['Te_up'] = sim_cfg.get('Te_up', 3.0)
    params['Ti'] = sim_cfg.get('Ti', 2.0)
    params['Tn'] = sim_cfg.get('Tn', 300.0)
    params['n0'] = sim_cfg.get('n0', 1e20)
    params['Accel'] = sim_cfg.get('Accel', 0.1)
    params['Thresh'] = sim_cfg.get('Thresh', 10000.0)
    params['sim_mode'] = sim_cfg.get('sim_mode', 'Both')
    params['geometry'] = sim_cfg.get('geometry', 'half_hole')

    # --- RF co-extraction ---
    rf_cfg = config.get('rf_co_extraction', {})
    params['rf_enable'] = rf_cfg.get('rf_enable', False)
    params['rf_grid_idx'] = rf_cfg.get('rf_grid_idx', 0)
    params['rf_freq'] = rf_cfg.get('rf_freq', 13.56)
    params['rf_amp'] = rf_cfg.get('rf_amp', 500.0)

    # --- Neutralizer ---
    neut_cfg = config.get('neutralizer', {})
    params['neut_rate'] = neut_cfg.get('neut_rate', 0)
    params['Te'] = neut_cfg.get('Te', 5.0)

    # --- Grids ---
    params['grids'] = []
    for grid_cfg in config.get('grids', []):
        params['grids'].append({
        'V': grid_cfg.get('V', 0.0),
        't': grid_cfg.get('t', 1.0),
        'gap': grid_cfg.get('gap', 1.0),
        'r': grid_cfg.get('r', 1.0),
        'cham': grid_cfg.get('cham', 0.0)
        })

    if len(params['grids']) == 0: # Added a check to ensure that at least one grid is defined in the config file, since the simulation cannot run without any grids.
        raise ValueError("No grids found in config.json. Add at least one entry in the 'grids' list.")

    # --- Terminal output ---
    out_cfg = config.get('terminal_output', {})
    output_selection = {
        'grid_temperatures': out_cfg.get('grid_temperatures', True),
        'beam_divergence': out_cfg.get('beam_divergence', True),
        'saddle_point_potential': out_cfg.get('saddle_point_potential', True),
        'mean_particle_energy': out_cfg.get('mean_particle_energy', True),
        'iteration_time': out_cfg.get('iteration_time', True)
    }

    return params, output_selection

def run_simulation():
    """
    Initializes and runs the simulation from config.json.
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        params, output_selection = parse_config(config_path)
        print("Using config file:", config_path)
    except Exception as e:
        print(f"Error parsing config file: {e}")
        return

    sim = DigitalTwinSimulator()

    # Apply beam species
    mass_amu = params.pop('beam_mass_amu', 131.293)
    charge_state = params.pop('beam_charge_state', 1)
    sim.m_ion = mass_amu * 1.6605e-27
    sim.m_XE = sim.m_ion
    sim.Z_ion = charge_state
    sim.q_ion = charge_state * sim.q
    sim.m_e = sim.m_ion / params.get('m_e_ratio', 1000.0)
    print(f"Beam species: {mass_amu:.3f} amu, charge +{charge_state}")

    # Apply grid material
    mat_name, mat_props = params.pop('_material', ('Molybdenum', None))
    if mat_props:
        sim.set_material(props=mat_props)
    else:
        sim.set_material(name='Molybdenum')
    print(f"Grid material: {mat_name} (k={sim.mat_k:.1f}, rho={sim.mat_rho:.0f}, cp={sim.mat_cp:.0f})")

    # Apply user cross-sections
    cs_store = params.pop('_cs_store', {})
    sim.user_cs = cs_store
    if cs_store:
        print(f"Cross-section datasets loaded: {', '.join(cs_store.keys())}")

    print("Building simulation domain...")
    sim.build_domain(params)
    print(f"Domain built with {len(params['grids'])} grids")
    for i, g in enumerate(params['grids'], start=1): #Adding a print to confirm what the simulator received
        print(f"Built Grid {i}: V={g['V']} V, t={g['t']} mm, gap={g['gap']} mm, r={g['r']} mm, cham={g['cham']} deg")
    print("Domain built successfully.")

    print("Starting simulation...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            start_time = time.time()
            remeshed, min_pot, current_div, T_grids = sim.step(params)
            end_time = time.time()

            if sim.iteration % 10 == 0:
                output_str = f"Iteration: {sim.iteration} | "

                if output_selection.get('grid_temperatures'):
                    temp_str = ', '.join([f"G{i+1}: {T-273.15:.1f}C" for i, T in enumerate(T_grids)])
                    output_str += f"Grid Temps: [{temp_str}] | "

                if output_selection.get('beam_divergence'):
                    output_str += f"Divergence: {current_div:.2f} deg | "

                if output_selection.get('saddle_point_potential'):
                    output_str += f"Saddle Potential: {min_pot:.2f} V | "

                if output_selection.get('mean_particle_energy'):
                    if sim.num_p > 0:
                        v_sq = sim.p_vx[:sim.num_p]**2 + sim.p_vy[:sim.num_p]**2 + sim.p_vz[:sim.num_p]**2
                        mean_energy = np.mean((0.5 * sim.m_ion * v_sq) / sim.q)
                        output_str += f"Mean Ion Energy: {mean_energy:.2f} eV | "

                if output_selection.get('iteration_time'):
                    output_str += f"Step Time: {end_time - start_time:.4f} s"

                print(output_str)

            if remeshed:
                print("Domain has been remeshed due to thermal or erosion effects.")

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")

if __name__ == "__main__":
    run_simulation() 
"""
================================================================================
EXAMPLE config.ini file
================================================================================

[beam_species]
# --- Ion Beam Species ---
# Atomic or molecular mass of the beam ion in atomic mass units (amu)
mass_amu = 131.293
# Charge state of the beam ion (1 = singly charged, 2 = doubly charged, etc.)
charge_state = 1

[cross_sections]
# --- Reaction Cross-Section Tables (optional) ---
# Provide paths to CSV files with two columns: Energy (eV), Cross-Section (m^2)
# Lines starting with '#' are treated as comments and skipped.
# Leave blank or comment out to use built-in analytical models.
#
# Charge exchange cross-section file
cx_file =
# Secondary electron yield file (Energy [eV] vs yield [dimensionless])
see_file =
# Custom reaction cross-section file
custom_file =
# Spline smoothing factor (0 = interpolating, passes through all points)
spline_smoothing = 0.0

[grid_material]
# --- Grid Material ---
# Preset name: Molybdenum, Steel (SS316), Titanium, Graphite, or Custom
preset = Molybdenum
# Manual properties (only used when preset = Custom)
thermal_conductivity_W_per_mK = 138.0
density_kg_per_m3 = 10280.0
specific_heat_J_per_kgK = 250.0
emissivity = 0.80
thermal_expansion_1_per_K = 4.8e-6
youngs_modulus_Pa = 329e9
sputter_yield_coeff = 1.05e-4
sputter_threshold_eV = 30.0

[simulation]
# --- General Simulation Parameters ---
plasma_density_m-3 = 1e17
upstream_electron_temp_eV = 3.0
ion_temp_eV = 2.0
neutral_temp_K = 300
neutral_density_m-3 = 1e20
# Factor to accelerate thermal and erosion effects
acceleration_factor = 1.0
# Damage threshold for cell erosion before it's removed
cell_fail_threshold = 10000.0
# Simulation mode: Both, Thermal, or Erosion
simulation_mode = Both


[rf_co_extraction]
# --- RF Co-extraction Parameters ---
enable_rf = no
# 0 for Grid 1, 1 for Grid 2, etc.
rf_grid_index = 0
frequency_mhz = 13.56
amplitude_v = 500


[neutralizer]
# --- Neutralizer Parameters ---
# Number of electron macroparticles to inject per step
electron_injection_rate = 0
# Temperature of injected electrons in eV
electron_temp_eV = 5.0


# --- Grid Definitions ---
# Add or remove grid sections as needed. They are processed in order (grid_1, grid_2, ...).
[grid_1]
dc_voltage_v = 1650
thickness_mm = 1.0
gap_to_next_mm = 1.0
hole_radius_mm = 1.0
chamfer_deg = 0

[grid_2]
dc_voltage_v = -350
thickness_mm = 1.0
gap_to_next_mm = 1.0
hole_radius_mm = 0.6
chamfer_deg = 0


[terminal_output]
# --- Terminal Output Selection ---
# Set to 'yes' or 'no' to enable or disable printing of each parameter.
grid_temperatures = yes
beam_divergence = yes
saddle_point_potential = yes
mean_particle_energy = yes
iteration_time = yes

"""
