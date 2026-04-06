
"""
This script runs the PY-BEMCS simulation without a GUI.
All simulation parameters are loaded from a configuration file (config.ini).
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
import configparser
import time
import numpy as np
from physics_engine import DigitalTwinSimulator

def parse_config(config_path='config.ini'):
    """
    Parses the config.ini file and returns a parameters dictionary
    for the DigitalTwinSimulator.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    if not config.sections():
        raise FileNotFoundError(f"Config file '{config_path}' not found or is empty. "
                                f"Please create it and fill it with parameters.")

    params = {}

    # --- Simulation Parameters ---
    sim_params = config['simulation']
    params['n0_plasma'] = sim_params.getfloat('plasma_density_m-3', 1e17)
    params['Te_up'] = sim_params.getfloat('upstream_electron_temp_eV', 3.0)
    params['Ti'] = sim_params.getfloat('ion_temp_eV', 2.0)
    params['Tn'] = sim_params.getfloat('neutral_temp_K', 300)
    params['n0'] = sim_params.getfloat('neutral_density_m-3', 1e20)
    params['Accel'] = sim_params.getfloat('acceleration_factor', 1)
    params['Thresh'] = sim_params.getfloat('cell_fail_threshold', 10000.0)
    params['sim_mode'] = sim_params.get('simulation_mode', 'Both')

    # --- RF Co-extraction ---
    rf_params = config['rf_co_extraction']
    params['rf_enable'] = rf_params.getboolean('enable_rf', False)
    params['rf_grid_idx'] = rf_params.getint('rf_grid_index', 0)
    params['rf_freq'] = rf_params.getfloat('frequency_mhz', 13.56)
    params['rf_amp'] = rf_params.getfloat('amplitude_v', 500)

    # --- Neutralizer ---
    neut_params = config['neutralizer']
    params['neut_rate'] = neut_params.getfloat('electron_injection_rate', 0)
    params['Te'] = neut_params.getfloat('electron_temp_eV', 5.0)

    # --- Grids ---
    params['grids'] = []
    grid_sections = sorted([s for s in config.sections() if s.startswith('grid_')])
    for section_name in grid_sections:
        grid_section = config[section_name]
        grid = {
            'V': grid_section.getfloat('dc_voltage_v'),
            't': grid_section.getfloat('thickness_mm'),
            'gap': grid_section.getfloat('gap_to_next_mm'),
            'r': grid_section.getfloat('hole_radius_mm'),
            'cham': grid_section.getfloat('chamfer_deg'),
        }
        params['grids'].append(grid)

    # --- Terminal Output ---
    output_params = config['terminal_output']
    output_selection = {
        'grid_temperatures': output_params.getboolean('grid_temperatures', True),
        'beam_divergence': output_params.getboolean('beam_divergence', True),
        'saddle_point_potential': output_params.getboolean('saddle_point_potential', True),
        'mean_particle_energy': output_params.getboolean('mean_particle_energy', True),
        'iteration_time': output_params.getboolean('iteration_time', True),
    }

    return params, output_selection

def run_simulation():
    """
    Initializes and runs the simulation from the config file.
    """
    try:
        params, output_selection = parse_config()
    except Exception as e:
        print(f"Error parsing config file: {e}")
        return

    sim = DigitalTwinSimulator()

    print("Building simulation domain...")
    sim.build_domain(params)
    print("Domain built successfully.")

    print("Starting simulation...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            start_time = time.time()
            remeshed, min_pot, current_div, T_grids = sim.step(params)
            end_time = time.time()

            if sim.iteration % 10 == 0:  # Print every 10 iterations
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
                        v_sq = sim.p_vx[:sim.num_p]**2 + sim.p_vy[:sim.num_p]**2
                        mean_energy = np.mean((0.5 * sim.m_XE * v_sq) / sim.q)
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

if __name__ == '__main__':
    run_simulation()
    
"""
================================================================================
EXAMPLE config.ini file
================================================================================

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
