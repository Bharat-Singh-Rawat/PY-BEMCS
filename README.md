<p align="center">
  <img src="PY-BEMCS-ICON.png" alt="PY-BEMCS Ion Extraction Simulation" width="700">
</p>

<h1 align="center">
  Python Beam Extraction & Monte Carlo Simulator (PY-BEMCS)
</h1>

<p align="center">
  <strong>A simulation tool for ion beam extraction, charge-exchange (CEX) physics, ion optics erosion, and multiphysics studies in ion thrusters, NBI, and Ion sources.</strong>
</p>

---
## Repository Overview

This repository currently includes:

- **Plume MCC Simulator (`charge_exchange_code.m`)**
  - MATLAB model for CEX ion production, plume expansion, and downstream behavior with a custom Faraday probe for measuring CX flux.
- **Matlab Beam extraction model EOL (`TransientDigitalTwin.m`)**
  - MATLAB test model for accelerated life testing, sputter erosion, and structural failure of accelerator grids.
- **Python Digital Twin (latest modular implementation)**
  - **GUI app:** `Python/main.py`
  - **Physics backend:** `Python/physics_engine.py`
  - **Headless runner:** `Python/run_simulation_from_config.py` (reads `config.ini`)
- **Legacy Python single-file app:** `Python/transient_digital_twin.py`
- **C++ 3D PIC Framework (`Cpp3D/`)**
  - Full 3D C++ Particle-In-Cell simulation with Qt6 GUI, VTK visualization, and OpenCASCADE STEP import.
  - Boris pusher, CG Poisson solver, CEX collisions, sputtering/erosion, thermal model, and SEE.
  - See [`Cpp3D/README.md`](Cpp3D/README.md) for build instructions and details (This part is still under development).
---

## PY-BEMCS Demo

<p align="center">
  <i>This simulation demonstrates Xe+ beam extraction of a dual-grid ion optics system for Vscreen=1650V and Va=-350V ,n_plasma=1e16/m3 without a neutralizer. As the primary and the CEX ions erode the screen/accelerator grid, geometry and potential barriers evolve in time. The secondary electrons are emitted due to the impact of primary beam ions and the charge exchange ions on the grids.</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/0c6d243d-daea-444e-beed-82dcb215ad47" width="500px" autoplay loop muted playsinline>
  </video>
</p>

---
<p align="center">
  <i>Example of RF-based co-extraction of electrons and ions (me=mXe/100) .</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/ce1ca173-62e1-433d-aff4-9aaafdbe117a" width="500px" autoplay loop muted playsinline>
  </video>
</p>

---

<p align="center">
  <i>Space charge neutralization of the ion beam at Vs=800V and Va=600V for different electron injection rates.</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/f541d3a0-647e-4827-9dda-5e4ce4f7e235" width="500px" autoplay loop muted playsinline>
  </video>
</p>
---

### Core Physics

- **Vectorized particle updates** for high-throughput runtime performance.
- **Artificial electron mass approximation (m_e = M_ion/1000)** to speed up the simulation time.
- **Self-consistent beam extraction** from plasma meniscus and Bohm criteria.
- **CEX collision modeling** with probabilistic scattering (Birdsall/Roy plume model), or user-imported cross-section data.
- **Dynamic erosion and failure logic** due to CEX ions with in-situ remeshing behavior.
- **RF-based Coextraction** of electron and ion beam.
- **User-defined multi-grid beam extraction** for different kinds of sources.

### Configurable Ion Beam Species (Menubar: Beam > Ion Species)
- Define the beam ion by **atomic/molecular mass** (amu) and **charge state** (+1, +2, etc.).
- Built-in presets for common species: Xe, Kr, Ar, N2, O2, H2, He, Hg, Cs.
- Custom mass and charge for any user-defined ion.

### Cross-Section Data Import (Menubar: Beam > Cross-Section Manager)
- Import CSV tables of **Energy (eV) vs Cross-Section (m^2)** for different reaction channels:
  - Charge Exchange (CX)
  - Secondary Electron Yield (SEE)
  - Custom reactions (e.g., air-mixture gas interactions)
- **Visualise** imported data on a log-log plot.
- **Fit a cubic spline** (in log-log space) with adjustable smoothing so that intermediate energy values are interpolated during the simulation.
- Multiple datasets can be loaded, inspected, and removed.
- When a fitted CX or SEE spline is present it automatically replaces the built-in analytical model during the run.

### Grid Material Selection (Menubar: Materials > Grid Material)
- Choose from built-in presets or define fully custom properties:

| Material | k (W/m/K) | rho (kg/m^3) | cp (J/kg/K) | alpha (1/K) | E (GPa) |
|---|---|---|---|---|---|
| Molybdenum | 138 | 10 280 | 250 | 4.8e-6 | 329 |
| Steel (SS316) | 16.3 | 8 000 | 500 | 16.0e-6 | 193 |
| Titanium | 21.9 | 4 507 | 520 | 8.6e-6 | 116 |
| Graphite | 120 | 2 200 | 710 | 3.0e-6 | 11 |

- Configurable parameters: thermal conductivity, density, specific heat, emissivity, thermal expansion coefficient, Young's modulus, sputter yield coefficient, and sputter threshold energy.

### Cantilever Thermal Deformation
- Grids deform like a **cantilever beam**: clamped at the top wall (`Y = Ly`), free at the aperture edge (`Y = r`).
- Tip deflection driven by thermal stress: `delta = alpha * dT * L^2 / (2*t)`.
- Quadratic (Euler-Bernoulli) deflection profile applied to the grid boundary mask, producing visible bowing in the beam trajectory plot.
- Materials with high thermal expansion and low conductivity (e.g., Steel) exhibit significantly faster deformation than refractory metals (e.g., Mo).

### Python Multiphysics Additions
- **Poisson field update with space charge** using both ion and electron density contributions.
- **Neutralizer electron model** with configurable electron injection rate and electron temperature.
- **Thermal-erosion coupling** with simulation modes:
  - `Both`
  - `Thermal`
  - `Erosion`
- **Thermal monitoring** for screen and accelerator grids.

### Visualization and Data Export

- Live plasma and damage-map plots.
- Electron backstreaming and beam divergence telemetry.
- Grid temperature map visualization.
- CSV export (iteration, electron backstreaming potential, divergence, grid temperatures).
- Particle kinematics export (time, position, velocity, energy, particle type).
- GIF recording/export via Pillow.
---

## Installation and Usage

### MATLAB Workflow

1. Prerequisite: MATLAB R2015b or newer.
2. Open MATLAB in the repository root.
3. Run:

```matlab
TransientDigitalTwin  % Grid erosion and EOL study
charge_exchange_code  % Plume/CEX study
```

### Python Workflow (Recommended, Modular)

1. Use Python 3.10+.
2. Install dependencies:

```bash
pip install numpy scipy matplotlib PyQt5 Pillow taichi
```

3. Launch the GUI app:

```bash
python Python/main.py
```

> **Qt + OpenFOAM users:** If your shell sources OpenFOAM (or any toolchain
> that injects `/usr/lib/x86_64-linux-gnu` into `LD_LIBRARY_PATH`), the system
> Qt libraries can shadow PyQt5's bundled copies and cause
> `Could not load the Qt platform plugin "xcb"`. `main.py` now self-heals by
> prepending the PyQt5 wheel's own `Qt5/lib` directory to `LD_LIBRARY_PATH`
> and re-exec-ing once; a shell-level equivalent is also provided as
> `Python/run.sh`.

4. Or run headless from a configuration file:

```bash
python Python/run_simulation_from_config.py
```

Edit `Python/config.ini` to set beam species, grid material, cross-section file paths, grid geometry, plasma parameters, and terminal output options.

### Python Workflow (Legacy Single File)

```bash
python Python/TrainsientDigitalTwin.py
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. 

This means you are free to use, share, and adapt this code for academic, research, and personal projects, provided you give appropriate credit. It **may not** be used for commercial purposes.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

*For commercial inquiries or alternative licensing, please contact the author.*

---
## Citation

If you use this software in your work, please cite it using the following formats:

**Plain Text:**
Bharat Singh Rawat. (2026). *Python Beam Extraction and Monte Carlo Simulator (PY-BEMCS)* [Source code]. GitHub. https://github.com/Bharat26031992/PY-BEMCS

**BibTeX:**
```bibtex
@software{PY-BEMCS,
  author       = {Bharat Singh Rawat},
  title        = {Python Beam Extraction and Monte Carlo Simulator (PY-BEMCS)},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/Bharat26031992/PY-BEMCS](https://github.com/Bharat26031992/PY-BEMCS)}},
}
```

## Author
**Dr.Bharat Singh Rawat** 

Feel free to reach out for collaborations or questions regarding the code on bharat.bharat22@gmail.com!
