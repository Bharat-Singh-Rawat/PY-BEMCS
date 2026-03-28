<p align="center">
  <img src="PY-BEMCS-ICON.png" alt="PY-BEMCS Ion Extraction Simulation" width="500">
</p>

<h1 align="center">
  Python Beam Extraction & Monte Carlo Simulator (PY-BEMCS)
</h1>

<p align="center">
  <strong>A simulation tool for ion beam extraction, charge-exchange (CEX) physics, ion optics erosion, and multiphysics studies in ion thrusters.</strong>
</p>

---

## Repository Overview

This repository currently includes:

- **Plume MCC Simulator (`charge_exchange_code.m`)**
  - MATLAB model for CEX ion production, plume expansion, and downstream behavior.
- **Grid Digital Twin / EOL (`TransientDigitalTwin.m`)**
  - MATLAB model for accelerated life testing, sputter erosion, and structural failure of accelerator grids.
- **Python Digital Twin (latest modular implementation)**
  - **Runner:** `Python/main.py`
  - **GUI layer:** `Python/gui_window.py`
  - **Physics backend:** `Python/physics_engine.py`
- **Legacy Python single-file app:** `Python/TrainsientDigitalTwin.py`

---

## PY-BEMCS Demo

<p align="center">
  <i>This simulation demonstrates Xe+ beam extraction of a dual-grid ion optics system for Vscreen=1650V and Va=-350V without a neutralizer. As the primary (blue) and the CEX (red) ions erode the screen/accelerator grid, geometry and potential barriers evolve in time. The secondary electrons (green) are emitted due to the impact of primary beam ions and the charge exchange ions on the grids.</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/68f347d0-b163-49d8-8dc5-8b52536709c6" width="900px" autoplay loop muted playsinline>
  </video>
</p>

---
### Key Features

### Core Physics

- **Vectorized particle updates** for high-throughput runtime performance.
- **Reduced electron mass (m_e = M_Xe/100)** to speed up the simulation time.
- **Self-consistent beam extraction** from plasma meniscus and Bohm criteria.
- **CEX collision modeling** with probabilistic scattering, with user defined region in the source code.
- **Dynamic erosion and failure logic** due to CEX ions with in-situ remeshing behavior.

### Python Multiphysics Additions (Latest)
- **Poisson field update with space charge** using both ion and electron density contributions.
- **Neutralizer electron model** with configurable electron injection rate and electron temperature.
- **Thermal-erosion coupling** with simulation modes:
  - `Both`
  - `Thermal`
  - `Erosion`
- **Thermal telemetry** for screen and accelerator grids.

### Visualization and Data export

- Live plasma and damage-map plots.
- Electron backstreaming and beam divergence telemetry.
- Grid temperature map visualization.
- CSV export (iteration, electron backstreaming potential, divergence, grid temperatures).
- GIF recording/export via Pillow.

---

## Installation and Usage

- Current release only simulates physics for Xe+ ions (This is an ongoing project).

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
pip install numpy scipy matplotlib PyQt5 Pillow
```

3. Launch the modular app:

```bash
python Python/main.py
```

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
## Author

**Dr.Bharat Singh Rawat** 

Feel free to reach out for collaborations or questions regarding the code on bharat.bharat22@gmail.com!
