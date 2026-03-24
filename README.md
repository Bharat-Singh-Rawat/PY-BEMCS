<h1 align="center">
  <big>Ion Thruster Digital Twin and MCC Plume Simulator</big>
</h1>

<p align="center">
  <strong>Simulation tools for charge-exchange (CEX) physics, ion optics erosion, and multiphysics digital twin studies in ion thrusters.</strong>
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

## Digital Twin Demo

<p align="center">
  <i>This simulation demonstrates end-of-life transition of a dual-grid ion optics system. As CEX ions erode the accelerator grid, geometry and potential barriers evolve in time.</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/f7e544f9-890e-43f9-bb1e-c623c5e220ad" width="900px" controls autoplay loop muted>
    Your browser does not support the video tag.
  </video>
</p>

---

## Key Features

### Core Physics

- **Vectorized particle updates** for high-throughput runtime performance.
- **CEX collision modeling** with probabilistic scattering.
- **Dynamic erosion and failure logic** with in-situ remeshing behavior.

### Python Multiphysics Additions (Latest)

- **Poisson field update with space charge** using both ion and electron density contributions.
- **Neutralizer electron model** with configurable electron injection rate and electron temperature.
- **Thermal-erosion coupling** with simulation modes:
  - `Both`
  - `Thermal`
  - `Erosion`
- **Thermal telemetry** for screen and accelerator grids.

### Visualization and Telemetry

- Live plasma and damage-map plots.
- Electron backstreaming and beam divergence telemetry.
- Grid temperature map visualization.
- CSV telemetry export (iteration, potential, divergence, temperatures).
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
