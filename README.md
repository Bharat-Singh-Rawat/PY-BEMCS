<p align="center">
  <img src="PY-BEMCS-ICON.png" alt="PY-BEMCS Ion Extraction Simulation" width="700">
</p>

<h1 align="center">
  Python Beam Extraction & Monte Carlo Simulator (PY-BEMCS)
</h1>

<p align="center">
  <strong>A high-fidelity 2D3V electrostatic Particle-in-Cell (PIC) & Monte Carlo Collision (MCC) simulation suite for gridded ion thrusters, neutral beam injectors (NBI), and advanced plasma extraction systems.</strong>
</p>

<p align="center">
  <a href="#-repository-overview">Repository Overview</a> •
  <a href="#-key-features--capabilities">Key Features</a> •
  <a href="#-numerical-controls--physics-validation">Physics Controls</a> •
  <a href="#-py-bemcs-demos">Demos</a> •
  <a href="#-installation--setup">Installation</a> •
  <a href="#-usage-workflows">Usage</a> •
  <a href="#-configuration-guide-configjson">Configuration</a> •
  <a href="#-multi-core-benchmarks">Benchmarks</a> •
  <a href="#-credits--authorship">Credits</a> •
  <a href="#-license--citation">Citation</a>
</p>

---

## 📁 Repository Overview

This repository includes a comprehensive multi-tier simulation ecosystem:

- **Python Digital Twin (Latest Modular Implementation — Version 2.0)**
  - **GUI Application (`Python/main.py`):** Interactive PyQt5 desktop suite featuring live plasma visualization, multi-grid optics geometry editor, RF co-extraction controls, IEDF/EEDF diagnostics, cross-section manager, and an in-app PyInstaller `.exe` compiler.
  - **Physics Engine (`Python/physics_engine.py`):** High-performance `DigitalTwinSimulator` featuring a dual-kernel architecture (Taichi GPU acceleration + pure-NumPy CPU fallback), 2D3V Boris particle pusher, SuperLU/CuPy Poisson solver, in-situ sputter erosion, thermal conduction, and automated numerical validation checks.
  - **Headless Batch Runner (`Python/run_simulation_from_config.py`):** Cluster- and CLI-friendly headless runner reading configuration from `Python/config.json`.
  - **Multi-Core Benchmarking Suite (`Python/benchmarks/`):** Parallel sweep scripts for CEX erosion rates, Electron Backstreaming (EBS) limits, primary beam grid impingement, and perveance-divergence curves.
  - **Documentation & User Manual (`Python/docs/`):** Complete LaTeX reference manual (`PY-BEMCS_Manual.tex`) and compiled PDF guide (`PY-BEMCS_Manual.pdf`).
- **Legacy Python Single-File App (`Python/transient_digital_twin.py`)**
  - Standalone monolithic GUI reference implementation.
- **Plume MCC Simulator (`Matlab/charge_exchange_code.m`)**
  - MATLAB model for CEX ion production, plume expansion, and downstream behavior with a custom Faraday probe model.
- **MATLAB Beam Extraction Model EOL (`Matlab/TransientDigitalTwin.m`)**
  - MATLAB test model for accelerated life testing, sputter erosion, and structural failure of accelerator grids.
- **C++ 3D PIC Framework (`Cpp3D/`)**
  - Full 3D Particle-In-Cell simulation with Qt6 GUI, VTK visualization, and OpenCASCADE STEP CAD import (under active development; see [`Cpp3D/README.md`](Cpp3D/README.md)).

---

## 🌟 Key Features & Capabilities

### ⚡ Dual-Kernel Hardware Acceleration
- **Taichi GPU Acceleration:** Particle pusher (2D3V Boris algorithm), bilinear charge density deposition, and 2D thermal conduction run natively on GPU via Taichi (supporting Vulkan, CUDA, and Metal backends).
- **Vectorized CPU Fallback:** Seamless, pure-NumPy CPU fallback kernels ensure 100% functionality on CPU-only machines or within frozen executable bundles.
- **Sparse Poisson Field Solvers:** High-performance direct Poisson solving via SciPy SuperLU with optional GPU-accelerated CuPy sparse LU support.

### 🔬 Advanced Core Physics
- **Self-Consistent Beam Extraction:** Meniscus formation governed by upstream Bohm velocity criterion and sheath thermodynamics.
- **Artificial Electron Mass Approximation:** Accelerated multi-scale time-stepping ($m_e = M_{\mathrm{ion}} / 100\dots 1000$).
- **Charge-Exchange (CEX) Collisions:** Monte Carlo collision modeling with probabilistic scattering (Birdsall/Roy model) and customizable cross-section data tables.
- **In-Situ Dynamic Sputter Erosion:** Real-time grid mass removal and hole geometry enlargement driven by primary and CEX ion impacts.
- **Thermo-Mechanical Cantilever Grid Deflection:** Dynamic aperture distortion based on Euler–Bernoulli cantilever thermal expansion ($\delta = \frac{\alpha \Delta T L^2}{2t}$), with live mesh updating and Poisson refactorization.
- **RF Co-Extraction:** Modulated RF extraction potentials ($V_{\mathrm{RF}}\sin(2\pi f t)$) for simultaneous or alternating electron-ion beam extraction.
- **Neutralizer Electron Injection:** Configurable downstream electron injection rate and electron thermal temperature.
- **Arbitrary $N$-Grid Ion Optics:** Fully customizable multi-grid systems (Screen, Accel, Decel, Ground) with per-grid thickness, gap, aperture radius, chamfer angle, and DC potential.
- **Configurable Particle Injection Cutoff:** Option to specify finite injection windows (`inj_time_us`) for pulse extraction or decay studies.

### 📊 Live Diagnostics, Telemetry & GUI Tools
- **Live Visualizations:** Real-time ion/electron trajectory tracking, grid temperature heatmaps, sputter damage maps, and radial accelerator grid erosion groove profiles.
- **95th-Percentile Beam Divergence ($\theta_{95}$):** Continuous tracking of beam envelope divergence.
- **Saddle-Point Potential & EBS Monitoring:** Centerline saddle-point potential tracking to detect Electron Backstreaming risk barriers ($V_{\mathrm{saddle}} \le -5\,\text{V}$).
- **Energy Distribution Function Viewer (IEDF / EEDF):** Dedicated diagnostic window to inspect ion and electron energy spectra.
- **Cross-Section Manager:** Visual log-log cubic spline fitting and interpolation for custom Charge Exchange (CX), Secondary Electron Emission (SEE), and gas mixture interaction data.
- **Material Preset Library:** Built-in properties for Molybdenum, SS316 Stainless Steel, Titanium, and Graphite, with fully customizable thermal and sputtering coefficients.
- **Configurable Ion Species:** Mass (amu) and charge state presets (Xe, Kr, Ar, N2, O2, H2, He, Hg, Cs) or custom user-defined ions.
- **In-App Standalone Executable Builder:** Build single-file `.exe` packages directly from the menu (**Settings → Build .exe...**) using non-blocking background threads (`QThread`) with live progress milestone reporting.
- **Asynchronous Data Exports:** Export telemetry logs to CSV, full particle kinematics ($x, y, v_x, v_y, v_z, E$) to CSV, and multi-frame GIF recordings with live frame counters.

---

## 🔬 Numerical Controls & Physics Validation

When generating the simulation domain (`build_domain`), the physics engine automatically verifies the numerical and physical consistency of your grid resolution and time step:

1. **Debye Length Resolution ($\lambda_{\mathrm{D}}$)**:
   $$\lambda_{\mathrm{D}} = \sqrt{\frac{\varepsilon_0 T_{\mathrm{e,up}}}{e \cdot (0.61 n_0)}}, \qquad \Delta x \le \lambda_{\mathrm{D}} \quad \text{and} \quad \Delta y \le \lambda_{\mathrm{D}}$$
   Prevents unphysical numerical grid heating and non-physical space-charge oscillations.

2. **Plasma Frequency Integration Limit**:
   $$\omega_{\mathrm{pi}} = \sqrt{\frac{n_0 e^2}{m_{\mathrm{ion}} \varepsilon_0}}, \qquad \Delta t \le \frac{2\pi}{\omega_{\mathrm{pi}}}$$
   Ensures the temporal resolution captures collective plasma oscillations.

3. **Courant–Friedrichs–Lewy (CFL) Velocity Bound**:
   $$v_{\max} = v_{\mathrm{Bohm}} + 4 v_{\mathrm{thermal,i}} = \sqrt{\frac{Z e T_{\mathrm{e}}}{m_{\mathrm{ion}}}} + 4\sqrt{\frac{Z e T_{\mathrm{i}}}{m_{\mathrm{ion}}}}, \qquad \frac{\Delta x}{\Delta t} \ge v_{\max}, \quad \frac{\Delta y}{\Delta t} \ge v_{\max}$$
   Guarantees that particles do not traverse more than one mesh cell per time increment.

4. **Dynamic Domain & Geometry Modes**:
   - **Axial Domain Length:** $L_x = 1.5\,\text{mm} + \sum_k (t_k + g_k) + 3.0\,\text{mm}$ automatically computed from the grid geometry.
   - **`half_hole`:** Axisymmetric half-aperture with specular symmetry reflection at $y = 0$ (default, fastest).
   - **`one_hole`:** Full single-aperture extraction domain with periodic boundary conditions along $y$.
   - **`two_holes`:** Dual-aperture coupling domain with periodic boundary conditions along $y$.

---

## 🎬 PY-BEMCS Demos

<p align="center">
  <i>Simulation demonstrating Xe+ beam extraction in a dual-grid ion optics system (Vscreen=1650V, Vaccel=-350V, n_plasma=1e16 m⁻³) without a neutralizer. Geometry and potential barriers dynamically evolve as primary and CEX ions erode the grids.</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/0c6d243d-daea-444e-beed-82dcb215ad47" width="500px" autoplay loop muted playsinline>
  </video>
</p>

---

<p align="center">
  <i>RF-based co-extraction of electrons and ions (m_e = m_Xe / 100).</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/ce1ca173-62e1-433d-aff4-9aaafdbe117a" width="500px" autoplay loop muted playsinline>
  </video>
</p>

---

<p align="center">
  <i>Space charge neutralization of an ion beam (Vs=800V, Va=-600V) under various electron injection rates.</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/f541d3a0-647e-4827-9dda-5e4ce4f7e235" width="500px" autoplay loop muted playsinline>
  </video>
</p>

---

## 🛠️ Grid Material Properties

Users can select built-in presets or configure custom thermophysical and sputtering properties:

| Material | $k$ (W/m·K) | $\rho$ (kg/m³) | $c_p$ (J/kg·K) | $\alpha$ (10⁻⁶/K) | $E$ (GPa) | $Y_{\mathrm{coeff}}$ | $E_{\mathrm{th}}$ (eV) |
|---|---|---|---|---|---|---|---|
| **Molybdenum** | 138.0 | 10 280 | 250 | 4.8 | 329 | 1.05e-4 | 30.0 |
| **Steel (SS316)** | 16.3 | 8 000 | 500 | 16.0 | 193 | 2.80e-4 | 25.0 |
| **Titanium** | 21.9 | 4 507 | 520 | 8.6 | 116 | 1.80e-4 | 20.0 |
| **Graphite** | 120.0 | 2 200 | 710 | 3.0 | 11 | 3.50e-4 | 15.0 |

---

## 🚀 Installation & Setup

### Python Workflow (Recommended)

#### 1. Create and Activate Virtual Environment
Python 3.10+ is required (Python 3.12 recommended).

```bash
# Navigate to the repository root
cd PY-BEMCS

# Linux / macOS
python3 -m venv Python/.venv
source Python/.venv/bin/activate

# Windows (PowerShell)
python -m venv Python\.venv
Python\.venv\Scripts\Activate.ps1
```

#### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install numpy scipy matplotlib PyQt5 Pillow taichi pyinstaller
```

*(Optional for GPU Poisson)*: On Linux with NVIDIA CUDA 12:
```bash
pip install 'cupy-cuda12x<14' nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cuda-runtime-cu12
```

---

## 💻 Usage Workflows

### 1. Interactive Graphical Application (GUI)
Launch the primary PyQt5 GUI interface:
```bash
python Python/main.py
```
- **Hot-Reload Configurations:** Switch simulation parameters on the fly via **Settings → Open Config JSON...** or **Settings → Reload config.json**.
- **Build Standalone Executable:** Click **Settings → Build .exe...**, choose an output path, and track compilation.
- **Diagnostics & Windows:** Inspect energy spectra via **Show Energy Dist. (IEDF/EEDF)**, manage materials in **Materials → Grid Material...**, and inspect cross-sections in **Beam → Cross-Section Manager...**.

### 2. Headless Batch Execution via `config.json`
Run high-throughput simulations on remote clusters or headless servers:
```bash
python Python/run_simulation_from_config.py
```

### 3. Parallel Multi-Core Benchmarking Sweeps
Execute parametric physics sweeps (each grid gap or operating point runs on an independent core):
```bash
python Python/benchmarks/benchmark_cex.py
python Python/benchmarks/benchmark_ebs.py
python Python/benchmarks/benchmark_impingement.py
python Python/benchmarks/benchmark_perveance.py
python Python/benchmarks/benchmark_perveance_Vs_Sweep.py
```

### 4. MATLAB Workflow
Prerequisite: MATLAB R2015b or newer. Open MATLAB in `Matlab/` and run:
```matlab
TransientDigitalTwin   % Accelerated grid erosion and EOL study
charge_exchange_code   % Plume expansion & CEX study
```

---

## ⚙️ Configuration Guide (`config.json`)

Simulations are configured via `Python/config.json`:

```json
{
  "beam_species": {
    "mass_amu": 131.293,
    "charge_state": 1
  },
  "grid_material": {
    "preset": "Molybdenum"
  },
  "advanced_settings": {
    "neut_x": 19.9,
    "neut_r": 3.0,
    "V_plasma_offset": 1000.0,
    "m_e_ratio": 1000.0,
    "Lx": 20.0,
    "Ly": 3.0
  },
  "simulation": {
    "n0_plasma": 7.73e17,
    "Te_up": 8.5,
    "Ti": 0.034,
    "Tn": 400.0,
    "n0": 5.27e18,
    "Accel": 0.5,
    "Thresh": 10000.0,
    "sim_mode": "Thermal",
    "geometry": "half_hole"
  },
  "rf_co_extraction": {
    "rf_enable": false,
    "rf_grid_idx": 0,
    "rf_freq": 13.56,
    "rf_amp": 500.0
  },
  "neutralizer": {
    "neut_rate": 0,
    "Te": 5.0
  },
  "grids": [
    { "V": 1300.0, "t": 1.2, "gap": 0.9, "r": 1.3, "cham": 20.0 },
    { "V": -150.0, "t": 0.6, "gap": 0.5, "r": 0.8, "cham": 0.0 },
    { "V": 50.0,   "t": 0.5, "gap": 4.0, "r": 1.2, "cham": 0.0 }
  ],
  "cross_sections": {
    "cx_file": "",
    "see_file": "",
    "custom_file": "",
    "spline_smoothing": 0.0
  },
  "terminal_output": {
    "grid_temperatures": true,
    "beam_divergence": true,
    "saddle_point_potential": true,
    "mean_particle_energy": true,
    "iteration_time": true
  }
}
```

---

## 📊 Multi-Core Benchmarking Suite

| Benchmark Script | Physics Parameter Swept | Objective |
| :--- | :--- | :--- |
| **`benchmark_cex.py`** | Neutral density ($n_{\mathrm{n}} = 10^{18}\dots 5\times 10^{19}\,\text{m}^{-3}$) | Quantify accelerator grid erosion rates from CEX ions across various grid gaps. |
| **`benchmark_ebs.py`** | Accelerator voltage ($V_{\mathrm{a}} = -400\dots 0\,\text{V}$) | Map centerline saddle-point potential against the $-5\,\text{V}$ Electron Backstreaming threshold. |
| **`benchmark_impingement.py`** | Plasma density / Perveance ($P = I_{\mathrm{ion}}/V_{\mathrm{tot}}^{3/2}$) | Measure primary beam interception and direct grid scraping fractions. |
| **`benchmark_perveance.py`** | Plasma density (60 points) | Trace perveance-divergence curves identifying underfocused, optimal, and overfocused regimes. |
| **`benchmark_perveance_Vs_Sweep.py`** | Screen voltage ($V_{\mathrm{s}} = 600\dots 2000\,\text{V}$) | Evaluate extraction voltage scaling on beam divergence at constant plasma density. |

---

## 👥 Credits & Authorship

- **Original Creator & Lead Author:**
  - **Dr. Bharat Singh Rawat** — [GitHub](https://github.com/Bharat26031992/PY-BEMCS) | [Email](mailto:bharat.bharat22@gmail.com)
- **Version 2.0 Architecture, Physics Controls & Diagnostic Extensions:**
  - **Nick Magrin** — [GitHub](https://github.com/nickmagrin-pixel/PY-BEMCS) | [Email](mailto:nick.magrin@gmail.com)

---

## 📄 License & Citation

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. 

You are free to use, share, and adapt this software for academic and non-commercial research, provided proper attribution is given.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

### Citation

If you use this software in your research, please cite:

```bibtex
@software{PY-BEMCS,
  author       = {Bharat Singh Rawat},
  title        = {Python Beam Extraction and Monte Carlo Simulator (PY-BEMCS)},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/nickmagrin-pixel/PY-BEMCS}},
}
```

