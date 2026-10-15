<p align="center">
  <img src="Original/PY-BEMCS-ICON.png" alt="PY-BEMCS Ion Extraction Simulation" width="700">
</p>

<h1 align="center">
  Python Beam Extraction & Monte Carlo Simulator (PY-BEMCS)
</h1>

<p align="center">
  <strong>A high-fidelity 2D3V electrostatic Particle-in-Cell (PIC) & Monte Carlo Collision (MCC) simulation suite for gridded ion thrusters, neutral beam injectors (NBI), and advanced plasma extraction systems.</strong>
</p>

<p align="center">
  <a href="#-key-features--upgrades">Key Features</a> •
  <a href="#-numerical-controls--physics-validation">Physics Controls</a> •
  <a href="#-installation--setup">Installation</a> •
  <a href="#-running-the-code">Usage</a> •
  <a href="#-configuration-guide-configjson">Configuration</a> •
  <a href="#-multi-core-benchmarks">Benchmarks</a> •
  <a href="#-credits--authorship">Credits</a>
</p>

---

## 🌟 Key Features & Upgrades

* **Dual-Kernel Architecture (Taichi GPU + CPU Fallback)**: High-throughput 2D3V Boris particle advancement and bilinear charge deposition accelerated with Taichi (CUDA, Metal, Vulkan) with seamless, vectorized pure-NumPy CPU fallback kernels for maximum portability across any machine.
* **Automated Numerical Physics Controls**: Built-in automated checks for spatial Debye length resolution ($\Delta x, \Delta y \le \lambda_{\mathrm{D}}$), plasma frequency integration limits ($\Delta t \le 2\pi/\omega_{\mathrm{pi}}$), and CFL velocity limits ($v_{\max} \Delta t \le \Delta x$).
* **Customizable Upstream Gap & Sheath Ratio Controls**: Interactive modal dialog (**Settings** button aside Upstream Gap) allowing users to inspect and adjust the physical calculation ratio (Presheath: $k \cdot r_{\mathrm{screen}}$; Entire Bulk Plasma: $N \cdot \lambda_{\mathrm{D}}$), reset to literature recommendations, or set a direct custom gap in millimeters that remains locked during parameter updates.
* **Granular Export Parameter Selectors**: Dedicated pop-up checklist dialogs (**Parameters** buttons beside Data and Particle exports) to selectively choose which simulation variables and particle kinematics ($t, x, y, v_x, v_y, v_z, E, \text{type}$) are saved to `.csv` or `.txt`.
* **Interactive Diagnostics with Matplotlib Toolbars & Window Controls**: Multi-window diagnostic suite (Electric Potential, Mesh Discretization, Total Energy, Total Charge, PPC, IEDF, Hardware Performance) with built-in Matplotlib navigation toolbars (Pan, Zoom, Home, Subplot Configuration, Save) and native desktop window controls (minimize, maximize, close).
* **Modular Clean Architecture**: Refactored modular structure cleanly partitioned into `gui/app/` (UI layout, controls, export, canvas), `physics/mixins/` (domain, field solve, particle push, thermal expansion), and `diagnostics/` for high stability and testability.
* **Modular JSON Configuration (`config.json`)**: Extensible JSON configuration supporting arbitrary $N$-grid optics, discharge chamber parameters, and reaction cross-section tables, complete with in-app hot-reloading.
* **In-App Standalone Executable Builder**: Compile the application into a single-file executable directly from the GUI (**Settings → Build .exe...**) using an asynchronous background thread (`QThread`) with live progress milestone tracking.
* **Multi-Core Benchmarking Suite (`benchmarks/`)**: Automated parallel multi-process parametric sweeps for CEX erosion rates, Electron Backstreaming (EBS) limits, primary beam impingement / grid scraping, and perveance-divergence curves.
* **Thermo-Mechanical Cantilever Grid Deflection**: Real-time modeling of grid aperture deformation driven by Euler--Bernoulli cantilever thermal expansion ($\delta = \frac{\alpha \Delta T L^2}{2t}$), with dynamic mesh distortion and Poisson refactorization.
* **Live Telemetry & Divergence Analysis**: Real-time tracking of 95th-percentile beam divergence ($\theta_{95}$), midline divergence, centerline saddle-point potential, and interactive axial divergence evaluation plane controls positioned cleanly under the telemetry graphs.

---

## 🔬 Numerical Controls & Physics Validation

When generating the simulation domain (`build_domain`), the engine automatically verifies the physical consistency of your grid resolution, sheath boundaries, and numerical time step:

1. **Debye Length Resolution ($\lambda_{\mathrm{D}}$)**:
   $$\lambda_{\mathrm{D}} = \sqrt{\frac{\varepsilon_0 T_{\mathrm{e,up}}}{e \cdot (0.61 n_0)}}, \qquad \Delta x \le \lambda_{\mathrm{D}} \quad \text{and} \quad \Delta y \le \lambda_{\mathrm{D}}$$
   Prevents unphysical numerical grid heating and space-charge instabilities.
2. **Plasma Frequency Integration Limit**:
   $$\omega_{\mathrm{pi}} = \sqrt{\frac{n_0 e^2}{m_{\mathrm{ion}} \varepsilon_0}}, \qquad \Delta t \le \frac{2\pi}{\omega_{\mathrm{pi}}}$$
   Ensures the time step is fine enough to resolve high-frequency collective plasma oscillations.
3. **Courant--Friedrichs--Lewy (CFL) Velocity Bound**:
   $$v_{\max} = v_{\mathrm{Bohm}} + 4 v_{\mathrm{thermal,i}} = \sqrt{\frac{Z e T_{\mathrm{e}}}{m_{\mathrm{ion}}}} + 4\sqrt{\frac{Z e T_{\mathrm{i}}}{m_{\mathrm{ion}}}}, \qquad \frac{\Delta x}{\Delta t} \ge v_{\max}, \quad \frac{\Delta y}{\Delta t} \ge v_{\max}$$
   Guarantees that fast particles do not skip entire mesh cells within a single time increment.
4. **Upstream Sheath & Extraction Gap Sizing**:
   - **Presheath Mode (Default)**: Automatically sized from the screen grid aperture radius: $\text{gap}_{\mathrm{up}} = 0.75 \cdot r_{\mathrm{screen}}$.
   - **Entire Bulk Plasma Mode**: Sized via Debye sheath criteria: $\text{gap}_{\mathrm{up}} = N \cdot \lambda_{\mathrm{D}}$, where $N = 80$ ($n_0 \le 10^{17}\,\si{\per\meter\cubed}$), $N = 40$ ($10^{17} < n_0 \le 4\times 10^{17}\,\si{\per\meter\cubed}$), and $N = 30$ ($n_0 > 4\times 10^{17}\,\si{\per\meter\cubed}$).
   - **Custom Manual Override**: Users can specify an arbitrary physical gap directly in millimeters via the **Settings** dialog.
5. **Dynamic Domain & Geometry Modes**:
   - **Axial Extent**: $L_x = \text{gap}_{\mathrm{up}} + \sum_k (t_k + g_k) + L_{\mathrm{downstream}}$ dynamically computed from the grid stack.
   - **`half_hole`**: Axisymmetric half-aperture with specular symmetry reflection at $y = 0$.
   - **`one_hole`**: Full single-aperture extraction domain with periodic boundary conditions along $y$.
   - **`two_holes`**: Dual-aperture coupling domain with periodic boundaries along $y$.

---

## 📁 Repository Structure

```text
Python/
├── main.py                         # PyQt5 GUI application entry point
├── physics_engine.py               # DigitalTwinSimulator compatibility wrapper
├── run_simulation_from_config.py   # Headless simulation runner reading config.json
├── config.json                     # Primary JSON simulation configuration
├── sample.txt                      # Example custom cross-section / SEE data table
├── gui/                            # Modular PyQt5 GUI implementation
│   ├── app/                        # Application layout, control mixins & canvas
│   │   ├── main_window.py          # Central DigitalTwinApp window
│   │   ├── ui_builder.py           # Control panel, buttons & layout builder
│   │   ├── dialogs_mixin.py        # Pop-up dialog openers & state handlers
│   │   ├── config_mixin.py         # Config persistence, JSON syncing & gap auto-update
│   │   ├── sim_control.py          # Worker lifecycle & simulation loop controls
│   │   ├── export.py               # Selective CSV data & particle export routines
│   │   └── canvas.py               # Real-time Matplotlib canvas & telemetry plots
│   ├── dialogs.py                  # Modal configuration dialogs (Upstream Gap, Exports, etc.)
│   ├── widgets.py                  # Specialized GUI widgets (ScientificSpinBox)
│   └── workers/                    # Background threads (SimulationWorker, PyInstallerWorker)
├── physics/                        # Computational PIC-MCC physics core
│   ├── simulator.py                # DigitalTwinSimulator state container & orchestrator
│   ├── mesh_utils.py               # Debye gap & automatic domain zoning routines
│   ├── kernels_taichi.py           # GPU-accelerated PIC particle push & deposition
│   ├── kernels_cpu.py              # Vectorized NumPy CPU fallback kernels
│   └── mixins/                     # Physics mixins (domain, fields, particles, thermal)
├── diagnostics/                    # Standalone interactive diagnostic windows
│   ├── charge_window.py            # Total charge conservation & net current monitor
│   ├── energy_window.py            # Total, kinetic & potential energy monitor
│   ├── field_window.py             # 2D electric potential contour & vector visualizer
│   ├── mesh_window.py              # Computational grid & zoning inspection window
│   ├── iedf_window.py              # Ion energy distribution function (IEDF) spectrum
│   ├── ppc_window.py               # Particles-per-cell (PPC) density diagnostic
│   └── perf_window.py              # Real-time hardware performance & loop timer
├── benchmarks/                     # Parallel multi-core physics sweep scripts
│   ├── benchmark_cex.py            # CEX erosion rate vs. neutral density sweep
│   ├── benchmark_ebs.py            # EBS saddle-point potential vs. accelerator voltage
│   ├── benchmark_impingement.py    # Primary ion grid impingement fraction vs. perveance
│   ├── benchmark_perveance.py      # Beam divergence vs. perveance across multiple gaps
│   └── benchmark_perveance_Vs_Sweep.py # Divergence vs. Screen grid voltage sweep
├── tests/                          # Automated unit test suite
│   ├── test_upstream_gap_dialog.py # Tests for upstream gap ratios & custom input
│   ├── test_export_parameters.py   # Tests for selective parameter export dialogs
│   ├── test_divergence_plane.py    # Tests for divergence evaluation plane dialog
│   └── test_mesh_window.py         # Tests for mesh inspection diagnostic
├── docs/                           # Physics and user manuals (LaTeX & PDF)
│   ├── PY-BEMCS_Manual.tex         # Complete LaTeX physics reference manual
│   └── PY-BEMCS_Manual.pdf         # Compiled user manual
└── README.md                       # This documentation file
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Python 3.10+ (Python 3.12/3.13 recommended).

```bash
# Navigate to the Python directory
cd Python

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies
```bash
pip install numpy scipy matplotlib PyQt5 Pillow taichi pyinstaller
```

---

## 💻 Running the Code

### 1. Interactive Graphical Application (GUI)
```bash
python main.py
```
* **Upstream Gap & Sheath Settings**: Click the small **Settings** button next to **Upstream Gap (mm)** to inspect the active ratio, adjust the calculation factor, or enter a custom physical gap.
* **Selective Data & Particle Export**: Click **Parameters** beside either **Export Data** or **Export Particle Data** to choose exactly which variables and kinematics columns to write.
* **Interactive Diagnostics**: Launch external diagnostic windows (**Diagnostics** menu or buttons) equipped with Matplotlib navigation toolbars (Pan/Zoom/Save) and desktop window minimize/maximize controls.
* **Hot-Reloading Configurations**: Go to **Settings → Open Config JSON...** or **Settings → Reload config.json** to swap configurations on the fly.
* **Build Standalone Executable**: Click **Settings → Build .exe...**, select a destination directory, and track real-time compilation progress.
* **Configure Species & Materials**: Manage species via **Beam → Ion Species...**, material presets via **Materials → Grid Material...**, and custom reaction tables via **Beam → Cross-Section Manager...**.

### 2. Headless Batch Execution via `config.json`
Run simulations in headless / cluster mode without launching a GUI:
```bash
python run_simulation_from_config.py
```

### 3. Parallel Multi-Core Benchmarking Sweeps
Execute multi-core parametric physics sweeps (each grid gap runs on an independent CPU core):
```bash
python benchmarks/benchmark_cex.py
python benchmarks/benchmark_ebs.py
python benchmarks/benchmark_impingement.py
python benchmarks/benchmark_perveance.py
python benchmarks/benchmark_perveance_Vs_Sweep.py
```

---

## ⚙️ Configuration Guide (`config.json`)

The simulator is driven by `config.json`. Below is a standard multi-grid configuration:

```json
{
  "beam_species": {
    "mass_amu": 131.293,
    "charge_state": 1
  },
  "grid_material": {
    "preset": "Molybdenum"
  },
  "simulation": {
    "n0_plasma": 7.73e17,
    "Te_up": 8.5,
    "Ti": 0.034,
    "Tn": 400.0,
    "n0": 5.27e18,
    "Accel": 0.5,
    "Thresh": 10000.0,
    "sim_mode": "Both",
    "geometry": "half_hole"
  },
  "discharge_chamber": {
    "radius_m": 0.02,
    "length_m": 0.10,
    "n_cusp": 4,
    "pitch_mm": 3.0
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

## 📊 Multi-Core Benchmarks

| Benchmark Script | Physics Parameter Swept | Objective |
| :--- | :--- | :--- |
| **`benchmark_cex.py`** | Neutral density ($n_{\mathrm{n}} = 10^{18}\dots 5\times 10^{19}\,\si{\per\meter\cubed}$) | Quantify accelerator grid erosion rates from CEX ions across multiple grid gaps. |
| **`benchmark_ebs.py`** | Accelerator voltage ($V_{\mathrm{a}} = -400\dots 0\,\si{\volt}$) | Track centerline saddle-point potential against the $-5\,\si{\volt}$ EBS risk barrier. |
| **`benchmark_impingement.py`** | Upstream plasma density / Perveance ($P = I_{\mathrm{ion}}/V_{\mathrm{tot}}^{1.5}$) | Measure the primary beam interception and grid scraping percentage. |
| **`benchmark_perveance.py`** | Plasma density over 60 points | Map beam divergence angle vs. perveance curves (underfocused, optimal, overfocused). |
| **`benchmark_perveance_Vs_Sweep.py`** | Screen voltage ($V_{\mathrm{s}} = 600\dots 2000\,\si{\volt}$) | Evaluate voltage extraction efficiency on beam divergence at constant density. |

---

## 👥 Credits & Authorship

* **Original Creator & Lead Author**:
  * **Dr. Bharat Singh Rawat** — [GitHub](https://github.com/Bharat-Singh-Rawat/PY-BEMCS) | [Email](mailto:bharat.bharat22@gmail.com)
* **Version 2.0 Architecture, Physics Controls & Diagnostic Extensions**:
  * **Nick Magrin** — [GitHub](https://github.com/nickmagrin-pixel/PY-BEMCS) | [Email](mailto:nick.magrin@gmail.com)

---

## 📄 License & Citation

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

```bibtex
@software{PY-BEMCS,
  author       = {Bharat Singh Rawat and Nick Magrin},
  title        = {Python Beam Extraction and Monte Carlo Simulator (PY-BEMCS)},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/nickmagrin-pixel/PY-BEMCS}},
}
```
