
# BEMCS-3D (Under development)

<img width="1280" height="720" alt="3D" src="https://github.com/user-attachments/assets/3768c522-2b8a-4171-b665-2929015710a0" />

**3D Particle-In-Cell Beam Extraction & Monte Carlo Simulation**

There may be many issues related to 3D code that need to be fixed; this section is under an experimental phase.
A full 3D C++ extension of the PYBEMCS code for studying beam extraction and sputtering/erosion physics in ion thruster grids. Developed by Dr. Bharat Singh Rawat with Claude code.

---
**An example showing the erosion of the ion optics (accelerated damage threshold)**

<img width="800" height="720" alt="3D" src="https://github.com/user-attachments/assets/9db221b9-0526-48c4-b872-24163dbfb0f1" />

## Features

### Physics Engine
- **3D PIC Solver** — Full 3D electrostatic Particle-In-Cell with Boris particle pusher
- **Poisson Solver** — Conjugate Gradient with Jacobi preconditioner and Boltzmann electron fluid model
- **Charge Exchange (CEX)** — Rapp-Francis cross section for Xe+ on Xe collisions
- **Sputtering/Erosion** — Empirical Xe+ on Mo yield model with damage accumulation and dynamic cell removal
- **Thermal Model** — 3D finite-difference heat conduction, radiative cooling (Stefan-Boltzmann), and particle impact heating
- **Secondary Electron Emission** — Energy-dependent SEE yield from grid surfaces
- **RF Co-Extraction** — Time-varying grid potentials for electron extraction studies

### Geometry & Meshing
- **STEP File Import** — Load CAD geometry via OpenCASCADE with configurable tessellation quality
- **Interactive 3D Face/Body Picking** — Embedded VTK preview in the import dialog lets you click directly on faces or bodies in the 3D view to select them, with bidirectional sync to the assignment tree
- **Face / Body Selection Mode** — Toggle between Face mode (individual CAD surfaces) and Body mode (entire solids) for faster bulk assignment of voltages or particle sources
- **Cut Plane Preview** — Clip the 3D preview along XZ, XY, or YZ planes to reveal and select internal/hidden faces
- **Per-Face Color Coding** — Each CAD face is rendered with a distinct color; selected faces highlight in yellow while others dim for clear visual feedback
- **Automatic Voxelization** — Converts imported surface meshes to structured PIC grids
- **Built-in Multi-Aperture Grids** — Parametric grid optics with voltage, thickness, gap, radius, and chamfer
- **Meshing Controls** — Adjustable cell size, auto-sizing, refinement, and domain padding

### Visualization & GUI
- **VTK 3D Rendering** — Interactive 3D view with rotation, zoom, and pan
- **Solid Surface Rendering** — Grid geometry rendered as solid surfaces with transparent domain box
- **Cut Plane Visualization** — Half-section clipping along X, Y, or Z axes via VTK clipping planes
- **Particle Visualization** — Color-coded by species (ions=blue, CEX=red, electrons=green)
- **Field Slicing** — Potential and temperature fields on configurable XY/XZ/YZ slice planes
- **Real-Time Diagnostics** — Beam divergence, saddle point potential, mean energy, grid temperatures
- **Process Log Window** — Docked bottom panel with timestamped simulation messages
- **Sputtering & Thermal Maps** — Dedicated dock windows for sputtering damage and thermal contour visualization
- **Erosion Map (Accel Grid, Downstream)** — Live 2D heat-map of cumulative groove depth across the full (x, y) downstream face of the accel grid with a colorbar in μm; every column contributes, so the display is statistically much smoother than a 1D slice and doesn't need ions to land on a centre-line
- **Animated GIF Export** — Record simulation frames and export as animated GIF (File > Record GIF / Save GIF)
- **Dimensional Scaling** — 1x/10x/100x self-similar PIC scaling: divides domain, cell size, grid optics, neutralizer positions, voltages and timestep by the factor; multiplies plasma and neutral densities by the factor squared. E-field on particles is preserved and the Debye length shrinks with the geometry.
- **Dark Theme UI** — Modern Qt6 interface with scrollable control panel
- **Data Export** — CSV export of all particle positions, velocities, and energies

### Performance
- **OpenMP Parallelism** — All compute-heavy loops (field solve, particle push, thermal conduction)
- **Structure-of-Arrays** — Cache-friendly particle storage for optimal memory access
- **Configurable Batched Rendering** — Steps/Frame spinbox (1–500, default 10) controls physics steps per render frame; increase to advance the beam faster visually
- **Adaptive Rendering** — Particle sub-sampling for smooth visualization at high particle counts

---

## Dependencies

| Library | Version | Purpose | Required |
|---------|---------|---------|----------|
| **CMake** | >= 3.20 | Build system | Yes |
| **C++17 Compiler** | MSVC 2019+ / GCC 9+ / Clang 10+ | Compilation | Yes |
| **Eigen3** | >= 3.3 (or 5.x) | Linear algebra (header-only) | Yes |
| **Qt6** | >= 6.2 | GUI framework | Yes |
| **VTK** | >= 9.0 | 3D visualization | Optional (fallback OpenGL) |
| **OpenCASCADE** | >= 7.6 (7.9+ recommended) | STEP file import | Optional |
| **OpenMP** | Any | Parallel acceleration | Optional |

### Installing Dependencies on Windows (Conda — Recommended)

Using Miniforge/Conda is the fastest way to get all pre-built dependencies on Windows:

```powershell
# Install Miniforge (if not already installed)
winget install CondaForge.Miniforge3

# Create environment with all dependencies
conda create -n pybemcs3d python=3.11
conda activate pybemcs3d
conda install -c conda-forge eigen qt6-main vtk occt cmake ninja
```

You also need the MSVC compiler (Visual Studio Build Tools):
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --quiet --wait"
```

### Installing Dependencies on Windows (vcpkg — Alternative)

```powershell
# Install vcpkg if you haven't already
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies (compiles from source — can take 30+ minutes)
.\vcpkg install eigen3:x64-windows
.\vcpkg install "qtbase[widgets,opengl]:x64-windows"
.\vcpkg install vtk:x64-windows
.\vcpkg install opencascade:x64-windows
```

### Installing Dependencies on Ubuntu 24.04 (Tested)

> **Important — VTK on Ubuntu 24.04 is NOT usable with this project.**
> The archive ships `libvtk9-dev` compiled against **Qt5**, but this project requires **Qt6**. Trying to configure with system VTK fails with `INTERFACE_QT_MAJOR_VERSION of Qt5::Core does not agree with ... QT_MAJOR_VERSION ... determined for PYBEMCS3D`. You must supply a Qt6-compiled VTK. Two options below.

**Step 1 — Install everything except VTK via apt:**

```bash
sudo apt install -y cmake g++ libeigen3-dev \
    qt6-base-dev libqt6opengl6-dev \
    libocct-modeling-algorithms-dev libocct-data-exchange-dev
```

**Step 2 — Get a Qt6-compiled VTK. Pick one:**

**Option A — Conda (recommended, fastest):**
```bash
# Install Miniforge if you don't have conda
wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init bash    # open a new shell afterwards

# Env with Qt6-based VTK
conda create -n pybemcs3d -c conda-forge -y vtk qt6-main
```
Every session: `conda activate pybemcs3d` before building or running.

**Option B — Build VTK from source against Qt6** (no conda needed; ~30-60 min compile):
```bash
cd ~ && git clone --branch v9.3.1 --depth 1 https://gitlab.kitware.com/vtk/vtk.git
cmake -S vtk -B vtk-build -DCMAKE_INSTALL_PREFIX=$HOME/vtk-install \
      -DVTK_GROUP_ENABLE_Qt=YES -DVTK_QT_VERSION=6 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build vtk-build -j$(nproc)
cmake --install vtk-build
```
The install lands in `$HOME/vtk-install`.

> **Gotcha: `paraview` conflicts with `libvtk9-dev`.** If you ever install the Ubuntu `libvtk9-dev` anyway, apt will refuse because `python3-paraview` pins a different `python3-vtk9`. Solutions: `sudo apt install --no-install-recommends libvtk9-dev libvtk9-qt-dev`, or temporarily `sudo apt remove python3-paraview paraview` and reinstall after. **But note:** `libvtk9-dev` is Qt5 on Ubuntu 24.04 and still won't build this project — it's not a shortcut.

### Installing Dependencies on macOS (Homebrew)

```bash
brew install cmake eigen qt@6 vtk opencascade
```

---

## Building

> **Common pitfall.** `cmake ..` (configure) and `cmake --build .` (build) are **two separate commands** — do not combine them with `--build` in the configure line. `cmake` will reject `-j$(nproc)` there with `Unknown argument -j12`.

### Full Build on Ubuntu (Tested)

**With conda (Option A from above):**
```bash
conda activate pybemcs3d
cd Cpp3D && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
         -DUSE_VTK=ON -DUSE_OCCT=ON -DUSE_OPENMP=ON
cmake --build . -j$(nproc)
```

**With VTK built from source in `$HOME/vtk-install` (Option B from above):**
```bash
cd Cpp3D && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DVTK_DIR=$HOME/vtk-install/lib/cmake/vtk-9.3 \
         -DUSE_VTK=ON -DUSE_OCCT=ON -DUSE_OPENMP=ON
cmake --build . -j$(nproc)
```

### Full Build (generic, assuming VTK is discoverable)

```bash
cd Cpp3D
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_VTK=ON \
         -DUSE_OCCT=ON \
         -DUSE_OPENMP=ON
cmake --build . --config Release
```

### With Conda (Windows — Recommended)

```powershell
conda activate pybemcs3d
cd Cpp3D
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64 `
         -DCMAKE_PREFIX_PATH="$env:CONDA_PREFIX\Library" `
         -DUSE_VTK=ON -DUSE_OCCT=ON -DUSE_OPENMP=ON
cmake --build . --config Release
```

### With vcpkg (Windows)

```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release `
         -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake `
         -DUSE_VTK=ON -DUSE_OCCT=ON -DUSE_OPENMP=ON
cmake --build . --config Release
```

### Minimal Build (Qt only, no VTK/OCCT)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_VTK=OFF \
         -DUSE_OCCT=OFF
cmake --build . --config Release
```

---

## Running

```bash
./PYBEMCS3D          # Linux/macOS (if VTK/Qt in default library path)
.\Release\PYBEMCS3D  # Windows
```

### Ubuntu: if VTK isn't installed system-wide

If you built VTK into `$HOME/vtk-install` (Option B above), point the loader at its libs:

```bash
LD_LIBRARY_PATH=$HOME/vtk-install/lib ./PYBEMCS3D
```

To make this permanent, add to `~/.bashrc`:
```bash
export LD_LIBRARY_PATH="$HOME/vtk-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

If you built with conda (Option A), activating the env is enough:
```bash
conda activate pybemcs3d && ./PYBEMCS3D
```

### Ubuntu: VS Code snap terminal issue

If you launched the terminal from a **snap-packaged VS Code**, running the binary fails with:
```
symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0:
undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
```
The snap injects Ubuntu 20.04's glibc into the shell, which is binary-incompatible with a binary compiled against the system glibc. Fixes:

- **Preferred:** launch the binary from a native terminal (GNOME Terminal via `Ctrl+Alt+T`, or any `.deb`-installed terminal).
- **Workaround in the snap shell** — strip the snap env:
  ```bash
  env -i HOME=$HOME DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY \
      PATH=/usr/local/bin:/usr/bin:/bin \
      LD_LIBRARY_PATH=$HOME/vtk-install/lib \
      ./PYBEMCS3D
  ```
- **Permanent fix:** replace snap VS Code with the `.deb` from [code.visualstudio.com](https://code.visualstudio.com/download).

---

## Usage

### Quick Start

1. **Launch** the application
2. **Configure grid optics** in the left panel (voltages, aperture sizes, etc.)
3. **Set plasma parameters** (density, temperature, neutral density)
4. Click **BUILD DOMAIN** to generate the 3D mesh and initial field solve — this also computes the correct macro-particle weight for injection
5. Click **START SIMULATION** to begin the PIC loop
6. Use **View Controls** to rotate, slice, and toggle field/particle display

### Tuning Beam Visibility

Two controls under **4. 3D DOMAIN** let you trade physical accuracy for faster visual feedback:

| Control | Default | Effect |
|---|---|---|
| **Timestep dt (ns)** | 1 ns | Physical time per step. Raise to 5–10 ns to see ions traverse the grid stack in fewer iterations. The ion CFL limit at 1650 V / 0.05 mm cells is ~1 ns; values above this reduce accuracy. |
| **Steps/Frame** | 10 | Physics steps computed per render frame. Raise to 50–100 to advance the beam faster on screen without changing per-step accuracy. |

> **Important:** always click **BUILD DOMAIN** before starting the simulation. The macro-particle weight is computed from plasma density, electron temperature, aperture area, and `dt` at build time. Skipping this step results in near-zero ion injection.

### Importing CAD Geometry

1. Click **Import STEP Geometry...** or go to File > Import STEP
2. Browse to your `.step` or `.stp` file
3. Set the input units and tessellation quality
4. Click **Import & Preview** — the 3D preview shows all CAD faces in distinct colors
5. **Select faces visually**: Click on any face in the 3D preview to select it. The clicked face highlights in yellow and the corresponding tree row is selected automatically
6. **Use Body mode** (dropdown above the tree) to select entire solids at once — clicking any face on a body selects all its faces
7. **Enable Cut Plane** (checkbox above the 3D view) to clip the model and reveal internal faces. Choose the cut axis (XZ, XY, or YZ) to expose hidden geometry
8. Assign boundary voltages or particle source definitions to selected faces/bodies using the **Set Voltage** and **Set as Particle Source** buttons
9. Click **Apply to Simulation**, then **BUILD DOMAIN**

### Meshing Options

Click **Meshing Options...** to configure:
- **Cell Size** — Target voxel size in mm (smaller = more accurate but slower)
- **Auto-size** — Automatically compute cell size from geometry dimensions
- **Max Cells** — Safety limit to prevent memory exhaustion
- **Domain Padding** — Extra space around geometry for field boundary conditions

---

## Project Structure

```
Cpp3D/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
└── src/
    ├── main.cpp                # Application entry point
    ├── core/                   # Physics engine
    │   ├── Constants.h         # Physical constants (SI)
    │   ├── Vec3.h              # 3D vector type
    │   ├── Particle.h          # Particle data structures & SimParams
    │   ├── Grid3D.h/cpp        # 3D structured grid with field storage
    │   ├── PoissonSolver3D.h/cpp  # CG Poisson solver + Boltzmann electrons
    │   ├── BorisPusher3D.h/cpp # 3D Boris particle pusher
    │   ├── ParticleManager.h/cpp  # Injection, removal, charge accumulation
    │   ├── CollisionHandler.h/cpp # CEX collisions & SEE
    │   ├── SputteringModel.h/cpp  # Erosion damage model
    │   ├── ThermalSolver3D.h/cpp  # 3D heat conduction & radiative cooling
    │   └── Simulator3D.h/cpp   # Simulation loop orchestrator
    ├── geometry/               # CAD import & meshing
    │   ├── Mesh3D.h/cpp        # Surface/volume mesh data structures
    │   ├── STEPImporter.h/cpp  # OpenCASCADE STEP reader
    │   └── MeshGenerator.h/cpp # Voxelization & grid generation
    └── gui/                    # Qt6 user interface
        ├── MainWindow.h/cpp    # Main application window
        ├── SimulationView3D.h/cpp # VTK 3D visualization widget
        ├── ControlPanel.h/cpp  # Parameter sidebar & diagnostics
        ├── MeshingDialog.h/cpp # Mesh settings dialog
        ├── GeometryImportDialog.h/cpp # STEP import dialog
        └── GifWriter.h         # Animated GIF recording & export
```

---

## MSVC Compatibility Notes

The codebase includes fixes for building with MSVC (Visual Studio 2022):

- **OpenMP 2.0**: MSVC only supports OpenMP 2.0, which requires signed integer loop variables. All `#pragma omp parallel for` loops use `int` instead of `size_t`.
- **`M_E` macro conflict**: The electron mass constant is named `M_ELECTRON` to avoid collision with the `M_E` (Euler's number) macro from `<cmath>`.
- **OpenCASCADE 7.9+**: Uses the new `TKDESTEP`/`TKDE` libraries instead of the legacy `TKSTEP`/`TKSTEPBase`/`TKSTEPAttr`/`TKXDESTEP`.
- **Eigen 5.x**: Compatible with both Eigen 3.x and the new Eigen 5.x release.
- **OpenMP `collapse`**: The `collapse` clause on nested loops is ignored by MSVC but does not cause errors.

---

## Physics Model Summary

### Coordinate Convention
- **Z-axis** is the beam/extraction axis
- **XY-plane** is the transverse plane
- Default domain: Lx=3 mm, Ly=3 mm, Lz=10 mm

### Poisson Equation (Electrostatics)
```
∇²V = -ρ/ε₀
```
Solved with Preconditioned Conjugate Gradient on a uniform 3D grid. Boltzmann electron density provides the plasma sheath:
```
n_e = n₀ · exp((V - V_plasma) / T_e)
```

### Boris Particle Pusher
Standard leapfrog Boris algorithm with full 3D electric and magnetic field support:
1. Half E-field acceleration
2. Magnetic rotation (cross products)
3. Half E-field acceleration
4. Position update

### Sputtering Yield (Xe+ → Mo)
```
Y(E) = min(1.05×10⁻⁴ · max(E - 30, 0)^1.5, 1.0)
```

### CEX Cross Section (Xe+ + Xe → Xe + Xe+)
Rapp-Francis parameterization:
```
σ(g) = (-0.8821·ln(g) + 15.1262)² × 10⁻²⁰ m²
```

---

## License

This project is part of the PYBEMCS suite developed by Dr.Bharat Singh Rawat.
