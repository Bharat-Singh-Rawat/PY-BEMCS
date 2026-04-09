# PYBEMCS-3D (C++ Source)

A 3D Particle-In-Cell (PIC) simulation framework for beam extraction, sputtering/erosion, and electrostatic plasma dynamics in multi-aperture ion thruster grids.

## Features

- Full 3D PIC simulation loop: injection, Poisson solve, Boris push, collisions, sputtering, thermal effects
- Conjugate Gradient Poisson solver with Jacobi preconditioner and Boltzmann electron fluid model
- Boris leapfrog particle mover with E and B field support
- Charge exchange (CEX) collisions via Rapp-Francis cross sections
- Sputtering/erosion model (Xe+ on Mo) with dynamic cell removal
- 3D thermal solver: impact heating, conduction, radiative cooling
- Secondary electron emission (SEE)
- STEP/CAD geometry import via OpenCASCADE
- VTK-based 3D visualization: particles, field slices, damage maps
- Qt6 GUI with parameter controls, diagnostics, and view management
- OpenMP parallelism throughout

## Source Structure

```
src/
├── main.cpp                    # Application entry point
├── core/                       # Physics engine
│   ├── Constants.h             # Physical constants (SI)
│   ├── Vec3.h                  # 3D vector class
│   ├── Particle.h              # Particle data (SoA), Species, SimParams, GridOptic
│   ├── Grid3D.h/cpp            # 3D structured grid (potential, charge density, E/B fields)
│   ├── Simulator3D.h/cpp       # Simulation orchestrator (main loop)
│   ├── PoissonSolver3D.h/cpp   # CG Poisson solver
│   ├── BorisPusher3D.h/cpp     # Boris leapfrog particle pusher
│   ├── ParticleManager.h/cpp   # Injection, boundary removal, charge accumulation
│   ├── CollisionHandler.h/cpp  # CEX collisions, secondary electron emission
│   ├── SputteringModel.h/cpp   # Sputter yield, damage accumulation, cell erosion
│   └── ThermalSolver3D.h/cpp   # Heat equation, radiative cooling
├── geometry/                   # CAD import and meshing
│   ├── Mesh3D.h                # Triangle/tetrahedron data structures
│   ├── MeshGenerator.h/cpp     # Voxelization, parametric grid builder
│   └── STEPImporter.h/cpp      # STEP file reader (OpenCASCADE)
└── gui/                        # Qt6 user interface
    ├── MainWindow.h/cpp        # Main window, menu bar, status bar
    ├── ControlPanel.h/cpp      # Parameter controls, diagnostics panel
    ├── SimulationView3D.h/cpp  # VTK 3D renderer
    ├── MeshingDialog.h/cpp     # Mesh configuration dialog
    └── GeometryImportDialog.h/cpp  # STEP import dialog
```

## Dependencies

### Required

| Library | Version | Purpose |
|---------|---------|---------|
| **CMake** | >= 3.20 | Build system |
| **C++17 compiler** | MSVC 2019+, GCC 9+, or Clang 10+ | Language standard |
| **Eigen3** | >= 3.3 | Linear algebra (header-only) |
| **Qt6** | >= 6.2 | GUI framework (Widgets, OpenGLWidgets) |

### Optional

| Library | Version | Purpose |
|---------|---------|---------|
| **OpenMP** | Any | Multi-threaded parallelism |
| **VTK** | >= 9.0 | 3D visualization (particles, fields, damage) |
| **OpenCASCADE (OCCT)** | >= 7.6 | STEP/IGES CAD file import |

## Installing Dependencies

### Windows (vcpkg)

```bash
# Install vcpkg if not already available
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && bootstrap-vcpkg.bat

# Install dependencies
vcpkg install eigen3 qt6 vtk opencascade
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install build-essential cmake libeigen3-dev \
    qt6-base-dev libvtk9-dev libocct-*-dev
```

### macOS (Homebrew)

```bash
brew install cmake eigen qt@6 vtk opencascade
```

## Building

```bash
cd Cpp3D
mkdir build && cd build

# Full build (all optional features enabled)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_VTK=ON \
         -DUSE_OCCT=ON \
         -DUSE_OPENMP=ON

cmake --build . --config Release
```

On Windows with vcpkg, add the toolchain file:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake \
         -DUSE_VTK=ON -DUSE_OCCT=ON -DUSE_OPENMP=ON

cmake --build . --config Release
```

### CMake Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_OPENMP` | ON | Enable OpenMP parallelism |
| `USE_VTK` | ON | Enable VTK 3D visualization |
| `USE_OCCT` | ON | Enable STEP CAD import |

## Running

```bash
# Linux / macOS
./PYBEMCS3D

# Windows
.\Release\PYBEMCS3D.exe
```

### Workflow

1. Launch the application
2. Configure grid optics (voltages, aperture radii, gaps, thicknesses) in the left panel
3. Set plasma parameters (density, electron/ion temperature, neutral density)
4. Click **BUILD DOMAIN** to generate the 3D mesh and run the initial field solve
5. Click **START SIMULATION** to begin the PIC loop
6. Use view controls to switch between particle, potential, temperature, and damage views
7. Optionally import CAD geometry via **Import STEP Geometry...**

## Physics Models

- **Poisson solver:** CG with Jacobi preconditioner, tolerance 1e-6, max 500 iterations
- **Particle push:** Boris leapfrog with trilinear field interpolation
- **CEX cross section:** sigma(g) = (-0.8821 ln(g) + 15.1262)^2 x 1e-20 m^2 (Rapp-Francis, Xe+ + Xe)
- **Sputter yield:** Y(E) = min(1.05e-4 * max(E - 30, 0)^1.5, 1.0) for Xe+ on Mo
- **SEE yield:** gamma = 0.05 + 1e-4 * E_eV (capped at 1.0)
- **Thermal:** Finite-difference heat equation with Stefan-Boltzmann radiative cooling
