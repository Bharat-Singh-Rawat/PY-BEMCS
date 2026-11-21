# Implementation Report: Sub-Grid Structural Surface & Thermal Representation (Plan A)

## 1. Executive Summary

In Particle-in-Cell (PIC-MCC) simulations of gridded ion optics, representing grid electrodes as discrete rasterized boolean cells (`isBound[iy, ix]`) creates severe physical and numerical limitations:
1. **Chunky Cell-Dropping Erosion**: Sputtering erosion proceeds in discrete jumps of size $\Delta y \times \Delta x$, creating unphysical $90^\circ$ stepped cavities whenever cell damage exceeds threshold.
2. **Artificial Corner Scattering**: Staircased electrode corners distort local radial electric fields and over-scatter periphery ions.
3. **Under-Resolved Thermal Conduction**: 2D finite-difference heat conduction on coarse vacuum cells artificially blunts the steep thermal gradient across the grid cantilever.

**Plan A (Sub-Grid Structural Surface & Thermal Representation)** decouples the physical grid boundary geometry, continuous sputtering erosion, and cantilever thermal conduction from the background PIC cell spacing.

---

## 2. Mathematical & Physical Formulation

### 2.1 Continuous 1D Boundary Polylines (`SubgridApertureProfile`)
Each grid aperture is discretized into continuous 1D boundary segments ($\Delta s \sim 5\dots 10\,\mu\mathrm{m}$):
* **Upstream Face**: $x = g_{\mathrm{start}} + \delta_{\mathrm{bow}}$, $y \in [r_{\mathrm{aperture}}, L_y]$ with outward unit normal $\hat{\mathbf{n}} = (-1, 0)$.
* **Aperture Barrel / Conical Chamfer**: Connecting $(g_{\mathrm{start}} + \delta_{\mathrm{bow}}, r_{\mathrm{front}})$ to $(g_{\mathrm{end}} + \delta_{\mathrm{bow}}, r_{\mathrm{back}})$ with conical taper $r(x) = \max\left(0, r_{\mathrm{ap}} - (x - g_{\mathrm{start}} - \delta_{\mathrm{bow}}) \tan \alpha\right)$ and inward normal $\hat{\mathbf{n}} = (-\sin \alpha, -\cos \alpha)$.
* **Downstream Face**: $x = g_{\mathrm{end}} + \delta_{\mathrm{bow}}$, $y \in [r_{\mathrm{back}}, L_y]$ with outward unit normal $\hat{\mathbf{n}} = (+1, 0)$.

### 2.2 Continuous Sputtering & Yamamura Angular Enhancement
When an ion strikes the grid at continuous coordinates $(x_i, y_i)$ with velocity $\mathbf{v}_i$:
1. The nearest boundary segment $k$ with normal $\hat{\mathbf{n}}_k$ is identified.
2. The exact incident angle $\theta_i$ is computed:
   $$\cos \theta_i = \max\left(0.1, \; \frac{-\mathbf{v}_i \cdot \hat{\mathbf{n}}_k}{\|\mathbf{v}_i\|}\right)$$
3. Sputtering yield is evaluated with Yamamura angular enhancement:
   $$Y(E_i, \theta_i) = Y_{\mathrm{coeff}} \cdot \max(0, E_i - E_{\mathrm{th}}) \cdot f(\theta_i), \qquad f(\theta_i) = \min\left(3.0, \; \frac{1}{\cos \theta_i}\right)$$
4. Continuous recession depth is accumulated on segment $k$:
   $$\Delta d_k = \frac{Y(E_i, \theta_i) \cdot w_{\mathrm{macro}}}{\text{Thresh}} \cdot \Delta y$$
   yielding smooth, sub-micrometer depth evolution without waiting for cell dropping.

### 2.3 Dedicated 1D Radial Structural Thermal Conduction (`SubgridThermalModel`)
The solid metal grid plate is modeled as a 1D radial finite-volume continuum $r \in [r_{\mathrm{aperture}}, L_y]$:
$$\rho c_p t_{\mathrm{grid}} \frac{\partial T}{\partial t} = \frac{1}{r} \frac{\partial}{\partial r}\left( k t_{\mathrm{grid}} r \frac{\partial T}{\partial r} \right) + \dot{q}_{\mathrm{dep}}(r) - 2 \epsilon \sigma \left(T^4 - 300^4\right)$$
with:
* Clamped chassis boundary: $T(L_y) = 300\text{ K}$.
* Aperture edge: $\partial T / \partial r = 0$ (adiabatic / radiation only).
* Euler–Bernoulli cantilever deflection:
  $$\delta_{\mathrm{cant}} = \frac{\alpha \cdot (\bar{T} - 300) \cdot L_{\mathrm{cant}}^2}{2 t_{\mathrm{grid}}}$$

---

## 3. Architecture & Integration in PY-BEMCS

```
[ Domain Build (domain.py) ]
      │
      ▼
Instantiate SubgridManager (structural_surface.py)
   ├── SubgridApertureProfile (continuous 1D polylines for each grid)
   └── SubgridThermalModel (1D radial conduction for each grid plate)
      │
      ├───────────────────────────────┬───────────────────────────────┐
      ▼                               ▼                               ▼
[ Ion Interactions (ion_walls.py) ]  [ Thermal Conduction (thermal.py) ] [ Diagnostics (diagnostics.py) ]
 • Continuous Yamamura yield          • 1D radial metal diffusion     • get_groove_profile() returns
 • Smooth segment recession d_k       • Euler-Bernoulli cantilever      smooth high-resolution curves
 • Synchronizes sim.eroded_depth        deflection δ(T)                 to GUI without staircasing
```

---

## 4. Verification & Test Results

A dedicated unit test suite ([tests/test_subgrid_surface.py](file:///c:/Users/Nick/Documents/Learning%20stuff/Uni/Dispense/Fifth%20year/Space%20propulsion%20Lab/Project/Grids/PY-BEMCS/Python/tests/test_subgrid_surface.py)) validates all components:

| Test Case | Description | Result |
| :--- | :--- | :--- |
| `test_subgrid_segment_normals` | Upstream $(-1, 0)$, Downstream $(+1, 0)$, and Inward Barrel $(0, -1)$ unit normals | **PASSED** |
| `test_aperture_profile_and_hit_recording` | Continuous erosion recession and Yamamura angular scaling | **PASSED** |
| `test_thermal_model_conduction_and_deflection` | 1D radial heat diffusion, Stefan-Boltzmann radiation, and cantilever deflection | **PASSED** |
| `test_subgrid_manager_integration_in_simulator` | End-to-end integration with `sim.build_domain`, `sim.eroded_depth`, and `get_groove_profile` | **PASSED** |
| Regression Test Suite | `test_multi_grid_erosion_plots.py`, `test_stability_and_diagnostics_fixes.py`, `test_axisymmetric_solver.py` | **ALL 21 PASSED** |
