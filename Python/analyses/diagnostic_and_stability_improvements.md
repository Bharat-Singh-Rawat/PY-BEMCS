# PY-BEMCS Diagnostic, Stability & Telemetry Improvements Report

## 1. Executive Summary

This report documents the architectural improvements, physical root causes, and verification tests implemented to resolve three core operational and diagnostic limitations in the PY-BEMCS simulation suite:

1. **Ion Current Stability Criterion**: Inclusion of exit ion current ($I_{\mathrm{ion,exit}}$) alongside total energy, system charge, and potential as a stationarity convergence criterion for steady-state detection in both standalone and parametric sweep modes.
2. **2D Plasma Density Diagnostics (PPC-Mesh Ground Truth)**: Resolution of the sparse/empty 2D density contour plot and 1D radial profiles by computing number density directly from the full mesh Particles-Per-Cell (`current_ppc_map` and `accum_ppc_map`) scaled by macroparticle weight and cell volumes, operating accurately in both instant and cumulative time-averaged modes.
3. **Dynamic Midline Potential Range Autoscaling**: Elimination of graph truncation when high space charge pushes centerline potential above $+100\,\mathrm{V}$ (e.g. at $n_0 = 10^{19}\,\mathrm{m}^{-3}$), accompanied by the removal of dead-code branches in the sweep snapshot rendering pipeline.

---

## 2. Stability Criterion: Exit Ion Current Stationarity

### 2.1 Problem Formulation
Previously, the steady-state convergence detector (`diagnostics/steady_state_window.py`, `gui/app/sweep_mixin.py`, and `physics/headless_sweep.py`) evaluated convergence based on rolling relative variations of:
* Total System Energy ($E_{\mathrm{tot}} = E_{\mathrm{kin}} + E_{\mathrm{pot}}$)
* Total Net Charge in Domain ($Q_{\mathrm{tot}}$)
* Centerline Saddle-Point Potential ($V_{\mathrm{saddle}}$)

While these quantities capture global Poisson and particle confinement equilibria, they occasionally stabilized before the downstream ion extraction current reached true hydrodynamic stationarity. During initial sheath expansion and virtual anode relaxation, temporary current dips or surges occur. Without tracking $I_{\mathrm{ion,exit}}$, simulations could declare premature convergence and terminate or advance to the next sweep scenario while the beam current was still undergoing transient oscillations.

### 2.2 Mathematical Definition & Tracking
During the probe evaluation step (`physics/step/probes.py`), total exit ion current is accumulated across the downstream boundary Faraday cup:
$$I_{\mathrm{ion,exit}} = \frac{e \, w_{\mathrm{macro}} \sum_{\mathrm{exit}} 1}{\Delta t}$$

The simulator (`physics/simulator.py`) maintains a rolling history buffer:
```python
self.ion_current_history: List[float] = []
self.exit_ion_current_history = self.ion_current_history  # backward-compatible alias
```

The rolling relative variation over a window $W$ is computed as:
$$\delta I_{\mathrm{ion}} = \frac{\max_{k \in [N-W, N]} I_k - \min_{k \in [N-W, N]} I_k}{\bar{I}} \times 100\%$$

In both standalone simulation (`SteadyStateWindow._check_steady_state`) and parallel/headless sweep (`physics/headless_sweep.py:check_steady_state_convergence`), the exit ion current is now included in the multi-metric criterion:
$$\max(\delta E_{\mathrm{tot}},\, \delta Q_{\mathrm{tot}},\, \delta V_{\mathrm{saddle}},\, \delta I_{\mathrm{ion}}) \le \varepsilon_{\mathrm{threshold}}$$

If $\delta I_{\mathrm{ion}} > \varepsilon_{\mathrm{threshold}}$, steady state is rejected, ensuring the extraction current has fully leveled out before scenario completion.

---

## 3. 2D Plasma Density Diagnostics (PPC-Mesh Ground Truth)

### 3.1 Problem Formulation & Root Cause
Users observed that in the 2D Density Window (`diagnostics/density_window.py`), the density contour plot appeared almost completely empty (black screen with sparse individual dots), and 1D radial profiles across the exit plane registered near zero.

Investigation revealed two compounding causes:
1. **Particle Subsampling for GUI IPC**: To maintain a responsive 60 FPS GUI and avoid IPC memory bloat during parallel sweeps, headless workers subsample particle coordinates to `scatter_limit = 1,000` points.
2. **Re-Binning Subsampled Points on Fine Mesh**: The density window attempted to compute number density by running 2D histogram bincounting (`np.histogram2d`) on the subsampled 1,000 particles across a $500 \times 300 = 150{,}000$ cell mesh. Consequently, 99.33% of cells contained 0 particles. When rendered with Matplotlib's logarithmic colormap (`LogNorm(vmin=1e14, vmax=n0)`), cells with 0 particles were masked out as uncolored (black background), leaving only a few isolated dots.

### 3.2 Solution: Direct PPC-to-Density Field Mapping
In the PIC-MCC solver, the full ensemble of $>100{,}000$ physical macroparticles is already deposited onto the mesh each step into `current_ppc_map` (and accumulated into `accum_ppc_map`).

The density diagnostic (`physics/mixins/diagnostics.py:get_density_map`) was re-engineered to directly map the PPC field into physical plasma number density:
$$n(x_i, y_j) = \frac{w_{\mathrm{macro}} \cdot \mathrm{PPC}(i, j)}{V_{\mathrm{cell}}(i, j)}$$

Where the cell volume $V_{\mathrm{cell}}(i, j)$ accounts for coordinate geometry:
* **Cartesian 2D**: $V_{\mathrm{cell}} = \Delta x_i \cdot \Delta y_j \cdot 1.0\,\mathrm{m}$
* **Axisymmetric ($r\text{--}z$)**: $V_{\mathrm{cell}} = 2\pi \, r_j \, \Delta r_j \, \Delta x_i$

Furthermore, the Density Window supports two complementary viewing modes:
* **Instantaneous Density**: Uses `current_ppc_map` reflecting the active time step.
* **Cumulative Time-Averaged Density**: Uses `accum_ppc_map / ppc_steps_count`, providing smooth, publication-grade density contours completely free of statistical PIC Monte Carlo grain noise.

---

## 4. Centerline Midline Potential Range Autoscaling

### 4.1 Root Cause of Potential Truncation
In the main window telemetry dashboard, the bottom-left plot displays the axial potential profile along the midline ($y = 0$).

When simulating high plasma densities ($n_0 = 10^{19}\,\mathrm{m}^{-3}$), the space-charge density of positive ions inside the screen-accelerator gap reaches:
$$\rho_{\mathrm{space}} \approx e \cdot n_0 \cdot \exp(-1/2) \sim 10^{-19} \cdot 10^{19} \approx 1\,\mathrm{C/m}^3$$
This positive charge concentration produces space-charge choking and creates a virtual anode where the axial potential rises above $+100\,\mathrm{V}$ (reaching $+150\,\mathrm{V}\dots +300\,\mathrm{V}$).

However, the Y-axis limits of `ax_ebs` were constrained by legacy hardcoded clamps:
```python
self.ax_ebs.set_ylim(-100, 100)
```
Moreover, in `gui/app/sweep_mixin.py:_render_sweep_scenario_snapshot`, an `elif hasattr(self, 'ax_ebs')` branch after `_update_ebs_xlim` prevented Y-axis updates from executing during scenario switching, freezing the view at $[-100, 100]\,\mathrm{V}$. Consequently, the potential curve rose off the top of the canvas, rendering the graph visually blank.

### 4.2 Solution: Unified Dynamic Limit Updater
A dedicated autoscaler method was added to `gui/app/rendering.py`:
```python
def _update_ebs_ylim(self):
    """Autoscale ax_ebs Y-limits based on visible midline potential history."""
    if not hasattr(self, 'ax_ebs') or self.ax_ebs is None:
        return
    vals = [v for v in getattr(self, 'ebs_history', []) if v is not None and not np.isnan(v)]
    if not vals:
        self.ax_ebs.set_ylim(-100, 100)
        return
    min_v = min(vals)
    max_v = max(vals)
    y_lower = min(-100.0, float(min_v) - 20.0)
    y_upper = max(100.0, float(max_v) + 20.0)
    self.ax_ebs.set_ylim(y_lower, y_upper)
```

The sweep snapshot renderer (`_render_sweep_scenario_snapshot`) was updated to decouple X and Y limit calls, ensuring that midline potential curves spanning any positive or negative range ($[-400, +500]\,\mathrm{V}$) remain fully framed and visible at all iterations.

---

## 5. Verification & Automated Test Matrix

The updates are covered by 7 unit and integration tests in `tests/test_stability_and_diagnostics_fixes.py` and regression suites across the entire repository:

| Test Case | Module | Description | Status |
| :--- | :--- | :--- | :--- |
| `test_ion_current_history_tracked_on_sim` | `test_stability_and_diagnostics_fixes.py` | Validates `ion_current_history` recording on `DigitalTwinSimulator`. | **PASSED** |
| `test_steady_state_window_reads_sim_ion_current_without_app` | `test_stability_and_diagnostics_fixes.py` | Verifies `SteadyStateWindow` operates directly from simulator without requiring GUI app reference. | **PASSED** |
| `test_transient_ion_current_prevents_premature_steady_state` | `test_stability_and_diagnostics_fixes.py` | Proves transient exit current prevents false convergence when energy is stable. | **PASSED** |
| `test_density_map_uses_full_ppc_grid` | `test_stability_and_diagnostics_fixes.py` | Verifies `get_density_map` resolves number density from `current_ppc_map` with exact volume weighting. | **PASSED** |
| `test_density_window_instant_and_time_averaged` | `test_stability_and_diagnostics_fixes.py` | Tests instant vs. cumulative time-averaged density rendering in `DensityWindow`. | **PASSED** |
| `test_rendering_mixin_ebs_ylim_autoscales_high_positive_potential` | `test_stability_and_diagnostics_fixes.py` | Validates Y-limits expand to $[-100, 270]\,\mathrm{V}$ when midline potential reaches $+250\,\mathrm{V}$. | **PASSED** |
| `test_rendering_mixin_ebs_ylim_empty_fallback` | `test_stability_and_diagnostics_fixes.py` | Validates $[-100, 100]\,\mathrm{V}$ baseline fallback when history is empty. | **PASSED** |
| `test_parallel_sweep.py` (23 tests) | Regression Suite | Verifies multi-process sweep, top-bar scenario switching, non-blocking modal test guards. | **PASSED** |
| `test_sweep_mode.py` (55 tests) | Regression Suite | Comprehensive sequential and parallel sweep integration. | **PASSED** |

