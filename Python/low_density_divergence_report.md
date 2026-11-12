# Root-Cause Report: Underestimated Beam Divergence at Low Plasma Densities in PY-BEMCS (NSTAR Case Study)

## Executive Summary

When simulating gridded ion optics in **PY-BEMCS** at low upstream plasma densities ($n_0 \sim 10^{15}\text{–}10^{17}\text{ m}^{-3}$, i.e., the under-perveant / crossover regime), the predicted beam divergence angle is substantially lower than experimental **NSTAR** data ($\sim 20^\circ\text{–}29^\circ$ for 95% enclosed beam current across throttle points TH0–TH15) and remains depressed **regardless of the divergence metric** (`95%`, `90%`, `Direct RMS`, `RMS (Twiss)`, `FWHM`) or **evaluation plane** (`x_exit` vs. `x_mid`).

Code inspection and controlled diagnostic runs on [`configNSTAR.json`](configNSTAR.json) identify **five computational root causes** inside the solver and **four physical/modeling discrepancies** relative to whole-thruster NSTAR measurements. The single largest computational cause is that the default `half_hole` symmetry mode **deletes over-focused crossover ions at $y < 0$ instead of reflecting them**, artificially stripping out the highest-angle trajectories and collapsing mid-plume divergence by more than a factor of $4\times$ at low density.

---

## 1. Observed Diagnostic Evidence in PY-BEMCS

Controlled 800-step (`dt = 0.5 ns`, `target_ppc = 40`, `sim_mode = "Thermal"`) runs of [`configNSTAR.json`](configNSTAR.json) ($V_s = 1100\text{ V}$, $V_a = -180\text{ V}$, $T_{e,\text{up}} = 4.3\text{ eV}$, $r_s = 0.95\text{ mm}$, $r_a = 0.57\text{ mm}$) at $n_0 = 10^{16}\text{ m}^{-3}$ demonstrate the magnitude of the solver-induced suppression across all five metrics implemented in [`compute_beam_divergence`](physics/step/ion_walls.py#L40-L156):

| Case ($n_0 = 10^{16}\text{ m}^{-3}$) | Plane | $n_{\text{sample}}$ | Transp. | $V_{\text{exit}}$ [V] | `95%` | `90%` | `Direct RMS` | `RMS (Twiss)` | `FWHM` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline `half_hole`** | `x_exit` (2.18 mm) | 1598 | 0.672 | $-162.0$ | $13.39^\circ$ | $12.22^\circ$ | $7.13^\circ$ | $4.31^\circ$ | $5.27^\circ$ |
| **1. Baseline `half_hole`** | `x_mid` (3.68 mm) | **595** | 0.672 | $-162.0$ | **$4.66^\circ$** | **$4.12^\circ$** | **$2.41^\circ$** | **$1.89^\circ$** | **$2.31^\circ$** |
| **2. `one_hole`** (periodic $y$) | `x_exit` (2.18 mm) | 4802 | 0.853 | $-153.2$ | $21.64^\circ$ | $19.34^\circ$ | $11.15^\circ$ | $11.13^\circ$ | $8.14^\circ$ |
| **2. `one_hole`** (periodic $y$) | `x_mid` (3.68 mm) | **4809** | 0.853 | $-153.2$ | **$22.57^\circ$** | **$20.10^\circ$** | **$11.86^\circ$** | **$11.82^\circ$** | **$9.22^\circ$** |
| **3. `one_hole` + $V(L_x)=0\text{ V}$** | `x_exit` (2.18 mm) | 4997 | 0.854 | $0.0$ | $21.95^\circ$ | $19.58^\circ$ | $11.43^\circ$ | $11.38^\circ$ | $9.42^\circ$ |
| **3. `one_hole` + $V(L_x)=0\text{ V}$** | `x_mid` (3.68 mm) | 4988 | 0.854 | $0.0$ | **$23.39^\circ$** | **$20.94^\circ$** | **$13.02^\circ$** | **$12.97^\circ$** | **$11.02^\circ$** |

Similarly, across a 1000-step sweep in `half_hole` (`target_ppc = 40`):
- At $n_0 = 10^{15}\text{ m}^{-3}$: `div_exit = 14.88°`, `div_mid = 4.63°` (plume-wide: `95% = 10.19°`, `Direct RMS = 4.81°`, `Twiss = 3.75°`, `FWHM = 3.82°`).
- At $n_0 = 10^{17}\text{ m}^{-3}$: `div_exit = 15.71°`, `div_mid = 4.73°`.
- At nominal peak density $n_0 = 7\times 10^{17}\text{ m}^{-3}$: `div_exit = 9.28°`, `div_mid = 17.89°`, `V_exit = +338.5 V`.
- In [`benchmark_perveance.py`](benchmarks/benchmark_perveance.py) (`half_hole`, thick grids $t_s = t_a = 1.0\text{ mm}$, $n_0 \in [4\times 10^{14}, 5\times 10^{17}]\text{ m}^{-3}$), the saved curve [`divergence_vs_perveance_benchmark_multicore.png`](results/divergence_vs_perveance_benchmark_multicore.png) bottoms out at $\sim 1.5^\circ$ and reaches at most $\sim 8^\circ$ even at the lowest perveance.

---

## 2. Computational Reasons (Verified in Codebase)

### 2.1 Missing Specular Reflection at $y = 0$ in `half_hole` (Crossover Ion Deletion) — *Primary Cause*
- **Documentation vs. Code Mismatch**: The manual ([`PY-BEMCS_Manual.tex`](docs/PY-BEMCS_Manual.tex#L198)) states that `half_hole` (the default geometry in [`configNSTAR.json`](configNSTAR.json#L29)) models *"specular symmetry reflection at $y = 0$"*. However, neither [`step_push`](physics/step/push.py#L12-L119) nor the push kernels implement any reflection ($y \to -y,\; v_y \to -v_y$) at $y = 0$.
- Instead, both the CPU boundary handler ([`step_ion_interactions`](physics/step/ion_walls.py#L413-L419)) and the Taichi GPU kernel ([`_check_particle_boundaries_taichi_kernel`](physics/kernels_taichi.py#L199-L200)) treat `p_y < 0.0` when `periodic_y == False` as **out-of-bounds (`is_oob = 1`) and immediately delete the macroparticle**.
- **Why this selectively destroys low-density divergence**:
  1. At low perveance (low $n_0$), the concave sheath and electrostatic lens strongly **over-focus** outer-radius ions toward the aperture axis ($v_y < 0$), causing **crossover** across $y = 0$ inside the accel barrel or near the exit plume.
  2. Ions that cross $y = 0$ *before* `x_exit` are deleted before reaching the first divergence plane (`trans` drops from $0.853$ in `one_hole` to $0.672$ in `half_hole`, cutting `x_exit` `95%` divergence from $21.64^\circ$ to $13.39^\circ$).
  3. Surviving inward-converging ions that pass `x_exit` subsequently cross $y = 0$ between `x_exit` and `x_mid`: in our test, **63% of the plume sample vanished** between `2.18 mm` ($n = 1598$) and `3.68 mm` ($n = 595$). Only nearly paraxial ions ($v_y \approx 0$) survive to `x_mid`, collapsing divergence across **every metric** (`95%`: $4.66^\circ$, `Direct RMS`: $2.41^\circ$, `FWHM`: $2.31^\circ$).

### 2.2 2D Planar Cartesian Formulation vs. Axisymmetric ($r\text{--}z$) Cylindrical Optics
- **Code Mechanism**: [`DomainMixin.build_sparse_matrix`](physics/mixins/domain.py#L72-L112) assembles a **2D Cartesian (planar slit)** 5-point finite-volume Laplacian with uniform slab depth $\Delta z = 1\text{ mm}$ ([`cell_vol_1d`](physics/mixins/domain.py#L365)), omitting the cylindrical radial metric $\frac{1}{r}\frac{\partial}{\partial r}(r \frac{\partial V}{\partial r})$ and $2\pi r$ volume weighting. Simultaneously, [`step_inject`](physics/step/injection.py#L90-L103) samples $y$ uniformly rather than with $r\,dr$ annular weighting.
- **Consequences for Divergence**:
  1. **Severe under-weighting of peripheral high-angle ions**: In a circular NSTAR aperture, the outer $30\%$ of the radius ($r \in [0.7 r_s, r_s]$) carries $1 - 0.7^2 = \mathbf{51\%}$ of the extracted beamlet current, whereas a uniform Cartesian slab places only **$30\%$** of particles there. Because peripheral ions sit closest to the curved meniscus and sharp electrode edges where radial fields and crossover aberrations are strongest, planar weighting shifts `95%`, `90%`, `Direct RMS`, `Twiss`, and `FWHM` downward.
  2. **Altered lens focal power and Child–Langmuir scaling**: By the Davisson–Calbick thin-lens relations, a slit lens ($f_{\text{slit}} = 2V_0/\Delta E$) and a circular aperture ($f_{\text{circ}} = 4V_0/\Delta E$) differ by a factor of 2 in focusing power, while planar space charge ($\phi \sim \ln r$) relaxes differently from cylindrical space charge ($\phi \sim 1/r$), shifting the perveance-crossover curve.
  3. **2D angular projection**: [`compute_beam_divergence`](physics/step/ion_walls.py#L67) evaluates $\theta = \arctan2(v_y, v_x)$ without out-of-plane $v_z$ or 3D spherical $\sin\theta\,d\theta$ current-density integration.

### 2.3 Aperture-Truncated Injection and Sheath Truncation at Low Density
- **Injection Restricted to $|y - y_c| \le r_s$**: [`step_inject`](physics/step/injection.py#L46-L104) injects ions *only* across the open screen aperture radius `screen_r` (`0.95 mm`), never across the full pitch $L_y$ above the solid screen grid webbing. At low $n_0$, the physical Debye sheath expands ($\lambda_D$ grows) and curves upstream over the screen webbing; ions originating at $r > r_s$ are funneled radially inward around the curved meniscus and enter the optics at steep angles, contributing strongly to crossover. Omitting $r > r_s$ removes this peripheral high-angle population.
- **Presheath Gap Truncation & Pinned Dirichlet Plane at $x = 0$**: Under the default `entire_bulk_plasma = False`, [`build_domain`](physics/mixins/domain.py#L167-L169) sets `upstream_gap_mm = 0.75 * screen_r` (`0.7125 mm` in `configNSTAR.json`) and pins a hard Dirichlet boundary $V[:, 0] = V_{\text{plasma}}$ ([lines 513–516](physics/mixins/domain.py#L513-L516)), while injecting ions at $x = 1.5\,\Delta x$ with purely axial Bohm velocity $v_x = v_{\text{Bohm}}$ and small thermal spread $v_{\text{th},i}$.
  - At nominal density ($n_0 = 7\times 10^{17}\text{ m}^{-3}$), $\lambda_D \approx 0.024\text{ mm}$, so the upstream gap spans $\sim 30\,\lambda_D$.
  - At low density ($n_0 = 10^{16}\text{ m}^{-3}$), $\lambda_D \approx 0.20\text{ mm}$, shrinking the gap to **$3.5\,\lambda_D$**; at $n_0 = 10^{15}\text{ m}^{-3}$, $\lambda_D \approx 0.63\text{ mm}$, leaving only **$1.1\,\lambda_D$**.
  - Pinning a flat $V = V_{\text{plasma}}$ equipotential just $1\text{–}3\,\lambda_D$ upstream of the screen grid prevents the low-density sheath from recessing and curving naturally upstream, flattening the plasma meniscus and weakening its radial focusing curvature.

### 2.4 Unclamped Downstream Plume Potential (Neumann $\partial V/\partial x = 0$ with `neut_rate = 0`)
- **Code Mechanism**: In 2-grid optics (`configNSTAR.json` and `benchmark_perveance.py`) there is no grounded deceleration grid, neutralizer electron injection is disabled (`neut_rate = 0`, `neut_match_ion = False`), and [`build_sparse_matrix`](physics/mixins/domain.py#L56-L59) enforces a homogeneous Neumann boundary $V[:, n_x-1] = V[:, n_x-2]$ at $x = L_x$.
- **Measured Effect**: At low density ($n_0 = 10^{15}\text{–}10^{16}\text{ m}^{-3}$), space charge is too weak to pull the plume up to ground ($0\text{ V}$); instead, `V_exit` remains trapped near the accelerator potential (**$-162\text{ V}$ to $-165\text{ V}$**).
- **Why this lowers divergence**:
  1. **Missing accel-exit diverging electrostatic lens**: In real thrusters, neutralizer electrons clamp the plume potential near $0\text{ to }+5\text{ V}$, creating a strong deceleration gradient ($-180\text{ V} \to 0\text{ V}$) immediately downstream of the accel grid whose outward radial fringe fields defocus the beamlet. Without potential recovery, this exit defocusing kick vanishes.
  2. **Inflated axial velocity $v_x$**: Ions retain $\sim 1209\text{ eV}$ kinetic energy instead of decelerating to $\sim 1100\text{–}1126\text{ eV}$, directly reducing $\theta = \arctan(v_y/v_x)$. Forcing $V(L_x) = 0\text{ V}$ in Case 3 increased `x_mid` `Direct RMS` from $11.86^\circ \to 13.02^\circ$ and `FWHM` from $9.22^\circ \to 11.02^\circ$.

### 2.5 Negligible CEX Ion Wings in Short Plume / Zero-Neutral Benchmarks
- `benchmark_perveance.py` disables neutral gas (`n0 = 0`), while `configNSTAR.json` simulates only $\sim 3\text{ mm}$ of downstream plume with $n_n = 10^{18}\text{ m}^{-3}$ ([`step_cex`](physics/step/collisions.py#L9-L70)), yielding CEX optical depth $\sim 10^{-3}$. Consequently, the slow, high-angle ($30^\circ\text{–}90^\circ$) CEX ion population present in ground-facility Faraday sweeps is absent.

*(Note: Low particle count / shot noise is **not** a cause here, because [`build_domain`](physics/mixins/domain.py#L404) scales `macro_weight` linearly with `n0`, maintaining $\sim 167\text{–}340$ injected macroparticles/step even at $n_0 = 10^{15}\text{ m}^{-3}$.)*

---

## 3. Physical Reasons (Comparing Single-Beamlet Simulation vs. NSTAR Data)

Even after correcting the numerical boundary conditions above (`one_hole` + $V(L_x)=0\text{ V}$), single-aperture flat-grid simulations measure a fundamentally different physical quantity than published whole-thruster NSTAR divergence data:

1. **Single-Beamlet Micro-Divergence vs. Whole-Thruster Dished Macro-Divergence**:
   - NSTAR uses **spherically dished** 30-cm molybdenum grids (concave toward the discharge chamber) with radial hole-pitch compensation. Published NSTAR experimental divergence half-angles (**$\sim 20^\circ\text{–}29^\circ$** at 95% beam current across throttle levels TH0–TH15) are measured in the far-field plume and convolve:
     1. **Macroscopic spherical dish canting** of peripheral apertures relative to the thruster centerline,
     2. **Off-axis beamlet steering** from intentional/thermal relative aperture misalignment between screen and accel grids, and
     3. **Individual beamlet micro-divergence**.
   - By contrast, a flat single-aperture model computes **micro-divergence only**, which in single-beamlet ion optics literature (e.g., JPL `CEX2D`, CSU `ffx`) is typically **$1^\circ\text{–}8^\circ$** near optimal perveance and only rises above $15^\circ\text{–}25^\circ$ deep in crossover or over-perveance.
2. **Radial Discharge Density Profile Across the Thruster Face**:
   - Real NSTAR discharge chambers have a peaked, bell-shaped radial plasma density profile $n_0(r)$: when global beam current is throttled down (e.g., TH0 at $0.5\text{ kW}$), outer-radius apertures drop sharply into deep crossover (or even grid impingement/cutoff) while center apertures remain near optimal perveance. Moreover, real NSTAR throttling lowers **both** $n_0$ and screen voltage ($V_s \approx 650\text{ V}$ at TH0 vs. $1100\text{ V}$ at TH15), whereas a constant-$V_s$ density sweep varies perveance differently.
3. **Thick-Grid Collimation in Benchmark Scripts**:
   - In [`benchmark_perveance.py`](benchmarks/benchmark_perveance.py#L20), the screen/accel grid thicknesses are set to $t_s = t_a = 1.0\text{ mm}$ (nearly $2\text{–}2.6\times$ thicker than NSTAR's $0.38\text{ mm} / 0.51\text{ mm}$), which geometrically scrapes high-angle crossover rays on the barrel walls before they can exit into the plume.

---

## 4. Summary Checklist & Recommended Diagnostic Actions

| Priority | Issue | Impact on Low-Density Divergence | Verification / Fix |
| :---: | :--- | :--- | :--- |
| **1** | No specular reflection at $y=0$ in `half_hole` | **Severe** (deletes crossover rays; drops `x_mid` to $\sim 4.6^\circ$) | Reflect particles crossing $y < 0$ ($y \leftarrow -y,\; v_y \leftarrow -v_y$) or run `one_hole`. |
| **2** | Neumann $x=L_x$ with `neut_rate=0` | **Moderate** ($V_{\text{exit}} \approx -165\text{ V}$; removes accel exit lens) | Enable `neut_match_ion` or clamp $V(L_x) = 0\text{ V}$ downstream. |
| **3** | Planar 2D vs. Axisymmetric ($r\text{--}z$) | **Moderate–High** (under-weights outer annulus $51\% \to 30\%$) | Weight particle statistics by $y$ (or implement axisymmetric $r\text{--}z$ Poisson). |
| **4** | Short `0.75 * r_s` gap & `r_s`-only injection | **Moderate** (flattens meniscus when $\lambda_D \sim 0.2\text{–}0.6\text{ mm}$) | Enable `entire_bulk_plasma = True` ($30\text{–}80\,\lambda_D$) and inject across full pitch $L_y$. |
| **5** | Flat single-beamlet vs. dished NSTAR | **High** (structural comparison mismatch) | Compare against single-beamlet `CEX2D`/`ffx` data, or convolve beamlet distribution over NSTAR dish geometry and $n_0(r)$ profile. |

---

## 5. Resolution & Implementation Status (Phases 1 & 2)

Both recommended solver enhancements have been implemented and verified:

1. **Phase 1: Specular Boundary Reflection** (Implemented & committed in `d130ee2`):
   - Transverse boundary checks in [`physics/kernels_cpu.py`](physics/kernels_cpu.py), [`physics/kernels_taichi.py`](physics/kernels_taichi.py), and [`physics/step/push.py`](physics/step/push.py) now reflect particles crossing $y < 0$ ($y \leftarrow -y,\; v_y \leftarrow -v_y$) and $y > L_y$.
   - Crossover ions are preserved; NSTAR low-density ($n_0 = 10^{16}\text{ m}^{-3}$) 95% divergence at `x_mid` rose immediately from **$4.66^\circ \to 21.80^\circ$**.
2. **Phase 2: Axisymmetric ($r\text{--}z$) 2D3V Cylindrical Formulation** (Implemented & committed in `c9dad2d`):
   - Added `"coordinate_system": "axisymmetric"` mapping $x \equiv z$ and $y \equiv r \ge 0$ with true annular control volumes $V_{\mathrm{cell}} = 2\pi r \Delta r h_i^x$.
   - Formulated cylindrical Poisson stencil with on-axis L'Hôpital boundary row ($c_{N,0} = 4.0$) enforcing $E_r(0, z) = 0$.
   - Implemented Birdsall & Langdon local-Cartesian rotation conserving canonical angular momentum $L_z = m r v_\theta$ and transverse kinetic energy $E_\perp = \frac{1}{2}m(v_r^2 + v_\theta^2)$.
   - Added uniform radial injection $r = r_{\mathrm{inj}}\sqrt{U}$ and 3D cone divergence $\theta_{\mathrm{3D}} = \arctan2(\sqrt{v_r^2 + v_\theta^2}, v_x)$.
   - Comparative 400-step benchmark at $n_0 = 10^{16}\text{ m}^{-3}$:
     - **Cartesian**: $\theta_{95}(\text{exit}) = 19.28^\circ$, $\theta_{95}(\text{mid}) = 21.90^\circ$
     - **Axisymmetric**: $\theta_{95}(\text{exit}) = 17.68^\circ$, $\theta_{95}(\text{mid}) = 17.00^\circ$
   - Both match published single-beamlet CEX2D crossover divergence trends.

---

## References

1. **Goebel, D. M., & Katz, I.** (2008). *Fundamentals of Electric Propulsion: Ion and Hall Thrusters*, Wiley / JPL Space Science and Technology Series, **Chapter 5: Ion Thruster Accelerator Grids** (perveance, Child–Langmuir limits, crossover vs. perveance impingement, dished grid beamlet steering, radial density profiles).
2. **Pollard, J. E., Polk, J. E., Soulas, J. E., Goebel, D. M., & Katz, I.** (NASA / JPL / Glenn Research Center NSTAR characterization & life-test studies, e.g., AIAA/IEPC papers on NSTAR 30-cm ion optics performance across TH0–TH15 throttle levels; whole-thruster 95% divergence half-angles of $\sim 20^\circ\text{–}29^\circ$ and thrust divergence factors $F_t \approx 0.97\text{–}0.98$).
3. **Katz, I., Anderson, J. R., Polk, J. E., & Brophy, J. R.** (JPL `CEX2D` / `CEX3D` ion optics modeling literature: single-beamlet divergence vs. perveance, upstream Boltzmann sheath sensitivity, and low-density crossover limits).
4. **Farnell, C. C., Williams, J. D., & Wilbur, P. J.** (Colorado State University `ffx` / `igx` / `ERODE` 2D/3D ion optics studies: comparison of 2D vs. 3D crossover limits and single-beamlet divergence curves).
5. **Davisson, C. J., & Calbick, C. J.** (1931/1932). Electrostatic thin-lens focal length relations for circular ($f = 4V_0/\Delta E$) vs. two-dimensional slit ($f = 2V_0/\Delta E$) apertures.
6. **PY-BEMCS Documentation & Source**: [`PY-BEMCS_Manual.tex`](docs/PY-BEMCS_Manual.tex), [`ion_walls.py`](physics/step/ion_walls.py), [`domain.py`](physics/mixins/domain.py), [`injection.py`](physics/step/injection.py).

