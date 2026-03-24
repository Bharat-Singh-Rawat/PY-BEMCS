<h1 align="center">
  <big>🚀 Ion Thruster Digital Twin & MCC Plume Simulator (MATLAB)</big>
</h1>

<p align="center">
  <strong>A suite of high-performance, vectorized MATLAB applications for modeling Charge Exchange (CEX) dynamics and Grid Erosion in Ion thrusters.</strong>
</p>

---

### 📂 Repository Overview

This repository hosts two distinct, high-fidelity simulation environments:

* **Plume MCC Simulator (`charge_exchange_code.m`):** Focuses on CEX ion production, plume expansion, and downstream flux collection.
* **Grid Digital Twin / EOL (`TransientDigitalTwin.m`):** Focuses on "Accelerated Life Testing," modeling the physical sputtering and eventual structural failure of accelerator grids.

---

## 📺 Digital Twin Demo: Grid Erosion & Failure

<p align="center">
  <i>This simulation demonstrates the end-of-life (EOL) transition of a dual-grid ion optics system. As CEX ions pit the accelerator grid, the geometry thins, and the negative potential barrier collapses.</i>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/f7e544f9-890e-43f9-bb1e-c623c5e220ad" width="900px" controls autoplay loop muted>
    Your browser does not support the video tag.
  </video>
</p>

---

## <big>✨ Key Features</big>

<br>

<h3><big>⚡ Vectorized Physics Engine</big></h3>
Both versions replace traditional particle-tracking loops with <b>matrix operations</b>, enabling real-time simulation of thousands of particles natively in MATLAB. This ensures massive performance gains and smooth GUI interactivity.

<br>

<h3><big>🛠️ Dynamic Sputter Morphing (Digital Twin)</big></h3>

* **Yamamura Yield Integration:** Calculates material removal based on ion impact energy and angle.
* **Self-Consistent Laplace Solver:** As the grid "melts" away, the code remeshes the geometry and recalculates the electrostatic potential profile on the fly to reflect the changing field topology.
* **Structural Failure Logic:** Removes grid cells once they cross a cumulative damage threshold, allowing for realistic "hole-to-hole" erosion modeling.

<br>

<h3><big>📊 Real-Time Telemetry & Analytics</big></h3>

* **EBS Monitoring:** Tracks the minimum centerline potential to predict the onset of **Electron Backstreaming**.
* **Beam Diagnostics:** Live calculation of **95% Beam Divergence** half-angles.
* **Directional Flux Probes (Plume Ver.):** Point-and-click to place physical probes in the domain. Includes adjustable collection radii and surface angles to simulate realistic downstream "shadowing."

<br>

<h3><big>🎥 Live 3D CAD Projection</big></h3>
A synchronized 3D window projects the 2D axisymmetric physics into a **revolved 3D view**. This allows for real-time inspection of "barrel erosion" and "pit and groove" patterns from any angle during the simulation.

<br>

<h3><big>💾 Engineering Exports</big></h3>

* **Built-in GIF Recorder:** Silently buffers frames into memory while running, allowing for high-quality `.gif` export of plume dynamics or grid destruction.
* **CSV Data Export:** Pause and export raw probe time-series or telemetry history directly to formatted `.csv` files for post-processing.

---

## <big>🛠️ Installation & Usage</big>

1.  **Prerequisites:** You need **MATLAB R2015b or newer**. No extra toolboxes (e.g., Signal Processing or Image Processing) are required.
2.  **Download:** Clone this repository or download the `.m` files.
3.  **Run:** Open MATLAB, navigate to the folder containing the files, and type the following in the Command Window:

```matlab
TransientDigitalTwin  % To study Grid Erosion & EOL
charge_exchange_code  % To study Plume Dynamics & Flux
