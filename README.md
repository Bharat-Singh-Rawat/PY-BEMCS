
<h1 align="center">
  <big>🚀 Ion Thruster Digital Twin & MCC Plume Simulator (MATLAB)</big>
</h1>

A suite of vectorized MATLAB applications for modeling Charge Exchange (CEX) dynamics and Grid Erosion in Ion thrusters.

This repository hosts two major versions:
Plume MCC Simulator: Focuses on CEX ion production and downstream flux collection (Plume dynamics) (charge_exchange_code.m).
Grid Digital Twin (EOL): Focuses on "Accelerated Life Testing," modeling the physical sputtering and eventual structural failure of accelerator grids (TransientDigitalTwin.m).

📺 Digital Twin Demo: Grid Erosion & Failure (CXfail.mp4)
This simulation demonstrates the end-of-life (EOL) transition of a dual-grid ion optics system. As CEX ions pit the accelerator grid, the geometry thins, and the negative potential barrier collapses.

<p align="center">
<video src="https://github.com/user-attachments/assets/f7e544f9-890e-43f9-bb1e-c623c5e220ad" width="900px" controls autoplay loop muted>
Your browser does not support the video tag.
</video>
</p>

✨ Key Features
⚡ Vectorized Physics Engine
Both versions replace traditional particle-tracking loops with matrix operations, enabling real-time simulation of thousands of particles natively in MATLAB.

🛠️ Dynamic Sputter Morphing (Digital Twin)
Yamamura Yield Integration: Calculates material removal based on ion impact energy and angle.

Self-Consistent Laplace Solver: As the grid "melts" away, the code remeshes the geometry and recalculates the electrostatic potential profile on the fly to reflect the changing field topology.

Structural Failure Logic: Removes grid cells once they cross a cumulative damage threshold, allowing for realistic "hole-to-hole" erosion modeling.

📊 Real-Time Telemetry & Analytics
EBS Monitoring: Tracks the minimum centerline potential to predict the onset of Electron Backstreaming.

Beam Diagnostics: Live calculation of 95% Beam Divergence half-angles.

Directional Flux Probes (Plume Ver.): Point-and-click probes with adjustable collection radii and surface angles to capture directional particle flux and downstream shadowing.

🎥 Live 3D CAD Projection
A synchronized 3D window projects the 2D axisymmetric physics into a revolved 3D view. This allows for real-time inspection of "barrel erosion" and "pit and groove" patterns from any angle.

💾 Engineering Exports
GIF Recorder: Buffer frames silently into memory and export high-quality animations.

CSV Data Export: Save probe time-series or telemetry history (EBS/Divergence) for post-processing and validation.

🛠️ Installation & Usage
Prerequisites: You need MATLAB R2015b or newer. No additional toolboxes (like Signal Processing or Image Processing) are required.

Download: Clone this repository or download the .m files.

Run: Open MATLAB, navigate to the folder, and run either:

Matlab
TransientDigitalTwin  % For Grid Erosion & EOL Study
charge_exchange_code  % For Plume Dynamics & Flux Probes
🔬 Physics Context

These tools can help students and engineers visualize the primary failure mechanism of gridded ion engines.

Charge Exchange (CEX): Occurs when a fast-moving beam ion steals an electron from a slow-moving neutral atom.

Grid Erosion: The resulting slow-moving CEX ion is accelerated into the accelerator grid by the negative potential, causing sputtering.

EOL: The thruster reaches its end-of-life when the grid can no longer prevent electrons from flowing back into the discharge chamber (EBS failure) or when structural integrity is lost.

Adapted from the original Java-based PIC/MCC demonstration by Lubos Brieda, Particle In Cell Consulting (https://www.particleincell.com/2011/mcc/).

🤝 Community & Feedback
I am actively seeking feedback from the #ElectricPropulsion community. Future roadmap items include:

Integrating self-consistent space charge effects (Poisson solver).

Adding support for triple-grid (3-grid) systems.

Implementing more complex sputtering models for different materials (Graphite, Titanium).

If you find a bug or have a suggestion for the physics model, please open an Issue or PR!

#ElectricPropulsion #PlasmaPhysics #MATLAB #AerospaceEngineering #DigitalTwin #Simulation
