"""
Matplotlib figure and axes setup for PY-BEMCS GUI.
"""
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def setup_figure_layout(app):
    """Initialise and configure the 3x3 GridSpec layout on app.fig."""
    grid = plt.GridSpec(3, 3, height_ratios=[1.2, 1, 0.9])

    # Row 0: wide beam extraction plot + narrow temperature map
    app.ax_live = app.fig.add_subplot(grid[0, 0:2])
    app.ax_live.set_title("Ion Beam Extraction & Particle Tracking", fontsize=10)
    app.ax_live.set_xlabel("Axial Position [mm]")
    app.ax_live.set_ylabel("Radial Position [mm]")

    app.ax_temp = app.fig.add_subplot(grid[0, 2])
    app.ax_temp.set_title("Grid Temperature", fontsize=10)
    app.ax_temp.set_xlabel("Axial Position [mm]", fontsize=8)
    app.ax_temp.set_ylabel("Radial Position [mm]", fontsize=8)

    # Row 1: damage map + diagnostics
    app.ax_dmg = app.fig.add_subplot(grid[1, 0])
    app.ax_dmg.set_title("Sputter Damage Map", fontsize=10)
    app.ax_dmg.set_xlabel("Axial Position [mm]", fontsize=8)
    app.ax_dmg.set_ylabel("Radial Position [mm]", fontsize=8)

    app.ax_ebs = app.fig.add_subplot(grid[1, 1])
    app.ax_ebs.set_title("Saddle-Point Potential", fontsize=10)
    app.ax_ebs.set_xlabel("Iteration", fontsize=8)
    app.ax_ebs.set_ylabel("[V]", fontsize=8)

    app.ax_div = app.fig.add_subplot(grid[1, 2])
    app.ax_div.set_title("Beam Divergence", fontsize=10)
    app.ax_div.set_xlabel("Iteration", fontsize=8)
    app.ax_div.set_ylabel("[°]", fontsize=8)

    # Row 2: full-width erosion profile
    app.ax_groove = app.fig.add_subplot(grid[2, :])
    app.ax_groove.set_title("Accel Grid Erosion Profile", fontsize=10)
    app.ax_groove.set_xlabel("Radial Position [mm]")
    app.ax_groove.set_ylabel("Erosion Depth [um]")
    app.ax_groove.grid(True, alpha=0.3)
    app.ax_groove.invert_yaxis()

    app.line_ebs, = app.ax_ebs.plot([], [], "m-", lw=2)
    app.line_div, = app.ax_div.plot([], [], "b-", lw=2)
    app.line_groove, = app.ax_groove.plot([], [], "r-", lw=1.5)

    app.scat_prim = None
    app.scat_cex = None
    app.scat_elec = app.ax_live.scatter([], [], s=1, c="#00FF00", alpha=0.5)

    app.fig.tight_layout(rect=(0, 0, 0.97, 1), h_pad=1.5, w_pad=0.8)
