"""
Matplotlib figure and axes setup for PY-BEMCS GUI.
"""
import math
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec


def rebuild_groove_axes(app, n_grids):
    """Rebuild Row 2 of app.fig with one erosion profile subplot per grid (2 columns per row)."""
    if not hasattr(app, "_main_grid") or app._main_grid is None:
        return

    n_grids = max(1, int(n_grids or 1))

    # Remove any previous groove axes
    old_axes = list(getattr(app, "ax_grooves", []))
    old_single = getattr(app, "ax_groove", None)
    if old_single is not None and old_single not in old_axes:
        old_axes.append(old_single)
    for ax in old_axes:
        if ax in app.fig.axes:
            app.fig.delaxes(ax)

    n_cols = 1 if n_grids <= 1 else 2
    n_rows = math.ceil(n_grids / n_cols)

    subgrid = GridSpecFromSubplotSpec(
        n_rows,
        n_cols,
        subplot_spec=app._main_grid[2, :],
        wspace=0.25,
        hspace=0.65 if n_rows > 1 else 0.35,
    )

    ax_grooves = []
    line_grooves = []
    title_fs = 8.5 if n_rows > 1 else 10
    label_fs = 8 if n_rows > 1 else 9

    for i in range(n_grids):
        r, c = divmod(i, n_cols)
        ax = app.fig.add_subplot(subgrid[r, c])
        ax.set_title(f"Grid {i + 1} Erosion Profile", fontsize=title_fs, pad=3)
        if r == n_rows - 1 or i + n_cols >= n_grids:
            ax.set_xlabel("Radial Position [mm]", fontsize=label_fs, labelpad=2)
        if c == 0:
            ax.set_ylabel("Erosion Depth [um]", fontsize=label_fs, labelpad=2)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        line, = ax.plot([], [], "r-", lw=1.5)
        ax_grooves.append(ax)
        line_grooves.append(line)

    app.ax_grooves = ax_grooves
    app.line_grooves = line_grooves
    app.ax_groove = ax_grooves[0]
    app.line_groove = line_grooves[0]

    app.fig.tight_layout(rect=(0, 0, 0.97, 1), h_pad=1.5, w_pad=0.8)

    if hasattr(app, "_reposition_div_menu_btn"):
        app._reposition_div_menu_btn()
    if hasattr(app, "_reposition_xaxis_menu_btns"):
        app._reposition_xaxis_menu_btns()
    if hasattr(app, "_reposition_div_scrollbar"):
        app._reposition_div_scrollbar()
    if hasattr(app, "_reposition_ebs_scrollbar"):
        app._reposition_ebs_scrollbar()


def setup_figure_layout(app):
    """Initialise and configure the 3x3 GridSpec layout on app.fig."""
    grid = plt.GridSpec(3, 3, height_ratios=[1.2, 1, 0.9])
    app._main_grid = grid

    # Row 0: wide beam extraction plot + narrow temperature map (isolated subgrid so only row 0 is adjusted without expanding space across other graphs)
    subgrid_row0 = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=grid[0, :],
        width_ratios=[2.0, 0.92],
        wspace=0.24,
    )
    app.ax_live = app.fig.add_subplot(subgrid_row0[0, 0])
    app.ax_live.set_title("Ion Beam Extraction & Particle Tracking", fontsize=10)
    app.ax_live.set_xlabel("Axial Position [mm]")
    app.ax_live.set_ylabel("Radial Position [mm]")

    app.ax_temp = app.fig.add_subplot(subgrid_row0[0, 1])
    app.ax_temp.set_title("Grid Temperature", fontsize=10)
    app.ax_temp.set_xlabel("Axial Position [mm]", fontsize=8)
    app.ax_temp.set_ylabel("r [mm]", fontsize=8, labelpad=2)
    app.ax_temp.tick_params(axis="y", labelsize=7, pad=2)

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

    app.line_ebs, = app.ax_ebs.plot([], [], "m-", lw=2)
    app.line_div, = app.ax_div.plot([], [], color="#1f77b4", lw=1.8, label="Grid Exit (θ95)")
    app.line_div_mid, = app.ax_div.plot([], [], color="#e67e22", lw=1.8, linestyle="-", label="Mid Plume (θ95)")
    app.line_div_plane = None

    app.scat_prim = None
    app.scat_cex = None
    app.scat_elec = app.ax_live.scatter([], [], s=1, c="#00FF00", alpha=0.5)

    # Row 2: one erosion profile subplot per configured grid (2 columns per row)
    app.ax_grooves = []
    app.line_grooves = []
    app.ax_groove = None
    app.line_groove = None
    n_grids = max(1, len(getattr(app, "grid_widgets", [])))
    rebuild_groove_axes(app, n_grids)
