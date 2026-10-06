"""
Unit test for divergence graph interactive checkmark legend and mid-plume divergence quantity.
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYBEMCS_FORCE_CPU"] = "1"

import numpy as np
from PyQt5.QtWidgets import QApplication

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.app import DigitalTwinApp


def test_divergence_graph():
    qt_app = QApplication.instance() or QApplication(sys.argv)
    app = DigitalTwinApp()

    # 1. Verify lines exist
    assert hasattr(app, 'line_div'), "app missing line_div"
    assert hasattr(app, 'line_div_mid'), "app missing line_div_mid"
    assert hasattr(app, '_div_curve_visible'), "app missing _div_curve_visible"
    assert hasattr(app, '_div_legend_map'), "app missing _div_legend_map"
    assert hasattr(app, 'div_mid_history'), "app missing div_mid_history"

    assert app._div_curve_visible['grid'] is True
    assert app._div_curve_visible['mid'] is True
    assert app.line_div.get_visible() is True
    assert app.line_div_mid.get_visible() is True

    # 2. Verify legend entries and pick mapping
    assert len(app._div_legend_map) > 0, "Legend pick map should not be empty"
    grid_artists = [a for a, k in app._div_legend_map.items() if k == 'grid']
    mid_artists = [a for a, k in app._div_legend_map.items() if k == 'mid']
    assert len(grid_artists) > 0, "No artists mapped to 'grid'"
    assert len(mid_artists) > 0, "No artists mapped to 'mid'"

    class DummyPickEvent:
        def __init__(self, artist):
            self.artist = artist

    # 3. Test toggle 'grid' off and on
    app._on_legend_pick(DummyPickEvent(grid_artists[0]))
    assert app._div_curve_visible['grid'] is False, "Grid curve should now be invisible"
    assert app.line_div.get_visible() is False, "line_div artist should be hidden"
    assert app.line_div_mid.get_visible() is True, "line_div_mid should still be visible"

    grid_artists_new = [a for a, k in app._div_legend_map.items() if k == 'grid']
    app._on_legend_pick(DummyPickEvent(grid_artists_new[0]))
    assert app._div_curve_visible['grid'] is True, "Grid curve should be visible again"
    assert app.line_div.get_visible() is True, "line_div artist should be visible again"

    # 4. Test toggle 'mid' off and on
    mid_artists_cur = [a for a, k in app._div_legend_map.items() if k == 'mid']
    app._on_legend_pick(DummyPickEvent(mid_artists_cur[0]))
    assert app._div_curve_visible['mid'] is False, "Mid curve should now be invisible"
    assert app.line_div_mid.get_visible() is False, "line_div_mid artist should be hidden"
    assert app.line_div.get_visible() is True, "line_div should still be visible"

    mid_artists_new = [a for a, k in app._div_legend_map.items() if k == 'mid']
    app._on_legend_pick(DummyPickEvent(mid_artists_new[0]))
    assert app._div_curve_visible['mid'] is True, "Mid curve should be visible again"
    assert app.line_div_mid.get_visible() is True, "line_div_mid artist should be visible again"

    # 5. Test _process_snap appends both quantities
    app.build_domain()
    mock_snap = {
        'params': {},
        'remeshed': False,
        'min_pot': -50.0,
        'current_div': 12.5,
        'current_div_mid': 8.3,
        'T_grids': [300.0, 310.0],
        'trans_last_frame': 0.85,
        'transparency': 0.85,
        'iteration': 1,
        'dt': 1e-10,
        'num_p': 100,
        'num_e': 50,
        'exit_vx_mean': 30000.0,
        'exit_v_mean': 31000.0,
        'exit_energy_mean_eV': 1200.0,
        'exit_count_step': 10,
        'exit_ion_current_step': 0.05,
        'exit_ion_current_avg': 0.05,
        'transmitted_ions_step': 10,
        'total_active_cells': 200,
        'low_ppc_cells': 5,
        'tracking_data': np.array([]),
        'mask_grids': [],
        'Tmap': None,
        'damage_map': None,
        'damage_version': 0,
        'X': np.zeros((10, 10)),
        'Y': np.zeros((10, 10)),
        'isBound': np.zeros((10, 10), dtype=bool),
        'injection_enabled': True,
        'has_active_particles': True,
        'perf_monitor': None,
        'prim_xy': np.empty((0, 2)),
        'cex_xy': np.empty((0, 2)),
        'elec_xy': np.empty((0, 2)),
        'e_prim_eV': np.array([]),
        'e_cex_eV': np.array([]),
        'prim_mask': np.array([], dtype=bool),
        'cex_mask': np.array([], dtype=bool),
        'p_isCEX': np.array([], dtype=bool),
        'p_vx': np.array([]),
        'p_vy': np.array([]),
        'e_x': np.array([]),
        'e_vx': np.array([]),
        'e_vy': np.array([]),
        'ions_subsampled': False,
        'ions_subsample_ratio': 1,
        'elec_subsampled': False,
    }
    app._process_snap(mock_snap)

    assert 12.5 in app.div_history, "div_history should contain 12.5"
    assert 8.3 in app.div_mid_history, "div_mid_history should contain 8.3"
    xdata_grid, ydata_grid = app.line_div.get_data()
    xdata_mid, ydata_mid = app.line_div_mid.get_data()
    assert ydata_grid[-1] == 12.5, f"line_div last y should be 12.5, got {ydata_grid[-1]}"
    assert ydata_mid[-1] == 8.3, f"line_div_mid last y should be 8.3, got {ydata_mid[-1]}"

    # Clean up
    app.close()
    print("ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_divergence_graph()
