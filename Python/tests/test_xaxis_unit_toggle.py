"""
Unit test for X-axis unit toggle feature (Iterations vs Microseconds)
on Beam Divergence and Midline Potential graphs, including independent graph toggling.
"""
import os
import sys
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYBEMCS_FORCE_CPU"] = "1"

from PyQt5.QtWidgets import QApplication

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.app import DigitalTwinApp


def test_xaxis_unit_toggle():
    qt_app = QApplication.instance() or QApplication(sys.argv)
    app = DigitalTwinApp()
    app.build_domain()

    # 1. Verify buttons and menus exist
    assert hasattr(app, 'btn_ebs_x_menu'), "app missing btn_ebs_x_menu"
    assert hasattr(app, 'btn_div_x_menu'), "app missing btn_div_x_menu"
    assert hasattr(app, 'btn_xaxis_menu'), "app missing btn_xaxis_menu alias"
    assert app.btn_ebs_x_menu.text() == "▼"
    assert app.btn_div_x_menu.text() == "▼"
    assert app.btn_ebs_x_menu.parent() == app.canvas
    assert app.btn_div_x_menu.parent() == app.canvas

    # Menus for each graph
    assert hasattr(app, 'menu_ebs_xaxis'), "app missing menu_ebs_xaxis"
    assert hasattr(app, '_ebs_xaxis_actions'), "app missing _ebs_xaxis_actions"
    assert "iterations" in app._ebs_xaxis_actions
    assert "microseconds" in app._ebs_xaxis_actions

    assert hasattr(app, 'menu_div_xaxis_btn'), "app missing menu_div_xaxis_btn"
    assert hasattr(app, '_div_xaxis_actions'), "app missing _div_xaxis_actions"
    assert "iterations" in app._div_xaxis_actions
    assert "microseconds" in app._div_xaxis_actions

    # Menu in divergence dropdown
    assert hasattr(app, 'menu_div_xaxis'), "app missing menu_div_xaxis"
    assert hasattr(app, '_div_xaxis_unit_actions'), "app missing _div_xaxis_unit_actions"

    # 2. Check initial state: both graphs use iterations
    assert app.ebs_xaxis_unit == "iterations"
    assert app.div_xaxis_unit == "iterations"
    assert app._ebs_xaxis_actions["iterations"].isChecked() is True
    assert app._ebs_xaxis_actions["microseconds"].isChecked() is False
    assert app._div_xaxis_actions["iterations"].isChecked() is True
    assert app._div_xaxis_actions["microseconds"].isChecked() is False
    assert app.ax_ebs.get_xlabel() == "Iteration"
    assert app.ax_div.get_xlabel() == "Iteration"

    # 3. Simulate snapshot in iterations mode
    mock_snap = {
        'params': {},
        'remeshed': False,
        'min_pot': -60.0,
        'current_div': 14.2,
        'current_div_mid': 9.1,
        'T_grids': [300.0, 310.0],
        'trans_last_frame': 0.85,
        'transparency': 0.85,
        'iteration': 100,
        'dt': 1e-9,  # 1 ns -> 100 iters = 0.1 µs
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

    xdata_ebs, ydata_ebs = app.line_ebs.get_data()
    xdata_div, ydata_div = app.line_div.get_data()
    assert xdata_ebs[-1] == 100
    assert xdata_div[-1] == 100

    # 4. INDEPENDENCE TEST: Switch ONLY Midline Potential (ax_ebs) to microseconds
    app._ebs_xaxis_actions["microseconds"].trigger()

    # Midline potential changed to microseconds
    assert app.ebs_xaxis_unit == "microseconds"
    assert app._ebs_xaxis_actions["microseconds"].isChecked() is True
    assert app.ax_ebs.get_xlabel() == "Time [µs]"
    xdata_ebs_us, _ = app.line_ebs.get_data()
    assert np.isclose(xdata_ebs_us[-1], 0.1)

    # BUT Beam Divergence MUST STILL BE in iterations!
    assert app.div_xaxis_unit == "iterations"
    assert app._div_xaxis_actions["iterations"].isChecked() is True
    assert app._div_xaxis_actions["microseconds"].isChecked() is False
    assert app.ax_div.get_xlabel() == "Iteration"
    xdata_div_it, _ = app.line_div.get_data()
    assert xdata_div_it[-1] == 100

    # 5. INDEPENDENCE TEST: Switch Beam Divergence to microseconds as well
    app._div_xaxis_actions["microseconds"].trigger()

    assert app.div_xaxis_unit == "microseconds"
    assert app.ax_div.get_xlabel() == "Time [µs]"
    xdata_div_us, _ = app.line_div.get_data()
    assert np.isclose(xdata_div_us[-1], 0.1)

    # 6. Process another step while both in microseconds mode
    mock_snap2 = dict(mock_snap)
    mock_snap2['iteration'] = 200  # 200 ns = 0.2 µs
    mock_snap2['min_pot'] = -62.0
    mock_snap2['current_div'] = 13.8
    app._process_snap(mock_snap2)

    xdata_ebs_us2, _ = app.line_ebs.get_data()
    xdata_div_us2, _ = app.line_div.get_data()
    assert np.isclose(xdata_ebs_us2[-1], 0.2)
    assert np.isclose(xdata_div_us2[-1], 0.2)

    # 7. INDEPENDENCE TEST: Switch ONLY Midline Potential back to iterations
    app._ebs_xaxis_actions["iterations"].trigger()

    # Midline potential back to iterations
    assert app.ebs_xaxis_unit == "iterations"
    assert app.ax_ebs.get_xlabel() == "Iteration"
    xdata_ebs_back, _ = app.line_ebs.get_data()
    assert xdata_ebs_back[-1] == 200

    # Beam Divergence MUST STILL REMAIN in microseconds!
    assert app.div_xaxis_unit == "microseconds"
    assert app.ax_div.get_xlabel() == "Time [µs]"
    xdata_div_remain, _ = app.line_div.get_data()
    assert np.isclose(xdata_div_remain[-1], 0.2)

    # 8. Switch Beam Divergence back to iterations
    app._div_xaxis_actions["iterations"].trigger()
    assert app.div_xaxis_unit == "iterations"
    assert app.ax_div.get_xlabel() == "Iteration"
    xdata_div_back, _ = app.line_div.get_data()
    assert xdata_div_back[-1] == 200

    # 9. Test direct methods set_ebs_xaxis_unit and set_div_xaxis_unit
    app.set_ebs_xaxis_unit("microseconds")
    assert app.ebs_xaxis_unit == "microseconds"
    assert app.div_xaxis_unit == "iterations"

    app.set_div_xaxis_unit("microseconds")
    assert app.ebs_xaxis_unit == "microseconds"
    assert app.div_xaxis_unit == "microseconds"

    # 10. Test config export
    params = app.get_params()
    assert params.get('ebs_xaxis_unit') == 'microseconds'
    assert params.get('div_xaxis_unit') == 'microseconds'

    # Clean up
    app.close()
    print("ALL INDEPENDENT X-AXIS UNIT TOGGLE TESTS PASSED!")
