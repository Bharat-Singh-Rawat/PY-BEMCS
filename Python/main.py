"""
PY-BEMCS GUI Application Entry Point.
All simulation defaults are loaded from config.json.
Modules are cleanly partitioned in the `gui/` and `physics/` packages.
"""
import os
import sys

from gui.bootstrap import init_environment
init_environment()

from PyQt5.QtWidgets import QApplication, QMessageBox
from gui.config_io import _config_path, load_json_config, load_cross_sections_from_config
from gui.widgets import ScientificSpinBox
from gui.workers import PyInstallerWorker, SimulationWorker
from gui.dialogs import (
    BeamSpeciesDialog,
    GridMaterialDialog,
    AdvancedSettingsDialog,
    GUISettingsDialog,
)
from gui.app.main_window import DigitalTwinApp

__all__ = [
    "DigitalTwinApp",
    "ScientificSpinBox",
    "PyInstallerWorker",
    "SimulationWorker",
    "BeamSpeciesDialog",
    "GridMaterialDialog",
    "AdvancedSettingsDialog",
    "GUISettingsDialog",
    "load_json_config",
    "load_cross_sections_from_config",
    "_config_path",
    "main",
]


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DigitalTwinApp()

    if len(sys.argv) > 1:
        cfg = load_json_config(sys.argv[1])
        if cfg is not None:
            try:
                resolved_path = sys.argv[1]
                if not os.path.isfile(resolved_path):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    cand1 = os.path.join(script_dir, "configurations", sys.argv[1])
                    if os.path.isfile(cand1):
                        resolved_path = cand1
                window.current_config_path = resolved_path
                window.current_config_name = os.path.basename(resolved_path)
                window.apply_config(cfg, config_name=window.current_config_name)
            except Exception as e:
                QMessageBox.critical(window, "Config Error", f"Failed to load config:\n{e}")
        else:
            QMessageBox.warning(window, "File Not Found",
                                f"Config file not found:\n{sys.argv[1]}")

    window.show()
    exit_code = app.exec_()
    try:
        window._shutdown_background_tasks()
    except Exception:
        pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()