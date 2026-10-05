"""
Configuration I/O and cross-section dataset loading.
"""
import os
import sys
import json
import numpy as np
from scipy.interpolate import UnivariateSpline


def _config_path():
    """
    Returns the path to config.json located next to the .exe (or next to
    main.py when running from source). Works on any PC, no hardcoding.
    """
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller .exe — use the folder containing the .exe
        base = os.path.dirname(sys.executable)
    else:
        # Running as normal Python — use the Python root folder (parent of gui/)
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "config.json")


def load_json_config(config_path=None):
    """
    Loads and parses a JSON configuration file for the simulation.
    If no path is provided, it defaults to the 'config.json' located next to the
    executable or in the project root. Returns a dictionary with configuration parameters if
    the file exists, or None if the file is not found (allowing the caller to fall
    back to default values).
    """
    if config_path is None:
        config_path = _config_path()

    if not os.path.isfile(config_path):
        return None  # no config present — caller will use defaults

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cross_sections_from_config(config):
    """Load and spline cross-section datasets defined in configuration."""
    cs_store = {}
    cs_cfg = config.get("cross_sections", {})
    smoothing = cs_cfg.get("spline_smoothing", 0.0)

    cs_files = {
        "CX": cs_cfg.get("cx_file", "").strip(),
        "SEE": cs_cfg.get("see_file", "").strip(),
        "Custom": cs_cfg.get("custom_file", "").strip(),
    }

    for label, fpath in cs_files.items():
        if not fpath:
            continue
        if not os.path.isfile(fpath):
            print(f"Warning: cross-section file not found for {label}: {fpath}")
            continue

        try:
            try:
                raw = np.loadtxt(fpath, delimiter=None, comments="#")
            except Exception:
                raw = np.loadtxt(fpath, delimiter=",", comments="#", skiprows=1)

            if raw.ndim != 2 or raw.shape[1] < 2:
                print(f"Warning: {fpath} must have at least 2 columns. Skipping.")
                continue

            energy = raw[:, 0]
            cs = raw[:, 1]
            order  = np.argsort(energy)
            energy = energy[order]
            cs = cs[order]

            log_e  = np.log10(np.maximum(energy, 1e-30))
            log_cs = np.log10(np.maximum(cs, 1e-50))
            spline = UnivariateSpline(log_e, log_cs, s=smoothing, k=3)

            cs_store[label] = {
                "energy": energy,
                "cs": cs,
                "spline": spline,
                "type": label
            }
        except Exception as e:
            print(f"Warning: failed to load {label} cross-section from {fpath}: {e}")

    return cs_store
