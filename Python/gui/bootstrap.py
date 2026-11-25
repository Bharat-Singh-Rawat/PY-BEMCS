"""
Environment and library path bootstrapping for PY-BEMCS.
Handles LD_LIBRARY_PATH patching for PyQt5 and CUDA runtime libraries.
"""
import os
import sys


def _collect_extra_lib_paths():
    """Return library dirs that must be on LD_LIBRARY_PATH before we load
    PyQt5 or CuPy. Qt5 ships inside the PyQt5 wheel; CUDA 12 runtime libs
    ship as separate nvidia-*-cu12 wheels that CuPy 13 doesn't auto-load."""
    paths = []

    try:
        import PyQt5  # noqa: F401
        qt_lib = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "lib")
        if os.path.isdir(qt_lib):
            paths.append(qt_lib)
    except Exception:
        pass

    try:
        import sysconfig
        site_pkgs = sysconfig.get_paths()["purelib"]
        nvidia_root = os.path.join(site_pkgs, "nvidia")
        if os.path.isdir(nvidia_root):
            for sub in sorted(os.listdir(nvidia_root)):
                lib = os.path.join(nvidia_root, sub, "lib")
                if os.path.isdir(lib):
                    paths.append(lib)
    except Exception:
        pass

    return paths


def _ensure_lib_paths():
    if os.environ.get("PYBEMCS_LD_PATCHED") == "1":
        return

    extras = _collect_extra_lib_paths()
    if not extras:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = current.split(":") if current else []
    missing = [p for p in extras if p not in current_parts[:len(extras)]]
    if not missing:
        return

    merged = ":".join(extras + [p for p in current_parts if p and p not in extras])
    os.environ["LD_LIBRARY_PATH"] = merged
    os.environ["PYBEMCS_LD_PATCHED"] = "1"
    # Re-exec so the dynamic linker picks up the prepended paths
    os.execv(sys.executable, [sys.executable] + sys.argv)


def init_environment():
    """Initialize library paths and platform environment variables."""
    _ensure_lib_paths()

    if not os.environ.get("QT_QPA_PLATFORM"):
        if sys.platform == "win32":
            os.environ["QT_QPA_PLATFORM"] = "windows"
        else:
            os.environ["QT_QPA_PLATFORM"] = "xcb;wayland;wayland-egl"


def _safe_start_dir():
    try:
        return os.getcwd()
    except Exception:
        return os.path.expanduser("~")
