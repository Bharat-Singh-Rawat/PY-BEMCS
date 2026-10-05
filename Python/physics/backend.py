"""
Backend detection and hardware initialization (Taichi Vulkan / CPU, CuPy GPU Poisson).
"""
import sys
import os
import numpy as np

# Taichi-on when running from Python, off when running from a frozen .exe
# Set PYBEMCS_FORCE_CPU=1 to force the NumPy CPU path (deterministic)
USE_TAICHI = (not getattr(sys, "frozen", False)) and os.environ.get("PYBEMCS_FORCE_CPU", "0") != "1"

if USE_TAICHI:
    import taichi as ti
    ti.init(arch=ti.vulkan, default_fp=ti.f32)
    _TI_FP = ti.f32
else:
    ti = None
    _TI_FP = None

_NP_FP = np.float32

# —- Optional GPU Poisson via CuPy (falls back silently if unavailable) —-
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    from cupyx.scipy.sparse.linalg import splu as cp_splu
    _GPU_POISSON_AVAILABLE = True
except ImportError:
    cp = None
    cp_sp = None
    cp_splu = None
    _GPU_POISSON_AVAILABLE = False

_USE_GPU_POISSON = False
_GPU_POISSON = _GPU_POISSON_AVAILABLE and _USE_GPU_POISSON


def _ti_arr(arr):
    """Return a C-contiguous float32 copy suitable for Taichi ndarray args."""
    return np.ascontiguousarray(arr, dtype=_NP_FP)


def _ti_arr_i32(arr):
    """Return a C-contiguous int32 copy suitable for Taichi ndarray args."""
    return np.ascontiguousarray(arr, dtype=np.int32)
