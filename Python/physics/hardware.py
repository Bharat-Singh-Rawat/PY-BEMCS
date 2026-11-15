"""
Hardware discovery, memory estimation, and resource management for PY-BEMCS.

Provides dependency-free physical RAM, available RAM, and CPU detection using
ctypes on Windows and /proc/meminfo / os.sysconf on POSIX systems.
Supports saving and loading hardware profiles from configurations/hardware_profile.json.
"""
import os
import sys
import json
import ctypes
import platform
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any


@dataclass
class HardwareProfile:
    """System hardware capabilities and user-configurable resource ceilings."""
    total_ram_gb: float
    available_ram_gb: float
    logical_cores: int
    physical_cores: int
    max_ram_usage_percent: float = 80.0
    reserved_os_ram_gb: float = 4.0
    max_cores_to_use: int = 8
    max_particle_budget: float = 1_000_000.0
    allow_ssd_paging: bool = False
    vulkan_available: bool = False

    @property
    def usable_ram_gb(self) -> float:
        """Maximum RAM in GB that PY-BEMCS is permitted to allocate."""
        percent_limit = self.total_ram_gb * (self.max_ram_usage_percent / 100.0)
        reserve_limit = max(1.0, self.total_ram_gb - self.reserved_os_ram_gb)
        return min(percent_limit, reserve_limit, self.available_ram_gb * 1.1)


def get_raw_system_memory() -> Tuple[float, float]:
    """
    Query OS for (total_ram_bytes, available_ram_bytes).
    Returns (total_bytes, avail_bytes), or (16 GB, 8 GB) as fallback.
    """
    total_bytes = 0
    avail_bytes = 0

    if platform.system() == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_bytes = int(stat.ullTotalPhys)
                avail_bytes = int(stat.ullAvailPhys)
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        meminfo[key] = int(val) * 1024  # kB to bytes
                total_bytes = meminfo.get("MemTotal", 0)
                avail_bytes = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            total_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
            avail_bytes = total_bytes // 2
        except Exception:
            pass

    # Fallback if OS query failed
    if total_bytes <= 0:
        total_bytes = 16 * (1024 ** 3)
    if avail_bytes <= 0:
        avail_bytes = total_bytes // 2

    return float(total_bytes), float(avail_bytes)


def is_vulkan_supported() -> bool:
    """Check if Taichi Vulkan architecture is available without throwing."""
    try:
        from .backend import _GPU_AVAILABLE, _GPU_BACKEND
        return bool(_GPU_AVAILABLE and _GPU_BACKEND in ('vulkan', 'cuda'))
    except Exception:
        return False


def detect_hardware_profile(config_path: Optional[str] = None) -> HardwareProfile:
    """
    Detect machine hardware and optionally overlay settings from configurations/hardware_profile.json.
    """
    total_bytes, avail_bytes = get_raw_system_memory()
    total_gb = round(total_bytes / (1024 ** 3), 2)
    avail_gb = round(avail_bytes / (1024 ** 3), 2)
    logical_cores = os.cpu_count() or 4
    physical_cores = max(1, logical_cores // 2)

    profile = HardwareProfile(
        total_ram_gb=total_gb,
        available_ram_gb=avail_gb,
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        max_ram_usage_percent=80.0,
        reserved_os_ram_gb=4.0,
        max_cores_to_use=max(1, logical_cores - 2 if logical_cores > 4 else logical_cores),
        max_particle_budget=1_000_000.0,
        allow_ssd_paging=False,
        vulkan_available=is_vulkan_supported(),
    )

    # Locate config file if not provided
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_file = os.path.join(base_dir, "configurations", "hardware_profile.json")
    else:
        cfg_file = config_path

    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                data = json.load(f)
            sys_cfg = data.get("system", {})
            profile.max_ram_usage_percent = float(sys_cfg.get("max_ram_usage_percent", profile.max_ram_usage_percent))
            profile.reserved_os_ram_gb = float(sys_cfg.get("reserved_os_ram_gb", profile.reserved_os_ram_gb))
            profile.max_cores_to_use = int(sys_cfg.get("max_cores_to_use", profile.max_cores_to_use))
            profile.allow_ssd_paging = bool(sys_cfg.get("allow_ssd_paging", profile.allow_ssd_paging))

            budget_cfg = data.get("particle_budget", {})
            profile.max_particle_budget = max(
                1_000_000.0,
                float(budget_cfg.get("base_budget", profile.max_particle_budget))
            )
        except Exception as e:
            print(f"[Hardware Warning] Failed to load {cfg_file}: {e}")

    return profile


def save_hardware_profile(profile: HardwareProfile, config_path: Optional[str] = None) -> str:
    """Save hardware profile to JSON file."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_file = os.path.join(base_dir, "configurations", "hardware_profile.json")
    else:
        cfg_file = config_path

    os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
    out_dict = {
        "hardware_detected": {
            "total_ram_gb": profile.total_ram_gb,
            "available_ram_gb": profile.available_ram_gb,
            "logical_cores": profile.logical_cores,
            "physical_cores": profile.physical_cores,
            "vulkan_available": profile.vulkan_available,
        },
        "system": {
            "max_ram_usage_percent": profile.max_ram_usage_percent,
            "reserved_os_ram_gb": profile.reserved_os_ram_gb,
            "max_cores_to_use": profile.max_cores_to_use,
            "allow_ssd_paging": profile.allow_ssd_paging,
        },
        "particle_budget": {
            "base_budget": profile.max_particle_budget,
            "hard_particle_ceiling": 2_500_000.0,
        },
    }
    with open(cfg_file, "w") as f:
        json.dump(out_dict, f, indent=2)
    return cfg_file


def estimate_mesh_memory_gb(nx: int, ny: int, num_particles: int = 500_000) -> float:
    """
    Accurately estimate peak RAM footprint (in GB) for a 2D PIC-MCC simulation step.

    Components:
      1. Discrete Laplacian 2D SuperLU sparse factorization:
         Nonzero entries scale as O(nx * ny^1.45) for banded Poisson stencils.
         Double precision factor storage = nonzeros * 16 bytes.
      2. Particle arrays: 6 x float64 / bool per ion + electrons + CEX.
      3. Field arrays, potential maps, temperature maps, charge maps.
    """
    total_cells = max(nx * ny, 1)

    # SuperLU sparse factorization fill-in estimate
    lu_bytes = 16.0 * nx * (float(ny) ** 1.45)

    # Particle buffers (ions + electrons + scratch)
    particles = max(num_particles, 50_000)
    particle_bytes = particles * 48.0  # px, py, pvx, pvy, pvz (float64) + isCEX (bool)

    # 2D Grid arrays (V, rho, E_x, E_y, Tmap, isBound, masks, cell volumes)
    grid_bytes = total_cells * 64.0 * 8.0

    total_bytes = (lu_bytes + particle_bytes + grid_bytes) * 1.25  # 25% safety overhead
    return total_bytes / (1024.0 ** 3)


def estimate_payload_ram_gb(payload: dict) -> float:
    """
    Estimate the memory required by a scenario payload before launching it.
    """
    param_id = payload.get("param_id", "")
    val = payload.get("parameter_value", 1e17)
    base_params = payload.get("base_params", {})

    from physics.headless_sweep import apply_sweep_param_to_dict
    params = apply_sweep_param_to_dict(
        base_params,
        param_id,
        val,
        sweep_config=payload.get("sweep_config"),
    )

    n0 = float(params.get("n0_plasma", 1e17))
    Te_up = float(params.get("Te_up", 2.9))
    debye_mm = (8.854e-12 * Te_up / (1.602e-19 * max(n0, 1e12) * 0.61)) ** 0.5 * 1e3
    ratio = float(params.get(r"\deltax/debye_length", params.get("dx_over_debye", 0.8)))
    dx = max(0.001, ratio * debye_mm)
    dy = dx
    Lx = float(params.get("Lx", 12.0)) if params.get("Lx") else 12.0
    Ly = float(params.get("Ly", 5.0)) if params.get("Ly") else 5.0
    nx = max(12, int(Lx / dx) + 1)
    ny = max(12, int(Ly / dy) + 1)
    return estimate_mesh_memory_gb(nx, ny) + 1.2


def estimate_optimal_worker_concurrency(
    scenario_cells_list: list,
    hardware_profile: Optional[HardwareProfile] = None,
) -> Dict[int, int]:
    """
    For a list of total cell counts across sweep scenarios, compute the dynamic
    worker concurrency limit per scenario so total memory never exceeds usable RAM.
    Returns mapping {scenario_idx: max_concurrency}.
    """
    profile = hardware_profile or detect_hardware_profile()
    usable_ram = profile.usable_ram_gb
    max_cores = profile.max_cores_to_use

    concurrency_map = {}
    for idx, cells in enumerate(scenario_cells_list):
        # Approximate nx, ny if only total cells given
        ratio = 2.0  # typical Lx / Ly ratio
        ny = max(12, int((cells / ratio) ** 0.5))
        nx = max(12, int(cells / ny))
        scen_ram = estimate_mesh_memory_gb(nx, ny)

        # Allow at least 1 worker; cap concurrency by available RAM
        slots = max(1, min(max_cores, int(usable_ram / max(scen_ram, 0.2))))
        concurrency_map[idx] = slots

    return concurrency_map
