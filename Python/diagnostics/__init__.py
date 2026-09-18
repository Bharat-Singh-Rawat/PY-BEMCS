"""
Diagnostics package for PY-BEMCS.

Contains performance monitoring, physical conservation budgets (energy & charge),
and Qt diagnostic pop-up visualization windows.
"""

from .performance_monitor import PerformanceMonitor, StepDiagnostics
from .metrics import compute_energy_budget, compute_charge_budget
from .cs_viewer_window import CrossSectionViewerWindow
from .iedf_window import IEDFWindow
from .ppc_window import PPCWindow
from .energy_window import PhysicalConstraintsWindow, TotalEnergyWindow
from .charge_window import TotalChargeMonitorWindow, ChargeMonitorWindow
from .perf_window import PerformanceMonitorWindow

__all__ = [
    "PerformanceMonitor",
    "StepDiagnostics",
    "compute_energy_budget",
    "compute_charge_budget",
    "CrossSectionViewerWindow",
    "IEDFWindow",
    "PPCWindow",
    "PhysicalConstraintsWindow",
    "TotalEnergyWindow",
    "TotalChargeMonitorWindow",
    "ChargeMonitorWindow",
    "PerformanceMonitorWindow",
]
