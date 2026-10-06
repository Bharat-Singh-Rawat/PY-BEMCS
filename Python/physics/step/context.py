"""
State container passed across sub-steps during a single simulation iteration.
"""
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np


@dataclass
class StepContext:
    params: dict
    sim_mode: str
    t_current: float
    grids: list

    # Mesh & arrays prepared during field solve
    _dy: np.float32 = np.float32(0.0)
    zone_b_ti: Any = None
    zone_dx_ti: Any = None
    zone_off_ti: Any = None
    num_zones: int = 0

    # Old coordinates before Boris push
    num_p_step: int = 0
    p_x_old: Optional[np.ndarray] = None
    p_y_old: Optional[np.ndarray] = None

    # Step outputs & diagnostics
    remeshed: bool = False
    current_div: float = np.nan
    current_div_mid: float = np.nan
    min_pot: float = np.nan
    trans_last_frame: float = 0.0
