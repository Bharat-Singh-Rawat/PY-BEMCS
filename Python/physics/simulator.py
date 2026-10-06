"""
Primary digital twin simulator orchestrating domain, Poisson solver, particles, and step phases.
"""
import time as _time_mod
import numpy as np

from .backend import _NP_FP
from .constants import (
    Q_ELEM, M_XE, K_B, EPS_0, SIGMA_SB, MATERIAL_PRESETS
)
from .mixins.mesh import MeshMixin
from .mixins.materials import MaterialsMixin
from .mixins.particles import ParticlesMixin
from .mixins.diagnostics import DiagnosticsMixin
from .mixins.domain import DomainMixin
from .mixins.poisson import PoissonMixin

from .step.context import StepContext
from .step.injection import step_rf, step_inject
from .step.field_solve import step_deposit_and_solve
from .step.push import step_push
from .step.ion_walls import step_ion_interactions
from .step.electrons import step_electrons
from .step.thermal import step_thermal
from .step.probes import step_midhole_probe, step_end_accounting
from .step.collisions import step_cex


class DigitalTwinSimulator(
    MeshMixin,
    MaterialsMixin,
    ParticlesMixin,
    DiagnosticsMixin,
    DomainMixin,
    PoissonMixin,
):
    """Full PIC-MCC / electro-thermal simulator for multi-grid ion extraction systems."""

    def __init__(self):
        self.dt = 5e-10
        self.injection_stop_time = None
        self.injection_enabled   = True

        self.q      = Q_ELEM
        self.m_XE   = M_XE

        self.m_ion  = self.m_XE
        self.Z_ion  = 1
        self.q_ion  = self.q
        self.kB     = K_B
        self.eps0   = EPS_0
        self.m_e    = self.m_XE / 100.0

        self.user_cs = {}
        self.MATERIAL_PRESETS = MATERIAL_PRESETS

        mat = self.MATERIAL_PRESETS['Molybdenum']
        self.mat_k         = mat['k']
        self.mat_rho       = mat['rho']
        self.mat_cp        = mat['cp']
        self.emissivity    = mat['emissivity']
        self.alpha_thermal = mat['alpha']
        self.E_modulus     = mat['E_mod']
        self.sputter_Y_coeff = mat['Y_coeff']
        self.sputter_E_th    = mat['E_th']

        self.macro_weight        = 3e5
        self.sb_sigma            = SIGMA_SB
        self.thermal_accel       = 1e7
        self.injected_ions       = 0.0
        self.injected_ions_step  = 0.0
        self.transmitted_ions    = 0.0
        self.transmitted_ions_step = 0.0
        self.exit_ion_current_step = 0.0
        self.exit_ion_current_avg  = 0.0
        self.entered_optics      = 0.0
        self.entered_optics_step = 0.0
        self.lost_to_oob_step    = 0.0

        self._perf_monitor = None
        self._last_injection_area = 0.0
        self._last_neut_injected = 0
        self.last_poisson_iters = 0
        self.last_poisson_delta_V = 0.0
        self.last_poisson_rms = 0.0
        self.last_poisson_converged = True
        self.last_poisson_status = 'converged'
        self.last_poisson_anderson_steps = 0
        self.last_poisson_backtracks = 0

        self.Lx = 3
        self.Ly = 3
        self.dx = 0.02
        self.dy = 0.02

        self.nx = int(self.Lx / self.dx) + 1
        self.ny = int(self.Ly / self.dy) + 1
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)

        self.x_coords = np.linspace(0, self.Lx, self.nx)
        self.y_coords = np.linspace(0, self.Ly, self.ny)
        self.x_pts = self.x_coords
        self.y_pts = self.y_coords
        self.xpts = self.x_coords
        self.ypts = self.y_coords
        self.dx_cells = np.diff(self.x_coords)
        self.dy_cells = np.diff(self.y_coords)
        self.dx_min = float(self.dx)
        self.dx_max = float(self.dx)

        hx = np.zeros(self.nx, dtype=np.float64)
        hx[0] = 0.5 * self.dx_cells[0]
        hx[1:-1] = 0.5 * (self.dx_cells[:-1] + self.dx_cells[1:])
        hx[-1] = 0.5 * self.dx_cells[-1]
        self.hx_cells = hx
        self.cell_vol_1d = (hx * 1e-3) * (self.dy * 1e-3) * 1e-3
        self.cell_vol_2d = np.tile(self.cell_vol_1d, (self.ny, 1))
        self.cell_area_1d = (hx * 1e-3) * (self.dy * 1e-3)
        self.cell_area_2d = np.tile(self.cell_area_1d, (self.ny, 1))

        self.zone_boundaries = np.array([0.0, self.Lx / 3.0, 2.0 * self.Lx / 3.0, self.Lx], dtype=np.float64)
        self.zone_dx = np.array([self.dx, self.dx, self.dx], dtype=np.float64)
        self.zone_offsets = np.array([0, int(self.nx / 3), int(2 * self.nx / 3)], dtype=np.int32)

        self._recompute_cell_constants()

        self.T_grids  = []
        self.mask_grids = []
        self.V_dc     = None

        self.X, self.Y = np.meshgrid(self.x_pts, self.y_pts)

        self.iteration  = 0
        self.Tmap       = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
        self.T_map      = self.Tmap
        self.T_map_new  = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
        self.Tmapnew    = self.T_map_new

        self.laplacian_lu   = None
        self.is_interior_mask = None
        self.is_bound_mask  = None

        self.exit_vx_mean        = np.nan
        self.exit_v_mean         = np.nan
        self.exit_vx_std         = np.nan
        self.exit_v_std          = np.nan
        self.exit_energy_mean_eV = np.nan
        self.exit_count_step     = 0
        self.current_div         = np.nan
        self.current_div_mid     = np.nan
        self.total_active_cells  = 0
        self.low_ppc_cells       = 0
        self.min_ppc_threshold   = 3

        self.reset_arrays()

    def reset_arrays(self):
        self.max_p = 100000
        self.max_e = 100000

        self.p_x     = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_y     = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_vx    = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_vy    = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_vz    = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_isCEX = np.zeros(self.max_p, dtype=bool)
        self.num_p   = 0

        self.e_x  = np.zeros(self.max_e, dtype=_NP_FP)
        self.e_y  = np.zeros(self.max_e, dtype=_NP_FP)
        self.e_vx = np.zeros(self.max_e, dtype=_NP_FP)
        self.e_vy = np.zeros(self.max_e, dtype=_NP_FP)
        self.e_vz = np.zeros(self.max_e, dtype=_NP_FP)
        self.num_e = 0
        self.iteration = 0

        self.V        = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.rho      = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.isBound  = np.zeros((self.ny, self.nx), dtype=bool)
        self.V_fixed  = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.damage_map  = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._damage_version = 0
        self.eroded_depth= np.zeros((self.ny, self.nx), dtype=np.float64)
        self.Ex = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.Ey = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.Bx = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.By = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.Bz = np.zeros((self.ny, self.nx), dtype=_NP_FP)

        self.accum_ppc_map   = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.ppc_steps_count = 0
        self.current_ppc_map = np.zeros((self.ny, self.nx), dtype=int)

        self.energy_history_t    = []
        self.energy_history_ke_i = []
        self.energy_history_ke_e = []
        self.energy_history_fe   = []
        self.energy_history_tot  = []
        self._prev_total_energy  = None
        self.energy_warning_threshold = 0.05

        self.charge_history_t    = []
        self.charge_history_q_i  = []
        self.charge_history_q_e  = []
        self.charge_history_q_b  = []
        self.charge_history_q_free = []
        self.charge_history_q_net= []
        self._prev_total_charge  = None
        self.charge_warning_threshold = 0.10

    def step(self, params):
        if self.laplacian_lu is None:
            return False, np.nan, np.nan, self.T_grids, 0.0

        sim_mode = params.get('sim_mode', 'Both')
        self.iteration += 1
        t_current = self.iteration * self.dt
        grids = params.get('grids', [])

        self._step_t0 = _time_mod.perf_counter()
        self.transmitted_ions_step = 0.0
        self.transmitted3_step = 0.0
        self.exit_ion_current_step = 0.0
        self.entered_optics_step = 0.0
        self.injected_ions_step = 0.0
        self.lost_to_grid_step = 0.0
        self.lost_to_oob_step = 0.0

        ctx = StepContext(
            params=params,
            sim_mode=sim_mode,
            t_current=t_current,
            grids=grids
        )

        step_rf(self, ctx)
        step_inject(self, ctx)
        step_deposit_and_solve(self, ctx)
        step_push(self, ctx)
        step_ion_interactions(self, ctx)
        step_electrons(self, ctx)
        step_thermal(self, ctx)
        step_midhole_probe(self, ctx)
        step_cex(self, ctx)

        return step_end_accounting(self, ctx)
