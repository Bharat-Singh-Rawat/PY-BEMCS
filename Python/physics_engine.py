# EXPERIMENTAL VERSION
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
import taichi as ti
import sys

# Taichi-on when running from Python, off when running from a frozen .exe
USE_TAICHI = not getattr(sys, "frozen", False)

if USE_TAICHI:
    ti.init(arch=ti.vulkan, default_fp=ti.f32)
_TI_FP = ti.f32
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

# Disabled by default: CuPy's sparse LU + per-iteration kernel launches are
# slower than SciPy's SuperLU for grids this small (~30k unknowns). Flip to
# True to experiment on larger grids where the GPU solver may actually win.
_USE_GPU_POISSON = False
_GPU_POISSON = _GPU_POISSON_AVAILABLE and _USE_GPU_POISSON

# =============================================================================
# TAICHI KERNELS
# =============================================================================

@ti.kernel
def accumulate_rho_taichi(
    x: ti.types.ndarray(dtype=_TI_FP),
    y: ti.types.ndarray(dtype=_TI_FP),
    rho: ti.types.ndarray(dtype=_TI_FP),
    num_p: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    charge_density: ti.f32
):
    """Parallel density accumulation replacing np.add.at"""
    for i in range(num_p):
        ix = ti.cast(ti.round(x[i] / dx), ti.i32)
        iy = ti.cast(ti.round(y[i] / dy), ti.i32)

        # Clamp to avoid out-of-bounds access
        ix = ti.max(1, ti.min(ix, nx - 2))
        iy = ti.max(1, ti.min(iy, ny - 2))

        # Taichi handles atomic additions automatically on the GPU
        rho[iy, ix] += charge_density


@ti.kernel
def push_particles_boris_taichi(
    x:   ti.types.ndarray(dtype=_TI_FP),
    y:   ti.types.ndarray(dtype=_TI_FP),
    vx:  ti.types.ndarray(dtype=_TI_FP),
    vy:  ti.types.ndarray(dtype=_TI_FP),
    vz:  ti.types.ndarray(dtype=_TI_FP),
    Ex:  ti.types.ndarray(dtype=_TI_FP),
    Ey:  ti.types.ndarray(dtype=_TI_FP),
    Bx:  ti.types.ndarray(dtype=_TI_FP),
    By:  ti.types.ndarray(dtype=_TI_FP),
    Bz:  ti.types.ndarray(dtype=_TI_FP),
    num_p: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    dt: ti.f32,
    q_m: ti.f32
):
    """Parallel bilinear interpolation with 2D3V Boris Algorithm"""
    for i in range(num_p):
        px = x[i]
        py = y[i]

        idx_x = px / dx
        idx_y = py / dy

        ix0 = ti.cast(ti.floor(idx_x), ti.i32)
        iy0 = ti.cast(ti.floor(idx_y), ti.i32)

        ix0 = ti.max(0, ti.min(ix0, nx - 2))
        iy0 = ti.max(0, ti.min(iy0, ny - 2))

        fx = idx_x - ti.cast(ix0, ti.f32)
        fy = idx_y - ti.cast(iy0, ti.f32)

        Ex_p = Ex[iy0, ix0]*(1.0-fx)*(1.0-fy) + Ex[iy0, ix0+1]*fx*(1.0-fy) + \
               Ex[iy0+1, ix0]*(1.0-fx)*fy      + Ex[iy0+1, ix0+1]*fx*fy
        Ey_p = Ey[iy0, ix0]*(1.0-fx)*(1.0-fy) + Ey[iy0, ix0+1]*fx*(1.0-fy) + \
               Ey[iy0+1, ix0]*(1.0-fx)*fy      + Ey[iy0+1, ix0+1]*fx*fy

        Bx_p = Bx[iy0, ix0]*(1.0-fx)*(1.0-fy) + Bx[iy0, ix0+1]*fx*(1.0-fy) + \
               Bx[iy0+1, ix0]*(1.0-fx)*fy      + Bx[iy0+1, ix0+1]*fx*fy
        By_p = By[iy0, ix0]*(1.0-fx)*(1.0-fy) + By[iy0, ix0+1]*fx*(1.0-fy) + \
               By[iy0+1, ix0]*(1.0-fx)*fy      + By[iy0+1, ix0+1]*fx*fy
        Bz_p = Bz[iy0, ix0]*(1.0-fx)*(1.0-fy) + Bz[iy0, ix0+1]*fx*(1.0-fy) + \
               Bz[iy0+1, ix0]*(1.0-fx)*fy      + Bz[iy0+1, ix0+1]*fx*fy

        # Boris pusher
        v_minus_x = vx[i] + (q_m * Ex_p * dt) / 2.0
        v_minus_y = vy[i] + (q_m * Ey_p * dt) / 2.0
        v_minus_z = vz[i]

        t_x = (q_m * Bx_p * dt) / 2.0
        t_y = (q_m * By_p * dt) / 2.0
        t_z = (q_m * Bz_p * dt) / 2.0
        t_mag_sq = t_x**2 + t_y**2 + t_z**2

        s_x = 2.0 * t_x / (1.0 + t_mag_sq)
        s_y = 2.0 * t_y / (1.0 + t_mag_sq)
        s_z = 2.0 * t_z / (1.0 + t_mag_sq)

        v_prime_x = v_minus_x + (v_minus_y * t_z - v_minus_z * t_y)
        v_prime_y = v_minus_y + (v_minus_z * t_x - v_minus_x * t_z)
        v_prime_z = v_minus_z + (v_minus_x * t_y - v_minus_y * t_x)

        # Cross product 2: v_plus = v_minus + (v_prime x s)
        v_plus_x = v_minus_x + (v_prime_y * s_z - v_prime_z * s_y)
        v_plus_y = v_minus_y + (v_prime_z * s_x - v_prime_x * s_z)
        v_plus_z = v_minus_z + (v_prime_x * s_y - v_prime_y * s_x)

        # STEP 3: Second half E-field acceleration
        vx[i] = v_plus_x + (q_m * Ex_p * dt) / 2.0
        vy[i] = v_plus_y + (q_m * Ey_p * dt) / 2.0
        vz[i] = v_plus_z

        # =====================================================================
        # KINEMATIC UPDATE
        # =====================================================================
        x[i] += vx[i] * dt * 1000.0
        y[i] += vy[i] * dt * 1000.0


@ti.kernel
def thermal_conduction_taichi(
    T:    ti.types.ndarray(dtype=_TI_FP),
    T_new:ti.types.ndarray(dtype=_TI_FP),
    mask: ti.types.ndarray(dtype=ti.i32),
    nx: ti.i32,
    ny: ti.i32,
    Fo_x: ti.f32,
    Fo_y: ti.f32
):
    """Parallel 2D Finite Difference Thermal Conduction"""
    for iy, ix in ti.ndrange(ny, nx):
        if mask[iy, ix] == 1:
            T_l = T[iy, ix]
            if ix > 0 and mask[iy, ix-1] == 1: T_l = T[iy, ix-1]
            T_r = T[iy, ix]
            if ix < nx-1 and mask[iy, ix+1] == 1: T_r = T[iy, ix+1]
            T_u = T[iy, ix]
            if iy < ny-1 and mask[iy+1, ix] == 1: T_u = T[iy+1, ix]
            T_d = T[iy, ix]
            if iy > 0 and mask[iy-1, ix] == 1: T_d = T[iy-1, ix]
            dT = Fo_x*(T_l - 2.0*T[iy, ix] + T_r) + Fo_y*(T_d - 2.0*T[iy, ix] + T_u)
            T_new[iy, ix] = ti.max(T[iy, ix] + dT, 300.0)
        else:
            T_new[iy, ix] = T[iy, ix]


# =============================================================================
# CPU FALLBACK VERSIONS
# =============================================================================

def accumulate_rho_cpu(x, y, rho, num_p, dx, dy, nx, ny, charge_density):
    """Vectorised NGP charge deposition — replaces the Python particle loop."""
    n = int(num_p)
    if n == 0:
        return
    ix = np.clip(np.round(x[:n] / dx).astype(np.int32), 1, nx - 2)
    iy = np.clip(np.round(y[:n] / dy).astype(np.int32), 1, ny - 2)
    np.add.at(rho, (iy, ix), charge_density)


def push_particles_boris_cpu(x, y, vx, vy, vz,
                              Ex, Ey, Bx, By, Bz,
                              num_p, dx, dy, nx, ny, dt, q_m):
    """Vectorised 2D3V Boris pusher — replaces the Python particle loop."""
    n = int(num_p)
    if n == 0:
        return

    # —- Bilinear (CIL) field interpolation —-
    nx_m1 = nx - 1
    ny_m1 = ny - 1

    idx_x = x[:n] / dx
    idx_y = y[:n] / dy
    ix0 = np.clip(np.floor(idx_x).astype(np.int32), 0, nx_m1 - 1)
    iy0 = np.clip(np.floor(idx_y).astype(np.int32), 0, ny_m1 - 1)
    fx  = idx_x - ix0
    fy  = idx_y - iy0
    ix1 = np.minimum(ix0 + 1, nx_m1)
    iy1 = np.minimum(iy0 + 1, ny_m1)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx          * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx          * fy

    def interp(F):
        return F[iy0, ix0]*w00 + F[iy0, ix1]*w10 + F[iy1, ix0]*w01 + F[iy1, ix1]*w11

    Ex_p = interp(Ex);  Ey_p = interp(Ey)
    Bx_p = interp(Bx);  By_p = interp(By);  Bz_p = interp(Bz)

    # —- Boris algorithm (vectorised) —-
    hqmdt = 0.5 * q_m * dt          # half charge-mass-time factor

    # Step 1: first half E-field kick  ->  v_minus
    vmx = vx[:n] + hqmdt * Ex_p
    vmy = vy[:n] + hqmdt * Ey_p
    vmz = vz[:n]  # no Ez in 2D

    # Step 2: magnetic rotation
    tx = hqmdt * Bx_p
    ty = hqmdt * By_p
    tz = hqmdt * Bz_p
    t2 = tx*tx + ty*ty + tz*tz
    sx = 2.0 * tx / (1.0 + t2)
    sy = 2.0 * ty / (1.0 + t2)
    sz = 2.0 * tz / (1.0 + t2)

    # v_prime = v_minus + v_minus x t
    vpx = vmx + (vmy*tz - vmz*ty)
    vpy = vmy + (vmz*tx - vmx*tz)
    vpz = vmz + (vmx*ty - vmy*tx)

    # v_plus = v_minus + v_prime x s
    vpx2 = vmx + (vpy*sz - vpz*sy)
    vpy2 = vmy + (vpz*sx - vpx*sz)
    vpz2 = vmz + (vpx*sy - vpy*sx)

    # Step 3: second half E-field kick
    vx[:n] = vpx2 + hqmdt * Ex_p
    vy[:n] = vpy2 + hqmdt * Ey_p
    vz[:n] = vpz2

    # Position update (mm units — same convention as Taichi kernel)
    x[:n] += vx[:n] * dt * 1000.0
    y[:n] += vy[:n] * dt * 1000.0


# =============================================================================
# HELPER: contiguous f32 view for Taichi
# =============================================================================

def _ti_arr(arr):
    """Return a C-contiguous float32 copy suitable for Taichi ndarray args."""
    return np.ascontiguousarray(arr, dtype=_NP_FP)


# =============================================================================
# STANDALONE HELPERS  (usable without instantiating the simulator)
# =============================================================================

def compute_debye_upstream_gap(n0: float, Te_up: float) -> float:
    """Return the physics-recommended upstream gap [mm] based on Debye length.

    Sheath-formation criterion (literature-based):
      n0 <= 1e17  m^-3  →  80 * λ_D
      1e17 < n0 <= 4e17 →  40 * λ_D
      n0 > 4e17         →  30 * λ_D

    Parameters
    —————
    n0 : float
        Upstream plasma density [m^-3]
    Te_up : float
        Upstream electron temperature [eV]

    Returns
    ———-
    float
        Recommended upstream gap in [mm]
    """
    eps0 = 8.854e-12
    q    = 1.6e-19
    debye_m  = np.sqrt(eps0 * Te_up / (q * n0 * 0.61))
    debye_mm = debye_m * 1e3

    if n0 <= 1e17:
        n_debye = 80
    elif n0 <= 4e17:
        n_debye = 40
    else:
        n_debye = 30

    return n_debye * debye_mm


# =============================================================================
# MAIN SIMULATOR CLASS
# =============================================================================

class DigitalTwinSimulator:
    def __init__(self):
        self.dt = 5e-10
        self.injection_stop_time = None
        self.injection_enabled   = True

        self.q      = 1.602e-19 #charge of ion in SI units
        self.m_XE   = 131.293 * 1.6605e-27 #mass of xenon ion in SI units

        self.m_ion  = self.m_XE     #mass of ion in SI units
        self.Z_ion  = 1             #charge state of ion
        self.q_ion  = self.q        #charge of ion in SI units
        self.kB     = 1.380649e-23  #boltzmann constant in SI units
        self.eps0   = 8.854e-12     #permittivity of free space in SI units
        self.m_e    = self.m_XE / 100.0 #mass of electron in SI units

        self.user_cs = {}

        self.MATERIAL_PRESETS = {
            'Molybdenum':   {'k': 138.0,  'rho': 10280.0, 'cp': 250.0,
                             'emissivity': 0.80, 'alpha': 4.8e-6,
                             'E_mod': 329e9, 'Y_coeff': 1.05e-4, 'E_th': 30.0},
            'Steel (SS316)':{'k': 16.3,   'rho': 8000.0,  'cp': 500.0,
                             'emissivity': 0.60, 'alpha': 16.0e-6,
                             'E_mod': 193e9, 'Y_coeff': 2.8e-4,  'E_th': 25.0},
            'Titanium':     {'k': 21.9,   'rho': 4507.0,  'cp': 520.0,
                             'emissivity': 0.50, 'alpha': 8.6e-6,
                             'E_mod': 116e9, 'Y_coeff': 1.8e-4,  'E_th': 20.0},
            'Graphite':     {'k': 120.0,  'rho': 2200.0,  'cp': 710.0,
                             'emissivity': 0.85, 'alpha': 3.0e-6,
                             'E_mod': 11e9,  'Y_coeff': 3.5e-4,  'E_th': 15.0},
        }

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
        self.sb_sigma            = 5.67e-8
        self.thermal_accel       = 1e7
        self.injected_ions       = 0.0
        self.injected_ions_step  = 0.0
        self.transmitted_ions    = 0.0
        self.transmitted_ions_step = 0.0
        self.entered_optics      = 0.0
        self.entered_optics_step = 0.0

        self.Lx = 3
        self.Ly = 3
        self.dx = 0.02 # original  0.015
        self.dy = 0.02 # original  0.015

        self.nx = int(self.Lx / self.dx) + 1
        self.ny = int(self.Ly / self.dy) + 1
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)

        self._recompute_cell_constants()

        self.T_grids  = []
        self.mask_grids = []
        self.V_dc     = None

        self.x_pts = np.linspace(0, self.Lx, self.nx)
        self.y_pts = np.linspace(0, self.Ly, self.ny)
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
        self.total_active_cells  = 0
        self.low_ppc_cells       = 0

        self.reset_arrays()

    # —————————————————————————————————
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
        self.eroded_depth= np.zeros((self.ny, self.nx), dtype=np.float64)
        self.Ex = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.Ey = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.Bx = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.By = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.Bz = np.zeros((self.ny, self.nx), dtype=_NP_FP)

        self.accum_ppc_map   = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.ppc_steps_count = 0
        self.current_ppc_map = np.zeros((self.ny, self.nx), dtype=int)

        # Energy conservation tracking
        self.energy_history_t    = []   # simulation time [s]
        self.energy_history_ke_i = []   # ion kinetic energy [J]
        self.energy_history_ke_e = []   # electron kinetic energy [J]
        self.energy_history_fe   = []   # electrostatic field energy [J]
        self.energy_history_tot  = []   # total energy [J]
        self._prev_total_energy  = None # for step-to-step change detection
        # Warning threshold: relative change in total energy per step (5%)
        self.energy_warning_threshold = 0.05

    def _recompute_cell_constants(self):
        self.C_cell = self.mat_rho * (self.dx*1e-3) * (self.dy*1e-3) * 1e-3 * self.mat_cp
        self.A_cell = 2 * (self.dx*1e-3) * 1e-3

    def set_material(self, name=None, props=None):
        """
        Set grid material by preset name or custom property dict.
        props keys: k, rho, cp, emissivity, alpha, Y_coeff, E_th
        """
        if name and name in self.MATERIAL_PRESETS:
            mat = self.MATERIAL_PRESETS[name]
        elif props:
            mat = props
        else:
            return
        self.mat_k         = mat['k']
        self.mat_rho       = mat['rho']
        self.mat_cp        = mat['cp']
        self.emissivity    = mat['emissivity']
        self.alpha_thermal = mat['alpha']
        self.E_modulus     = mat['E_mod']
        self.sputter_Y_coeff = mat['Y_coeff']
        self.sputter_E_th    = mat['E_th']
        self._recompute_cell_constants()

    def lookup_user_cs(self, cs_type_prefix, energy_eV):
        for label, ds in self.user_cs.items():
            if label.startswith(cs_type_prefix) and ds.get('spline') is not None:
                log_e = np.log10(np.maximum(energy_eV, 1e-30))
                e_min = np.log10(max(ds['energy'][0], 1e-30))
                e_max = np.log10(ds['energy'][-1])
                log_e = np.clip(log_e, e_min, e_max)
                log_cs = ds['spline'](log_e)
                return 10.0 ** log_cs
        return None

    def _add_ions(self, x, y, vx, vy, vz, is_cex):
        n_new = len(x)
        if self.num_p + n_new > self.max_p:
            new_max = max(self.max_p * 2, self.num_p + n_new)
            self.p_x     = np.pad(self.p_x,     (0, new_max - self.max_p))
            self.p_y     = np.pad(self.p_y,     (0, new_max - self.max_p))
            self.p_vx    = np.pad(self.p_vx,    (0, new_max - self.max_p))
            self.p_vy    = np.pad(self.p_vy,    (0, new_max - self.max_p))
            self.p_vz    = np.pad(self.p_vz,    (0, new_max - self.max_p))
            self.p_isCEX = np.pad(self.p_isCEX, (0, new_max - self.max_p))
            self.max_p   = new_max
        s = self.num_p; e = s + n_new
        self.p_x[s:e]     = x
        self.p_y[s:e]     = y
        self.p_vx[s:e]    = vx
        self.p_vy[s:e]    = vy
        self.p_vz[s:e]    = vz
        self.p_isCEX[s:e] = is_cex
        self.num_p += n_new

    def _add_electrons(self, x, y, vx, vy, vz):
        n_new = len(x)
        if self.num_e + n_new > self.max_e:
            new_max = max(self.max_e * 2, self.num_e + n_new)
            self.e_x  = np.pad(self.e_x,  (0, new_max - self.max_e))
            self.e_y  = np.pad(self.e_y,  (0, new_max - self.max_e))
            self.e_vx = np.pad(self.e_vx, (0, new_max - self.max_e))
            self.e_vy = np.pad(self.e_vy, (0, new_max - self.max_e))
            self.e_vz = np.pad(self.e_vz, (0, new_max - self.max_e))
            self.max_e = new_max
        s = self.num_e; e = s + n_new
        self.e_x[s:e]  = x
        self.e_y[s:e]  = y
        self.e_vx[s:e] = vx
        self.e_vy[s:e] = vy
        self.e_vz[s:e] = vz
        self.num_e += n_new

    # —————————————————————————————————
    def build_sparse_matrix(self):
        periodic_y = getattr(self, 'periodic_y', False)
        N   = self.nx * self.ny
        idx = np.arange(N)
        y_  = idx // self.nx
        x_  = idx  % self.nx

        is_bound   = self.isBound.flatten()
        is_right   = (x_ == self.nx-1) & ~is_bound

        if periodic_y:
            # Bottom and top non-bound rows become interior with periodic coupling
            is_top       = np.zeros(N, dtype=bool)
            is_bottom    = np.zeros(N, dtype=bool)
            is_per_bot   = (y_ == 0)          & ~is_bound & ~is_right
            is_per_top   = (y_ == self.ny-1)  & ~is_bound & ~is_right
            is_interior  = ~is_bound & ~is_right & ~is_per_bot & ~is_per_top
        else:
            is_top      = (y_ == self.ny-1) & ~is_bound & ~is_right
            is_bottom   = (y_ == 0)         & ~is_bound & ~is_right & ~is_top
            is_interior = ~is_bound & ~is_right & ~is_top & ~is_bottom
            is_per_bot  = np.zeros(N, dtype=bool)
            is_per_top  = np.zeros(N, dtype=bool)

        self.is_interior_mask  = is_interior
        self.is_bound_mask     = is_bound
        # All non-Dirichlet, non-Neumann-right cells get rho as RHS source
        self.is_rhs_rho_mask   = is_interior | is_per_bot | is_per_top

        row, col, data = [], [], []

        # —- Dirichlet (fixed-voltage) cells —-
        idx_b = idx[is_bound]
        row.append(idx_b); col.append(idx_b); data.append(np.ones_like(idx_b))

        # —- Neumann right: V[iy, nx-1] = V[iy, nx-2] —-
        idx_r = idx[is_right]
        row.append(idx_r); col.append(idx_r);   data.append( np.ones_like(idx_r))
        row.append(idx_r); col.append(idx_r-1); data.append(-np.ones_like(idx_r))

        if not periodic_y:
            # —- Neumann top: V[ny-1, ix] = V[ny-2, ix] —-
            idx_t = idx[is_top]
            row.append(idx_t); col.append(idx_t);         data.append( np.ones_like(idx_t))
            row.append(idx_t); col.append(idx_t-self.nx); data.append(-np.ones_like(idx_t))

            # —- Neumann bottom: V[0, ix] = V[1, ix] —-
            idx_bot = idx[is_bottom]
            row.append(idx_bot); col.append(idx_bot);         data.append( np.ones_like(idx_bot))
            row.append(idx_bot); col.append(idx_bot+self.nx); data.append(-np.ones_like(idx_bot))

        # —- Standard 5-point interior stencil —-
        idx_in = idx[is_interior]
        row.append(idx_in); col.append(idx_in);          data.append(np.full_like(idx_in, -4.0))
        row.append(idx_in); col.append(idx_in-1);        data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in+1);        data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in-self.nx);  data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in+self.nx);  data.append(np.ones_like(idx_in))

        if periodic_y:
            # —- Periodic bottom row (iy=0): 5-pt with south neighbour = iy=ny-1 —-
            idx_pb   = idx[is_per_bot]
            col_south = (self.ny - 1) * self.nx + (idx_pb % self.nx)
            row.append(idx_pb); col.append(idx_pb);         data.append(np.full_like(idx_pb, -4.0))
            row.append(idx_pb); col.append(idx_pb - 1);     data.append(np.ones_like(idx_pb))  # west
            row.append(idx_pb); col.append(idx_pb + 1);     data.append(np.ones_like(idx_pb))  # east
            row.append(idx_pb); col.append(idx_pb + self.nx); data.append(np.ones_like(idx_pb)) # north (iy=1)
            row.append(idx_pb); col.append(col_south);      data.append(np.ones_like(idx_pb))  # periodic south

            # —- Periodic top row (iy=ny-1): 5-pt with north neighbour = iy=0 —-
            idx_pt   = idx[is_per_top]
            col_north = idx_pt % self.nx  # iy=0
            row.append(idx_pt); col.append(idx_pt);         data.append(np.full_like(idx_pt, -4.0))
            row.append(idx_pt); col.append(idx_pt - 1);     data.append(np.ones_like(idx_pt))  # west
            row.append(idx_pt); col.append(idx_pt + 1);     data.append(np.ones_like(idx_pt))  # east
            row.append(idx_pt); col.append(idx_pt - self.nx); data.append(np.ones_like(idx_pt)) # south (iy=ny-2)
            row.append(idx_pt); col.append(col_north);      data.append(np.ones_like(idx_pt))  # periodic north

        row  = np.concatenate(row)
        col  = np.concatenate(col)
        data = np.concatenate(data)

        A = sp.coo_matrix((data,(row,col)), shape=(N,N)).tocsc()
        self.laplacian_lu = factorized(A)

        self.laplacian_lu_gpu    = None
        self.is_bound_mask_gpu   = None
        self.is_interior_mask_gpu= None
        if _GPU_POISSON:
            try:
                A_csc = A.astype(np.float64)
                A_gpu = cp_sp.csc_matrix(
                    (cp.asarray(A_csc.data),
                     cp.asarray(A_csc.indices),
                     cp.asarray(A_csc.indptr)),
                    shape=A_csc.shape)
                self.laplacian_lu_gpu     = cp_splu(A_gpu)
                self.is_bound_mask_gpu    = cp.asarray(self.is_bound_mask)
                self.is_interior_mask_gpu = cp.asarray(self.is_interior_mask)
            except Exception as exc:
                print(f"[Poisson] GPU factorization failed, using CPU: {exc}")
                self.laplacian_lu_gpu = None

    # —————————————————————————————————
    def build_domain(self, params, preserve_state=False):
        grids   = params.get("grids", [])
        if grids:
            total_grid_thickness = sum(g['t'] + g['gap'] for g in grids)
        else:
            total_grid_thickness = 0.0
        
        pitch   = params.get("pitch_mm", params.get("discharge_chamber", {}).get("pitch_mm", 0.0) if isinstance(params.get("discharge_chamber"), dict) else 0.0)
        Te_up   = params.get('Te_up', 3.0)

        n0_n      = params.get("n0_plasma", 1e17)
        entire_bulk_plasma = params.get("entire_bulk_plasma", False)

        # Debye length is always needed for grid-spacing validation
        debye_length = np.sqrt(self.eps0 * Te_up / (self.q * n0_n * 0.61))
        debye_mm = debye_length * 1e3   # Debye length in mm

        # Physics-based recommended upstream gap depending on simulation mode
        if entire_bulk_plasma:
            # Bulk plasma mode: Debye-length-based sheath-formation criterion
            if n0_n <= 1e17:
                n_debye = 80
            elif n0_n <= 4e17:
                n_debye = 40
            else:
                n_debye = 30
            self.recommended_upstream_gap_mm = n_debye * debye_mm
        else:
            # Presheath mode (default): 0.75 × screen radius
            screen_r = grids[0]['r'] if grids else 0.80
            self.recommended_upstream_gap_mm = 0.75 * screen_r

        # Use user value if non-zero, otherwise fall back to mode-appropriate default
        user_gap = float(params.get('upstream_gap_mm', 0.0))
        upstream_gap = user_gap if user_gap > 0.0 else self.recommended_upstream_gap_mm
        upstream_gap = max(upstream_gap, self.dx * 2)  # minimum: 2 cells from boundary
        self.upstream_gap_mm = upstream_gap

        grid_Lx = (upstream_gap + total_grid_thickness) if grids else 3.0
        user_Lx = params.get('Lx', None)
        if user_Lx is not None and float(user_Lx) > 0:
            min_Lx = upstream_gap + total_grid_thickness if grids else 1.0
            self.Lx = max(float(user_Lx), min_Lx)
        else:
            self.Lx = grid_Lx

        if self.dx * 1e-3 > debye_length or self.dy * 1e-3 > debye_length:
            raise ValueError(
                f"Grid spacing too large for Debye resolution: "
                f"dx={self.dx:.6f} mm, dy={self.dy:.6f} mm, "
                f"lambda_D={debye_length*1e3:.6f} mm. "
                f"Choose dx, dy <= lambda_D."
            )


        plasma_freq = np.sqrt(n0_n * self.q**2 / (self.m_ion * self.eps0))
        elet_freq = np.sqrt(n0_n * self.q**2 / (self.m_e * self.eps0))
        dt = 2*3.14159 / plasma_freq
        dt_e = 2*3.14159 / elet_freq
        if dt < self.dt:
            raise ValueError(
                f"Too low time step: "
                f"dt used ={self.dt:.8e} s, dt minimum ={dt:.6e} s, "
            )
        Ti = params.get('Ti', 0.1)
        v_bohm = np.sqrt(self.q_ion * Te_up / self.m_ion)
        v_spread = np.sqrt(self.q_ion * Ti / self.m_ion)
        vmax = v_bohm + 4*v_spread

        if self.dx * 1e-3 / self.dt < vmax or self.dy * 1e-3 / self.dt < vmax:
            raise ValueError(
                f"Grid spacing and time step too large for velocity resolution: "
                f"dx/dt={self.dx*1e-3/self.dt:.2e} m/s, dy/dt={self.dy*1e-3/self.dt:.2e} m/s, "
                f"vmax={vmax:.2e} m/s. "
                f"Choose dx, dy and dt such that dx/dt >= vmax and dy/dt >= vmax."
            )
        # ————————————————————————————————
        # GEOMETRY MODE: selects domain height and hole layout
        # 'half_hole' — half-pitch symmetry plane at y=0 (default)
        # 'one_hole'  — full single-aperture domain
        # 'two_holes' — full dual-aperture domain (one pitch)
        # ————————————————————————————————
        geometry = params.get('geometry', 'half_hole')
        self.geometry = geometry  # stored so step() can reuse without re-reading params
        # y=0 and y=Ly are physical symmetry planes only in half_hole.
        # For one_hole / two_holes use periodic BCs so neither edge breaks symmetry.
        self.periodic_y = geometry in ('one_hole', 'two_holes')

        if grids:
            screen_r = grids[0]['r']
            if geometry == 'two_holes':
                grid_Ly = screen_r + 2.0 * screen_r + pitch      # full dual-hole pitch
            elif geometry == 'one_hole':
                grid_Ly = 3.0 * screen_r                         # full single-hole domain
            else:  # 'half_hole' (default)
                grid_Ly = 0.5 * screen_r + 0.30 * pitch         # half-pitch symmetry
        else:
            grid_Ly = 3.0

        user_Ly = params.get('Ly', None)
        if user_Ly is not None and float(user_Ly) > 0:
            min_Ly = screen_r if grids else 0.5
            self.Ly = max(float(user_Ly), min_Ly)
        else:
            self.Ly = grid_Ly

        self.nx = int(self.Lx / self.dx) + 1
        self.ny = int(self.Ly / self.dy) + 1
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)

        self.xpts = np.linspace(0, self.Lx, self.nx)
        self.ypts = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.xpts, self.ypts)

        if not preserve_state:
            self.Tmap       = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
            self.T_map      = self.Tmap
            self.T_map_new  = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
            self.Tmapnew    = self.T_map_new
            self.reset_arrays()
            self.iteration  = 0
            self.injected_ions = 0.0
            self.injected_ions_step = 0.0
            self.transmitted_ions = 0.0
            self.transmitted_ions_step = 0.0
            self.entered_optics = 0.0
            self.entered_optics_step = 0.0
            self._domain_built = True

        inj_time = params.get("inj_time", 0.0)

        if inj_time > 0.0:
            self.injection_stop_time = inj_time
            self.injection_enabled   = True
        else:
            self.injection_stop_time = None
            self.injection_enabled   = True

        n0         = params.get('n0_plasma', 1e17)
        target_ppc = 80.0
        cell_vol   = (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3
        self.macro_weight = max(n0 * cell_vol / target_ppc, 1e3)
        self.mask_grids = []
        self.T_grids    = []
        self.isBound.fill(False)
        self.V_fixed.fill(0.0)

        if not hasattr(self, 'grid_deflections') or len(self.grid_deflections) != len(grids):
            self.grid_deflections = [0.0] * len(grids)

        current_x = self.upstream_gap_mm  # left face of screen grid position [mm]

        # Hole centres depend on geometry mode
        if grids:
            screen_r = grids[0]['r']
            if geometry == 'two_holes':
                y_c1 = 1.5 * screen_r
                hole_centers = [y_c1, y_c1 + pitch]
            elif geometry == 'one_hole':
                hole_centers = [1.5 * screen_r]
            else:  # half_hole
                hole_centers = [0.0]
        else:
            hole_centers = []

        self.hole_centers = hole_centers  # stored for diagnostics

        self.grid_x_starts = []
        self.grid_x_ends   = []
        for i, grid in enumerate(grids):
            gstart = current_x
            gend   = gstart + grid["t"]
            self.grid_x_starts.append(gstart)
            self.grid_x_ends.append(gend)
            delta  = self.grid_deflections[i]

            if abs(delta) > 1e-6:
                if geometry == 'two_holes':
                    y_web = 1.5 * screen_r + 0.5 * pitch
                    eta = np.clip(1.0 - np.abs(self.Y - y_web) / max(pitch * 0.5, 1e-6), 0.0, 1.0)
                    dxbow = delta * eta**2
                elif geometry == 'one_hole':
                    y_mid = 1.5 * screen_r
                    eta = np.clip(1.0 - np.abs(self.Y - y_mid) / max(self.Ly * 0.5, 1e-6), 0.0, 1.0)
                    dxbow = delta * eta**2
                else:  # half_hole
                    Lcant = self.Ly - grid["r"]
                    if Lcant > 0:
                        eta   = np.clip((self.Ly - self.Y) / Lcant, 0.0, 1.0)
                        dxbow = delta * eta**2
                    else:
                        dxbow = 0.0
            else:
                dxbow = 0.0

            ingrid_x = (self.X >= gstart + dxbow) & (self.X <= gend + dxbow)
            mask     = ingrid_x.copy()

            for yc in hole_centers:
                local_r = grid["r"] - np.maximum(0.0, self.X - gstart - dxbow) * \
                          np.tan(np.radians(grid["cham"]))
                local_r = np.maximum(local_r, 0.0)
                hole    = ingrid_x & (np.abs(self.Y - yc) <= local_r)
                mask   &= ~hole

            self.isBound[mask] = True
            self.V_fixed[mask] = grid["V"]
            self.mask_grids.append(mask)
            if np.any(mask):
                self.T_grids.append(float(np.mean(self.Tmap[mask])))
            else:
                self.T_grids.append(300.0)

            current_x = gend + grid["gap"]

        self.Vdc = np.copy(self.V_fixed)

        # Set plasma boundary at high reference voltage (opposite of negated grids)
        # This creates accelerating voltage for positive ions from plasma toward grids
        v_plasma_bound = (grids[0]["V"] + params.get("V_plasma_offset", 20.0)
                          if grids else 1000.0 + params.get("V_plasma_offset", 20.0))
        self.V_fixed[:, 0] = v_plasma_bound
        self.isBound[:, 0] = True

        if not preserve_state:
            self.Tmap[self.isBound] = 300.0
        self.build_sparse_matrix()
        self.recalc_poisson(iterations=30 if not preserve_state else 10, params=params)

    # —————————————————————————————————
    def recalc_poisson(self, iterations=5, params=None):
        if self.laplacian_lu is None:
            return

        dx_m2 = (self.dx * 1e-3)**2
        coeff  = dx_m2 / self.eps0

        if params is None:
            params = {}
        grids    = params.get('grids', [{'V': 1000}])
        v_offset = params.get('V_plasma_offset', 20.0)
        V_plasma = grids[0]['V'] + v_offset
        Te_up    = params.get('Te_up', 3.0)
        n0       = params.get('n0_plasma', 1e17)
        omega    = 0.2

        if _GPU_POISSON and self.laplacian_lu_gpu is not None:
            self._recalc_poisson_gpu(iterations, coeff, V_plasma, Te_up, n0, omega)
        else:
            self._recalc_poisson_cpu(iterations, coeff, V_plasma, Te_up, n0, omega)

        self.Ey, self.Ex = np.gradient(-self.V, self.dy*1e-3, self.dx*1e-3)

        # Fix Ey at the y-domain edges: np.gradient uses one-sided differences there,
        # which assumes zero-gradient (Neumann). In periodic mode we need central
        # differences with wrap-around.
        if getattr(self, 'periodic_y', False):
            two_dy_m = 2.0 * self.dy * 1e-3
            # iy=0: central = (V[ny-1] - V[1]) / (2*dy)  ->  Ey = (V[ny-1] - V[1]) / (2*dy)
            self.Ey[0, :]       = (self.V[self.ny-1, :] - self.V[1, :])       / two_dy_m
            # iy=ny-1: central = (V[ny-2] - V[0]) / (2*dy)
            self.Ey[self.ny-1, :] = (self.V[self.ny-2, :] - self.V[0, :]) / two_dy_m

        self.Ey = self.Ey.astype(_NP_FP)
        self.Ex = self.Ex.astype(_NP_FP)

    def _recalc_poisson_cpu(self, iterations, coeff, V_plasma, Te_up, n0, omega):
        b = np.zeros(self.nx * self.ny, dtype=np.float64)
        V_fixed_flat = self.V_fixed.flatten()
        rhs_rho_mask = getattr(self, 'is_rhs_rho_mask', self.is_interior_mask)

        for _ in range(iterations):
            rho_e     = -self.q * n0 * np.exp((np.minimum(self.V, V_plasma) - V_plasma) / Te_up)
            rho_total = self.rho + rho_e
            rho_flat  = rho_total.flatten()

            b.fill(0.0)
            b[self.is_bound_mask] = V_fixed_flat[self.is_bound_mask]
            b[rhs_rho_mask]       = -coeff * rho_flat[rhs_rho_mask]

            V_new_flat = self.laplacian_lu(b)
            V_new      = V_new_flat.reshape((self.ny, self.nx))
            self.V     = ((1-omega)*self.V + omega*V_new).astype(np.float64)

    def _recalc_poisson_gpu(self, iterations, coeff, V_plasma, Te_up, n0, omega):
        V_gpu            = cp.asarray(self.V, dtype=cp.float64)
        rho_gpu          = cp.asarray(self.rho).astype(cp.float64)
        V_fixed_flat_gpu = cp.asarray(self.V_fixed.ravel(), dtype=cp.float64)
        b_gpu            = cp.zeros(self.nx * self.ny, dtype=cp.float64)
        q                = self.q

        rhs_rho_mask = getattr(self, 'is_rhs_rho_mask', self.is_interior_mask)
        rhs_rho_mask_gpu = cp.asarray(rhs_rho_mask)
        bound_mask_gpu   = cp.asarray(self.is_bound_mask)

        for _ in range(iterations):
            rho_e_gpu    = -q * n0 * cp.exp((cp.minimum(V_gpu, V_plasma) - V_plasma) / Te_up)
            rho_flat_gpu = (rho_gpu + rho_e_gpu).ravel()
            b_gpu[:] = 0.0
            b_gpu[bound_mask_gpu]   = V_fixed_flat_gpu[bound_mask_gpu]
            b_gpu[rhs_rho_mask_gpu] = -coeff * rho_flat_gpu[rhs_rho_mask_gpu]
            V_new_flat = self.laplacian_lu_gpu.solve(b_gpu)
            V_new_gpu  = V_new_flat.reshape((self.ny, self.nx))
            V_gpu      = (1.0-omega)*V_gpu + omega*V_new_gpu

        self.V = cp.asnumpy(V_gpu)

    # —————————————————————————————————
    def compute_particle_substeps(self, x, y, vx, vy, vz, qm, dt, frac=0.25):
        """
        Dynamically calculates the number of sub-steps (n_sub) required for particle pushing.
        
        Sub-stepping ensures that fast-moving or strongly accelerated particles do not travel
        more than a fraction of a grid cell (frac * min(dx, dy)) in a single push. This prevents
        particles from skipping cells, avoids non-physical trajectory errors, ensures accurate
        electric field interpolation, and prevents tunneling through grid boundaries.
        """
        if len(x) == 0:
            return 1, 0.0, frac * min(self.dx, self.dy) * 1e-3

        ix0 = np.clip(np.floor(x / self.dx).astype(int), 0, self.nx - 2)
        iy0 = np.clip(np.floor(y / self.dy).astype(int), 0, self.ny - 2)
        fx = x / self.dx - ix0
        fy = y / self.dy - iy0
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        def interp(F):
            return (
                F[iy0, ix0] * (1-fx) * (1-fy) +
                F[iy0, ix1] * fx * (1-fy) +
                F[iy1, ix0] * (1-fx) * fy +
                F[iy1, ix1] * fx * fy
            )

        Ex_p = interp(self.Ex)
        Ey_p = interp(self.Ey)

        a_mag = np.sqrt((qm * Ex_p)**2 + (qm * Ey_p)**2)
        v_mag = np.sqrt(vx*vx + vy*vy + vz*vz)

        ds_pred = v_mag * dt + 0.5 * a_mag * dt**2
        ds_max = float(np.max(ds_pred))
        ds_lim = frac * min(self.dx, self.dy) * 1e-3

        n_sub = max(1, int(np.ceil(ds_max / max(ds_lim, 1e-30))))
        return n_sub, ds_max, ds_lim
    
    # —————————————————————————————————
    def step(self, params):
        if self.laplacian_lu is None:
            return False, np.nan, np.nan, self.T_grids, 0.0

        sim_mode = params.get('sim_mode', 'Both')
        self.iteration += 1
        t_current = self.iteration * self.dt
        grids = params.get('grids', [])

        self.transmitted_ions_step = 0.0
        self.transmitted3_step = 0.0
        self.entered_optics_step = 0.0
        self.injected_ions_step = 0.0
        self.lost_to_grid_step = 0.0

        # —- RF CO-EXTRACTION —-
        if params.get('rf_enable') and grids:
            rf_idx = params.get('rf_grid_idx', 0)
            if rf_idx < len(grids):
                f_hz = params.get('rf_freq', 13.56) * 1e6
                v_rf = params.get('rf_amp', 100.0) * np.sin(2.0 * np.pi * f_hz * t_current)
                self.V_fixed[self.mask_grids[rf_idx]] = self.Vdc[self.mask_grids[rf_idx]] + v_rf
                self.recalc_poisson(iterations=2, params=params)

        # ————————————————————————————————
        # A. INJECT PARTICLES
        # ————————————————————————————————
        if self.injection_enabled and (
            self.injection_stop_time is None or t_current <= self.injection_stop_time
        ):
            n0 = params.get('n0_plasma', 1e17)
            Te_up = params.get('Te_up', 3.0)
            Ti = params.get('Ti', 0.1)

            v_bohm = np.sqrt(self.q_ion * Te_up / self.m_ion)

            injection_area_scale = 0.005 # 0.005
            #injection_area_scale = 0.1
            injection_area = self.Ly * 1e-3 * injection_area_scale
            # Presheath mode: include Bohm factor 0.61 at the injection plane.
            # Entire Bulk Plasma mode: no 0.61 (full density assumed at boundary).
            entire_bulk_plasma = params.get('entire_bulk_plasma', False)
            bohm_factor = 1.0 if entire_bulk_plasma else 0.61
            I_ion = self.q_ion * bohm_factor * n0 * v_bohm * injection_area # 3.7-28 Goebbels
            charge_per_macro = self.q_ion * self.macro_weight

            num_inject_float = (I_ion * self.dt) / charge_per_macro
            # Use Poisson sampling: gives statistically correct particle count including
            # sub-particle fractional contributions, without the arbitrary +3 hack.
            num_inject = np.random.poisson(num_inject_float)

            if num_inject > 0:
                geometry = getattr(self, 'geometry', 'half_hole')
                if grids:
                    screen_r = grids[0]['r']
                    pitch_inj = params.get('pitch_mm', 0.0)
                    if geometry == 'two_holes':
                        # Inject uniformly in the aperture band around each of the two holes
                        y_c1 = 1.5 * screen_r
                        span_limits = np.array([
                            [max(self.dy, 0.5 * screen_r),
                             min(2.5 * screen_r, self.Ly - self.dy)],
                            [max(self.dy, 0.5 * screen_r + pitch_inj),
                             min(2.5 * screen_r + pitch_inj, self.Ly - self.dy)],
                        ], dtype=_NP_FP)
                        which_hole = np.random.randint(0, 2, size=num_inject)
                        y_lo  = span_limits[which_hole, 0]
                        y_hi  = span_limits[which_hole, 1]
                        new_y = (y_lo + (y_hi - y_lo) * np.random.rand(num_inject)).astype(_NP_FP)
                    elif geometry == 'one_hole':
                        # Inject in the aperture band around the single centred hole
                        ylow  = max(self.dy, 0.5 * screen_r)
                        yhigh = min(2.5 * screen_r, self.Ly - self.dy)
                        new_y = np.random.uniform(ylow, yhigh, num_inject).astype(_NP_FP)
                    else:  # half_hole
                        # Inject within the aperture band only (hole centred at y=0)
                        # The half-aperture spans from the axis up to the screen radius
                        ylow  = 0.0
                        yhigh = min(screen_r, self.Ly - self.dy)
                        new_y = np.random.uniform(ylow, yhigh, num_inject).astype(_NP_FP)
                else:
                    new_y = np.random.uniform(self.dy, self.Ly - self.dy, num_inject).astype(_NP_FP)

                # Inject at the upstream boundary wall (x ≈ 0)
                new_x = np.full(num_inject, self.dx * 1.5, dtype=_NP_FP)

                v_spread = np.sqrt(self.q_ion * Ti / self.m_ion)
                #new_vx = np.abs(v_bohm + np.random.randn(num_inject) * v_spread).astype(_NP_FP)
                new_vx = np.full(num_inject, v_bohm, dtype=np.float64) + np.random.randn(num_inject).astype(np.float64) * v_spread
                new_vy = (np.random.randn(num_inject) * v_spread).astype(_NP_FP)
                new_vz = (np.random.randn(num_inject) * v_spread).astype(_NP_FP)
                new_cex = np.zeros(num_inject, dtype=bool)

                self._add_ions(new_x, new_y, new_vx, new_vy, new_vz, new_cex)
                self.injected_ions_step += float(num_inject)
                self.injected_ions += float(num_inject)

            if params.get('rf_enable'):
                v_e_th_source = np.sqrt(2.0 * self.q * Te_up / self.m_e)
                I_e = self.q * 0.25 * n0 * v_e_th_source * injection_area
                num_e_float = (I_e * self.dt) / charge_per_macro
                num_inj_e = int(num_e_float)
                if np.random.rand() < (num_e_float - num_inj_e):
                    num_inj_e += 1

                if num_inj_e > 0:
                    y_margin = max(self.dy, 1e-9)
                    y_low = y_margin
                    y_high = max(y_low, self.Ly - y_margin)
                    new_ex = np.full(num_inj_e, 0.1, dtype=_NP_FP)
                    new_ey = np.random.uniform(y_low, y_high, num_inj_e).astype(_NP_FP)
                    new_evx = (np.abs(np.random.randn(num_inj_e)) * v_e_th_source + v_bohm).astype(_NP_FP)
                    new_evy = (np.random.randn(num_inj_e) * v_e_th_source).astype(_NP_FP)
                    new_evz = (np.random.randn(num_inj_e) * v_e_th_source).astype(_NP_FP)
                    self._add_electrons(new_ex, new_ey, new_evx, new_evy, new_evz)

        # —- NEUTRALIZER —-
        num_e_neut = int(params.get('neut_rate', 30))
        Te_eV = params.get('Te', 5.0)
        # Default neutralizer position: 9/10 of the downstream plume region
        # (between last grid exit and domain end), unless the user has overridden it.
        if hasattr(self, 'grid_x_ends') and self.grid_x_ends:
            _x_exit = self.grid_x_ends[-1]
            _neut_default = _x_exit + 0.9 * (self.Lx - _x_exit)
        else:
            _neut_default = self.Lx - 0.5
        neut_x_param = params.get('neut_x', _neut_default)
        neut_r_param = params.get('neut_r', self.Ly)
        if neut_x_param > self.Lx or neut_x_param < 0.0:
            if not getattr(self, '_warned_neut_oob', False):
                print(f"[Warning] Neutralizer position x={neut_x_param:.3f} mm is outside x-domain [0, {self.Lx:.3f} mm]. Setting to x = Lx ({self.Lx:.3f} mm).")
                self._warned_neut_oob = True
            neut_x = float(self.Lx)
        else:
            self._warned_neut_oob = False
            neut_x = float(neut_x_param)
        neut_r = float(np.clip(neut_r_param, self.dy, self.Ly))
        if num_e_neut > 0:
            new_ey = np.random.uniform(0.0, neut_r, num_e_neut).astype(_NP_FP)
            new_ex = np.full(num_e_neut, neut_x, dtype=_NP_FP)
            v_e_th = np.sqrt(2.0 * self.q * Te_eV / self.m_e)
            new_evx = (np.random.randn(num_e_neut) * v_e_th).astype(_NP_FP)
            new_evy = (np.random.randn(num_e_neut) * v_e_th).astype(_NP_FP)
            new_evz = (np.random.randn(num_e_neut) * v_e_th).astype(_NP_FP)
            self._add_electrons(new_ex, new_ey, new_evx, new_evy, new_evz)

        # ————————————————————————————————
        # B. POISSON SOLVER
        # ————————————————————————————————
        self.rho.fill(0.0)
        cell_vol = (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3
        charge_per_particle = self.q * self.macro_weight

        if self.num_p > 0:
            if USE_TAICHI:
                accumulate_rho_taichi(
                    _ti_arr(self.p_x[:self.num_p]),
                    _ti_arr(self.p_y[:self.num_p]),
                    self.rho,
                    self.num_p,
                    np.float32(self.dx),
                    np.float32(self.dy),
                    self.nx,
                    self.ny,
                    np.float32(charge_per_particle / cell_vol)
                )
            else:
                accumulate_rho_cpu(
                    self.p_x[:self.num_p],
                    self.p_y[:self.num_p],
                    self.rho,
                    self.num_p,
                    self.dx,
                    self.dy,
                    self.nx,
                    self.ny,
                    charge_per_particle / cell_vol
                )

        if self.num_e > 0:
            if USE_TAICHI:
                accumulate_rho_taichi(
                    _ti_arr(self.e_x[:self.num_e]),
                    _ti_arr(self.e_y[:self.num_e]),
                    self.rho,
                    self.num_e,
                    np.float32(self.dx),
                    np.float32(self.dy),
                    self.nx,
                    self.ny,
                    np.float32(-charge_per_particle / cell_vol)
                )
            else:
                accumulate_rho_cpu(
                    self.e_x[:self.num_e],
                    self.e_y[:self.num_e],
                    self.rho,
                    self.num_e,
                    self.dx,
                    self.dy,
                    self.nx,
                    self.ny,
                    -charge_per_particle / cell_vol
                )

        if self.iteration % 2 == 0:
            self.recalc_poisson(iterations=5, params=params)

        # ————————————————————————————————
        # C. PUSH PARTICLES
        # ————————————————————————————————
        num_p_step = int(self.num_p)

        p_x_old = self.p_x[:num_p_step].copy()
        p_y_old = self.p_y[:num_p_step].copy()

        _dx = np.float32(self.dx)
        _dy = np.float32(self.dy)
        _dt = np.float32(self.dt)
        _qm_ion = np.float32(self.q_ion / self.m_ion)
        _qm_e = np.float32(-self.q / self.m_e)

        # Runtime adaptive substepping based on current fields and particle states
        if num_p_step > 0:
            n_sub_ion, ds_ion, ds_lim_ion = self.compute_particle_substeps(
                self.p_x[:num_p_step], self.p_y[:num_p_step],
                self.p_vx[:num_p_step], self.p_vy[:num_p_step], self.p_vz[:num_p_step],
                self.q_ion / self.m_ion, self.dt
            )
        else:
            n_sub_ion = 1

        if n_sub_ion > 1:
            print(
                f"[Warning] Ion displacement criterion violated at iter {self.iteration}: "
                f"predicted ds_max = {ds_ion:.3e} m, allowed = {ds_lim_ion:.3e} m, "
                f"required substeps = {n_sub_ion}"
            )

        num_e_step = int(self.num_e)
        if num_e_step > 0:
            n_sub_e, ds_e, ds_lim_e = self.compute_particle_substeps(
                self.e_x[:num_e_step], self.e_y[:num_e_step],
                self.e_vx[:num_e_step], self.e_vy[:num_e_step], self.e_vz[:num_e_step],
                -self.q / self.m_e, self.dt
            )
        else:
            n_sub_e = 1

        if n_sub_e > 1:
            print(
                f"[Warning] Electron displacement criterion violated at iter {self.iteration}: "
                f"predicted ds_max = {ds_e:.3e} m, allowed = {ds_lim_e:.3e} m, "
                f"required substeps = {n_sub_e}"
            )
            
        if num_p_step > 0:
            dt_ion = np.float32(self.dt / n_sub_ion)

            if USE_TAICHI:
                px_ti = _ti_arr(self.p_x[:num_p_step])
                py_ti = _ti_arr(self.p_y[:num_p_step])
                pvx_ti = _ti_arr(self.p_vx[:num_p_step])
                pvy_ti = _ti_arr(self.p_vy[:num_p_step])
                pvz_ti = _ti_arr(self.p_vz[:num_p_step])

                for _ in range(n_sub_ion):
                    push_particles_boris_taichi(
                        px_ti, py_ti, pvx_ti, pvy_ti, pvz_ti,
                        self.Ex, self.Ey, self.Bx, self.By, self.Bz,
                        num_p_step, _dx, _dy, self.nx, self.ny, dt_ion, _qm_ion
                    )

                self.p_x[:num_p_step] = px_ti
                self.p_y[:num_p_step] = py_ti
                self.p_vx[:num_p_step] = pvx_ti
                self.p_vy[:num_p_step] = pvy_ti
                self.p_vz[:num_p_step] = pvz_ti

            else:
                for _ in range(n_sub_ion):
                    push_particles_boris_cpu(
                        self.p_x[:num_p_step],
                        self.p_y[:num_p_step],
                        self.p_vx[:num_p_step],
                        self.p_vy[:num_p_step],
                        self.p_vz[:num_p_step],
                        self.Ex, self.Ey, self.Bx, self.By, self.Bz,
                        num_p_step, self.dx, self.dy, self.nx, self.ny,
                        self.dt / n_sub_ion,
                        self.q_ion / self.m_ion
                    )

        if num_e_step > 0:
            dt_e = np.float32(self.dt / n_sub_e)

            if USE_TAICHI:
                ex_ti = _ti_arr(self.e_x[:num_e_step])
                ey_ti = _ti_arr(self.e_y[:num_e_step])
                evx_ti = _ti_arr(self.e_vx[:num_e_step])
                evy_ti = _ti_arr(self.e_vy[:num_e_step])
                evz_ti = _ti_arr(self.e_vz[:num_e_step])

                for _ in range(n_sub_e):
                    push_particles_boris_taichi(
                        ex_ti, ey_ti, evx_ti, evy_ti, evz_ti,
                        self.Ex, self.Ey, self.Bx, self.By, self.Bz,
                        num_e_step, _dx, _dy, self.nx, self.ny, dt_e, _qm_e
                    )

                self.e_x[:num_e_step] = ex_ti
                self.e_y[:num_e_step] = ey_ti
                self.e_vx[:num_e_step] = evx_ti
                self.e_vy[:num_e_step] = evy_ti
                self.e_vz[:num_e_step] = evz_ti

            else:
                for _ in range(n_sub_e):
                    push_particles_boris_cpu(
                        self.e_x[:num_e_step],
                        self.e_y[:num_e_step],
                        self.e_vx[:num_e_step],
                        self.e_vy[:num_e_step],
                        self.e_vz[:num_e_step],
                        self.Ex, self.Ey, self.Bx, self.By, self.Bz,
                        num_e_step, self.dx, self.dy, self.nx, self.ny,
                        self.dt / n_sub_e,
                        -self.q / self.m_e
                    )

        # ————————————————————————————————
        # Periodic y wrap-around (one_hole / two_holes only)
        # Replaces boundary-kill for y-edges; must happen BEFORE hit detection
        # ————————————————————————————————
        if getattr(self, 'periodic_y', False):
            if num_p_step > 0:
                self.p_y[:num_p_step] = np.mod(self.p_y[:num_p_step], self.Ly)
            if self.num_e > 0:
                self.e_y[:self.num_e] = np.mod(self.e_y[:self.num_e], self.Ly)

        p_x = self.p_x[:num_p_step].copy()
        p_y = self.p_y[:num_p_step].copy()
        p_vx = self.p_vx[:num_p_step].copy()
        p_vy = self.p_vy[:num_p_step].copy()
        p_vz = self.p_vz[:num_p_step].copy()
        p_cex = self.p_isCEX[:num_p_step].copy()

        if grids and hasattr(self, "grid_x_starts") and len(self.grid_x_starts) == len(grids):
            x_start_first_grid = self.grid_x_starts[0]
            x_entry_first_grid = self.grid_x_starts[0]
            x_exit_last = self.grid_x_ends[-1]
            x_plume_boundary = self.grid_x_ends[-1] + grids[-1]["gap"]
        else:
            x_start_first_grid = 0.5
            x_entry_first_grid = 0.5
            x_exit_last = 3.0
            x_plume_boundary = 3.0

        # ————————————————————————————————
        # D. ION DIAGNOSTICS / HITS / EROSION / SEE
        # ————————————————————————————————
        entered_first_grid_mask = (
            (p_x_old < x_entry_first_grid) &
            (p_x >= x_entry_first_grid) &
            (p_vx > 0.0)
        )
        n_entered = int(np.count_nonzero(entered_first_grid_mask))
        self.entered_optics_step = float(n_entered)
        self.entered_optics += float(n_entered)
        self.prev_entered = self.entered_optics_step

        ix = np.clip(np.round(p_x / self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip(np.round(p_y / self.dy).astype(int), 0, self.ny - 1)

        hit_grid_final = self.isBound[iy, ix]
        hit_grid_path, path_ix, path_iy = self._segment_hits_grid(p_x_old, p_y_old, p_x, p_y, samples=8)
        hit_grid = hit_grid_final | hit_grid_path
        
        # Use first impact coordinates for any interaction (thermal, sputtering)
        # If it hit the boundary directly on the final step without triggering path (rare), keep ix/iy
        impact_ix = np.where(hit_grid_path, path_ix, ix)
        impact_iy = np.where(hit_grid_path, path_iy, iy)

        # ————————————————————————————————
        # 1. Out of Bounds
        # ————————————————————————————————
        if getattr(self, 'periodic_y', False):
            # In periodic mode, p_y is already wrapped to [0, Ly], so it can never be OOB.
            # Particles are tracked across the full domain to avoid artificial charge
            # discontinuities; diagnostic plane remains at x_exit_last.
            out_of_bounds = (
                (p_x < 0.0) |
                (p_x > self.Lx) |
                np.isnan(p_x)
            )
        else:
            out_of_bounds = (
                (p_x < 0.0) |
                (p_x > self.Lx) |
                (p_y < 0.0) |
                (p_y > self.Ly) |
                np.isnan(p_x)
            )

        remeshed = False

        valid_thermal_hit = hit_grid & (p_x > 0.25)
        if sim_mode in ("Thermal", "Both") and np.any(valid_thermal_hit):
            v_mag_sq = (
                p_vx[valid_thermal_hit] ** 2 +
                p_vy[valid_thermal_hit] ** 2 +
                p_vz[valid_thermal_hit] ** 2
            )
            E_joules = 0.5 * self.m_ion * v_mag_sq * self.macro_weight
            dT_heat = (E_joules / self.C_cell) * self.thermal_accel

            # —- Diagnostic: Check hits on Grid 2 (Accelerator Grid) —-
            if len(self.mask_grids) > 1:
                hit_iy_sub = impact_iy[valid_thermal_hit]
                hit_ix_sub = impact_ix[valid_thermal_hit]
                hit_g2 = self.mask_grids[1][hit_iy_sub, hit_ix_sub]
                if np.any(hit_g2):
                    n_g2 = int(np.count_nonzero(hit_g2))
                    max_dT = float(np.max(dT_heat[hit_g2]))
                    x_hits = hit_ix_sub[hit_g2] * self.dx
                    y_hits = hit_iy_sub[hit_g2] * self.dy
                    e_ev_g2 = (0.5 * self.m_ion * v_mag_sq[hit_g2]) / self.q
                    # print(
                    #    f"[Iter {self.iteration}] Grid 2 hit by {n_g2} macroparticle(s)! "
                    #    f"x: [{x_hits.min():.3f}, {x_hits.max():.3f}] mm, "
                    #    f"y: [{y_hits.min():.3f}, {y_hits.max():.3f}] mm, "
                    #    f"mean energy: {float(np.mean(e_ev_g2)):.1f} eV, "
                    #    f"max dT: {max_dT:.1f} K"
                    # )

            np.add.at(self.Tmap, (impact_iy[valid_thermal_hit], impact_ix[valid_thermal_hit]), dT_heat)

        valid_see_hit = hit_grid & (p_x > 0.25)
        if np.any(valid_see_hit):
            v_mag_sq = (
                p_vx[valid_see_hit] ** 2 +
                p_vy[valid_see_hit] ** 2 +
                p_vz[valid_see_hit] ** 2
            )
            E_eV = (0.5 * self.m_ion * v_mag_sq) / self.q
            see_user = self.lookup_user_cs('SEE', E_eV)
            gamma = np.clip(see_user if see_user is not None else 0.05 + 1e-4 * E_eV, 0.0, 1.0)
            spawn_mask = np.random.rand(len(gamma)) < gamma

            if np.any(spawn_mask):
                num_see = int(np.sum(spawn_mask))
                see_x = (
                    p_x[valid_see_hit][spawn_mask] -
                    p_vx[valid_see_hit][spawn_mask] * self.dt * 1000.0 / 1.5
                ).astype(_NP_FP)
                see_y = (
                    p_y[valid_see_hit][spawn_mask] -
                    p_vy[valid_see_hit][spawn_mask] * self.dt * 1000.0 / 1.5
                ).astype(_NP_FP)
                T_see = 2.0
                v_see_th = np.sqrt(2.0 * self.q * T_see / self.m_e)
                see_vx = (np.random.randn(num_see) * v_see_th).astype(_NP_FP)
                see_vy = (np.random.randn(num_see) * v_see_th).astype(_NP_FP)
                see_vz = (np.random.randn(num_see) * v_see_th).astype(_NP_FP)
                self._add_electrons(see_x, see_y, see_vx, see_vy, see_vz)

        is_erosion_hit = hit_grid & (p_x > 0.25)
        if sim_mode in ("Erosion", "Both") and np.any(is_erosion_hit):
            E_eV = (
                0.5 * self.m_ion * (
                    p_vx[is_erosion_hit] ** 2 +
                    p_vy[is_erosion_hit] ** 2 +
                    p_vz[is_erosion_hit] ** 2
                )
            ) / self.q

            valid_among_hits = E_eV > self.sputter_E_th

            if np.any(valid_among_hits):
                E_valid = E_eV[valid_among_hits]
                yield_rate = self.sputter_Y_coeff * (E_valid - self.sputter_E_th)
                damage = yield_rate * self.macro_weight
                iy_hit = impact_iy[is_erosion_hit][valid_among_hits]
                ix_hit = impact_ix[is_erosion_hit][valid_among_hits]
                np.add.at(
                    self.damage_map,
                    (iy_hit, ix_hit),
                    damage
                )

            broken_cells = (self.damage_map > params.get('Thresh', 1e5)) & self.isBound
            if np.any(broken_cells):
                self.eroded_depth[broken_cells] += self.dy
                self.isBound[broken_cells] = False
                self.damage_map[broken_cells] = 0.0
                self.build_sparse_matrix()
                remeshed = True

        exited_mask = (p_x > x_exit_last) & (p_vx > 0.0)
        crossed_mask = (p_x_old <= x_exit_last) & (p_x > x_exit_last) & (p_vx > 0.0)

        if np.count_nonzero(crossed_mask) > 0:
            vx_exit = p_vx[crossed_mask].copy()
            vy_exit = p_vy[crossed_mask].copy()
            vz_exit = p_vz[crossed_mask].copy()
            cex_exit = p_cex[crossed_mask].copy()
            v_exit = np.sqrt(vx_exit ** 2 + vy_exit ** 2 + vz_exit ** 2)
            E_exit_eV = 0.5 * self.m_ion * v_exit ** 2 / self.q

            self.exit_vx_mean = float(np.mean(vx_exit))
            self.exit_v_mean = float(np.mean(v_exit))
            self.exit_vx_std = float(np.std(vx_exit))
            self.exit_v_std = float(np.std(v_exit))
            self.exit_energy_mean_eV = float(np.mean(E_exit_eV))
            self.exit_count_step = int(np.count_nonzero(crossed_mask))

            prim_exit = ~cex_exit
            self.exit_v_mean_primary = float(np.mean(v_exit[prim_exit])) if np.any(prim_exit) else np.nan
            self.exit_v_mean_cex = float(np.mean(v_exit[cex_exit])) if np.any(cex_exit) else np.nan
        else:
            self.exit_vx_mean = np.nan
            self.exit_v_mean = np.nan
            self.exit_vx_std = np.nan
            self.exit_v_std = np.nan
            self.exit_energy_mean_eV = np.nan
            self.exit_count_step = 0
            self.exit_v_mean_primary = np.nan
            self.exit_v_mean_cex = np.nan

        current_div = (
            np.percentile(
                np.abs(np.arctan2(p_vy[crossed_mask], p_vx[crossed_mask])) * 180.0 / np.pi,
                95
            )
            if np.count_nonzero(crossed_mask) > 5 else np.nan
        )

        if np.any(exited_mask):
            n_transmitted = int(np.count_nonzero(crossed_mask))
            self.transmitted_ions_step = float(n_transmitted)
            self.transmitted_ions += float(n_transmitted)
            self.transmitted3_step = float(n_transmitted)

        grid_hit_mask = hit_grid & ~out_of_bounds
        self.lost_to_grid_step = float(np.count_nonzero(grid_hit_mask))

        dead_mask = hit_grid | out_of_bounds
        alive_mask = ~dead_mask
        n_alive = int(np.sum(alive_mask))

        if n_alive < num_p_step:
            self.p_x[:n_alive] = p_x[alive_mask]
            self.p_y[:n_alive] = p_y[alive_mask]
            self.p_vx[:n_alive] = p_vx[alive_mask]
            self.p_vy[:n_alive] = p_vy[alive_mask]
            self.p_vz[:n_alive] = p_vz[alive_mask]
            self.p_isCEX[:n_alive] = p_cex[alive_mask]
            self.num_p = n_alive
        else:
            self.num_p = num_p_step

        p_x = self.p_x[:self.num_p]
        p_y = self.p_y[:self.num_p]
        p_vx = self.p_vx[:self.num_p]
        p_vy = self.p_vy[:self.num_p]
        p_vz = self.p_vz[:self.num_p]
        p_cex = self.p_isCEX[:self.num_p]

        # Particles are now tracked across the full domain (up to self.Lx).
        # The secondary plume-boundary purge has been removed: ions are only
        # removed when they reach the actual domain edge (already handled by
        # out_of_bounds above), eliminating the artificial charge discontinuity
        # that occurred at x_plume_boundary.

        # ————————————————————————————————
        # E. PURGE DEAD ELECTRONS
        # ————————————————————————————————
        if self.num_e > 0:
            e_x = self.e_x[:self.num_e]
            e_y = self.e_y[:self.num_e]
            e_vx = self.e_vx[:self.num_e]
            e_vy = self.e_vy[:self.num_e]
            e_vz = self.e_vz[:self.num_e]

            ix_e = np.clip(np.round(e_x / self.dx).astype(int), 0, self.nx - 1)
            iy_e = np.clip(np.round(e_y / self.dy).astype(int), 0, self.ny - 1)
            hit_grid_e = self.isBound[iy_e, ix_e]
            out_e = (
                (e_x < 0.0) |
                (e_x > self.Lx) |
                (e_y < 0.0) |
                (e_y > self.Ly) |
                np.isnan(e_x)
            )
            dead_e = hit_grid_e | out_e
            alive_e = ~dead_e
            n_alive_e = int(np.sum(alive_e))

            if n_alive_e < self.num_e:
                self.e_x[:n_alive_e] = e_x[alive_e]
                self.e_y[:n_alive_e] = e_y[alive_e]
                self.e_vx[:n_alive_e] = e_vx[alive_e]
                self.e_vy[:n_alive_e] = e_vy[alive_e]
                self.e_vz[:n_alive_e] = e_vz[alive_e]
                self.num_e = n_alive_e

        # ————————————————————————————————
        # F. THERMAL
        # ————————————————————————————————
        if sim_mode in ['Thermal', 'Both']:
            T_bound = self.Tmap[self.isBound]
            cooling_factor = (self.emissivity * self.sb_sigma * self.A_cell * self.dt * self.thermal_accel) / self.C_cell
            dT_cool = cooling_factor * (T_bound ** 4 - 300.0 ** 4)
            self.Tmap[self.isBound] -= dT_cool
            self.Tmap = self.Tmap.astype(_NP_FP)

            alpha_diff = self.mat_k / (self.mat_rho * self.mat_cp)
            dt_thermal = self.dt * self.thermal_accel
            dx_m = self.dx * 1e-3
            dy_m = self.dy * 1e-3
            Fo_x = alpha_diff * dt_thermal / dx_m ** 2
            Fo_y = alpha_diff * dt_thermal / dy_m ** 2

            max_Fo = 0.2
            if Fo_x > max_Fo or Fo_y > max_Fo:
                scale = max_Fo / max(Fo_x, Fo_y)
                Fo_x *= scale
                Fo_y *= scale

            if USE_TAICHI:
                for _ in range(10):
                    thermal_conduction_taichi(
                        self.Tmap,
                        self.T_map_new,
                        self.isBound.astype(np.int32),
                        self.nx,
                        self.ny,
                        np.float32(Fo_x),
                        np.float32(Fo_y)
                    )
                    self.Tmap, self.T_map_new = self.T_map_new, self.Tmap
            else:
                T = self.Tmap
                T_new = self.T_map_new
                for _ in range(10):
                    T_new[:] = T[:]
                    for iy_ in range(1, self.ny - 1):
                        for ix_ in range(1, self.nx - 1):
                            if self.isBound[iy_, ix_]:
                                T_l = T[iy_, ix_ - 1] if self.isBound[iy_, ix_ - 1] else T[iy_, ix_]
                                T_r = T[iy_, ix_ + 1] if self.isBound[iy_, ix_ + 1] else T[iy_, ix_]
                                T_d = T[iy_ - 1, ix_] if self.isBound[iy_ - 1, ix_] else T[iy_, ix_]
                                T_u = T[iy_ + 1, ix_] if self.isBound[iy_ + 1, ix_] else T[iy_, ix_]
                                dT = Fo_x * (T_l - 2.0 * T[iy_, ix_] + T_r) + Fo_y * (T_d - 2.0 * T[iy_, ix_] + T_u)
                                T_new[iy_, ix_] = max(T[iy_, ix_] + dT, 300.0)
                    T, T_new = T_new, T
                self.Tmap, self.T_map_new = T, T_new

            self.Tmap[self.isBound] = np.maximum(self.Tmap[self.isBound], 300.0)

            needs_remesh = False
            for i, mask in enumerate(self.mask_grids):
                if np.any(mask):
                    self.T_grids[i] = np.mean(self.Tmap[mask])

            geometry = getattr(self, 'geometry', 'half_hole')
            for i, grid in enumerate(grids):
                dT = self.T_grids[i] - 300.0
                if geometry == 'two_holes':
                    pitch_val = params.get("pitch_mm", params.get("discharge_chamber", {}).get("pitch_mm", 3.0) if isinstance(params.get("discharge_chamber"), dict) else 3.0)
                    L_cant_mm = max(0.2, pitch_val - 2.0 * grid['r'])
                elif geometry == 'one_hole':
                    L_cant_mm = max(0.2, 0.5 * (self.Ly - 2.0 * grid['r']))
                else:  # half_hole
                    L_cant_mm = max(0.2, self.Ly - grid['r'])

                L_cant_m = L_cant_mm * 1e-3
                t_m = grid['t'] * 1e-3
                if t_m > 0 and L_cant_m > 0:
                    delta_m = self.alpha_thermal * dT * L_cant_m ** 2 / (2.0 * t_m)
                    new_defl = delta_m * 1e3
                else:
                    new_defl = 0.0

                if abs(new_defl - self.grid_deflections[i]) > 0.005:
                    self.grid_deflections[i] = new_defl
                    needs_remesh = True

            if needs_remesh:
                self.build_domain(params, preserve_state=True)
                remeshed = True

        # ————————————————————————————————
        # G. MID-HOLE POTENTIAL DIAGNOSTIC
        # Evaluate centrerline potential at the axial midpoint of the
        # first downstream grid (grid 2 if present, else grid 1),
        # at the radial centre of the relevant aperture(s).
        # ————————————————————————————————
        geometry = getattr(self, 'geometry', 'half_hole')
        hole_centers = getattr(self, 'hole_centers', [0.0])

        if len(grids) >= 2:
            x_grid_mm = 0.5 * (self.grid_x_starts[1] + self.grid_x_ends[1])
        elif len(grids) == 1:
            x_grid_mm = 0.5 * (self.grid_x_starts[0] + self.grid_x_ends[0])
        else:
            x_grid_mm = self.Lx * 0.5

        x_idx = int(np.clip(round(x_grid_mm / self.dx), 0, self.nx - 1))

        if hole_centers:
            # For each hole centre, sample the potential and take the minimum
            # (the critical saddle point for electron backstreaming)
            pot_samples = []
            for yc in hole_centers:
                y_idx = int(np.clip(round(yc / self.dy), 0, self.ny - 1))
                pot_samples.append(self.V[y_idx, x_idx])
            min_pot = min(pot_samples)
        else:
            min_pot = self.V[self.ny // 2, x_idx]


        # ————————————————————————————————
        # H. CEX COLLISIONS
        # ————————————————————————————————
        if self.num_p > 0:
            p_x = self.p_x[:self.num_p]
            p_y = self.p_y[:self.num_p]
            p_vx = self.p_vx[:self.num_p]
            p_vy = self.p_vy[:self.num_p]
            p_vz = self.p_vz[:self.num_p]
            p_cex = self.p_isCEX[:self.num_p]

            primary_mask = (~p_cex) & (p_x >= 1.0) & (p_x <= self.Lx)
            if np.any(primary_mask):
                n_primary = int(np.sum(primary_mask))
                px_m = p_x[primary_mask]
                py_m = p_y[primary_mask]

                r_T = self.Ly * 1e-3
                z_m = np.maximum((px_m - 1.0) * 1e-3, 0.0)
                r_m = py_m * 1e-3
                a_corr = 1.0 / (1.0 - 1.0 / np.sqrt(2.0))
                R_dist = np.sqrt(r_m ** 2 + (z_m + r_T) ** 2)
                theta = np.arctan2(r_m, z_m + r_T)

                n_local = params.get('n0', 1e20) * a_corr * (
                    1.0 - 1.0 / np.sqrt(1.0 + (r_T / np.maximum(R_dist, 1e-12)) ** 2)
                ) * np.cos(theta)
                n_local = np.maximum(n_local, 0.0)

                v_mag = np.sqrt(
                    p_vx[primary_mask] ** 2 +
                    p_vy[primary_mask] ** 2 +
                    p_vz[primary_mask] ** 2
                )
                g = np.maximum(v_mag, 1.0)
                E_eV_cex = (0.5 * self.m_ion * g ** 2) / self.q

                sigma_user = self.lookup_user_cs('CX', E_eV_cex)
                sigma = sigma_user if sigma_user is not None else ((-0.8821 * np.log(g) + 15.1262) ** 2) * 1e-20
                prob = 1.0 - np.exp(-n_local * sigma * g * self.dt)
                collided = np.random.rand(n_primary) < prob

                if np.any(collided):
                    c_idx = np.where(primary_mask)[0][collided]
                    n_coll = len(c_idx)
                    neut_vth = np.sqrt(2.0 * self.kB * params.get('Tn', 300.0) / self.m_ion)

                    fM_x = 2.0 * (
                        np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5
                    )
                    fM_y = 2.0 * (
                        np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5
                    )
                    fM_z = 2.0 * (
                        np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5
                    )

                    self.p_vx[c_idx] = (neut_vth * fM_x).astype(_NP_FP)
                    self.p_vy[c_idx] = (neut_vth * fM_y).astype(_NP_FP)
                    self.p_vz[c_idx] = (neut_vth * fM_z).astype(_NP_FP)
                    self.p_isCEX[c_idx] = True

        trans_last_frame = self.get_third_grid_transparency_frame()
        # 1. Isolate the active particles
        p_x_active = self.p_x[:self.num_p]
        p_y_active = self.p_y[:self.num_p]

        # 2. Map continuous coordinates to grid cell indices.
        # Particle positions (p_x, p_y) are stored in mm; self.dx/self.dy are also in mm.
        # Dividing mm by mm gives a dimensionless cell index — do NOT convert dx to metres here.
        ix = np.clip((p_x_active / self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip((p_y_active / self.dy).astype(int), 0, self.ny - 1)

        # 3. Convert 2D indices to a 1D flat index for bincount
        flat_idx = iy * self.nx + ix

        # 4. Count particles in each cell
        # minlength ensures the output array matches your full grid size
        ppc_flat = np.bincount(flat_idx, minlength=self.nx * self.ny)
        ppc_map = ppc_flat.reshape((self.ny, self.nx))
        self.current_ppc_map = ppc_map
        self.accum_ppc_map += ppc_map
        self.ppc_steps_count += 1

        # 5. Evaluate the threshold
        # We ignore cells with 0 particles (vacuum/solid grids) to avoid false positives
        low_ppc_mask = (ppc_map > 0) & (ppc_map < 3)
        self.total_active_cells = int(np.count_nonzero(ppc_map > 0))
        self.low_ppc_cells = int(np.count_nonzero(low_ppc_mask))

        # 6. Check if any cells violate your condition
        if self.low_ppc_cells > 0:
            num_violating_cells = self.low_ppc_cells
            total_active_cells = self.total_active_cells
            pct_low = (num_violating_cells / total_active_cells * 100.0) if total_active_cells > 0 else 0.0
            min_ppc_found = np.min(ppc_map[ppc_map > 0]) # Lowest non-zero count
            
            print(f"Warning: {num_violating_cells} active cells have fewer than 3 macroparticles.")
            print(f"Total active cells: {total_active_cells}")
            print(f"Percentage of low PPC cells (< 3): {pct_low:.2f}%")
            print(f"Lowest non-zero PPC found: {min_ppc_found}")
            
            # Optional: Find exact coordinates of violating cells
            # bad_iy, bad_ix = np.where(low_ppc_mask)
            # print(f"First violating cell at x={bad_ix[0]*self.dx:.2f}mm, y={bad_iy[0]*self.dy:.2f}mm")
        return remeshed, min_pot, current_div, self.T_grids, trans_last_frame

    def reset_ppc_accumulator(self):
        """Resets the time-averaged PPC spatial accumulator."""
        if hasattr(self, 'accum_ppc_map'):
            self.accum_ppc_map.fill(0.0)
        self.ppc_steps_count = 0

    def get_avg_ppc_map(self):
        """Returns the time-averaged 2D PPC map across accumulated steps."""
        if hasattr(self, 'accum_ppc_map') and self.ppc_steps_count > 0:
            return self.accum_ppc_map / float(self.ppc_steps_count)
        if hasattr(self, 'current_ppc_map'):
            return self.current_ppc_map.astype(np.float64)
        return np.zeros((self.ny, self.nx), dtype=np.float64)

    def get_total_energy(self):
        """
        Compute and store the instantaneous total energy of the PIC system:

            E_total = E_kinetic_ions + E_kinetic_electrons + E_field

        where:
            E_kinetic_ions  = sum_p  0.5 * m_ion * (vx^2+vy^2+vz^2) * macro_weight
            E_kinetic_electrons = sum_e  0.5 * m_e  * (vx^2+vy^2+vz^2) * macro_weight
            E_field         = (eps0/2) * sum_cells (Ex^2 + Ey^2) * dx*dy*dz

        Also appends results to the energy history and prints a terminal warning
        if the total energy changes by more than `energy_warning_threshold` (5%) in a
        single step, which signals a violation of energy conservation.

        Returns
        -------
        dict with keys: t_s, ke_ions_J, ke_elec_J, field_J, total_J
        """
        t_now = self.iteration * self.dt

        # --- Ion kinetic energy ---
        if self.num_p > 0:
            v2 = (self.p_vx[:self.num_p]**2
                  + self.p_vy[:self.num_p]**2
                  + self.p_vz[:self.num_p]**2)
            ke_i = float(np.sum(v2)) * 0.5 * self.m_ion * self.macro_weight
        else:
            ke_i = 0.0

        # --- Electron kinetic energy ---
        # Electrons share the same macro_weight as ions (quasi-neutrality assumption)
        if self.num_e > 0:
            v2e = (self.e_vx[:self.num_e]**2
                   + self.e_vy[:self.num_e]**2
                   + self.e_vz[:self.num_e]**2)
            ke_e = float(np.sum(v2e)) * 0.5 * self.m_e * self.macro_weight
        else:
            ke_e = 0.0

        # --- Electrostatic field energy: (eps0/2) * integral(|E|^2) dV ---
        # Cell volume uses 2D slice with unit depth of 1 m (standard 2D PIC convention)
        cell_area = (self.dx * 1e-3) * (self.dy * 1e-3)  # [m^2]
        e_field = float(np.sum(self.Ex**2 + self.Ey**2)) * 0.5 * self.eps0 * cell_area

        total = ke_i + ke_e + e_field

        # --- Step-to-step conservation warning ---
        if self._prev_total_energy is not None and self._prev_total_energy > 0.0:
            rel_change = abs(total - self._prev_total_energy) / self._prev_total_energy
            if rel_change > self.energy_warning_threshold:
                print(
                    f"[Energy Warning] iter={self.iteration}: "
                    f"total energy changed by {rel_change*100:.1f}% in one step "
                    f"(prev={self._prev_total_energy:.4e} J, now={total:.4e} J). "
                    f"KE_i={ke_i:.3e} J, KE_e={ke_e:.3e} J, E_field={e_field:.3e} J"
                )
        self._prev_total_energy = total

        # Append to history
        self.energy_history_t.append(t_now)
        self.energy_history_ke_i.append(ke_i)
        self.energy_history_ke_e.append(ke_e)
        self.energy_history_fe.append(e_field)
        self.energy_history_tot.append(total)

        return dict(t_s=t_now, ke_ions_J=ke_i, ke_elec_J=ke_e, field_J=e_field, total_J=total)

    # —————————————————————————————————
    def get_third_grid_transparency_frame(self):
        if self.entered_optics_step <= 0.0: return 0.0
        return self.transmitted3_step / self.entered_optics_step

    def get_transparency(self):
        if self.entered_optics <= 0.0: return 0.0
        return self.transmitted_ions / self.entered_optics

    def has_active_particles(self):
        return (self.num_p > 0) or (self.num_e > 0)
    
    def _segment_hits_grid(self, x0, y0, x1, y1, samples=8):
        hit = np.zeros(len(x0), dtype=bool)
        # Default to ending cell (will be overwritten for hits)
        hit_ix = np.clip(np.round(x1 / self.dx).astype(int), 0, self.nx - 1)
        hit_iy = np.clip(np.round(y1 / self.dy).astype(int), 0, self.ny - 1)
        
        for i in range(1, samples+1):
            f = i / float(samples)
            xi = x0 + (x1 - x0) * f
            yi = y0 + (y1 - y0) * f
            c_x = np.clip(np.round(xi / self.dx).astype(int), 0, self.nx - 1)
            c_y = np.clip(np.round(yi / self.dy).astype(int), 0, self.ny - 1)
            step_hit = self.isBound[c_y, c_x]
            
            # Record coordinates only for the FIRST hit
            new_hits = step_hit & ~hit
            hit_ix[new_hits] = c_x[new_hits]
            hit_iy[new_hits] = c_y[new_hits]
            
            hit |= step_hit
            
        return hit, hit_ix, hit_iy
    
    def get_groove_profile(self, grid_idx, thresh=None, accumulate_subcell=True, face='upstream'):
        if grid_idx < 0 or grid_idx >= len(self.mask_grids):
            return np.array([]), np.array([])

        mask = self.mask_grids[grid_idx]
        if not np.any(mask): return np.array([]), np.array([])

        depth = self.eroded_depth
        if accumulate_subcell and thresh and thresh > 0:
            depth = depth + (self.damage_map/thresh)*self.dy

        y_mm  = np.arange(self.ny)*self.dy
        per_y = np.zeros(self.ny, dtype=np.float64)
        has_any = mask.any(axis=1)

        if face == 'any':
            cols = np.any(mask, axis=0)
            if not np.any(cols): return np.array([]), np.array([])
            per_y[has_any] = depth[has_any][:,cols].max(axis=1)
        elif face == 'upstream':
            # np.argmax returns first True index along x for each y
            first_idx = np.argmax(mask, axis=1)
            per_y[has_any] = depth[np.where(has_any)[0], first_idx[has_any]]
        elif face == 'downstream':
            last_idx = self.nx-1-np.argmax(mask[:,::-1], axis=1)
            per_y[has_any] = depth[np.where(has_any)[0], last_idx[has_any]]
        else:
            raise ValueError(f"face must be 'upstream','downstream', or 'any' (got {face!r})")

        return y_mm, per_y*1000.0

    def get_particle_kinematics(self):
        t_current = self.iteration * self.dt
        
        # Note: Added p_vz and e_vz to the output stack to provide the full 3D velocity vector.
        # This means the output now has 8 columns: [time, x, y, vx, vy, vz, energy_eV, type]
        if self.num_p > 0:
            p_x,p_y = self.p_x[:self.num_p], self.p_y[:self.num_p]
            p_vx,p_vy,p_vz = self.p_vx[:self.num_p],self.p_vy[:self.num_p],self.p_vz[:self.num_p]
            p_cex = self.p_isCEX[:self.num_p]
            v_sq_i     = p_vx**2+p_vy**2+p_vz**2
            energy_eV_i= (0.5*self.m_ion*v_sq_i)/self.q
            ions = np.column_stack((np.full(self.num_p,t_current),
                                    p_x,p_y,p_vx,p_vy,p_vz,
                                    energy_eV_i, p_cex.astype(int)))
        else:
            ions = np.empty((0,8))

        if self.num_e > 0:
            e_x,e_y = self.e_x[:self.num_e],self.e_y[:self.num_e]
            e_vx,e_vy,e_vz = self.e_vx[:self.num_e],self.e_vy[:self.num_e],self.e_vz[:self.num_e]
            v_sq_e     = e_vx**2+e_vy**2+e_vz**2
            energy_eV_e= (0.5*self.m_e*v_sq_e)/self.q
            type_e     = np.where(e_x<=4.0, 2, 3)
            elecs = np.column_stack((np.full(self.num_e,t_current),
                                     e_x,e_y,e_vx,e_vy,e_vz,
                                     energy_eV_e, type_e))
        else:
            elecs = np.empty((0,8))

        return np.vstack((ions, elecs))