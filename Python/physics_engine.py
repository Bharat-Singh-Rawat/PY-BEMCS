import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
import taichi as ti

# --- Optional GPU Poisson via CuPy (falls back silently if unavailable) ---
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
# ARCHITECTURE & PRECISION SWITCHING GUIDE
# =============================================================================
# To switch how this simulation runs, change the `ti.init` line below.
# 
# OPTION 1: Max Speed / Broad GPU Compatibility (Standard Consumer GPUs, Apple Silicon)
# ti.init(arch=ti.gpu, default_fp=ti.f32)
# -> IMPORTANT: If using this, all kernel type hints must be `ti.f32`
#
# OPTION 2: High Precision / CPU Fallback (Use if f64 is strictly required)
# ti.init(arch=ti.cpu, default_fp=_TI_FP)
# -> IMPORTANT: If using this, all kernel type hints must be changed to `_TI_FP`
# =============================================================================

# --- Precision switch: flip these 3 lines to go back to CPU + f64 ---
# ti.init(arch=ti.cpu, default_fp=_TI_FP)
# _TI_FP = _TI_FP
# _NP_FP = np.float64
ti.init(arch=ti.gpu, default_fp=ti.f32)
_TI_FP = ti.f32
_NP_FP = np.float32

# =============================================================================
# TAICHI KERNELS (Compiled for GPU using 32-bit floats)
# =============================================================================

@ti.kernel
def accumulate_rho_taichi(
    x: ti.types.ndarray(dtype=_TI_FP),
    y: ti.types.ndarray(dtype=_TI_FP),
    rho: ti.types.ndarray(dtype=_TI_FP),
    num_p: ti.i32,
    dx: _TI_FP,
    dy: _TI_FP,
    nx: ti.i32,
    ny: ti.i32,
    charge_density: _TI_FP
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
    x: ti.types.ndarray(dtype=_TI_FP),
    y: ti.types.ndarray(dtype=_TI_FP),
    vx: ti.types.ndarray(dtype=_TI_FP),
    vy: ti.types.ndarray(dtype=_TI_FP),
    vz: ti.types.ndarray(dtype=_TI_FP), # NEW: 3rd Velocity Component
    Ex: ti.types.ndarray(dtype=_TI_FP),
    Ey: ti.types.ndarray(dtype=_TI_FP),
    Bx: ti.types.ndarray(dtype=_TI_FP), # NEW: Magnetic Field grids
    By: ti.types.ndarray(dtype=_TI_FP),
    Bz: ti.types.ndarray(dtype=_TI_FP),
    num_p: ti.i32,
    dx: _TI_FP,
    dy: _TI_FP,
    nx: ti.i32,
    ny: ti.i32,
    dt: _TI_FP,
    q_m: _TI_FP
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

        fx = idx_x - ti.cast(ix0, _TI_FP)
        fy = idx_y - ti.cast(iy0, _TI_FP)

        # Interpolate Electric Field (Assuming Ez = 0 for 2D electrostatics)
        Ex_p = Ex[iy0, ix0] * (1.0 - fx) * (1.0 - fy) + Ex[iy0, ix0 + 1] * fx * (1.0 - fy) + \
               Ex[iy0 + 1, ix0] * (1.0 - fx) * fy + Ex[iy0 + 1, ix0 + 1] * fx * fy
        Ey_p = Ey[iy0, ix0] * (1.0 - fx) * (1.0 - fy) + Ey[iy0, ix0 + 1] * fx * (1.0 - fy) + \
               Ey[iy0 + 1, ix0] * (1.0 - fx) * fy + Ey[iy0 + 1, ix0 + 1] * fx * fy

        # Interpolate Magnetic Field
        Bx_p = Bx[iy0, ix0] * (1.0 - fx) * (1.0 - fy) + Bx[iy0, ix0 + 1] * fx * (1.0 - fy) + \
               Bx[iy0 + 1, ix0] * (1.0 - fx) * fy + Bx[iy0 + 1, ix0 + 1] * fx * fy
        By_p = By[iy0, ix0] * (1.0 - fx) * (1.0 - fy) + By[iy0, ix0 + 1] * fx * (1.0 - fy) + \
               By[iy0 + 1, ix0] * (1.0 - fx) * fy + By[iy0 + 1, ix0 + 1] * fx * fy
        Bz_p = Bz[iy0, ix0] * (1.0 - fx) * (1.0 - fy) + Bz[iy0, ix0 + 1] * fx * (1.0 - fy) + \
               Bz[iy0 + 1, ix0] * (1.0 - fx) * fy + Bz[iy0 + 1, ix0 + 1] * fx * fy

        # =====================================================================
        # BORIS PUSHER ALGORITHM
        # =====================================================================
        # STEP 1: First half E-field acceleration (v_minus)
        v_minus_x = vx[i] + (q_m * Ex_p * dt) / 2.0
        v_minus_y = vy[i] + (q_m * Ey_p * dt) / 2.0
        v_minus_z = vz[i] # Ez is 0

        # STEP 2: Magnetic Field Rotation (v_plus)
        # Calculates rotation vectors t and s
        t_x = (q_m * Bx_p * dt) / 2.0
        t_y = (q_m * By_p * dt) / 2.0
        t_z = (q_m * Bz_p * dt) / 2.0
        t_mag_sq = t_x**2 + t_y**2 + t_z**2
        
        s_x = 2.0 * t_x / (1.0 + t_mag_sq)
        s_y = 2.0 * t_y / (1.0 + t_mag_sq)
        s_z = 2.0 * t_z / (1.0 + t_mag_sq)

        # Cross product 1: v_prime = v_minus + (v_minus x t)
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
        vz[i] = v_plus_z # Ez is 0

        # =====================================================================
        # KINEMATIC UPDATE
        # =====================================================================
        x[i] += vx[i] * dt * 1000.0
        y[i] += vy[i] * dt * 1000.0

@ti.kernel
def thermal_conduction_taichi(
    T: ti.types.ndarray(dtype=_TI_FP),
    T_new: ti.types.ndarray(dtype=_TI_FP),
    mask: ti.types.ndarray(dtype=ti.i32),
    nx: ti.i32,
    ny: ti.i32,
    Fo_x: _TI_FP,
    Fo_y: _TI_FP
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

            dT = Fo_x * (T_l - 2.0*T[iy, ix] + T_r) + Fo_y * (T_d - 2.0*T[iy, ix] + T_u)
            T_new[iy, ix] = ti.max(T[iy, ix] + dT, 300.0)
        else:
            T_new[iy, ix] = T[iy, ix]

# =============================================================================
# MAIN SIMULATOR CLASS
# =============================================================================

class DigitalTwinSimulator:
    def __init__(self):
        # Constants for simulation
        self.dt = 1e-9
        self.q = 1.602e-19
        self.m_XE = 131.293 * 1.6605e-27
        self.m_ion = self.m_XE          # configurable ion mass (set by GUI)
        self.Z_ion = 1                   # charge state (set by GUI)
        self.q_ion = self.q              # ion charge = Z * e
        self.kB = 1.380649e-23
        self.eps0 = 8.854e-12

        self.m_e = self.m_XE / 100.0

        # User-imported cross-section splines: {label: {energy, cs, spline, type}}
        self.user_cs = {}

        # --- Grid Material Presets ---
        # Each entry: (k [W/m/K], rho [kg/m³], cp [J/kg/K], emissivity,
        #              alpha_thermal [1/K], sputter_coeff, sputter_threshold_eV)
        self.MATERIAL_PRESETS = {
            'Molybdenum': {'k': 138.0, 'rho': 10280.0, 'cp': 250.0,
                           'emissivity': 0.80, 'alpha': 4.8e-6,
                           'E_mod': 329e9,
                           'Y_coeff': 1.05e-4, 'E_th': 30.0},
            'Steel (SS316)': {'k': 16.3, 'rho': 8000.0, 'cp': 500.0,
                              'emissivity': 0.60, 'alpha': 16.0e-6,
                              'E_mod': 193e9,
                              'Y_coeff': 2.8e-4, 'E_th': 25.0},
            'Titanium':  {'k': 21.9, 'rho': 4507.0, 'cp': 520.0,
                          'emissivity': 0.50, 'alpha': 8.6e-6,
                          'E_mod': 116e9,
                          'Y_coeff': 1.8e-4, 'E_th': 20.0},
            'Graphite':  {'k': 120.0, 'rho': 2200.0, 'cp': 710.0,
                          'emissivity': 0.85, 'alpha': 3.0e-6,
                          'E_mod': 11e9,
                          'Y_coeff': 3.5e-4, 'E_th': 15.0},
        }

        # Active material properties (default: Molybdenum)
        mat = self.MATERIAL_PRESETS['Molybdenum']
        self.mat_k = mat['k']
        self.mat_rho = mat['rho']
        self.mat_cp = mat['cp']
        self.emissivity = mat['emissivity']
        self.alpha_thermal = mat['alpha']
        self.E_modulus = mat['E_mod']
        self.sputter_Y_coeff = mat['Y_coeff']
        self.sputter_E_th = mat['E_th']

        self.macro_weight = 3e5
        self.sb_sigma = 5.67e-8
        self.thermal_accel = 1e7

        # Mesh parameters
        self.Lx = 20
        self.Ly = 3
        self.dx = 0.015
        self.dy = 0.015
        self.nx = int(self.Lx / self.dx) + 1
        self.ny = int(self.Ly / self.dy) + 1

        self._recompute_cell_constants()
        
        # Dynamic Grid Arrays
        self.T_grids = []
        self.mask_grids = []
        self.V_dc = None
        
        self.x_pts = np.linspace(0, self.Lx, self.nx)
        self.y_pts = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x_pts, self.y_pts)
        
        self.iteration = 0
        self.T_map = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
        self.T_map_new = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)

        # Sparse Matrix
        self.laplacian_lu = None
        self.is_interior_mask = None
        self.is_bound_mask = None

        self.reset_arrays()

    def reset_arrays(self):
        # PRE-ALLOCATED PARTICLE BUFFERS
        self.max_p = 50000
        self.max_e = 50000

        self.p_x = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_y = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_vx = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_vy = np.zeros(self.max_p, dtype=_NP_FP)
        self.p_vz = np.zeros(self.max_p, dtype=_NP_FP) # Added vz
        self.p_isCEX = np.zeros(self.max_p, dtype=bool)
        self.num_p = 0

        self.e_x = np.zeros(self.max_e, dtype=_NP_FP)
        self.e_y = np.zeros(self.max_e, dtype=_NP_FP)
        self.e_vx = np.zeros(self.max_e, dtype=_NP_FP)
        self.e_vy = np.zeros(self.max_e, dtype=_NP_FP)
        self.e_vz = np.zeros(self.max_e, dtype=_NP_FP) # Added vz
        self.num_e = 0

        # V stays f64 — Poisson LU solve is on CPU/SciPy, benefits from f64.
        self.V = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.rho = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.isBound = np.zeros((self.ny, self.nx), dtype=bool)
        self.V_fixed = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.damage_map = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.Ex = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.Ey = np.zeros((self.ny, self.nx), dtype=_NP_FP)

        # Static Magnetic Field Grids (Initialize to Zero)
        self.Bx = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.By = np.zeros((self.ny, self.nx), dtype=_NP_FP)
        self.Bz = np.zeros((self.ny, self.nx), dtype=_NP_FP)

    def _recompute_cell_constants(self):
        """Recompute cell thermal mass and area from current material + mesh."""
        self.C_cell = self.mat_rho * (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3 * self.mat_cp
        self.A_cell = 2 * (self.dx * 1e-3) * 1e-3

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
        self.mat_k = mat['k']
        self.mat_rho = mat['rho']
        self.mat_cp = mat['cp']
        self.emissivity = mat['emissivity']
        self.alpha_thermal = mat['alpha']
        self.E_modulus = mat['E_mod']
        self.sputter_Y_coeff = mat['Y_coeff']
        self.sputter_E_th = mat['E_th']
        self._recompute_cell_constants()

    def lookup_user_cs(self, cs_type_prefix, energy_eV):
        """
        Look up cross-section from user-imported spline data.
        cs_type_prefix: 'CX' or 'SEE' or 'Custom' — matches first dataset whose label starts with this.
        energy_eV: array of energies in eV
        Returns sigma array in m², or None if no matching user data.
        """
        for label, ds in self.user_cs.items():
            if label.startswith(cs_type_prefix) and ds.get('spline') is not None:
                log_e = np.log10(np.maximum(energy_eV, 1e-30))
                # Clamp to data range
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
            self.p_x = np.pad(self.p_x, (0, new_max - self.max_p))
            self.p_y = np.pad(self.p_y, (0, new_max - self.max_p))
            self.p_vx = np.pad(self.p_vx, (0, new_max - self.max_p))
            self.p_vy = np.pad(self.p_vy, (0, new_max - self.max_p))
            self.p_vz = np.pad(self.p_vz, (0, new_max - self.max_p))
            self.p_isCEX = np.pad(self.p_isCEX, (0, new_max - self.max_p))
            self.max_p = new_max
        
        s = self.num_p
        e = s + n_new
        self.p_x[s:e] = x
        self.p_y[s:e] = y
        self.p_vx[s:e] = vx
        self.p_vy[s:e] = vy
        self.p_vz[s:e] = vz
        self.p_isCEX[s:e] = is_cex
        self.num_p += n_new

    def _add_electrons(self, x, y, vx, vy, vz):
        n_new = len(x)
        if self.num_e + n_new > self.max_e:
            new_max = max(self.max_e * 2, self.num_e + n_new)
            self.e_x = np.pad(self.e_x, (0, new_max - self.max_e))
            self.e_y = np.pad(self.e_y, (0, new_max - self.max_e))
            self.e_vx = np.pad(self.e_vx, (0, new_max - self.max_e))
            self.e_vy = np.pad(self.e_vy, (0, new_max - self.max_e))
            self.e_vz = np.pad(self.e_vz, (0, new_max - self.max_e))
            self.max_e = new_max
        
        s = self.num_e
        e = s + n_new
        self.e_x[s:e] = x
        self.e_y[s:e] = y
        self.e_vx[s:e] = vx
        self.e_vy[s:e] = vy
        self.e_vz[s:e] = vz
        self.num_e += n_new

    def build_sparse_matrix(self):
        N = self.nx * self.ny
        idx = np.arange(N)
        y = idx // self.nx
        x = idx % self.nx

        is_bound = self.isBound.flatten()
        is_right = (x == self.nx - 1) & ~is_bound
        is_top = (y == self.ny - 1) & ~is_bound & ~is_right
        is_bottom = (y == 0) & ~is_bound & ~is_right & ~is_top
        is_interior = ~is_bound & ~is_right & ~is_top & ~is_bottom

        self.is_interior_mask = is_interior
        self.is_bound_mask = is_bound

        row, col, data = [], [], []

        idx_b = idx[is_bound]
        row.append(idx_b); col.append(idx_b); data.append(np.ones_like(idx_b))

        idx_r = idx[is_right]
        row.append(idx_r); col.append(idx_r); data.append(np.ones_like(idx_r))
        row.append(idx_r); col.append(idx_r - 1); data.append(-np.ones_like(idx_r))

        idx_t = idx[is_top]
        row.append(idx_t); col.append(idx_t); data.append(np.ones_like(idx_t))
        row.append(idx_t); col.append(idx_t - self.nx); data.append(-np.ones_like(idx_t))

        idx_bot = idx[is_bottom]
        row.append(idx_bot); col.append(idx_bot); data.append(np.ones_like(idx_bot))
        row.append(idx_bot); col.append(idx_bot + self.nx); data.append(-np.ones_like(idx_bot))

        idx_in = idx[is_interior]
        row.append(idx_in); col.append(idx_in); data.append(np.full_like(idx_in, -4.0))
        row.append(idx_in); col.append(idx_in - 1); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in + 1); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in - self.nx); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in + self.nx); data.append(np.ones_like(idx_in))

        row = np.concatenate(row)
        col = np.concatenate(col)
        data = np.concatenate(data)
        
        A = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsc()
        self.laplacian_lu = factorized(A)

        # --- GPU LU factorization (mirrors the CPU one for fast re-solves) ---
        self.laplacian_lu_gpu = None
        self.is_bound_mask_gpu = None
        self.is_interior_mask_gpu = None
        if _GPU_POISSON:
            try:
                A_csc = A.astype(np.float64)
                A_gpu = cp_sp.csc_matrix(
                    (cp.asarray(A_csc.data),
                     cp.asarray(A_csc.indices),
                     cp.asarray(A_csc.indptr)),
                    shape=A_csc.shape,
                )
                self.laplacian_lu_gpu = cp_splu(A_gpu)
                self.is_bound_mask_gpu = cp.asarray(self.is_bound_mask)
                self.is_interior_mask_gpu = cp.asarray(self.is_interior_mask)
            except Exception as exc:
                print(f"[Poisson] GPU factorization failed, using CPU path: {exc}")
                self.laplacian_lu_gpu = None

    def build_domain(self, params):
        # Apply GUI Advanced Parameters
        self.Lx = params.get('Lx', self.Lx)
        self.Ly = params.get('Ly', self.Ly)
        self.nx = int(self.Lx / self.dx) + 1
        self.ny = int(self.Ly / self.dy) + 1
        
        self.m_e = self.m_ion / params.get('m_e_ratio', 1000.0)
        v_offset = params.get('V_plasma_offset', 20.0)

        # Re-initialize coordinates and thermal arrays with potentially new shape
        self.x_pts = np.linspace(0, self.Lx, self.nx)
        self.y_pts = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x_pts, self.y_pts)
        
        self.T_map = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
        self.T_map_new = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)

        # Reset particle and field arrays to matching shape
        self.reset_arrays()
        self.iteration = 0

        cell_vol = (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3 
        n0 = params.get('n0_plasma', 1e17) 
        target_ppc = 40.0 
        self.macro_weight = max((n0 * cell_vol) / target_ppc, 1e3)

        self.mask_grids = []
        self.T_grids = []
        grids = params.get('grids', [])

        # Initialise per-grid cantilever deflections (mm, in x-direction at free end)
        if not hasattr(self, 'grid_deflections') or len(self.grid_deflections) != len(grids):
            self.grid_deflections = [0.0] * len(grids)

        current_x = 1.0  # Starting position for the first grid

        for i, grid in enumerate(grids):
            g_start = current_x
            g_end = g_start + grid['t']
            delta = self.grid_deflections[i]  # tip deflection in mm

            # Cantilever profile: clamped at Y = Ly (top wall), free at Y = r (aperture)
            # Normalised distance from fixed end: 0 at Ly, 1 at r
            L_cant = self.Ly - grid['r']       # cantilever length (mm)
            if L_cant > 0 and abs(delta) > 1e-6:
                eta = np.clip((self.Ly - self.Y) / L_cant, 0.0, 1.0)
                dx_bow = delta * eta**2          # quadratic cantilever shape (mm)
            else:
                dx_bow = 0.0

            in_grid = (self.X >= g_start + dx_bow) & (self.X <= g_end + dx_bow)
            R_grid = grid['r'] + np.maximum(0, self.X - (g_start + dx_bow)) * np.tan(np.radians(grid['cham']))

            mask = in_grid & (self.Y >= R_grid)
            self.isBound[mask] = True
            self.V_fixed[mask] = grid['V']

            self.mask_grids.append(mask)
            self.T_grids.append(300.0)

            current_x = g_end + grid['gap']

        self.V_dc = np.copy(self.V_fixed) # Store DC potentials for RF superimposition

        self.isBound[:, 0] = True
        # Set left boundary to first grid voltage + dynamic offset
        v_plasma_bound = grids[0]['V'] + v_offset if grids else 1000 + v_offset
        self.V_fixed[:, 0] = v_plasma_bound
        
        self.T_map[~self.isBound] = 300.0
        
        self.build_sparse_matrix()
        self.recalc_poisson(iterations=30, params=params)

    def recalc_poisson(self, iterations=5, params=None):
        if self.laplacian_lu is None: return

        dx_m2 = (self.dx * 1e-3)**2
        coeff = dx_m2 / self.eps0

        grids = params.get('grids', [{'V': 1000}]) if params else [{'V': 1000}]
        v_offset = params.get('V_plasma_offset', 20.0) if params else 20.0
        V_plasma = grids[0]['V'] + v_offset
        Te_up = params.get('Te_up', 3.0)
        n0 = params.get('n0_plasma', 1e17)
        omega = 0.2

        if _GPU_POISSON and self.laplacian_lu_gpu is not None:
            self._recalc_poisson_gpu(iterations, coeff, V_plasma, Te_up, n0, omega)
        else:
            self._recalc_poisson_cpu(iterations, coeff, V_plasma, Te_up, n0, omega)

        self.Ey, self.Ex = np.gradient(-self.V, self.dy * 1e-3, self.dx * 1e-3)
        self.Ey = self.Ey.astype(_NP_FP)
        self.Ex = self.Ex.astype(_NP_FP)

    def _recalc_poisson_cpu(self, iterations, coeff, V_plasma, Te_up, n0, omega):
        b = np.zeros(self.nx * self.ny, dtype=np.float64)
        V_fixed_flat = self.V_fixed.flatten()

        # Fluid model for plasma sheath with Boltzmann electron distribution
        for _ in range(iterations):
            rho_e = -self.q * n0 * np.exp((np.minimum(self.V, V_plasma) - V_plasma) / Te_up)
            rho_total = self.rho + rho_e
            rho_flat = rho_total.flatten()

            b.fill(0.0)
            b[self.is_bound_mask] = V_fixed_flat[self.is_bound_mask]
            b[self.is_interior_mask] = -coeff * rho_flat[self.is_interior_mask]

            V_new_flat = self.laplacian_lu(b)
            V_new = V_new_flat.reshape((self.ny, self.nx))

            self.V = ((1 - omega) * self.V + omega * V_new).astype(np.float64)

    def _recalc_poisson_gpu(self, iterations, coeff, V_plasma, Te_up, n0, omega):
        V_gpu = cp.asarray(self.V, dtype=cp.float64)
        rho_gpu = cp.asarray(self.rho).astype(cp.float64)
        V_fixed_flat_gpu = cp.asarray(self.V_fixed.ravel(), dtype=cp.float64)
        b_gpu = cp.zeros(self.nx * self.ny, dtype=cp.float64)
        q = self.q

        for _ in range(iterations):
            rho_e_gpu = -q * n0 * cp.exp((cp.minimum(V_gpu, V_plasma) - V_plasma) / Te_up)
            rho_flat_gpu = (rho_gpu + rho_e_gpu).ravel()

            b_gpu[:] = 0.0
            b_gpu[self.is_bound_mask_gpu] = V_fixed_flat_gpu[self.is_bound_mask_gpu]
            b_gpu[self.is_interior_mask_gpu] = -coeff * rho_flat_gpu[self.is_interior_mask_gpu]

            V_new_flat = self.laplacian_lu_gpu.solve(b_gpu)
            V_new_gpu = V_new_flat.reshape((self.ny, self.nx))

            V_gpu = (1.0 - omega) * V_gpu + omega * V_new_gpu

        self.V = cp.asnumpy(V_gpu)

    def step(self, params):
        if not np.any(self.Ex): return False, np.nan, np.nan, self.T_grids
        sim_mode = params.get('sim_mode', 'Both')
        self.iteration += 1
        
        t_current = self.iteration * self.dt
        grids = params.get('grids', [])
        
        # --- RF CO-EXTRACTION LOGIC ---
        if params.get('rf_enable') and grids:
            rf_idx = params.get('rf_grid_idx', 0)
            if rf_idx < len(grids):
                f_hz = params.get('rf_freq', 13.56) * 1e6
                v_rf = params.get('rf_amp', 100) * np.sin(2 * np.pi * f_hz * t_current)
                self.V_fixed[self.mask_grids[rf_idx]] = self.V_dc[self.mask_grids[rf_idx]] + v_rf
                self.recalc_poisson(iterations=2, params=params)

        # --- A. INJECT PARTICLES ---
        n0 = params.get('n0_plasma', 1e17)
        Te_up = params.get('Te_up', 3.0)
        v_bohm = np.sqrt(self.q_ion * Te_up / self.m_ion)

        injection_area = (grids[0]['r'] - 0.05) * 1e-3 * 1.0 if grids else 1e-3
        I_ion = self.q_ion * 0.61 * n0 * v_bohm * injection_area

        charge_per_macro = self.q_ion * self.macro_weight
        num_inject_float = (I_ion * self.dt) / charge_per_macro
        num_inject = int(num_inject_float) + (1 if np.random.rand() < (num_inject_float % 1) else 0)

        r_max = grids[0]['r'] - 0.05 if grids else 1.0

        if num_inject > 0:
            new_y = np.random.uniform(0.02, r_max, num_inject).astype(np.float64)
            new_x = np.full(num_inject, 0.1, dtype=np.float64)
            v_spread = np.sqrt(self.q_ion * params.get('Ti', 0.1) / self.m_ion)
            new_vx = np.full(num_inject, v_bohm, dtype=np.float64) + np.random.randn(num_inject).astype(np.float64) * v_spread
            new_vy = (np.random.randn(num_inject) * v_spread).astype(np.float64)
            new_vz = (np.random.randn(num_inject) * v_spread).astype(np.float64)
            new_cex = np.zeros(num_inject, dtype=bool)
            self._add_ions(new_x, new_y, new_vx, new_vy, new_vz, new_cex)
            
        # Source Electrons (for true RF Co-extraction)
        if params.get('rf_enable'):
            v_e_th_source = np.sqrt(2 * self.q * Te_up / self.m_e)
            I_e = self.q * 0.25 * n0 * v_e_th_source * injection_area
            num_e_float = (I_e * self.dt) / charge_per_macro
            num_inj_e = int(num_e_float) + (1 if np.random.rand() < (num_e_float % 1) else 0)
            
            if num_inj_e > 0:
                new_ey = np.random.uniform(0.02, r_max, num_inj_e).astype(np.float64)
                new_ex = np.full(num_inj_e, 0.1, dtype=np.float64)
                new_evx = (np.abs(np.random.randn(num_inj_e)) * v_e_th_source + v_bohm).astype(np.float64)
                new_evy = (np.random.randn(num_inj_e) * v_e_th_source).astype(np.float64)
                new_evz = (np.random.randn(num_inj_e) * v_e_th_source).astype(np.float64)
                self._add_electrons(new_ex, new_ey, new_evx, new_evy, new_evz)

        # --- NEUTRALIZER CONTROL ---
        num_e_neut = int(params.get('neut_rate', 30))
        Te_eV = params.get('Te', 5.0)

        # 1. Fetch custom position and radius (fallback to domain edges)
        neut_x = params.get('neut_x', self.Lx - 0.1)
        neut_r = params.get('neut_r', self.Ly)

        if num_e_neut > 0:
            # Emit uniformly between the centerline (0) and the specified radius (neut_r)
            new_ey = np.random.uniform(0.0, neut_r, num_e_neut).astype(np.float64)
            new_ex = np.full(num_e_neut, neut_x, dtype=np.float64)
            v_e_th = np.sqrt(2 * self.q * Te_eV / self.m_e)
            
            # 2. Purely thermal emission (Gaussian centered at 0)
            # This shoots ~50% of electrons left and ~50% right automatically
            new_evx = (np.random.randn(num_e_neut) * v_e_th).astype(np.float64)
            new_evy = (np.random.randn(num_e_neut) * v_e_th).astype(np.float64)
            new_evz = (np.random.randn(num_e_neut) * v_e_th).astype(np.float64)
            self._add_electrons(new_ex, new_ey, new_evx, new_evy, new_evz)

        p_x = self.p_x[:self.num_p]
        p_y = self.p_y[:self.num_p]
        p_vx = self.p_vx[:self.num_p]
        p_vy = self.p_vy[:self.num_p]
        p_vz = self.p_vz[:self.num_p]
        p_cex = self.p_isCEX[:self.num_p]

        e_x = self.e_x[:self.num_e]
        e_y = self.e_y[:self.num_e]
        e_vx = self.e_vx[:self.num_e]
        e_vy = self.e_vy[:self.num_e]
        e_vz = self.e_vz[:self.num_e]

        # B. POISSON SOLVER
        self.rho.fill(0.0)
        cell_vol = (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3 
        charge_per_particle = self.q * self.macro_weight
        
        # TAICHI: Parallel Density Accumulation
        if self.num_p > 0:
            accumulate_rho_taichi(
                p_x, p_y, self.rho, self.num_p, 
                self.dx, self.dy, self.nx, self.ny, 
                charge_per_particle / cell_vol
            )
            
        if self.num_e > 0:
            accumulate_rho_taichi(
                e_x, e_y, self.rho, self.num_e, 
                self.dx, self.dy, self.nx, self.ny, 
                -charge_per_particle / cell_vol
            )

        if self.iteration % 2 == 0:
            self.recalc_poisson(iterations=5, params=params)

        # C. PARTICLE PUSH ALGORITHM (TAICHI BORIS PUSHER)
        if self.num_p > 0:
            push_particles_boris_taichi(
                p_x, p_y, p_vx, p_vy, p_vz,
                self.Ex, self.Ey, self.Bx, self.By, self.Bz,
                self.num_p, self.dx, self.dy, self.nx, self.ny,
                self.dt, self.q_ion / self.m_ion
            )

        if self.num_e > 0:
            push_particles_boris_taichi(
                e_x, e_y, e_vx, e_vy, e_vz, 
                self.Ex, self.Ey, self.Bx, self.By, self.Bz,
                self.num_e, self.dx, self.dy, self.nx, self.ny, 
                self.dt, -self.q / self.m_e
            )

        # Calculate limits dynamically
        max_grid_x = 1.0 + sum([g['t'] + g['gap'] for g in grids]) if grids else 3.0
        post_grid = (~p_cex) & (p_x > max_grid_x)
        current_div = np.percentile(np.abs(np.arctan2(p_vy[post_grid], p_vx[post_grid])) * 180 / np.pi, 95) if np.sum(post_grid) > 5 else np.nan
        
        # Fixing the saddle point at the centre of the SECOND grid
        if len(grids) >= 2:
            grid2_start_mm = 1.0 + grids[0]['t'] + grids[0]['gap']
            grid2_center_mm = grid2_start_mm + (grids[1]['t'] / 2.0)
            x_idx = int(grid2_center_mm / self.dx)
            x_idx = np.clip(x_idx, 0, self.nx - 1) 
            min_pot = self.V[0, x_idx]
        elif len(grids) == 1:
            grid1_center_mm = 1.0 + (grids[0]['t'] / 2.0)
            x_idx = int(grid1_center_mm / self.dx)
            min_pot = self.V[0, np.clip(x_idx, 0, self.nx - 1)]
        else:
            min_pot = np.min(self.V[0, :])

        # D. 2D THERMAL CALCULATIONS & HIT DETECTION 
        ix = np.clip(np.round(p_x / self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip(np.round(p_y / self.dy).astype(int), 0, self.ny - 1)
        hit_grid = self.isBound[iy, ix]
        out_of_bounds = (p_x < 0) | (p_x > self.Lx) | (p_y < 0) | (p_y > self.Ly) | np.isnan(p_x)
        
        valid_thermal_hit = hit_grid & (p_x > 0.5)
        remeshed = False
        
        if sim_mode in ['Thermal', 'Both'] and np.any(valid_thermal_hit):
            v_mag_sq = p_vx[valid_thermal_hit]**2 + p_vy[valid_thermal_hit]**2 + p_vz[valid_thermal_hit]**2
            E_joules = 0.5 * self.m_ion * v_mag_sq * self.macro_weight
            dT_heat = (E_joules / self.C_cell) * self.thermal_accel
            
            # Use Numpy's add.at for this small subset
            np.add.at(self.T_map, (iy[valid_thermal_hit], ix[valid_thermal_hit]), dT_heat)

        if sim_mode in ['Thermal', 'Both']:
            T_bound = self.T_map[self.isBound]
            cooling_factor = (self.emissivity * self.sb_sigma * self.A_cell * self.dt * self.thermal_accel) / self.C_cell
            dT_cool = cooling_factor * (T_bound**4 - 300.0**4)
            
            self.T_map[self.isBound] -= dT_cool

            # Force cast back to kernel float type before sending to Taichi
            self.T_map = self.T_map.astype(_NP_FP)
            
            # --- THERMAL CONDUCTION (TAICHI) ---
            alpha_diff = self.mat_k / (self.mat_rho * self.mat_cp)
            
            dt_thermal = self.dt * self.thermal_accel
            dx_m = self.dx * 1e-3
            dy_m = self.dy * 1e-3
            
            Fo_x = alpha_diff * dt_thermal / (dx_m**2)
            Fo_y = alpha_diff * dt_thermal / (dy_m**2)
            
            max_Fo = 0.2
            scale = 1.0
            if Fo_x > max_Fo or Fo_y > max_Fo:
                scale = max_Fo / max(Fo_x, Fo_y)
                
            Fo_x *= scale
            Fo_y *= scale
            
            # Pass arrays to Taichi for massive 2D finite difference speedup
            thermal_conduction_taichi(
                self.T_map, self.T_map_new, self.isBound.astype(np.int32), 
                self.nx, self.ny, Fo_x, Fo_y
            )
            
            # Swap buffers
            self.T_map[:] = self.T_map_new[:]
            # -------------------------------------------------

            self.T_map[self.isBound] = np.maximum(self.T_map[self.isBound], 300.0) 
            
            needs_remesh = False
            for i, mask in enumerate(self.mask_grids):
                if np.any(mask):
                    self.T_grids[i] = np.mean(self.T_map[mask])

            # --- Cantilever thermal bowing ---
            # Each grid is clamped at Y=Ly (top wall), free at Y=r (aperture).
            # Thermal stress from uniform ΔT on a constrained cantilever produces
            # a tip deflection:  δ = α · ΔT · L² / (2·t)
            # where L = cantilever length, t = grid thickness.
            for i, grid in enumerate(grids):
                dT = self.T_grids[i] - 300.0
                L_cant_m = (self.Ly - grid['r']) * 1e-3   # cantilever length [m]
                t_m = grid['t'] * 1e-3                     # thickness [m]
                if t_m > 0 and L_cant_m > 0:
                    delta_m = self.alpha_thermal * dT * L_cant_m**2 / (2.0 * t_m)
                    new_defl = delta_m * 1e3                # convert back to mm
                else:
                    new_defl = 0.0

                if abs(new_defl - self.grid_deflections[i]) > 0.005:
                    self.grid_deflections[i] = new_defl
                    needs_remesh = True

            if needs_remesh:
                self.build_domain(params)
                remeshed = True

        # --- SECONDARY ELECTRON EMISSION 
        valid_see_hit = hit_grid & (p_x > 0.5)
        
        if np.any(valid_see_hit):
            v_mag_sq = p_vx[valid_see_hit]**2 + p_vy[valid_see_hit]**2 + p_vz[valid_see_hit]**2
            E_eV = (0.5 * self.m_ion * v_mag_sq) / self.q

            # Use user-imported SEE yield spline if available
            see_user = self.lookup_user_cs('SEE', E_eV)
            if see_user is not None:
                gamma = np.clip(see_user, 0.0, 1.0)
            else:
                gamma = np.clip(0.05 + 1e-4 * E_eV, 0.0, 1.0)
            spawn_mask = np.random.rand(len(gamma)) < gamma
            
            if np.any(spawn_mask):
                num_see = np.sum(spawn_mask)
                see_x = (p_x[valid_see_hit][spawn_mask] - p_vx[valid_see_hit][spawn_mask] * self.dt * 1000 * 1.5).astype(np.float64)
                see_y = (p_y[valid_see_hit][spawn_mask] - p_vy[valid_see_hit][spawn_mask] * self.dt * 1000 * 1.5).astype(np.float64)
                
                T_see = 2.0 
                v_see_th = np.sqrt(2 * self.q * T_see / self.m_e)
                see_vx = (np.random.randn(num_see) * v_see_th).astype(np.float64)
                see_vy = (np.random.randn(num_see) * v_see_th).astype(np.float64)
                see_vz = (np.random.randn(num_see) * v_see_th).astype(np.float64)
                
                self._add_electrons(see_x, see_y, see_vx, see_vy, see_vz)
                
                e_x = self.e_x[:self.num_e]
                e_y = self.e_y[:self.num_e]
                e_vx = self.e_vx[:self.num_e]
                e_vy = self.e_vy[:self.num_e]
                e_vz = self.e_vz[:self.num_e]

        # --- EROSION LOGIC (Sputtering)
        is_erosion_hit = hit_grid & (p_x > 0.5)
        if sim_mode in ['Erosion', 'Both'] and np.any(is_erosion_hit):
            E_eV = (0.5 * self.m_ion * (p_vx[is_erosion_hit]**2 + p_vy[is_erosion_hit]**2 + p_vz[is_erosion_hit]**2)) / self.q
            Y_yield = np.minimum(self.sputter_Y_coeff * (np.maximum(E_eV - self.sputter_E_th, 0))**1.5, 1.0)
            np.add.at(self.damage_map, (iy[is_erosion_hit], ix[is_erosion_hit]), Y_yield * params.get('Accel', 1))

            broken_cells = (self.damage_map > params.get('Thresh', 1e5)) & self.isBound
            if np.any(broken_cells):
                self.isBound[broken_cells] = False
                self.damage_map[broken_cells] = 0
                self.build_sparse_matrix() 
                remeshed = True

        # --- PURGING DEAD PARTICLES---
        dead_mask = hit_grid | out_of_bounds
        alive_mask = ~dead_mask
        n_alive = np.sum(alive_mask)
        
        if n_alive < self.num_p:
            self.p_x[:n_alive] = p_x[alive_mask]
            self.p_y[:n_alive] = p_y[alive_mask]
            self.p_vx[:n_alive] = p_vx[alive_mask]
            self.p_vy[:n_alive] = p_vy[alive_mask]
            self.p_vz[:n_alive] = p_vz[alive_mask]
            self.p_isCEX[:n_alive] = p_cex[alive_mask]
            self.num_p = n_alive
            
            p_x = self.p_x[:self.num_p]
            p_y = self.p_y[:self.num_p]
            p_vx = self.p_vx[:self.num_p]
            p_vy = self.p_vy[:self.num_p]
            p_vz = self.p_vz[:self.num_p]
            p_cex = self.p_isCEX[:self.num_p]

        if self.num_e > 0:
            ix_e = np.clip(np.round(e_x / self.dx).astype(int), 0, self.nx - 1)
            iy_e = np.clip(np.round(e_y / self.dy).astype(int), 0, self.ny - 1)
            hit_grid_e = self.isBound[iy_e, ix_e]
            out_e = (e_x < 0) | (e_x > self.Lx) | (e_y < 0) | (e_y > self.Ly) | np.isnan(e_x)
            
            dead_e = hit_grid_e | out_e
            alive_e = ~dead_e
            n_alive_e = np.sum(alive_e)
            
            if n_alive_e < self.num_e:
                self.e_x[:n_alive_e] = e_x[alive_e]
                self.e_y[:n_alive_e] = e_y[alive_e]
                self.e_vx[:n_alive_e] = e_vx[alive_e]
                self.e_vy[:n_alive_e] = e_vy[alive_e]
                self.e_vz[:n_alive_e] = e_vz[alive_e]
                self.num_e = n_alive_e

        # --- E. CEX COLLISIONS (MCC Method — Birdsall / Roy plume model) ---
        # Ref: Birdsall, "Plasma Physics via Computer Simulation"
        #      Roy, analytical neutral plume density profile
        #
        # Neutral density profile (Roy's dissertation, Eq. 3.17):
        #   n_n(r,z) = n_n0 * a * [1 - {1 + (r_T/R)^2}^{-1/2}] * cos(theta)
        #   where a = [1 - 1/sqrt(2)]^{-1},  R = sqrt(r^2 + (z + r_T)^2),
        #   theta = arctan(r / (z + r_T)),  r_T = max transverse domain size
        #
        # Velocity replacement uses Birdsall Maxwellian sampling (M=3):
        #   f_M = 2*(R1 + R2 + R3 - 1.5)  where R_i ~ U(0,1)
        #   v_new = v_th,n * f_M
        if self.num_p > 0:
            primary_mask = (~p_cex) & (p_x >= 1) & (p_x <= self.Lx)
            if np.any(primary_mask):
                n_primary = np.sum(primary_mask)
                px_m = p_x[primary_mask]  # positions in mm
                py_m = p_y[primary_mask]

                # r_T: maximum transverse dimension of the domain [m]
                r_T = self.Ly * 1e-3

                # Convert particle positions to metres for density calc
                # x = beam axis (axial distance from thruster exit ≈ grid region start)
                z_m = (px_m - 1.0) * 1e-3   # axial distance from first grid [m]
                z_m = np.maximum(z_m, 0.0)
                r_m = py_m * 1e-3            # radial distance from axis [m]

                # Roy's analytical neutral density profile
                a_corr = 1.0 / (1.0 - 1.0 / np.sqrt(2.0))  # correction factor
                R_dist = np.sqrt(r_m**2 + (z_m + r_T)**2)
                theta = np.arctan2(r_m, z_m + r_T)
                n_local = params.get('n0', 1e20) * a_corr * (
                    1.0 - 1.0 / np.sqrt(1.0 + (r_T / np.maximum(R_dist, 1e-12))**2)
                ) * np.cos(theta)
                n_local = np.maximum(n_local, 0.0)

                # Collision probability (Birdsall): P = 1 - exp(-n_n * sigma * g * dt)
                v_mag = np.sqrt(p_vx[primary_mask]**2 + p_vy[primary_mask]**2 + p_vz[primary_mask]**2)
                g = np.maximum(v_mag, 1.0)

                # Use user-imported CX cross-section spline if available
                E_eV_cex = (0.5 * self.m_ion * g**2) / self.q
                sigma_user = self.lookup_user_cs('CX', E_eV_cex)
                if sigma_user is not None:
                    sigma = sigma_user
                else:
                    # Fallback: Birdsall analytical formula (Xe+ default)
                    sigma = ((-0.8821 * np.log(g) + 15.1262)**2) * 1e-20

                prob = 1.0 - np.exp(-n_local * sigma * g * self.dt)
                collided = np.random.rand(n_primary) < prob

                if np.any(collided):
                    c_idx = np.where(primary_mask)[0][collided]
                    n_coll = len(c_idx)

                    # Birdsall Maxwellian sampling (M=3):
                    # f_M = 2*(R1 + R2 + R3 - 1.5)
                    neut_vth = np.sqrt(2 * self.kB * params.get('Tn', 300) / self.m_ion)
                    fM_x = 2.0 * (np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5)
                    fM_y = 2.0 * (np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5)
                    fM_z = 2.0 * (np.random.rand(n_coll) + np.random.rand(n_coll) + np.random.rand(n_coll) - 1.5)
                    p_vx[c_idx] = (neut_vth * fM_x).astype(np.float64)
                    p_vy[c_idx] = (neut_vth * fM_y).astype(np.float64)
                    p_vz[c_idx] = (neut_vth * fM_z).astype(np.float64)
                    p_cex[c_idx] = True

        # --- OLD CEX MODEL (uniform density, Gaussian velocity replacement) ---
        # if self.num_p > 0:
        #     primary_mask = (~p_cex) & (p_x >= 1) & (p_x <= self.Lx)
        #     if np.any(primary_mask):
        #         v_mag = np.sqrt(p_vx[primary_mask]**2 + p_vy[primary_mask]**2 + p_vz[primary_mask]**2)
        #         g = np.maximum(v_mag, 1)
        #         sigma = ((-0.8821 * np.log(g) + 15.1262)**2) * 1e-20
        #         prob = 1 - np.exp(-params.get('n0', 1e20) * sigma * g * self.dt)
        #         collided = np.random.rand(np.sum(primary_mask)) < prob
        #         if np.any(collided):
        #             c_idx = np.where(primary_mask)[0][collided]
        #             neut_vth = np.sqrt(2 * self.kB * params.get('Tn', 300) / self.m_XE)
        #             p_vx[c_idx] = (np.random.randn(len(c_idx)) * neut_vth).astype(np.float64)
        #             p_vy[c_idx] = (np.random.randn(len(c_idx)) * neut_vth).astype(np.float64)
        #             p_vz[c_idx] = (np.random.randn(len(c_idx)) * neut_vth).astype(np.float64)
        #             p_cex[c_idx] = True

        return remeshed, min_pot, current_div, self.T_grids

    def get_particle_kinematics(self):
        t_current = self.iteration * self.dt
        
        # Note: Added p_vz and e_vz to the output stack to provide the full 3D velocity vector.
        # This means the output now has 8 columns: [time, x, y, vx, vy, vz, energy_eV, type]
        if self.num_p > 0:
            p_x, p_y = self.p_x[:self.num_p], self.p_y[:self.num_p]
            p_vx, p_vy, p_vz = self.p_vx[:self.num_p], self.p_vy[:self.num_p], self.p_vz[:self.num_p]
            p_cex = self.p_isCEX[:self.num_p]
            v_sq_i = p_vx**2 + p_vy**2 + p_vz**2
            energy_eV_i = (0.5 * self.m_ion * v_sq_i) / self.q
            type_i = p_cex.astype(int) 
            ions = np.column_stack((np.full(self.num_p, t_current), p_x, p_y, p_vx, p_vy, p_vz, energy_eV_i, type_i))
        else:
            ions = np.empty((0, 8))
            
        if self.num_e > 0:
            e_x, e_y = self.e_x[:self.num_e], self.e_y[:self.num_e]
            e_vx, e_vy, e_vz = self.e_vx[:self.num_e], self.e_vy[:self.num_e], self.e_vz[:self.num_e]
            v_sq_e = e_vx**2 + e_vy**2 + e_vz**2
            energy_eV_e = (0.5 * self.m_e * v_sq_e) / self.q
            type_e = np.where(e_x <= 4.0, 2, 3)
            elecs = np.column_stack((np.full(self.num_e, t_current), e_x, e_y, e_vx, e_vy, e_vz, energy_eV_e, type_e))
        else:
            elecs = np.empty((0, 8))
            
        return np.vstack((ions, elecs))