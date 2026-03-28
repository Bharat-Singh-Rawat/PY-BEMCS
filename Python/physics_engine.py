import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

class DigitalTwinSimulator:
    def __init__(self):
        # Fundamental Constants
        self.dt = 1e-9
        self.q = 1.602e-19
        self.m_XE = 131.293 * 1.6605e-27
        self.kB = 1.380649e-23
        self.eps0 = 8.854e-12
        
        # ARTIFICIAL ELECTRON MASS: 100x lighter than Xenon
        self.m_e = self.m_XE / 100.0 
        
        # --- THERMAL MULTIPHYSICS CONSTANTS ---
        self.macro_weight = 3e5  
        self.alpha_moly = 4.8e-6 
        self.sb_sigma = 5.67e-8  
        self.emissivity = 0.8    
        self.thermal_accel = 1e7 
        
        self.C_cell = 10280 * (0.05e-3)**2 * 1e-3 * 250 
        self.A_cell = 2 * (0.05e-3) * 1e-3 
        
        self.T_screen = 300.0 
        self.T_accel = 300.0 
        
        # Mesh parameters (High Accuracy)
        self.Lx = 8
        self.Ly = 3
        self.dx = 0.01
        self.dy = 0.01
        self.nx = int(self.Lx / self.dx) + 1
        self.ny = int(self.Ly / self.dy) + 1
        
        self.x_pts = np.linspace(0, self.Lx, self.nx)
        self.y_pts = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x_pts, self.y_pts)
        
        self.iteration = 0
        self.T_map = np.full((self.ny, self.nx), 300.0)
        
        # Sparse Matrix References
        self.laplacian_lu = None
        self.is_interior_mask = None
        self.is_bound_mask = None

        self.reset_arrays()

    def reset_arrays(self):
        self.p_x, self.p_y = np.array([]), np.array([])
        self.p_vx, self.p_vy = np.array([]), np.array([])
        self.p_isCEX = np.array([], dtype=bool)
        
        self.e_x, self.e_y = np.array([]), np.array([])
        self.e_vx, self.e_vy = np.array([]), np.array([])
        
        self.V = np.zeros((self.ny, self.nx))
        self.rho = np.zeros((self.ny, self.nx)) 
        self.isBound = np.zeros((self.ny, self.nx), dtype=bool)
        self.V_fixed = np.zeros((self.ny, self.nx))
        self.damage_map = np.zeros((self.ny, self.nx))
        self.Ex, self.Ey = np.zeros((self.ny, self.nx)), np.zeros((self.ny, self.nx))
        self.interp_Ex, self.interp_Ey = None, None

    def build_sparse_matrix(self):
        """Builds the Laplacian Matrix and pre-computes the LU factorization for extreme speed."""
        N = self.nx * self.ny
        idx = np.arange(N)
        y = idx // self.nx
        x = idx % self.nx

        # Define boundary regions
        is_bound = self.isBound.flatten()
        is_right = (x == self.nx - 1) & ~is_bound
        is_top = (y == self.ny - 1) & ~is_bound & ~is_right
        is_bottom = (y == 0) & ~is_bound & ~is_right & ~is_top
        is_interior = ~is_bound & ~is_right & ~is_top & ~is_bottom

        self.is_interior_mask = is_interior
        self.is_bound_mask = is_bound

        row, col, data = [], [], []

        # 1. Fixed Boundaries (Dirichlet)
        idx_b = idx[is_bound]
        row.append(idx_b); col.append(idx_b); data.append(np.ones_like(idx_b))

        # 2. Right Boundary (Neumann dV/dx = 0)
        idx_r = idx[is_right]
        row.append(idx_r); col.append(idx_r); data.append(np.ones_like(idx_r))
        row.append(idx_r); col.append(idx_r - 1); data.append(-np.ones_like(idx_r))

        # 3. Top Boundary (Neumann dV/dy = 0)
        idx_t = idx[is_top]
        row.append(idx_t); col.append(idx_t); data.append(np.ones_like(idx_t))
        row.append(idx_t); col.append(idx_t - self.nx); data.append(-np.ones_like(idx_t))

        # 4. Bottom Boundary (Symmetry axis, dV/dy = 0)
        idx_bot = idx[is_bottom]
        row.append(idx_bot); col.append(idx_bot); data.append(np.ones_like(idx_bot))
        row.append(idx_bot); col.append(idx_bot + self.nx); data.append(-np.ones_like(idx_bot))

        # 5. Interior nodes (Standard 5-point Laplacian stencil)
        idx_in = idx[is_interior]
        row.append(idx_in); col.append(idx_in); data.append(np.full_like(idx_in, -4.0))
        row.append(idx_in); col.append(idx_in - 1); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in + 1); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in - self.nx); data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in + self.nx); data.append(np.ones_like(idx_in))

        # Assemble Sparse Matrix
        row = np.concatenate(row)
        col = np.concatenate(col)
        data = np.concatenate(data)
        
        A = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsc()
        self.laplacian_lu = factorized(A) # Pre-factorize for instant solves

    def build_domain(self, params):
        self.reset_arrays()
        self.iteration = 0

        # --- AUTO-SCALE MACRO WEIGHT FOR OPTIMAL PERFORMANCE ---
        cell_vol = (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3 
        n0 = params.get('n0_plasma', 1e17) 
        target_ppc = 40.0 
        self.macro_weight = max((n0 * cell_vol) / target_ppc, 1e3)

        # --- MESH GENERATION ---
        screen_start = 1.0
        screen_end = screen_start + params['ts']
        accel_start = screen_end + params['gap']
        accel_end = accel_start + params['ta']

        in_screen = (self.X >= screen_start) & (self.X <= screen_end)
        in_accel = (self.X >= accel_start) & (self.X <= accel_end)

        R_screen = params['rs'] + np.maximum(0, self.X - screen_start) * np.tan(np.radians(params['cham_s']))
        R_accel = params['ra'] + np.maximum(0, self.X - accel_start) * np.tan(np.radians(params['cham_a']))

        self.mask_s = in_screen & (self.Y >= R_screen)
        self.isBound[self.mask_s] = True
        self.V_fixed[self.mask_s] = params['Vs']

        self.mask_a = in_accel & (self.Y >= R_accel)
        self.isBound[self.mask_a] = True
        self.V_fixed[self.mask_a] = params['Va']

        self.isBound[:, 0] = True
        self.V_fixed[:, 0] = params['Vs'] + 50
        
        self.T_map[~self.isBound] = 300.0
        
        # Build Sparse Math Operator
        self.build_sparse_matrix()
        
        # Initial settle (30 iterations is enough with a direct solver)
        self.recalc_poisson(iterations=30, params=params)

    def recalc_poisson(self, iterations=5, params=None):
        if self.laplacian_lu is None: return
        
        dx_m2 = (self.dx * 1e-3)**2 
        coeff = dx_m2 / self.eps0
        
        V_plasma = params['Vs'] + 50 if params else 1050 
        Te_up = params.get('Te_up', 3.0) 
        n0 = params.get('n0_plasma', 1e17) 
        
        omega = 0.2 # Under-relaxation for Picard Iterations

        b = np.zeros(self.nx * self.ny)
        V_fixed_flat = self.V_fixed.flatten()

        for _ in range(iterations):
            # Calculate Boltzmann electrons
            rho_e = -self.q * n0 * np.exp((np.minimum(self.V, V_plasma) - V_plasma) / Te_up)
            rho_total = self.rho + rho_e
            rho_flat = rho_total.flatten()

            # Assemble RHS vector (b)
            b.fill(0.0)
            b[self.is_bound_mask] = V_fixed_flat[self.is_bound_mask]
            b[self.is_interior_mask] = -coeff * rho_flat[self.is_interior_mask]

            # Direct Solve (Instant)
            V_new_flat = self.laplacian_lu(b)
            V_new = V_new_flat.reshape((self.ny, self.nx))

            # Apply Under-relaxation
            self.V = (1 - omega) * self.V + omega * V_new

        # Compute Electric Fields
        self.Ey, self.Ex = np.gradient(-self.V, self.dy * 1e-3, self.dx * 1e-3)
        self.interp_Ex = RegularGridInterpolator((self.y_pts, self.x_pts), self.Ex, bounds_error=False, fill_value=0)
        self.interp_Ey = RegularGridInterpolator((self.y_pts, self.x_pts), self.Ey, bounds_error=False, fill_value=0)

    def step(self, params):
        if not np.any(self.Ex): return False, np.nan, np.nan, self.T_screen, self.T_accel
        sim_mode = params.get('sim_mode', 'Both')
        self.iteration += 1
        
        # --- A. INJECT PARTICLES ---
        n0 = params.get('n0_plasma', 1e17) 
        Te_up = params.get('Te_up', 3.0)
        v_bohm = np.sqrt(self.q * Te_up / self.m_XE)
        
        injection_area = (params['rs'] - 0.05) * 1e-3 * 1.0 
        I_ion = self.q * 0.61 * n0 * v_bohm * injection_area 
        
        charge_per_macro = self.q * self.macro_weight
        num_inject_float = (I_ion * self.dt) / charge_per_macro
        
        num_inject = int(num_inject_float) + (1 if np.random.rand() < (num_inject_float % 1) else 0)

        if num_inject > 0:
            new_y = np.random.uniform(0.02, params['rs'] - 0.05, num_inject)
            new_x = np.full(num_inject, 0.1)
            v_spread = np.sqrt(self.q * params.get('Ti', 0.1) / self.m_XE)

            self.p_x = np.concatenate((self.p_x, new_x))
            self.p_y = np.concatenate((self.p_y, new_y))
            self.p_vx = np.concatenate((self.p_vx, np.full(num_inject, v_bohm) + np.random.randn(num_inject) * v_spread))
            self.p_vy = np.concatenate((self.p_vy, np.random.randn(num_inject) * v_spread))
            self.p_isCEX = np.concatenate((self.p_isCEX, np.zeros(num_inject, dtype=bool)))

        # --- DYNAMIC NEUTRALIZER CONTROL ---
        num_e = int(params.get('neut_rate', 30))
        Te_eV = params.get('Te', 5.0)

        if num_e > 0:
            new_ey = np.random.rand(num_e) * self.Ly
            new_ex = np.full(num_e, self.Lx - 0.1)
            v_e_th = np.sqrt(2 * self.q * Te_eV / self.m_e)
            
            self.e_x = np.concatenate((self.e_x, new_ex))
            self.e_y = np.concatenate((self.e_y, new_ey))
            self.e_vx = np.concatenate((self.e_vx, -np.abs(np.random.randn(num_e)) * v_e_th - v_e_th*0.2)) 
            self.e_vy = np.concatenate((self.e_vy, np.random.randn(num_e) * v_e_th))

        # --- B. POISSON & SPACE CHARGE ---
        self.rho.fill(0.0)
        cell_vol = (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3 
        charge_per_particle = self.q * self.macro_weight
        
        if len(self.p_x) > 0:
            ix_rho = np.clip(np.round(self.p_x / self.dx).astype(int), 1, self.nx - 2)
            iy_rho = np.clip(np.round(self.p_y / self.dy).astype(int), 1, self.ny - 2)
            np.add.at(self.rho, (iy_rho, ix_rho), charge_per_particle / cell_vol)
            
        if len(self.e_x) > 0:
            ix_rho_e = np.clip(np.round(self.e_x / self.dx).astype(int), 1, self.nx - 2)
            iy_rho_e = np.clip(np.round(self.e_y / self.dy).astype(int), 1, self.ny - 2)
            np.add.at(self.rho, (iy_rho_e, ix_rho_e), -charge_per_particle / cell_vol)

        if self.iteration % 2 == 0:
            self.recalc_poisson(iterations=5, params=params)

        # --- C. PUSH PARTICLES ---
        pts = np.column_stack((self.p_y, self.p_x))
        Ex_p, Ey_p = self.interp_Ex(pts), self.interp_Ey(pts)
        self.p_vx += (self.q / self.m_XE) * Ex_p * self.dt
        self.p_vy += (self.q / self.m_XE) * Ey_p * self.dt
        self.p_x += self.p_vx * self.dt * 1000
        self.p_y += self.p_vy * self.dt * 1000

        if len(self.e_x) > 0:
            pts_e = np.column_stack((self.e_y, self.e_x))
            Ex_e, Ey_e = self.interp_Ex(pts_e), self.interp_Ey(pts_e)
            self.e_vx += (-self.q / self.m_e) * Ex_e * self.dt
            self.e_vy += (-self.q / self.m_e) * Ey_e * self.dt
            self.e_x += self.e_vx * self.dt * 1000
            self.e_y += self.e_vy * self.dt * 1000

        max_grid_x = 1.0 + params['ts'] + params['gap'] + params['ta']
        post_grid = (~self.p_isCEX) & (self.p_x > max_grid_x)
        current_div = np.percentile(np.abs(np.arctan2(self.p_vy[post_grid], self.p_vx[post_grid])) * 180 / np.pi, 95) if np.sum(post_grid) > 5 else np.nan
        min_pot = np.min(self.V[0, :])

        # --- D. 2D THERMAL MULTIPHYSICS & HIT DETECTION ---
        ix = np.clip(np.round(self.p_x / self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip(np.round(self.p_y / self.dy).astype(int), 0, self.ny - 1)
        hit_grid = self.isBound[iy, ix]
        out_of_bounds = (self.p_x < 0) | (self.p_x > self.Lx) | (self.p_y < 0) | (self.p_y > self.Ly) | np.isnan(self.p_x)
        
        valid_thermal_hit = hit_grid & (self.p_x > 0.5)
        remeshed = False
        
        if sim_mode in ['Thermal', 'Both'] and np.any(valid_thermal_hit):
            v_mag_sq = self.p_vx[valid_thermal_hit]**2 + self.p_vy[valid_thermal_hit]**2
            E_joules = 0.5 * self.m_XE * v_mag_sq * self.macro_weight
            dT_heat = (E_joules / self.C_cell) * self.thermal_accel
            np.add.at(self.T_map, (iy[valid_thermal_hit], ix[valid_thermal_hit]), dT_heat)

        if sim_mode in ['Thermal', 'Both']:
            T_bound = self.T_map[self.isBound]
            cooling_factor = (self.emissivity * self.sb_sigma * self.A_cell * self.dt * self.thermal_accel) / self.C_cell
            dT_cool = cooling_factor * (T_bound**4 - 300.0**4)
            
            self.T_map[self.isBound] -= dT_cool
            self.T_map[self.isBound] = np.maximum(self.T_map[self.isBound], 300.0) 
            
            if hasattr(self, 'mask_s') and np.any(self.mask_s): 
                self.T_screen = np.mean(self.T_map[self.mask_s])
            if hasattr(self, 'mask_a') and np.any(self.mask_a): 
                self.T_accel = np.mean(self.T_map[self.mask_a])

            delta_gap = self.alpha_moly * params['gap'] * (self.T_accel - 300) 
            if abs(delta_gap) > 0.05:
                params['gap'] -= delta_gap 
                self.build_domain(params) 
                remeshed = True

        # --- SECONDARY ELECTRON EMISSION (MOLYBDENUM) ---
        valid_see_hit = hit_grid & (self.p_x > 0.5)
        
        if np.any(valid_see_hit):
            v_mag_sq = self.p_vx[valid_see_hit]**2 + self.p_vy[valid_see_hit]**2
            E_eV = (0.5 * self.m_XE * v_mag_sq) / self.q
            
            gamma = np.clip(0.05 + 1e-4 * E_eV, 0.0, 1.0)
            spawn_mask = np.random.rand(len(gamma)) < gamma
            
            if np.any(spawn_mask):
                num_see = np.sum(spawn_mask)
                see_x = self.p_x[valid_see_hit][spawn_mask] - self.p_vx[valid_see_hit][spawn_mask] * self.dt * 1000 * 1.5
                see_y = self.p_y[valid_see_hit][spawn_mask] - self.p_vy[valid_see_hit][spawn_mask] * self.dt * 1000 * 1.5
                
                T_see = 2.0 
                v_see_th = np.sqrt(2 * self.q * T_see / self.m_e)
                see_vx = np.random.randn(num_see) * v_see_th
                see_vy = np.random.randn(num_see) * v_see_th
                
                self.e_x = np.concatenate((self.e_x, see_x))
                self.e_y = np.concatenate((self.e_y, see_y))
                self.e_vx = np.concatenate((self.e_vx, see_vx))
                self.e_vy = np.concatenate((self.e_vy, see_vy))

        # --- EROSION LOGIC ---
        is_erosion_hit = hit_grid & (self.p_x > 0.5)
        
        if sim_mode in ['Erosion', 'Both'] and np.any(is_erosion_hit):
            E_eV = (0.5 * self.m_XE * (self.p_vx[is_erosion_hit]**2 + self.p_vy[is_erosion_hit]**2)) / self.q
            
            Y_yield = np.where(E_eV > 30, 1.05e-4 * (E_eV - 30)**1.5, 0)
            Y_yield = np.clip(Y_yield, 0, 1.0) 
            
            np.add.at(self.damage_map, (iy[is_erosion_hit], ix[is_erosion_hit]), Y_yield * params['Accel'])

            broken_cells = (self.damage_map > params['Thresh']) & self.isBound
            if np.any(broken_cells):
                self.isBound[broken_cells] = False
                self.damage_map[broken_cells] = 0
                self.build_sparse_matrix() 
                remeshed = True

        # Purge dead Ions
        dead_mask = hit_grid | out_of_bounds
        self.p_x, self.p_y = self.p_x[~dead_mask], self.p_y[~dead_mask]
        self.p_vx, self.p_vy = self.p_vx[~dead_mask], self.p_vy[~dead_mask]
        self.p_isCEX = self.p_isCEX[~dead_mask]
        
        # Purge dead Electrons
        if len(self.e_x) > 0:
            ix_e = np.clip(np.round(self.e_x / self.dx).astype(int), 0, self.nx - 1)
            iy_e = np.clip(np.round(self.e_y / self.dy).astype(int), 0, self.ny - 1)
            hit_grid_e = self.isBound[iy_e, ix_e]
            out_e = (self.e_x < 0) | (self.e_x > self.Lx) | (self.e_y < 0) | (self.e_y > self.Ly) | np.isnan(self.e_x)
            
            dead_e = hit_grid_e | out_e
            self.e_x, self.e_y = self.e_x[~dead_e], self.e_y[~dead_e]
            self.e_vx, self.e_vy = self.e_vx[~dead_e], self.e_vy[~dead_e]

        # --- E. CEX COLLISIONS ---
        primary_mask = (~self.p_isCEX) & (self.p_x >= 1) & (self.p_x <= 8.0)
        if np.any(primary_mask):
            v_mag = np.sqrt(self.p_vx[primary_mask]**2 + self.p_vy[primary_mask]**2)
            g = np.maximum(v_mag, 1)
            sigma = ((-0.8821 * np.log(g) + 15.1262)**2) * 1e-20
            prob = 1 - np.exp(-params['n0'] * sigma * g * self.dt)
            collided = np.random.rand(np.sum(primary_mask)) < prob
            
            if np.any(collided):
                c_idx = np.where(primary_mask)[0][collided]
                neut_vth = np.sqrt(2 * self.kB * params['Tn'] / self.m_XE)
                self.p_vx[c_idx] = np.random.randn(len(c_idx)) * neut_vth
                self.p_vy[c_idx] = np.random.randn(len(c_idx)) * neut_vth
                self.p_isCEX[c_idx] = True

        return remeshed, min_pot, current_div, self.T_screen, self.T_accel

    # --- UPDATED: KINEMATICS DATA EXTRACTOR (IONS + ELECTRONS) ---
    def get_particle_kinematics(self):
        """
        Extracts a snapshot of all particles (Ions + Electrons).
        Columns: [Time(s), Z(mm), R(mm), Vz(m/s), Vr(m/s), Energy(eV), Particle_Type]
        Types: 0=Primary Ion, 1=CEX Ion, 2=Grid Electron (SEE), 3=Plume Electron (Neut)
        """
        t_current = self.iteration * self.dt
        
        # --- 1. Process Ions ---
        if len(self.p_x) > 0:
            v_sq_i = self.p_vx**2 + self.p_vy**2
            energy_eV_i = (0.5 * self.m_XE * v_sq_i) / self.q
            type_i = self.p_isCEX.astype(int) # 0 for Primary, 1 for CEX
            
            ions = np.column_stack((
                np.full(len(self.p_x), t_current),
                self.p_x, self.p_y, self.p_vx, self.p_vy, energy_eV_i, type_i
            ))
        else:
            ions = np.empty((0, 7))
            
        # --- 2. Process Electrons ---
        if len(self.e_x) > 0:
            v_sq_e = self.e_vx**2 + self.e_vy**2
            energy_eV_e = (0.5 * self.m_e * v_sq_e) / self.q
            
            # Assign Type 2 for SEE (x <= 4mm) and Type 3 for Neutralizer (x > 4mm)
            type_e = np.where(self.e_x <= 4.0, 2, 3)
            
            elecs = np.column_stack((
                np.full(len(self.e_x), t_current),
                self.e_x, self.e_y, self.e_vx, self.e_vy, energy_eV_e, type_e
            ))
        else:
            elecs = np.empty((0, 7))
            
        # --- 3. Combine and Return ---
        snapshot = np.vstack((ions, elecs))
        return snapshot