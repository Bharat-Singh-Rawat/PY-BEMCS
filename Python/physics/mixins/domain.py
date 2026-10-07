"""
Domain construction and discrete Laplacian assembly mixin for DigitalTwinSimulator.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from ..backend import (
    _NP_FP, _GPU_POISSON, cp, cp_sp, cp_splu
)
from ..mesh_utils import estimate_automatic_mesh_zones


class DomainMixin:
    """Matrix assembly and physical domain setup methods."""

    def build_sparse_matrix(self):
        periodic_y = getattr(self, 'periodic_y', False)
        N   = self.nx * self.ny
        idx = np.arange(N)
        y_  = idx // self.nx
        x_  = idx  % self.nx

        is_bound   = self.isBound.flatten()
        is_right   = (x_ == self.nx - 1) & ~is_bound

        if periodic_y:
            # Bottom and top non-bound rows become interior with periodic coupling
            is_top       = np.zeros(N, dtype=bool)
            is_bottom    = np.zeros(N, dtype=bool)
            is_per_bot   = (y_ == 0)          & ~is_bound & ~is_right
            is_per_top   = (y_ == self.ny - 1)  & ~is_bound & ~is_right
            is_interior  = ~is_bound & ~is_right & ~is_per_bot & ~is_per_top
        else:
            is_top      = (y_ == self.ny - 1) & ~is_bound & ~is_right
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
        row.append(idx_r); col.append(idx_r - 1); data.append(-np.ones_like(idx_r))

        if not periodic_y:
            # —- Neumann top: V[ny-1, ix] = V[ny-2, ix] —-
            idx_t = idx[is_top]
            row.append(idx_t); col.append(idx_t);         data.append( np.ones_like(idx_t))
            row.append(idx_t); col.append(idx_t - self.nx); data.append(-np.ones_like(idx_t))

            # —- Neumann bottom: V[0, ix] = V[1, ix] —-
            idx_bot = idx[is_bottom]
            row.append(idx_bot); col.append(idx_bot);         data.append( np.ones_like(idx_bot))
            row.append(idx_bot); col.append(idx_bot + self.nx); data.append(-np.ones_like(idx_bot))

        # Finite-volume stencil coefficients for non-uniform x and uniform y
        dx_m = self.x_coords[1:-1] - self.x_coords[:-2]
        dx_p = self.x_coords[2:] - self.x_coords[1:-1]
        hx = 0.5 * (dx_m + dx_p)

        cW = np.zeros(self.nx, dtype=np.float64)
        cE = np.zeros(self.nx, dtype=np.float64)
        cW[1:-1] = (self.dy ** 2) / (hx * dx_m)
        cE[1:-1] = (self.dy ** 2) / (hx * dx_p)
        cP = -(cW + cE + 2.0)

        # —- Standard 5-point interior stencil —-
        idx_in = idx[is_interior]
        x_in   = x_[is_interior]
        row.append(idx_in); col.append(idx_in);          data.append(cP[x_in])
        row.append(idx_in); col.append(idx_in - 1);        data.append(cW[x_in])
        row.append(idx_in); col.append(idx_in + 1);        data.append(cE[x_in])
        row.append(idx_in); col.append(idx_in - self.nx);  data.append(np.ones_like(idx_in))
        row.append(idx_in); col.append(idx_in + self.nx);  data.append(np.ones_like(idx_in))

        if periodic_y:
            # —- Periodic bottom row (iy=0): 5-pt with south neighbour = iy=ny-1 —-
            idx_pb   = idx[is_per_bot]
            x_pb     = x_[is_per_bot]
            col_south = (self.ny - 1) * self.nx + x_pb
            row.append(idx_pb); col.append(idx_pb);         data.append(cP[x_pb])
            row.append(idx_pb); col.append(idx_pb - 1);     data.append(cW[x_pb])
            row.append(idx_pb); col.append(idx_pb + 1);     data.append(cE[x_pb])
            row.append(idx_pb); col.append(idx_pb + self.nx); data.append(np.ones_like(idx_pb)) # north (iy=1)
            row.append(idx_pb); col.append(col_south);      data.append(np.ones_like(idx_pb))  # periodic south

            # —- Periodic top row (iy=ny-1): 5-pt with north neighbour = iy=0 —-
            idx_pt   = idx[is_per_top]
            x_pt     = x_[is_per_top]
            col_north = x_pt  # iy=0
            row.append(idx_pt); col.append(idx_pt);         data.append(cP[x_pt])
            row.append(idx_pt); col.append(idx_pt - 1);     data.append(cW[x_pt])
            row.append(idx_pt); col.append(idx_pt + 1);     data.append(cE[x_pt])
            row.append(idx_pt); col.append(idx_pt - self.nx); data.append(np.ones_like(idx_pt)) # south (iy=ny-2)
            row.append(idx_pt); col.append(col_north);      data.append(np.ones_like(idx_pt))  # periodic north

        row  = np.concatenate(row)
        col  = np.concatenate(col)
        data = np.concatenate(data)

        A = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsc()
        self.laplacian_matrix = A
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
                self.laplacian_matrix_gpu = A_gpu
                self.laplacian_lu_gpu     = cp_splu(A_gpu)
                self.is_bound_mask_gpu    = cp.asarray(self.is_bound_mask)
                self.is_interior_mask_gpu = cp.asarray(self.is_interior_mask)
            except Exception as exc:
                print(f"[Poisson] GPU factorization failed, using CPU: {exc}")
                self.laplacian_lu_gpu = None

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
            if n0_n <= 1e17:
                n_debye = 80
            elif n0_n <= 4e17:
                n_debye = 40
            else:
                n_debye = 30
            self.recommended_upstream_gap_mm = n_debye * debye_mm
        else:
            screen_r = grids[0]['r'] if grids else 0.80
            self.recommended_upstream_gap_mm = 0.75 * screen_r

        ratio_debye = params.get(r"\deltax/debye_length", params.get("dx_over_debye", None))
        if ratio_debye is not None:
            _lambda_D_m = np.sqrt(self.eps0 * Te_up * self.q / (n0_n * self.q**2))
            _lambda_D_mm = _lambda_D_m * 1e3
            _dxy_mm = float(ratio_debye) * _lambda_D_mm
            self.dx = _dxy_mm
            self.dy = _dxy_mm

        user_gap = float(params.get('upstream_gap_mm', 0.0))
        upstream_gap = user_gap if user_gap > 0.0 else self.recommended_upstream_gap_mm
        upstream_gap = max(upstream_gap, self.dx * 2)
        self.upstream_gap_mm = upstream_gap

        grid_Lx = (upstream_gap + total_grid_thickness) if grids else 3.0
        user_Lx = params.get('Lx', None)
        if user_Lx is not None and float(user_Lx) > 0:
            min_Lx = upstream_gap + total_grid_thickness if grids else 1.0
            self.Lx = max(float(user_Lx), min_Lx)
        else:
            self.Lx = grid_Lx

        if self.dx * 1e-3 > debye_length or self.dy * 1e-3 > debye_length:
            warn_msg = (
                f"Grid spacing too large for Debye resolution: "
                f"dx={self.dx:.6f} mm, dy={self.dy:.6f} mm, "
                f"lambda_D={debye_length*1e3:.6f} mm."
            )
            if ratio_debye is not None and float(ratio_debye) > 1.0:
                print(f"[Domain Warning] {warn_msg} Allowed by user override (dx/debye={ratio_debye}).")
            else:
                raise ValueError(
                    f"{warn_msg} Choose dx, dy <= lambda_D or configure \\deltax/debye_length in Advanced Settings."
                )

        plasma_freq = np.sqrt(n0_n * self.q ** 2 / (self.m_ion * self.eps0))
        elet_freq = np.sqrt(n0_n * self.q ** 2 / (self.m_e * self.eps0))
        dt = 2 * 3.14159 / plasma_freq
        dt_e = 2 * 3.14159 / elet_freq
        if dt < self.dt:
            raise ValueError(
                f"Too low time step: "
                f"dt used ={self.dt:.8e} s, dt minimum ={dt:.6e} s, "
            )
        Ti = params.get('Ti', 0.1)
        v_bohm = np.sqrt(self.q_ion * Te_up / self.m_ion)
        v_spread = np.sqrt(self.q_ion * Ti / self.m_ion)
        vmax = v_bohm + 4 * v_spread

        if self.dx * 1e-3 / self.dt < vmax or self.dy * 1e-3 / self.dt < vmax:
            raise ValueError(
                f"Grid spacing and time step too large for velocity resolution: "
                f"dx/dt={self.dx*1e-3/self.dt:.2e} m/s, dy/dt={self.dy*1e-3/self.dt:.2e} m/s, "
                f"vmax={vmax:.2e} m/s. "
                f"Choose dx, dy and dt such that dx/dt >= vmax and dy/dt >= vmax."
            )

        geometry = params.get('geometry', 'half_hole')
        self.geometry = geometry
        self.periodic_y = geometry in ('one_hole', 'two_holes')

        if grids:
            screen_r = grids[0]['r']
            if geometry == 'two_holes':
                grid_Ly = screen_r + 2.0 * screen_r + pitch
            elif geometry == 'one_hole':
                grid_Ly = 3.0 * screen_r
            else:
                grid_Ly = 0.5 * screen_r + 0.30 * pitch
        else:
            grid_Ly = 3.0

        user_Ly = params.get('Ly', None)
        if user_Ly is not None and float(user_Ly) > 0:
            min_Ly = screen_r if grids else 0.5
            self.Ly = max(float(user_Ly), min_Ly)
        else:
            self.Ly = grid_Ly

        if grids:
            x_screen_start = self.upstream_gap_mm
            x_last_grid_end = self.upstream_gap_mm + sum(g['t'] + g['gap'] for g in grids) - grids[-1]['gap']
        else:
            x_screen_start = self.Lx / 3.0
            x_last_grid_end = 2.0 * self.Lx / 3.0

        use_json_mesh_zones = bool(params.get('use_json_mesh_zones', False))

        if use_json_mesh_zones:
            mesh_zones_cfg = params.get('mesh_zones', {})
            presheath_factor = float(mesh_zones_cfg.get('presheath_factor', 1.0))
            optics_factor    = float(mesh_zones_cfg.get('optics_factor', 1.0))
            plume_factor     = float(mesh_zones_cfg.get('plume_factor', 4.0))

            zone_configs = [
                {'name': 'Presheath', 'x_start': 0.0, 'x_end': min(x_screen_start, self.Lx), 'factor': presheath_factor},
                {'name': 'Optics',    'x_start': min(x_screen_start, self.Lx), 'x_end': min(x_last_grid_end, self.Lx), 'factor': optics_factor},
                {'name': 'Plume',     'x_start': min(x_last_grid_end, self.Lx), 'x_end': self.Lx, 'factor': plume_factor},
            ]
        else:
            zone_configs = estimate_automatic_mesh_zones(
                grids=grids,
                upstream_gap_mm=self.upstream_gap_mm,
                Lx=self.Lx,
                dx0=self.dx
            )

        self.zone_configs = zone_configs

        self.x_coords, self.zone_boundaries, self.zone_dx, self.zone_offsets = (
            self._build_nonuniform_x_coords(zone_configs, self.Lx, self.dx)
        )
        self.xpts = self.x_coords
        self.x_pts = self.x_coords
        self.nx = len(self.x_coords)
        self.dx_cells = np.diff(self.x_coords)
        self.dx_min = float(np.min(self.dx_cells))
        self.dx_max = float(np.max(self.dx_cells))

        self.ny = int(self.Ly / self.dy) + 1
        self.dy = self.Ly / (self.ny - 1)
        self.ypts = np.linspace(0, self.Ly, self.ny)
        self.y_coords = self.ypts
        self.y_pts = self.y_coords
        self.dy_cells = np.diff(self.y_coords)

        hx = np.zeros(self.nx, dtype=np.float64)
        hx[0] = 0.5 * self.dx_cells[0]
        hx[1:-1] = 0.5 * (self.dx_cells[:-1] + self.dx_cells[1:])
        hx[-1] = 0.5 * self.dx_cells[-1]
        self.hx_cells = hx
        self.cell_vol_1d = (hx * 1e-3) * (self.dy * 1e-3) * 1e-3
        self.cell_vol_2d = np.tile(self.cell_vol_1d, (self.ny, 1))
        self.cell_area_1d = (hx * 1e-3) * (self.dy * 1e-3)
        self.cell_area_2d = np.tile(self.cell_area_1d, (self.ny, 1))

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
            self.exit_ion_current_step = 0.0
            self.exit_ion_current_avg  = 0.0
            self.entered_optics = 0.0
            self.entered_optics_step = 0.0
            self._last_neut_injected = 0
            self._domain_built = True

        inj_time = params.get("inj_time", 0.0)

        if inj_time > 0.0:
            self.injection_stop_time = inj_time
            self.injection_enabled   = True
        else:
            self.injection_stop_time = None
            self.injection_enabled   = True

        n0         = params.get('n0_plasma', 1e17)
        self.target_ppc = float(params.get('target_ppc', 40.0))
        cell_vol   = self.cell_vol_1d[0]
        entire_bulk_plasma = params.get('entire_bulk_plasma', False)
        bohm_factor = 1.0 if entire_bulk_plasma else 0.61
        self.macro_weight = max(bohm_factor * n0 * cell_vol / self.target_ppc, 1.0)
        self.mask_grids = []
        self.T_grids    = []
        self.isBound.fill(False)
        self.V_fixed.fill(0.0)

        self.energy_history_t.clear()
        self.energy_history_ke_i.clear()
        self.energy_history_ke_e.clear()
        self.energy_history_fe.clear()
        self.energy_history_tot.clear()
        self._prev_total_energy = None

        self.charge_history_t.clear()
        self.charge_history_q_i.clear()
        self.charge_history_q_e.clear()
        self.charge_history_q_b.clear()
        self.charge_history_q_free.clear()
        self.charge_history_q_net.clear()
        self._prev_total_charge = None

        if not hasattr(self, 'grid_deflections') or len(self.grid_deflections) != len(grids):
            self.grid_deflections = [0.0] * len(grids)

        current_x = self.upstream_gap_mm

        if grids:
            screen_r = grids[0]['r']
            if geometry == 'two_holes':
                y_c1 = 1.5 * screen_r
                hole_centers = [y_c1, y_c1 + pitch]
            elif geometry == 'one_hole':
                hole_centers = [1.5 * screen_r]
            else:
                hole_centers = [0.0]
        else:
            hole_centers = []

        self.hole_centers = hole_centers

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
                    dxbow = delta * eta ** 2
                elif geometry == 'one_hole':
                    y_mid = 1.5 * screen_r
                    eta = np.clip(1.0 - np.abs(self.Y - y_mid) / max(self.Ly * 0.5, 1e-6), 0.0, 1.0)
                    dxbow = delta * eta ** 2
                else:
                    Lcant = self.Ly - grid["r"]
                    if Lcant > 0:
                        eta   = np.clip((self.Ly - self.Y) / Lcant, 0.0, 1.0)
                        dxbow = delta * eta ** 2
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

        v_plasma_bound = (grids[0]["V"] + params.get("V_plasma_offset", 20.0)
                          if grids else 1000.0 + params.get("V_plasma_offset", 20.0))
        self.V_fixed[:, 0] = v_plasma_bound
        self.isBound[:, 0] = True

        if not preserve_state:
            self.Tmap[self.isBound] = 300.0
        self.build_sparse_matrix()
        self.recalc_poisson(iterations=30 if not preserve_state else 10, params=params)

        self.grids = grids
        if hasattr(self, 'grid_x_starts') and len(self.grid_x_starts) > 1:
            x_accel_center = 0.5 * (self.grid_x_starts[1] + self.grid_x_ends[1])
            x_idx = self._x_to_ix(x_accel_center)
        else:
            x_idx = self._x_to_ix(self.Lx * 0.5)

        hole_centers = getattr(self, 'hole_centers', [0.0])
        y_c_first = hole_centers[0] if hole_centers else 0.0
        y_idx = int(np.clip(round(y_c_first / self.dy), 0, self.ny - 1))
        self.min_pot = float(self.V[y_idx, x_idx])
        self.saddle_point_potential = self.min_pot
