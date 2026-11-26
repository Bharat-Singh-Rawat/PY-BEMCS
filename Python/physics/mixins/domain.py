"""
Domain construction and discrete Laplacian assembly mixin for DigitalTwinSimulator.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from ..backend import (
    _NP_FP, _GPU_POISSON, cp, cp_sp, cp_splu
)
from ..mesh_utils import (
    estimate_automatic_mesh_zones,
    compute_efield_constrained_spacing,
    evaluate_mesh_efield_variation,
    compute_debye_upstream_gap,
)
from ..hardware import (
    detect_hardware_profile,
    estimate_mesh_memory_gb,
)


class DomainMixin:
    """Matrix assembly and physical domain setup methods."""

    def build_sparse_matrix(self):
        periodic_y = getattr(self, 'periodic_y', False)
        is_axisymmetric = getattr(self, 'is_axisymmetric', False)
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
            is_axis_bot  = np.zeros(N, dtype=bool)
            is_interior  = ~is_bound & ~is_right & ~is_per_bot & ~is_per_top
        elif is_axisymmetric:
            is_top       = (y_ == self.ny - 1) & ~is_bound & ~is_right
            is_bottom    = np.zeros(N, dtype=bool)
            is_axis_bot  = (y_ == 0) & ~is_bound & ~is_right
            is_interior  = ~is_bound & ~is_right & ~is_top & ~is_axis_bot
            is_per_bot   = np.zeros(N, dtype=bool)
            is_per_top   = np.zeros(N, dtype=bool)
        else:
            is_top      = (y_ == self.ny - 1) & ~is_bound & ~is_right
            is_bottom   = (y_ == 0)         & ~is_bound & ~is_right & ~is_top
            is_interior = ~is_bound & ~is_right & ~is_top & ~is_bottom
            is_per_bot  = np.zeros(N, dtype=bool)
            is_per_top  = np.zeros(N, dtype=bool)
            is_axis_bot = np.zeros(N, dtype=bool)

        self.is_interior_mask  = is_interior | is_axis_bot
        self.is_bound_mask     = is_bound
        # All non-Dirichlet, non-Neumann-right cells get rho as RHS source
        self.is_rhs_rho_mask   = is_interior | is_axis_bot | is_per_bot | is_per_top

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

            if not is_axisymmetric:
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

        # —- Interior stencil —-
        idx_in = idx[is_interior]
        x_in   = x_[is_interior]
        y_in   = y_[is_interior]

        if is_axisymmetric:
            j_in = y_in.astype(np.float64)
            cS_in = 1.0 - 0.5 / j_in
            cN_in = 1.0 + 0.5 / j_in
            row.append(idx_in); col.append(idx_in);          data.append(cP[x_in])
            row.append(idx_in); col.append(idx_in - 1);        data.append(cW[x_in])
            row.append(idx_in); col.append(idx_in + 1);        data.append(cE[x_in])
            row.append(idx_in); col.append(idx_in - self.nx);  data.append(cS_in)
            row.append(idx_in); col.append(idx_in + self.nx);  data.append(cN_in)

            # —- On-axis (j=0) cylindrical FVM row —-
            if np.any(is_axis_bot):
                idx_ax = idx[is_axis_bot]
                x_ax   = x_[is_axis_bot]
                cP_ax  = -(cW[x_ax] + cE[x_ax] + 4.0)
                row.append(idx_ax); col.append(idx_ax);          data.append(cP_ax)
                row.append(idx_ax); col.append(idx_ax - 1);        data.append(cW[x_ax])
                row.append(idx_ax); col.append(idx_ax + 1);        data.append(cE[x_ax])
                row.append(idx_ax); col.append(idx_ax + self.nx);  data.append(np.full_like(idx_ax, 4.0, dtype=np.float64))
        else:
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
        import copy
        self._stop_requested = False
        if isinstance(params, dict):
            self._last_params = copy.deepcopy(params)
        if params.get("beam_mass_amu") is not None and float(params["beam_mass_amu"]) > 0:
            self.m_ion = float(params["beam_mass_amu"]) * 1.6605e-27
            self.m_XE = self.m_ion
        if params.get("beam_charge_state") is not None and int(params["beam_charge_state"]) > 0:
            self.Z_ion = int(params["beam_charge_state"])
            self.q_ion = self.Z_ion * self.q
        if params.get("m_e_ratio") is not None and float(params["m_e_ratio"]) > 0:
            self.m_e = self.m_ion / float(params["m_e_ratio"])
        if isinstance(params.get("mat_props"), dict) and params["mat_props"]:
            self.set_material(props=params["mat_props"])
        elif params.get("mat_name") and hasattr(self, "MATERIAL_PRESETS") and params["mat_name"] in self.MATERIAL_PRESETS:
            self.set_material(name=params["mat_name"])

        grids   = params.get("grids", [])
        if grids:
            total_grid_thickness = sum(g['t'] + g['gap'] for g in grids)
        else:
            total_grid_thickness = 0.0

        pitch   = params.get("pitch_mm", params.get("discharge_chamber", {}).get("pitch_mm", 0.0) if isinstance(params.get("discharge_chamber"), dict) else 0.0)
        Te_up   = float(params.get('Te_up', 3.0))

        n0_n    = float(params.get("n0_plasma", params.get("n0", 1e17)))
        self.n0_plasma = n0_n
        self.n0 = n0_n
        self.Te_up = Te_up
        entire_bulk_plasma = params.get("entire_bulk_plasma", False)

        # Debye length is always needed for grid-spacing validation
        debye_length = np.sqrt(self.eps0 * Te_up / (self.q * n0_n * 0.61))
        debye_mm = debye_length * 1e3   # Debye length in mm
        self.debye_length_mm = float(debye_mm)

        # Physics-based recommended upstream gap depending on simulation mode
        if entire_bulk_plasma:
            self.recommended_upstream_gap_mm = compute_debye_upstream_gap(n0_n, Te_up)
        else:
            screen_r = grids[0]['r'] if grids else 0.80
            self.recommended_upstream_gap_mm = 0.75 * screen_r

        ratio_debye = params.get(r"\deltax/debye_length", params.get("dx_over_debye", None))
        enforce_efield = bool(params.get("enforce_efield_mesh_limit", True))
        max_jump_allowed = float(params.get("max_efield_jump", 0.10))

        is_refining = getattr(self, "_efield_refinement_active", False)
        _dxy_mm = getattr(self, "_original_dxy_mm", None)

        if not is_refining and not preserve_state:
            if ratio_debye is not None:
                _lambda_D_m = np.sqrt(self.eps0 * Te_up * self.q / (n0_n * self.q**2))
                _lambda_D_mm = _lambda_D_m * 1e3
                _dxy_mm = float(ratio_debye) * _lambda_D_mm
                self._original_dxy_mm = _dxy_mm
                if enforce_efield and grids:
                    dx_efield, dy_efield = compute_efield_constrained_spacing(
                        grids=grids,
                        n0=n0_n,
                        Te_up=Te_up,
                        ratio_debye=float(ratio_debye),
                        max_rel_jump=max_jump_allowed,
                    )
                    self.dx = min(_dxy_mm, dx_efield)
                    self.dy = min(_dxy_mm, dy_efield)
                else:
                    self.dx = _dxy_mm
                    self.dy = _dxy_mm
            elif enforce_efield and grids:
                dx_efield, dy_efield = compute_efield_constrained_spacing(
                    grids=grids,
                    n0=n0_n,
                    Te_up=Te_up,
                    ratio_debye=0.8,
                    max_rel_jump=max_jump_allowed,
                )
                self.dx = dx_efield
                self.dy = dy_efield

        user_gap = float(params.get('upstream_gap_mm', 0.0))
        upstream_gap = user_gap if user_gap > 0.0 else self.recommended_upstream_gap_mm
        is_debye_gap = entire_bulk_plasma or (
            isinstance(params.get('upstream_gap_settings'), dict)
            and params['upstream_gap_settings'].get('method') == 'ratio'
            and params['upstream_gap_settings'].get('ratio_base') == 'debye'
        )
        if is_debye_gap and upstream_gap > 4.0:
            print(f"[Warning] Debye-evaluated upstream gap ({upstream_gap:.3f} mm) exceeds 4 mm limit. Clamped to 4.0 mm.")
            upstream_gap = 4.0
        upstream_gap = max(upstream_gap, self.dx * 2)
        self.upstream_gap_mm = upstream_gap

        x_last_exit = upstream_gap + (sum(g['t'] + g['gap'] for g in grids) - grids[-1]['gap']) if grids else 1.0
        min_Lx = x_last_exit + (self.dx * 2.0)
        grid_Lx = (upstream_gap + total_grid_thickness) if grids else 3.0
        user_Lx = params.get('Lx', None)
        if user_Lx is not None and float(user_Lx) > 0:
            self.Lx = max(float(user_Lx), min_Lx)
        else:
            self.Lx = grid_Lx

        # Optional downstream plume truncation to avoid excessive vacuum mesh cells
        do_truncate = bool(params.get('truncate_downstream_plume', False) or getattr(self, 'truncate_downstream_plume', False))
        self.truncate_downstream_plume = do_truncate
        if grids and do_truncate:
            r_accel_bound = grids[-1]['r']
            max_plume = float(params.get('max_plume_length_mm', 2.5 * r_accel_bound))
            trunc_Lx = round(x_last_exit + max_plume, 3)
            if trunc_Lx < self.Lx:
                print(f"[Domain Plume] Truncating downstream plume Lx from {self.Lx:.3f} mm to {trunc_Lx:.3f} mm (plume length={max_plume:.2f} mm).")
                self.Lx = max(trunc_Lx, min_Lx)

        if 'div_plane_x' in params and params['div_plane_x'] is not None:
            self.div_plane_x = float(params['div_plane_x'])
        if 'div_method' in params and params['div_method'] is not None:
            self.div_method = str(params['div_method'])
        if 'div_percentile' in params and params['div_percentile'] is not None:
            self.div_percentile = float(params['div_percentile'])
        if 'twiss_n_sigma' in params and params['twiss_n_sigma'] is not None:
            self.twiss_n_sigma = float(params['twiss_n_sigma'])

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

        user_dt = params.get('dt', None)
        if user_dt is not None and float(user_dt) > 0:
            self.dt = float(user_dt)
        elif params.get('dt_ns') is not None and float(params.get('dt_ns')) > 0:
            self.dt = float(params.get('dt_ns')) * 1e-9

        plasma_freq = np.sqrt(n0_n * self.q ** 2 / (self.m_ion * self.eps0))
        elet_freq   = np.sqrt(n0_n * self.q ** 2 / (self.m_e * self.eps0))
        dt_max_ion  = 2.0 * np.pi / plasma_freq
        self.dt_max_electron = 2.0 * np.pi / elet_freq

        if self.dt > dt_max_ion:
            raise ValueError(
                f"Time step too large for ion plasma frequency: "
                f"dt used = {self.dt:.4e} s exceeds max allowable dt = {dt_max_ion:.4e} s "
                f"(f_pi = {plasma_freq / (2.0 * np.pi):.2e} Hz)."
            )
        Ti = params.get('Ti', 0.1)
        v_bohm = np.sqrt(self.q_ion * Te_up / self.m_ion)
        v_spread = np.sqrt(self.q_ion * Ti / self.m_ion)
        vmax = v_bohm + 4 * v_spread

        if self.dx * 1e-3 / self.dt < vmax or self.dy * 1e-3 / self.dt < vmax:
            dt_cfl = 0.95 * min(self.dx * 1e-3, self.dy * 1e-3) / vmax
            old_dt = self.dt
            self.dt = min(self.dt, dt_cfl)
            print(
                f"[Domain CFL] Grid spacing dx={self.dx:.4f} mm requires finer time step: "
                f"dx/dt was {self.dx*1e-3/old_dt:.2e} m/s < vmax ({vmax:.2e} m/s). "
                f"Automatically adapted dt from {old_dt:.2e} s to {self.dt:.2e} s to satisfy CFL velocity resolution."
            )

        coord_sys = str(params.get('coordinate_system', getattr(self, 'coordinate_system', 'cartesian'))).strip().lower()
        self.is_axisymmetric = coord_sys in ('axisymmetric', 'rz', 'r-z')
        self.coordinate_system = 'axisymmetric' if self.is_axisymmetric else 'cartesian'

        geometry = params.get('geometry', 'half_hole')
        if self.is_axisymmetric:
            # Axisymmetric single-aperture formulation requires on-axis symmetry at y = 0
            geometry = 'half_hole'

        self.geometry = geometry
        self.periodic_y = geometry in ('one_hole', 'two_holes')
        self.reflect_y0 = (geometry == 'half_hole')
        half_hole_top_bc = str(params.get('half_hole_top_bc', 'reflect')).strip().lower()
        self.reflect_y_top = (geometry == 'half_hole' and half_hole_top_bc == 'reflect')

        if grids:
            screen_r = grids[0]['r']
            pitch = params.get("pitch_mm", params.get("discharge_chamber", {}).get("pitch_mm", 0.0) if isinstance(params.get("discharge_chamber"), dict) else 0.0)
            if not pitch:
                pitch = 3.0

            user_hdist = params.get('hole_boundary_dist_mm', params.get('hole_to_boundary_mm', 0.0))
            if user_hdist is not None and float(user_hdist) > 0.0:
                d_bound = float(user_hdist)
            elif self.is_axisymmetric and str(params.get('r_cell_mode', '')).strip().lower() == 'wigner_seitz':
                d_bound = np.sqrt(np.sqrt(3.0) / (2.0 * np.pi)) * pitch
            else:
                d_bound = 0.5 * (screen_r + pitch)
            self.hole_boundary_dist_mm = d_bound

            if geometry == 'two_holes':
                grid_Ly = 4.0 * d_bound
            elif geometry == 'one_hole':
                grid_Ly = 2.0 * d_bound
            else:
                grid_Ly = d_bound
        else:
            grid_Ly = 3.0
            self.hole_boundary_dist_mm = 1.5

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
        explicit_zone_configs = params.get('zone_configs', None) if preserve_state else None

        if (
            not use_json_mesh_zones
            and isinstance(explicit_zone_configs, list)
            and len(explicit_zone_configs) > 0
            and abs(float(explicit_zone_configs[-1].get('x_end', 0.0)) - self.Lx) <= 1e-3
        ):
            import copy
            zone_configs = copy.deepcopy(explicit_zone_configs)
        elif use_json_mesh_zones:
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
                dx0=self.dx,
                max_rel_jump=max_jump_allowed,
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

        self.ny = max(12, int(self.Ly / self.dy) + 1)
        self.dy = self.Ly / (self.ny - 1)
        self.ypts = np.linspace(0, self.Ly, self.ny)
        self.y_coords = self.ypts
        self.y_pts = self.y_coords
        self.x_coords_m = self.x_coords * 1e-3
        self.y_coords_m = self.y_coords * 1e-3
        self.two_dy_m = 2.0 * self.dy * 1e-3
        self.dy_cells = np.diff(self.y_coords)

        hx = np.zeros(self.nx, dtype=np.float64)
        hx[0] = 0.5 * self.dx_cells[0]
        hx[1:-1] = 0.5 * (self.dx_cells[:-1] + self.dx_cells[1:])
        hx[-1] = 0.5 * self.dx_cells[-1]
        self.hx_cells = hx
        hx_m = hx * 1e-3
        dy_m = self.dy * 1e-3

        if self.is_axisymmetric:
            r_coords_m = self.y_coords * 1e-3
            V_rad = np.zeros(self.ny, dtype=np.float64)
            # Node 0 (on axis r=0) half-cell cross-sectional area spans [0, dy_m / 2]
            V_rad[0] = np.pi * (0.5 * dy_m) ** 2
            # Interior nodes j=1..ny-2 span [r_j - dy_m/2, r_j + dy_m/2]
            if self.ny > 2:
                V_rad[1:-1] = 2.0 * np.pi * r_coords_m[1:-1] * dy_m
            # Top boundary node j=ny-1 spans [r_max - dy_m/2, r_max]
            V_rad[-1] = np.pi * (r_coords_m[-1] ** 2 - (r_coords_m[-1] - 0.5 * dy_m) ** 2)
            self.cvol_r_1d = V_rad
            self.cvol_x_1d = hx_m
            self.cell_vol_2d = np.outer(V_rad, hx_m)
            self.cell_vol_1d = self.cell_vol_2d[0, :].copy()
            self.cell_area_1d = (hx * 1e-3) * (self.dy * 1e-3)
            self.cell_area_2d = np.tile(self.cell_area_1d, (self.ny, 1))
        else:
            self.cvol_r_1d = np.full(self.ny, dy_m, dtype=np.float64)
            self.cvol_x_1d = hx_m * 1e-3
            self.cell_vol_1d = (hx * 1e-3) * (self.dy * 1e-3) * 1e-3
            self.cell_vol_2d = np.tile(self.cell_vol_1d, (self.ny, 1))
            self.cell_area_1d = (hx * 1e-3) * (self.dy * 1e-3)
            self.cell_area_2d = np.tile(self.cell_area_1d, (self.ny, 1))

        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        self._recompute_cell_constants()
        self._update_taichi_mesh_constants()

        if not preserve_state:
            self.Tmap       = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
            self.T_map      = self.Tmap
            self.T_map_new  = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
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
            self.first_grid_entry_iter = None
            self._last_neut_injected = 0
            self._plume_remesh_deferred = False
            from ..mesh_utils import compute_plume_mean_dx
            base_p_dx = compute_plume_mean_dx(self, zone_configs)
            self.plume_history_iterations = [0]
            self.plume_history_dx = [base_p_dx]
            self.plume_dx_history = [(0, base_p_dx)]
            self.zone_history_dx = {
                str(z.get('name', 'Zone')): [float(self.dx * float(z.get('factor', 1.0)))]
                for z in (zone_configs or [])
            }
            far_z = next((z for z in zone_configs if z['name'] == 'Far_Plume'), None)
            self._current_plume_factor = float(far_z['factor']) if far_z else 4.0
            self._domain_built = True
        else:
            # Ensure 2D spatial grid arrays match new (ny, nx) if mesh shape changed
            if not hasattr(self, 'isBound') or self.isBound.shape != (self.ny, self.nx):
                self.V        = np.zeros((self.ny, self.nx), dtype=np.float64)
                self.rho      = np.zeros((self.ny, self.nx), dtype=_NP_FP)
                self.isBound  = np.zeros((self.ny, self.nx), dtype=bool)
                self.V_fixed  = np.zeros((self.ny, self.nx), dtype=np.float64)
                self.damage_map = np.zeros((self.ny, self.nx), dtype=np.float64)
                self.eroded_depth = np.zeros((self.ny, self.nx), dtype=np.float64)
                self.Ex       = np.zeros((self.ny, self.nx), dtype=_NP_FP)
                self.Ey       = np.zeros((self.ny, self.nx), dtype=_NP_FP)
                self.Bx       = np.zeros((self.ny, self.nx), dtype=_NP_FP)
                self.By       = np.zeros((self.ny, self.nx), dtype=_NP_FP)
                self.Bz       = np.zeros((self.ny, self.nx), dtype=_NP_FP)
                self.Tmap     = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
                self.T_map    = self.Tmap
                self.T_map_new = np.full((self.ny, self.nx), 300.0, dtype=_NP_FP)
                self.accum_ppc_map   = np.zeros((self.ny, self.nx), dtype=np.float64)
                self.accum_cex_ppc_map = np.zeros((self.ny, self.nx), dtype=np.float64)
                self.current_ppc_map = np.zeros((self.ny, self.nx), dtype=int)
                self.current_cex_ppc_map = np.zeros((self.ny, self.nx), dtype=int)
                self.current_prim_ppc_map = np.zeros((self.ny, self.nx), dtype=int)
            self._domain_built = True

        inj_time = params.get("inj_time", 0.0)

        if inj_time > 0.0:
            self.injection_stop_time = inj_time
            self.injection_enabled   = True
        else:
            self.injection_stop_time = None
            self.injection_enabled   = True

        n0 = float(params.get('n0_plasma', getattr(self, 'n0_plasma', 1e17)))
        self.n0_plasma = n0
        self.n0 = n0
        self.target_ppc = float(params.get('target_ppc', 40.0))
        entire_bulk_plasma = params.get('entire_bulk_plasma', False)
        bohm_factor = 1.0 if entire_bulk_plasma else 0.61

        user_inj_y = params.get('inj_y_length_mm', params.get('inj_y_length', None))
        geometry = getattr(self, 'geometry', 'half_hole')
        max_inj_y = (0.5 * self.Ly) if geometry == 'one_hole' else (self.Ly - self.dy)
        if user_inj_y is not None and float(user_inj_y) > 0.0:
            inj_y_span = min(float(user_inj_y), max_inj_y)
        else:
            inj_y_span = min(float(grids[0]['r']), max_inj_y) if grids else max_inj_y

        # Hardware-aware budget ceiling: bounded to at least 1,000,000 for extreme scenarios
        hw = detect_hardware_profile()
        user_budget = params.get('max_particle_budget', None)
        if user_budget is not None and float(user_budget) > 0.0:
            MAX_PARTICLE_BUDGET = max(1_000_000.0, float(user_budget))
        else:
            MAX_PARTICLE_BUDGET = max(1_000_000.0, float(hw.max_particle_budget))

        # Real presheath injection geometry
        x_screen_mm = float(grids[0].get('x', getattr(self, 'upstream_gap_mm', 0.8))) if grids else getattr(self, 'upstream_gap_mm', 0.8)
        x_screen_m = max(x_screen_mm, 0.1) * 1e-3

        # Count active cells in the presheath injection cylinder
        ix_presheath = int(np.searchsorted(self.x_coords, x_screen_mm, side='right'))
        iy_inj = int(np.searchsorted(self.y_coords, inj_y_span, side='right'))
        n_presheath_cells = max(1, ix_presheath * iy_inj)

        if self.is_axisymmetric and grids:
            r_inj_m = inj_y_span * 1e-3
            presheath_vol = x_screen_m * np.pi * (r_inj_m ** 2)
            avg_cell_vol = np.pi * r_inj_m * dy_m * hx_m[0]
        else:
            _unit_depth_m = 1e-3
            presheath_vol = x_screen_m * (inj_y_span * 1e-3) * _unit_depth_m
            cell_vol = self.cell_vol_1d[0]
            avg_cell_vol = cell_vol

        n_real_presheath = bohm_factor * n0 * presheath_vol

        # Expected steady-state particle population based on real injection flux & transit time
        v_bohm_est = np.sqrt(self.q_ion * Te_up / self.m_ion)
        v_beam_est = max(v_bohm_est * 5.0, np.sqrt(v_bohm_est**2 + 2.0 * self.q_ion * 1000.0 / self.m_ion))
        speed_ratio = v_beam_est / max(v_bohm_est, 1.0)
        beam_length_mm = max(0.0, self.Lx - x_screen_mm)
        ratio_beam_to_presheath = (beam_length_mm / max(x_screen_mm, 0.1)) / max(speed_ratio, 1.0)

        n_nominal_presheath_macro = n_presheath_cells * self.target_ppc
        est_total_macroparticles = n_nominal_presheath_macro * (1.0 + ratio_beam_to_presheath)

        # Conditional Presheath Budgeting:
        # If expected total particles <= MAX_PARTICLE_BUDGET, leave target_ppc 100% UNMODIFIED!
        if est_total_macroparticles <= MAX_PARTICLE_BUDGET:
            effective_ppc = self.target_ppc
            modulated_budget = False
        else:
            budget_scale = MAX_PARTICLE_BUDGET / max(est_total_macroparticles, 1.0)
            effective_ppc = max(10.0, min(self.target_ppc, self.target_ppc * budget_scale))
            modulated_budget = True

        base_weight = bohm_factor * n0 * avg_cell_vol / effective_ppc
        w_presheath_budget = n_real_presheath / MAX_PARTICLE_BUDGET

        self.macro_weight = max(base_weight, w_presheath_budget, 1.0)
        self.effective_ppc = effective_ppc
        self.max_particle_budget = MAX_PARTICLE_BUDGET

        # Pre-flight Downstream Plume Memory Check & Advisory Warning
        total_cells = self.nx * self.ny
        r_accel_exit = grids[-1]['r'] if grids else 1.0
        plume_len = max(0.0, self.Lx - x_last_grid_end)
        recommended_plume_len = min(plume_len, 2.5 * r_accel_exit)
        self.recommended_Lx = round(x_last_grid_end + recommended_plume_len, 3)

        if plume_len > 3.0 * r_accel_exit and total_cells > 300_000:
            self.plume_warning_active = True
            self.plume_excess_cells = int(total_cells * (1.0 - self.recommended_Lx / max(self.Lx, 1e-6)))
            ram_gb = estimate_mesh_memory_gb(self.nx, self.ny)
            if modulated_budget:
                print(
                    f"[Hardware & Physics Warning] Downstream plume region is very long (plume={plume_len:.2f} mm > 3*r_accel, "
                    f"Lx={self.Lx:.2f} mm, mesh={total_cells:,} cells, est. RAM={ram_gb:.2f} GB). "
                    f"Because downstream plume is not truncated, Particles Per Cell (PPC) must be modulated "
                    f"(effective PPC={effective_ppc:.1f} vs target={self.target_ppc:.1f}) to remain within "
                    f"the hardware memory budget ({MAX_PARTICLE_BUDGET:,.0f} particles). "
                    f"Truncating downstream plume to Lx <= {self.recommended_Lx:.2f} mm will reduce mesh by "
                    f"~{self.plume_excess_cells:,} cells and allow higher unconstrained PPC."
                )
            else:
                print(
                    f"[Hardware Advisory] Downstream plume region is long (plume={plume_len:.2f} mm, "
                    f"Lx={self.Lx:.2f} mm, mesh={total_cells:,} cells, est. RAM={ram_gb:.2f} GB). "
                    f"Consider truncating downstream plume to Lx <= {self.recommended_Lx:.2f} mm to save compute time."
                )
        else:
            self.plume_warning_active = False
        self.mask_grids = []
        self.T_grids    = []
        self.isBound.fill(False)
        self._isBound_int = None
        self._damage_version = 0
        self.V_fixed.fill(0.0)

        if not preserve_state:
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

            if hasattr(self, 'potential_history'):
                self.potential_history.clear()

            if hasattr(self, 'ion_current_history'):
                self.ion_current_history.clear()
            if hasattr(self, 'exit_ion_current_history'):
                self.exit_ion_current_history.clear()

            if hasattr(self, 'substep_history_iter'):
                self.substep_history_iter.clear()
                self.substep_history_ion.clear()
                self.substep_history_e.clear()
                self.last_n_sub_ion = 1
                self.last_n_sub_e = 1

        if not hasattr(self, 'grid_deflections') or len(self.grid_deflections) != len(grids):
            self.grid_deflections = [0.0] * len(grids)

        current_x = self.upstream_gap_mm

        if grids:
            screen_r = grids[0]['r']
            pitch = params.get("pitch_mm", params.get("discharge_chamber", {}).get("pitch_mm", 0.0) if isinstance(params.get("discharge_chamber"), dict) else 0.0)
            if not pitch:
                pitch = 3.0
            d_bound = getattr(self, 'hole_boundary_dist_mm', 0.5 * (screen_r + pitch))
            if geometry == 'two_holes':
                if abs(self.Ly - 4.0 * d_bound) < 1e-4:
                    hole_centers = [d_bound, 3.0 * d_bound]
                else:
                    hole_centers = [0.25 * self.Ly, 0.75 * self.Ly]
            elif geometry == 'one_hole':
                hole_centers = [0.5 * self.Ly]
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
                    y_web = 0.5 * (hole_centers[0] + hole_centers[1]) if len(hole_centers) >= 2 else self.Ly * 0.5
                    w_half = 0.5 * abs(hole_centers[1] - hole_centers[0]) if len(hole_centers) >= 2 else max(pitch * 0.5, 1e-6)
                    eta = np.clip(1.0 - np.abs(self.Y - y_web) / max(w_half, 1e-6), 0.0, 1.0)
                    dxbow = delta * eta ** 2
                elif geometry == 'one_hole':
                    y_mid = hole_centers[0] if hole_centers else self.Ly * 0.5
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

        from ..structural_surface import SubgridManager
        mat_props = {
            'k': getattr(self, 'mat_k', 138.0),
            'rho': getattr(self, 'mat_rho', 10280.0),
            'cp': getattr(self, 'mat_cp', 250.0),
            'emissivity': getattr(self, 'emissivity', 0.8),
            'alpha': getattr(self, 'alpha_thermal', 4.8e-6),
        }
        if not preserve_state or getattr(self, 'subgrid_manager', None) is None:
            self.subgrid_manager = SubgridManager(
                grids=grids,
                upstream_gap_mm=self.upstream_gap_mm,
                Ly=self.Ly,
                geometry=geometry,
                hole_centers=hole_centers,
                mat_props=mat_props,
            )
        else:
            self.subgrid_manager.update_deflections(self.grid_deflections)

        self.Vdc = np.copy(self.V_fixed)

        v_plasma_bound = (grids[0]["V"] + params.get("V_plasma_offset", 20.0)
                          if grids else 1000.0 + params.get("V_plasma_offset", 20.0))
        self.V_fixed[:, 0] = v_plasma_bound
        self.isBound[:, 0] = True

        if not preserve_state:
            self.Tmap[self.isBound] = 300.0
        self.build_sparse_matrix()
        if not preserve_state:
            self.recalc_poisson(iterations=30, params=params)

        # Stage 2: Electric field jump evaluation & pre-flight adaptive refinement
        if enforce_efield and grids and not preserve_state:
            max_jump_x, max_jump_y, compliant = evaluate_mesh_efield_variation(
                self.Ex, self.Ey, is_bound=self.isBound, max_rel_jump=max_jump_allowed
            )
            self.last_max_efield_jump_x = max_jump_x
            self.last_max_efield_jump_y = max_jump_y
            self.last_efield_jump_compliant = compliant

            # Only adapt when cell spacing was constrained by geometry/E-field (low-density regime)
            orig_dxy = getattr(self, "_original_dxy_mm", None)
            is_low_density = (ratio_debye is not None and orig_dxy is not None and (orig_dxy > self.dx * 1.05)) or (ratio_debye is None)
            if not compliant and is_low_density and not getattr(self, "_efield_refinement_active", False):
                refine_x = min(2.5, max(1.0, 1.25 * max_jump_x / max_jump_allowed))
                refine_y = min(2.5, max(1.0, 1.25 * max_jump_y / max_jump_allowed))
                if refine_x > 1.10 or refine_y > 1.10:
                    print(
                        f"[Mesh E-Field] Cell-to-cell jump (x: {max_jump_x*100:.1f}%, y: {max_jump_y*100:.1f}%) "
                        f"exceeds target {max_jump_allowed*100:.0f}%. Adapting mesh (refine_x={refine_x:.2f}, refine_y={refine_y:.2f})."
                    )
                    self._efield_refinement_active = True
                    self.dx = self.dx / refine_x
                    self.dy = self.dy / refine_y
                    self.build_domain(params, preserve_state=preserve_state)
                    self._efield_refinement_active = False
                    return
        else:
            self.last_max_efield_jump_x = 0.0
            self.last_max_efield_jump_y = 0.0
            self.last_efield_jump_compliant = True

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

    def reshape_downstream_plume(self, new_Lx=None, params=None):
        """
        Dynamically truncate the downstream plume region at runtime to reduce memory
        and accelerate execution while preserving particle state inside the valid domain.
        """
        target_Lx = new_Lx or getattr(self, 'recommended_Lx', None)
        if target_Lx is None or target_Lx >= self.Lx:
            return False

        old_Lx = self.Lx
        cfg = dict(getattr(self, '_last_params', {}))
        if params:
            cfg.update(params)
        cfg['Lx'] = target_Lx

        # Update last grid gap to match truncated plume
        if 'grids' in cfg and cfg['grids']:
            import copy
            grids_copy = copy.deepcopy(cfg['grids'])
            x_last_exit = self.upstream_gap_mm + sum(g['t'] + g['gap'] for g in grids_copy) - grids_copy[-1]['gap']
            grids_copy[-1]['gap'] = max(0.05, round(target_Lx - x_last_exit, 3))
            cfg['grids'] = grids_copy

        # Retain particles that fall within the new domain bounds
        if self.num_p > 0:
            valid_p = (self.p_x[:self.num_p] < target_Lx)
            if not np.all(valid_p):
                valid_idx = np.where(valid_p)[0]
                n_rem = len(valid_idx)
                self.p_x[:n_rem] = self.p_x[valid_idx]
                self.p_y[:n_rem] = self.p_y[valid_idx]
                self.p_vx[:n_rem] = self.p_vx[valid_idx]
                self.p_vy[:n_rem] = self.p_vy[valid_idx]
                self.p_vz[:n_rem] = self.p_vz[valid_idx]
                self.p_isCEX[:n_rem] = self.p_isCEX[valid_idx]
                self.num_p = n_rem

        self.build_domain(cfg, preserve_state=True)
        self.recalc_poisson(iterations=10, params=cfg)
        print(f"[Plume Reshape] Downstream plume truncated from {old_Lx:.3f} mm to {self.Lx:.3f} mm. Mesh now has {self.nx * self.ny:,} cells.")
        return True

    def execute_plume_remesh(self, params=None, remesh_cfg=None, force=False):
        """
        Execute automatic adaptive plume remeshing:
        - Automatically computes downstream plume coarsening factor based on live plume ion density
          and Debye length expansion ratio.
        - Keeps upstream presheath and inter-grid optics strictly at baseline resolution.
        - If plume factor changes significantly (or force=True), rebuilds the domain,
          interpolates fields, preserves particles, re-deposits charge, and relaxes potential.
        - Records iteration and mean plume dx into plume cell size history.
        """
        from ..mesh_utils import estimate_automatic_plume_mesh_zones, compute_plume_mean_dx

        cfg = dict(getattr(self, '_last_params', {}))
        if params:
            cfg.update(params)

        remesh_settings = remesh_cfg or cfg.get("adaptive_remesh_settings", {})
        if not remesh_settings and isinstance(cfg.get("advanced_settings"), dict):
            remesh_settings = cfg["advanced_settings"].get("adaptive_remesh_settings", {})
        grids = cfg.get("grids", [])

        new_zones = estimate_automatic_plume_mesh_zones(
            sim=self,
            grids=grids,
            upstream_gap_mm=self.upstream_gap_mm,
            Lx=self.Lx,
            dx0=self.dx,
            remesh_cfg=remesh_settings
        )

        far_zone = next((z for z in new_zones if z['name'] == 'Far_Plume'), None)
        new_far_factor = far_zone['factor'] if far_zone else 4.0
        old_zones = getattr(self, 'zone_configs', None)

        # Check if any zone factor across the domain changed by >= 0.15
        max_factor_diff = 0.0
        if isinstance(old_zones, list) and len(old_zones) == len(new_zones):
            for oz, nz in zip(old_zones, new_zones):
                max_factor_diff = max(
                    max_factor_diff,
                    abs(float(nz.get('factor', 1.0)) - float(oz.get('factor', 1.0)))
                )
        else:
            old_far_factor = getattr(self, '_current_plume_factor', None)
            max_factor_diff = abs(new_far_factor - old_far_factor) if old_far_factor is not None else 999.0

        if not force and max_factor_diff < 0.15:
            if not getattr(self, '_plume_remesh_deferred', False):
                current_dx = compute_plume_mean_dx(self)
                if not hasattr(self, 'plume_history_iterations'):
                    self.plume_history_iterations = []
                    self.plume_history_dx = []
                    self.plume_dx_history = []
                if not hasattr(self, 'zone_history_dx') or not isinstance(self.zone_history_dx, dict):
                    self.zone_history_dx = {}
                if not self.plume_history_iterations or self.plume_history_iterations[-1] != self.iteration:
                    self.plume_history_iterations.append(self.iteration)
                    self.plume_history_dx.append(current_dx)
                    self.plume_dx_history.append((self.iteration, current_dx))
                    curr_zones = getattr(self, 'zone_configs', None) or new_zones
                    n_prev = len(self.plume_history_iterations) - 1
                    for z in (curr_zones or []):
                        z_name = str(z.get('name', 'Zone'))
                        z_dx = float(self.dx * float(z.get('factor', 1.0)))
                        if z_name not in self.zone_history_dx:
                            self.zone_history_dx[z_name] = [z_dx] * n_prev
                        self.zone_history_dx[z_name].append(z_dx)
            return False

        old_x = self.x_coords.copy()
        old_y = self.y_coords.copy()
        old_V = self.V.copy() if hasattr(self, 'V') and self.V is not None else None
        old_Tmap = self.Tmap.copy() if getattr(self, 'Tmap', None) is not None else None
        old_damage = self.damage_map.copy() if getattr(self, 'damage_map', None) is not None else None
        old_eroded = self.eroded_depth.copy() if getattr(self, 'eroded_depth', None) is not None else None
        old_accum_ppc = self.accum_ppc_map.copy() if getattr(self, 'accum_ppc_map', None) is not None else None
        old_accum_cex = self.accum_cex_ppc_map.copy() if getattr(self, 'accum_cex_ppc_map', None) is not None else None
        old_ppc_steps = getattr(self, 'ppc_steps_count', 0)
        old_iter = self.iteration

        num_p = self.num_p
        old_p_x = self.p_x[:num_p].copy()
        old_p_y = self.p_y[:num_p].copy()
        old_p_vx = self.p_vx[:num_p].copy()
        old_p_vy = self.p_vy[:num_p].copy()
        old_p_vz = self.p_vz[:num_p].copy()
        old_isCEX = self.p_isCEX[:num_p].copy() if hasattr(self, 'p_isCEX') and self.p_isCEX is not None else None

        cfg["zone_configs"] = new_zones
        cfg["use_json_mesh_zones"] = False
        cfg["Lx"] = self.Lx
        cfg["Ly"] = self.Ly
        cfg["truncate_downstream_plume"] = bool(getattr(self, "truncate_downstream_plume", False))

        # Rebuild domain geometry and matrices on the newly partitioned mesh
        self.build_domain(cfg, preserve_state=True)

        # Restore particle state
        self.num_p = num_p
        self.p_x[:num_p] = old_p_x
        self.p_y[:num_p] = old_p_y
        self.p_vx[:num_p] = old_p_vx
        self.p_vy[:num_p] = old_p_vy
        self.p_vz[:num_p] = old_p_vz
        if old_isCEX is not None and hasattr(self, 'p_isCEX'):
            self.p_isCEX[:num_p] = old_isCEX
        self.iteration = old_iter

        # Interpolate 2D Eulerian scalar fields
        from scipy.interpolate import RegularGridInterpolator
        pts_new = (self.Y, self.X)

        if old_V is not None:
            interp_V = RegularGridInterpolator((old_y, old_x), old_V, bounds_error=False, fill_value=None)
            self.V = interp_V(pts_new)
            self.V[self.isBound] = self.V_fixed[self.isBound]

        if old_Tmap is not None:
            interp_T = RegularGridInterpolator((old_y, old_x), old_Tmap, bounds_error=False, fill_value=300.0)
            self.Tmap = interp_T(pts_new).astype(_NP_FP)
            self.T_map = self.Tmap
            self.T_map_new = self.Tmap.copy()

        if old_damage is not None:
            interp_dmg = RegularGridInterpolator((old_y, old_x), old_damage, bounds_error=False, fill_value=0.0)
            self.damage_map = interp_dmg(pts_new)

        if old_eroded is not None:
            interp_ero = RegularGridInterpolator((old_y, old_x), old_eroded, bounds_error=False, fill_value=0.0)
            self.eroded_depth = interp_ero(pts_new)

        if old_accum_ppc is not None:
            interp_ppc = RegularGridInterpolator((old_y, old_x), old_accum_ppc, bounds_error=False, fill_value=0.0)
            self.accum_ppc_map = np.maximum(0.0, interp_ppc(pts_new))
            self.ppc_steps_count = old_ppc_steps

        if old_accum_cex is not None:
            interp_cex = RegularGridInterpolator((old_y, old_x), old_accum_cex, bounds_error=False, fill_value=0.0)
            self.accum_cex_ppc_map = np.maximum(0.0, interp_cex(pts_new))

        # Recompute current_ppc_map on the new mesh so PPC & Density diagnostics stay continuous
        if self.num_p > 0:
            ix_new = self._x_to_ix(self.p_x[:self.num_p])
            iy_new = np.clip(np.round(self.p_y[:self.num_p] / self.dy).astype(int), 0, self.ny - 1)
            flat_new = iy_new * self.nx + ix_new
            self.current_ppc_map = np.bincount(flat_new, minlength=self.nx * self.ny).reshape((self.ny, self.nx))
            if hasattr(self, 'p_isCEX') and self.p_isCEX is not None:
                is_cex_now = self.p_isCEX[:self.num_p]
                if np.any(is_cex_now):
                    self.current_cex_ppc_map = np.bincount(flat_new[is_cex_now], minlength=self.nx * self.ny).reshape((self.ny, self.nx))
                else:
                    self.current_cex_ppc_map = np.zeros((self.ny, self.nx), dtype=np.int32)
            self.current_prim_ppc_map = np.maximum(0, self.current_ppc_map - self.current_cex_ppc_map)

        # Re-deposit charge on the adapted grid
        self.rho.fill(0.0)
        from ..step.field_solve import USE_TAICHI, accumulate_rho_taichi, accumulate_rho_cpu
        charge_per_particle = float(self.q_ion * self.macro_weight)
        if getattr(self, 'zone_b_ti', None) is None or getattr(self, 'cvol_r_ti', None) is None:
            self._update_taichi_mesh_constants()
        is_axisym = 1 if getattr(self, 'is_axisymmetric', False) else 0

        if self.num_p > 0:
            if USE_TAICHI and hasattr(self, 'zone_b_ti') and self.zone_b_ti is not None:
                accumulate_rho_taichi(
                    self.p_x[:self.num_p],
                    self.p_y[:self.num_p],
                    self.rho,
                    self.cvol_r_ti,
                    self.cvol_x_ti,
                    self.num_p,
                    self._dy_f32,
                    self.nx,
                    self.ny,
                    np.float32(charge_per_particle),
                    self.zone_b_ti, self.zone_dx_ti, self.zone_off_ti, self.num_zones,
                    is_axisym
                )
            else:
                accumulate_rho_cpu(
                    self.p_x[:self.num_p],
                    self.p_y[:self.num_p],
                    self.rho,
                    self.num_p,
                    self.dy,
                    self.nx,
                    self.ny,
                    charge_per_particle,
                    self.zone_boundaries,
                    self.zone_dx,
                    self.zone_offsets,
                    self.cell_vol_1d,
                    cvol_r=self.cvol_r_1d,
                    cvol_x=self.cvol_x_1d,
                    is_axisymmetric=bool(is_axisym)
                )

        if getattr(self, 'num_e', 0) > 0:
            if USE_TAICHI and hasattr(self, 'zone_b_ti') and self.zone_b_ti is not None:
                accumulate_rho_taichi(
                    self.e_x[:self.num_e],
                    self.e_y[:self.num_e],
                    self.rho,
                    self.cvol_r_ti,
                    self.cvol_x_ti,
                    self.num_e,
                    self._dy_f32,
                    self.nx,
                    self.ny,
                    np.float32(-charge_per_particle),
                    self.zone_b_ti, self.zone_dx_ti, self.zone_off_ti, self.num_zones,
                    is_axisym
                )
            else:
                accumulate_rho_cpu(
                    self.e_x[:self.num_e],
                    self.e_y[:self.num_e],
                    self.rho,
                    self.num_e,
                    self.dy,
                    self.nx,
                    self.ny,
                    -charge_per_particle,
                    self.zone_boundaries,
                    self.zone_dx,
                    self.zone_offsets,
                    self.cell_vol_1d,
                    cvol_r=self.cvol_r_1d,
                    cvol_x=self.cvol_x_1d,
                    is_axisymmetric=bool(is_axisym)
                )

        # Execute Picard relaxation steps to smoothly reconcile fields on the new mesh
        self.recalc_poisson(iterations=5, params=cfg)

        self._current_plume_factor = new_far_factor
        self._has_stability_remeshed = True
        self._stability_remesh_iter = self.iteration

        # Record plume & all-zone cell size history
        mean_plume_dx = compute_plume_mean_dx(self, new_zones)
        if not hasattr(self, 'plume_history_iterations'):
            self.plume_history_iterations = []
            self.plume_history_dx = []
            self.plume_dx_history = []
        if not hasattr(self, 'zone_history_dx') or not isinstance(self.zone_history_dx, dict):
            self.zone_history_dx = {}
        n_prev = len(self.plume_history_iterations)
        self.plume_history_iterations.append(self.iteration)
        self.plume_history_dx.append(mean_plume_dx)
        self.plume_dx_history.append((self.iteration, mean_plume_dx))
        for z in (new_zones or []):
            z_name = str(z.get('name', 'Zone'))
            z_dx = float(self.dx * float(z.get('factor', 1.0)))
            if z_name not in self.zone_history_dx:
                self.zone_history_dx[z_name] = [z_dx] * n_prev
            self.zone_history_dx[z_name].append(z_dx)

        zone_str = ", ".join(f"{z['name']}:{z['factor']:.1f}x" for z in new_zones)
        print(f"[Adaptive Remesh] Iter {self.iteration}: Adapted domain mesh ({zone_str}, plume mean Δx {mean_plume_dx:.4f} mm, grid {self.ny}x{self.nx}).")
        return True

    def execute_stability_remesh(self, params=None, remesh_cfg=None):
        """Backward-compatible alias for execute_plume_remesh."""
        return self.execute_plume_remesh(params=params, remesh_cfg=remesh_cfg, force=True)
