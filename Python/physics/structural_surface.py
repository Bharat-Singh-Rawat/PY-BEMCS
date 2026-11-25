"""
Sub-Grid Structural Surface & Thermal Representation.

Decouples physical grid boundary geometry, continuous sputtering erosion,
and structural thermal conduction from the background Eulerian PIC mesh.
"""
import numpy as np

from .constants import SIGMA_SB


class SubgridSegment:
    """A fine 1D line element along an electrode boundary face or chamfer."""

    __slots__ = (
        'x0', 'y0', 'x1', 'y1', 'x_mid', 'y_mid', 'length',
        'nx', 'ny', 'face', 'grid_idx', 'damage', 'eroded_depth_mm',
        'temperature_k'
    )

    def __init__(self, x0, y0, x1, y1, face, grid_idx):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x_mid = 0.5 * (self.x0 + self.x1)
        self.y_mid = 0.5 * (self.y0 + self.y1)
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        self.length = max(1e-9, float(np.hypot(dx, dy)))

        # Unit normal pointing outward from metal into plasma
        if face == 'upstream':
            # Front face: normal points upstream (-x)
            self.nx = -1.0
            self.ny = 0.0
        elif face == 'downstream':
            # Back face: normal points downstream (+x)
            self.nx = 1.0
            self.ny = 0.0
        elif face in ('barrel', 'chamfer'):
            # Aperture barrel: normal points radially inward toward hole axis
            # Tangent is (dx, dy). Normal into hole: (-dy, dx) normalized
            tang_x = dx / self.length
            tang_y = dy / self.length
            # For y > yc, inward is -y
            nx_cand = -tang_y
            ny_cand = tang_x
            if ny_cand > 0:
                nx_cand = -nx_cand
                ny_cand = -ny_cand
            norm = max(1e-9, float(np.hypot(nx_cand, ny_cand)))
            self.nx = nx_cand / norm
            self.ny = ny_cand / norm
        else:
            self.nx = 0.0
            self.ny = 1.0

        self.face = face
        self.grid_idx = int(grid_idx)
        self.damage = 0.0
        self.eroded_depth_mm = 0.0
        self.temperature_k = 300.0


class SubgridApertureProfile:
    """Continuous 1D polyline boundary and sub-element profile for one grid aperture."""

    def __init__(self, grid_idx, grid_dict, gstart, gend, dxbow, Ly, hole_centers, n_sub=80):
        self.grid_idx = grid_idx
        self.grid_dict = grid_dict
        self.gstart = float(gstart)
        self.gend = float(gend)
        self.dxbow = float(dxbow)
        self.Ly = float(Ly)
        self.hole_centers = list(hole_centers) if hole_centers else [0.0]
        self.n_sub = int(max(20, n_sub))

        self.segments = []
        self._build_profile()

    def _build_profile(self):
        """Construct fine subgrid boundary segments along upstream face, chamfer, and downstream face."""
        self.segments.clear()
        t = float(self.grid_dict.get('t', 0.5))
        r_ap = float(self.grid_dict.get('r', 0.8))
        cham_deg = float(self.grid_dict.get('cham', 0.0))
        cham_rad = np.radians(cham_deg)

        x_front = self.gstart + self.dxbow
        x_back = self.gend + self.dxbow

        for yc in self.hole_centers:
            # Radius at front and back
            r_front = r_ap
            # local_r = r - (x - gstart) * tan(cham)
            r_back = max(0.0, r_ap - t * np.tan(cham_rad))

            # 1. Upstream Face: from (x_front, yc + r_front) to (x_front, Ly)
            y_pts_up = np.linspace(yc + r_front, self.Ly, self.n_sub + 1)
            for j in range(self.n_sub):
                seg = SubgridSegment(
                    x_front, y_pts_up[j], x_front, y_pts_up[j + 1],
                    face='upstream', grid_idx=self.grid_idx
                )
                self.segments.append(seg)

            # 2. Aperture Barrel / Chamfer: from (x_front, yc + r_front) to (x_back, yc + r_back)
            x_pts_ch = np.linspace(x_front, x_back, self.n_sub + 1)
            y_pts_ch = yc + np.maximum(0.0, r_ap - (x_pts_ch - x_front) * np.tan(cham_rad))
            face_name = 'chamfer' if cham_deg > 0.0 else 'barrel'
            for j in range(self.n_sub):
                seg = SubgridSegment(
                    x_pts_ch[j], y_pts_ch[j], x_pts_ch[j + 1], y_pts_ch[j + 1],
                    face=face_name, grid_idx=self.grid_idx
                )
                self.segments.append(seg)

            # 3. Downstream Face: from (x_back, yc + r_back) to (x_back, Ly)
            y_pts_dn = np.linspace(yc + r_back, self.Ly, self.n_sub + 1)
            for j in range(self.n_sub):
                seg = SubgridSegment(
                    x_back, y_pts_dn[j], x_back, y_pts_dn[j + 1],
                    face='downstream', grid_idx=self.grid_idx
                )
                self.segments.append(seg)

        # Cache midpoints as arrays for fast nearest-neighbor projection
        self._x_mid = np.array([s.x_mid for s in self.segments], dtype=np.float64)
        self._y_mid = np.array([s.y_mid for s in self.segments], dtype=np.float64)

    def find_nearest_segment_idx(self, px, py):
        """Find the index of the nearest subgrid segment to point (px, py)."""
        dist_sq = (self._x_mid - px) ** 2 + (self._y_mid - py) ** 2
        return int(np.argmin(dist_sq))

    def record_hit(self, px, py, vx, vy, vz, energy_eV, macro_weight, Y_coeff, E_th, thresh, dy):
        """Record a single particle impact on the nearest subgrid segment with Yamamura angular factor."""
        if energy_eV <= E_th:
            return 0.0, None

        idx = self.find_nearest_segment_idx(px, py)
        seg = self.segments[idx]

        v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
        if v_mag > 1e-6:
            # cos(theta) = -(v . n) / |v|
            cos_theta = -(vx * seg.nx + vy * seg.ny) / v_mag
            cos_theta = max(0.1, min(1.0, cos_theta))
            # Yamamura angular enhancement factor (bounded to 3.0 max)
            f_theta = min(3.0, 1.0 / cos_theta)
        else:
            f_theta = 1.0

        yield_rate = Y_coeff * (energy_eV - E_th) * f_theta
        damage = yield_rate * macro_weight
        seg.damage += damage

        # Continuous recession depth in mm: damage normalized to thresh * dy
        delta_depth = (damage / max(thresh, 1.0)) * dy
        seg.eroded_depth_mm += delta_depth

        return delta_depth, seg

    def get_face_profile(self, face='downstream'):
        """Return (y_coords, eroded_depth_mm) along a specific grid face."""
        matching = [s for s in self.segments if s.face == face]
        if not matching:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        if face in ('upstream', 'downstream'):
            matching.sort(key=lambda s: s.y_mid)
            y_arr = np.array([s.y_mid for s in matching], dtype=np.float64)
            d_arr = np.array([s.eroded_depth_mm for s in matching], dtype=np.float64)
            return y_arr, d_arr
        else:
            matching.sort(key=lambda s: s.x_mid)
            x_arr = np.array([s.x_mid for s in matching], dtype=np.float64)
            d_arr = np.array([s.eroded_depth_mm for s in matching], dtype=np.float64)
            return x_arr, d_arr


class SubgridThermalModel:
    """Dedicated 1D radial finite-volume thermal conduction model for one grid plate."""

    def __init__(self, grid_idx, r_min, r_max, thickness_mm, k, rho, cp, emissivity, alpha, n_r=50):
        self.grid_idx = grid_idx
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.thickness_m = float(thickness_mm) * 1e-3
        self.k = float(k)
        self.rho = float(rho)
        self.cp = float(cp)
        self.emissivity = float(emissivity)
        self.alpha = float(alpha)
        self.n_r = int(max(10, n_r))

        self.r_nodes = np.linspace(self.r_min, self.r_max, self.n_r)
        self.dr_m = (self.r_max - self.r_min) * 1e-3 / max(1, self.n_r - 1)
        self.T = np.full(self.n_r, 300.0, dtype=np.float64)
        self.heat_flux = np.zeros(self.n_r, dtype=np.float64)
        self.T_mean = 300.0

    def add_heat(self, r_mm, energy_joules):
        """Accumulate heat energy into the nearest radial cell."""
        idx = int(np.clip(np.round((r_mm - self.r_min) / ((self.r_max - self.r_min) / max(1, self.n_r - 1))), 0, self.n_r - 1))
        self.heat_flux[idx] += float(energy_joules)

    def solve_step(self, dt, thermal_accel=1e7):
        """Solve 1D radial conduction with Stefan-Boltzmann radiative cooling (numerically bounded & stable)."""
        dt_eff = dt * thermal_accel
        r_m = self.r_nodes * 1e-3
        dr = max(self.dr_m, 1e-6)

        # Radiative cooling on both faces: 2 * eps * sigma * (T^4 - 300^4)
        q_rad = 2.0 * self.emissivity * SIGMA_SB * (np.clip(self.T, 300.0, 3000.0) ** 4 - 300.0 ** 4)
        vol_cell = 2.0 * np.pi * np.maximum(r_m, 1e-6) * dr * self.thickness_m
        q_vol = self.heat_flux / np.maximum(vol_cell, 1e-12)
        self.heat_flux.fill(0.0)

        # Bounded explicit Fourier diffusion
        alpha_diff = self.k / (self.rho * self.cp)
        raw_fo = alpha_diff * dt_eff / (dr ** 2)
        fo = min(raw_fo, 0.20)  # Numerical stability limit for 1D explicit diffusion

        T_new = self.T.copy()
        # Interior points: (1/r) d/dr (r dT/dr)
        for i in range(1, self.n_r - 1):
            r_p = 0.5 * (r_m[i] + r_m[i + 1])
            r_m_ = 0.5 * (r_m[i] + r_m[i - 1])
            cond = (r_p * (self.T[i + 1] - self.T[i]) - r_m_ * (self.T[i] - self.T[i - 1])) / r_m[i]
            dT_heat = q_vol[i] / (self.rho * self.cp)
            dT_rad = (dt_eff / (self.rho * self.cp)) * (q_rad[i] / self.thickness_m)
            dT = fo * cond + dT_heat - dT_rad
            T_new[i] = float(np.clip(self.T[i] + dT, 300.0, 3000.0))

        # Inner boundary (r_min): adiabatic / radiation only (dT/dr = 0)
        T_new[0] = float(np.clip(T_new[1] - (dt_eff / (self.rho * self.cp)) * (q_rad[0] / self.thickness_m), 300.0, 3000.0))
        # Outer boundary (r_max): clamped to 300K chassis
        T_new[-1] = 300.0

        self.T = T_new
        self.T_mean = float(np.mean(self.T))
        return self.T_mean

    def compute_deflection(self, L_cant_mm, max_defl_mm=0.5):
        """Compute Euler-Bernoulli cantilever deflection in mm, physically bounded."""
        dT = max(0.0, min(1500.0, self.T_mean - 300.0))
        L_cant_m = float(L_cant_mm) * 1e-3
        if self.thickness_m > 0 and L_cant_m > 0:
            delta_m = self.alpha * dT * (L_cant_m ** 2) / (2.0 * self.thickness_m)
            defl = float(delta_m * 1e3)
            return float(np.clip(defl, -max_defl_mm, max_defl_mm))
        return 0.0


class SubgridManager:
    """Orchestrates continuous subgrid aperture profiles and thermal models for all grids."""

    def __init__(self, grids, upstream_gap_mm, Ly, geometry='half_hole', hole_centers=None, mat_props=None):
        self.grids = list(grids) if grids else []
        self.upstream_gap_mm = float(upstream_gap_mm)
        self.Ly = float(Ly)
        self.geometry = geometry
        self.hole_centers = list(hole_centers) if hole_centers else [0.0]
        self.mat_props = mat_props or {}
        self.thermal_enabled = True

        self.profiles = []
        self.thermal_models = []
        self._build()

    def _build(self):
        self.profiles.clear()
        self.thermal_models.clear()
        if not self.grids:
            return

        current_x = self.upstream_gap_mm
        k = self.mat_props.get('k', 138.0)
        rho = self.mat_props.get('rho', 10280.0)
        cp = self.mat_props.get('cp', 250.0)
        emissivity = self.mat_props.get('emissivity', 0.8)
        alpha = self.mat_props.get('alpha', 4.8e-6)

        for i, grid in enumerate(self.grids):
            t = float(grid.get('t', 0.5))
            gap = float(grid.get('gap', 1.0))
            r = float(grid.get('r', 0.8))
            gstart = current_x
            gend = gstart + t

            prof = SubgridApertureProfile(
                grid_idx=i,
                grid_dict=grid,
                gstart=gstart,
                gend=gend,
                dxbow=0.0,
                Ly=self.Ly,
                hole_centers=self.hole_centers,
                n_sub=80
            )
            self.profiles.append(prof)

            therm = SubgridThermalModel(
                grid_idx=i,
                r_min=r,
                r_max=self.Ly,
                thickness_mm=t,
                k=k,
                rho=rho,
                cp=cp,
                emissivity=emissivity,
                alpha=alpha,
                n_r=50
            )
            self.thermal_models.append(therm)

            current_x = gend + gap

    def update_deflections(self, deflections):
        """Update aperture positions when grid cantilever deflection changes."""
        for i, delta in enumerate(deflections):
            if i < len(self.profiles):
                self.profiles[i].dxbow = float(delta)
                self.profiles[i]._build_profile()

    def record_erosion_hits(self, px, py, vx, vy, vz, sim, params):
        """
        Process continuous erosion impacts on subgrid aperture segments.
        Updates continuous segment recession and synchronizes with sim.eroded_depth.
        """
        if not self.profiles or len(px) == 0:
            return

        thresh = float(params.get('Thresh', 1e5))
        macro_weight = float(sim.macro_weight)
        Y_coeff = float(sim.sputter_Y_coeff)
        E_th = float(sim.sputter_E_th)
        dy = float(sim.dy)
        m_ion = float(sim.m_ion)
        q = float(sim.q)

        v_sq = vx * vx + vy * vy + vz * vz
        E_eV = 0.5 * m_ion * v_sq / q
        valid_mask = E_eV > E_th

        if not np.any(valid_mask):
            return

        valid_px = px[valid_mask]
        valid_py = py[valid_mask]
        valid_vx = vx[valid_mask]
        valid_vy = vy[valid_mask]
        valid_vz = vz[valid_mask]
        valid_E = E_eV[valid_mask]

        # For each impacting particle, find which grid profile it hit
        for i in range(len(valid_px)):
            x_i = valid_px[i]
            y_i = valid_py[i]

            # Find matching grid
            matched_prof = None
            for prof in self.profiles:
                if (prof.gstart + prof.dxbow - 0.1) <= x_i <= (prof.gend + prof.dxbow + 0.1):
                    matched_prof = prof
                    break
            if matched_prof is None and self.profiles:
                # Nearest profile by axial distance
                dists = [abs(x_i - (p.gstart + p.dxbow)) for p in self.profiles]
                matched_prof = self.profiles[int(np.argmin(dists))]

            if matched_prof is not None:
                d_depth, seg = matched_prof.record_hit(
                    x_i, y_i, valid_vx[i], valid_vy[i], valid_vz[i],
                    valid_E[i], macro_weight, Y_coeff, E_th, thresh, dy
                )

                # Feed heat to thermal model
                if self.thermal_enabled and matched_prof.grid_idx < len(self.thermal_models):
                    E_joules = valid_E[i] * q * macro_weight
                    self.thermal_models[matched_prof.grid_idx].add_heat(y_i, E_joules)

        # Synchronize continuous subgrid depths into sim.eroded_depth
        self.sync_to_domain(sim)

    def sync_to_domain(self, sim):
        """Map fine subgrid erosion depths onto the 2D background array sim.eroded_depth."""
        if not hasattr(sim, 'eroded_depth') or not self.profiles:
            return

        if sim.eroded_depth.shape != (sim.ny, sim.nx):
            sim.eroded_depth = np.zeros((sim.ny, sim.nx), dtype=np.float64)

        for prof in self.profiles:
            grid_idx = prof.grid_idx
            if grid_idx >= len(sim.mask_grids):
                continue
            mask = sim.mask_grids[grid_idx]
            if not np.any(mask):
                continue

            # Downstream face mapping
            y_dn, d_dn = prof.get_face_profile(face='downstream')
            if len(y_dn) > 0 and len(d_dn) > 0:
                has_any = mask.any(axis=1)
                last_cols = np.array([np.where(mask[iy])[0][-1] if has_any[iy] else 0 for iy in range(sim.ny)])
                iy_valid = np.where(has_any)[0]
                y_sim = sim.y_coords[iy_valid]
                # Interpolate subgrid depth onto sim y_coords
                d_interp = np.interp(y_sim, y_dn, d_dn, left=0.0, right=0.0)
                for k, iy in enumerate(iy_valid):
                    col = last_cols[iy]
                    sim.eroded_depth[iy, col] = max(sim.eroded_depth[iy, col], d_interp[k])

    def get_groove_profile(self, grid_idx, face='downstream'):
        """Return continuous (y_mm, depth_um) for high-resolution plotting."""
        if grid_idx < 0 or grid_idx >= len(self.profiles):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        prof = self.profiles[grid_idx]
        y_mm, d_mm = prof.get_face_profile(face=face)
        # Return depth in micrometers for UI plotting
        return y_mm, d_mm * 1e3
