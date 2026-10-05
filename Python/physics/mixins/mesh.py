"""
Mesh coordinates and indexing mixin for DigitalTwinSimulator.
"""
import numpy as np


class MeshMixin:
    """Mesh creation, coordinate transforms, and cell indexing."""

    def _recompute_cell_constants(self):
        self.C_cell = self.mat_rho * (self.dx * 1e-3) * (self.dy * 1e-3) * 1e-3 * self.mat_cp
        self.A_cell = 2 * (self.dx * 1e-3) * 1e-3

    def _x_to_ix(self, x):
        """Vectorized O(1) nearest-grid-point node index lookup for piecewise-uniform mesh."""
        if np.isscalar(x):
            z = int(np.searchsorted(self.zone_boundaries[1:-1], x))
            ix = self.zone_offsets[z] + int(np.round((x - self.zone_boundaries[z]) / self.zone_dx[z]))
            return int(np.clip(ix, 0, self.nx - 1))
        else:
            x_arr = np.asarray(x)
            if len(x_arr) == 0:
                return np.empty(0, dtype=np.int32)
            z = np.searchsorted(self.zone_boundaries[1:-1], x_arr)
            ix = self.zone_offsets[z] + np.round((x_arr - self.zone_boundaries[z]) / self.zone_dx[z]).astype(np.int32)
            return np.clip(ix, 0, self.nx - 1)

    def _x_to_ix_floor(self, x):
        """Vectorized O(1) cell left-node index lookup for piecewise-uniform mesh."""
        if np.isscalar(x):
            z = int(np.searchsorted(self.zone_boundaries[1:-1], x))
            ix0 = self.zone_offsets[z] + int(np.floor((x - self.zone_boundaries[z]) / self.zone_dx[z]))
            return int(np.clip(ix0, 0, self.nx - 2))
        else:
            x_arr = np.asarray(x)
            if len(x_arr) == 0:
                return np.empty(0, dtype=np.int32)
            z = np.searchsorted(self.zone_boundaries[1:-1], x_arr)
            ix0 = self.zone_offsets[z] + np.floor((x_arr - self.zone_boundaries[z]) / self.zone_dx[z]).astype(np.int32)
            return np.clip(ix0, 0, self.nx - 2)

    def _build_nonuniform_x_coords(self, zone_configs, Lx, dx0):
        """
        Build piecewise-uniform x_coords across 3 axial zones.
        zone_configs: list of dicts with keys 'x_start', 'x_end', 'factor'
        Returns:
            x_coords (1D ndarray), bounds (1D ndarray), dxs (1D ndarray), offsets (1D ndarray)
        """
        node_lists = []
        offsets = []
        dxs = []
        bounds = []
        curr_offset = 0
        for i, z in enumerate(zone_configs):
            x_start = float(z['x_start'])
            x_end = float(z['x_end'])
            factor = float(z.get('factor', 1.0))
            L = x_end - x_start
            if L <= 1e-6:
                continue
            dx_target = dx0 * factor
            n = max(1, int(round(L / dx_target)))
            dx_act = L / n
            nodes = np.linspace(x_start, x_end, n + 1)
            if len(node_lists) == 0:
                node_lists.append(nodes)
            else:
                node_lists.append(nodes[1:])
            bounds.append(x_start)
            dxs.append(dx_act)
            offsets.append(curr_offset)
            curr_offset += n

        bounds.append(Lx)
        x_coords = np.concatenate(node_lists) if node_lists else np.linspace(0, Lx, int(Lx / dx0) + 1)
        return x_coords, np.array(bounds, dtype=np.float64), np.array(dxs, dtype=np.float64), np.array(offsets, dtype=np.int32)
