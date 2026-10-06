"""
Simulation worker running DigitalTwinSimulator.step off the GUI thread.
"""
import threading
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class SimulationWorker(QThread):
    """Background worker that runs sim.step() off the GUI thread.

    Emits *step_done* with a snapshot dict containing every piece of data the
    GUI needs to update plots and labels.  The snapshot uses **copies** of
    numpy arrays so the GUI can safely consume them while the worker is
    already computing the next step.

    Improvements
    ------------
    - Particle XY columns and energy-per-particle are pre-computed here on
      the worker thread so the GUI thread only needs ``set_offsets`` /
      ``set_array`` calls.
    - Scatter arrays are randomly subsampled to ``MAX_SCATTER`` points when
      the particle count exceeds that threshold, bounding render cost.
    - ``damage_map`` is only copied when it changed (version counter).
    - ``Tmap`` is only copied when thermal is active in the current step.
    - Adaptive ``msleep``: sleeps for at least as long as the GUI took to
      process the last snapshot, preventing queue overflow at low particle
      counts.

    Flow
    ----
    1. GUI calls ``worker.start_sim(params)`` -> sets ``_running = True`` and
       starts the thread (or resumes the loop).
    2. Worker runs ``sim.step(params)`` in a loop, emitting ``step_done``
       after each iteration.
    3. GUI calls ``worker.pause()`` -> sets ``_running = False``; the loop
       finishes the current iteration and stops.
    """

    # Maximum particles sent to scatter plots when no runtime override is active
    MAX_SCATTER = 100_000

    # Runtime scatter limit — set by the GUI Settings dialog:
    #   None  -> hide all particles (display nothing)
    #   0     -> no limit (show every particle)
    #   N > 0 -> show at most N particles (randomly subsampled)
    # When _scatter_limit is not set it falls back to MAX_SCATTER.
    _scatter_limit: "int | None | 'all'" = MAX_SCATTER

    # Signal carries a dict with ALL snapshot data for the GUI update
    step_done = pyqtSignal(dict)
    # Emitted when the worker thread finishes its run loop
    sim_finished = pyqtSignal()

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self._params = {}
        self._running = False
        self._lock = threading.Lock()   # protects _params / _running
        self._last_damage_version = -1   # tracks when damage_map changed
        self._last_damage_copy = None    # last copied damage_map

    # ---- control interface (called from GUI thread) ----
    def start_sim(self, params):
        """Begin or resume the simulation loop with the given *params*."""
        with self._lock:
            self._params = params
            self._running = True
        if not self.isRunning():
            self.start()            # QThread.start()

    def update_params(self, params):
        """Hot-swap parameters while the worker is running."""
        with self._lock:
            self._params = params

    def pause(self):
        """Ask the worker to stop after the current step."""
        with self._lock:
            self._running = False

    @property
    def running(self):
        with self._lock:
            return self._running

    # ---- internal helpers ----
    def _subsample(self, arr, idx):
        """Return arr[idx] if arr is non-empty, else arr."""
        return arr[idx] if len(arr) > 0 else arr

    # ---- thread body ----
    def run(self):
        while True:
            with self._lock:
                if not self._running:
                    break
                params = self._params.copy()  # shallow copy is fine (values are immutable / lists)

            # ----- Heavy computation (runs off the GUI thread) -----
            try:
                step_out = self.sim.step(params)
            except Exception as exc:
                print(f"[SimulationWorker] sim.step() raised: {exc}")
                with self._lock:
                    self._running = False
                break

            if len(step_out) == 5:
                remeshed, min_pot, current_div, T_grids, trans_last_frame = step_out
            else:
                remeshed, min_pot, current_div, T_grids = step_out
                trans_last_frame = 0.0

            # Compute derived quantities while still on the worker thread
            self.sim.get_total_energy()
            self.sim.get_total_charge()
            transparency = self.sim.get_transparency()

            # ---- Strategy 2: pre-compute particle data on worker thread ----
            num_p = self.sim.num_p
            num_e = self.sim.num_e
            m_ion = self.sim.m_ion
            m_e   = self.sim.m_e
            q     = self.sim.q

            # -- Ion arrays (raw slices, before subsampling) --
            if num_p > 0:
                p_isCEX_raw = self.sim.p_isCEX[:num_p].copy()
                p_x_raw  = self.sim.p_x[:num_p].copy()
                p_y_raw  = self.sim.p_y[:num_p].copy()
                p_vx_raw = self.sim.p_vx[:num_p].copy()
                p_vy_raw = self.sim.p_vy[:num_p].copy()
                p_vz_raw = self.sim.p_vz[:num_p].copy()
            else:
                p_isCEX_raw = np.empty(0, dtype=bool)
                p_x_raw = p_y_raw = p_vx_raw = p_vy_raw = p_vz_raw = np.empty(0)

            # -- Subsample ions based on runtime scatter limit --
            scatter_limit = getattr(self, '_scatter_limit', self.MAX_SCATTER)

            if scatter_limit is None:
                # Hide all particles
                p_isCEX = np.empty(0, dtype=bool)
                p_x = p_y = p_vx = p_vy = p_vz = np.empty(0)
                ions_subsampled = False
                ions_subsample_ratio = 0.0
            elif scatter_limit == 0 or num_p <= scatter_limit:
                # No limit or already below threshold — use full arrays
                p_isCEX = p_isCEX_raw
                p_x = p_x_raw; p_y = p_y_raw
                p_vx = p_vx_raw; p_vy = p_vy_raw; p_vz = p_vz_raw
                ions_subsampled = False
                ions_subsample_ratio = 1.0
            else:
                # Subsample to scatter_limit
                ion_idx = np.random.choice(num_p, scatter_limit, replace=False)
                ion_idx.sort()  # keep spatial ordering for visual coherence
                p_isCEX = p_isCEX_raw[ion_idx]
                p_x  = p_x_raw[ion_idx];  p_y  = p_y_raw[ion_idx]
                p_vx = p_vx_raw[ion_idx]; p_vy = p_vy_raw[ion_idx]; p_vz = p_vz_raw[ion_idx]
                ions_subsampled = True
                ions_subsample_ratio = num_p / scatter_limit

            # -- Pre-compute scatter XY and per-particle energy (worker thread) --
            prim_mask = ~p_isCEX if len(p_isCEX) > 0 else np.empty(0, dtype=bool)
            cex_mask  =  p_isCEX if len(p_isCEX) > 0 else np.empty(0, dtype=bool)

            prim_xy = (np.column_stack((p_x[prim_mask], p_y[prim_mask]))
                       if np.any(prim_mask) else np.empty((0, 2)))
            cex_xy  = (np.column_stack((p_x[cex_mask],  p_y[cex_mask]))
                       if np.any(cex_mask)  else np.empty((0, 2)))

            if np.any(prim_mask):
                vsq = p_vx[prim_mask]**2 + p_vy[prim_mask]**2 + p_vz[prim_mask]**2
                e_prim_eV = (0.5 * m_ion * vsq) / q
            else:
                e_prim_eV = np.empty(0)

            if np.any(cex_mask):
                vsq = p_vx[cex_mask]**2 + p_vy[cex_mask]**2 + p_vz[cex_mask]**2
                e_cex_eV = (0.5 * m_ion * vsq) / q
            else:
                e_cex_eV = np.empty(0)

            # -- Electron arrays (subsample if needed) --
            if num_e > 0:
                e_x_raw  = self.sim.e_x[:num_e].copy()
                e_y_raw  = self.sim.e_y[:num_e].copy()
                e_vx_raw = self.sim.e_vx[:num_e].copy()
                e_vy_raw = self.sim.e_vy[:num_e].copy()
            else:
                e_x_raw = e_y_raw = e_vx_raw = e_vy_raw = np.empty(0)

            if scatter_limit is None:
                e_x = e_y = e_vx = e_vy = np.empty(0)
                elec_subsampled = False
            elif scatter_limit == 0 or num_e <= scatter_limit:
                e_x = e_x_raw; e_y = e_y_raw; e_vx = e_vx_raw; e_vy = e_vy_raw
                elec_subsampled = False
            else:
                elec_idx = np.random.choice(num_e, scatter_limit, replace=False)
                e_x  = e_x_raw[elec_idx];  e_y  = e_y_raw[elec_idx]
                e_vx = e_vx_raw[elec_idx]; e_vy = e_vy_raw[elec_idx]
                elec_subsampled = True

            elec_xy = (np.column_stack((e_x, e_y)) if num_e > 0 else np.empty((0, 2)))

            # -- Strategy 2: conditional Tmap copy (only if thermal is active) --
            sim_mode = params.get('sim_mode', 'Both')
            thermal_active = sim_mode in ('Thermal', 'Both')
            Tmap_snap = self.sim.Tmap.copy() if (thermal_active and self.sim.Tmap is not None) else None

            # -- Strategy 2: damage_map — only copy when it changed or shape resized --
            dmg_ver = getattr(self.sim, '_damage_version', 0)
            if (dmg_ver != self._last_damage_version
                    or self._last_damage_copy is None
                    or self._last_damage_copy.shape != self.sim.damage_map.shape):
                self._last_damage_copy = self.sim.damage_map.copy()
                self._last_damage_version = dmg_ver
            damage_snap = self._last_damage_copy

            # Build snapshot dict
            snap = {
                'remeshed':            remeshed,
                'min_pot':             min_pot,
                'current_div':         current_div,
                'current_div_mid':     getattr(self.sim, 'current_div_mid', np.nan),
                'T_grids':             list(T_grids),
                'trans_last_frame':    trans_last_frame,
                'transparency':        transparency,
                'iteration':           self.sim.iteration,
                'dt':                  self.sim.dt,
                'num_p':               num_p,
                'num_e':               num_e,
                'exit_vx_mean':        self.sim.exit_vx_mean,
                'exit_v_mean':         self.sim.exit_v_mean,
                'exit_energy_mean_eV': self.sim.exit_energy_mean_eV,
                'exit_count_step':     self.sim.exit_count_step,
                'exit_ion_current_step': self.sim.exit_ion_current_step,
                'exit_ion_current_avg':  self.sim.exit_ion_current_avg,
                'transmitted_ions_step': self.sim.transmitted_ions_step,
                'total_active_cells':  self.sim.total_active_cells,
                'low_ppc_cells':       self.sim.low_ppc_cells,
                'injection_enabled':   self.sim.injection_enabled,
                'has_active_particles': self.sim.has_active_particles(),
                'm_ion':               m_ion,
                'm_e':                 m_e,
                'q':                   q,
                'Lx':                  self.sim.Lx,
                'Ly':                  self.sim.Ly,
                'dx':                  self.sim.dx,
                'dy':                  self.sim.dy,
                'nx':                  self.sim.nx,
                'ny':                  self.sim.ny,
                # Pre-computed scatter data (Strategy 3)
                'prim_xy':             prim_xy,
                'cex_xy':              cex_xy,
                'elec_xy':             elec_xy,
                'e_prim_eV':           e_prim_eV,
                'e_cex_eV':            e_cex_eV,
                'prim_mask':           prim_mask,
                'cex_mask':            cex_mask,
                # Raw particle arrays (needed for IEDF window)
                'p_isCEX':    p_isCEX_raw,
                'p_vx':       p_vx_raw,
                'p_vy':       p_vy_raw,
                'e_x':        e_x_raw,
                'e_vx':       e_vx_raw,
                'e_vy':       e_vy_raw,
                # Subsampling metadata
                'ions_subsampled':      ions_subsampled,
                'ions_subsample_ratio': ions_subsample_ratio,
                'elec_subsampled':      elec_subsampled,
                # Maps
                'Tmap':       Tmap_snap,
                'isBound':    self.sim.isBound,    # read-only, safe to share
                'mask_grids': self.sim.mask_grids,  # read-only
                'damage_map': damage_snap,
                'X':          self.sim.X,
                'Y':          self.sim.Y,
                'V':          self.sim.V,
                'x_coords':   getattr(self.sim, 'x_coords', None),
                'y_coords':   getattr(self.sim, 'y_coords', None),
                # Perf monitor
                'perf_monitor': self.sim._perf_monitor,
                # params echo-back
                'params':     params,
            }

            self.step_done.emit(snap)

            # Strategy 1 (adaptive sleep): wait at least as long as the GUI
            # took to process the previous snapshot so the event queue never
            # fills up faster than it can be drained.
            gui_ms = max(1, int(params.get('_last_gui_ms', 1)))
            self.msleep(gui_ms)

        self.sim_finished.emit()
