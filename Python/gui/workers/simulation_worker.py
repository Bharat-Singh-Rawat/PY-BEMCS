"""
Simulation worker running DigitalTwinSimulator.step off the GUI thread.
"""
import time
import math
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
    MAX_SCATTER = 2000

    # Runtime scatter limit — set by the GUI Settings dialog:
    #   None  -> hide all particles (display nothing)
    #   0     -> no limit (show every particle, capped at 100k for blitted display)
    #   N > 0 -> show at most N particles (randomly subsampled)
    # When _scatter_limit is not set it falls back to MAX_SCATTER.
    _scatter_limit: "int | None | 'all'" = 2000

    # Signal carries a dict with ALL snapshot data for the GUI update
    step_done = pyqtSignal(dict)
    # Emitted when the worker thread finishes its run loop
    sim_finished = pyqtSignal()

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self._params = {}
        self._running = False
        self._run_id = 0
        self._lock = threading.Lock()   # protects _params / _running / _run_id
        self._last_damage_version = -1   # tracks when damage_map changed
        self._last_damage_copy = None    # last copied damage_map
        self._last_snap_time = 0.0
        self._min_snap_interval = 0.033  # 30 FPS target for decoupled GUI rendering
        self._last_emitted_iter = -1     # tracks the latest iteration sent to GUI

    # ---- control interface (called from GUI thread) ----
    def start_sim(self, params, run_id=0):
        """Begin or resume the simulation loop with the given *params* and *run_id*."""
        with self._lock:
            self._params = params
            self._run_id = run_id
            self._running = True
        if hasattr(self, 'sim') and self.sim is not None:
            self.sim._stop_requested = False
        if self.isRunning() and not self.running:
            self.wait(1000)
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

    def stop(self, wait_ms=3000):
        """Cooperatively and cleanly stop the worker thread without killing via terminate()."""
        with self._lock:
            self._running = False
        if hasattr(self, 'sim') and self.sim is not None:
            self.sim._stop_requested = True
        if self.isRunning():
            self.wait(wait_ms)

    @property
    def running(self):
        with self._lock:
            return self._running

    # ---- internal helpers ----
    def _subsample(self, arr, idx):
        """Return arr[idx] if arr is non-empty, else arr."""
        return arr[idx] if len(arr) > 0 else arr

    def _emit_snapshot(self, params, run_id, remeshed, min_pot, current_div, T_grids, trans_last_frame):
        """Construct and emit full snapshot dict to GUI thread."""
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

        # -- Phase 4 Downsampled Async Particle Extraction --
        scatter_limit = getattr(self, '_scatter_limit', params.get('scatter_limit', self.MAX_SCATTER))
        eff_limit = 100000 if (scatter_limit == 0) else scatter_limit

        # Direct Downsampled Extraction for Ions (Zero full-array heap allocations)
        if scatter_limit is None or num_p == 0:
            p_isCEX = np.empty(0, dtype=bool)
            p_x = p_y = p_vx = p_vy = p_vz = np.empty(0, dtype=np.float32)
            ions_subsampled = False
            ions_subsample_ratio = 0.0
        elif eff_limit and num_p > eff_limit:
            stride = max(1, math.ceil(num_p / eff_limit))
            sl = slice(0, num_p, stride)
            p_isCEX = self.sim.p_isCEX[sl].copy()
            p_x  = self.sim.p_x[sl].copy()
            p_y  = self.sim.p_y[sl].copy()
            p_vx = self.sim.p_vx[sl].copy()
            p_vy = self.sim.p_vy[sl].copy()
            p_vz = self.sim.p_vz[sl].copy()
            ions_subsampled = True
            ions_subsample_ratio = num_p / max(1, len(p_x))
        else:
            p_isCEX = self.sim.p_isCEX[:num_p].copy()
            p_x  = self.sim.p_x[:num_p].copy()
            p_y  = self.sim.p_y[:num_p].copy()
            p_vx = self.sim.p_vx[:num_p].copy()
            p_vy = self.sim.p_vy[:num_p].copy()
            p_vz = self.sim.p_vz[:num_p].copy()
            ions_subsampled = False
            ions_subsample_ratio = 1.0

        # Pre-compute scatter XY and per-particle energy on worker thread
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

        # Direct Downsampled Extraction for Electrons
        if scatter_limit is None or num_e == 0:
            e_x = e_y = e_vx = e_vy = np.empty(0, dtype=np.float32)
            elec_subsampled = False
        elif eff_limit and num_e > eff_limit:
            stride_e = max(1, math.ceil(num_e / eff_limit))
            sl_e = slice(0, num_e, stride_e)
            e_x  = self.sim.e_x[sl_e].copy()
            e_y  = self.sim.e_y[sl_e].copy()
            e_vx = self.sim.e_vx[sl_e].copy()
            e_vy = self.sim.e_vy[sl_e].copy()
            elec_subsampled = True
        else:
            e_x  = self.sim.e_x[:num_e].copy()
            e_y  = self.sim.e_y[:num_e].copy()
            e_vx = self.sim.e_vx[:num_e].copy()
            e_vy = self.sim.e_vy[:num_e].copy()
            elec_subsampled = False

        elec_xy = (np.column_stack((e_x, e_y)) if len(e_x) > 0 else np.empty((0, 2)))
        if len(e_vx) > 0:
            vsq_e = e_vx**2 + e_vy**2
            e_elec_eV = (0.5 * m_e * vsq_e) / q
        else:
            e_elec_eV = np.empty(0, dtype=np.float32)

        # IEDF Window Data (Only extracted when window is actively open)
        iedf_open = params.get('iedf_visible', False)
        if iedf_open and num_p > 0:
            stride_iedf = max(1, num_p // 10000)
            p_isCEX_iedf = self.sim.p_isCEX[:num_p:stride_iedf].copy()
            p_vx_iedf    = self.sim.p_vx[:num_p:stride_iedf].copy()
            p_vy_iedf    = self.sim.p_vy[:num_p:stride_iedf].copy()
        else:
            p_isCEX_iedf = np.empty(0, dtype=bool)
            p_vx_iedf = np.empty(0, dtype=np.float32)
            p_vy_iedf = np.empty(0, dtype=np.float32)

        if iedf_open and num_e > 0:
            stride_iedf_e = max(1, num_e // 10000)
            e_x_iedf  = self.sim.e_x[:num_e:stride_iedf_e].copy()
            e_vx_iedf = self.sim.e_vx[:num_e:stride_iedf_e].copy()
            e_vy_iedf = self.sim.e_vy[:num_e:stride_iedf_e].copy()
        else:
            e_x_iedf = np.empty(0, dtype=np.float32)
            e_vx_iedf = np.empty(0, dtype=np.float32)
            e_vy_iedf = np.empty(0, dtype=np.float32)

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
            'n_sub_ion':           getattr(self.sim, 'last_n_sub_ion', 1),
            'n_sub_e':             getattr(self.sim, 'last_n_sub_e', 1),
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
            'e_elec_eV':           e_elec_eV,
            'prim_mask':           prim_mask,
            'cex_mask':            cex_mask,
            # Particle arrays (supplied on-demand for IEDF window)
            'p_isCEX':    p_isCEX_iedf,
            'p_vx':       p_vx_iedf,
            'p_vy':       p_vy_iedf,
            'e_x':        e_x_iedf,
            'e_vx':       e_vx_iedf,
            'e_vy':       e_vy_iedf,
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
            'run_id':     run_id,
        }

        self.step_done.emit(snap)
        self._last_emitted_iter = self.sim.iteration

    # ---- thread body ----
    def run(self):
        last_step_out = None
        last_params = None
        last_run_id = 0

        while True:
            with self._lock:
                if not self._running:
                    break
                params = self._params.copy()  # shallow copy is fine (values are immutable / lists)
                run_id = self._run_id

            # ----- Heavy computation (runs off the GUI thread) -----
            try:
                step_out = self.sim.step(params)
            except Exception as exc:
                print(f"[SimulationWorker] sim.step() raised: {exc}")
                with self._lock:
                    self._running = False
                break

            with self._lock:
                if not self._running:
                    break

            last_step_out = step_out
            last_params = params
            last_run_id = run_id

            if len(step_out) == 5:
                remeshed, min_pot, current_div, T_grids, trans_last_frame = step_out
            else:
                remeshed, min_pot, current_div, T_grids = step_out
                trans_last_frame = 0.0

            # Check if a snapshot should be generated and emitted to the GUI
            now = time.perf_counter()
            with self._lock:
                is_running = self._running

            # Dynamically adapt snapshot rate to match GUI rendering speed
            last_gui_ms = float(params.get('_last_gui_ms', 20.0))
            eff_snap_interval = max(self._min_snap_interval, (last_gui_ms * 1.1) / 1000.0)
            should_emit_snap = (now - self._last_snap_time >= eff_snap_interval) or (not is_running) or remeshed

            if should_emit_snap:
                self._last_snap_time = now
                self._emit_snapshot(params, run_id, remeshed, min_pot, current_div, T_grids, trans_last_frame)

            # Cooperative yield to prevent starving GUI and background threads
            self.msleep(1)

        # Guarantee that the very latest simulated iteration is delivered to GUI on pause
        if not getattr(self.sim, '_stop_requested', False):
            if last_step_out is not None and self._last_emitted_iter != self.sim.iteration:
                if len(last_step_out) == 5:
                    remeshed, min_pot, current_div, T_grids, trans_last_frame = last_step_out
                else:
                    remeshed, min_pot, current_div, T_grids = last_step_out
                    trans_last_frame = 0.0
                try:
                    self._emit_snapshot(last_params, last_run_id, remeshed, min_pot, current_div, T_grids, trans_last_frame)
                except Exception:
                    pass

        self.sim_finished.emit()
