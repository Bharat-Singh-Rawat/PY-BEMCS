"""
Active duration clock and timer tracking utilities.

Provides an unbiased monotonic clock and an active duration tracker that
automatically freeze and exclude periods when the operating system is in
suspension, sleep, hibernation, or Modern Standby (S0 low-power idle).
"""

import sys
import time
import threading
from typing import Optional

# --------------------------------------------------------------------------- #
#  OS Unbiased Clock Resolver                                                 #
# --------------------------------------------------------------------------- #

_win_unbiased_clock = None

if sys.platform == "win32":
    import ctypes

    try:
        # Windows 10 / Server 2016+ provides sub-microsecond unbiased interrupt time.
        # This count does NOT advance during system sleep or hibernation.
        _kernelbase = ctypes.windll.kernelbase
        _q_unbiased_precise = _kernelbase.QueryUnbiasedInterruptTimePrecise

        def _get_win_unbiased_time_precise() -> float:
            t = ctypes.c_ulonglong()
            _q_unbiased_precise(ctypes.byref(t))
            return t.value * 1e-7  # 100-nanosecond units to seconds

        # Test call to ensure function is present and callable
        _get_win_unbiased_time_precise()
        _win_unbiased_clock = _get_win_unbiased_time_precise
    except Exception:
        try:
            # Fallback for Windows 7 / 8
            _kernel32 = ctypes.windll.kernel32
            _q_unbiased = _kernel32.QueryUnbiasedInterruptTime

            def _get_win_unbiased_time() -> float:
                t = ctypes.c_ulonglong()
                _q_unbiased(ctypes.byref(t))
                return t.value * 1e-7

            _get_win_unbiased_time()
            _win_unbiased_clock = _get_win_unbiased_time
        except Exception:
            _win_unbiased_clock = None


def get_active_clock() -> float:
    """
    Returns the current monotonic timestamp in seconds, using an unbiased
    clock where supported by the operating system.

    - On Windows: Uses QueryUnbiasedInterruptTimePrecise (or QueryUnbiasedInterruptTime),
      which halts during system sleep and hibernation.
    - On Linux: Uses CLOCK_MONOTONIC via time.monotonic(), which excludes suspend time.
    - Fallback: Uses time.perf_counter().
    """
    if _win_unbiased_clock is not None:
        return _win_unbiased_clock()
    elif hasattr(time, 'monotonic'):
        return time.monotonic()
    return time.perf_counter()


# --------------------------------------------------------------------------- #
#  Active Duration Tracker with Heartbeat Gap Rejection                       #
# --------------------------------------------------------------------------- #

class ActiveDurationTracker:
    """
    Tracks active computation duration, ensuring that periods when the system
    is suspended, sleeping, hibernating, or throttled/frozen (such as during
    Modern Standby or lid closure) are NOT accumulated.

    Features
    --------
    1. Unbiased OS monotonic clock:
       Uses QueryUnbiasedInterruptTime on Windows / CLOCK_MONOTONIC on Linux.
    2. Heartbeat Delta Gap Rejection:
       Any delta between ticks or updates exceeding `max_gap` (default 2.0 s)
       is detected as a suspension or sleep event and completely rejected from
       the accumulated elapsed duration.
    3. Thread-safe control:
       Safe to query or update from GUI or worker threads.
    """

    DEFAULT_MAX_GAP = 2.0  # seconds

    def __init__(self, max_gap: float = DEFAULT_MAX_GAP):
        self.max_gap = float(max_gap)
        self.total_elapsed: float = 0.0
        self.last_tick: Optional[float] = None
        self.running: bool = False
        self._lock = threading.Lock()

    def start(self):
        """Starts or resumes active time tracking."""
        with self._lock:
            self.running = True
            self.last_tick = get_active_clock()

    def pause(self):
        """Pauses active time tracking, committing any active slice since last tick."""
        with self._lock:
            if self.running and self.last_tick is not None:
                now = get_active_clock()
                dt = now - self.last_tick
                if 0.0 < dt < self.max_gap:
                    self.total_elapsed += dt
            self.running = False
            self.last_tick = None

    def reset(self, start_running: bool = False):
        """Resets the accumulated active duration to zero."""
        with self._lock:
            self.total_elapsed = 0.0
            if start_running:
                self.running = True
                self.last_tick = get_active_clock()
            else:
                self.running = False
                self.last_tick = None

    def tick(self) -> float:
        """
        Periodic heartbeat update (e.g. called on every 100ms timer timeout).

        Accumulates elapsed time since the previous tick if and only if:
        - The tracker is running.
        - The delta is positive and strictly less than `max_gap`.

        If `dt >= max_gap`, a suspension / sleep / freeze event occurred:
        the gap is discarded and not added to `total_elapsed`.
        """
        with self._lock:
            if not self.running:
                self.last_tick = None
                return self.total_elapsed

            now = get_active_clock()
            if self.last_tick is not None:
                dt = now - self.last_tick
                if 0.0 < dt < self.max_gap:
                    self.total_elapsed += dt
                # dt >= max_gap: suspension gap! Discarded.

            self.last_tick = now
            return self.total_elapsed

    def get_duration(self) -> float:
        """
        Returns the total accumulated active duration in seconds, including
        the active sub-tick delta if currently running.
        """
        with self._lock:
            elapsed = self.total_elapsed
            if self.running and self.last_tick is not None:
                now = get_active_clock()
                dt = now - self.last_tick
                if 0.0 < dt < self.max_gap:
                    elapsed += dt
            return elapsed

