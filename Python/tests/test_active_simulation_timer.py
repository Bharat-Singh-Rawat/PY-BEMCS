"""
Tests for active simulation timing, unbiased clock, and suspension gap rejection.
Ensures that system sleep, suspension, and hibernation do not artificially inflate
the simulation duration or sweep scenario timers.
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from physics.active_clock import get_active_clock, ActiveDurationTracker
from gui.app.sim_control import SimControlMixin


class DummyMainWindow(SimControlMixin):
    """Minimal dummy window implementing SimControlMixin for unit testing."""

    def __init__(self):
        self.sim_isRunning = False
        self.sim_wall_elapsed = 0.0
        self.sim_wall_start_time = None
        self._ensure_active_timer()
        self.lbl_sim_timer = MagicMock()
        self.lbl_sim_status_badge = MagicMock()

    def is_simulation_running(self):
        return getattr(self, 'sim_isRunning', False)


def test_get_active_clock_returns_monotonic_float():
    t1 = get_active_clock()
    assert isinstance(t1, float)
    assert t1 > 0
    time.sleep(0.01)
    t2 = get_active_clock()
    assert t2 >= t1


def test_active_duration_tracker_normal_flow():
    tracker = ActiveDurationTracker(max_gap=1.0)
    assert tracker.get_duration() == 0.0
    assert not tracker.running

    tracker.start()
    assert tracker.running
    time.sleep(0.05)
    tracker.tick()
    d1 = tracker.get_duration()
    assert 0.04 <= d1 <= 0.25

    tracker.pause()
    assert not tracker.running
    time.sleep(0.05)
    d2 = tracker.get_duration()
    assert abs(d2 - d1) < 0.005  # Paused, so no time added

    tracker.start()
    time.sleep(0.05)
    tracker.tick()
    d3 = tracker.get_duration()
    assert d3 > d2

    tracker.reset()
    assert tracker.get_duration() == 0.0
    assert not tracker.running


def test_active_duration_tracker_rejects_10h_suspension_gap():
    """
    Simulate a user closing laptop screen for 10 hours (36,000 seconds).
    Verify that the 10-hour hiatus is completely discarded from the duration.
    """
    tracker = ActiveDurationTracker(max_gap=2.0)
    tracker.start()

    # Active running for 0.1s
    with patch("physics.active_clock.get_active_clock") as mock_clock:
        mock_clock.return_value = 1000.0
        tracker.start()

        # Step 1: 0.1s later (normal tick)
        mock_clock.return_value = 1000.1
        tracker.tick()
        assert abs(tracker.get_duration() - 0.1) < 1e-4

        # Step 2: User closes the laptop screen for 10 hours (36,000 seconds)
        # Clock jumps to 1000.1 + 36000.0 = 37000.1
        mock_clock.return_value = 37000.1
        tracker.tick()

        # Duration MUST NOT be 10 hours! It should remain 0.1s!
        assert abs(tracker.get_duration() - 0.1) < 1e-4

        # Step 3: Simulation resumes actively for another 0.1s
        mock_clock.return_value = 37000.2
        tracker.tick()
        assert abs(tracker.get_duration() - 0.2) < 1e-4


def test_sim_control_mixin_suspension_rejection():
    """
    Verify that SimControlMixin's update_simulation_timer and get_simulation_duration
    do not advance by 10 hours when the system wakes up from suspension.
    """
    win = DummyMainWindow()
    assert win.get_simulation_duration() == 0.0

    with patch("physics.active_clock.get_active_clock") as mock_clock:
        mock_clock.return_value = 5000.0
        win.sim_isRunning = True
        win._resume_simulation_timer()

        # Normal active ticks
        mock_clock.return_value = 5000.1
        win.update_simulation_timer()
        mock_clock.return_value = 5000.2
        win.update_simulation_timer()

        d_active = win.get_simulation_duration()
        assert 0.19 <= d_active <= 0.21

        # Laptop screen closed for 10 hours (36,000s)
        mock_clock.return_value = 5000.2 + 36000.0  # 41000.2
        win.update_simulation_timer()

        # Duration should STILL be ~0.2s, definitely NOT 10 hours (36000s)
        d_after_wake = win.get_simulation_duration()
        assert d_after_wake < 1.0
        assert abs(d_after_wake - 0.2) < 1e-3

        # UI label check
        win.lbl_sim_timer.setText.assert_called()
        last_label_text = win.lbl_sim_timer.setText.call_args[0][0]
        assert "10:" not in last_label_text
        assert "00:00:00.2" in last_label_text


def test_sim_control_mixin_reset_timer():
    win = DummyMainWindow()
    win.sim_isRunning = True
    win._resume_simulation_timer()
    win._active_sim_timer.total_elapsed = 42.0

    win.reset_simulation_timer()
    assert win.get_simulation_duration() < 0.01
    assert win.sim_wall_elapsed < 0.01

    # Also test reset when not running
    win.sim_isRunning = False
    win._active_sim_timer.total_elapsed = 15.0
    win.reset_simulation_timer()
    assert win.get_simulation_duration() == 0.0
