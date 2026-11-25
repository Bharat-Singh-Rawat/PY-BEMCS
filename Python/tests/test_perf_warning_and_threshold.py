import os
import sys
import unittest
import numpy as np
from io import StringIO
from unittest.mock import patch

from diagnostics.performance_monitor import PerformanceMonitor, StepDiagnostics
from diagnostics.perf_window import PerformanceMonitorWindow

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)


class DummySim:
    def __init__(self):
        self.iteration = 10
        self.dt = 1e-9
        self.num_p = 100
        self.num_e = 100
        self.max_p = 1000
        self.last_poisson_status = 'converged'
        self.last_poisson_iters = 5
        self.last_poisson_delta_V = 0.02
        self.last_poisson_rms = 0.005
        self.min_ppc_threshold = 3
        self._perf_monitor = None


class TestPerfWarningAndThreshold(unittest.TestCase):
    def test_no_perf_warning_terminal_message_on_exceeded_limits(self):
        """Verify that PerformanceMonitor does not print [PERF WARNING] on long step or high RAM."""
        mon = PerformanceMonitor(warn_step_time_ms=500.0, warn_memory_mb=100.0)
        sim = DummySim()

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # Step taking 1500 ms (exceeds 500 ms threshold)
            mon.record_step(sim, step_wall_time_s=1.5)
            output = mock_stdout.getvalue()

            self.assertNotIn("[PERF WARNING]", output)
            self.assertEqual(len(mon.history), 1)
            self.assertAlmostEqual(mon.history[0].wall_time_ms, 1500.0)

    def test_perf_window_plot4_threshold_line_and_legend(self):
        """Verify that PerformanceMonitorWindow includes the threshold line in Plot 4."""
        sim = DummySim()
        mon = PerformanceMonitor(warn_step_time_ms=800.0, warn_memory_mb=1500.0)
        sim._perf_monitor = mon

        # Record a couple steps
        mon.record_step(sim, step_wall_time_s=0.2)
        mon.record_step(sim, step_wall_time_s=0.9)

        win = PerformanceMonitorWindow()
        self.assertIn('p4_warn', win.curve_visibility)
        self.assertTrue(win.curve_visibility['p4_warn'])

        win.update_plot(sim)

        # Verify ax4 has the warn limit threshold line
        ax4 = win.ax4
        # Check lines in ax4: one for dt_ms, one axhline for warn threshold
        self.assertGreaterEqual(len(ax4.lines), 2)

        # Check legend labels on ax4
        legend = ax4.get_legend()
        self.assertIsNotNone(legend)
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertTrue(any("Warn Limit (800 ms)" in lbl or "800" in lbl for lbl in labels))

    def test_export_csv_and_txt(self):
        """Verify export to both .csv and .txt in PerformanceMonitor and PerformanceMonitorWindow."""
        import tempfile

        sim = DummySim()
        mon = PerformanceMonitor()
        sim._perf_monitor = mon
        mon.record_step(sim, step_wall_time_s=0.05)
        mon.record_step(sim, step_wall_time_s=0.08)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "perf.csv")
            txt_path = os.path.join(tmpdir, "perf.txt")

            # Direct export_csv and export_txt
            mon.export_csv(csv_path)
            mon.export_txt(txt_path)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(txt_path))

            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()
                self.assertIn("iteration,sim_time_s", csv_content)
                self.assertIn(",", csv_content)

            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_content = f.read()
                self.assertIn("iteration\tsim_time_s", txt_content)
                self.assertIn("\t", txt_content)

            # Test export_data automatic routing
            csv_auto = os.path.join(tmpdir, "auto.csv")
            txt_auto = os.path.join(tmpdir, "auto.txt")
            mon.export_data(csv_auto)
            mon.export_data(txt_auto)
            with open(csv_auto, 'r', encoding='utf-8') as f:
                self.assertIn(",", f.readline())
            with open(txt_auto, 'r', encoding='utf-8') as f:
                self.assertIn("\t", f.readline())

            # Test window _export_data logic with mock QFileDialog
            win = PerformanceMonitorWindow()
            win._last_sim = sim

            mock_txt_dest = os.path.join(tmpdir, "window_export.txt")
            with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName", return_value=(mock_txt_dest, "Text Files (*.txt)")):
                with patch("PyQt5.QtWidgets.QMessageBox.information"):
                    win._export_data()
            self.assertTrue(os.path.exists(mock_txt_dest))
            with open(mock_txt_dest, 'r', encoding='utf-8') as f:
                self.assertIn("\t", f.readline())


if __name__ == '__main__':
    unittest.main()
