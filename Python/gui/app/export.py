"""
Export mixin for CSV telemetry, particle kinetics, GIF recording, and PyInstaller exe generation.
"""
import os
import sys
import time
import csv
import numpy as np
from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QFileDialog, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage

from ..bootstrap import _safe_start_dir
from ..workers.pyinstaller_worker import PyInstallerWorker


class ExportMixin:
    """Mixin managing file exports and binary compilation."""

    def _create_progress_dialog(self, title, label, total):
        progress = QProgressDialog(label, "Cancel", 0, max(1, total), self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        return progress

    def export_csv(self):
        n = len(self.iter_history)
        if n == 0:
            QMessageBox.warning(self, "No Data", "No simulation history data to export.")
            return

        startdir = _safe_start_dir()
        if not os.access(startdir, os.W_OK):
            startdir = os.path.expanduser("~")
        suggested = os.path.join(startdir, time.strftime("pybemcs_%Y%m%d%H%M%S.csv"))

        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Data", suggested, "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
        )
        if not file_name:
            return

        if "(*.txt)" in selected_filter and file_name.lower().endswith(".csv"):
            file_name = file_name[:-4] + ".txt"
        elif not os.path.splitext(file_name)[1]:
            if "(*.txt)" in selected_filter:
                file_name += ".txt"
            else:
                file_name += ".csv"

        was_running = self.sim_isRunning
        self.sim_isRunning = False

        progress = self._create_progress_dialog("Exporting Data", "Preparing export...", n)
        cancelled = False

        try:
            with open(file_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = [
                    'iteration', 't_sim_s',
                    'minpotential',
                    'beamdivergence°',
                    'beamdivergence_mid_plume°',
                    'transparency',
                    'ion_current_exit_A',
                    'ion_current_exit_avg_A',
                    'transmitted_ions_step',
                    'total_active_cells',
                    f'cells_less_than_{getattr(self.sim, "min_ppc_threshold", 3)}_macroparticles'
                ]
                for i in range(len(self.T_histories)):
                    header.append(f"grid_{i+1}_temp_K")
                writer.writerow(header)

                chunk_size = max(1, n // 100)
                for j in range(n):
                    if progress.wasCanceled():
                        cancelled = True
                        break

                    row = [
                        self.iter_history[j],
                        self.time_history[j] if j < len(self.time_history) else '',
                        self.ebs_history[j],
                        self.div_history[j],
                        self.div_mid_history[j] if j < len(self.div_mid_history) else '',
                        self.transparency_history[j] if j < len(self.transparency_history) else '',
                        self.ion_current_exit_history[j] if j < len(self.ion_current_exit_history) else '',
                        self.ion_current_exit_avg_history[j] if j < len(self.ion_current_exit_avg_history) else '',
                        self.transmitted_ions_history[j] if j < len(self.transmitted_ions_history) else '',
                        self.active_cells_history[j] if j < len(self.active_cells_history) else '',
                        self.low_ppc_cells_history[j] if j < len(self.low_ppc_cells_history) else ''
                    ]
                    for i in range(len(self.T_histories)):
                        row.append(self.T_histories[i][j] if j < len(self.T_histories[i]) else "")
                    writer.writerow(row)

                    if (j + 1) % chunk_size == 0 or j == n - 1:
                        pct = int(((j + 1) / n) * 100)
                        progress.setValue(j + 1)
                        progress.setLabelText(f"Writing row {j+1} of {n} ({pct}%)...")
                        self.lbl_status.setText(f"Exporting data: {j+1}/{n} ({pct}%)...")
                        QApplication.processEvents()

            if cancelled:
                if os.path.exists(file_name):
                    try:
                        os.remove(file_name)
                    except Exception:
                        pass
                self.lbl_status.setText("Data export cancelled.")
                QMessageBox.information(self, "Cancelled", "Data export was cancelled.")
            else:
                progress.setValue(n)
                self.lbl_status.setText(f"Data saved: {file_name}")
                QMessageBox.information(self, "Success", f"Data exported:\n{file_name}")
        except Exception as e:
            self.lbl_status.setText("Data export failed.")
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{e}")
        finally:
            progress.close()
            self.sim_isRunning = was_running

    def exporttrackingdata(self):
        if len(self.tracking_buffer) == 0:
            QMessageBox.warning(self, "No Data", "No particle tracking data recorded.")
            return

        startdir = _safe_start_dir()
        if not os.access(startdir, os.W_OK):
            startdir = os.path.expanduser("~")

        suggested = os.path.join(startdir, time.strftime("pybemcs_particles_%Y%m%d%H%M%S.csv"))
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Particle Data", suggested, "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
        )
        if not filename:
            return

        if "(*.txt)" in selected_filter and filename.lower().endswith(".csv"):
            filename = filename[:-4] + ".txt"
        elif not os.path.splitext(filename)[1]:
            if "(*.txt)" in selected_filter:
                filename += ".txt"
            else:
                filename += ".csv"

        was_running = self.sim_isRunning
        self.sim_isRunning = False

        try:
            data = np.vstack(self.tracking_buffer)
            n_rows = len(data)
            ncols = data.shape[1] if data.ndim > 1 else 0

            if ncols == 8:
                header = ['time_s', 'x_mm', 'y_mm', 'vx_ms', 'vy_ms', 'vz_ms', 'energy_eV', 'type']
            elif ncols == 6:
                header = ['x_mm', 'y_mm', 'vx_ms', 'vy_ms', 'vz_ms', 'isCEX']
            else:
                header = [f"col_{c}" for c in range(ncols)]

            progress = self._create_progress_dialog("Exporting Particle Data", "Writing particle tracking data...", n_rows)
            cancelled = False

            chunk_size = max(500, n_rows // 100)
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for idx in range(0, n_rows, chunk_size):
                    if progress.wasCanceled():
                        cancelled = True
                        break

                    chunk = data[idx:idx + chunk_size]
                    writer.writerows(chunk.tolist())

                    written = min(idx + chunk_size, n_rows)
                    pct = int((written / n_rows) * 100)
                    progress.setValue(written)
                    progress.setLabelText(f"Writing particle {written} of {n_rows} ({pct}%)...")
                    self.lbl_status.setText(f"Exporting particles: {written}/{n_rows} ({pct}%)...")
                    QApplication.processEvents()

            if cancelled:
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except Exception:
                        pass
                self.lbl_status.setText("Particle export cancelled.")
                QMessageBox.information(self, "Cancelled", "Particle data export was cancelled.")
            else:
                progress.setValue(n_rows)
                self.lbl_status.setText(f"Particle data exported: {filename}")
                QMessageBox.information(self, "Success", f"Particle data exported:\n{filename}")
        except PermissionError:
            alt = os.path.join(os.path.expanduser("~"), os.path.basename(filename))
            try:
                data = np.vstack(self.tracking_buffer)
                with open(alt, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(data.tolist())
                QMessageBox.information(self, "Saved to Home",
                    f"Permission denied at original path.\nSaved to:\n{alt}")
            except Exception as e2:
                QMessageBox.critical(self, "Export Error", f"Failed to export particle data:\n{e2}")
        except Exception as e:
            self.lbl_status.setText("Particle export failed.")
            QMessageBox.critical(self, "Export Error", f"Failed to export particle data:\n{e}")
        finally:
            if 'progress' in locals():
                progress.close()
            self.sim_isRunning = was_running

    def save_gif(self):
        if len(self.recorded_frames) == 0:
            QMessageBox.warning(self, "No Frames", "No recorded frames to save.")
            return

        suggested = os.path.join(_safe_start_dir(), time.strftime("pybemcs_%Y%m%d_%H%M%S.gif"))
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Animation", suggested, "GIF Files (*.gif)")
        if not file_name:
            return

        was_running = self.sim_isRunning
        self.sim_isRunning = False

        total = len(self.recorded_frames)
        progress = self._create_progress_dialog("Saving GIF Animation", "Preparing GIF export...", total + 1)
        cancelled = False

        try:
            try:
                from PIL import Image
            except ImportError:
                progress.close()
                QMessageBox.warning(self, "Error", "Install Pillow: pip install Pillow")
                return

            palettized_frames = []
            for i, qpix in enumerate(self.recorded_frames):
                if progress.wasCanceled():
                    cancelled = True
                    break

                qimg = qpix.toImage().convertToFormat(QImage.Format_RGBA8888)
                width = qimg.width()
                height = qimg.height()
                ptr = qimg.bits()
                ptr.setsize(qimg.byteCount())
                arr = np.array(ptr, dtype=np.uint8).reshape(height, width, 4)
                im = Image.fromarray(arr[:, :, :3])
                palettized_frames.append(im.quantize(colors=256, method=Image.Quantize.FASTOCTREE))

                pct = int(((i + 1) / (total + 1)) * 100)
                progress.setValue(i + 1)
                progress.setLabelText(f"Processing frame {i+1} of {total} ({pct}%)...")
                self.lbl_status.setText(f"Saving GIF: {i+1}/{total} ({pct}%)...")
                QApplication.processEvents()

            if not cancelled and palettized_frames:
                progress.setLabelText("Finalizing & writing GIF to disk (98%)...")
                QApplication.processEvents()

                palettized_frames[0].save(
                    file_name,
                    save_all=True,
                    append_images=palettized_frames[1:],
                    duration=50,
                    loop=0,
                    optimize=False,
                )

                progress.setValue(total + 1)
                progress.setLabelText("Export completed (100%)")
                QApplication.processEvents()

            if cancelled:
                if os.path.exists(file_name):
                    try:
                        os.remove(file_name)
                    except Exception:
                        pass
                self.lbl_status.setText("GIF export cancelled.")
                QMessageBox.information(self, "Cancelled", "GIF export was cancelled.")
            else:
                self.recorded_frames.clear()
                self.chk_record.setChecked(False)
                self.chk_record.setText("Record Frames (0)")
                self.lbl_status.setText(f"GIF saved: {file_name}")
                QMessageBox.information(self, "Success", f"GIF saved to:\n{file_name}")
        except Exception as e:
            self.lbl_status.setText("GIF save failed.")
            QMessageBox.critical(self, "Error", f"GIF save failed:\n{e}")
        finally:
            progress.close()
            self.sim_isRunning = was_running

    def build_exe(self):
        """Compile main.py into a standalone executable using PyInstaller."""
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "PyInstaller Not Found",
                "PyInstaller is not installed in the current Python environment.\n"
                "Install it with:\n\n    pip install pyinstaller"
            )
            return

        dest_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder for .exe",
            _safe_start_dir(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not dest_dir:
            return

        # Resolve main.py location
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_path = os.path.join(root_dir, "main.py")
        if not os.path.isfile(script_path):
            QMessageBox.critical(self, "Error", f"Cannot find main.py:\n{script_path}")
            return

        progress = QProgressDialog(
            "Initialising PyInstaller...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Building .exe")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(550)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        self._build_log_lines = []
        worker = PyInstallerWorker(script_path, dest_dir, parent=self)
        self._build_worker = worker

        def _on_line(line):
            self._build_log_lines.append(line)
            stripped = line.strip()
            if stripped:
                display = stripped if len(stripped) <= 80 else stripped[:77] + "..."
                progress.setLabelText(display)
            QApplication.processEvents()

        def _on_pct(pct):
            if not progress.wasCanceled():
                progress.setValue(pct)
            QApplication.processEvents()

        def _on_finished(success, message):
            progress.setValue(100)
            progress.close()
            if success:
                self.lbl_status.setText(f"Build complete -> {message}")
                QMessageBox.information(
                    self, "Build Successful",
                    f"Executable created successfully:\n\n{message}\n\n"
                    "config.json has been copied alongside the .exe (if present)."
                )
            else:
                self.lbl_status.setText("Build failed.")
                log_snippet = "\n".join(self._build_log_lines[-20:])
                QMessageBox.critical(
                    self, "Build Failed",
                    f"{message}\n\n—- Last build output —-\n{log_snippet}"
                )

        def _on_cancel():
            if worker.isRunning():
                worker.terminate()
                worker.wait(2000)
                self.lbl_status.setText("Build cancelled.")

        worker.progress_line.connect(_on_line)
        worker.progress_pct.connect(_on_pct)
        worker.finished.connect(_on_finished)
        progress.canceled.connect(_on_cancel)

        worker.start()
