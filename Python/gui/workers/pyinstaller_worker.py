"""
PyInstaller background compilation worker.
"""
import os
import sys
import shutil
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal


class PyInstallerWorker(QThread):
    """Background worker thread that executes PyInstaller builds and streams progress to the GUI."""
    progress_line = pyqtSignal(str)   # each new stdout/stderr line
    progress_pct  = pyqtSignal(int)   # 0-100 estimated percentage
    finished      = pyqtSignal(bool, str)  # (success, message)

    # PyInstaller log lines that signal meaningful milestones (order matters)
    _MILESTONES = [
        ("checking python",         5),
        ("running analysis",        10),
        ("processing module hooks", 20),
        ("looking for implied",     30),
        ("copying dependencies",    40),
        ("building pyz",            50),
        ("building pkg",            60),
        ("building exe",            70),
        ("appending archive",       80),
        ("building exe from",       85),
        ("exe successfully",        95),
    ]

    def __init__(self, script_path: str, dest_dir: str, parent=None):
        super().__init__(parent)
        self._script_path = script_path
        self._dest_dir    = dest_dir

    def run(self):
        script_dir  = os.path.dirname(os.path.abspath(self._script_path))
        script_name = os.path.splitext(os.path.basename(self._script_path))[0]

        # Build inside a temporary work directory so we don't pollute the source tree
        build_dir = os.path.join(script_dir, "_exe_build_tmp")
        dist_dir  = os.path.join(build_dir,  "dist")
        work_dir  = os.path.join(build_dir,  "work")

        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--name",    script_name,
            "--distpath", dist_dir,
            "--workpath", work_dir,
            "--specpath", build_dir,
            "--noconfirm",
            self._script_path,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=script_dir,
            )

            self.progress_pct.emit(2)
            current_pct = 2

            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                self.progress_line.emit(line)
                lower = line.lower()
                for keyword, pct in self._MILESTONES:
                    if keyword in lower and pct > current_pct:
                        current_pct = pct
                        self.progress_pct.emit(current_pct)
                        break

            proc.wait()

            if proc.returncode != 0:
                self.finished.emit(False, "PyInstaller exited with errors. See log above.")
                return

            # Locate the produced executable
            exe_name = script_name + (".exe" if sys.platform == "win32" else "")
            src_exe  = os.path.join(dist_dir, exe_name)
            if not os.path.isfile(src_exe):
                self.finished.emit(False, f"Build succeeded but executable not found:\n{src_exe}")
                return

            # Copy to user-selected destination
            dst_exe = os.path.join(self._dest_dir, exe_name)
            shutil.copy2(src_exe, dst_exe)

            # Also copy config.json next to the exe if it exists
            cfg_src = os.path.join(script_dir, "config.json")
            if os.path.isfile(cfg_src):
                shutil.copy2(cfg_src, os.path.join(self._dest_dir, "config.json"))

            self.progress_pct.emit(100)
            self.finished.emit(True, dst_exe)

        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            # Clean up temporary build artefacts
            try:
                if os.path.isdir(build_dir):
                    shutil.rmtree(build_dir, ignore_errors=True)
            except Exception:
                pass
