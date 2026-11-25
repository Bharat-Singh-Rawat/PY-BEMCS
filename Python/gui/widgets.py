"""
Custom Qt widgets for PY-BEMCS GUI.
"""
from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtGui import QValidator


class ScientificSpinBox(QDoubleSpinBox):
    """Custom QDoubleSpinBox supporting scientific notation (e.g., 1e16 or 2.5e16) for input, display, and validation."""
    def textFromValue(self, value):
        s = f"{value:.4g}"
        if "e+" in s:
            return s.replace("e+", "e")
        elif "e-" in s:
            parts = s.split("e-")
            return f"{parts[0]}e-{int(parts[1])}"
        elif "e" not in s and (abs(value) >= 1e4 or (0 < abs(value) <= 1e-3)):
            return f"{value:.3e}".replace("e+0", "e").replace("e+", "e")
        return s

    def valueFromText(self, text):
        clean = text.replace(self.suffix(), "").strip()
        try:
            return float(clean)
        except ValueError:
            return 0.0

    def validate(self, text, pos):
        clean = text.replace(self.suffix(), "").strip()
        if clean in ("", "-", "+", ".", "e", "E", "-e", "+e"):
            return (QValidator.Intermediate, text, pos)
        low = clean.lower()
        if low.endswith("e") or low.endswith("e+") or low.endswith("e-"):
            try:
                prefix = low.rstrip("e+-")
                if prefix:
                    float(prefix)
                return (QValidator.Intermediate, text, pos)
            except ValueError:
                return (QValidator.Invalid, text, pos)
        try:
            float(clean.replace("E", "e"))
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            return (QValidator.Invalid, text, pos)
