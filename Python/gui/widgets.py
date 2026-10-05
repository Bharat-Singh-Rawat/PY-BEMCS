"""
Custom Qt widgets for PY-BEMCS GUI.
"""
from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtGui import QValidator


class ScientificSpinBox(QDoubleSpinBox):
    """Custom QDoubleSpinBox supporting scientific notation (e.g., 1.000e+17) for input, display, and validation."""
    def textFromValue(self, value):
        return f"{value:.3e}"

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return 0.0

    def validate(self, text, pos):
        try:
            float(text.replace("E", "e"))
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            if text in ("", "-", "+", ".", "e", "E", "-e", "+e"):
                return (QValidator.Intermediate, text, pos)
            return (QValidator.Invalid, text, pos)
