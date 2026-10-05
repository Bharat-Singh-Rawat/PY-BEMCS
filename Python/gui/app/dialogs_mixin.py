"""
Dialog launcher mixin for DigitalTwinApp.
Manages popup dialogs and secondary diagnostics windows.
"""
from PyQt5.QtWidgets import QDialog, QMessageBox

from ..dialogs import (
    GUISettingsDialog,
    AdvancedSettingsDialog,
    BeamSpeciesDialog,
    GridMaterialDialog,
)
from physics_engine import compute_debye_upstream_gap
from diagnostics import (
    CrossSectionViewerWindow,
    IEDFWindow,
    PPCWindow,
    PhysicalConstraintsWindow,
    TotalChargeMonitorWindow,
    PerformanceMonitorWindow,
)


class DialogsMixin:
    """Mixin managing modal dialogs and external diagnostic windows."""

    def open_gui_settings(self):
        """Open the GUI Settings dialog."""
        dialog = GUISettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            new_limit = dialog.get_scatter_limit()
            self._sim_worker._scatter_limit = new_limit

            new_diag_ms = dialog.get_diag_interval_ms()
            self._DIAG_MIN_INTERVAL_MS = new_diag_ms

            if new_limit is None:
                lim_str = "particles hidden"
            elif new_limit == 0:
                lim_str = "all particles shown"
            else:
                lim_str = f"≤ {new_limit:,} particles shown"
            self.lbl_status.setText(
                f"GUI Settings applied — scatter: {lim_str}; "
                f"diag interval: {new_diag_ms} ms"
            )

    def open_advanced_settings(self):
        """Open Advanced Parameters dialog."""
        auto_Lx, auto_Ly = self.compute_grid_domain_size()
        if not getattr(self, '_user_overrode_Lx', False):
            self.adv_params["Lx"] = auto_Lx
        if not getattr(self, '_user_overrode_Ly', False):
            self.adv_params["Ly"] = auto_Ly

        dialog = AdvancedSettingsDialog(self.adv_params, self, default_Lx=auto_Lx, default_Ly=auto_Ly)
        if dialog.exec_() == QDialog.Accepted:
            new_vals = dialog.get_values()
            if abs(new_vals.get("Lx", auto_Lx) - auto_Lx) > 1e-4:
                self._user_overrode_Lx = True
            else:
                self._user_overrode_Lx = False

            if abs(new_vals.get("Ly", auto_Ly) - auto_Ly) > 1e-4:
                self._user_overrode_Ly = True
            else:
                self._user_overrode_Ly = False

            self.adv_params.update(new_vals)

            if hasattr(self, '_last_auto_gap'):
                current = round(self.inputs["upstream_gap_mm"].value(), 3)
                if abs(current - self._last_auto_gap) <= 0.001:
                    if self.adv_params.get("entire_bulk_plasma", False):
                        n0    = self.inputs["n0_plasma"].value()
                        Te_up = self.inputs["Te_up"].value()
                        if n0 > 0 and Te_up > 0:
                            new_gap = round(compute_debye_upstream_gap(n0, Te_up), 3)
                        else:
                            new_gap = self._last_auto_gap
                    else:
                        screen_r = self.grid_widgets[0]["r"].value() if self.grid_widgets else 0.80
                        new_gap = round(0.75 * screen_r, 3)
                    self._last_auto_gap = new_gap
                    self.inputs["upstream_gap_mm"].blockSignals(True)
                    self.inputs["upstream_gap_mm"].setValue(new_gap)
                    self.inputs["upstream_gap_mm"].blockSignals(False)
            QMessageBox.information(
                self, "Settings Updated",
                "Advanced settings updated in memory. Click '1. BUILD DOMAIN' to apply."
            )

    def open_beam_species(self):
        """Open Ion Beam Species dialog."""
        dialog = BeamSpeciesDialog(self.beam_mass_amu, self.beam_charge_state, self)
        if dialog.exec_() == QDialog.Accepted:
            self.beam_mass_amu, self.beam_charge_state = dialog.get_values()
            QMessageBox.information(
                self, "Species Updated",
                f"Beam ion: {self.beam_mass_amu:.3f} amu, charge +{self.beam_charge_state}.\n"
                f"Click '1. BUILD DOMAIN' to apply."
            )

    def open_cs_viewer(self):
        """Open Cross-Section Manager dialog window."""
        if self.cs_viewer_window is None:
            self.cs_viewer_window = CrossSectionViewerWindow(self.cs_store)
        self.cs_viewer_window.show()
        self.cs_viewer_window.raise_()
        self.cs_viewer_window.activateWindow()

    def open_grid_material(self):
        """Open Grid Material dialog."""
        dialog = GridMaterialDialog(self.mat_name, self.mat_props, self)
        if dialog.exec_() == QDialog.Accepted:
            self.mat_name, self.mat_props = dialog.get_values()
            if hasattr(self, 'lbl_material'):
                self.lbl_material.setText(f"Grid Material: {self.mat_name}")
            QMessageBox.information(
                self, "Material Updated",
                f"Grid material: {self.mat_name}\n"
                f"Click '1. BUILD DOMAIN' to apply."
            )

    def open_iedf_window(self):
        """Open Ion and Electron Energy Distribution Function window."""
        if self.iedf_window is None:
            self.iedf_window = IEDFWindow()
        self.iedf_window.show()

    def open_ppc_window(self):
        """Open Particles-Per-Cell monitor window."""
        if self.ppc_window is None:
            self.ppc_window = PPCWindow(self)
        self.ppc_window.show()
        self.ppc_window.raise_()
        self.ppc_window.activateWindow()
        if hasattr(self, 'sim'):
            self.ppc_window.update_plot(self.sim)

    def open_phys_window(self):
        """Open Total Energy and Physical Constraints monitor window."""
        if self.phys_window is None:
            self.phys_window = PhysicalConstraintsWindow(self)
        self.phys_window.show()
        self.phys_window.raise_()
        self.phys_window.activateWindow()
        if hasattr(self, 'sim'):
            self.phys_window.update_plot(self.sim)

    def open_charge_window(self):
        """Open Total Charge Conservation window."""
        if self.charge_window is None:
            self.charge_window = TotalChargeMonitorWindow(self)
        self.charge_window.show()
        self.charge_window.raise_()
        self.charge_window.activateWindow()
        if hasattr(self, 'sim'):
            self.charge_window.update_plot(self.sim)

    def open_perf_window(self):
        """Open Performance and Execution Profiling window."""
        if self.perf_window is None:
            self.perf_window = PerformanceMonitorWindow(self)
        self.perf_window.show()
        self.perf_window.raise_()
        self.perf_window.activateWindow()
        if hasattr(self, 'sim'):
            self.perf_window.update_plot(self.sim)
