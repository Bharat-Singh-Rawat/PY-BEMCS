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
    DivergencePlaneDialog,
    ExportParametersDialog,
    ExportParticleParametersDialog,
    UpstreamGapDialog,
)
from physics_engine import compute_debye_upstream_gap
from diagnostics import (
    CrossSectionViewerWindow,
    IEDFWindow,
    PPCWindow,
    PhysicalConstraintsWindow,
    TotalChargeMonitorWindow,
    PerformanceMonitorWindow,
    MeshWindow,
    DisplacementSubstepsWindow,
    FieldDiagnosticWindow,
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
                    settings = getattr(self, 'upstream_gap_settings', None)
                    if not (settings and settings.get("method") == "custom"):
                        if settings and settings.get("method") == "ratio":
                            ratio = float(settings.get("ratio", 0.75))
                            base = settings.get("ratio_base", "screen_r")
                            if base == "screen_r":
                                screen_r = self.grid_widgets[0]["r"].value() if self.grid_widgets else 0.80
                                new_gap = round(ratio * screen_r, 3)
                            else:
                                n0    = self.inputs["n0_plasma"].value()
                                Te_up = self.inputs["Te_up"].value()
                                if n0 > 0 and Te_up > 0:
                                    debye_mm = (8.854e-12 * Te_up / (1.6e-19 * n0 * 0.61))**0.5 * 1e3
                                    new_gap = round(ratio * debye_mm, 3)
                                else:
                                    new_gap = self._last_auto_gap
                        else:
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

    def open_mesh_window(self):
        """Open Computational Domain & Mesh Grid Inspector window."""
        if getattr(self, 'mesh_window', None) is None:
            self.mesh_window = MeshWindow(self)
        self.mesh_window.show()
        self.mesh_window.raise_()
        self.mesh_window.activateWindow()
        if hasattr(self, 'sim'):
            params = self.get_params() if hasattr(self, 'get_params') else None
            self.mesh_window.update_plot(self.sim, params=params)

    def open_substep_window(self):
        """Open Displacement Substeps monitor window."""
        if getattr(self, 'substep_window', None) is None:
            self.substep_window = DisplacementSubstepsWindow(self)
        self.substep_window.show()
        self.substep_window.raise_()
        self.substep_window.activateWindow()
        if hasattr(self, 'sim'):
            self.substep_window.update_plot(self.sim)

    def open_field_window(self):
        """Open Electric Field & Potential Diagnostics window."""
        if getattr(self, 'field_window', None) is None:
            self.field_window = FieldDiagnosticWindow(self)
        self.field_window.show()
        self.field_window.raise_()
        self.field_window.activateWindow()
        if hasattr(self, 'sim'):
            self.field_window.update_plot(self.sim)

    def get_divergence_plane_limits(self):
        """Return (x_min, x_max) in mm defining the allowed range for the divergence evaluation plane."""
        if (
            hasattr(self, 'sim')
            and hasattr(self.sim, 'grid_x_ends')
            and len(self.sim.grid_x_ends) > 0
            and hasattr(self.sim, 'Lx')
            and self.sim.Lx > 0
        ):
            x_min = float(self.sim.grid_x_ends[-1])
            x_max = float(self.sim.Lx)
        else:
            upstream = (
                self.inputs["upstream_gap_mm"].value()
                if hasattr(self, 'inputs') and "upstream_gap_mm" in self.inputs
                else 0.6
            )
            grids_thickness = 0.0
            if hasattr(self, 'grid_widgets') and self.grid_widgets:
                for i, gw in enumerate(self.grid_widgets):
                    t = gw["t"].value()
                    gap = gw["gap"].value() if i < len(self.grid_widgets) - 1 else 0.0
                    grids_thickness += t + gap
            x_min = round(upstream + grids_thickness, 3)
            auto_Lx, _ = self.compute_grid_domain_size() if hasattr(self, 'compute_grid_domain_size') else (20.0, 3.0)
            x_max = float(getattr(self, 'adv_params', {}).get('Lx', auto_Lx))

        if x_max <= x_min:
            x_max = round(x_min + 1.0, 3)

        return round(x_min, 3), round(x_max, 3)

    def open_divergence_plane_dialog(self):
        """Open the modal dialog to configure the axial divergence evaluation plane."""
        x_min, x_max = self.get_divergence_plane_limits()
        current_x = getattr(self, 'div_plane_x', None)
        if current_x is None:
            current_x = round(x_min + 0.5 * (x_max - x_min), 3)

        dialog = DivergencePlaneDialog(current_x, x_min, x_max, self)
        if dialog.exec_() == QDialog.Accepted:
            new_plane_x = dialog.get_plane_x()
            self.set_divergence_plane(new_plane_x)

    def open_export_parameters_dialog(self):
        """Open the checklist popup to configure which data columns to export."""
        current_sel = getattr(self, 'export_selected_params', None)
        dialog = ExportParametersDialog(current_sel, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.export_selected_params = dialog.get_selected_params()
            count = sum(1 for v in self.export_selected_params.values() if v)
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(f"Export parameters updated: {count} active.")

    def open_export_particle_parameters_dialog(self):
        """Open the checklist popup to configure which particle data columns to export."""
        current_sel = getattr(self, 'export_selected_particle_params', None)
        dialog = ExportParticleParametersDialog(current_sel, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.export_selected_particle_params = dialog.get_selected_params()
            count = sum(1 for v in self.export_selected_particle_params.values() if v)
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(f"Particle export parameters updated: {count} active.")

    def open_upstream_gap_dialog(self):
        """Open modal dialog to view/modify the upstream gap calculation ratio or set a custom value."""
        current_gap = self.inputs["upstream_gap_mm"].value() if (hasattr(self, 'inputs') and "upstream_gap_mm" in self.inputs) else 0.0
        screen_r = self.grid_widgets[0]["r"].value() if getattr(self, 'grid_widgets', None) else 0.80
        n0 = self.inputs["n0_plasma"].value() if (hasattr(self, 'inputs') and "n0_plasma" in self.inputs) else 1e18
        Te_up = self.inputs["Te_up"].value() if (hasattr(self, 'inputs') and "Te_up" in self.inputs) else 3.0
        bulk = self.adv_params.get("entire_bulk_plasma", False) if hasattr(self, 'adv_params') else False
        saved_settings = getattr(self, 'upstream_gap_settings', None)

        dialog = UpstreamGapDialog(
            current_gap=current_gap,
            screen_r=screen_r,
            n0=n0,
            Te_up=Te_up,
            is_bulk=bulk,
            saved_settings=saved_settings,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            new_gap, settings = dialog.get_result()
            self.upstream_gap_settings = settings
            self._last_auto_gap = new_gap
            if hasattr(self, 'inputs') and "upstream_gap_mm" in self.inputs:
                self.inputs["upstream_gap_mm"].blockSignals(True)
                self.inputs["upstream_gap_mm"].setValue(new_gap)
                self.inputs["upstream_gap_mm"].blockSignals(False)
            if hasattr(self, 'lbl_status'):
                method_desc = "ratio" if settings.get("method") == "ratio" else "custom value"
                self.lbl_status.setText(f"Upstream gap set to {new_gap:.3f} mm via {method_desc}.")



