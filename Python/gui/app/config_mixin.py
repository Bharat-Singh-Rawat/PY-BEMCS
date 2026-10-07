"""
Configuration synchronization, Debye length calculations, and domain sizing mixin.
"""
import os
import numpy as np
from PyQt5.QtWidgets import QMessageBox

from ..config_io import _config_path, load_json_config, load_cross_sections_from_config
from ..dialogs import GridMaterialDialog
from physics_engine import compute_debye_upstream_gap


class ConfigMixin:
    """Mixin managing configuration loading, defaults, and parameter synchronization."""

    def _apply_defaults(self):
        """Populate the UI with safe hardcoded defaults when no config.json is present."""
        self.beam_mass_amu  = 131.293   # Xenon
        self.beam_charge_state = 1

        self.mat_name  = "Molybdenum"
        self.mat_props = GridMaterialDialog.PRESETS["Molybdenum"].copy()
        if hasattr(self, 'lbl_material'):
            self.lbl_material.setText(f"Grid Material: {self.mat_name}")

        self.adv_params = {
            "V_plasma_offset":  20.0,
            "m_e_ratio":      1000.0,
            "entire_bulk_plasma": False,  # default: presheath mode
            "dx_over_debye": 0.8,
            r"\deltax/debye_length": 0.8,
        }

        self.inputs["n0_plasma"].setValue(1e17)
        self.inputs["Te_up"].setValue(3.0)
        self.inputs["Ti"].setValue(0.5)
        self.inputs["Tn"].setValue(300.0)
        self.inputs["n0"].setValue(1e18)
        self.inputs["Accel"].setValue(1.0)
        self.inputs["Thresh"].setValue(1e6)
        self.inputs["target_ppc"].setValue(40.0)
        self.inputs["inj_time_µs"].setValue(0.0)

        _default_screen_r = 0.80
        _auto_gap = round(0.75 * _default_screen_r, 3)
        self._last_auto_gap = _auto_gap
        self.inputs["upstream_gap_mm"].setValue(_auto_gap)

        self.chk_neut_match_ion.setChecked(False)
        self.inputs["neut_rate"].setValue(30.0)
        self.inputs["neut_rate"].setEnabled(True)
        self.inputs["Te"].setValue(2.0)
        self.inputs["neut_r"].setValue(1.5)

        self.chk_rf.setChecked(False)
        self.spin_rf_freq.setValue(13.56)
        self.spin_rf_amp.setValue(0.0)

        self.combo_geometry.setCurrentText("half_hole")

        self.clear_grid_ui()
        self.add_grid_ui(1100.0, 0.38, 0.64, 0.80, 0.0)   # Screen grid
        self.add_grid_ui(-200.0, 0.38, 0.64, 0.70, 0.0)   # Accel grid
        self.add_grid_ui(  0.0,  0.38, 2.00, 0.75, 0.0)   # Deccel grid

        auto_Lx, auto_Ly = self.compute_grid_domain_size()
        self.adv_params["Lx"] = auto_Lx
        self.adv_params["Ly"] = auto_Ly
        self._user_overrode_Lx = False
        self._user_overrode_Ly = False

        x_exit = _auto_gap + (0.38 + 0.64) + (0.38 + 0.64) + 0.38
        auto_neut_x = round(x_exit + 0.9 * (auto_Lx - x_exit), 3)
        self._last_auto_neut_x = auto_neut_x
        self.inputs["neut_x"].setValue(auto_neut_x)

        self.lbl_status.setText("Status: No config.json found — using default values.")

    def compute_grid_domain_size(self):
        """Compute default Lx and Ly directly from grid geometry and simulation settings."""
        if hasattr(self, 'grid_widgets') and self.grid_widgets:
            grids = [{
                "t": gw["t"].value(),
                "gap": gw["gap"].value(),
                "r": gw["r"].value()
            } for gw in self.grid_widgets]
        elif hasattr(self, 'config') and self.config and "grids" in self.config:
            grids = self.config["grids"]
        else:
            grids = []

        if grids:
            total_grid_thickness = sum(float(g["t"]) + float(g["gap"]) for g in grids)
            screen_r = float(grids[0]["r"])
        else:
            total_grid_thickness = 0.0
            screen_r = 0.80

        upstream_gap = 0.0
        if hasattr(self, 'inputs') and "upstream_gap_mm" in self.inputs:
            upstream_gap = self.inputs["upstream_gap_mm"].value()
        if upstream_gap <= 0.0:
            bulk = self.adv_params.get("entire_bulk_plasma", False) if hasattr(self, 'adv_params') else False
            if bulk and hasattr(self, 'inputs') and "n0_plasma" in self.inputs and "Te_up" in self.inputs:
                upstream_gap = compute_debye_upstream_gap(
                    self.inputs["n0_plasma"].value(),
                    self.inputs["Te_up"].value()
                )
            else:
                upstream_gap = 0.75 * screen_r

        auto_Lx = round(upstream_gap + total_grid_thickness, 3)

        geom = "half_hole"
        if hasattr(self, 'combo_geometry'):
            geom = self.combo_geometry.currentText()
        elif hasattr(self, 'config') and self.config and "simulation" in self.config:
            geom = self.config["simulation"].get("geometry", "half_hole")

        pitch = getattr(self, "pitch_mm", 3.0)

        if geom == "two_holes":
            auto_Ly = screen_r + 2.0 * screen_r + pitch
        elif geom == "one_hole":
            auto_Ly = 3.0 * screen_r
        else:  # half_hole
            auto_Ly = 0.5 * screen_r + 0.30 * pitch

        auto_Ly = round(auto_Ly, 3)
        return auto_Lx, auto_Ly

    def _on_neut_match_toggled(self, state):
        """Enable or disable manual neutralizer injection rate when matched to ion rate."""
        is_matched = bool(state)
        self.inputs["neut_rate"].setEnabled(not is_matched)

    def _validate_neut_x(self):
        """Warn and clamp neutralizer x position to Lx if it is settled outside the domain."""
        if hasattr(self, 'sim') and hasattr(self.sim, 'Lx') and self.sim.Lx > 0:
            val = self.inputs["neut_x"].value()
            if val > self.sim.Lx or val < 0.0:
                QMessageBox.warning(
                    self,
                    "Neutralizer Warning",
                    f"Neutralizer axial position (x = {val:.3f} mm) is settled outside the x-domain [0, {self.sim.Lx:.3f} mm].\n\n"
                    f"Setting position to x = Lx ({self.sim.Lx:.3f} mm)."
                )
                self.inputs["neut_x"].blockSignals(True)
                self.inputs["neut_x"].setValue(round(self.sim.Lx, 3))
                self.inputs["neut_x"].blockSignals(False)
                self._last_auto_neut_x = round(self.sim.Lx, 3)
                print(f"[Warning] Neutralizer position x={val:.3f} mm is outside domain [0, {self.sim.Lx:.3f} mm]. Settled to x = Lx ({self.sim.Lx:.3f} mm).")

    def _update_debye_gap(self):
        """Auto-update the Upstream Gap spinbox when n0 or Te_up change."""
        if not hasattr(self, '_last_auto_gap'):
            return
        current = round(self.inputs["upstream_gap_mm"].value(), 3)
        if abs(current - self._last_auto_gap) > 0.001:
            return
        bulk = self.adv_params.get("entire_bulk_plasma", False) if hasattr(self, 'adv_params') else False
        if bulk:
            n0    = self.inputs["n0_plasma"].value()
            Te_up = self.inputs["Te_up"].value()
            if n0 <= 0 or Te_up <= 0:
                return
            new_gap = round(compute_debye_upstream_gap(n0, Te_up), 3)
        else:
            if self.grid_widgets:
                screen_r = self.grid_widgets[0]["r"].value()
            else:
                screen_r = 0.80
            new_gap = round(0.75 * screen_r, 3)
        self._last_auto_gap = new_gap
        self.inputs["upstream_gap_mm"].blockSignals(True)
        self.inputs["upstream_gap_mm"].setValue(new_gap)
        self.inputs["upstream_gap_mm"].blockSignals(False)

    def apply_config(self, config, config_name=None):
        if config_name:
            self.current_config_name = config_name
        self.update_config_title()

        self.config = config
        discharge_chamber = config.get("discharge_chamber", {})
        self.pitch_mm = discharge_chamber["pitch_mm"]

        beam = config.get("beam_species", {})
        self.beam_mass_amu = beam["mass_amu"]
        self.beam_charge_state = beam["charge_state"]

        mat_cfg = config.get("grid_material", {})
        preset = mat_cfg.get("preset", "Custom")
        if preset in GridMaterialDialog.PRESETS and GridMaterialDialog.PRESETS[preset] is not None:
            self.mat_name = preset
            self.mat_props = GridMaterialDialog.PRESETS[preset].copy()
        else:
            self.mat_name = "Custom"
            self.mat_props = {
                "k": mat_cfg["k"],
                "rho": mat_cfg["rho"],
                "cp": mat_cfg["cp"],
                "emissivity": mat_cfg["emissivity"],
                "alpha": mat_cfg["alpha"],
                "E_mod": mat_cfg["E_mod"],
                "Y_coeff": mat_cfg["Y_coeff"],
                "E_th": mat_cfg["E_th"],
            }

        adv = config.get("advanced_settings", {})
        val_dx_debye = adv.get(r"\deltax/debye_length", adv.get("dx_over_debye", adv.get("deltax_debye_length", 0.8)))
        self.adv_params = {
            "V_plasma_offset": adv.get("V_plasma_offset", 20.0),
            "m_e_ratio": adv.get("m_e_ratio", 1000.0),
            "entire_bulk_plasma": bool(adv.get("entire_bulk_plasma", False)),
            "use_json_mesh_zones": bool(adv.get("use_json_mesh_zones", False)),
            "dx_over_debye": float(val_dx_debye),
            r"\deltax/debye_length": float(val_dx_debye),
        }

        self.mesh_zones = config.get("mesh_zones", {
            "presheath_factor": 1.0,
            "optics_factor": 1.0,
            "plume_factor": 4.0
        })

        sim = config.get("simulation", {})
        self.inputs["n0_plasma"].blockSignals(True)
        self.inputs["Te_up"].blockSignals(True)
        self.inputs["n0_plasma"].setValue(sim["n0_plasma"])
        self.inputs["Te_up"].setValue(sim["Te_up"])
        self.inputs["n0_plasma"].blockSignals(False)
        self.inputs["Te_up"].blockSignals(False)

        self.inputs["Ti"].setValue(sim["Ti"])
        self.inputs["Tn"].setValue(sim["Tn"])
        self.inputs["n0"].setValue(sim["n0"])
        self.inputs["Accel"].setValue(sim["Accel"])
        self.inputs["Thresh"].setValue(sim["Thresh"])
        self.inputs["target_ppc"].setValue(float(sim.get("target_ppc", 40.0)))

        mode = sim["sim_mode"]
        idx = self.combo_mode.findText(mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)

        geom = sim.get("geometry", "half_hole")
        idx_geom = self.combo_geometry.findText(geom)
        if idx_geom >= 0:
            self.combo_geometry.setCurrentIndex(idx_geom)

        grids_cfg = config.get("grids", [])
        if self.adv_params["entire_bulk_plasma"]:
            _gap_val = round(compute_debye_upstream_gap(sim["n0_plasma"], sim["Te_up"]), 3)
        else:
            _screen_r = grids_cfg[0]["r"] if grids_cfg else 0.80
            _gap_val = round(0.75 * _screen_r, 3)
        self._last_auto_gap = _gap_val
        self.inputs["upstream_gap_mm"].setValue(_gap_val)

        rf = config.get("rf_co_extraction", {})
        self.chk_rf.setChecked(rf["rf_enable"])
        self.spin_rf_freq.setValue(rf["rf_freq"])
        self.spin_rf_amp.setValue(rf["rf_amp"])

        neut = config.get("neutralizer", {})
        neut_matched = bool(neut.get("neut_match_ion", False))
        self.chk_neut_match_ion.setChecked(neut_matched)
        self.inputs["neut_rate"].setValue(neut.get("neut_rate", 30.0))
        self.inputs["neut_rate"].setEnabled(not neut_matched)
        self.inputs["Te"].setValue(neut.get("Te", 2.0))
        self.inputs["neut_r"].setValue(neut.get("neut_r", adv.get("neut_r", 1.5)))

        self.clear_grid_ui()
        grids = config.get("grids", [])
        if len(grids) == 0:
            self._apply_defaults()
            return

        for g in grids:
            self.add_grid_ui(g["V"], g["t"], g["gap"], g["r"], g["cham"])

        rf_idx = rf["rf_grid_idx"]
        if 0 <= rf_idx < self.combo_rf_grid.count():
            self.combo_rf_grid.setCurrentIndex(rf_idx)

        self.cs_store = load_cross_sections_from_config(config)
        self.pitch_mm = config.get("discharge_chamber", {}).get("pitch_mm", 3.0)

        auto_Lx, auto_Ly = self.compute_grid_domain_size()
        if "Lx" in adv:
            self.adv_params["Lx"] = adv["Lx"]
            self._user_overrode_Lx = True
        else:
            self.adv_params["Lx"] = auto_Lx
            self._user_overrode_Lx = False

        if "Ly" in adv:
            self.adv_params["Ly"] = adv["Ly"]
            self._user_overrode_Ly = True
        else:
            self.adv_params["Ly"] = auto_Ly
            self._user_overrode_Ly = False

        # Neutralizer axial distance (x):
        if "neut_x" in neut or "neut_x" in adv:
            val_neut_x = float(neut.get("neut_x", adv.get("neut_x")))
            if val_neut_x > auto_Lx or val_neut_x < 0.0:
                print(f"[Warning] Neutralizer position x={val_neut_x:.3f} mm in config is outside domain [0, {auto_Lx:.3f} mm]. Settled to x = Lx ({auto_Lx:.3f} mm).")
                val_neut_x = auto_Lx
            self.inputs["neut_x"].setValue(val_neut_x)
            self._last_auto_neut_x = val_neut_x
        else:
            if grids:
                upstream = self.inputs["upstream_gap_mm"].value()
                x_exit = upstream + sum(g["t"] + g["gap"] for g in grids[:-1]) + grids[-1]["t"]
                auto_neut_x = round(x_exit + 0.9 * (auto_Lx - x_exit), 3)
            else:
                auto_neut_x = round(auto_Lx - 0.5, 3)
            self.inputs["neut_x"].setValue(auto_neut_x)
            self._last_auto_neut_x = auto_neut_x

        cfg_name = getattr(self, "current_config_name", "config.json")
        self.lbl_status.setText(f"Loaded {cfg_name} | {len(grids)} grids")
        self.lbl_temp.setText("Grid Temps: " + " | ".join([f"G{i+1}: ready" for i in range(len(grids))]))
        if hasattr(self, 'lbl_material'):
            self.lbl_material.setText(f"Grid Material: {self.mat_name}")

    def reload_config(self):
        cfg_path = getattr(self, "current_config_path", None) or _config_path()
        cfg_name = getattr(self, "current_config_name", "config.json")
        cfg = load_json_config(cfg_path)
        if cfg is None:
            QMessageBox.warning(self, "No Config Found",
                                f"{cfg_name} not found. Using current defaults.")
            return
        try:
            self.apply_config(cfg, config_name=cfg_name)
            QMessageBox.information(self, "Config Reloaded",
                                    f"{cfg_name} was reloaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Config Error", f"Failed to reload config:\n{e}")

    def apply_advanced_settings_to_sim(self):
        self.sim.Lx = self.adv_params["Lx"]
        self.sim.Ly = self.adv_params["Ly"]
        self.sim.nx = int(self.sim.Lx / self.sim.dx) + 1
        self.sim.ny = int(self.sim.Ly / self.sim.dy) + 1

        self.sim.m_ion = self.beam_mass_amu * 1.6605e-27
        self.sim.m_XE = self.sim.m_ion
        self.sim.Z_ion = self.beam_charge_state
        self.sim.q_ion = self.beam_charge_state * self.sim.q
        self.sim.m_e = self.sim.m_ion / self.adv_params["m_e_ratio"]

        self.sim.set_material(props=self.mat_props)

        self.sim.user_cs = {}
        for label, ds in self.cs_store.items():
            if ds.get("spline") is not None:
                self.sim.user_cs[label] = ds

        self.sim.x_pts = np.linspace(0, self.sim.Lx, self.sim.nx)
        self.sim.y_pts = np.linspace(0, self.sim.Ly, self.sim.ny)
        self.sim.X, self.sim.Y = np.meshgrid(self.sim.x_pts, self.sim.y_pts)

    def get_params(self):
        params = {k: v.value() for k, v in self.inputs.items()}
        params.update(self.adv_params)
        params["sim_mode"] = self.combo_mode.currentText()
        params["geometry"] = self.combo_geometry.currentText()
        params["rf_enable"] = self.chk_rf.isChecked()
        params["rf_grid_idx"] = self.combo_rf_grid.currentIndex()
        params["rf_freq"] = self.spin_rf_freq.value()
        params["rf_amp"] = self.spin_rf_amp.value()
        params["neut_match_ion"] = self.chk_neut_match_ion.isChecked()
        params["pitch_mm"] = getattr(self, "pitch_mm", 0.0)

        grids = []
        for gw in self.grid_widgets:
            grids.append({
                "V": gw["V"].value(),
                "t": gw["t"].value(),
                "gap": gw["gap"].value(),
                "r": gw["r"].value(),
                "cham": gw["cham"].value(),
            })
        params["grids"] = grids

        inj_time_µs = self.inputs.get("inj_time_µs", None)
        if inj_time_µs is not None:
            params["inj_time"] = inj_time_µs.value() * 1e-6
        else:
            params["inj_time"] = 0.0

        params['macro_weight'] = self.sim.macro_weight
        params['use_json_mesh_zones'] = self.adv_params.get('use_json_mesh_zones', False)
        params['mesh_zones'] = getattr(self, 'mesh_zones', {
            "presheath_factor": 1.0,
            "optics_factor": 1.0,
            "plume_factor": 4.0
        })
        params['div_percentile'] = getattr(self, 'div_percentile', 95.0)
        params['div_method'] = getattr(self, 'div_method', '95%')
        return params
