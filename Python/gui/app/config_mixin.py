"""
Configuration synchronization, Debye length calculations, and domain sizing mixin.
"""
import os
import copy
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from ..config_io import (
    _config_path,
    load_json_config,
    save_json_config,
    load_cross_sections_from_config
)
from ..bootstrap import _safe_start_dir
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
            "dt_ns": 0.50,
            "dt": 5e-10,
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
        val_dt_ns = adv.get("dt_ns", adv.get("dt", 5e-10) * 1e9 if "dt" in adv else 0.50)
        self.adv_params = {
            "V_plasma_offset": adv.get("V_plasma_offset", 20.0),
            "m_e_ratio": adv.get("m_e_ratio", 1000.0),
            "entire_bulk_plasma": bool(adv.get("entire_bulk_plasma", False)),
            "use_json_mesh_zones": bool(adv.get("use_json_mesh_zones", False)),
            "dx_over_debye": float(val_dx_debye),
            r"\deltax/debye_length": float(val_dx_debye),
            "dt_ns": float(val_dt_ns),
            "dt": float(val_dt_ns) * 1e-9,
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

        if "div_plane_x" in sim and sim["div_plane_x"] is not None:
            self.div_plane_x = float(sim["div_plane_x"])
            if hasattr(self, 'sim'):
                self.sim.div_plane_x = self.div_plane_x
        elif "div_plane_x" in adv and adv["div_plane_x"] is not None:
            self.div_plane_x = float(adv["div_plane_x"])
            if hasattr(self, 'sim'):
                self.sim.div_plane_x = self.div_plane_x
        else:
            self.div_plane_x = None
            if hasattr(self, 'sim'):
                self.sim.div_plane_x = None

        div_method_val = sim.get("div_method", adv.get("div_method", None))
        if div_method_val and hasattr(self, 'set_divergence_method'):
            self.set_divergence_method(str(div_method_val))

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

    def get_current_config_dict(self):
        """
        Assemble the full JSON configuration dictionary reflecting the current
        application state and GUI inputs.
        """
        if hasattr(self, 'config') and isinstance(self.config, dict):
            cfg = copy.deepcopy(self.config)
        else:
            cfg = {}

        # 1. Beam species
        cfg["beam_species"] = {
            "mass_amu": float(getattr(self, "beam_mass_amu", 131.293)),
            "charge_state": int(getattr(self, "beam_charge_state", 1))
        }

        # 2. Grid material
        mat_name = getattr(self, "mat_name", "Molybdenum")
        mat_props = getattr(self, "mat_props", {})
        if mat_name in GridMaterialDialog.PRESETS and mat_name != "Custom":
            cfg["grid_material"] = {
                "preset": mat_name
            }
        else:
            cfg["grid_material"] = {
                "preset": "Custom",
                "k": float(mat_props.get("k", 138.0)),
                "rho": float(mat_props.get("rho", 10280.0)),
                "cp": float(mat_props.get("cp", 250.0)),
                "emissivity": float(mat_props.get("emissivity", 0.80)),
                "alpha": float(mat_props.get("alpha", 4.8e-6)),
                "E_mod": float(mat_props.get("E_mod", 329e9)),
                "Y_coeff": float(mat_props.get("Y_coeff", 1.05e-4)),
                "E_th": float(mat_props.get("E_th", 30.0))
            }

        # 3. Advanced settings
        adv_cfg = cfg.get("advanced_settings", {})
        adv_params = getattr(self, "adv_params", {})
        adv_cfg["V_plasma_offset"] = float(adv_params.get("V_plasma_offset", 20.0))
        adv_cfg["m_e_ratio"] = float(adv_params.get("m_e_ratio", 1000.0))

        if "entire_bulk_plasma" in adv_params:
            adv_cfg["entire_bulk_plasma"] = bool(adv_params["entire_bulk_plasma"])
        if "use_json_mesh_zones" in adv_params:
            adv_cfg["use_json_mesh_zones"] = bool(adv_params["use_json_mesh_zones"])
        if "dx_over_debye" in adv_params:
            adv_cfg["dx_over_debye"] = float(adv_params["dx_over_debye"])
        if "dt_ns" in adv_params:
            adv_cfg["dt_ns"] = float(adv_params["dt_ns"])

        if getattr(self, "_user_overrode_Lx", False) or "Lx" in adv_cfg:
            adv_cfg["Lx"] = float(adv_params.get("Lx", 20.0))
        if getattr(self, "_user_overrode_Ly", False) or "Ly" in adv_cfg:
            adv_cfg["Ly"] = float(adv_params.get("Ly", 3.0))

        cfg["advanced_settings"] = adv_cfg

        # 4. Mesh zones
        cfg["mesh_zones"] = getattr(self, "mesh_zones", {
            "presheath_factor": 1.0,
            "optics_factor": 2.0,
            "plume_factor": 4.0
        })

        # 5. Simulation
        sim_cfg = cfg.get("simulation", {})
        inputs = getattr(self, "inputs", {})
        sim_cfg["n0_plasma"] = float(inputs["n0_plasma"].value()) if "n0_plasma" in inputs else float(sim_cfg.get("n0_plasma", 1e17))
        sim_cfg["Te_up"] = float(inputs["Te_up"].value()) if "Te_up" in inputs else float(sim_cfg.get("Te_up", 3.0))
        sim_cfg["Ti"] = float(inputs["Ti"].value()) if "Ti" in inputs else float(sim_cfg.get("Ti", 0.5))
        sim_cfg["Tn"] = float(inputs["Tn"].value()) if "Tn" in inputs else float(sim_cfg.get("Tn", 300.0))
        sim_cfg["n0"] = float(inputs["n0"].value()) if "n0" in inputs else float(sim_cfg.get("n0", 1e18))
        sim_cfg["Accel"] = float(inputs["Accel"].value()) if "Accel" in inputs else float(sim_cfg.get("Accel", 1.0))
        sim_cfg["Thresh"] = float(inputs["Thresh"].value()) if "Thresh" in inputs else float(sim_cfg.get("Thresh", 1e6))
        sim_cfg["sim_mode"] = str(self.combo_mode.currentText()) if hasattr(self, "combo_mode") else str(sim_cfg.get("sim_mode", "Thermal"))
        sim_cfg["geometry"] = str(self.combo_geometry.currentText()) if hasattr(self, "combo_geometry") else str(sim_cfg.get("geometry", "half_hole"))
        if "target_ppc" in inputs:
            sim_cfg["target_ppc"] = float(inputs["target_ppc"].value())
        if "inj_time_µs" in inputs and inputs["inj_time_µs"].value() > 0:
            sim_cfg["inj_time_µs"] = float(inputs["inj_time_µs"].value())
        if getattr(self, "div_plane_x", None) is not None:
            sim_cfg["div_plane_x"] = float(self.div_plane_x)
        if getattr(self, "div_method", None) is not None:
            sim_cfg["div_method"] = str(self.div_method)

        cfg["simulation"] = sim_cfg

        # 6. RF co-extraction
        cfg["rf_co_extraction"] = {
            "rf_enable": bool(self.chk_rf.isChecked()) if hasattr(self, "chk_rf") else False,
            "rf_grid_idx": int(self.combo_rf_grid.currentIndex()) if hasattr(self, "combo_rf_grid") else 0,
            "rf_freq": float(self.spin_rf_freq.value()) if hasattr(self, "spin_rf_freq") else 13.56,
            "rf_amp": float(self.spin_rf_amp.value()) if hasattr(self, "spin_rf_amp") else 0.0
        }

        # 7. Neutralizer
        neut_rate_val = inputs["neut_rate"].value() if "neut_rate" in inputs else 30.0
        neut_rate_num = int(neut_rate_val) if neut_rate_val == int(neut_rate_val) else float(neut_rate_val)
        cfg["neutralizer"] = {
            "neut_rate": neut_rate_num,
            "neut_match_ion": bool(self.chk_neut_match_ion.isChecked()) if hasattr(self, "chk_neut_match_ion") else False,
            "Te": float(inputs["Te"].value()) if "Te" in inputs else 2.0,
            "neut_x": float(inputs["neut_x"].value()) if "neut_x" in inputs else 10.0,
            "neut_r": float(inputs["neut_r"].value()) if "neut_r" in inputs else 1.5
        }

        # 8. Grids
        grids_data = []
        if hasattr(self, "grid_widgets"):
            for gw in self.grid_widgets:
                grids_data.append({
                    "V": float(gw["V"].value()),
                    "t": float(gw["t"].value()),
                    "gap": float(gw["gap"].value()),
                    "r": float(gw["r"].value()),
                    "cham": float(gw["cham"].value()),
                })
        cfg["grids"] = grids_data

        # 9. Additional sections preserved or defaulted
        if "cross_sections" not in cfg:
            cfg["cross_sections"] = {
                "cx_file": "",
                "see_file": "",
                "custom_file": "",
                "spline_smoothing": 0.0
            }

        if "terminal_output" not in cfg:
            cfg["terminal_output"] = {
                "grid_temperatures": True,
                "beam_divergence": True,
                "saddle_point_potential": True,
                "mean_particle_energy": True,
                "iteration_time": True
            }

        if "discharge_chamber" not in cfg:
            cfg["discharge_chamber"] = {
                "radius_m": 0.02,
                "length_m": 0.10,
                "n_cusp": 4,
                "pitch_mm": float(getattr(self, "pitch_mm", 3.0))
            }
        else:
            cfg["discharge_chamber"]["pitch_mm"] = float(getattr(self, "pitch_mm", cfg["discharge_chamber"].get("pitch_mm", 3.0)))

        if "grid_transparency" not in cfg:
            cfg["grid_transparency"] = {
                "beta_i": 0.7,
                "beta_g": 0.5
            }

        if "power_and_flow" not in cfg:
            cfg["power_and_flow"] = {
                "B": 0.05,
                "massflow_in": 2.18e-7,
                "Pw": 45.0,
                "Te_0": 3.0
            }

        return cfg

    def save_config(self):
        """Save the current configuration to the active .json file after confirmation."""
        reply = QMessageBox.question(
            self,
            "Save Configuration",
            "Are you sure you want to save this configuration?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        target_path = getattr(self, "current_config_path", None)
        if not target_path:
            target_path = _config_path()

        try:
            cfg_dict = self.get_current_config_dict()
            save_json_config(cfg_dict, target_path)
            self.config = cfg_dict
            self.current_config_path = target_path
            self.current_config_name = os.path.basename(target_path)
            self.update_config_title()
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(f"Configuration saved to {self.current_config_name}")
            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Configuration successfully saved to:\n{target_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration:\n{e}")

    def save_as_config(self):
        """Save the current configuration to a user-selected .json file."""
        start_dir = (
            os.path.dirname(self.current_config_path)
            if getattr(self, "current_config_path", None) and os.path.dirname(self.current_config_path)
            else _safe_start_dir()
        )
        suggested_name = getattr(self, "current_config_name", "config.json")
        default_save_path = os.path.join(start_dir, suggested_name)

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration As",
            default_save_path,
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_name:
            return

        if not file_name.lower().endswith(".json"):
            file_name += ".json"

        try:
            cfg_dict = self.get_current_config_dict()
            save_json_config(cfg_dict, file_name)
            self.config = cfg_dict
            self.current_config_path = file_name
            self.current_config_name = os.path.basename(file_name)
            self.update_config_title()
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(f"Configuration saved to {self.current_config_name}")
            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Configuration successfully saved to:\n{file_name}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration:\n{e}")

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
        if "dt" in self.adv_params and self.adv_params["dt"] > 0:
            self.sim.dt = float(self.adv_params["dt"])

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
        params['div_plane_x'] = getattr(self, 'div_plane_x', None)
        params['twiss_n_sigma'] = getattr(self, 'twiss_n_sigma', 1.0)
        return params
