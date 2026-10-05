"""
Materials and cross-sections mixin for DigitalTwinSimulator.
"""
import numpy as np


class MaterialsMixin:
    """Material assignment and reaction cross-section lookups."""

    def set_material(self, name=None, props=None):
        """
        Set grid material by preset name or custom property dict.
        props keys: k, rho, cp, emissivity, alpha, Y_coeff, E_th
        """
        if name and name in self.MATERIAL_PRESETS:
            mat = self.MATERIAL_PRESETS[name]
        elif props:
            mat = props
        else:
            return
        self.mat_k         = mat['k']
        self.mat_rho       = mat['rho']
        self.mat_cp        = mat['cp']
        self.emissivity    = mat['emissivity']
        self.alpha_thermal = mat['alpha']
        self.E_modulus     = mat['E_mod']
        self.sputter_Y_coeff = mat['Y_coeff']
        self.sputter_E_th    = mat['E_th']
        self._recompute_cell_constants()

    def lookup_user_cs(self, cs_type_prefix, energy_eV):
        for label, ds in self.user_cs.items():
            if label.startswith(cs_type_prefix) and ds.get('spline') is not None:
                log_e = np.log10(np.maximum(energy_eV, 1e-30))
                e_min = np.log10(max(ds['energy'][0], 1e-30))
                e_max = np.log10(ds['energy'][-1])
                log_e = np.clip(log_e, e_min, e_max)
                log_cs = ds['spline'](log_e)
                return 10.0 ** log_cs
        return None
