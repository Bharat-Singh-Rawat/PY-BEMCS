"""
Physical conservation metrics calculation routines for PY-BEMCS.

Computes instantaneous system energy and charge budgets from the PIC simulator state.
"""

import numpy as np


def compute_energy_budget(sim):
    """
    Compute and store the instantaneous total energy of the PIC system:

        E_total = E_kinetic_ions + E_kinetic_electrons + E_field

    where:
        E_kinetic_ions      = sum_p  0.5 * m_ion * (vx^2 + vy^2 + vz^2) * macro_weight
        E_kinetic_electrons = sum_e  0.5 * m_e   * (vx^2 + vy^2 + vz^2) * macro_weight
        E_field             = (eps0/2) * integral(|E|^2) dV

    Appends results to sim.energy_history_* and prints a console warning if
    energy changes abruptly (> 5% by default) in a single step.

    Returns
    -------
    dict with keys: t_s, ke_ions_J, ke_elec_J, field_J, total_J
    """
    t_now = sim.iteration * sim.dt

    # --- Ion kinetic energy ---
    if sim.num_p > 0:
        v2 = (sim.p_vx[:sim.num_p]**2
              + sim.p_vy[:sim.num_p]**2
              + sim.p_vz[:sim.num_p]**2)
        ke_i = float(np.sum(v2)) * 0.5 * sim.m_ion * sim.macro_weight
    else:
        ke_i = 0.0

    # --- Electron kinetic energy ---
    if sim.num_e > 0:
        v2e = (sim.e_vx[:sim.num_e]**2
               + sim.e_vy[:sim.num_e]**2
               + sim.e_vz[:sim.num_e]**2)
        ke_e = float(np.sum(v2e)) * 0.5 * sim.m_e * sim.macro_weight
    else:
        ke_e = 0.0

    # --- Electrostatic field energy: (eps0/2) * integral(|E|^2) dV ---
    cell_area = (sim.dx * 1e-3) * (sim.dy * 1e-3)  # [m^2]
    e_field = float(np.sum(sim.Ex**2 + sim.Ey**2)) * 0.5 * sim.eps0 * cell_area

    total = ke_i + ke_e + e_field

    # --- Step-to-step conservation warning ---
    if sim._prev_total_energy is not None and sim._prev_total_energy > 0.0:
        rel_change = abs(total - sim._prev_total_energy) / sim._prev_total_energy
        if rel_change > sim.energy_warning_threshold:
            print(
                f"[Energy Warning] iter={sim.iteration}: "
                f"total energy changed by {rel_change*100:.1f}% in one step "
                f"(prev={sim._prev_total_energy:.4e} J, now={total:.4e} J). "
                f"KE_i={ke_i:.3e} J, KE_e={ke_e:.3e} J, E_field={e_field:.3e} J"
            )
    sim._prev_total_energy = total

    # Append to history
    sim.energy_history_t.append(t_now)
    sim.energy_history_ke_i.append(ke_i)
    sim.energy_history_ke_e.append(ke_e)
    sim.energy_history_fe.append(e_field)
    sim.energy_history_tot.append(total)

    return dict(t_s=t_now, ke_ions_J=ke_i, ke_elec_J=ke_e, field_J=e_field, total_J=total)


def compute_charge_budget(sim):
    """
    Compute and store the instantaneous charge budget of the PIC system:

        Q_ions        = sum_p  q_ion * macro_weight
        Q_electrons   = - sum_e e * macro_weight
        Q_boltzmann   = integral (rho_e_continuum) dV
        Q_free        = Q_ions + Q_electrons   (discrete macroparticles)
        Q_net         = Q_free + Q_boltzmann   (total space charge)

    Returns
    -------
    dict with keys: t_s, q_ions_C, q_elec_C, q_boltz_C, q_free_C, q_net_C
    """
    t_now = sim.iteration * sim.dt

    # --- Ion charge ---
    if sim.num_p > 0:
        q_i = float(sim.num_p) * sim.q_ion * sim.macro_weight
    else:
        q_i = 0.0

    # --- Kinetic Electron charge ---
    if sim.num_e > 0:
        q_e = - float(sim.num_e) * sim.q * sim.macro_weight
    else:
        q_e = 0.0

    # --- Boltzmann fluid electron charge ---
    if hasattr(sim, 'V') and sim.V is not None:
        cell_vol = (sim.dx * 1e-3) * (sim.dy * 1e-3) * 1e-3
        grids = getattr(sim, 'grids', [{'V': 1000}])
        v_offset = getattr(sim, 'V_plasma_offset', 20.0)
        V_p = grids[0]['V'] + v_offset if grids else 1020.0
        Te = getattr(sim, 'Te_up', 3.0)
        n0 = getattr(sim, 'n0_plasma', 1e17)

        interior = ~sim.isBound if hasattr(sim, 'isBound') else slice(None)
        bf = np.exp((np.minimum(sim.V[interior], V_p) - V_p) / Te)
        q_b = float(np.sum(-sim.q * n0 * bf)) * cell_vol
    else:
        q_b = 0.0

    q_free = q_i + q_e
    q_net  = q_free + q_b

    # Append to history
    sim.charge_history_t.append(t_now)
    sim.charge_history_q_i.append(q_i)
    sim.charge_history_q_e.append(q_e)
    sim.charge_history_q_b.append(q_b)
    sim.charge_history_q_free.append(q_free)
    sim.charge_history_q_net.append(q_net)

    sim._prev_total_charge = q_net

    return dict(
        t_s=t_now,
        q_ions_C=q_i,
        q_elec_C=q_e,
        q_boltz_C=q_b,
        q_free_C=q_free,
        q_net_C=q_net,
    )
