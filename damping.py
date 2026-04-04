"""Damping post-processing — computes pitch/yaw damping assessment quantities.

Separated from dynamics.py for clarity.  The hot-loop Mandell damping term
lives in aerodynamics.py (part of the @njit force model); this module handles
only the post-flight diagnostic pass.
"""
from __future__ import annotations

import math

import numpy as np

from aerodynamics import cn_alpha_comp_at, _interp3
from atmosphere import isa_at_site

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dynamics import TrajectoryProfile, SimParams


_EPS_V = 1.0e-6  # minimum velocity for meaningful aero quantities


def compute_damping(profile: TrajectoryProfile, params: SimParams) -> None:
    """Compute damping quantities as a post-processing pass over the ascent.

    Mutates *profile* in place, filling all damping-related fields.
    Quantities are only meaningful during the ascent phase (up to apogee);
    descent and rail values are NaN.

    Definitions (ref: "Advanced Topics in Model Rocketry" pp. 201-202):
      C1   — corrective moment coefficient:  0.5 rho V^2 S_ref C_Nalpha (X_CP - X_CG)
      C2   — damping moment coefficient:     C2A + C2R
      C2A  — aerodynamic damping:            Sigma_j 0.5 rho V S_ref C_Nalpha_j (X_CPj - X_CG)^2
      C2R  — jet damping (eq. 99):           mdot (L_ne - X_CG)^2
    where L_ne is the nozzle exit position (distance from nose tip).
    """
    p = params
    K = len(profile.time)
    n_comp = p.cn_alpha_comp.shape[0] if p.has_components else 0

    # Find apogee index
    apogee_idx = int(np.argmax(profile.altitude))

    # Allocate output arrays (NaN everywhere)
    c1 = np.full(K, math.nan, dtype=np.float64)
    c2 = np.full(K, math.nan, dtype=np.float64)
    c2a = np.full(K, math.nan, dtype=np.float64)
    c2r = np.full(K, math.nan, dtype=np.float64)
    zeta = np.full(K, math.nan, dtype=np.float64)
    omega_n = np.full(K, math.nan, dtype=np.float64)
    omega_d = np.full(K, math.nan, dtype=np.float64)
    max_roll_hz = np.full(K, math.nan, dtype=np.float64)

    cn_alpha_comp_buf = np.full((n_comp, K), math.nan, dtype=np.float64)
    cp_comp_buf = np.full((n_comp, K), math.nan, dtype=np.float64)
    c1_comp_buf = np.full((n_comp, K), math.nan, dtype=np.float64)
    c2a_comp_buf = np.full((n_comp, K), math.nan, dtype=np.float64)

    if not p.has_components or apogee_idx == 0:
        # Cannot compute damping without per-component data
        profile.c1 = c1
        profile.c2 = c2
        profile.c2a = c2a
        profile.c2r = c2r
        profile.zeta = zeta
        profile.omega_n = omega_n
        profile.omega_d = omega_d
        profile.max_roll_rate_hz = max_roll_hz
        profile.cn_alpha_comp = cn_alpha_comp_buf
        profile.cp_comp = cp_comp_buf
        profile.c1_comp = c1_comp_buf
        profile.c2a_comp = c2a_comp_buf
        profile.comp_names = p.comp_names
        return

    # Roll rate characteristic radius
    r_roll = (p.diameter + p.fin_span) / 2.0

    for i in range(apogee_idx + 1):
        h = max(float(profile.altitude[i]), 0.0)
        M = float(profile.mach[i])

        _, _, rho, a, mu = isa_at_site(h, p.site_elevation, p.t_offset)
        V = M * a  # reconstruct velocity from Mach
        if V < _EPS_V:
            continue

        Re = rho * V * p.length / mu if mu > 0.0 else 0.0

        cg      = float(profile.cg[i])
        I_lat   = float(profile.I_lateral[i])
        I_roll  = float(profile.I_roll[i])
        m_dot   = float(profile.mdot[i])

        if math.isnan(cg) or math.isnan(I_lat) or math.isnan(m_dot):
            continue

        # Whole-vehicle CN_alpha (sum of per-component)
        cna_total = 0.0
        for j in range(n_comp):
            cna_j = cn_alpha_comp_at(p.mach_g, p.re_g, p.cn_alpha_comp, M, Re, j)
            cn_alpha_comp_buf[j, i] = cna_j
            cna_total += cna_j

        # Whole-vehicle CP (from per-component CN_alpha-weighted average)
        cp_total = 0.0
        if cna_total > 1e-9:
            for j in range(n_comp):
                cp_j = _interp3(p.mach_g, p.re_g, p.alpha_g, p.cp_comp[j], M, Re, 2.0)
                cp_comp_buf[j, i] = cp_j
                cp_total += cn_alpha_comp_buf[j, i] * cp_j
            cp_total /= cna_total
        else:
            cp_total = cg
            for j in range(n_comp):
                cp_j = _interp3(p.mach_g, p.re_g, p.alpha_g, p.cp_comp[j], M, Re, 2.0)
                cp_comp_buf[j, i] = cp_j

        # C1 — corrective moment coefficient (p. 201)
        c1_val = 0.5 * rho * V * V * p.A_ref * cna_total * (cp_total - cg)
        c1[i] = c1_val

        # C1 and C2A per-component contributions
        c2a_val = 0.0
        for j in range(n_comp):
            c_cp = cp_comp_buf[j, i]
            c_cna = cn_alpha_comp_buf[j, i]
            if math.isnan(c_cp):
                c_cp = cg
            lever = c_cp - cg
            # C1_j = 0.5 rho V^2 S_ref CNalpha_j (CP_j - CG)
            c1_comp_buf[j, i] = 0.5 * rho * V * V * p.A_ref * c_cna * lever
            # C2A_j = 0.5 rho V S_ref CNalpha_j (CP_j - CG)^2
            contrib = 0.5 * rho * V * p.A_ref * c_cna * lever * lever
            c2a_comp_buf[j, i] = contrib
            c2a_val += contrib
        c2a[i] = c2a_val

        # C2R — jet damping moment coefficient (eq. 99): mdot (L_ne - X_CG)^2
        lever = p.nozzle_position - cg
        c2r_val = m_dot * lever * lever
        c2r[i] = c2r_val

        # C2 — total damping moment coefficient
        c2_val = c2a_val + c2r_val
        c2[i] = c2_val

        # Damping ratio — coupled (rolling) form: I_L replaced by (I_L + I_R)
        I_total = I_lat + I_roll
        if c1_val > 0.0 and I_total > 0.0:
            product = c1_val * I_total
            zeta_val = c2_val / (2.0 * math.sqrt(product))
            zeta[i] = zeta_val

            omega_n_val = math.sqrt(c1_val / I_total)
            omega_n[i] = omega_n_val

            discriminant = 1.0 - zeta_val * zeta_val
            if discriminant > 0.0:
                omega_d[i] = omega_n_val * math.sqrt(discriminant)
            else:
                omega_d[i] = 0.0

        # Max permissible roll rate
        if r_roll > 0.0:
            max_roll_hz[i] = V / r_roll / (2.0 * math.pi)

    # Store results on profile
    profile.c1 = c1
    profile.c2 = c2
    profile.c2a = c2a
    profile.c2r = c2r
    profile.zeta = zeta
    profile.omega_n = omega_n
    profile.omega_d = omega_d
    profile.max_roll_rate_hz = max_roll_hz
    profile.cn_alpha_comp = cn_alpha_comp_buf
    profile.cp_comp = cp_comp_buf
    profile.c1_comp = c1_comp_buf
    profile.c2a_comp = c2a_comp_buf
    profile.comp_names = p.comp_names
