"""Tests for aerodynamics.py — per-component local AoA damping.

Verifies that a pure pitch-rate perturbation produces pitch moment
matching the Mandell linearisation to within 1% for small q (§18.2).

Synthetic model: nosecone at CP=0.5 m (C_Nα≈1.5), fins at CP=2.0 m (C_Nα≈0.5).
CG = 1.0 m.  Mandell: τ = −½·V·A_ref·Σᵢ(C_Nα_i·aᵢ²)·q
"""

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from aerodynamics import aero_forces_moments, build_aero_model


# ---------------------------------------------------------------------------
# Synthetic data — same geometry as test_aero_interp.py
# ---------------------------------------------------------------------------

# Nosecone: C_Nα=1.5 rad⁻¹, CP=0.5 m
_NOSECONE_CSV = textwrap.dedent("""\
    Mach,Reynolds,AoA_deg,CA,CN,CP_m
    0.5,1000000,0.0,0.300,0.000000,0.500
    0.5,1000000,5.0,0.300,0.130900,0.500
    0.5,1000000,10.0,0.300,0.261799,0.500
    1.5,1000000,0.0,0.300,0.000000,0.500
    1.5,1000000,5.0,0.300,0.130900,0.500
    1.5,1000000,10.0,0.300,0.261799,0.500
""")

# Fins: C_Nα=0.5 rad⁻¹, CP=2.0 m
_FIN_CSV = textwrap.dedent("""\
    Mach,Reynolds,AoA_deg,CA,CN,CP_m
    0.5,1000000,0.0,0.200,0.000000,2.000
    0.5,1000000,5.0,0.200,0.043633,2.000
    0.5,1000000,10.0,0.200,0.087266,2.000
    1.5,1000000,0.0,0.200,0.000000,2.000
    1.5,1000000,5.0,0.200,0.043633,2.000
    1.5,1000000,10.0,0.200,0.087266,2.000
""")

CG = 1.0        # m from nosecone tip
V_FLIGHT = 100.0  # m/s airspeed
RHO = 1.225     # kg/m³
A_REF = 0.01    # m²  (arbitrary reference area)
MACH = 0.3      # supersonic-free test point within table range


def _model(tmp_path: Path):
    d = tmp_path / "aero"
    d.mkdir()
    (d / "nosecone.csv").write_text(_NOSECONE_CSV, encoding="utf-8")
    (d / "fins.csv").write_text(_FIN_CSV, encoding="utf-8")
    return build_aero_model(d)


def _mandell_tau_pitch(q_rate: float) -> float:
    """Mandell linearised pitch damping moment.

    τ = −½·ρ·V·A_ref · Σᵢ(C_Nα_i · aᵢ²) · q
    where aᵢ = CP_i − CG.
    """
    cna_nose, a_nose = 1.5, 0.5 - CG   # −0.5 m
    cna_fins, a_fins = 0.5, 2.0 - CG   # +1.0 m
    damp_sum = cna_nose * a_nose**2 + cna_fins * a_fins**2  # = 0.875
    return -0.5 * RHO * V_FLIGHT * A_REF * damp_sum * q_rate


# ---------------------------------------------------------------------------
# Damping matches Mandell linearisation at small q (§18.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_rate", [1e-4, 1e-3, 1e-2])
def test_pitch_damping_matches_mandell_small_q(tmp_path, q_rate):
    """Pure pitch rate (no bulk AoA): τ_pitch matches Mandell to within 1%."""
    m = _model(tmp_path)

    # u_rel = V, v_rel = w_rel = 0  → zero bulk AoA, pure rotation
    _, _, _, tau_pitch, _, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table, m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components,
        MACH, 1e6,
        RHO, V_FLIGHT, A_REF,
        V_FLIGHT, 0.0, 0.0,   # u_rel, v_rel, w_rel
        q_rate, 0.0,           # q_rate, r_rate
        CG,
    )

    expected = _mandell_tau_pitch(q_rate)
    assert tau_pitch == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("q_rate", [1e-4, 1e-3, 1e-2])
def test_yaw_damping_matches_mandell_small_r(tmp_path, q_rate):
    """Pure yaw rate: τ_yaw matches Mandell to within 1% (symmetry of model)."""
    m = _model(tmp_path)

    _, _, _, _, tau_yaw, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table, m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components,
        MACH, 1e6,
        RHO, V_FLIGHT, A_REF,
        V_FLIGHT, 0.0, 0.0,
        0.0, q_rate,           # q_rate=0, r_rate=q_rate
        CG,
    )

    expected = _mandell_tau_pitch(q_rate)  # symmetric model → same magnitude
    assert tau_yaw == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Damping sign convention
# ---------------------------------------------------------------------------

def test_pitch_damping_opposes_positive_q(tmp_path):
    """Positive pitch rate → negative (nose-down restoring) pitch moment."""
    m = _model(tmp_path)
    _, _, _, tau_pitch, _, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table, m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components,
        MACH, 1e6,
        RHO, V_FLIGHT, A_REF,
        V_FLIGHT, 0.0, 0.0,
        0.01, 0.0,
        CG,
    )
    assert tau_pitch < 0.0


def test_yaw_damping_opposes_positive_r(tmp_path):
    """Positive yaw rate → negative yaw moment."""
    m = _model(tmp_path)
    _, _, _, _, tau_yaw, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table, m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components,
        MACH, 1e6,
        RHO, V_FLIGHT, A_REF,
        V_FLIGHT, 0.0, 0.0,
        0.0, 0.01,
        CG,
    )
    assert tau_yaw < 0.0


# ---------------------------------------------------------------------------
# Restoring moment still present without rotation
# ---------------------------------------------------------------------------

def test_restoring_moment_no_rotation(tmp_path):
    """Static AoA with no rotation gives a non-zero pitch moment."""
    m = _model(tmp_path)
    alpha = math.radians(5.0)
    u_rel = V_FLIGHT * math.cos(alpha)
    w_rel = V_FLIGHT * math.sin(alpha)

    _, _, _, tau_pitch, _, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table, m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components,
        MACH, 1e6,
        RHO, V_FLIGHT, A_REF,
        u_rel, 0.0, w_rel,
        0.0, 0.0,
        CG,
    )
    # At α=5°, CP=0.875 m is forward of CG=1.0 m (unstable geometry):
    # positive w_rel → nose-up → destabilising τ_pitch > 0
    assert tau_pitch > 0.0


# ---------------------------------------------------------------------------
# cp_whole from per-component loop
# ---------------------------------------------------------------------------

def test_cp_whole_at_nonzero_aoa(tmp_path):
    """cp_whole = Σ(CN_i·CP_i)/Σ(CN_i) at static AoA ≈ 0.875 m."""
    m = _model(tmp_path)
    alpha = math.radians(5.0)
    u_rel = V_FLIGHT * math.cos(alpha)
    w_rel = V_FLIGHT * math.sin(alpha)

    _, _, _, _, _, cp_whole = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table, m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components,
        MACH, 1e6,
        RHO, V_FLIGHT, A_REF,
        u_rel, 0.0, w_rel,
        0.0, 0.0,
        CG,
    )
    assert cp_whole == pytest.approx(0.875, rel=1e-3)


def test_cp_whole_fallback_to_cg_at_zero_aoa(tmp_path):
    """At zero AoA (no rotation), cp_whole falls back to CG."""
    m = _model(tmp_path)
    _, _, _, _, _, cp_whole = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table, m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components,
        MACH, 1e6,
        RHO, V_FLIGHT, A_REF,
        V_FLIGHT, 0.0, 0.0,  # pure axial, zero lateral
        0.0, 0.0,
        CG,
    )
    assert cp_whole == pytest.approx(CG, abs=1e-6)
