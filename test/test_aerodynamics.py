"""Tests for aerodynamics.py — table loading, coefficient interpolation, and
per-component local AoA damping.

Damping tests verify that a pure pitch-rate perturbation produces pitch moment
matching the Mandell linearisation to within 1% for small q (§18.2).

Synthetic model: nosecone at CP=0.5 m (C_Nα≈1.5), fins at CP=2.0 m (C_Nα≈0.5).
CG = 1.0 m.  Mandell: τ = −½·V·A_ref·Σᵢ(C_Nα_i·aᵢ²)·q
"""

import math
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from aerodynamics import (
    AeroModel,
    build_aero_model,
    aero_forces_moments,
    ca_at,
    cn_cp_at,
    cn_alpha_comp_at,
    _interp2,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic CSV data
# ---------------------------------------------------------------------------

# Single whole-vehicle file.
# CA = 0.5 (constant), CN = 2.0·α_rad (C_Nα = 2.0), CP = 1.5 m (constant).
# Mach = [0.5, 1.5], Re = [1e6], AoA = [0, 5, 10] deg.
# CN at 5° = 2.0·sin(5°) ≈ 2.0·0.087266 = 0.174533
# CN at 10° = 2.0·sin(10°) ≈ 2.0·0.174533 = 0.349066
_WHOLE_CSV = textwrap.dedent("""\
    Mach,Reynolds,AoA_deg,CA,CN,CP_m
    0.5,1000000,0.0,0.500,0.000000,1.500
    0.5,1000000,5.0,0.500,0.174533,1.500
    0.5,1000000,10.0,0.500,0.349066,1.500
    1.5,1000000,0.0,0.500,0.000000,1.500
    1.5,1000000,5.0,0.500,0.174533,1.500
    1.5,1000000,10.0,0.500,0.349066,1.500
""")

# Per-component nosecone file.
# CA=0.3, CN=1.5·α_rad, CP=0.5 m, CN_alpha=1.5/rad
# CN at 5° = 1.5·0.087266 = 0.130900; at 10° = 0.261799
_NOSECONE_CSV = textwrap.dedent("""\
    Mach,Reynolds,AoA_deg,CA_off,CA_on,CN,CP_m,CN_alpha_per_rad
    0.5,1000000,0.0,0.300,0.300,0.000000,0.500,1.5
    0.5,1000000,5.0,0.300,0.300,0.130900,0.500,1.5
    0.5,1000000,10.0,0.300,0.300,0.261799,0.500,1.5
    1.5,1000000,0.0,0.300,0.300,0.000000,0.500,1.5
    1.5,1000000,5.0,0.300,0.300,0.130900,0.500,1.5
    1.5,1000000,10.0,0.300,0.300,0.261799,0.500,1.5
""")

# Per-component fin file (stem contains 'fin').
# CA=0.2, CN=0.5·α_rad, CP=2.0 m, CN_alpha=0.5/rad
# CN at 5° = 0.5·0.087266 = 0.043633; at 10° = 0.087266
_FIN_CSV = textwrap.dedent("""\
    Mach,Reynolds,AoA_deg,CA_off,CA_on,CN,CP_m,CN_alpha_per_rad
    0.5,1000000,0.0,0.200,0.200,0.000000,2.000,0.5
    0.5,1000000,5.0,0.200,0.200,0.043633,2.000,0.5
    0.5,1000000,10.0,0.200,0.200,0.087266,2.000,0.5
    1.5,1000000,0.0,0.200,0.200,0.000000,2.000,0.5
    1.5,1000000,5.0,0.200,0.200,0.043633,2.000,0.5
    1.5,1000000,10.0,0.200,0.200,0.087266,2.000,0.5
""")


def _write(tmp_path: Path, name: str, content: str) -> None:
    (tmp_path / name).write_text(content, encoding="utf-8")


def _single_dir(tmp_path: Path) -> Path:
    d = tmp_path / "aero"
    d.mkdir()
    _write(d, "vehicle.csv", _WHOLE_CSV)
    return d


def _component_dir(tmp_path: Path) -> Path:
    d = tmp_path / "aero"
    d.mkdir()
    _write(d, "nosecone.csv", _NOSECONE_CSV)
    _write(d, "fins.csv", _FIN_CSV)
    return d


# Damping test constants
_DAMP_CG = 1.0         # m from nosecone tip
_DAMP_V = 100.0         # m/s airspeed
_DAMP_RHO = 1.225       # kg/m³
_DAMP_A_REF = 0.01      # m²  (arbitrary reference area)
_DAMP_MACH = 0.3        # supersonic-free test point within table range


def _damp_model(tmp_path: Path):
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
    cna_nose, a_nose = 1.5, 0.5 - _DAMP_CG   # −0.5 m
    cna_fins, a_fins = 0.5, 2.0 - _DAMP_CG   # +1.0 m
    damp_sum = cna_nose * a_nose**2 + cna_fins * a_fins**2  # = 0.875
    return -0.5 * _DAMP_RHO * _DAMP_V * _DAMP_A_REF * damp_sum * q_rate


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------

def test_build_single_file_returns_aeromodel(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_aero_model(_single_dir(tmp_path))
    assert isinstance(model, AeroModel)


def test_single_file_has_components_false(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_aero_model(_single_dir(tmp_path))
    assert model.has_components is False


def test_single_file_emits_warning(tmp_path):
    with pytest.warns(UserWarning, match="whole-vehicle"):
        build_aero_model(_single_dir(tmp_path))


def test_multi_component_has_components_true(tmp_path):
    model = build_aero_model(_component_dir(tmp_path))
    assert model.has_components is True


def test_no_fin_file_emits_warning(tmp_path):
    d = tmp_path / "aero"
    d.mkdir()
    _write(d, "nosecone.csv", _NOSECONE_CSV)
    _write(d, "body.csv", _NOSECONE_CSV)  # no 'fin' in name
    with pytest.warns(UserWarning, match="fin"):
        build_aero_model(d)


def test_empty_dir_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        build_aero_model(d)


def test_single_file_path_direct_returns_aeromodel(tmp_path):
    """Passing a .csv file directly (not a directory) should work."""
    f = tmp_path / "vehicle.csv"
    f.write_text(_WHOLE_CSV, encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_aero_model(f)
    assert isinstance(model, AeroModel)


def test_single_file_path_direct_has_components_false(tmp_path):
    f = tmp_path / "vehicle.csv"
    f.write_text(_WHOLE_CSV, encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_aero_model(f)
    assert model.has_components is False


def test_single_file_path_direct_emits_warning(tmp_path):
    f = tmp_path / "vehicle.csv"
    f.write_text(_WHOLE_CSV, encoding="utf-8")
    with pytest.warns(UserWarning, match="whole-vehicle"):
        build_aero_model(f)


def test_grid_shapes_consistent(tmp_path):
    model = build_aero_model(_component_dir(tmp_path))
    NM, NR, NA = len(model.mach_grid), len(model.re_grid), len(model.alpha_grid)
    N = model.cn_comp.shape[0]
    assert model.ca_table_off.shape == (NM, NR, NA)
    assert model.ca_table_on.shape == (NM, NR, NA)
    assert model.cn_comp.shape == (N, NM, NR, NA)
    assert model.cp_comp.shape == (N, NM, NR, NA)
    assert model.fin_comp_idx >= 0


def test_grids_are_sorted(tmp_path):
    model = build_aero_model(_component_dir(tmp_path))
    assert np.all(np.diff(model.mach_grid) > 0)
    assert np.all(np.diff(model.re_grid) > 0)
    assert np.all(np.diff(model.alpha_grid) > 0)


def test_real_aero_dir_loads(aero_dir):
    """The committed aero_tables directory must load without errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_aero_model(aero_dir)
    assert isinstance(model, AeroModel)


# ---------------------------------------------------------------------------
# ca_at interpolation tests
# ---------------------------------------------------------------------------

def test_ca_at_exact_grid_point(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    # At M=0.5, Re=1e6, alpha=0 → CA=0.5
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table_off,
                   0.5, 1e6, 0.0)
    assert result == pytest.approx(0.5, rel=1e-6)


def test_ca_at_constant_table(tmp_path):
    """CA=0.5 everywhere, so interpolated value at any (M,Re,alpha) is 0.5."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table_off,
                   1.0, 1e6, math.radians(7.0))
    assert result == pytest.approx(0.5, rel=1e-4)


def test_ca_at_clamped_below_min_mach(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table_off,
                   0.01, 1e6, 0.0)
    assert result == pytest.approx(0.5, rel=1e-4)


def test_ca_at_clamped_above_max_alpha(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table_off,
                   0.5, 1e6, math.radians(90.0))
    assert result == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# cn_cp_at interpolation tests
# ---------------------------------------------------------------------------

def test_cn_zero_at_zero_aoa(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    cn, _ = cn_cp_at(m.mach_grid, m.re_grid, m.alpha_grid,
                     m.cn_table, m.cp_table, 0.5, 1e6, 0.0)
    assert cn == pytest.approx(0.0, abs=1e-10)


def test_cn_at_exact_grid_point(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    # At alpha=5°: CN = 2.0 * sin(5°) ≈ 2.0 * 0.08727 = 0.17453... ≈ 0.175
    cn, _ = cn_cp_at(m.mach_grid, m.re_grid, m.alpha_grid,
                     m.cn_table, m.cp_table, 0.5, 1e6, math.radians(5.0))
    assert cn == pytest.approx(0.175, abs=2e-3)


def test_cp_constant_table(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    _, cp = cn_cp_at(m.mach_grid, m.re_grid, m.alpha_grid,
                     m.cn_table, m.cp_table, 0.5, 1e6, math.radians(5.0))
    assert cp == pytest.approx(1.5, rel=1e-4)


# ---------------------------------------------------------------------------
# Multi-component whole-vehicle assembly tests
# ---------------------------------------------------------------------------

def test_ca_total_is_sum_of_components(tmp_path):
    """At any (M, Re, alpha): CA_total = CA_nosecone + CA_fins."""
    m = build_aero_model(_component_dir(tmp_path))
    # Both components have constant CA: 0.3 + 0.2 = 0.5
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table_off,
                   0.5, 1e6, math.radians(5.0))
    assert result == pytest.approx(0.5, rel=1e-4)


def test_cn_total_is_sum_of_components(tmp_path):
    """Sum of per-component CN tables equals expected whole-vehicle CN."""
    m = build_aero_model(_component_dir(tmp_path))
    # CN_nose + CN_fins at M=0.5, Re=1e6, alpha=5°:
    # 0.130900 + 0.043633 = 0.174533 ≈ 0.175
    alpha_deg = math.degrees(math.radians(5.0))
    cn_total = 0.0
    for i in range(m.cn_comp.shape[0]):
        cn_total += float(np.interp(alpha_deg, m.alpha_grid,
                                    m.cn_comp[i, 0, 0, :]))
    assert cn_total == pytest.approx(0.175, abs=2e-3)


def test_cp_moment_balance(tmp_path):
    """CP_whole from aero_forces_moments = Σ(CN_i·CP_i)/Σ(CN_i) ≈ 0.875 m."""
    m = build_aero_model(_component_dir(tmp_path))
    # Static case: no rotation, bulk AoA = 5°, unit conditions
    alpha_rad = math.radians(5.0)
    V = 100.0
    u_rel = V * math.cos(alpha_rad)
    w_rel = V * math.sin(alpha_rad)
    _, _, _, _, _, cp_whole = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table_off, m.ca_table_on, False,
        m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components, m.cn_alpha_comp,
        0.5, 1e6,
        1.0, V, 1.0,        # rho, V, A_ref
        u_rel, 0.0, w_rel,  # u_rel, v_rel, w_rel
        0.0, 0.0,           # q_rate, r_rate
        1.0,                # cg
    )
    assert cp_whole == pytest.approx(0.875, rel=1e-3)


# ---------------------------------------------------------------------------
# fin_comp_idx / cn_alpha_comp tests
# ---------------------------------------------------------------------------

def test_fin_comp_idx_valid_with_fins_file(tmp_path):
    m = build_aero_model(_component_dir(tmp_path))
    assert m.fin_comp_idx >= 0
    result = _interp2(m.mach_grid, m.re_grid,
                      m.cn_alpha_comp[m.fin_comp_idx], 0.5, 1e6)
    assert result == pytest.approx(0.5, rel=1e-3)


def test_fin_comp_idx_negative_single_file(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    assert m.fin_comp_idx == -1


# ---------------------------------------------------------------------------
# Damping matches Mandell linearisation at small q (§18.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_rate", [1e-4, 1e-3, 1e-2])
def test_pitch_damping_matches_mandell_small_q(tmp_path, q_rate):
    """Pure pitch rate (no bulk AoA): τ_pitch matches Mandell to within 1%."""
    m = _damp_model(tmp_path)

    # u_rel = V, v_rel = w_rel = 0  → zero bulk AoA, pure rotation
    _, _, _, tau_pitch, _, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table_off, m.ca_table_on, False,
        m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components, m.cn_alpha_comp,
        _DAMP_MACH, 1e6,
        _DAMP_RHO, _DAMP_V, _DAMP_A_REF,
        _DAMP_V, 0.0, 0.0,   # u_rel, v_rel, w_rel
        q_rate, 0.0,           # q_rate, r_rate
        _DAMP_CG,
    )

    expected = _mandell_tau_pitch(q_rate)
    assert tau_pitch == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("q_rate", [1e-4, 1e-3, 1e-2])
def test_yaw_damping_matches_mandell_small_r(tmp_path, q_rate):
    """Pure yaw rate: τ_yaw matches Mandell to within 1% (symmetry of model)."""
    m = _damp_model(tmp_path)

    _, _, _, _, tau_yaw, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table_off, m.ca_table_on, False,
        m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components, m.cn_alpha_comp,
        _DAMP_MACH, 1e6,
        _DAMP_RHO, _DAMP_V, _DAMP_A_REF,
        _DAMP_V, 0.0, 0.0,
        0.0, q_rate,           # q_rate=0, r_rate=q_rate
        _DAMP_CG,
    )

    expected = _mandell_tau_pitch(q_rate)  # symmetric model → same magnitude
    assert tau_yaw == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Damping sign convention
# ---------------------------------------------------------------------------

def test_pitch_damping_opposes_positive_q(tmp_path):
    """Positive pitch rate → negative (nose-down restoring) pitch moment."""
    m = _damp_model(tmp_path)
    _, _, _, tau_pitch, _, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table_off, m.ca_table_on, False,
        m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components, m.cn_alpha_comp,
        _DAMP_MACH, 1e6,
        _DAMP_RHO, _DAMP_V, _DAMP_A_REF,
        _DAMP_V, 0.0, 0.0,
        0.01, 0.0,
        _DAMP_CG,
    )
    assert tau_pitch < 0.0


def test_yaw_damping_opposes_positive_r(tmp_path):
    """Positive yaw rate → negative yaw moment."""
    m = _damp_model(tmp_path)
    _, _, _, _, tau_yaw, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table_off, m.ca_table_on, False,
        m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components, m.cn_alpha_comp,
        _DAMP_MACH, 1e6,
        _DAMP_RHO, _DAMP_V, _DAMP_A_REF,
        _DAMP_V, 0.0, 0.0,
        0.0, 0.01,
        _DAMP_CG,
    )
    assert tau_yaw < 0.0


# ---------------------------------------------------------------------------
# Restoring moment still present without rotation
# ---------------------------------------------------------------------------

def test_restoring_moment_no_rotation(tmp_path):
    """Static AoA with no rotation gives a non-zero pitch moment."""
    m = _damp_model(tmp_path)
    alpha = math.radians(5.0)
    u_rel = _DAMP_V * math.cos(alpha)
    w_rel = _DAMP_V * math.sin(alpha)

    _, _, _, tau_pitch, _, _ = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table_off, m.ca_table_on, False,
        m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components, m.cn_alpha_comp,
        _DAMP_MACH, 1e6,
        _DAMP_RHO, _DAMP_V, _DAMP_A_REF,
        u_rel, 0.0, w_rel,
        0.0, 0.0,
        _DAMP_CG,
    )
    # At α=5°, CP=0.875 m is forward of CG=1.0 m (unstable geometry):
    # positive w_rel → nose-up → destabilising τ_pitch > 0
    assert tau_pitch > 0.0


# ---------------------------------------------------------------------------
# cp_whole from per-component loop
# ---------------------------------------------------------------------------

def test_cp_whole_at_nonzero_aoa(tmp_path):
    """cp_whole = Σ(CN_i·CP_i)/Σ(CN_i) at static AoA ≈ 0.875 m."""
    m = _damp_model(tmp_path)
    alpha = math.radians(5.0)
    u_rel = _DAMP_V * math.cos(alpha)
    w_rel = _DAMP_V * math.sin(alpha)

    _, _, _, _, _, cp_whole = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table_off, m.ca_table_on, False,
        m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components, m.cn_alpha_comp,
        _DAMP_MACH, 1e6,
        _DAMP_RHO, _DAMP_V, _DAMP_A_REF,
        u_rel, 0.0, w_rel,
        0.0, 0.0,
        _DAMP_CG,
    )
    assert cp_whole == pytest.approx(0.875, rel=1e-3)


def test_cp_whole_fallback_to_cg_at_zero_aoa(tmp_path):
    """At zero AoA (no rotation), cp_whole falls back to CG."""
    m = _damp_model(tmp_path)
    _, _, _, _, _, cp_whole = aero_forces_moments(
        m.mach_grid, m.re_grid, m.alpha_grid,
        m.ca_table_off, m.ca_table_on, False,
        m.cn_table, m.cp_table,
        m.cn_comp, m.cp_comp, m.has_components, m.cn_alpha_comp,
        _DAMP_MACH, 1e6,
        _DAMP_RHO, _DAMP_V, _DAMP_A_REF,
        _DAMP_V, 0.0, 0.0,  # pure axial, zero lateral
        0.0, 0.0,
        _DAMP_CG,
    )
    assert cp_whole == pytest.approx(_DAMP_CG, abs=1e-6)
