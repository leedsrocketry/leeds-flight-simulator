"""Tests for aerodynamics.py — table loading and coefficient interpolation."""

import math
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from aerodynamics import (
    AeroModel,
    build_aero_model,
    ca_at,
    cn_cp_at,
    cn_alpha_fins_at,
    damping_sum_at,
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
# CA=0.3, CN=1.5·α_rad, CP=0.5 m
# CN at 5° = 1.5·0.087266 = 0.130900; at 10° = 0.261799
_NOSECONE_CSV = textwrap.dedent("""\
    Mach,Reynolds,AoA_deg,CA,CN,CP_m
    0.5,1000000,0.0,0.300,0.000000,0.500
    0.5,1000000,5.0,0.300,0.130900,0.500
    0.5,1000000,10.0,0.300,0.261799,0.500
    1.5,1000000,0.0,0.300,0.000000,0.500
    1.5,1000000,5.0,0.300,0.130900,0.500
    1.5,1000000,10.0,0.300,0.261799,0.500
""")

# Per-component fin file (stem contains 'fin').
# CA=0.2, CN=0.5·α_rad, CP=2.0 m
# CN at 5° = 0.5·0.087266 = 0.043633; at 10° = 0.087266
_FIN_CSV = textwrap.dedent("""\
    Mach,Reynolds,AoA_deg,CA,CN,CP_m
    0.5,1000000,0.0,0.200,0.000000,2.000
    0.5,1000000,5.0,0.200,0.043633,2.000
    0.5,1000000,10.0,0.200,0.087266,2.000
    1.5,1000000,0.0,0.200,0.000000,2.000
    1.5,1000000,5.0,0.200,0.043633,2.000
    1.5,1000000,10.0,0.200,0.087266,2.000
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


def test_grid_shapes_consistent(tmp_path):
    model = build_aero_model(_component_dir(tmp_path))
    NM, NR, NA = len(model.mach_grid), len(model.re_grid), len(model.alpha_grid)
    assert model.ca_table.shape == (NM, NR, NA)
    assert model.cn_table.shape == (NM, NR, NA)
    assert model.cp_table.shape == (NM, NR, NA)
    assert model.cna_sum.shape == (NM, NR)
    assert model.cn_alpha_fins.shape == (NM, NR)


def test_grids_are_sorted(tmp_path):
    model = build_aero_model(_component_dir(tmp_path))
    assert np.all(np.diff(model.mach_grid) > 0)
    assert np.all(np.diff(model.re_grid) > 0)
    assert np.all(np.diff(model.alpha_grid) > 0)


def test_real_aero_dir_loads():
    """The committed input/aero_tables directory must load without errors."""
    real = Path(__file__).parent.parent / "input" / "aero_tables"
    if not real.exists() or not list(real.glob("*.csv")):
        pytest.skip("input/aero_tables not present or empty")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_aero_model(real)
    assert isinstance(model, AeroModel)


# ---------------------------------------------------------------------------
# ca_at interpolation tests
# ---------------------------------------------------------------------------

def test_ca_at_exact_grid_point(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    # At M=0.5, Re=1e6, alpha=0 → CA=0.5
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table,
                   0.5, 1e6, 0.0)
    assert result == pytest.approx(0.5, rel=1e-6)


def test_ca_at_constant_table(tmp_path):
    """CA=0.5 everywhere, so interpolated value at any (M,Re,alpha) is 0.5."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table,
                   1.0, 1e6, math.radians(7.0))
    assert result == pytest.approx(0.5, rel=1e-4)


def test_ca_at_clamped_below_min_mach(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table,
                   0.01, 1e6, 0.0)
    assert result == pytest.approx(0.5, rel=1e-4)


def test_ca_at_clamped_above_max_alpha(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table,
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
    result = ca_at(m.mach_grid, m.re_grid, m.alpha_grid, m.ca_table,
                   0.5, 1e6, math.radians(5.0))
    assert result == pytest.approx(0.5, rel=1e-4)


def test_cn_total_is_sum_of_components(tmp_path):
    m = build_aero_model(_component_dir(tmp_path))
    # C_Nα_nose=1.5, C_Nα_fins=0.5, total C_Nα=2.0
    # At alpha=5°: CN ≈ 2.0 * rad(5°) ≈ 0.175
    cn, _ = cn_cp_at(m.mach_grid, m.re_grid, m.alpha_grid,
                     m.cn_table, m.cp_table, 0.5, 1e6, math.radians(5.0))
    assert cn == pytest.approx(0.175, abs=2e-3)


def test_cp_moment_balance(tmp_path):
    """CP_total = Σ(CN_i·CP_i)/Σ(CN_i) = (1.5·0.5 + 0.5·2.0)/2.0 = 0.875 m."""
    m = build_aero_model(_component_dir(tmp_path))
    _, cp = cn_cp_at(m.mach_grid, m.re_grid, m.alpha_grid,
                     m.cn_table, m.cp_table, 0.5, 1e6, math.radians(5.0))
    assert cp == pytest.approx(0.875, rel=1e-3)


# ---------------------------------------------------------------------------
# cn_alpha_fins_at tests
# ---------------------------------------------------------------------------

def test_cn_alpha_fins_nonzero_with_fins_file(tmp_path):
    m = build_aero_model(_component_dir(tmp_path))
    result = cn_alpha_fins_at(m.mach_grid, m.re_grid, m.cn_alpha_fins,
                              0.5, 1e6)
    assert result == pytest.approx(0.5, rel=1e-3)


def test_cn_alpha_fins_zero_single_file(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(_single_dir(tmp_path))
    result = cn_alpha_fins_at(m.mach_grid, m.re_grid, m.cn_alpha_fins,
                              0.5, 1e6)
    assert result == pytest.approx(0.0, abs=1e-10)
