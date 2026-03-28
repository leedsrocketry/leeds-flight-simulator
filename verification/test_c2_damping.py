"""Tests for aerodynamics.py — C_Nα fitting and pitch/yaw damping sums."""

import math
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from aerodynamics import build_aero_model, damping_sum_at


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Nosecone: C_Nα=1.5, CP=0.5 m; Fins: C_Nα=0.5, CP=2.0 m
# Same data as test_aero_interp.py so expected analytical values are shared.
_NOSECONE_CSV = textwrap.dedent("""\
    Mach,Reynolds,AoA_deg,CA,CN,CP_m
    0.5,1000000,0.0,0.300,0.000000,0.500
    0.5,1000000,5.0,0.300,0.130900,0.500
    0.5,1000000,10.0,0.300,0.261799,0.500
    1.5,1000000,0.0,0.300,0.000000,0.500
    1.5,1000000,5.0,0.300,0.130900,0.500
    1.5,1000000,10.0,0.300,0.261799,0.500
""")

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


def _model(tmp_path: Path):
    d = tmp_path / "aero"
    d.mkdir()
    _write(d, "nosecone.csv", _NOSECONE_CSV)
    _write(d, "fins.csv", _FIN_CSV)
    return build_aero_model(d)


# ---------------------------------------------------------------------------
# C_Nα fitting
# ---------------------------------------------------------------------------

def test_cna_nose_value(tmp_path):
    """Nosecone C_Nα ≈ 1.5 rad⁻¹ from linear fit to synthetic data."""
    m = _model(tmp_path)
    # cna_sum = C_Nα_nose + C_Nα_fins = 1.5 + 0.5 = 2.0
    # cna_cp_sum = 1.5·0.5 + 0.5·2.0 = 1.75
    # Extract total C_Nα at (M=0.5, Re=1e6)
    im = np.searchsorted(m.mach_grid, 0.5)
    ir = np.searchsorted(m.re_grid, 1e6)
    assert m.cna_sum[im, ir] == pytest.approx(2.0, rel=1e-2)


def test_cna_fins_value(tmp_path):
    m = _model(tmp_path)
    im = np.searchsorted(m.mach_grid, 0.5)
    ir = np.searchsorted(m.re_grid, 1e6)
    assert m.cn_alpha_fins[im, ir] == pytest.approx(0.5, rel=1e-2)


def test_cna_sum_positive(tmp_path):
    m = _model(tmp_path)
    assert np.all(m.cna_sum >= 0.0)


def test_cna_cp_sum_value(tmp_path):
    """cna_cp_sum = Σ C_Nα_i · CP_i = 1.5·0.5 + 0.5·2.0 = 1.75."""
    m = _model(tmp_path)
    im = np.searchsorted(m.mach_grid, 0.5)
    ir = np.searchsorted(m.re_grid, 1e6)
    assert m.cna_cp_sum[im, ir] == pytest.approx(1.75, rel=1e-2)


def test_cna_cp2_sum_value(tmp_path):
    """cna_cp2_sum = 1.5·0.25 + 0.5·4.0 = 0.375 + 2.0 = 2.375."""
    m = _model(tmp_path)
    im = np.searchsorted(m.mach_grid, 0.5)
    ir = np.searchsorted(m.re_grid, 1e6)
    assert m.cna_cp2_sum[im, ir] == pytest.approx(2.375, rel=1e-2)


# ---------------------------------------------------------------------------
# damping_sum_at
# ---------------------------------------------------------------------------

def test_damping_sum_hand_calc(tmp_path):
    """At CG=1.0 m:  Σ C_Nα_i·(CP_i−CG)² = 1.5·0.25 + 0.5·1.0 = 0.875."""
    m = _model(tmp_path)
    result = damping_sum_at(
        m.mach_grid, m.re_grid,
        m.cna_sum, m.cna_cp_sum, m.cna_cp2_sum,
        0.5, 1e6, 1.0,
    )
    assert result == pytest.approx(0.875, rel=1e-2)


def test_damping_sum_at_cp_centroid(tmp_path):
    """At CG = CN-weighted mean CP, damping sum is minimised (≥ 0)."""
    m = _model(tmp_path)
    # CG = cna_cp_sum / cna_sum = 1.75 / 2.0 = 0.875
    cg_min = 1.75 / 2.0
    result = damping_sum_at(
        m.mach_grid, m.re_grid,
        m.cna_sum, m.cna_cp_sum, m.cna_cp2_sum,
        0.5, 1e6, cg_min,
    )
    assert result >= 0.0
    # Should be less than the CG=1.0 case
    result_1 = damping_sum_at(
        m.mach_grid, m.re_grid,
        m.cna_sum, m.cna_cp_sum, m.cna_cp2_sum,
        0.5, 1e6, 1.0,
    )
    assert result <= result_1


def test_damping_sum_nonnegative(tmp_path):
    """Damping sum must be ≥ 0 for any CG."""
    m = _model(tmp_path)
    for cg in np.linspace(0.0, 3.0, 30):
        result = damping_sum_at(
            m.mach_grid, m.re_grid,
            m.cna_sum, m.cna_cp_sum, m.cna_cp2_sum,
            0.5, 1e6, cg,
        )
        assert result >= 0.0


def test_damping_sum_zero_single_file(tmp_path):
    """Single-file model has zero damping sums."""
    _WHOLE = textwrap.dedent("""\
        Mach,Reynolds,AoA_deg,CA,CN,CP_m
        0.5,1000000,0.0,0.500,0.000,1.500
        0.5,1000000,5.0,0.500,0.175,1.500
        0.5,1000000,10.0,0.500,0.349,1.500
    """)
    d = tmp_path / "aero_single"
    d.mkdir()
    _write(d, "vehicle.csv", _WHOLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build_aero_model(d)
    result = damping_sum_at(
        m.mach_grid, m.re_grid,
        m.cna_sum, m.cna_cp_sum, m.cna_cp2_sum,
        0.5, 1e6, 1.0,
    )
    assert result == pytest.approx(0.0, abs=1e-10)


def test_damping_sum_increases_with_separation(tmp_path):
    """Greater CG–CP separation → larger damping sum."""
    m = _model(tmp_path)
    ds_near = damping_sum_at(
        m.mach_grid, m.re_grid,
        m.cna_sum, m.cna_cp_sum, m.cna_cp2_sum,
        0.5, 1e6, 1.0,
    )
    ds_far = damping_sum_at(
        m.mach_grid, m.re_grid,
        m.cna_sum, m.cna_cp_sum, m.cna_cp2_sum,
        0.5, 1e6, 0.0,   # far from CP centroid at 0.875
    )
    assert ds_far > ds_near
