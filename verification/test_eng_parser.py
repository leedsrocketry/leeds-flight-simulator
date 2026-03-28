"""Tests for .eng parser (config.load_motor) and motor.py physics functions."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from config import load_motor, load_vehicle_config, MotorData
from motor import (
    build_motor_model,
    thrust_at,
    mdot_at,
    m_prop_at,
    inertia_at,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_ENG = """\
; Test motor
M2020 98 732 0 5.0 8.0 TestMfr
0.0    0.0
0.1  2000.0
1.0  2000.0
2.0     0.0
"""

_SIMPLE_ENG_NO_BURNOUT = """\
; No explicit burnout point
M2020 98 732 0 5.0 8.0 TestMfr
0.0    0.0
0.1  2000.0
2.0   50.0
"""


def _write_eng(tmp_path: Path, content: str, name: str = "motor.eng") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parse_basic(tmp_path):
    p = _write_eng(tmp_path, _SIMPLE_ENG)
    md = load_motor(p)
    assert isinstance(md, MotorData)
    assert md.name == "M2020"
    assert md.m_prop_kg == pytest.approx(5.0)
    assert md.m_motor_kg == pytest.approx(8.0)


def test_parse_time_thrust_arrays(tmp_path):
    p = _write_eng(tmp_path, _SIMPLE_ENG)
    md = load_motor(p)
    assert md.time_s[0] == pytest.approx(0.0)
    assert md.thrust_n[1] == pytest.approx(2000.0)
    assert md.thrust_n[-1] == pytest.approx(0.0)


def test_parse_arrays_are_float64(tmp_path):
    md = load_motor(_write_eng(tmp_path, _SIMPLE_ENG))
    assert md.time_s.dtype == np.float64
    assert md.thrust_n.dtype == np.float64


def test_burnout_appended_if_missing(tmp_path):
    """If the last thrust point is non-zero, a burnout point must be appended."""
    p = _write_eng(tmp_path, _SIMPLE_ENG_NO_BURNOUT)
    md = load_motor(p)
    assert md.thrust_n[-1] == pytest.approx(0.0)
    assert md.time_s[-1] == pytest.approx(md.time_s[-2])


def test_comments_ignored(tmp_path):
    content = "; full comment\n; another\n" + _SIMPLE_ENG
    md = load_motor(_write_eng(tmp_path, content))
    assert md.name == "M2020"


def test_missing_header_fields_raises(tmp_path):
    bad = "M2020 98 732 0 5.0\n0.0 0.0\n1.0 100.0\n"
    with pytest.raises(ValueError, match="7 fields"):
        load_motor(_write_eng(tmp_path, bad))


def test_total_less_than_prop_raises(tmp_path):
    bad = "M2020 98 732 0 5.0 4.0 Mfr\n0.0 0.0\n1.0 100.0\n2.0 0.0\n"
    with pytest.raises(ValueError, match="exceed propellant"):
        load_motor(_write_eng(tmp_path, bad))


def test_non_monotonic_times_raises(tmp_path):
    bad = "M2020 98 732 0 5.0 8.0 Mfr\n0.0 0.0\n2.0 100.0\n1.0 50.0\n3.0 0.0\n"
    with pytest.raises(ValueError, match="strictly increasing"):
        load_motor(_write_eng(tmp_path, bad))


def test_real_motor_file_loads():
    real = Path(__file__).parent.parent / "input" / "motor.eng"
    if not real.exists():
        pytest.skip("input/motor.eng not present")
    md = load_motor(real)
    assert md.m_prop_kg > 0
    assert md.thrust_n[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MotorModel construction
# ---------------------------------------------------------------------------

def _make_model(tmp_path):
    """Build a MotorModel from the simple eng + a matching vehicle config."""
    vehicle_yaml = tmp_path / "vehicle.yaml"
    vehicle_yaml.write_text(
        "geometry:\n  diameter: 0.1\n  length: 2.0\n"
        "mass:\n  wet: 20.0\n  dry: 15.0\n  cg_dry: 1.0\n  motor_cg_loaded: 1.8\n"
        "inertia:\n  I_R_wet: 0.01\n  I_R_dry: 0.008\n  I_L_wet: 4.0\n  I_L_dry: 3.0\n"
        "nozzle:\n  exit: 1.95\n"
        "recovery:\n  CdA_drogue: 0.15\n  CdA_main: 2.8\n  deploy_altitude_agl: 305\n"
        "roll:\n  r_fin: 0.09\n",
        encoding="utf-8",
    )
    md = load_motor(_write_eng(tmp_path, _SIMPLE_ENG))
    vc = load_vehicle_config(vehicle_yaml)
    return build_motor_model(md, vc), md, vc


def test_build_motor_model(tmp_path):
    from motor import MotorModel
    model, _, _ = _make_model(tmp_path)
    assert isinstance(model, MotorModel)


def test_total_impulse_positive(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert model.total_impulse > 0


def test_total_impulse_value(tmp_path):
    """Trapz of constant 2000 N over [0.1, 2.0] s ≈ 3800 N·s."""
    model, _, _ = _make_model(tmp_path)
    # From _SIMPLE_ENG: 0→0, 0.1→2000, 1.0→2000, 2.0→0
    # Trapz: 0.5*(0+2000)*0.1 + 2000*(1.0-0.1) + 0.5*(2000+0)*1.0 = 100+1800+1000 = 2900
    assert model.total_impulse == pytest.approx(2900.0, rel=1e-6)


def test_m_casing(tmp_path):
    model, md, _ = _make_model(tmp_path)
    assert model.m_casing == pytest.approx(md.m_motor_kg - md.m_prop_kg)


# ---------------------------------------------------------------------------
# thrust_at
# ---------------------------------------------------------------------------

def test_thrust_at_zero(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert thrust_at(model.times, model.thrusts, 0.0) == pytest.approx(0.0)


def test_thrust_at_midpoint(tmp_path):
    model, _, _ = _make_model(tmp_path)
    # Between t=0.1 (2000 N) and t=1.0 (2000 N) → 2000 N
    assert thrust_at(model.times, model.thrusts, 0.5) == pytest.approx(2000.0)


def test_thrust_at_burnout(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert thrust_at(model.times, model.thrusts, 2.0) == pytest.approx(0.0)


def test_thrust_after_burnout(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert thrust_at(model.times, model.thrusts, 10.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# m_prop_at
# ---------------------------------------------------------------------------

def test_m_prop_at_zero(tmp_path):
    model, md, _ = _make_model(tmp_path)
    assert m_prop_at(model.times, model.thrusts, model.m_prop_0,
                     model.total_impulse, 0.0) == pytest.approx(md.m_prop_kg)


def test_m_prop_at_burnout(tmp_path):
    model, _, _ = _make_model(tmp_path)
    mp = m_prop_at(model.times, model.thrusts, model.m_prop_0,
                   model.total_impulse, 2.0)
    assert mp == pytest.approx(0.0, abs=1e-10)


def test_m_prop_decreasing(tmp_path):
    model, _, _ = _make_model(tmp_path)
    t_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
    mp_vals = [m_prop_at(model.times, model.thrusts, model.m_prop_0,
                         model.total_impulse, t) for t in t_vals]
    assert all(mp_vals[i] >= mp_vals[i + 1] for i in range(len(mp_vals) - 1))


# ---------------------------------------------------------------------------
# mdot_at
# ---------------------------------------------------------------------------

def test_mdot_proportional_to_thrust(tmp_path):
    """ṁ = m_prop_0 * F / I_total, so ṁ/F = constant."""
    model, _, _ = _make_model(tmp_path)
    t1, t2 = 0.3, 0.8
    mdot1 = mdot_at(model.times, model.thrusts, model.m_prop_0,
                    model.total_impulse, t1)
    mdot2 = mdot_at(model.times, model.thrusts, model.m_prop_0,
                    model.total_impulse, t2)
    f1 = thrust_at(model.times, model.thrusts, t1)
    f2 = thrust_at(model.times, model.thrusts, t2)
    assert mdot1 / f1 == pytest.approx(mdot2 / f2, rel=1e-9)


# ---------------------------------------------------------------------------
# inertia_at
# ---------------------------------------------------------------------------

def test_inertia_at_zero_is_wet(tmp_path):
    model, _, vc = _make_model(tmp_path)
    I_R, I_L = inertia_at(model.times, model.thrusts, model.m_prop_0,
                           model.total_impulse,
                           model.I_R_wet, model.I_R_dry,
                           model.I_L_wet, model.I_L_dry, 0.0)
    assert I_R == pytest.approx(vc.inertia.I_R_wet)
    assert I_L == pytest.approx(vc.inertia.I_L_wet)


def test_inertia_at_burnout_is_dry(tmp_path):
    model, _, vc = _make_model(tmp_path)
    I_R, I_L = inertia_at(model.times, model.thrusts, model.m_prop_0,
                           model.total_impulse,
                           model.I_R_wet, model.I_R_dry,
                           model.I_L_wet, model.I_L_dry, 2.0)
    assert I_R == pytest.approx(vc.inertia.I_R_dry, abs=1e-10)
    assert I_L == pytest.approx(vc.inertia.I_L_dry, abs=1e-10)


def test_inertia_decreasing(tmp_path):
    model, _, _ = _make_model(tmp_path)
    I_L_vals = [
        inertia_at(model.times, model.thrusts, model.m_prop_0,
                   model.total_impulse,
                   model.I_R_wet, model.I_R_dry,
                   model.I_L_wet, model.I_L_dry, t)[1]
        for t in [0.0, 0.5, 1.0, 2.0]
    ]
    assert all(I_L_vals[i] >= I_L_vals[i + 1] - 1e-12
               for i in range(len(I_L_vals) - 1))
