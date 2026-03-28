"""Tests for .eng parser (config.load_motor) and motor.py physics functions."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from config import load_motor, load_vehicle_config, MotorData
from motor import (
    build_motor_model,
    thrust_at,
    thrust_corrected_at,
    mdot_at,
    m_prop_at,
    mass_at,
    cg_at,
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
# MotorModel construction helpers
# ---------------------------------------------------------------------------

# Vehicle yaml for tests — matches _SIMPLE_ENG (m_prop=5.0 kg)
_VEHICLE_YAML = (
    "motor: \"motor.eng\"\naero_tables: \"aero_tables\"\n"
    "geometry:\n  diameter: 0.1\n  length: 2.0\n"
    "  nozzle_position: 1.95\n  nozzle_diameter: 0.05\n  fin_cp_radius: 0.09\n"
    "mass:\n  wet_mass: 20.0\n  wet_cg: 0.90\n  wet_motor_cg: 1.40\n"
    "  propellant_inertia_roll: 0.001\n  propellant_inertia_lateral: 0.20\n"
    "  wet_inertia_lateral: 4.0\n  wet_inertia_roll: 0.010\n"
    "recovery:\n"
    "  drogue:\n    cd: 1.5\n    area: 0.05\n    threshold: apogee\n"
    "  main:\n    cd: 2.2\n    area: 2.5\n    threshold: 305\n"
)
# Derived values for hand-checking (m_prop=5.0 from _SIMPLE_ENG):
#   m_dry = 20.0 - 5.0 = 15.0 kg
#   cg_dry = (20.0*0.90 - 5.0*1.40) / 15.0 = 11.0/15.0 ≈ 0.7333 m
#   I_roll_dry = 0.010 - 0.001 = 0.009 kg·m²
#   d_prop_wet = 1.40 - 0.90 = 0.50 m
#   I_prop_lat_wet_cg = 0.20 + 5.0*0.25 = 1.45 kg·m²
#   I_lat_dry_wet_cg  = 4.0 - 1.45 = 2.55 kg·m²
#   d_dry_wet = 0.7333 - 0.90 = −0.1667 m
#   I_lateral_dry = 2.55 - 15.0*0.02778 = 2.1333 kg·m²


def _make_model(tmp_path):
    """Build a MotorModel from _SIMPLE_ENG + matching vehicle config."""
    vehicle_yaml = tmp_path / "vehicle.yaml"
    vehicle_yaml.write_text(_VEHICLE_YAML, encoding="utf-8")
    md = load_motor(_write_eng(tmp_path, _SIMPLE_ENG))
    vc = load_vehicle_config(vehicle_yaml)
    return build_motor_model(md, vc), md, vc


# ---------------------------------------------------------------------------
# MotorModel construction
# ---------------------------------------------------------------------------

def test_build_motor_model(tmp_path):
    from motor import MotorModel
    model, _, _ = _make_model(tmp_path)
    assert isinstance(model, MotorModel)


def test_total_impulse_positive(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert model.total_impulse > 0


def test_total_impulse_value(tmp_path):
    """Trapz of _SIMPLE_ENG: 0→0, 0.1→2000, 1.0→2000, 2.0→0 ≈ 2900 N·s."""
    model, _, _ = _make_model(tmp_path)
    assert model.total_impulse == pytest.approx(2900.0, rel=1e-6)


def test_m_casing(tmp_path):
    model, md, _ = _make_model(tmp_path)
    assert model.m_casing == pytest.approx(md.m_motor_kg - md.m_prop_kg)


def test_dry_mass_computed(tmp_path):
    model, md, _ = _make_model(tmp_path)
    assert model.m_dry == pytest.approx(20.0 - md.m_prop_kg)


def test_dry_cg_computed(tmp_path):
    model, md, vc = _make_model(tmp_path)
    m_dry = vc.mass.wet_mass - md.m_prop_kg
    expected = (vc.mass.wet_mass * vc.mass.wet_cg
                - md.m_prop_kg * vc.mass.wet_motor_cg) / m_dry
    assert model.cg_dry == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# thrust_at
# ---------------------------------------------------------------------------

def test_thrust_at_zero(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert thrust_at(model.times, model.thrusts, 0.0) == pytest.approx(0.0)


def test_thrust_at_midpoint(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert thrust_at(model.times, model.thrusts, 0.5) == pytest.approx(2000.0)


def test_thrust_at_burnout(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert thrust_at(model.times, model.thrusts, 2.0) == pytest.approx(0.0)


def test_thrust_after_burnout(tmp_path):
    model, _, _ = _make_model(tmp_path)
    assert thrust_at(model.times, model.thrusts, 10.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# thrust_corrected_at
# ---------------------------------------------------------------------------

def test_thrust_corrected_at_sea_level(tmp_path):
    """At altitude=0, pressure correction is zero."""
    model, _, _ = _make_model(tmp_path)
    F_base = thrust_at(model.times, model.thrusts, 0.5)
    F_corr = thrust_corrected_at(model.times, model.thrusts,
                                  model.nozzle_area, 0.0, 0.5)
    assert F_corr == pytest.approx(F_base)


def test_thrust_corrected_increases_with_altitude(tmp_path):
    """Higher altitude → lower ambient pressure → higher thrust."""
    model, _, _ = _make_model(tmp_path)
    F_sea = thrust_corrected_at(model.times, model.thrusts,
                                 model.nozzle_area, 0.0, 0.5)
    F_high = thrust_corrected_at(model.times, model.thrusts,
                                  model.nozzle_area, 5000.0, 0.5)
    assert F_high > F_sea


def test_thrust_corrected_zero_area_no_change(tmp_path):
    """With nozzle_area=0, correction is zero regardless of altitude."""
    model, _, _ = _make_model(tmp_path)
    F_base = thrust_at(model.times, model.thrusts, 0.5)
    F_corr = thrust_corrected_at(model.times, model.thrusts, 0.0, 8000.0, 0.5)
    assert F_corr == pytest.approx(F_base)


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
# mass_at
# ---------------------------------------------------------------------------

def test_mass_at_zero_is_wet(tmp_path):
    model, _, vc = _make_model(tmp_path)
    result = mass_at(model.times, model.thrusts, model.m_prop_0,
                     model.total_impulse, model.m_dry, 0.0)
    assert result == pytest.approx(vc.mass.wet_mass, rel=1e-6)


def test_mass_at_burnout_is_dry(tmp_path):
    model, _, _ = _make_model(tmp_path)
    result = mass_at(model.times, model.thrusts, model.m_prop_0,
                     model.total_impulse, model.m_dry, 2.0)
    assert result == pytest.approx(model.m_dry, abs=1e-10)


# ---------------------------------------------------------------------------
# cg_at
# ---------------------------------------------------------------------------

def test_cg_at_zero_is_wet(tmp_path):
    model, _, vc = _make_model(tmp_path)
    result = cg_at(model.times, model.thrusts, model.m_prop_0, model.total_impulse,
                   model.m_dry, model.cg_dry, model.motor_cg_loaded, 0.0)
    assert result == pytest.approx(vc.mass.wet_cg, rel=1e-6)


def test_cg_at_burnout_is_dry(tmp_path):
    model, _, _ = _make_model(tmp_path)
    result = cg_at(model.times, model.thrusts, model.m_prop_0, model.total_impulse,
                   model.m_dry, model.cg_dry, model.motor_cg_loaded, 2.0)
    assert result == pytest.approx(model.cg_dry, rel=1e-6)


def test_cg_moves_forward_during_burn(tmp_path):
    """Motor is aft of CG → CG shifts forward (toward nosecone) as propellant burns."""
    model, _, vc = _make_model(tmp_path)
    # wet_motor_cg (1.40) > wet_cg (0.90), so burning propellant moves CG forward
    cg_mid = cg_at(model.times, model.thrusts, model.m_prop_0, model.total_impulse,
                   model.m_dry, model.cg_dry, model.motor_cg_loaded, 1.0)
    assert cg_mid < vc.mass.wet_cg
    assert cg_mid > model.cg_dry


# ---------------------------------------------------------------------------
# inertia_at
# ---------------------------------------------------------------------------

def _inertia(model, t):
    return inertia_at(
        model.times, model.thrusts, model.m_prop_0, model.total_impulse,
        model.m_dry, model.cg_dry, model.motor_cg_loaded,
        model.I_roll_dry, model.I_lateral_dry,
        model.prop_I_roll, model.prop_I_lateral,
        t,
    )


def test_inertia_at_zero_is_wet(tmp_path):
    """At t=0 the PAT computation must recover the user-supplied wet inertias."""
    model, _, vc = _make_model(tmp_path)
    I_roll, I_lat = _inertia(model, 0.0)
    assert I_roll == pytest.approx(vc.mass.wet_inertia_roll, rel=1e-6)
    assert I_lat == pytest.approx(vc.mass.wet_inertia_lateral, rel=1e-6)


def test_inertia_at_burnout_is_dry(tmp_path):
    """At burnout the result must equal the derived dry inertias."""
    model, _, _ = _make_model(tmp_path)
    I_roll, I_lat = _inertia(model, 2.0)
    assert I_roll == pytest.approx(model.I_roll_dry, abs=1e-10)
    assert I_lat == pytest.approx(model.I_lateral_dry, abs=1e-10)


def test_inertia_decreasing(tmp_path):
    """Lateral inertia should decrease monotonically as propellant burns."""
    model, _, _ = _make_model(tmp_path)
    I_lat_vals = [_inertia(model, t)[1] for t in [0.0, 0.5, 1.0, 2.0]]
    assert all(I_lat_vals[i] >= I_lat_vals[i + 1] - 1e-12
               for i in range(len(I_lat_vals) - 1))


def test_inertia_roll_decreasing(tmp_path):
    """Roll inertia should decrease monotonically (no PAT term)."""
    model, _, _ = _make_model(tmp_path)
    I_roll_vals = [_inertia(model, t)[0] for t in [0.0, 0.5, 1.0, 2.0]]
    assert all(I_roll_vals[i] >= I_roll_vals[i + 1] - 1e-12
               for i in range(len(I_roll_vals) - 1))
