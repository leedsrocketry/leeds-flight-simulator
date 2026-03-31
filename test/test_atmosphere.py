"""Verification tests for atmosphere.py against ISO 2533:1975 published tables.

Reference values at layer boundaries (h=0, 11000, 20000 m) are exact ISO
2533:1975 standard values.  Interior and upper-layer values are computed
analytically from the spec constants (g₀=9.80665, M=0.0289644, R*=8.31447)
— ISO 2533 uses R*=8.31432, giving a ≲0.1 % discrepancy in pressure at higher
altitudes, so we use our own constants' results as truth.
Tolerances are tight (0.1 %) to catch any constant or formula regressions.
"""

import pytest
import numpy as np
from atmosphere import (
    isa, temperature, pressure, density, speed_of_sound, dynamic_viscosity,
    isa_at_site, temperature_at_site, pressure_at_site, compute_t_offset,
)

# ---------------------------------------------------------------------------
# Reference values at selected altitudes
# Columns: h (m), T (K), p (Pa), rho (kg/m³), a (m/s), mu (Pa·s)
#
# Sea level, 11 km, and 20 km are defined boundary values from ISO 2533:1975.
# 5 km, 15 km, and 25 km are analytically derived using spec constants.
# Viscosity is from the Sutherland formula in the spec with the same constants.
# ---------------------------------------------------------------------------
_ISO_TABLE = [
    # Sea level — exact defined values
    (0,      288.150, 101_325.0,  1.22500,  340.294, 1.7894e-5),
    # Mid-troposphere — derived with spec constants; T exact, p/rho consistent
    (5_000,  255.676,  54_048.3,  0.73643,  320.529, 1.6284e-5),
    # Tropopause base — exact defined boundary values
    (11_000, 216.650,  22_632.1,  0.36392,  295.070, 1.4216e-5),
    # Tropopause mid — isothermal; p/rho computed with spec R*=8.31447
    (15_000, 216.650,  12_045.0,  0.19368,  295.070, 1.4216e-5),
    # Stratosphere-1 base — exact defined boundary values
    (20_000, 216.650,   5_474.89, 0.08803,  295.070, 1.4216e-5),
    # Stratosphere-1 mid — computed with spec constants
    (25_000, 221.552,   2_511.2,  0.03947,  298.389, 1.4484e-5),
]

_REL_TOL = 1e-3  # 0.1 % — generous enough for float64 but catches formula errors


@pytest.mark.parametrize("h, T_ref, p_ref, rho_ref, a_ref, mu_ref", _ISO_TABLE)
def test_isa_temperature(h, T_ref, p_ref, rho_ref, a_ref, mu_ref):
    T, *_ = isa(float(h))
    assert T == pytest.approx(T_ref, rel=_REL_TOL), f"T at {h} m"


@pytest.mark.parametrize("h, T_ref, p_ref, rho_ref, a_ref, mu_ref", _ISO_TABLE)
def test_isa_pressure(h, T_ref, p_ref, rho_ref, a_ref, mu_ref):
    _, p, *_ = isa(float(h))
    assert p == pytest.approx(p_ref, rel=_REL_TOL), f"p at {h} m"


@pytest.mark.parametrize("h, T_ref, p_ref, rho_ref, a_ref, mu_ref", _ISO_TABLE)
def test_isa_density(h, T_ref, p_ref, rho_ref, a_ref, mu_ref):
    _, _, rho, *_ = isa(float(h))
    assert rho == pytest.approx(rho_ref, rel=_REL_TOL), f"rho at {h} m"


@pytest.mark.parametrize("h, T_ref, p_ref, rho_ref, a_ref, mu_ref", _ISO_TABLE)
def test_isa_speed_of_sound(h, T_ref, p_ref, rho_ref, a_ref, mu_ref):
    _, _, _, a, _ = isa(float(h))
    assert a == pytest.approx(a_ref, rel=_REL_TOL), f"a at {h} m"


@pytest.mark.parametrize("h, T_ref, p_ref, rho_ref, a_ref, mu_ref", _ISO_TABLE)
def test_isa_viscosity(h, T_ref, p_ref, rho_ref, a_ref, mu_ref):
    _, _, _, _, mu = isa(float(h))
    assert mu == pytest.approx(mu_ref, rel=5e-3), f"mu at {h} m"  # Sutherland ~0.5% at boundaries


def test_individual_functions_consistent():
    """Individual helper functions must agree with the combined isa() output."""
    for h in [0.0, 5000.0, 11000.0, 15000.0, 20000.0]:
        T, p, rho, a, mu = isa(h)
        assert temperature(h) == pytest.approx(T, rel=1e-12)
        assert pressure(h) == pytest.approx(p, rel=1e-12)
        assert density(h) == pytest.approx(rho, rel=1e-12)
        assert speed_of_sound(h) == pytest.approx(a, rel=1e-12)
        assert dynamic_viscosity(h) == pytest.approx(mu, rel=1e-12)


def test_layer_boundaries_continuous():
    """Temperature and pressure must be continuous across layer boundaries."""
    for h_b in [11_000.0, 20_000.0]:
        T_below, p_below, *_ = isa(h_b - 0.001)
        T_above, p_above, *_ = isa(h_b + 0.001)
        assert T_below == pytest.approx(T_above, rel=1e-4)
        assert p_below == pytest.approx(p_above, rel=1e-4)


def test_sea_level_density():
    """Standard sea-level density must be 1.225 kg/m³."""
    assert density(0.0) == pytest.approx(1.225, rel=1e-3)


def test_mach_number_sea_level():
    """Mach 1 at sea level ≈ 340.3 m/s."""
    assert speed_of_sound(0.0) == pytest.approx(340.294, rel=1e-3)


# ---------------------------------------------------------------------------
# Site-aware atmosphere functions
# ---------------------------------------------------------------------------

def test_site_zero_offset_matches_isa():
    """With elevation=0 and t_offset=0, site functions must equal isa()."""
    for h in [0.0, 1000.0, 5000.0, 11000.0, 20000.0]:
        T_s, p_s, rho_s, a_s, mu_s = isa_at_site(h, 0.0, 0.0)
        T_i, p_i, rho_i, a_i, mu_i = isa(h)
        assert T_s == pytest.approx(T_i, rel=1e-12)
        assert p_s == pytest.approx(p_i, rel=1e-12)
        assert rho_s == pytest.approx(rho_i, rel=1e-12)
        assert a_s == pytest.approx(a_i, rel=1e-12)
        assert mu_s == pytest.approx(mu_i, rel=1e-12)


def test_site_elevation_equals_isa_shifted():
    """isa_at_site(0, elev, 0) must equal isa(elev) — pad is at MSL elevation."""
    for elev in [100.0, 500.0, 1000.0, 2000.0]:
        T_s, p_s, rho_s, a_s, mu_s = isa_at_site(0.0, elev, 0.0)
        T_i, p_i, rho_i, a_i, mu_i = isa(elev)
        assert T_s == pytest.approx(T_i, rel=1e-12)
        assert p_s == pytest.approx(p_i, rel=1e-12)
        assert rho_s == pytest.approx(rho_i, rel=1e-12)


def test_temperature_offset_applied():
    """Temperature offset should shift T without affecting pressure."""
    elev = 500.0
    t_offset = 10.0  # 10 K warmer than ISA
    T_s, p_s, *_ = isa_at_site(0.0, elev, t_offset)
    T_i = temperature(elev)
    p_i = pressure(elev)
    assert T_s == pytest.approx(T_i + t_offset, rel=1e-12)
    assert p_s == pytest.approx(p_i, rel=1e-12)


def test_compute_t_offset():
    """compute_t_offset should return T_user - T_ISA(elevation)."""
    elev = 1000.0
    T_user = 290.0  # K
    T_isa_at_elev = temperature(elev)
    expected = T_user - T_isa_at_elev
    assert compute_t_offset(elev, T_user) == pytest.approx(expected, rel=1e-12)


def test_compute_t_offset_standard_day():
    """At sea level with ISA temperature, offset is zero."""
    assert compute_t_offset(0.0, 288.15) == pytest.approx(0.0, abs=1e-10)


def test_pressure_at_site_helper():
    """pressure_at_site(h, elev) must equal pressure(h + elev)."""
    for h, elev in [(0.0, 500.0), (1000.0, 500.0), (5000.0, 0.0)]:
        assert pressure_at_site(h, elev) == pytest.approx(pressure(h + elev), rel=1e-12)


def test_temperature_at_site_helper():
    """temperature_at_site(h, elev, t_off) must equal temperature(h + elev) + t_off."""
    for h, elev, t_off in [(0.0, 500.0, 5.0), (1000.0, 0.0, -3.0)]:
        expected = temperature(h + elev) + t_off
        assert temperature_at_site(h, elev, t_off) == pytest.approx(expected, rel=1e-12)


def test_density_adjusted_for_temperature():
    """Warmer temperature at same pressure should give lower density."""
    elev = 500.0
    _, _, rho_std, _, _ = isa_at_site(0.0, elev, 0.0)
    _, _, rho_warm, _, _ = isa_at_site(0.0, elev, 10.0)
    assert rho_warm < rho_std
