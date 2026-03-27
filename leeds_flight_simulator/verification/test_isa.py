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
from leeds_flight_simulator.atmosphere import isa, temperature, pressure, density, speed_of_sound, dynamic_viscosity

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
