"""Terminal descent verification (§18.2).

Tests:
- Terminal velocity: steady-state V_t = sqrt(2mg / (rho * CdA))
- No-drag free fall: h(t) = h0 + 0.5*g*t² (NED convention: rD increasing)
- Landing at ground: rD reaches 0
"""

import math

import numpy as np
import pytest

from dynamics import integrate_descent, SCENARIO_DROGUE_ONLY, SCENARIO_BALLISTIC
from atmosphere import density


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_wind():
    """Wind arrays with zero wind at all altitudes."""
    alt = np.array([0.0, 50000.0], dtype=np.float64)
    east = np.zeros(2, dtype=np.float64)
    north = np.zeros(2, dtype=np.float64)
    return alt, east, north


def _dummy_aero():
    """Minimal aero tables (only used for ballistic scenario)."""
    mach_g = np.array([0.0, 10.0], dtype=np.float64)
    re_g = np.array([0.0, 1e8], dtype=np.float64)
    alpha_g = np.array([0.0, 180.0], dtype=np.float64)
    ca_tbl = np.ones((2, 2, 2), dtype=np.float64)  # C_A = 1.0
    return mach_g, re_g, alpha_g, ca_tbl


# ---------------------------------------------------------------------------
# Test: terminal velocity under constant drag
# ---------------------------------------------------------------------------

def test_terminal_velocity_drogue():
    """Vehicle dropped from high altitude converges to terminal velocity.

    V_terminal = sqrt(2 * m * g / (rho * CdA))

    Use sea-level density for a short drop (altitude effect is small).
    """
    m = 10.0         # kg
    g = 9.80665      # m/s²
    cda = 0.5        # m² — drogue CdA
    h0 = 500.0       # m — drop altitude (low enough for ~constant rho)
    rho_sl = density(0.0)

    V_terminal = math.sqrt(2.0 * m * g / (rho_sl * cda))

    # Start from rest at h0
    state0 = np.array([0.0, 0.0, -h0, 0.0, 0.0, 0.0], dtype=np.float64)
    w_alt, w_east, w_north = _zero_wind()
    mach_g, re_g, alpha_g, ca_tbl = _dummy_aero()

    t_out, y_out, n = integrate_descent(
        0.0, state0,
        w_alt, w_east, w_north,
        mach_g, re_g, alpha_g, ca_tbl,
        A_ref=0.01, ref_length=1.0,
        m=m,
        drogue_cda=cda, main_cda=0.0, main_deploy_alt=0.0,
        scenario=SCENARIO_DROGUE_ONLY,
        rtol=1e-8, atol=1e-8,
    )

    # Final downward velocity (NED: vD > 0 means descending)
    vD_final = y_out[n - 1, 5]

    # Should be close to terminal velocity (within 2% — density varies slightly
    # over the 500 m drop)
    assert vD_final == pytest.approx(V_terminal, rel=0.02)


# ---------------------------------------------------------------------------
# Test: free fall (no drag)
# ---------------------------------------------------------------------------

def test_free_fall_no_drag():
    """With zero CdA, vehicle should free-fall: vD = g*t, rD = rD0 + 0.5*g*t².

    We use ballistic scenario with C_A = 0 everywhere.
    """
    m = 10.0
    g = 9.80665
    h0 = 200.0

    state0 = np.array([0.0, 0.0, -h0, 0.0, 0.0, 0.0], dtype=np.float64)
    w_alt, w_east, w_north = _zero_wind()

    # Zero C_A aero tables
    mach_g = np.array([0.0, 10.0], dtype=np.float64)
    re_g = np.array([0.0, 1e8], dtype=np.float64)
    alpha_g = np.array([0.0, 180.0], dtype=np.float64)
    ca_tbl = np.zeros((2, 2, 2), dtype=np.float64)

    t_out, y_out, n = integrate_descent(
        0.0, state0,
        w_alt, w_east, w_north,
        mach_g, re_g, alpha_g, ca_tbl,
        A_ref=0.01, ref_length=1.0,
        m=m,
        drogue_cda=0.0, main_cda=0.0, main_deploy_alt=0.0,
        scenario=SCENARIO_BALLISTIC,
        rtol=1e-9, atol=1e-9,
    )

    # Check a few interior points
    for i in range(1, min(n, 10)):
        t = t_out[i]
        rD_expected = -h0 + 0.5 * g * t * t
        vD_expected = g * t
        assert y_out[i, 2] == pytest.approx(rD_expected, rel=1e-4)
        assert y_out[i, 5] == pytest.approx(vD_expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Test: landing at ground (rD >= 0)
# ---------------------------------------------------------------------------

def test_descent_lands_at_ground():
    """Vehicle must reach rD >= 0 (ground) by the last stored step."""
    m = 10.0
    h0 = 300.0
    cda = 0.3

    state0 = np.array([0.0, 0.0, -h0, 0.0, 0.0, 0.0], dtype=np.float64)
    w_alt, w_east, w_north = _zero_wind()
    mach_g, re_g, alpha_g, ca_tbl = _dummy_aero()

    t_out, y_out, n = integrate_descent(
        0.0, state0,
        w_alt, w_east, w_north,
        mach_g, re_g, alpha_g, ca_tbl,
        A_ref=0.01, ref_length=1.0,
        m=m,
        drogue_cda=cda, main_cda=0.0, main_deploy_alt=0.0,
        scenario=SCENARIO_DROGUE_ONLY,
        rtol=1e-6, atol=1e-6,
    )

    assert y_out[n - 1, 2] >= 0.0


# ---------------------------------------------------------------------------
# Test: wind displaces landing point
# ---------------------------------------------------------------------------

def test_descent_wind_displacement():
    """Constant east wind should displace the landing point eastward."""
    m = 10.0
    h0 = 300.0
    cda = 0.5
    wind_speed = 10.0  # m/s eastward

    state0 = np.array([0.0, 0.0, -h0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Constant east wind
    w_alt = np.array([0.0, 50000.0], dtype=np.float64)
    w_east = np.array([wind_speed, wind_speed], dtype=np.float64)
    w_north = np.zeros(2, dtype=np.float64)
    mach_g, re_g, alpha_g, ca_tbl = _dummy_aero()

    # With wind
    _, y_wind, n_wind = integrate_descent(
        0.0, state0.copy(),
        w_alt, w_east, w_north,
        mach_g, re_g, alpha_g, ca_tbl,
        0.01, 1.0, m, cda, 0.0, 0.0,
        SCENARIO_DROGUE_ONLY, 1e-6, 1e-6,
    )

    # Without wind
    w_east_zero = np.zeros(2, dtype=np.float64)
    _, y_nowind, n_nowind = integrate_descent(
        0.0, state0.copy(),
        w_alt, w_east_zero, w_north,
        mach_g, re_g, alpha_g, ca_tbl,
        0.01, 1.0, m, cda, 0.0, 0.0,
        SCENARIO_DROGUE_ONLY, 1e-6, 1e-6,
    )

    # With east wind, landing East position (index 1) should be larger
    east_with = y_wind[n_wind - 1, 1]
    east_without = y_nowind[n_nowind - 1, 1]
    assert east_with > east_without + 1.0  # at least 1 m displacement
