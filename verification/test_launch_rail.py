"""Launch rail exit velocity verification (§18.2).

Compares simulate_rail output against analytical solutions for simple cases:
- Constant thrust, no drag, no gravity → V = sqrt(2aL)
- Constant thrust with gravity → energy balance
- Rail direction and exit position consistency
"""

import math

import numpy as np
import pytest

from dynamics import simulate_rail, _rail_direction


# ---------------------------------------------------------------------------
# Helpers: build minimal motor/aero arrays for rail simulation
# ---------------------------------------------------------------------------

def _constant_thrust_motor(thrust_n: float, burn_time: float):
    """Return (times, thrusts) for a constant-thrust motor."""
    times = np.array([0.0, burn_time], dtype=np.float64)
    thrusts = np.array([thrust_n, thrust_n], dtype=np.float64)
    return times, thrusts


def _dummy_aero_tables():
    """Minimal aero tables that return C_A = 0 everywhere.

    Single grid point: Mach=0, Re=0, AoA=0 with C_A=0.
    """
    mach_g = np.array([0.0, 10.0], dtype=np.float64)
    re_g = np.array([0.0, 1e8], dtype=np.float64)
    alpha_g = np.array([0.0, 180.0], dtype=np.float64)
    ca_tbl = np.zeros((2, 2, 2), dtype=np.float64)
    return mach_g, re_g, alpha_g, ca_tbl


def _nonzero_aero_tables(ca_val: float):
    """Aero tables returning constant C_A everywhere."""
    mach_g = np.array([0.0, 10.0], dtype=np.float64)
    re_g = np.array([0.0, 1e8], dtype=np.float64)
    alpha_g = np.array([0.0, 180.0], dtype=np.float64)
    ca_tbl = np.full((2, 2, 2), ca_val, dtype=np.float64)
    return mach_g, re_g, alpha_g, ca_tbl


# ---------------------------------------------------------------------------
# Test: constant thrust, no drag, no gravity (horizontal rail)
# ---------------------------------------------------------------------------

def test_rail_no_drag_no_gravity():
    """Horizontal rail, constant thrust, zero drag.

    Analytical: a = F/m (constant mass since we set m_prop_0 ~ 0).
    V_exit = sqrt(2 * a * L).
    """
    F = 1000.0       # N
    m_dry = 10.0      # kg
    m_prop_0 = 1e-6   # effectively zero propellant
    rail_length = 3.0  # m
    burn_time = 5.0    # s (longer than rail time)

    times, thrusts = _constant_thrust_motor(F, burn_time)
    total_impulse = float(np.trapz(thrusts, times))
    mach_g, re_g, alpha_g, ca_tbl = _dummy_aero_tables()

    # Horizontal rail (0° inclination) pointing north (0° azimuth)
    # → no gravity component along rail
    V_exit, t_exit, rN, rE, rD = simulate_rail(
        rail_azimuth_rad=0.0,
        rail_inclination_rad=0.0,
        rail_length=rail_length,
        motor_times=times,
        motor_thrusts=thrusts,
        nozzle_area=0.0,          # no altitude correction
        impulse_factor=1.0,
        m_prop_0=m_prop_0,
        total_impulse=total_impulse,
        m_dry=m_dry,
        mach_g=mach_g,
        re_g=re_g,
        alpha_g=alpha_g,
        ca_tbl=ca_tbl,
        A_ref=0.01,               # irrelevant with zero C_A
        length=1.0,
        rtol=1e-9,
        atol=1e-9,
    )

    a = F / m_dry
    V_analytical = math.sqrt(2.0 * a * rail_length)
    t_analytical = math.sqrt(2.0 * rail_length / a)

    assert V_exit == pytest.approx(V_analytical, rel=1e-4)
    assert t_exit == pytest.approx(t_analytical, rel=1e-4)


# ---------------------------------------------------------------------------
# Test: vertical rail with gravity, no drag
# ---------------------------------------------------------------------------

def test_rail_vertical_with_gravity():
    """Vertical rail, constant thrust, zero drag.

    Analytical: a_net = F/m - g.  V_exit = sqrt(2 * a_net * L).
    """
    F = 500.0
    m_dry = 10.0
    m_prop_0 = 1e-6
    rail_length = 3.0
    burn_time = 5.0
    g = 9.80665

    times, thrusts = _constant_thrust_motor(F, burn_time)
    total_impulse = float(np.trapz(thrusts, times))
    mach_g, re_g, alpha_g, ca_tbl = _dummy_aero_tables()

    V_exit, t_exit, rN, rE, rD = simulate_rail(
        rail_azimuth_rad=0.0,
        rail_inclination_rad=math.radians(90.0),
        rail_length=rail_length,
        motor_times=times,
        motor_thrusts=thrusts,
        nozzle_area=0.0,
        impulse_factor=1.0,
        m_prop_0=m_prop_0,
        total_impulse=total_impulse,
        m_dry=m_dry,
        mach_g=mach_g,
        re_g=re_g,
        alpha_g=alpha_g,
        ca_tbl=ca_tbl,
        A_ref=0.01,
        length=1.0,
        rtol=1e-9,
        atol=1e-9,
    )

    a_net = F / m_dry - g
    V_analytical = math.sqrt(2.0 * a_net * rail_length)

    assert V_exit == pytest.approx(V_analytical, rel=1e-3)


# ---------------------------------------------------------------------------
# Test: exit position matches rail direction × distance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("az_deg, inc_deg", [
    (0.0, 85.0),
    (90.0, 80.0),
    (45.0, 70.0),
])
def test_rail_exit_position_direction(az_deg, inc_deg):
    """Exit NED position should be approximately rail_length × e_rail."""
    F = 2000.0
    m_dry = 10.0
    m_prop_0 = 1e-6
    rail_length = 2.0
    burn_time = 5.0

    times, thrusts = _constant_thrust_motor(F, burn_time)
    total_impulse = float(np.trapz(thrusts, times))
    mach_g, re_g, alpha_g, ca_tbl = _dummy_aero_tables()

    az = math.radians(az_deg)
    inc = math.radians(inc_deg)

    V_exit, t_exit, rN, rE, rD = simulate_rail(
        az, inc, rail_length,
        times, thrusts, 0.0, 1.0,
        m_prop_0, total_impulse, m_dry,
        mach_g, re_g, alpha_g, ca_tbl,
        0.01, 1.0, 1e-9, 1e-9,
    )

    eN, eE, eD = _rail_direction(az, inc)
    # Position should be along rail direction
    pos = np.array([rN, rE, rD])
    expected = np.array([eN, eE, eD]) * rail_length
    # Allow some overshoot since integration may step slightly past rail_length
    np.testing.assert_allclose(pos / rail_length, expected / rail_length, atol=0.05)


# ---------------------------------------------------------------------------
# Test: impulse factor scaling
# ---------------------------------------------------------------------------

def test_rail_impulse_factor():
    """Doubling impulse factor should increase exit velocity.

    With constant mass (no propellant): V ∝ sqrt(F), so
    V(k=2) / V(k=1) = sqrt(2).
    """
    F = 1000.0
    m_dry = 10.0
    m_prop_0 = 1e-6
    rail_length = 3.0
    burn_time = 5.0

    times, thrusts = _constant_thrust_motor(F, burn_time)
    total_impulse = float(np.trapz(thrusts, times))
    mach_g, re_g, alpha_g, ca_tbl = _dummy_aero_tables()

    common = dict(
        rail_azimuth_rad=0.0,
        rail_inclination_rad=0.0,
        rail_length=rail_length,
        motor_times=times, motor_thrusts=thrusts,
        nozzle_area=0.0,
        m_prop_0=m_prop_0, total_impulse=total_impulse, m_dry=m_dry,
        mach_g=mach_g, re_g=re_g, alpha_g=alpha_g, ca_tbl=ca_tbl,
        A_ref=0.01, length=1.0, rtol=1e-9, atol=1e-9,
    )

    V1, *_ = simulate_rail(impulse_factor=1.0, **common)
    V2, *_ = simulate_rail(impulse_factor=2.0, **common)

    assert V2 / V1 == pytest.approx(math.sqrt(2.0), rel=1e-3)
