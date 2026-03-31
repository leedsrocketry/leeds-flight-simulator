"""Dynamics module verification — 6DoF, frames, launch rail, and descent (§18.2, §18.3).

Tests:
- Quaternion / DCM / frame transforms: round-trip identity, known rotations,
  normalisation, quaternion rate consistency, DCM orthogonality
- Launch rail exit velocity: constant thrust analytical solutions
- 6DoF gravity-only free fall, integrator convergence, derivative checks
- Terminal descent: terminal velocity, free fall, landing, wind displacement
"""

import math

import numpy as np
import pytest

from dynamics import (
    integrate_sixdof,
    _sixdof_deriv,
    quat_from_rail,
    quat_to_dcm_nb,
    quat_rate,
    quat_normalize,
    simulate_rail,
    _rail_direction,
    integrate_descent,
    SCENARIO_DROGUE_ONLY,
    SCENARIO_BALLISTIC,
)
from integrator import error_norm, optimal_step_factor
from atmosphere import density


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _zero_wind():
    alt = np.array([0.0, 50000.0], dtype=np.float64)
    east = np.zeros(2, dtype=np.float64)
    north = np.zeros(2, dtype=np.float64)
    return alt, east, north


# ---------------------------------------------------------------------------
# 6DoF helpers
# ---------------------------------------------------------------------------

def _zero_aero():
    """Aero tables with C_A = C_N = 0 everywhere."""
    mach_g = np.array([0.0, 10.0], dtype=np.float64)
    re_g = np.array([0.0, 1e8], dtype=np.float64)
    alpha_g = np.array([0.0, 180.0], dtype=np.float64)
    z3 = np.zeros((2, 2, 2), dtype=np.float64)
    cn_comp = np.zeros((1, 2, 2, 2), dtype=np.float64)
    cp_comp = np.zeros((1, 2, 2, 2), dtype=np.float64)
    cna_fins = np.zeros((2, 2), dtype=np.float64)
    return mach_g, re_g, alpha_g, z3, z3, z3, cn_comp, cp_comp, cna_fins


def _dead_motor():
    """Motor arrays with zero thrust (coast-only)."""
    times = np.array([0.0, 0.001], dtype=np.float64)
    thrusts = np.array([0.0, 0.0], dtype=np.float64)
    m_prop_0 = 1e-6
    total_impulse = 1e-6
    return times, thrusts, m_prop_0, total_impulse


# ---------------------------------------------------------------------------
# Rail helpers
# ---------------------------------------------------------------------------

def _constant_thrust_motor(thrust_n: float, burn_time: float):
    """Return (times, thrusts) for a constant-thrust motor."""
    times = np.array([0.0, burn_time], dtype=np.float64)
    thrusts = np.array([thrust_n, thrust_n], dtype=np.float64)
    return times, thrusts


def _dummy_rail_aero():
    """Minimal aero tables that return C_A = 0 everywhere."""
    mach_g = np.array([0.0, 10.0], dtype=np.float64)
    re_g = np.array([0.0, 1e8], dtype=np.float64)
    alpha_g = np.array([0.0, 180.0], dtype=np.float64)
    ca_tbl = np.zeros((2, 2, 2), dtype=np.float64)
    return mach_g, re_g, alpha_g, ca_tbl


def _nonzero_rail_aero(ca_val: float):
    """Aero tables returning constant C_A everywhere."""
    mach_g = np.array([0.0, 10.0], dtype=np.float64)
    re_g = np.array([0.0, 1e8], dtype=np.float64)
    alpha_g = np.array([0.0, 180.0], dtype=np.float64)
    ca_tbl = np.full((2, 2, 2), ca_val, dtype=np.float64)
    return mach_g, re_g, alpha_g, ca_tbl


# ---------------------------------------------------------------------------
# Descent helpers
# ---------------------------------------------------------------------------


# ===========================================================================
# FRAMES — quaternion / DCM / rail direction
# ===========================================================================

# ---------------------------------------------------------------------------
# quat_from_rail → DCM → body x-axis should align with rail direction
# ---------------------------------------------------------------------------

_RAIL_CASES = [
    # (azimuth_deg, inclination_deg, description)
    (0.0, 90.0, "vertical, north"),
    (0.0, 85.0, "5° from vertical, north"),
    (90.0, 85.0, "5° from vertical, east"),
    (180.0, 80.0, "10° from vertical, south"),
    (270.0, 75.0, "15° from vertical, west"),
    (45.0, 85.0, "5° from vertical, NE"),
    (0.0, 45.0, "45° from horizontal, north"),
]


@pytest.mark.parametrize("az_deg, inc_deg, desc", _RAIL_CASES)
def test_body_x_aligns_with_rail(az_deg, inc_deg, desc):
    """Body x-axis (via C_nb · [1,0,0]) should equal the rail direction in NED."""
    az = math.radians(az_deg)
    inc = math.radians(inc_deg)

    q0, q1, q2, q3 = quat_from_rail(az, inc)
    C_nb = quat_to_dcm_nb(q0, q1, q2, q3)

    # Body x-axis in NED = first column of C_nb
    body_x_ned = np.array([C_nb[0, 0], C_nb[1, 0], C_nb[2, 0]])

    eN, eE, eD = _rail_direction(az, inc)
    rail_ned = np.array([eN, eE, eD])

    np.testing.assert_allclose(body_x_ned, rail_ned, atol=1e-12)


# ---------------------------------------------------------------------------
# DCM properties: orthogonal, det = +1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("az_deg, inc_deg, desc", _RAIL_CASES)
def test_dcm_orthogonal(az_deg, inc_deg, desc):
    """C_nb must be orthogonal: C_nb^T C_nb = I."""
    az = math.radians(az_deg)
    inc = math.radians(inc_deg)
    q0, q1, q2, q3 = quat_from_rail(az, inc)
    C = quat_to_dcm_nb(q0, q1, q2, q3)
    np.testing.assert_allclose(C.T @ C, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("az_deg, inc_deg, desc", _RAIL_CASES)
def test_dcm_determinant(az_deg, inc_deg, desc):
    """det(C_nb) must be +1 (proper rotation)."""
    az = math.radians(az_deg)
    inc = math.radians(inc_deg)
    q0, q1, q2, q3 = quat_from_rail(az, inc)
    C = quat_to_dcm_nb(q0, q1, q2, q3)
    assert np.linalg.det(C) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Quaternion normalisation
# ---------------------------------------------------------------------------

def test_quat_normalize_unit():
    """Already-unit quaternion should be unchanged."""
    q0, q1, q2, q3 = quat_normalize(1.0, 0.0, 0.0, 0.0)
    assert (q0, q1, q2, q3) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-15)


def test_quat_normalize_scaled():
    """Scaled quaternion should normalise to unit length."""
    q0, q1, q2, q3 = quat_normalize(2.0, 0.0, 0.0, 0.0)
    assert (q0, q1, q2, q3) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-15)

    # General case
    q0, q1, q2, q3 = quat_normalize(3.0, 4.0, 0.0, 0.0)
    norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    assert norm == pytest.approx(1.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Quaternion rate: dq/dt consistency
# ---------------------------------------------------------------------------

def test_quat_rate_zero_omega():
    """Zero angular velocity → zero quaternion rate."""
    dq0, dq1, dq2, dq3 = quat_rate(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert (dq0, dq1, dq2, dq3) == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-15)


def test_quat_rate_pure_roll():
    """Pure roll (p ≠ 0, q = r = 0) at identity quaternion.

    dq/dt = 0.5 * [-p*q1, p*q0, p*q3, -p*q2]
    At identity q = [1,0,0,0]: dq = [0, 0.5*p, 0, 0]
    """
    p = 2.0
    dq0, dq1, dq2, dq3 = quat_rate(1.0, 0.0, 0.0, 0.0, p, 0.0, 0.0)
    assert dq0 == pytest.approx(0.0, abs=1e-15)
    assert dq1 == pytest.approx(0.5 * p, abs=1e-15)
    assert dq2 == pytest.approx(0.0, abs=1e-15)
    assert dq3 == pytest.approx(0.0, abs=1e-15)


def test_quat_rate_preserves_norm():
    """q · dq/dt = 0 (quaternion rate is tangent to the unit sphere)."""
    q0, q1, q2, q3 = quat_from_rail(math.radians(45), math.radians(80))
    p, q, r = 0.5, -0.3, 0.1
    dq0, dq1, dq2, dq3 = quat_rate(q0, q1, q2, q3, p, q, r)
    dot = q0 * dq0 + q1 * dq1 + q2 * dq2 + q3 * dq3
    assert dot == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Rail direction vector sanity
# ---------------------------------------------------------------------------

def test_rail_direction_vertical():
    """90° inclination should point straight up (NED Down = -1)."""
    eN, eE, eD = _rail_direction(0.0, math.radians(90.0))
    assert eN == pytest.approx(0.0, abs=1e-12)
    assert eE == pytest.approx(0.0, abs=1e-12)
    assert eD == pytest.approx(-1.0, abs=1e-12)


def test_rail_direction_horizontal_north():
    """0° inclination, 0° azimuth should point North."""
    eN, eE, eD = _rail_direction(0.0, 0.0)
    assert eN == pytest.approx(1.0, abs=1e-12)
    assert eE == pytest.approx(0.0, abs=1e-12)
    assert eD == pytest.approx(0.0, abs=1e-12)


def test_rail_direction_unit_vector():
    """Rail direction must have unit length."""
    for az_deg in [0, 45, 90, 135, 270]:
        for inc_deg in [0, 30, 60, 85, 90]:
            eN, eE, eD = _rail_direction(
                math.radians(az_deg), math.radians(inc_deg),
            )
            length = math.sqrt(eN**2 + eE**2 + eD**2)
            assert length == pytest.approx(1.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Gravity in body frame via DCM
# ---------------------------------------------------------------------------

def test_gravity_body_frame_vertical_rail():
    """On a vertical rail (90° inc), gravity in body frame should be along -x.

    g_NED = [0, 0, g]. Body x points up (NED Down = -1).
    grav_body = C_bn @ g_NED = C_nb^T @ [0,0,g].
    For vertical rail, body x = -NED_D, so grav_body_x = -g.
    """
    q0, q1, q2, q3 = quat_from_rail(0.0, math.radians(90.0))
    C_nb = quat_to_dcm_nb(q0, q1, q2, q3)
    g = 9.80665
    g_ned = np.array([0.0, 0.0, g])
    grav_body = C_nb.T @ g_ned

    assert grav_body[0] == pytest.approx(-g, rel=1e-10)
    assert grav_body[1] == pytest.approx(0.0, abs=1e-10)
    assert grav_body[2] == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# LAUNCH RAIL
# ===========================================================================

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
    mach_g, re_g, alpha_g, ca_tbl = _dummy_rail_aero()

    # Horizontal rail (0° inclination) pointing north (0° azimuth)
    # → no gravity component along rail
    V_exit, t_exit, rN, rE, rD, _, _, _, _ = simulate_rail(
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
        site_elevation=0.0,
        t_offset=0.0,
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
    mach_g, re_g, alpha_g, ca_tbl = _dummy_rail_aero()

    V_exit, t_exit, rN, rE, rD, _, _, _, _ = simulate_rail(
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
        site_elevation=0.0,
        t_offset=0.0,
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
    mach_g, re_g, alpha_g, ca_tbl = _dummy_rail_aero()

    az = math.radians(az_deg)
    inc = math.radians(inc_deg)

    V_exit, t_exit, rN, rE, rD, _, _, _, _ = simulate_rail(
        az, inc, rail_length,
        times, thrusts, 0.0, 1.0,
        m_prop_0, total_impulse, m_dry,
        mach_g, re_g, alpha_g, ca_tbl,
        0.01, 1.0, 0.0, 0.0, 1e-9, 1e-9,
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
    mach_g, re_g, alpha_g, ca_tbl = _dummy_rail_aero()

    common = dict(
        rail_azimuth_rad=0.0,
        rail_inclination_rad=0.0,
        rail_length=rail_length,
        motor_times=times, motor_thrusts=thrusts,
        nozzle_area=0.0,
        m_prop_0=m_prop_0, total_impulse=total_impulse, m_dry=m_dry,
        mach_g=mach_g, re_g=re_g, alpha_g=alpha_g, ca_tbl=ca_tbl,
        A_ref=0.01, length=1.0,
        site_elevation=0.0, t_offset=0.0,
        rtol=1e-9, atol=1e-9,
    )

    V1, *_ = simulate_rail(impulse_factor=1.0, **common)
    V2, *_ = simulate_rail(impulse_factor=2.0, **common)

    assert V2 / V1 == pytest.approx(math.sqrt(2.0), rel=1e-3)


# ===========================================================================
# 6DoF DYNAMICS
# ===========================================================================

# ---------------------------------------------------------------------------
# Test: gravity-only 6DoF — vertical rail, no aero, no thrust
# ---------------------------------------------------------------------------

def test_sixdof_gravity_only_apogee():
    """Launch vertically with initial velocity, no thrust or aero.

    The vehicle should rise to h = V0²/(2g) then trigger apogee.
    """
    V0 = 50.0  # m/s
    g = 9.80665
    h_expected = V0 ** 2 / (2.0 * g)

    q0, q1, q2, q3 = quat_from_rail(0.0, math.radians(90.0))
    state0 = np.array([
        0.0, 0.0, 0.0,         # position at origin (ground)
        q0, q1, q2, q3,
        V0, 0.0, 0.0,          # body velocity along x (= up)
        0.0, 0.0, 0.0,         # no rotation
    ], dtype=np.float64)

    mt, mth, mp0, ti = _dead_motor()
    mg, rg, ag, cat, cnt, cpt, cnc, cpc, cnaf = _zero_aero()
    wa, we, wn = _zero_wind()

    t_out, y_out, n, *_ = integrate_sixdof(
        0.0, state0, 0.001,   # t_burnout in the past
        mt, mth, mp0, ti,
        0.0, 1.0,             # nozzle_area, nozzle_position
        10.0, 0.5, 0.5,       # m_dry, cg_dry, motor_cg_loaded
        0.01, 0.01, 0.01, 0.0, 0.1,  # I_roll_dry, I_lat_dry, prop_r_outer, prop_r_inner_0, prop_length
        1.0,                   # impulse_factor
        mg, rg, ag, cat, cnt, cpt, cnc, cpc,
        False, cnaf,           # no components
        0.1, 1.0, 0.01, 0.05, # diameter, length, A_ref, fin_cp_radius
        wa, we, wn,
        0.0,                   # fin_cant
        0.0, 0.0,             # site_elevation, t_offset
        1.0,                   # sm_transition_mach
        -1e6, -1e6,           # sm_min (permissive — no check)
        math.pi, math.pi,     # aoa_max, sm_aoa_threshold (permissive)
        1e-8, 1e-8,           # tight tolerances
    )

    # Apogee altitude (h = -rD)
    apogee_alt = -y_out[n - 1, 2]
    assert apogee_alt == pytest.approx(h_expected, rel=0.01)


# ---------------------------------------------------------------------------
# Test: integrator convergence (§18.3)
# ---------------------------------------------------------------------------

def test_integrator_convergence():
    """Tighter tolerance should give consistent results (< 0.5% change).

    Compare apogee altitude at rtol=1e-6 vs rtol=1e-9.
    """
    V0 = 100.0
    q0, q1, q2, q3 = quat_from_rail(0.0, math.radians(85.0))
    state0 = np.array([
        0.0, 0.0, 0.0,
        q0, q1, q2, q3,
        V0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], dtype=np.float64)

    mt, mth, mp0, ti = _dead_motor()
    mg, rg, ag, cat, cnt, cpt, cnc, cpc, cnaf = _zero_aero()
    wa, we, wn = _zero_wind()

    common = dict(
        t0=0.0, state0=state0, t_burnout=0.001,
        motor_times=mt, motor_thrusts=mth,
        m_prop_0=mp0, total_impulse=ti,
        nozzle_area=0.0, nozzle_position=1.0,
        m_dry=10.0, cg_dry=0.5, motor_cg_loaded=0.5,
        I_roll_dry=0.01, I_lateral_dry=0.01,
        prop_r_outer=0.01, prop_r_inner_0=0.0, prop_length=0.1,
        impulse_factor=1.0,
        mach_g=mg, re_g=rg, alpha_g=ag,
        ca_tbl=cat, cn_tbl=cnt, cp_tbl=cpt,
        cn_comp=cnc, cp_comp=cpc,
        has_components=False, cn_alpha_fins_tbl=cnaf,
        diameter=0.1, ref_length=1.0, A_ref=0.01, fin_cp_radius=0.05,
        wind_alt=wa, wind_east=we, wind_north=wn,
        fin_cant_rad=0.0,
        site_elevation=0.0, t_offset=0.0,
        sm_transition_mach=1.0,
        sm_subsonic_min=-1e6, sm_supersonic_min=-1e6,
        aoa_max_rad=math.pi, sm_aoa_threshold_rad=math.pi,
    )

    _, y_loose, n_loose, *_ = integrate_sixdof(rtol=1e-6, atol=1e-6, **common)
    _, y_tight, n_tight, *_ = integrate_sixdof(rtol=1e-9, atol=1e-9, **common)

    h_loose = -y_loose[n_loose - 1, 2]
    h_tight = -y_tight[n_tight - 1, 2]

    # < 0.5% difference
    assert abs(h_loose - h_tight) / h_tight < 0.005


# ---------------------------------------------------------------------------
# Test: error_norm helper
# ---------------------------------------------------------------------------

def test_error_norm_zero_error():
    """Zero error vector → zero norm."""
    y = np.array([1.0, 2.0, 3.0])
    y_new = np.array([1.1, 2.1, 3.1])
    y_err = np.array([0.0, 0.0, 0.0])
    assert error_norm(y, y_new, y_err, 1e-6, 1e-6) == pytest.approx(0.0, abs=1e-15)


def test_error_norm_scales_correctly():
    """Error norm should scale linearly with error magnitude."""
    y = np.ones(3)
    y_new = np.ones(3)
    y_err_small = np.array([1e-8, 1e-8, 1e-8])
    y_err_large = np.array([1e-4, 1e-4, 1e-4])

    n_small = error_norm(y, y_new, y_err_small, 1e-6, 1e-6)
    n_large = error_norm(y, y_new, y_err_large, 1e-6, 1e-6)

    # 10000× larger error should give ~10000× larger norm
    assert n_large / n_small == pytest.approx(1e4, rel=0.01)


# ---------------------------------------------------------------------------
# Test: optimal_step_factor
# ---------------------------------------------------------------------------

def test_step_factor_accepted():
    """err < 1 → factor > 1 (step can grow)."""
    f = optimal_step_factor(0.5)
    assert f > 1.0


def test_step_factor_rejected():
    """err > 1 → factor < 1 (step must shrink)."""
    f = optimal_step_factor(2.0)
    assert f < 1.0


def test_step_factor_clamped():
    """Factor clamped to [0.2, 5.0]."""
    assert optimal_step_factor(1e-20) == pytest.approx(5.0)
    assert optimal_step_factor(1e6) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Test: 6DoF derivative — gravity direction
# ---------------------------------------------------------------------------

def test_sixdof_deriv_gravity_direction():
    """For a vertical stationary vehicle with no thrust/aero, the body-frame
    acceleration should be purely along -x (gravity pulling down = -body_x)."""
    q0, q1, q2, q3 = quat_from_rail(0.0, math.radians(90.0))
    state = np.array([
        0.0, 0.0, -100.0,      # 100 m altitude
        q0, q1, q2, q3,
        0.0, 0.0, 0.0,         # stationary
        0.0, 0.0, 0.0,
    ], dtype=np.float64)

    dy = np.empty(13)
    mt, mth, mp0, ti = _dead_motor()
    mg, rg, ag, cat, cnt, cpt, cnc, cpc, cnaf = _zero_aero()
    wa, we, wn = _zero_wind()

    _sixdof_deriv(
        0.0, state, dy,
        mt, mth, mp0, ti,
        0.0, 1.0,
        10.0, 0.5, 0.5,
        0.01, 0.01, 0.01, 0.0, 0.1,  # I_roll_dry, I_lat_dry, prop_r_outer, prop_r_inner_0, prop_length
        1.0,
        mg, rg, ag, cat, cnt, cpt, cnc, cpc, False, cnaf,
        0.1, 1.0, 0.01, 0.05,
        wa, we, wn,
        0.0,
        0.0, 0.0,
    )

    g = 9.80665
    # du (index 7) should be -g (gravity along -body_x for vertical vehicle)
    assert dy[7] == pytest.approx(-g, rel=1e-4)
    # dv, dw (indices 8, 9) should be ~0
    assert dy[8] == pytest.approx(0.0, abs=1e-10)
    assert dy[9] == pytest.approx(0.0, abs=1e-10)
    # No rotation
    assert dy[10] == pytest.approx(0.0, abs=1e-10)
    assert dy[11] == pytest.approx(0.0, abs=1e-10)
    assert dy[12] == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# TERMINAL DESCENT
# ===========================================================================

# ---------------------------------------------------------------------------
# Test: terminal velocity under constant drag
# ---------------------------------------------------------------------------

def test_descent_time_matches_terminal_velocity():
    """Descent time should match h / V_terminal for constant-density drop.

    The dynamic descent model integrates vD, so when initialised at terminal
    velocity the descent time should still be h0 / V_terminal (within 2%).
    """
    m = 10.0         # kg
    g = 9.80665      # m/s²
    cda = 0.5        # m² — drogue CdA
    h0 = 500.0       # m — drop altitude (low enough for ~constant rho)
    rho_sl = density(0.0)

    V_terminal = math.sqrt(2.0 * m * g / (rho_sl * cda))
    expected_time = h0 / V_terminal

    # 4-component state: [rN, rE, rD, vD] — initialise at terminal velocity
    state0 = np.array([0.0, 0.0, -h0, V_terminal], dtype=np.float64)
    w_alt, w_east, w_north = _zero_wind()

    t_out, y_out, _, n = integrate_descent(
        0.0, state0,
        w_alt, w_east, w_north,
        m=m,
        drogue_cda=cda, main_cda=0.0, main_deploy_alt=0.0,
        scenario=SCENARIO_DROGUE_ONLY,
        site_elevation=0.0, t_offset=0.0,
        rtol=1e-8, atol=1e-8,
    )

    # Within 2% — density varies slightly over the 500 m drop
    assert t_out[n - 1] == pytest.approx(expected_time, rel=0.02)


# ---------------------------------------------------------------------------
# Test: landing at ground (rD >= 0)
# ---------------------------------------------------------------------------

def test_descent_lands_at_ground():
    """Vehicle must reach rD >= 0 (ground) by the last stored step."""
    m = 10.0
    h0 = 300.0
    cda = 0.3

    rho_sl = density(0.0)
    V_terminal = math.sqrt(2.0 * 10.0 * 9.80665 / (rho_sl * cda))
    # 4-component state: [rN, rE, rD, vD]
    state0 = np.array([0.0, 0.0, -h0, V_terminal], dtype=np.float64)
    w_alt, w_east, w_north = _zero_wind()

    t_out, y_out, _, n = integrate_descent(
        0.0, state0,
        w_alt, w_east, w_north,
        m=m,
        drogue_cda=cda, main_cda=0.0, main_deploy_alt=0.0,
        scenario=SCENARIO_DROGUE_ONLY,
        site_elevation=0.0, t_offset=0.0,
        rtol=1e-6, atol=1e-6,
    )

    assert y_out[n - 1, 2] >= 0.0


# ---------------------------------------------------------------------------
# Test: wind displaces landing point
# ---------------------------------------------------------------------------

def test_descent_wind_displacement():
    """Constant east wind should displace the landing point eastward.

    With the simplified descent model the vehicle drifts at exactly the
    wind speed, so the east displacement should be wind_speed * descent_time.
    """
    m = 10.0
    h0 = 300.0
    cda = 0.5
    wind_speed = 10.0  # m/s eastward

    rho_sl = density(0.0)
    V_terminal = math.sqrt(2.0 * m * 9.80665 / (rho_sl * cda))
    # 4-component state: [rN, rE, rD, vD]
    state0 = np.array([0.0, 0.0, -h0, V_terminal], dtype=np.float64)

    # Constant east wind
    w_alt = np.array([0.0, 50000.0], dtype=np.float64)
    w_east = np.array([wind_speed, wind_speed], dtype=np.float64)
    w_north = np.zeros(2, dtype=np.float64)

    # With wind
    t_out, y_wind, _, n_wind = integrate_descent(
        0.0, state0.copy(),
        w_alt, w_east, w_north,
        m, cda, 0.0, 0.0,
        SCENARIO_DROGUE_ONLY, 0.0, 0.0, 1e-6, 1e-6,
    )

    descent_time = t_out[n_wind - 1]
    east_landing = y_wind[n_wind - 1, 1]
    expected_east = wind_speed * descent_time

    # Vehicle drifts at exactly wind speed — should match closely
    assert east_landing == pytest.approx(expected_east, rel=0.02)
