"""Quaternion / DCM / frame transform verification (§18.2).

Tests:
- Round-trip identity: quat → DCM → known vector → back
- Known rotations: rail orientations produce correct NED directions
- Quaternion normalisation
- Quaternion rate (dq/dt) consistency
- DCM orthogonality and determinant
"""

import math

import numpy as np
import pytest

from dynamics import (
    quat_from_rail,
    quat_to_dcm_nb,
    quat_rate,
    quat_normalize,
    _rail_direction,
)


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
