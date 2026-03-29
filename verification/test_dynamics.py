"""6DoF dynamics and integrator verification (§18.2, §18.3).

Tests:
- Gravity-only free fall (no thrust, no aero): body falls, position matches
- Integrator convergence: tighter tolerance → consistent results (§18.3)
- 6DoF derivative: gravity direction in body frame
- Apogee detection: vertical velocity sign change
"""

import math

import numpy as np
import pytest

from dynamics import (
    integrate_sixdof,
    _sixdof_deriv,
    quat_from_rail,
    quat_to_dcm_nb,
    simulate_rail,
)
from integrator import error_norm, optimal_step_factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_wind():
    alt = np.array([0.0, 50000.0], dtype=np.float64)
    east = np.zeros(2, dtype=np.float64)
    north = np.zeros(2, dtype=np.float64)
    return alt, east, north


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
        0.01, 0.01, 0.001, 0.001,  # inertias
        1.0,                   # impulse_factor
        mg, rg, ag, cat, cnt, cpt, cnc, cpc,
        False, cnaf,           # no components
        0.1, 1.0, 0.01, 0.05, # diameter, length, A_ref, fin_cp_radius
        wa, we, wn,
        0.0,                   # fin_cant
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
        prop_I_roll=0.001, prop_I_lateral=0.001,
        impulse_factor=1.0,
        mach_g=mg, re_g=rg, alpha_g=ag,
        ca_tbl=cat, cn_tbl=cnt, cp_tbl=cpt,
        cn_comp=cnc, cp_comp=cpc,
        has_components=False, cn_alpha_fins_tbl=cnaf,
        diameter=0.1, ref_length=1.0, A_ref=0.01, fin_cp_radius=0.05,
        wind_alt=wa, wind_east=we, wind_north=wn,
        fin_cant_rad=0.0,
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
        0.01, 0.01, 0.001, 0.001,
        1.0,
        mg, rg, ag, cat, cnt, cpt, cnc, cpc, False, cnaf,
        0.1, 1.0, 0.01, 0.05,
        wa, we, wn,
        0.0,
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
