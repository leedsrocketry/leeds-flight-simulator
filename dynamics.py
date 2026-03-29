"""6DoF, 3DoF, and launch-rail dynamics with adaptive Dormand-Prince integration.

Implements the equations of motion from specification sections 8.2–8.5:
    Phase 1 — Launch rail (constrained 1-D translation)
    Phase 2 — Free flight 6DoF (full translation + rotation)
    Phase 3 — Descent 3DoF (point-mass under drag + gravity)
    3DoF ascent — Simplified point-mass for optimisation (section 8.5)

All hot-loop functions are Numba ``@njit`` compiled with ``fastmath=True``.
The integration loops use the Dormand-Prince RK4(5) tableau from
``integrator.py``.

Performance notes
-----------------
- Derivative functions write into caller-supplied output buffers — no heap
  allocation per derivative evaluation.
- The DCM is computed as 9 scalar locals, not a 3×3 array — avoids
  allocation and array-indexing overhead on every call.
- The 6DoF derivative returns auxiliary diagnostics (alpha, CP, Mach) so
  the integration loop does not need a separate acceptance-check pass.
- FSAL is implemented via pointer swap (``k1, k7 = k7, k1``), no copy.

Public API
----------
Data structures (plain Python):
    SimParams       — all pre-processed data for one trajectory
    TrajectoryResult — output of a single trajectory

@njit quaternion/frame utilities:
    quat_from_rail, quat_to_dcm_nb, quat_rate, quat_normalize

@njit phase runners:
    simulate_rail         — Phase 1
    integrate_sixdof      — Phase 2
    integrate_descent     — Phase 3
    simulate_ascent_3dof  — 3DoF ascent (optimisation)

Top-level entry point:
    run_trajectory(params, scenario) → TrajectoryResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numba as nb

from atmosphere import isa
from wind import interpolate_wind
from aerodynamics import (
    aero_forces_moments,
    ca_at,
    cn_alpha_fins_at,
    cn_cp_at,
    _interp3,
)
from motor import (
    thrust_corrected_at,
    mdot_at,
    mass_at,
    cg_at,
    inertia_at,
    MotorModel,
)
from aerodynamics import AeroModel
from integrator import (
    DP_C, DP_A1, DP_A2, DP_A3, DP_A4, DP_A5, DP_B, DP_E,
    error_norm, optimal_step_factor, clamp_step,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_G0: float = 9.80665
_EPS_V: float = 1.0e-6   # minimum airspeed to avoid division by zero
_RAD2DEG: float = 180.0 / 3.141592653589793


# ---------------------------------------------------------------------------
# Descent scenario encoding (Numba cannot handle strings)
# ---------------------------------------------------------------------------
SCENARIO_NOMINAL: int = 0
SCENARIO_BALLISTIC: int = 1
SCENARIO_DROGUE_ONLY: int = 2
SCENARIO_PREMATURE_MAIN: int = 3

SCENARIO_MAP: dict[str, int] = {
    "nominal": SCENARIO_NOMINAL,
    "ballistic": SCENARIO_BALLISTIC,
    "drogue_only": SCENARIO_DROGUE_ONLY,
    "premature_main": SCENARIO_PREMATURE_MAIN,
}


# ---------------------------------------------------------------------------
# Data structures (plain Python — not Numba-visible)
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """All pre-processed data needed to run a single trajectory.

    Construct in ``montecarlo.py`` from ``MotorModel``, ``AeroModel``,
    ``WindEnsemble``, and the simulation/vehicle configs.
    """
    # Motor (from MotorModel)
    motor_times: np.ndarray       # (K,) float64
    motor_thrusts: np.ndarray     # (K,) float64
    m_prop_0: float
    total_impulse: float
    nozzle_area: float
    nozzle_position: float
    m_dry: float
    cg_dry: float
    motor_cg_loaded: float
    I_roll_dry: float
    I_lateral_dry: float
    prop_I_roll: float
    prop_I_lateral: float

    # Aero (from AeroModel)
    mach_g: np.ndarray
    re_g: np.ndarray
    alpha_g: np.ndarray
    ca_tbl: np.ndarray
    cn_tbl: np.ndarray
    cp_tbl: np.ndarray
    cn_comp: np.ndarray
    cp_comp: np.ndarray
    has_components: bool
    cn_alpha_fins: np.ndarray

    # Geometry
    diameter: float
    length: float
    A_ref: float
    fin_cp_radius: float

    # Wind (single profile for this sample)
    wind_alt: np.ndarray          # (M,) float64
    wind_east: np.ndarray         # (M,) float64
    wind_north: np.ndarray        # (M,) float64

    # Rail
    rail_azimuth_rad: float
    rail_inclination_rad: float
    rail_length: float

    # Stochastic draws for this sample
    fin_cant_rad: float
    impulse_factor: float

    # Recovery (CdA values; 0.0 if not configured)
    drogue_cda: float
    main_cda: float
    main_deploy_alt: float        # m AGL; negative sentinel if deploy-at-apogee
    has_drogue: bool
    has_main: bool

    # Acceptance thresholds
    sm_transition_mach: float
    sm_subsonic_min: float        # calibres
    sm_supersonic_min: float      # calibres
    aoa_max_rad: float
    sm_aoa_threshold_rad: float

    # Integration tolerances
    rtol: float = 1.0e-6
    atol: float = 1.0e-6


@dataclass
class TrajectoryResult:
    """Output of a single trajectory simulation."""
    apogee_altitude: float        # m AGL
    apogee_time: float            # s
    apogee_position: np.ndarray   # [N, E, D] NED [m]
    landing_position: np.ndarray  # [N, E, D] NED [m]
    landing_time: float           # s
    flight_time: float            # s  (total rail-exit to landing)
    max_mach: float
    max_aoa_deg: float
    min_sm_subsonic: float        # calibres (worst case)
    min_sm_supersonic: float      # calibres (worst case)
    rail_exit_velocity: float     # m/s
    peak_altitude_ft: float       # feet
    in_buffer: bool               # full trajectory inside buffered footprint
    below_ceiling: bool           # apogee below buffered altitude ceiling
    compliant: bool
    stability_compliant: bool
    violation_reason: str         # empty string if compliant
    # Time history (ascent only, for replay/verification)
    t_ascent: np.ndarray
    state_ascent: np.ndarray      # (N, 13) 6DoF state history


# ---------------------------------------------------------------------------
# Quaternion / frame utilities
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def quat_from_rail(azimuth_rad: float, inclination_rad: float
                   ) -> tuple[float, float, float, float]:
    """Quaternion (scalar-first) placing body x-axis along the launch rail.

    The rail direction in NED is::

        e_rail = [cos(inc)*cos(az), cos(inc)*sin(az), -sin(inc)]

    This is a yaw of *azimuth* about Down, then a pitch of *inclination*
    about the new East axis.  The resulting quaternion maps body → NED.
    """
    ha = azimuth_rad * 0.5
    hi = inclination_rad * 0.5
    cy, sy = math.cos(ha), math.sin(ha)
    ci, si = math.cos(hi), math.sin(hi)

    # Hamilton product: q_yaw ⊗ q_pitch
    # q_yaw  = (cy, 0, 0, sy)   — rotate about NED Down by azimuth
    # q_pitch = (ci, 0, si, 0)  — rotate about NED East by inclination (nose up)
    a0, a1, a2, a3 = cy, 0.0, 0.0, sy
    b0, b1, b2, b3 = ci, 0.0, si, 0.0

    q0 = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    q1 = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
    q2 = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
    q3 = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0

    n = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) ** 0.5
    return q0 / n, q1 / n, q2 / n, q3 / n


@nb.njit(cache=True, fastmath=True)
def quat_to_dcm_nb(q0: float, q1: float, q2: float, q3: float
                    ) -> np.ndarray:
    """Body-to-NED direction cosine matrix C_nb (3×3).

    Specification section 3.4.  Use only where an array is truly needed
    (e.g. for the ``run_trajectory`` NED velocity transform).  Inside the
    hot-loop derivative, use the inlined scalar version instead.
    """
    C = np.empty((3, 3))
    q0q0 = q0 * q0; q1q1 = q1 * q1; q2q2 = q2 * q2; q3q3 = q3 * q3
    q0q1 = q0 * q1; q0q2 = q0 * q2; q0q3 = q0 * q3
    q1q2 = q1 * q2; q1q3 = q1 * q3; q2q3 = q2 * q3

    C[0, 0] = 1.0 - 2.0 * (q2q2 + q3q3)
    C[0, 1] = 2.0 * (q1q2 - q0q3)
    C[0, 2] = 2.0 * (q1q3 + q0q2)
    C[1, 0] = 2.0 * (q1q2 + q0q3)
    C[1, 1] = 1.0 - 2.0 * (q1q1 + q3q3)
    C[1, 2] = 2.0 * (q2q3 - q0q1)
    C[2, 0] = 2.0 * (q1q3 - q0q2)
    C[2, 1] = 2.0 * (q2q3 + q0q1)
    C[2, 2] = 1.0 - 2.0 * (q1q1 + q2q2)
    return C


@nb.njit(cache=True, fastmath=True)
def quat_rate(q0: float, q1: float, q2: float, q3: float,
              p: float, q: float, r: float,
              ) -> tuple[float, float, float, float]:
    """Quaternion time derivative: dq/dt = 0.5 * Ω(ω) · q."""
    dq0 = 0.5 * (-p * q1 - q * q2 - r * q3)
    dq1 = 0.5 * (p * q0 + r * q2 - q * q3)
    dq2 = 0.5 * (q * q0 - r * q1 + p * q3)
    dq3 = 0.5 * (r * q0 + q * q1 - p * q2)
    return dq0, dq1, dq2, dq3


@nb.njit(cache=True, fastmath=True)
def quat_normalize(q0: float, q1: float, q2: float, q3: float
                   ) -> tuple[float, float, float, float]:
    """Renormalise a quaternion to unit length."""
    n = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) ** 0.5
    if n < 1.0e-15:
        return 1.0, 0.0, 0.0, 0.0
    return q0 / n, q1 / n, q2 / n, q3 / n


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _rail_direction(azimuth_rad: float, inclination_rad: float
                    ) -> tuple[float, float, float]:
    """Unit rail vector in NED: [cosθ cosψ, cosθ sinψ, −sinθ]."""
    ct = math.cos(inclination_rad)
    st = math.sin(inclination_rad)
    ca = math.cos(azimuth_rad)
    sa = math.sin(azimuth_rad)
    return ct * ca, ct * sa, -st


# ---------------------------------------------------------------------------
# Phase 1: Launch Rail
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _rail_deriv(
    t: float, s: float, V: float,
    sin_theta: float,
    motor_times: np.ndarray, motor_thrusts: np.ndarray,
    nozzle_area: float, altitude: float,
    impulse_factor: float,
    m_prop_0: float, total_impulse: float, m_dry: float,
    mach_g: np.ndarray, re_g: np.ndarray, alpha_g: np.ndarray,
    ca_tbl: np.ndarray,
    A_ref: float, length: float,
) -> tuple[float, float]:
    """Derivatives for launch rail: ds/dt = V, dV/dt = F_along / m."""
    _, _, rho, a_sound, mu = isa(altitude)

    F_thrust = thrust_corrected_at(
        motor_times, motor_thrusts, nozzle_area, altitude, t
    ) * impulse_factor

    m = mass_at(motor_times, motor_thrusts, m_prop_0, total_impulse, m_dry, t)

    V_abs = abs(V)
    if V_abs > _EPS_V:
        M = V_abs / a_sound
        Re = rho * V_abs * length / mu
        C_A = ca_at(mach_g, re_g, alpha_g, ca_tbl, M, Re, 0.0)
        F_drag = 0.5 * rho * V_abs * V_abs * A_ref * C_A
    else:
        F_drag = 0.0

    F_grav = m * _G0 * sin_theta
    a = (F_thrust - F_drag - F_grav) / m
    return V, a


@nb.njit(cache=True, fastmath=True)
def simulate_rail(
    rail_azimuth_rad: float,
    rail_inclination_rad: float,
    rail_length: float,
    motor_times: np.ndarray,
    motor_thrusts: np.ndarray,
    nozzle_area: float,
    impulse_factor: float,
    m_prop_0: float,
    total_impulse: float,
    m_dry: float,
    mach_g: np.ndarray,
    re_g: np.ndarray,
    alpha_g: np.ndarray,
    ca_tbl: np.ndarray,
    A_ref: float,
    length: float,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, float, float]:
    """Integrate launch-rail phase until CG travels ``rail_length``.

    Returns
    -------
    (V_exit, t_exit, rN_exit, rE_exit, rD_exit)
    """
    sin_theta = math.sin(rail_inclination_rad)
    eN, eE, eD = _rail_direction(rail_azimuth_rad, rail_inclination_rad)

    altitude = 0.0
    s = 0.0
    V = 0.0
    t = 0.0
    h = 1.0e-3
    h_min = 1.0e-4
    h_max = 0.05

    max_steps = 50000
    y = np.empty(2)
    y_new = np.empty(2)
    y_err = np.empty(2)
    k1 = np.empty(2)
    k2 = np.empty(2)
    k3 = np.empty(2)
    k4 = np.empty(2)
    k5 = np.empty(2)
    k6 = np.empty(2)
    k7 = np.empty(2)
    ys = np.empty(2)

    y[0] = s; y[1] = V

    ds, dV = _rail_deriv(
        t, y[0], y[1], sin_theta,
        motor_times, motor_thrusts, nozzle_area, altitude,
        impulse_factor, m_prop_0, total_impulse, m_dry,
        mach_g, re_g, alpha_g, ca_tbl, A_ref, length,
    )
    k1[0] = ds; k1[1] = dV

    for _ in range(max_steps):
        altitude = -y[0] * eD

        # Stage 2
        for j in range(2):
            ys[j] = y[j] + h * DP_A1[0] * k1[j]
        ds, dV = _rail_deriv(
            t + DP_C[1] * h, ys[0], ys[1], sin_theta,
            motor_times, motor_thrusts, nozzle_area, altitude,
            impulse_factor, m_prop_0, total_impulse, m_dry,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, length,
        )
        k2[0] = ds; k2[1] = dV

        # Stage 3
        for j in range(2):
            ys[j] = y[j] + h * (DP_A2[0] * k1[j] + DP_A2[1] * k2[j])
        ds, dV = _rail_deriv(
            t + DP_C[2] * h, ys[0], ys[1], sin_theta,
            motor_times, motor_thrusts, nozzle_area, altitude,
            impulse_factor, m_prop_0, total_impulse, m_dry,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, length,
        )
        k3[0] = ds; k3[1] = dV

        # Stage 4
        for j in range(2):
            ys[j] = y[j] + h * (DP_A3[0] * k1[j] + DP_A3[1] * k2[j]
                                 + DP_A3[2] * k3[j])
        ds, dV = _rail_deriv(
            t + DP_C[3] * h, ys[0], ys[1], sin_theta,
            motor_times, motor_thrusts, nozzle_area, altitude,
            impulse_factor, m_prop_0, total_impulse, m_dry,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, length,
        )
        k4[0] = ds; k4[1] = dV

        # Stage 5
        for j in range(2):
            ys[j] = y[j] + h * (DP_A4[0] * k1[j] + DP_A4[1] * k2[j]
                                 + DP_A4[2] * k3[j] + DP_A4[3] * k4[j])
        ds, dV = _rail_deriv(
            t + DP_C[4] * h, ys[0], ys[1], sin_theta,
            motor_times, motor_thrusts, nozzle_area, altitude,
            impulse_factor, m_prop_0, total_impulse, m_dry,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, length,
        )
        k5[0] = ds; k5[1] = dV

        # Stage 6
        for j in range(2):
            ys[j] = y[j] + h * (DP_A5[0] * k1[j] + DP_A5[1] * k2[j]
                                 + DP_A5[2] * k3[j] + DP_A5[3] * k4[j]
                                 + DP_A5[4] * k5[j])
        ds, dV = _rail_deriv(
            t + DP_C[5] * h, ys[0], ys[1], sin_theta,
            motor_times, motor_thrusts, nozzle_area, altitude,
            impulse_factor, m_prop_0, total_impulse, m_dry,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, length,
        )
        k6[0] = ds; k6[1] = dV

        # 5th-order solution
        for j in range(2):
            y_new[j] = y[j] + h * (
                DP_B[0] * k1[j] + DP_B[2] * k3[j] + DP_B[3] * k4[j]
                + DP_B[4] * k5[j] + DP_B[5] * k6[j]
            )

        # Stage 7 (FSAL)
        alt_new = -y_new[0] * eD
        ds, dV = _rail_deriv(
            t + h, y_new[0], y_new[1], sin_theta,
            motor_times, motor_thrusts, nozzle_area, alt_new,
            impulse_factor, m_prop_0, total_impulse, m_dry,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, length,
        )
        k7[0] = ds; k7[1] = dV

        # Error estimate
        for j in range(2):
            y_err[j] = h * (
                DP_E[0] * k1[j] + DP_E[2] * k3[j] + DP_E[3] * k4[j]
                + DP_E[4] * k5[j] + DP_E[5] * k6[j] + DP_E[6] * k7[j]
            )

        err = error_norm(y, y_new, y_err, rtol, atol)

        if err <= 1.0:
            s_prev = y[0]
            V_prev = y[1]
            t_prev = t
            t += h
            for j in range(2):
                y[j] = y_new[j]
            k1, k7 = k7, k1  # FSAL swap — no copy

            if y[0] >= rail_length:
                # Interpolate to exact rail exit (V² linear in s)
                if y[0] > s_prev:
                    frac = (rail_length - s_prev) / (y[0] - s_prev)
                    V2_interp = V_prev * V_prev + frac * (y[1] * y[1] - V_prev * V_prev)
                    V_exit = V2_interp ** 0.5
                    ds = rail_length - s_prev
                    V_avg = 0.5 * (V_prev + V_exit)
                    t_exit = t_prev + ds / V_avg if V_avg > 1.0e-15 else t_prev + frac * (t - t_prev)
                    return V_exit, t_exit, rail_length * eN, rail_length * eE, rail_length * eD
                break

        factor = optimal_step_factor(err)
        h = clamp_step(h * factor, h_min, h_max)

        # Don't overshoot rail end by too much
        remaining = rail_length - y[0]
        if remaining > 0.0 and y[1] > 0.0:
            h_est = remaining / y[1] * 1.05
            if h_est < h:
                h = clamp_step(h_est, h_min, h_max)

    V_exit = y[1]
    t_exit = t
    s_exit = y[0]
    return V_exit, t_exit, s_exit * eN, s_exit * eE, s_exit * eD


# ---------------------------------------------------------------------------
# Phase 2: Free Flight 6DoF
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _sixdof_deriv(
    t: float, state: np.ndarray, dy: np.ndarray,
    # Motor
    motor_times: np.ndarray, motor_thrusts: np.ndarray,
    m_prop_0: float, total_impulse: float,
    nozzle_area: float, nozzle_position: float,
    m_dry: float, cg_dry: float, motor_cg_loaded: float,
    I_roll_dry: float, I_lateral_dry: float,
    prop_I_roll: float, prop_I_lateral: float,
    impulse_factor: float,
    # Aero
    mach_g: np.ndarray, re_g: np.ndarray, alpha_g: np.ndarray,
    ca_tbl: np.ndarray, cn_tbl: np.ndarray, cp_tbl: np.ndarray,
    cn_comp: np.ndarray, cp_comp: np.ndarray,
    has_components: bool, cn_alpha_fins_tbl: np.ndarray,
    # Geometry
    diameter: float, ref_length: float, A_ref: float, fin_cp_radius: float,
    # Wind
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    # Roll
    fin_cant_rad: float,
) -> tuple[float, float, float]:
    """13-component derivative for the 6DoF state vector.

    Writes derivatives into *dy* (pre-allocated by caller — no allocation).
    Returns ``(alpha_rad, cp_whole, mach)`` for acceptance checking.

    State layout: [rN, rE, rD, q0, q1, q2, q3, u, v, w, p, q_rate, r_rate]
    """
    # Unpack state
    rD = state[2]
    q0 = state[3]; q1 = state[4]; q2 = state[5]; q3 = state[6]
    u = state[7]; v = state[8]; w = state[9]
    p_rate = state[10]; q_r = state[11]; r_r = state[12]

    h = -rD
    if h < 0.0:
        h = 0.0

    # --- Atmosphere ---
    _, _, rho, a_sound, mu = isa(h)

    # --- DCM as 9 scalars (no array allocation) ---
    q0q0 = q0 * q0; q1q1 = q1 * q1; q2q2 = q2 * q2; q3q3 = q3 * q3
    q0q1 = q0 * q1; q0q2 = q0 * q2; q0q3 = q0 * q3
    q1q2 = q1 * q2; q1q3 = q1 * q3; q2q3 = q2 * q3

    # C_nb (body → NED)
    c00 = 1.0 - 2.0 * (q2q2 + q3q3)
    c01 = 2.0 * (q1q2 - q0q3)
    c02 = 2.0 * (q1q3 + q0q2)
    c10 = 2.0 * (q1q2 + q0q3)
    c11 = 1.0 - 2.0 * (q1q1 + q3q3)
    c12 = 2.0 * (q2q3 - q0q1)
    c20 = 2.0 * (q1q3 - q0q2)
    c21 = 2.0 * (q2q3 + q0q1)
    c22 = 1.0 - 2.0 * (q1q1 + q2q2)

    # --- Wind in body frame: v_wind_b = C_bn · [vN_wind, vE_wind, 0] ---
    # C_bn = C_nb^T  →  C_bn[i,j] = C_nb[j,i]
    vN_wind, vE_wind = interpolate_wind(wind_alt, wind_east, wind_north, h)
    wb_x = c00 * vN_wind + c10 * vE_wind
    wb_y = c01 * vN_wind + c11 * vE_wind
    wb_z = c02 * vN_wind + c12 * vE_wind

    # --- Relative velocity (body frame) ---
    u_rel = u - wb_x
    v_rel = v - wb_y
    w_rel = w - wb_z
    V = (u_rel * u_rel + v_rel * v_rel + w_rel * w_rel) ** 0.5

    # --- Mach & Reynolds ---
    if V > _EPS_V:
        M = V / a_sound
        Re = rho * V * ref_length / mu
    else:
        M = 0.0
        Re = 0.0

    # --- Motor properties ---
    F_thrust = thrust_corrected_at(
        motor_times, motor_thrusts, nozzle_area, h, t
    ) * impulse_factor
    m = mass_at(motor_times, motor_thrusts, m_prop_0, total_impulse, m_dry, t)
    cg = cg_at(
        motor_times, motor_thrusts, m_prop_0, total_impulse,
        m_dry, cg_dry, motor_cg_loaded, t,
    )
    I_roll, I_lat = inertia_at(
        motor_times, motor_thrusts, m_prop_0, total_impulse,
        m_dry, cg_dry, motor_cg_loaded,
        I_roll_dry, I_lateral_dry, prop_I_roll, prop_I_lateral, t,
    )

    # --- Aerodynamic forces and moments ---
    Fx, Fy, Fz, tau_p_aero, tau_y_aero, cp_whole = aero_forces_moments(
        mach_g, re_g, alpha_g,
        ca_tbl, cn_tbl, cp_tbl,
        cn_comp, cp_comp, has_components,
        M, Re, rho, V, A_ref,
        u_rel, v_rel, w_rel, q_r, r_r, cg,
    )

    # --- Jet damping (during burn only, section 8.3.4) ---
    m_dot = mdot_at(motor_times, motor_thrusts, m_prop_0, total_impulse, t)
    lever = nozzle_position - cg
    C2R = m_dot * lever * lever
    tau_p_jet = -C2R * q_r
    tau_y_jet = -C2R * r_r

    # --- Roll torques (Barrowman, section 6.5) ---
    tau_roll = 0.0
    if has_components and V > _EPS_V:
        cna_fins = cn_alpha_fins_at(
            mach_g, re_g, cn_alpha_fins_tbl, M, Re,
        )
        r_fin_d = fin_cp_radius / diameter
        C_l_delta = cna_fins * r_fin_d
        C_l_p = -cna_fins * r_fin_d * r_fin_d * (4.0 / 3.0)
        tau_cant = 0.5 * rho * V * V * A_ref * C_l_delta * fin_cant_rad
        tau_damp = -0.5 * rho * V * A_ref * diameter * C_l_p * (
            p_rate * diameter / (2.0 * V)
        )
        tau_roll = tau_cant + tau_damp

    # --- Total moments ---
    tau_pitch = tau_p_aero + tau_p_jet
    tau_yaw = tau_y_aero + tau_y_jet

    # --- Gravity in body frame: C_bn · [0, 0, m*g₀] ---
    # grav_b[i] = C_nb[2, i] * m * g₀
    mg = m * _G0
    gx = c20 * mg
    gy = c21 * mg
    gz = c22 * mg

    # --- Translational dynamics (section 8.3.2) ---
    inv_m = 1.0 / m
    du = (Fx + F_thrust + gx) * inv_m - (q_r * w - r_r * v)
    dv = (Fy + gy) * inv_m - (r_r * u - p_rate * w)
    dw = (Fz + gz) * inv_m - (p_rate * v - q_r * u)

    # --- Rotational dynamics (section 8.3.3) ---
    if I_roll > 1.0e-12:
        dp = tau_roll / I_roll
    else:
        dp = 0.0
    if I_lat > 1.0e-12:
        dq = ((I_lat - I_roll) * r_r * p_rate + tau_pitch) / I_lat
        dr = ((I_roll - I_lat) * p_rate * q_r + tau_yaw) / I_lat
    else:
        dq = 0.0
        dr = 0.0

    # --- Kinematics: dr_e/dt = C_nb · v_b ---
    dy[0] = c00 * u + c01 * v + c02 * w
    dy[1] = c10 * u + c11 * v + c12 * w
    dy[2] = c20 * u + c21 * v + c22 * w

    # --- Quaternion rate ---
    dy[3] = 0.5 * (-p_rate * q1 - q_r * q2 - r_r * q3)
    dy[4] = 0.5 * (p_rate * q0 + r_r * q2 - q_r * q3)
    dy[5] = 0.5 * (q_r * q0 - r_r * q1 + p_rate * q3)
    dy[6] = 0.5 * (r_r * q0 + q_r * q1 - p_rate * q2)

    dy[7] = du; dy[8] = dv; dy[9] = dw
    dy[10] = dp; dy[11] = dq; dy[12] = dr

    # --- Auxiliary outputs for acceptance checking ---
    V_lat = (v_rel * v_rel + w_rel * w_rel) ** 0.5
    alpha_rad = math.atan2(V_lat, u_rel)

    return alpha_rad, cp_whole, M


@nb.njit(cache=True, fastmath=True)
def integrate_sixdof(
    t0: float, state0: np.ndarray,
    t_burnout: float,
    # Motor
    motor_times: np.ndarray, motor_thrusts: np.ndarray,
    m_prop_0: float, total_impulse: float,
    nozzle_area: float, nozzle_position: float,
    m_dry: float, cg_dry: float, motor_cg_loaded: float,
    I_roll_dry: float, I_lateral_dry: float,
    prop_I_roll: float, prop_I_lateral: float,
    impulse_factor: float,
    # Aero
    mach_g: np.ndarray, re_g: np.ndarray, alpha_g: np.ndarray,
    ca_tbl: np.ndarray, cn_tbl: np.ndarray, cp_tbl: np.ndarray,
    cn_comp: np.ndarray, cp_comp: np.ndarray,
    has_components: bool, cn_alpha_fins_tbl: np.ndarray,
    # Geometry
    diameter: float, ref_length: float, A_ref: float, fin_cp_radius: float,
    # Wind
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    # Roll
    fin_cant_rad: float,
    # Acceptance
    sm_transition_mach: float,
    sm_subsonic_min: float, sm_supersonic_min: float,
    aoa_max_rad: float, sm_aoa_threshold_rad: float,
    # Tolerances
    rtol: float, atol: float,
) -> tuple[np.ndarray, np.ndarray, int,
           float, float, float, float, float,
           bool, int]:
    """Integrate 6DoF free-flight from rail exit to apogee or violation.

    Returns
    -------
    t_out, y_out, n_steps,
    max_mach, max_aoa_deg, min_sm_sub, min_sm_sup, peak_alt,
    stability_compliant, violation_code
    """
    MAX_STEPS = 50000
    N = 13

    t_out = np.empty(MAX_STEPS)
    y_out = np.empty((MAX_STEPS, N))

    y = state0.copy()
    t = t0
    h_step = 1.0e-3
    h_min = 1.0e-4
    n = 0

    # Store initial state
    t_out[0] = t
    for j in range(N):
        y_out[0, j] = y[j]
    n = 1

    # Tracking variables
    max_mach = 0.0
    max_aoa_deg = 0.0
    min_sm_sub = 1.0e6
    min_sm_sup = 1.0e6
    peak_alt = -y[2]
    stability_compliant = True
    violation_code = 0
    prev_rD = y[2]

    # Pre-allocate all work arrays — zero heap allocations in the loop
    k1 = np.empty(N)
    k2 = np.empty(N)
    k3 = np.empty(N)
    k4 = np.empty(N)
    k5 = np.empty(N)
    k6 = np.empty(N)
    k7 = np.empty(N)
    y_new = np.empty(N)
    y_err = np.empty(N)
    ys = np.empty(N)

    # Derivative args — captured once to reduce line noise below
    # (Numba sees through local aliases with no overhead)
    _mt = motor_times; _mth = motor_thrusts
    _mp0 = m_prop_0; _ti = total_impulse
    _na = nozzle_area; _np_ = nozzle_position
    _md = m_dry; _cgd = cg_dry; _mcl = motor_cg_loaded
    _ird = I_roll_dry; _ild = I_lateral_dry
    _pir = prop_I_roll; _pil = prop_I_lateral
    _if = impulse_factor
    _mg = mach_g; _rg = re_g; _ag = alpha_g
    _cat = ca_tbl; _cnt = cn_tbl; _cpt = cp_tbl
    _cnc = cn_comp; _cpc = cp_comp
    _hc = has_components; _cnaf = cn_alpha_fins_tbl
    _d = diameter; _rl = ref_length; _ar = A_ref; _fr = fin_cp_radius
    _wa = wind_alt; _we = wind_east; _wn = wind_north
    _fc = fin_cant_rad

    # Initial k1
    _sixdof_deriv(
        t, y, k1,
        _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
        _ird, _ild, _pir, _pil, _if,
        _mg, _rg, _ag, _cat, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
        _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
    )

    for _ in range(MAX_STEPS * 2):
        # Max step depends on burn status
        h_max = 0.05 if t < t_burnout else 0.1
        h_step = clamp_step(h_step, h_min, h_max)

        # --- Dormand-Prince stages (writing into pre-allocated buffers) ---
        # Stage 2
        for j in range(N):
            ys[j] = y[j] + h_step * DP_A1[0] * k1[j]
        _sixdof_deriv(
            t + DP_C[1] * h_step, ys, k2,
            _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
            _ird, _ild, _pir, _pil, _if,
            _mg, _rg, _ag, _cat, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
        )

        # Stage 3
        for j in range(N):
            ys[j] = y[j] + h_step * (DP_A2[0] * k1[j] + DP_A2[1] * k2[j])
        _sixdof_deriv(
            t + DP_C[2] * h_step, ys, k3,
            _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
            _ird, _ild, _pir, _pil, _if,
            _mg, _rg, _ag, _cat, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
        )

        # Stage 4
        for j in range(N):
            ys[j] = y[j] + h_step * (
                DP_A3[0] * k1[j] + DP_A3[1] * k2[j] + DP_A3[2] * k3[j]
            )
        _sixdof_deriv(
            t + DP_C[3] * h_step, ys, k4,
            _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
            _ird, _ild, _pir, _pil, _if,
            _mg, _rg, _ag, _cat, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
        )

        # Stage 5
        for j in range(N):
            ys[j] = y[j] + h_step * (
                DP_A4[0] * k1[j] + DP_A4[1] * k2[j]
                + DP_A4[2] * k3[j] + DP_A4[3] * k4[j]
            )
        _sixdof_deriv(
            t + DP_C[4] * h_step, ys, k5,
            _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
            _ird, _ild, _pir, _pil, _if,
            _mg, _rg, _ag, _cat, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
        )

        # Stage 6
        for j in range(N):
            ys[j] = y[j] + h_step * (
                DP_A5[0] * k1[j] + DP_A5[1] * k2[j]
                + DP_A5[2] * k3[j] + DP_A5[3] * k4[j]
                + DP_A5[4] * k5[j]
            )
        _sixdof_deriv(
            t + DP_C[5] * h_step, ys, k6,
            _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
            _ird, _ild, _pir, _pil, _if,
            _mg, _rg, _ag, _cat, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
        )

        # 5th-order solution
        for j in range(N):
            y_new[j] = y[j] + h_step * (
                DP_B[0] * k1[j] + DP_B[2] * k3[j] + DP_B[3] * k4[j]
                + DP_B[4] * k5[j] + DP_B[5] * k6[j]
            )

        # Stage 7 (FSAL) — also returns auxiliaries for acceptance checking
        alpha_rad, cp_whole, mach_now = _sixdof_deriv(
            t + h_step, y_new, k7,
            _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
            _ird, _ild, _pir, _pil, _if,
            _mg, _rg, _ag, _cat, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
        )

        # Error estimate
        for j in range(N):
            y_err[j] = h_step * (
                DP_E[0] * k1[j] + DP_E[2] * k3[j] + DP_E[3] * k4[j]
                + DP_E[4] * k5[j] + DP_E[5] * k6[j] + DP_E[6] * k7[j]
            )

        err = error_norm(y, y_new, y_err, rtol, atol)

        if err <= 1.0:
            # Accept step
            t += h_step

            # Quaternion renormalisation
            qn = (y_new[3] ** 2 + y_new[4] ** 2
                  + y_new[5] ** 2 + y_new[6] ** 2) ** 0.5
            if qn > 1.0e-15:
                inv_qn = 1.0 / qn
                y_new[3] *= inv_qn
                y_new[4] *= inv_qn
                y_new[5] *= inv_qn
                y_new[6] *= inv_qn

            prev_rD = y[2]
            for j in range(N):
                y[j] = y_new[j]
            k1, k7 = k7, k1  # FSAL swap — no copy

            # Store
            if n < MAX_STEPS:
                t_out[n] = t
                for j in range(N):
                    y_out[n, j] = y[j]
                n += 1

            # Track peak altitude
            h_now = -y[2]
            if h_now > peak_alt:
                peak_alt = h_now

            # --- Acceptance checks using FSAL auxiliaries (no redundant computation) ---
            aoa_deg = alpha_rad * _RAD2DEG
            if aoa_deg > max_aoa_deg:
                max_aoa_deg = aoa_deg
            if mach_now > max_mach:
                max_mach = mach_now

            # CG at accepted time for SM calculation
            cg_now = cg_at(
                _mt, _mth, _mp0, _ti, _md, _cgd, _mcl, t,
            )
            sm_cal = (cp_whole - cg_now) / _d

            # AoA check
            if alpha_rad > aoa_max_rad:
                stability_compliant = False
                violation_code = 1
                break

            # SM check (only when AoA < threshold)
            if alpha_rad < sm_aoa_threshold_rad:
                if mach_now < sm_transition_mach:
                    if sm_cal < min_sm_sub:
                        min_sm_sub = sm_cal
                    if sm_cal < sm_subsonic_min:
                        stability_compliant = False
                        violation_code = 2
                        break
                else:
                    if sm_cal < min_sm_sup:
                        min_sm_sup = sm_cal
                    if sm_cal < sm_supersonic_min:
                        stability_compliant = False
                        violation_code = 2
                        break

            # Apogee detection: rD goes from decreasing to increasing
            if y[2] > prev_rD and n > 2:
                break

        # Adjust step size
        factor = optimal_step_factor(err)
        h_step = clamp_step(h_step * factor, h_min, h_max)

    return (t_out, y_out, n,
            max_mach, max_aoa_deg, min_sm_sub, min_sm_sup, peak_alt,
            stability_compliant, violation_code)


# ---------------------------------------------------------------------------
# Phase 3: Descent (3DoF)
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _descent_deriv(
    t: float, state: np.ndarray, dy: np.ndarray,
    # Wind
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    # Aero (for ballistic scenario)
    mach_g: np.ndarray, re_g: np.ndarray, alpha_g: np.ndarray,
    ca_tbl: np.ndarray,
    A_ref: float, ref_length: float,
    # Mass
    m: float,
    # Recovery
    drogue_cda: float, main_cda: float, main_deploy_alt: float,
    # Scenario
    scenario: int,
) -> None:
    """6-component derivative for descent: [rN, rE, rD, vN, vE, vD].

    Writes into *dy* (pre-allocated by caller).
    ``scenario`` encoding: 0=nominal, 1=ballistic, 2=drogue_only,
    3=premature_main.
    """
    rD = state[2]
    vN = state[3]; vE = state[4]; vD = state[5]
    h = -rD
    if h < 0.0:
        h = 0.0

    _, _, rho, a_sound, mu = isa(h)
    vN_wind, vE_wind = interpolate_wind(wind_alt, wind_east, wind_north, h)

    vN_rel = vN - vN_wind
    vE_rel = vE - vE_wind
    vD_rel = vD
    V_rel = (vN_rel * vN_rel + vE_rel * vE_rel + vD_rel * vD_rel) ** 0.5

    # CdA for this scenario
    if scenario == SCENARIO_NOMINAL:
        if h <= main_deploy_alt:
            cda = drogue_cda + main_cda
        else:
            cda = drogue_cda
    elif scenario == SCENARIO_BALLISTIC:
        if V_rel > _EPS_V:
            M = V_rel / a_sound
            Re = rho * V_rel * ref_length / mu
            C_A = ca_at(mach_g, re_g, alpha_g, ca_tbl, M, Re, 0.0)
            cda = A_ref * C_A
        else:
            cda = A_ref * ca_at(mach_g, re_g, alpha_g, ca_tbl, 0.0, 0.0, 0.0)
    elif scenario == SCENARIO_DROGUE_ONLY:
        cda = drogue_cda
    else:  # SCENARIO_PREMATURE_MAIN
        cda = drogue_cda + main_cda

    if V_rel > _EPS_V:
        F_over_mV = 0.5 * rho * V_rel * cda / m  # |F|/(m·V) = 0.5·ρ·V·CdA/m
        dy[3] = -F_over_mV * vN_rel
        dy[4] = -F_over_mV * vE_rel
        dy[5] = -F_over_mV * vD_rel + _G0
    else:
        dy[3] = 0.0
        dy[4] = 0.0
        dy[5] = _G0

    dy[0] = vN; dy[1] = vE; dy[2] = vD


@nb.njit(cache=True, fastmath=True)
def integrate_descent(
    t0: float, state0: np.ndarray,
    # Wind
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    # Aero (for ballistic)
    mach_g: np.ndarray, re_g: np.ndarray, alpha_g: np.ndarray,
    ca_tbl: np.ndarray,
    A_ref: float, ref_length: float,
    # Mass
    m: float,
    # Recovery
    drogue_cda: float, main_cda: float, main_deploy_alt: float,
    # Scenario
    scenario: int,
    # Tolerances
    rtol: float, atol: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Integrate descent until ground impact (rD >= 0).

    Returns (t_out, y_out, n_steps).
    """
    MAX_STEPS = 20000
    NS = 6
    t_out = np.empty(MAX_STEPS)
    y_out = np.empty((MAX_STEPS, NS))

    y = state0.copy()
    t = t0
    h_step = 0.01
    h_min = 1.0e-4
    h_max = 1.0

    n = 0
    t_out[0] = t
    for j in range(NS):
        y_out[0, j] = y[j]
    n = 1

    # Pre-allocate work arrays
    k1 = np.empty(NS); k2 = np.empty(NS); k3 = np.empty(NS)
    k4 = np.empty(NS); k5 = np.empty(NS); k6 = np.empty(NS)
    k7 = np.empty(NS)
    y_new = np.empty(NS); y_err = np.empty(NS); ys = np.empty(NS)

    _descent_deriv(
        t, y, k1, wind_alt, wind_east, wind_north,
        mach_g, re_g, alpha_g, ca_tbl, A_ref, ref_length,
        m, drogue_cda, main_cda, main_deploy_alt, scenario,
    )

    for _ in range(MAX_STEPS * 2):
        h_step = clamp_step(h_step, h_min, h_max)

        # Stage 2
        for j in range(NS):
            ys[j] = y[j] + h_step * DP_A1[0] * k1[j]
        _descent_deriv(
            t + DP_C[1] * h_step, ys, k2,
            wind_alt, wind_east, wind_north,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, ref_length,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
        )

        # Stage 3
        for j in range(NS):
            ys[j] = y[j] + h_step * (DP_A2[0] * k1[j] + DP_A2[1] * k2[j])
        _descent_deriv(
            t + DP_C[2] * h_step, ys, k3,
            wind_alt, wind_east, wind_north,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, ref_length,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
        )

        # Stage 4
        for j in range(NS):
            ys[j] = y[j] + h_step * (
                DP_A3[0] * k1[j] + DP_A3[1] * k2[j] + DP_A3[2] * k3[j]
            )
        _descent_deriv(
            t + DP_C[3] * h_step, ys, k4,
            wind_alt, wind_east, wind_north,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, ref_length,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
        )

        # Stage 5
        for j in range(NS):
            ys[j] = y[j] + h_step * (
                DP_A4[0] * k1[j] + DP_A4[1] * k2[j]
                + DP_A4[2] * k3[j] + DP_A4[3] * k4[j]
            )
        _descent_deriv(
            t + DP_C[4] * h_step, ys, k5,
            wind_alt, wind_east, wind_north,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, ref_length,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
        )

        # Stage 6
        for j in range(NS):
            ys[j] = y[j] + h_step * (
                DP_A5[0] * k1[j] + DP_A5[1] * k2[j]
                + DP_A5[2] * k3[j] + DP_A5[3] * k4[j]
                + DP_A5[4] * k5[j]
            )
        _descent_deriv(
            t + DP_C[5] * h_step, ys, k6,
            wind_alt, wind_east, wind_north,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, ref_length,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
        )

        # 5th-order solution
        for j in range(NS):
            y_new[j] = y[j] + h_step * (
                DP_B[0] * k1[j] + DP_B[2] * k3[j] + DP_B[3] * k4[j]
                + DP_B[4] * k5[j] + DP_B[5] * k6[j]
            )

        # Stage 7 (FSAL)
        _descent_deriv(
            t + h_step, y_new, k7,
            wind_alt, wind_east, wind_north,
            mach_g, re_g, alpha_g, ca_tbl, A_ref, ref_length,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
        )

        # Error
        for j in range(NS):
            y_err[j] = h_step * (
                DP_E[0] * k1[j] + DP_E[2] * k3[j] + DP_E[3] * k4[j]
                + DP_E[4] * k5[j] + DP_E[5] * k6[j] + DP_E[6] * k7[j]
            )

        err = error_norm(y, y_new, y_err, rtol, atol)

        if err <= 1.0:
            t += h_step
            for j in range(NS):
                y[j] = y_new[j]
            k1, k7 = k7, k1  # FSAL swap

            if n < MAX_STEPS:
                t_out[n] = t
                for j in range(NS):
                    y_out[n, j] = y[j]
                n += 1

            if y[2] >= 0.0:
                break

        factor = optimal_step_factor(err)
        h_step = clamp_step(h_step * factor, h_min, h_max)

    return t_out, y_out, n


# ---------------------------------------------------------------------------
# 3DoF Ascent (for optimisation, section 8.5)
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def simulate_ascent_3dof(
    rail_azimuth_rad: float,
    rail_inclination_rad: float,
    rail_length: float,
    motor_times: np.ndarray,
    motor_thrusts: np.ndarray,
    nozzle_area: float,
    impulse_factor: float,
    m_prop_0: float,
    total_impulse: float,
    m_dry: float,
    mach_g: np.ndarray,
    re_g: np.ndarray,
    alpha_g: np.ndarray,
    ca_tbl: np.ndarray,
    A_ref: float,
    ref_length: float,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, float, float, float]:
    """Simplified 3DoF ascent: point-mass, α=0, no roll.

    Runs rail phase then coasts to apogee along the launch axis direction
    (no attitude changes — velocity vector stays along initial rail heading).

    Returns (apogee_alt, apogee_N, apogee_E, apogee_D, t_apogee, V_apogee).
    """
    V_exit, t_exit, rN, rE, rD = simulate_rail(
        rail_azimuth_rad, rail_inclination_rad, rail_length,
        motor_times, motor_thrusts, nozzle_area, impulse_factor,
        m_prop_0, total_impulse, m_dry,
        mach_g, re_g, alpha_g, ca_tbl, A_ref, ref_length,
        rtol, atol,
    )

    eN, eE, eD = _rail_direction(rail_azimuth_rad, rail_inclination_rad)
    sin_theta = math.sin(rail_inclination_rad)

    t = t_exit
    V = V_exit
    pos_N = rN; pos_E = rE; pos_D = rD
    t_burnout = motor_times[motor_times.shape[0] - 1]
    max_steps = 100000

    for _ in range(max_steps):
        h_now = -pos_D
        if h_now < 0.0:
            h_now = 0.0

        _, _, rho, a_sound, mu = isa(h_now)
        m = mass_at(
            motor_times, motor_thrusts, m_prop_0, total_impulse, m_dry, t,
        )
        F_thrust = thrust_corrected_at(
            motor_times, motor_thrusts, nozzle_area, h_now, t,
        ) * impulse_factor

        V_abs = abs(V)
        if V_abs > _EPS_V:
            M = V_abs / a_sound
            Re = rho * V_abs * ref_length / mu
            C_A = ca_at(mach_g, re_g, alpha_g, ca_tbl, M, Re, 0.0)
            F_drag = 0.5 * rho * V_abs * V_abs * A_ref * C_A
        else:
            F_drag = 0.0

        a = (F_thrust - F_drag) / m - _G0 * sin_theta

        # Simple RK4 step
        dt = 0.01 if t < t_burnout else 0.05

        V_new = V + dt * a
        V_mid = V + 0.5 * dt * a
        pos_N += V_mid * dt * eN
        pos_E += V_mid * dt * eE
        pos_D += V_mid * dt * eD
        V = V_new
        t += dt

        if V <= 0.0:
            break

    return -pos_D, pos_N, pos_E, pos_D, t, V


# ---------------------------------------------------------------------------
# Top-level trajectory runner (plain Python)
# ---------------------------------------------------------------------------

def build_sim_params(
    motor: MotorModel,
    aero: AeroModel,
    wind_alt: np.ndarray,
    wind_east: np.ndarray,
    wind_north: np.ndarray,
    diameter: float,
    length: float,
    A_ref: float,
    fin_cp_radius: float,
    rail_azimuth_rad: float,
    rail_inclination_rad: float,
    rail_length: float,
    fin_cant_rad: float,
    impulse_factor: float,
    drogue_cda: float,
    main_cda: float,
    main_deploy_alt: float,
    has_drogue: bool,
    has_main: bool,
    sm_transition_mach: float,
    sm_subsonic_min: float,
    sm_supersonic_min: float,
    aoa_max_rad: float,
    sm_aoa_threshold_rad: float,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-6,
) -> SimParams:
    """Convenience constructor for SimParams from model objects."""
    return SimParams(
        motor_times=motor.times,
        motor_thrusts=motor.thrusts,
        m_prop_0=motor.m_prop_0,
        total_impulse=motor.total_impulse,
        nozzle_area=motor.nozzle_area,
        nozzle_position=motor.nozzle_position,
        m_dry=motor.m_dry,
        cg_dry=motor.cg_dry,
        motor_cg_loaded=motor.motor_cg_loaded,
        I_roll_dry=motor.I_roll_dry,
        I_lateral_dry=motor.I_lateral_dry,
        prop_I_roll=motor.prop_I_roll,
        prop_I_lateral=motor.prop_I_lateral,
        mach_g=aero.mach_grid,
        re_g=aero.re_grid,
        alpha_g=aero.alpha_grid,
        ca_tbl=aero.ca_table,
        cn_tbl=aero.cn_table,
        cp_tbl=aero.cp_table,
        cn_comp=aero.cn_comp,
        cp_comp=aero.cp_comp,
        has_components=aero.has_components,
        cn_alpha_fins=aero.cn_alpha_fins,
        diameter=diameter,
        length=length,
        A_ref=A_ref,
        fin_cp_radius=fin_cp_radius,
        wind_alt=wind_alt,
        wind_east=wind_east,
        wind_north=wind_north,
        rail_azimuth_rad=rail_azimuth_rad,
        rail_inclination_rad=rail_inclination_rad,
        rail_length=rail_length,
        fin_cant_rad=fin_cant_rad,
        impulse_factor=impulse_factor,
        drogue_cda=drogue_cda,
        main_cda=main_cda,
        main_deploy_alt=main_deploy_alt,
        has_drogue=has_drogue,
        has_main=has_main,
        sm_transition_mach=sm_transition_mach,
        sm_subsonic_min=sm_subsonic_min,
        sm_supersonic_min=sm_supersonic_min,
        aoa_max_rad=aoa_max_rad,
        sm_aoa_threshold_rad=sm_aoa_threshold_rad,
        rtol=rtol,
        atol=atol,
    )


def run_trajectory(
    params: SimParams,
    scenario: int,
    poly_e: np.ndarray | None = None,
    poly_n: np.ndarray | None = None,
    buffered_ceiling: float = float('inf'),
) -> TrajectoryResult:
    """Run a complete trajectory: rail → 6DoF ascent → 3DoF descent.

    Parameters
    ----------
    params : SimParams
        All pre-processed data for this sample.
    scenario : int
        Descent scenario (use SCENARIO_* constants).
    poly_e, poly_n : np.ndarray or None
        Buffered danger-area polygon exterior ring (east, north) in NED
        metres.  If ``None``, footprint containment checks are skipped.
    buffered_ceiling : float
        ``altitude_ceiling - buffer_distance`` in metres.  Defaults to
        ``inf`` (no ceiling check).

    Returns
    -------
    TrajectoryResult
    """
    p = params
    t_burnout = float(p.motor_times[-1])

    # ---- Phase 1: Launch rail ----
    V_exit, t_exit, rN_exit, rE_exit, rD_exit = simulate_rail(
        p.rail_azimuth_rad, p.rail_inclination_rad, p.rail_length,
        p.motor_times, p.motor_thrusts, p.nozzle_area, p.impulse_factor,
        p.m_prop_0, p.total_impulse, p.m_dry,
        p.mach_g, p.re_g, p.alpha_g, p.ca_tbl,
        p.A_ref, p.length,
        p.rtol, p.atol,
    )

    # ---- Phase 2 initial conditions ----
    q0, q1, q2, q3 = quat_from_rail(
        p.rail_azimuth_rad, p.rail_inclination_rad,
    )
    state0 = np.array([
        rN_exit, rE_exit, rD_exit,
        q0, q1, q2, q3,
        V_exit, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], dtype=np.float64)

    # ---- Phase 2: 6DoF free flight ----
    (t_hist, state_hist, n_steps,
     max_mach, max_aoa_deg, min_sm_sub, min_sm_sup, peak_alt,
     stab_ok, viol_code) = integrate_sixdof(
        t_exit, state0, t_burnout,
        p.motor_times, p.motor_thrusts, p.m_prop_0, p.total_impulse,
        p.nozzle_area, p.nozzle_position, p.m_dry, p.cg_dry,
        p.motor_cg_loaded,
        p.I_roll_dry, p.I_lateral_dry, p.prop_I_roll, p.prop_I_lateral,
        p.impulse_factor,
        p.mach_g, p.re_g, p.alpha_g,
        p.ca_tbl, p.cn_tbl, p.cp_tbl,
        p.cn_comp, p.cp_comp, p.has_components, p.cn_alpha_fins,
        p.diameter, p.length, p.A_ref, p.fin_cp_radius,
        p.wind_alt, p.wind_east, p.wind_north,
        p.fin_cant_rad,
        p.sm_transition_mach,
        p.sm_subsonic_min, p.sm_supersonic_min,
        p.aoa_max_rad, p.sm_aoa_threshold_rad,
        p.rtol, p.atol,
    )

    t_asc = t_hist[:n_steps].copy()
    s_asc = state_hist[:n_steps].copy()

    apogee_idx = n_steps - 1
    apogee_state = s_asc[apogee_idx]
    apogee_t = t_asc[apogee_idx]
    apogee_pos = apogee_state[:3].copy()
    apogee_alt = -apogee_state[2]

    # ---- Geofence checks (between ascent and descent) ----
    _below_ceiling = apogee_alt <= buffered_ceiling
    _in_buffer = True  # assume compliant until proven otherwise

    if poly_e is not None and poly_n is not None:
        from geofence import all_points_in_polygon
        _in_buffer = all_points_in_polygon(
            s_asc[:, 0], s_asc[:, 1], poly_e, poly_n,
        )

    if not _below_ceiling or not _in_buffer:
        # Skip descent — sample is already non-compliant.
        if not stab_ok:
            if viol_code == 1:
                viol_str = "AoA exceeded maximum"
            elif viol_code == 2:
                viol_str = "Static margin below minimum"
            else:
                viol_str = "Unknown stability violation"
        elif not _below_ceiling:
            viol_str = "Apogee above buffered ceiling"
        else:
            viol_str = "Trajectory exited buffered danger area"

        return TrajectoryResult(
            apogee_altitude=apogee_alt,
            apogee_time=apogee_t,
            apogee_position=apogee_pos,
            landing_position=apogee_pos,
            landing_time=apogee_t,
            flight_time=apogee_t - t_exit,
            max_mach=max_mach,
            max_aoa_deg=max_aoa_deg,
            min_sm_subsonic=min_sm_sub if min_sm_sub < 1.0e5 else float('nan'),
            min_sm_supersonic=min_sm_sup if min_sm_sup < 1.0e5 else float('nan'),
            rail_exit_velocity=V_exit,
            peak_altitude_ft=peak_alt * 3.28084,
            in_buffer=_in_buffer,
            below_ceiling=_below_ceiling,
            compliant=False,
            stability_compliant=stab_ok,
            violation_reason=viol_str,
            t_ascent=t_asc,
            state_ascent=s_asc,
        )

    # ---- Phase 3 initial conditions ----
    C_nb = quat_to_dcm_nb(
        apogee_state[3], apogee_state[4],
        apogee_state[5], apogee_state[6],
    )
    vb = apogee_state[7:10]
    vN = C_nb[0, 0] * vb[0] + C_nb[0, 1] * vb[1] + C_nb[0, 2] * vb[2]
    vE = C_nb[1, 0] * vb[0] + C_nb[1, 1] * vb[1] + C_nb[1, 2] * vb[2]
    vD = C_nb[2, 0] * vb[0] + C_nb[2, 1] * vb[1] + C_nb[2, 2] * vb[2]

    descent_state0 = np.array([
        apogee_pos[0], apogee_pos[1], apogee_pos[2],
        vN, vE, vD,
    ], dtype=np.float64)

    drogue_cda = p.drogue_cda if p.has_drogue else 0.0
    main_cda = p.main_cda if p.has_main else 0.0
    main_deploy_alt = p.main_deploy_alt

    effective_scenario = scenario
    if scenario == SCENARIO_NOMINAL and main_deploy_alt < 0.0:
        effective_scenario = SCENARIO_PREMATURE_MAIN

    # ---- Phase 3: Descent ----
    t_desc, y_desc, n_desc = integrate_descent(
        apogee_t, descent_state0,
        p.wind_alt, p.wind_east, p.wind_north,
        p.mach_g, p.re_g, p.alpha_g, p.ca_tbl,
        p.A_ref, p.length,
        p.m_dry, drogue_cda, main_cda, main_deploy_alt,
        effective_scenario,
        p.rtol, p.atol,
    )

    land_idx = n_desc - 1
    landing_pos = y_desc[land_idx, :3].copy()
    landing_t = t_desc[land_idx]

    # ---- Descent footprint check ----
    if poly_e is not None and poly_n is not None:
        from geofence import all_points_in_polygon
        if not all_points_in_polygon(
            y_desc[:n_desc, 0], y_desc[:n_desc, 1], poly_e, poly_n,
        ):
            _in_buffer = False

    if not stab_ok:
        if viol_code == 1:
            viol_str = "AoA exceeded maximum"
        elif viol_code == 2:
            viol_str = "Static margin below minimum"
        else:
            viol_str = "Unknown stability violation"
    elif not _in_buffer:
        viol_str = "Trajectory exited buffered danger area"
    else:
        viol_str = ""

    _compliant = stab_ok and _in_buffer and _below_ceiling

    return TrajectoryResult(
        apogee_altitude=apogee_alt,
        apogee_time=apogee_t,
        apogee_position=apogee_pos,
        landing_position=landing_pos,
        landing_time=landing_t,
        flight_time=landing_t - t_exit,
        max_mach=max_mach,
        max_aoa_deg=max_aoa_deg,
        min_sm_subsonic=min_sm_sub if min_sm_sub < 1.0e5 else float('nan'),
        min_sm_supersonic=min_sm_sup if min_sm_sup < 1.0e5 else float('nan'),
        rail_exit_velocity=V_exit,
        peak_altitude_ft=peak_alt * 3.28084,
        in_buffer=_in_buffer,
        below_ceiling=_below_ceiling,
        compliant=_compliant,
        stability_compliant=stab_ok,
        violation_reason=viol_str,
        t_ascent=t_asc,
        state_ascent=s_asc,
    )
