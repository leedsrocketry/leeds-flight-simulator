"""6DoF and launch-rail dynamics with adaptive Dormand-Prince integration.

Implements the equations of motion from specification sections 8.2–8.5:
    Phase 1 — Launch rail (constrained 1-D translation)
    Phase 2 — Free flight 6DoF (full translation + rotation)
    Phase 3 — Descent (point-mass under drag + gravity)

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
    SimParams         — all pre-processed data for one trajectory
    FlightSummary     — scalar outputs of a single trajectory
    TrajectoryProfile — unified time-history (only when requested)

@njit quaternion/frame utilities:
    quat_from_rail, quat_to_dcm_nb, quat_rate, quat_normalize

@njit phase runners:
    simulate_rail         — Phase 1
    integrate_sixdof      — Phase 2
    integrate_descent     — Phase 3

Top-level entry point:
    run_trajectory(params, scenario) → FlightSummary (+ TrajectoryProfile)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numba as nb

from atmosphere import isa_at_site, pressure_at_site
from wind import interpolate_wind
from aerodynamics import (
    aero_forces_moments,
    ca_at,
    cn_cp_at,
    cn_alpha_fins_at,
    cn_alpha_comp_at,
    _interp3,
)
from motor import (
    _interp as _motor_interp,
    mdot_at,
    m_prop_at,
    PropellantModel,
)
from integrator import (
    DP_C, DP_A1, DP_A2, DP_A3, DP_A4, DP_A5, DP_B, DP_E,
    error_norm, optimal_step_factor, clamp_step,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_G0: float = 9.80665
_CD_CYLINDER: float = 1.2  # crossflow drag coefficient for a circular cylinder
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
# Vehicle-level @njit helpers (time-varying mass properties + thrust)
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def thrust_corrected_at(
    times: np.ndarray,
    thrusts: np.ndarray,
    nozzle_area: float,
    altitude_m: float,
    site_elevation: float,
    t: float,
) -> float:
    """Altitude-corrected thrust [N] at time *t* and altitude *altitude_m*.

    The .eng thrust curve is assumed to be measured at sea level.  Thrust
    increases as ambient back-pressure falls:

        F(h) = F₀ + Aₑ · (p₀ − p_ISA(h + site_elevation))
    """
    F0 = _motor_interp(times, thrusts, t)
    if F0 <= 0.0:
        return 0.0
    delta_p = 101325.0 - pressure_at_site(altitude_m, site_elevation)
    return F0 + nozzle_area * delta_p


@nb.njit(cache=True, fastmath=True)
def mass_at(
    times: np.ndarray,
    thrusts: np.ndarray,
    m_prop_0: float,
    total_impulse: float,
    m_dry: float,
    t: float,
) -> float:
    """Total vehicle mass [kg] at time *t*."""
    return m_dry + m_prop_at(times, thrusts, m_prop_0, total_impulse, t)


@nb.njit(cache=True, fastmath=True)
def cg_at(
    times: np.ndarray,
    thrusts: np.ndarray,
    m_prop_0: float,
    total_impulse: float,
    m_dry: float,
    cg_dry: float,
    motor_cg_loaded: float,
    t: float,
) -> float:
    """Vehicle CG [m from nosecone] at time *t*.

    CG(t) = (m_dry · cg_dry + m_prop(t) · motor_cg_loaded) / (m_dry + m_prop(t))
    """
    m_p = m_prop_at(times, thrusts, m_prop_0, total_impulse, t)
    return (m_dry * cg_dry + m_p * motor_cg_loaded) / (m_dry + m_p)


@nb.njit(cache=True, fastmath=True)
def inertia_at(
    times: np.ndarray,
    thrusts: np.ndarray,
    m_prop_0: float,
    total_impulse: float,
    m_dry: float,
    cg_dry: float,
    motor_cg_loaded: float,
    I_roll_dry: float,
    I_lateral_dry: float,
    prop_r_outer: float,
    prop_r_inner_0: float,
    prop_length: float,
    t: float,
) -> tuple[float, float]:
    """Whole-vehicle (I_roll, I_lateral) [kg·m²] at time *t*.

    Propellant is an annular cylinder burning radially outward.  As the mass
    fraction *f* decreases the inner radius grows:

        r_inner(t) = sqrt(r_outer² − f · (r_outer² − r_inner_0²))

    Inertias are recomputed from the current annular geometry rather than
    scaled linearly with mass fraction.  The parallel-axis theorem transfers
    each contribution to the instantaneous vehicle CG.
    """
    m_p = m_prop_at(times, thrusts, m_prop_0, total_impulse, t)

    if m_p <= 0.0:
        # Burnout — only dry inertias remain
        return I_roll_dry, I_lateral_dry

    f = m_p / m_prop_0   # 1 at ignition → 0 at burnout

    # Current inner radius (radial burn outward from bore)
    r_o2 = prop_r_outer * prop_r_outer
    r_i0_2 = prop_r_inner_0 * prop_r_inner_0
    r_i2 = r_o2 - f * (r_o2 - r_i0_2)
    # Guard against floating-point overshoot
    if r_i2 < 0.0:
        r_i2 = 0.0

    # Current vehicle CG
    cg_t = (m_dry * cg_dry + m_p * motor_cg_loaded) / (m_dry + m_p)

    # Propellant roll inertia (annulus, no PAT needed)
    I_roll_prop = 0.5 * m_p * (r_o2 + r_i2)
    I_roll = I_roll_dry + I_roll_prop

    # Propellant lateral inertia about its own CG (annular cylinder)
    I_lat_prop_own = m_p * (3.0 * (r_o2 + r_i2) + prop_length * prop_length) / 12.0

    # Lateral — dry contribution about current CG (PAT)
    d_dry = cg_dry - cg_t
    I_lat_dry = I_lateral_dry + m_dry * d_dry * d_dry

    # Lateral — propellant contribution about current CG (PAT)
    d_prop = motor_cg_loaded - cg_t
    I_lat_prop = I_lat_prop_own + m_p * d_prop * d_prop

    return I_roll, I_lat_dry + I_lat_prop


# ---------------------------------------------------------------------------
# Data structures (plain Python — not Numba-visible)
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """All pre-processed data needed to run a single trajectory.

    Construct in ``montecarlo.py`` from ``PropellantModel``, ``AeroModel``,
    ``WindEnsemble``, and the simulation/vehicle configs.
    """
    # Motor / propellant (from PropellantModel + derived dry properties)
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
    prop_r_outer: float
    prop_r_inner_0: float
    prop_length: float

    # Aero (from AeroModel)
    mach_g: np.ndarray
    re_g: np.ndarray
    alpha_g: np.ndarray
    ca_tbl_off: np.ndarray
    ca_tbl_on: np.ndarray
    cn_tbl: np.ndarray
    cp_tbl: np.ndarray
    cn_comp: np.ndarray
    cp_comp: np.ndarray
    has_components: bool
    cn_alpha_fins: np.ndarray
    cn_alpha_comp: np.ndarray    # [N_comp, NM, NR] per-component C_Nα
    comp_names: list[str]

    # Geometry
    diameter: float
    length: float
    A_ref: float
    fin_cp_radius: float
    fin_span: float

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
    sm_subsonic_min: float        # calibres
    sm_supersonic_min: float      # calibres

    # Atmosphere site corrections
    site_elevation: float             # metres MSL
    t_offset: float                   # K — uniform temperature offset from ISA

    # Integration tolerances
    rtol: float = 1.0e-6
    atol: float = 1.0e-6


@dataclass
class FlightSummary:
    """Scalar outputs of a single trajectory simulation.

    Contains only summary values — no time-history arrays.  Modules
    outside ``dynamics.py`` should never need to know about flight
    phases (rail / ascent / descent); that knowledge is encapsulated
    here and in :class:`TrajectoryProfile`.
    """
    apogee: float                 # m AGL
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
    footprint_compliant: bool     # full trajectory inside buffered footprint
    ceiling_compliant: bool       # apogee below buffered altitude ceiling
    stability_compliant: bool


def check_stability_compliance(
    summary: FlightSummary,
    acceptance: "AcceptanceConfig",
    label: str,
) -> None:
    """Raise ``RuntimeError`` if *summary* violates stability thresholds.

    Parameters
    ----------
    summary : FlightSummary
        Result of a single trajectory run.
    acceptance : config.AcceptanceConfig
        Configured stability thresholds.
    label : str
        Human-readable identifier for the trajectory (used in the error
        message), e.g. ``"Baseline 'nominal'"`` or ``"Verification"``.
    """
    if not summary.stability_compliant:
        raise RuntimeError(
            f"{label} trajectory is stability-non-compliant "
            f"(min SM subsonic = {summary.min_sm_subsonic:.2f} cal "
            f"[limit {acceptance.sm_subsonic_min}], "
            f"min SM supersonic = {summary.min_sm_supersonic:.2f} cal "
            f"[limit {acceptance.sm_supersonic_min}]). "
            f"The vehicle does not meet the configured acceptance "
            f"thresholds even with zero uncertainties. Review the "
            f"transonic stability margins or adjust acceptance "
            f"thresholds in the config. "
            f"Use --no-termination to run regardless."
        )


@dataclass
class TrajectoryProfile:
    """Unified flight profile from ignition to landing.

    All arrays share a common time axis covering every phase of
    flight (rail, free flight, descent).  No consumer outside
    ``dynamics.py`` should ever need to know where one phase ends
    and another begins.

    Quantities that are undefined during part of the flight (e.g.
    angle of attack during parachute descent) are ``NaN``.
    """
    time: np.ndarray          # (K,) seconds from ignition
    position_ned: np.ndarray  # (K, 3) [N, E, D] metres
    altitude: np.ndarray      # (K,) metres AGL
    mach: np.ndarray          # (K,)
    aoa_deg: np.ndarray       # (K,) NaN during descent
    sm: np.ndarray            # (K,) calibres
    thrust: np.ndarray        # (K,) Newtons, 0 after burnout
    mass: np.ndarray          # (K,) kg
    cd: np.ndarray            # (K,) vehicle CD during ascent,
                              #       parachute CD from config during descent
    roll_rate_hz: np.ndarray  # (K,) roll rate in Hz; NaN during rail & parachute descent
    cg: np.ndarray            # (K,) centre of gravity from nose tip [m]; dry CG during descent
    I_roll: np.ndarray        # (K,) roll moment of inertia [kg·m²]; dry value during descent
    I_lateral: np.ndarray     # (K,) lateral moment of inertia [kg·m²]; dry value during descent
    mdot: np.ndarray          # (K,) mass flow rate [kg/s]; 0 after burnout

    # --- Damping quantities (computed by compute_damping, NaN until then) ---
    c1: np.ndarray | None = None            # (K,) corrective moment coefficient
    c2: np.ndarray | None = None            # (K,) damping moment coefficient
    c2a: np.ndarray | None = None           # (K,) aerodynamic damping moment coefficient
    c2r: np.ndarray | None = None           # (K,) jet damping moment coefficient
    zeta: np.ndarray | None = None          # (K,) damping ratio
    omega_n: np.ndarray | None = None       # (K,) natural frequency (rad/s)
    omega_d: np.ndarray | None = None       # (K,) damped frequency (rad/s)
    max_roll_rate_hz: np.ndarray | None = None  # (K,) max permissible roll rate
    # Per-component breakdown for damping plots
    cn_alpha_comp: np.ndarray | None = None  # (N_comp, K) CN_alpha per component
    cp_comp: np.ndarray | None = None        # (N_comp, K) CP per component
    c1_comp: np.ndarray | None = None        # (N_comp, K) per-component C1 contribution
    c2a_comp: np.ndarray | None = None       # (N_comp, K) per-component C2A contribution
    comp_names: list[str] | None = None      # component names


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
    ca_tbl_off: np.ndarray, ca_tbl_on: np.ndarray,
    t_burnout: float,
    A_ref: float, length: float,
    site_elevation: float, t_offset: float,
) -> tuple[float, float]:
    """Derivatives for launch rail: ds/dt = V, dV/dt = F_along / m."""
    _, _, rho, a_sound, mu = isa_at_site(altitude, site_elevation, t_offset)

    F_thrust = thrust_corrected_at(
        motor_times, motor_thrusts, nozzle_area, altitude, site_elevation, t
    ) * impulse_factor

    m = mass_at(motor_times, motor_thrusts, m_prop_0, total_impulse, m_dry, t)

    V_abs = abs(V)
    if V_abs > _EPS_V:
        M = V_abs / a_sound
        Re = rho * V_abs * length / mu
        ca_tbl = ca_tbl_on if t <= t_burnout else ca_tbl_off
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
    ca_tbl_off: np.ndarray,
    ca_tbl_on: np.ndarray,
    t_burnout: float,
    A_ref: float,
    length: float,
    site_elevation: float,
    t_offset: float,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, float, float,
           np.ndarray, np.ndarray, np.ndarray, int]:
    """Integrate launch-rail phase until CG travels ``rail_length``.

    Records ``(t, V, altitude)`` at each accepted integrator step so the
    full rail-phase trajectory is available to callers that need it (e.g.
    verification).  Callers that only need exit conditions can ignore the
    last four return values.

    Returns
    -------
    (V_exit, t_exit, rN_exit, rE_exit, rD_exit,
     t_hist, V_hist, alt_hist, n_hist)
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
    max_hist = 500
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

    t_hist = np.empty(max_hist, dtype=np.float64)
    V_hist = np.empty(max_hist, dtype=np.float64)
    alt_hist = np.empty(max_hist, dtype=np.float64)
    n_hist = 0

    y[0] = s; y[1] = V

    # Record initial state (t=0, V=0, alt=0)
    t_hist[0] = 0.0
    V_hist[0] = 0.0
    alt_hist[0] = 0.0
    n_hist = 1

    ds, dV = _rail_deriv(
        t, y[0], y[1], sin_theta,
        motor_times, motor_thrusts, nozzle_area, altitude,
        impulse_factor, m_prop_0, total_impulse, m_dry,
        mach_g, re_g, alpha_g, ca_tbl_off, ca_tbl_on,
        t_burnout, A_ref, length,
        site_elevation, t_offset,
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
            mach_g, re_g, alpha_g, ca_tbl_off, ca_tbl_on,
        t_burnout, A_ref, length,
            site_elevation, t_offset,
        )
        k2[0] = ds; k2[1] = dV

        # Stage 3
        for j in range(2):
            ys[j] = y[j] + h * (DP_A2[0] * k1[j] + DP_A2[1] * k2[j])
        ds, dV = _rail_deriv(
            t + DP_C[2] * h, ys[0], ys[1], sin_theta,
            motor_times, motor_thrusts, nozzle_area, altitude,
            impulse_factor, m_prop_0, total_impulse, m_dry,
            mach_g, re_g, alpha_g, ca_tbl_off, ca_tbl_on,
        t_burnout, A_ref, length,
            site_elevation, t_offset,
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
            mach_g, re_g, alpha_g, ca_tbl_off, ca_tbl_on,
        t_burnout, A_ref, length,
            site_elevation, t_offset,
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
            mach_g, re_g, alpha_g, ca_tbl_off, ca_tbl_on,
        t_burnout, A_ref, length,
            site_elevation, t_offset,
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
            mach_g, re_g, alpha_g, ca_tbl_off, ca_tbl_on,
        t_burnout, A_ref, length,
            site_elevation, t_offset,
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
            mach_g, re_g, alpha_g, ca_tbl_off, ca_tbl_on,
        t_burnout, A_ref, length,
            site_elevation, t_offset,
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
                    # Record the interpolated exit point (not the overshoot)
                    if n_hist < max_hist:
                        t_hist[n_hist] = t_exit
                        V_hist[n_hist] = V_exit
                        alt_hist[n_hist] = max(rail_length * (-eD), 0.0)
                        n_hist += 1
                    return (V_exit, t_exit,
                            rail_length * eN, rail_length * eE, rail_length * eD,
                            t_hist, V_hist, alt_hist, n_hist)
                break

            # Record accepted step (only if we didn't exit the rail above)
            if n_hist < max_hist:
                t_hist[n_hist] = t
                V_hist[n_hist] = y[1]
                alt_hist[n_hist] = max(-y[0] * eD, 0.0)
                n_hist += 1

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
    return (V_exit, t_exit, s_exit * eN, s_exit * eE, s_exit * eD,
            t_hist, V_hist, alt_hist, n_hist)


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
    prop_r_outer: float, prop_r_inner_0: float, prop_length: float,
    impulse_factor: float,
    # Aero
    mach_g: np.ndarray, re_g: np.ndarray, alpha_g: np.ndarray,
    ca_tbl_off: np.ndarray, ca_tbl_on: np.ndarray,
    t_burnout: float,
    cn_tbl: np.ndarray, cp_tbl: np.ndarray,
    cn_comp: np.ndarray, cp_comp: np.ndarray,
    has_components: bool, cn_alpha_fins_tbl: np.ndarray,
    # Geometry
    diameter: float, ref_length: float, A_ref: float, fin_cp_radius: float,
    # Wind
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    # Roll
    fin_cant_rad: float,
    # Atmosphere site corrections
    site_elevation: float, t_offset: float,
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
    _, _, rho, a_sound, mu = isa_at_site(h, site_elevation, t_offset)

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
        motor_times, motor_thrusts, nozzle_area, h, site_elevation, t
    ) * impulse_factor
    m = mass_at(motor_times, motor_thrusts, m_prop_0, total_impulse, m_dry, t)
    cg = cg_at(
        motor_times, motor_thrusts, m_prop_0, total_impulse,
        m_dry, cg_dry, motor_cg_loaded, t,
    )
    I_roll, I_lat = inertia_at(
        motor_times, motor_thrusts, m_prop_0, total_impulse,
        m_dry, cg_dry, motor_cg_loaded,
        I_roll_dry, I_lateral_dry,
        prop_r_outer, prop_r_inner_0, prop_length, t,
    )

    # --- Aerodynamic forces and moments ---
    power_on = t <= t_burnout
    Fx, Fy, Fz, tau_p_aero, tau_y_aero, cp_whole = aero_forces_moments(
        mach_g, re_g, alpha_g,
        ca_tbl_off, ca_tbl_on, power_on,
        cn_tbl, cp_tbl,
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
    prop_r_outer: float, prop_r_inner_0: float, prop_length: float,
    impulse_factor: float,
    # Aero
    mach_g: np.ndarray, re_g: np.ndarray, alpha_g: np.ndarray,
    ca_tbl_off: np.ndarray, ca_tbl_on: np.ndarray,
    cn_tbl: np.ndarray, cp_tbl: np.ndarray,
    cn_comp: np.ndarray, cp_comp: np.ndarray,
    has_components: bool, cn_alpha_fins_tbl: np.ndarray,
    # Geometry
    diameter: float, ref_length: float, A_ref: float, fin_cp_radius: float,
    # Wind
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    # Roll
    fin_cant_rad: float,
    # Atmosphere site corrections
    site_elevation: float, t_offset: float,
    # Acceptance
    sm_subsonic_min: float, sm_supersonic_min: float,
    # Tolerances
    rtol: float, atol: float,
    # Termination
    terminate_at_apogee: bool = True,
) -> tuple[np.ndarray, np.ndarray, int,
           float, float, float, float, float,
           bool, int]:
    """Integrate 6DoF free-flight from rail exit to apogee (or ground impact).

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

    # Hardcoded stability-check constants
    sm_transition_mach = 0.91
    sm_aoa_threshold_rad = 5.0 * math.pi / 180.0

    # Tracking variables
    max_mach = 0.0
    max_aoa_deg = 0.0
    min_sm_sub = 1.0e6
    min_sm_sup = 1.0e6
    peak_alt = -y[2]
    stability_compliant = True
    violation_code = 0
    past_apogee = False
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
    _pro = prop_r_outer; _pri = prop_r_inner_0; _pl = prop_length
    _if = impulse_factor
    _mg = mach_g; _rg = re_g; _ag = alpha_g
    _coff = ca_tbl_off; _con = ca_tbl_on; _tb = t_burnout
    _cnt = cn_tbl; _cpt = cp_tbl
    _cnc = cn_comp; _cpc = cp_comp
    _hc = has_components; _cnaf = cn_alpha_fins_tbl
    _d = diameter; _rl = ref_length; _ar = A_ref; _fr = fin_cp_radius
    _wa = wind_alt; _we = wind_east; _wn = wind_north
    _fc = fin_cant_rad
    _se = site_elevation; _to = t_offset

    # Initial k1
    _sixdof_deriv(
        t, y, k1,
        _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
        _ird, _ild, _pro, _pri, _pl, _if,
        _mg, _rg, _ag, _coff, _con, _tb, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
        _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
        _se, _to,
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
            _ird, _ild, _pro, _pri, _pl, _if,
            _mg, _rg, _ag, _coff, _con, _tb, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
            _se, _to,
        )

        # Stage 3
        for j in range(N):
            ys[j] = y[j] + h_step * (DP_A2[0] * k1[j] + DP_A2[1] * k2[j])
        _sixdof_deriv(
            t + DP_C[2] * h_step, ys, k3,
            _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
            _ird, _ild, _pro, _pri, _pl, _if,
            _mg, _rg, _ag, _coff, _con, _tb, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
            _se, _to,
        )

        # Stage 4
        for j in range(N):
            ys[j] = y[j] + h_step * (
                DP_A3[0] * k1[j] + DP_A3[1] * k2[j] + DP_A3[2] * k3[j]
            )
        _sixdof_deriv(
            t + DP_C[3] * h_step, ys, k4,
            _mt, _mth, _mp0, _ti, _na, _np_, _md, _cgd, _mcl,
            _ird, _ild, _pro, _pri, _pl, _if,
            _mg, _rg, _ag, _coff, _con, _tb, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
            _se, _to,
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
            _ird, _ild, _pro, _pri, _pl, _if,
            _mg, _rg, _ag, _coff, _con, _tb, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
            _se, _to,
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
            _ird, _ild, _pro, _pri, _pl, _if,
            _mg, _rg, _ag, _coff, _con, _tb, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
            _se, _to,
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
            _ird, _ild, _pro, _pri, _pl, _if,
            _mg, _rg, _ag, _coff, _con, _tb, _cnt, _cpt, _cnc, _cpc, _hc, _cnaf,
            _d, _rl, _ar, _fr, _wa, _we, _wn, _fc,
            _se, _to,
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

            # Detect apogee passage (altitude starts decreasing)
            if y[2] > prev_rD and n > 2:
                past_apogee = True

            # Stability checks apply only during ascent (up to apogee).
            # Post-apogee attitudes are not meaningful for vehicle
            # stability assessment.
            if not past_apogee:
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

            # Termination detection
            if terminate_at_apogee:
                if past_apogee:
                    break
            else:
                # Continue to ground impact (rD >= 0)
                if y[2] >= 0.0 and n > 2:
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
def _parachute_cda(
    h: float,
    drogue_cda: float, main_cda: float, main_deploy_alt: float,
    scenario: int,
) -> float:
    """Return parachute CdA for the current altitude and scenario."""
    if scenario == SCENARIO_NOMINAL:
        if h <= main_deploy_alt:
            return drogue_cda + main_cda
        return drogue_cda
    elif scenario == SCENARIO_DROGUE_ONLY:
        return drogue_cda
    else:  # SCENARIO_PREMATURE_MAIN
        return drogue_cda + main_cda


@nb.njit(cache=True, fastmath=True)
def _descent_deriv(
    t: float, state: np.ndarray, dy: np.ndarray,
    # Wind
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    # Mass
    m: float,
    # Recovery
    drogue_cda: float, main_cda: float, main_deploy_alt: float,
    # Scenario
    scenario: int,
    # Body horizontal drag
    body_cda: float,
    # Atmosphere site corrections
    site_elevation: float, t_offset: float,
) -> None:
    """6-component derivative for parachute descent: [rN, rE, rD, vN, vE, vD].

    Vertical velocity is decelerated by parachute drag.  Horizontal
    velocity is decelerated toward the local wind by body cylinder
    crossflow drag (CdA_body = Cd_cyl · L · d).

        dvD/dt = g − (ρ · CdA_chute · vD²) / (2m)
        dvH/dt = −(ρ · CdA_body · |v_h_rel| · v_h_rel) / (2m)
    """
    h = -state[2]
    if h < 0.0:
        h = 0.0

    vN_wind, vE_wind = interpolate_wind(wind_alt, wind_east, wind_north, h)

    vN = state[3]
    vE = state[4]
    vD = state[5]
    cda = _parachute_cda(h, drogue_cda, main_cda, main_deploy_alt, scenario)

    _, _, rho, _, _ = isa_at_site(h, site_elevation, t_offset)

    # --- Vertical: parachute drag ---
    if cda > 1.0e-12 and rho > 0.0:
        drag_accel_D = rho * cda * vD * vD / (2.0 * m)
    else:
        drag_accel_D = 0.0

    # --- Horizontal: body cylinder crossflow drag ---
    vN_rel = vN - vN_wind
    vE_rel = vE - vE_wind
    v_h_rel = math.sqrt(vN_rel * vN_rel + vE_rel * vE_rel)
    if body_cda > 1.0e-12 and rho > 0.0 and v_h_rel > 1.0e-12:
        h_factor = rho * body_cda * v_h_rel / (2.0 * m)
        drag_accel_N = h_factor * vN_rel
        drag_accel_E = h_factor * vE_rel
    else:
        drag_accel_N = 0.0
        drag_accel_E = 0.0

    dy[0] = vN
    dy[1] = vE
    dy[2] = vD
    dy[3] = -drag_accel_N
    dy[4] = -drag_accel_E
    dy[5] = _G0 - drag_accel_D


@nb.njit(cache=True, fastmath=True)
def integrate_descent(
    t0: float, state0: np.ndarray,
    # Wind
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    # Mass
    m: float,
    # Recovery
    drogue_cda: float, main_cda: float, main_deploy_alt: float,
    # Scenario
    scenario: int,
    # Body horizontal drag
    body_cda: float,
    # Atmosphere site corrections
    site_elevation: float, t_offset: float,
    # Tolerances
    rtol: float, atol: float,
    # Optional early stop altitude (m AGL); 0.0 = ground
    stop_alt: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Integrate parachute descent until altitude reaches *stop_alt* (rD >= -stop_alt).

    State is [rN, rE, rD, vN, vE, vD].  Vertical velocity is
    decelerated by parachute drag; horizontal velocity is decelerated
    toward the local wind by body cylinder crossflow drag.

    Returns (t_out, y_out, sm_out, n_steps).  SM is zero throughout
    (no aerodynamic stability under parachute).
    """
    MAX_STEPS = 20000
    NS = 6
    t_out = np.empty(MAX_STEPS)
    y_out = np.empty((MAX_STEPS, NS))
    sm_out = np.zeros(MAX_STEPS, dtype=np.float64)

    y = state0.copy()
    t = t0
    h_step = 0.5
    h_min = 1.0e-3
    h_max = 5.0

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
        m, drogue_cda, main_cda, main_deploy_alt, scenario,
        body_cda, site_elevation, t_offset,
    )

    for _ in range(MAX_STEPS * 2):
        h_step = clamp_step(h_step, h_min, h_max)

        # Stage 2
        for j in range(NS):
            ys[j] = y[j] + h_step * DP_A1[0] * k1[j]
        _descent_deriv(
            t + DP_C[1] * h_step, ys, k2,
            wind_alt, wind_east, wind_north,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
            body_cda, site_elevation, t_offset,
        )

        # Stage 3
        for j in range(NS):
            ys[j] = y[j] + h_step * (DP_A2[0] * k1[j] + DP_A2[1] * k2[j])
        _descent_deriv(
            t + DP_C[2] * h_step, ys, k3,
            wind_alt, wind_east, wind_north,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
            body_cda, site_elevation, t_offset,
        )

        # Stage 4
        for j in range(NS):
            ys[j] = y[j] + h_step * (
                DP_A3[0] * k1[j] + DP_A3[1] * k2[j] + DP_A3[2] * k3[j]
            )
        _descent_deriv(
            t + DP_C[3] * h_step, ys, k4,
            wind_alt, wind_east, wind_north,
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
            body_cda, site_elevation, t_offset,
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
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
            body_cda, site_elevation, t_offset,
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
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
            body_cda, site_elevation, t_offset,
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
            m, drogue_cda, main_cda, main_deploy_alt, scenario,
            body_cda, site_elevation, t_offset,
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

            if y[2] >= -stop_alt:
                break

        factor = optimal_step_factor(err)
        h_step = clamp_step(h_step * factor, h_min, h_max)

    return t_out, y_out, sm_out, n


def _descent_mach(
    y_desc: np.ndarray,
    n_desc: int,
    site_elevation: float,
    t_offset: float,
) -> np.ndarray:
    """Compute Mach at each descent step from the integrated vD."""
    mach = np.empty(n_desc, dtype=np.float64)
    for i in range(n_desc):
        h = max(-float(y_desc[i, 2]), 0.0)
        _, _, _, a, _ = isa_at_site(h, site_elevation, t_offset)
        vD = y_desc[i, 5]
        mach[i] = vD / a if a > 0.0 else 0.0
    return mach


# ---------------------------------------------------------------------------
# Damping post-processing (plain Python — vectorised NumPy over trajectory)
# ---------------------------------------------------------------------------

def compute_damping(profile: TrajectoryProfile, params: SimParams) -> None:
    """Compute damping quantities as a post-processing pass over the ascent.

    Mutates *profile* in place, filling all damping-related fields.
    Quantities are only meaningful during the ascent phase (up to apogee);
    descent and rail values are NaN.

    Definitions (ref: "Advanced Topics in Model Rocketry" pp. 201–202):
      C1   — corrective moment coefficient:  0.5 ρ V² S_ref C_Nα (X_CP − X_CG)
      C2   — damping moment coefficient:     C2A + C2R
      C2A  — aerodynamic damping:            Σ_j 0.5 ρ V S_ref C_Nα_j (X_CPj − X_CG)²
      C2R  — jet damping (eq. 99):           ṁ (L_ne − X_CG)²
    where L_ne is the nozzle exit position (distance from nose tip).
    """
    p = params
    K = len(profile.time)
    n_comp = p.cn_alpha_comp.shape[0] if p.has_components else 0

    # Find apogee index
    apogee_idx = int(np.argmax(profile.altitude))

    # Allocate output arrays (NaN everywhere)
    c1 = np.full(K, math.nan, dtype=np.float64)
    c2 = np.full(K, math.nan, dtype=np.float64)
    c2a = np.full(K, math.nan, dtype=np.float64)
    c2r = np.full(K, math.nan, dtype=np.float64)
    zeta = np.full(K, math.nan, dtype=np.float64)
    omega_n = np.full(K, math.nan, dtype=np.float64)
    omega_d = np.full(K, math.nan, dtype=np.float64)
    max_roll_hz = np.full(K, math.nan, dtype=np.float64)

    cn_alpha_comp_buf = np.full((n_comp, K), math.nan, dtype=np.float64)
    cp_comp_buf = np.full((n_comp, K), math.nan, dtype=np.float64)
    c1_comp_buf = np.full((n_comp, K), math.nan, dtype=np.float64)
    c2a_comp_buf = np.full((n_comp, K), math.nan, dtype=np.float64)

    if not p.has_components or apogee_idx == 0:
        # Cannot compute damping without per-component data
        profile.c1 = c1
        profile.c2 = c2
        profile.c2a = c2a
        profile.c2r = c2r
        profile.zeta = zeta
        profile.omega_n = omega_n
        profile.omega_d = omega_d
        profile.max_roll_rate_hz = max_roll_hz
        profile.cn_alpha_comp = cn_alpha_comp_buf
        profile.cp_comp = cp_comp_buf
        profile.c1_comp = c1_comp_buf
        profile.c2a_comp = c2a_comp_buf
        profile.comp_names = p.comp_names
        return

    # Max CP distance clip (stability guard, 1000 mm)
    max_dist_m = 1.0

    # Roll rate characteristic radius
    r_roll = (p.diameter + p.fin_span) / 2.0

    for i in range(apogee_idx + 1):
        h = max(float(profile.altitude[i]), 0.0)
        M = float(profile.mach[i])

        _, _, rho, a, mu = isa_at_site(h, p.site_elevation, p.t_offset)
        V = M * a  # reconstruct velocity from Mach
        if V < _EPS_V:
            continue

        Re = rho * V * p.length / mu if mu > 0.0 else 0.0

        cg    = float(profile.cg[i])
        I_lat = float(profile.I_lateral[i])
        m_dot = float(profile.mdot[i])

        if math.isnan(cg) or math.isnan(I_lat) or math.isnan(m_dot):
            continue

        # Whole-vehicle CN_alpha (sum of per-component)
        cna_total = 0.0
        for j in range(n_comp):
            cna_j = cn_alpha_comp_at(p.mach_g, p.re_g, p.cn_alpha_comp, M, Re, j)
            cn_alpha_comp_buf[j, i] = cna_j
            cna_total += cna_j

        # Whole-vehicle CP (from per-component CN_alpha-weighted average)
        cp_total = 0.0
        if cna_total > 1e-9:
            for j in range(n_comp):
                cp_j = _interp3(p.mach_g, p.re_g, p.alpha_g, p.cp_comp[j], M, Re, 2.0)
                cp_comp_buf[j, i] = cp_j
                cp_total += cn_alpha_comp_buf[j, i] * cp_j
            cp_total /= cna_total
        else:
            cp_total = cg
            for j in range(n_comp):
                cp_j = _interp3(p.mach_g, p.re_g, p.alpha_g, p.cp_comp[j], M, Re, 2.0)
                cp_comp_buf[j, i] = cp_j

        # C1 — corrective moment coefficient (p. 201)
        c1_val = 0.5 * rho * V * V * p.A_ref * cna_total * (cp_total - cg)
        c1[i] = c1_val

        # C1 and C2A per-component contributions
        c2a_val = 0.0
        for j in range(n_comp):
            c_cp = cp_comp_buf[j, i]
            c_cna = cn_alpha_comp_buf[j, i]
            if math.isnan(c_cp):
                c_cp = cg
            # Stability guard: clip CP distance from CG
            c_cp = max(cg - max_dist_m, min(c_cp, cg + max_dist_m))
            lever = c_cp - cg
            # C1_j = 0.5 ρ V² S_ref CNα_j (CP_j − CG)
            c1_comp_buf[j, i] = 0.5 * rho * V * V * p.A_ref * c_cna * lever
            # C2A_j = 0.5 ρ V S_ref CNα_j (CP_j − CG)²
            contrib = 0.5 * rho * V * p.A_ref * c_cna * lever * lever
            c2a_comp_buf[j, i] = contrib
            c2a_val += contrib
        c2a[i] = c2a_val

        # C2R — jet damping moment coefficient (eq. 99): ṁ (L_ne − X_CG)²
        lever = p.nozzle_position - cg
        c2r_val = m_dot * lever * lever
        c2r[i] = c2r_val

        # C2 — total damping moment coefficient
        c2_val = c2a_val + c2r_val
        c2[i] = c2_val

        # Damping ratio
        if c1_val > 0.0 and I_lat > 0.0:
            product = c1_val * I_lat
            zeta_val = c2_val / (2.0 * math.sqrt(product))
            zeta[i] = zeta_val

            omega_n_val = math.sqrt(c1_val / I_lat)
            omega_n[i] = omega_n_val

            discriminant = 1.0 - zeta_val * zeta_val
            if discriminant > 0.0:
                omega_d[i] = omega_n_val * math.sqrt(discriminant)
            else:
                omega_d[i] = 0.0

        # Max permissible roll rate
        if r_roll > 0.0:
            max_roll_hz[i] = V / r_roll / (2.0 * math.pi)

    # Store results on profile
    profile.c1 = c1
    profile.c2 = c2
    profile.c2a = c2a
    profile.c2r = c2r
    profile.zeta = zeta
    profile.omega_n = omega_n
    profile.omega_d = omega_d
    profile.max_roll_rate_hz = max_roll_hz
    profile.cn_alpha_comp = cn_alpha_comp_buf
    profile.cp_comp = cp_comp_buf
    profile.c1_comp = c1_comp_buf
    profile.c2a_comp = c2a_comp_buf
    profile.comp_names = p.comp_names


# ---------------------------------------------------------------------------
# Profile builder (plain Python — NOT exposed outside dynamics.py)
# ---------------------------------------------------------------------------

def _build_profile(
    params: SimParams,
    # Rail phase
    rail_t: np.ndarray, rail_V: np.ndarray, rail_alt: np.ndarray, n_rail: int,
    # 6DoF ascent phase
    t_asc: np.ndarray, state_asc: np.ndarray,
    # Descent phase (may be None for ballistic)
    t_desc: np.ndarray | None, state_desc: np.ndarray | None, n_desc: int,
    # Descent scenario (for parachute CD selection)
    scenario: int,
    # Burnout time for power-on/off CA selection
    t_burnout: float = 0.0,
) -> TrajectoryProfile:
    """Build a unified :class:`TrajectoryProfile` from phase-specific arrays.

    Computes all derived quantities (Mach, AoA, SM, thrust, mass, CD)
    at each integration step using the same functions the integrator
    calls, then concatenates into a single timeline.  This is the
    **only** place where phase stitching occurs.
    """
    p = params
    diameter = p.diameter

    # --- Rail phase ---
    nr = n_rail
    rail_pos = np.empty((nr, 3), dtype=np.float64)
    eN, eE, eD = _rail_direction(p.rail_azimuth_rad, p.rail_inclination_rad)
    for i in range(nr):
        # Reconstruct NED position from rail distance implied by altitude
        if eD != 0.0:
            s_i = -rail_alt[i] / eD if rail_alt[i] > 0.0 else 0.0
        else:
            s_i = 0.0
        rail_pos[i, 0] = s_i * eN
        rail_pos[i, 1] = s_i * eE
        rail_pos[i, 2] = s_i * eD

    rail_mach = np.empty(nr, dtype=np.float64)
    rail_thrust = np.empty(nr, dtype=np.float64)
    rail_mass = np.empty(nr, dtype=np.float64)
    rail_sm = np.empty(nr, dtype=np.float64)
    rail_cd = np.empty(nr, dtype=np.float64)
    rail_aoa = np.zeros(nr, dtype=np.float64)
    rail_roll_hz = np.full(nr, math.nan, dtype=np.float64)
    rail_cg = np.empty(nr, dtype=np.float64)
    rail_I_roll = np.empty(nr, dtype=np.float64)
    rail_I_lateral = np.empty(nr, dtype=np.float64)
    rail_mdot = np.empty(nr, dtype=np.float64)

    for i in range(nr):
        ti = float(rail_t[i])
        h = max(float(rail_alt[i]), 0.0)
        V = float(rail_V[i])

        _, _, rho, a, mu = isa_at_site(h, p.site_elevation, p.t_offset)
        M = V / a if a > 0.0 else 0.0
        rail_mach[i] = M

        rail_thrust[i] = thrust_corrected_at(
            p.motor_times, p.motor_thrusts, p.nozzle_area,
            h, p.site_elevation, ti,
        )
        rail_mass[i] = mass_at(
            p.motor_times, p.motor_thrusts, p.m_prop_0,
            p.total_impulse, p.m_dry, ti,
        )
        cg = cg_at(
            p.motor_times, p.motor_thrusts, p.m_prop_0, p.total_impulse,
            p.m_dry, p.cg_dry, p.motor_cg_loaded, ti,
        )
        i_roll, i_lat = inertia_at(
            p.motor_times, p.motor_thrusts, p.m_prop_0, p.total_impulse,
            p.m_dry, p.cg_dry, p.motor_cg_loaded,
            p.I_roll_dry, p.I_lateral_dry,
            p.prop_r_outer, p.prop_r_inner_0, p.prop_length, ti,
        )
        rail_cg[i] = cg
        rail_I_roll[i] = i_roll
        rail_I_lateral[i] = i_lat
        rail_mdot[i] = mdot_at(p.motor_times, p.motor_thrusts, p.m_prop_0, p.total_impulse, ti)
        Re = rho * V * p.length / mu if V > _EPS_V and mu > 0.0 else 0.0
        # SM from whole-vehicle (no lateral flow on rail)
        _, _, _, _, _, cp_whole = aero_forces_moments(
            p.mach_g, p.re_g, p.alpha_g,
            p.ca_tbl_off, p.ca_tbl_on, ti <= t_burnout,
            p.cn_tbl, p.cp_tbl,
            p.cn_comp, p.cp_comp, p.has_components,
            M, Re, rho, V, p.A_ref,
            V, 0.0, 0.0, 0.0, 0.0, cg,
        )
        rail_sm[i] = (cp_whole - cg) / diameter if diameter > 0.0 else 0.0

        if M >= p.mach_g[0]:
            ca_tbl = p.ca_tbl_on if ti <= t_burnout else p.ca_tbl_off
            ca = ca_at(p.mach_g, p.re_g, p.alpha_g, ca_tbl, M, Re, 0.0)
            rail_cd[i] = ca
        else:
            rail_cd[i] = math.nan

    # --- 6DoF ascent phase ---
    n_asc = len(t_asc)
    asc_pos = state_asc[:, :3].copy()               # [rN, rE, rD]
    asc_alt = -state_asc[:, 2].copy()

    asc_mach = np.empty(n_asc, dtype=np.float64)
    asc_thrust = np.empty(n_asc, dtype=np.float64)
    asc_mass = np.empty(n_asc, dtype=np.float64)
    asc_sm = np.empty(n_asc, dtype=np.float64)
    asc_cd = np.empty(n_asc, dtype=np.float64)
    asc_aoa = np.empty(n_asc, dtype=np.float64)
    asc_cg = np.empty(n_asc, dtype=np.float64)
    asc_I_roll = np.empty(n_asc, dtype=np.float64)
    asc_I_lateral = np.empty(n_asc, dtype=np.float64)
    asc_mdot = np.empty(n_asc, dtype=np.float64)
    # Roll rate: state index 10 is p_rate [rad/s], convert to Hz
    _RAD_PER_S_TO_HZ = 1.0 / (2.0 * math.pi)
    asc_roll_hz = np.empty(n_asc, dtype=np.float64)
    for i in range(n_asc):
        asc_roll_hz[i] = float(state_asc[i, 10]) * _RAD_PER_S_TO_HZ

    for i in range(n_asc):
        ti = float(t_asc[i])
        h = max(float(asc_alt[i]), 0.0)

        _, _, rho, a, mu = isa_at_site(h, p.site_elevation, p.t_offset)

        u = float(state_asc[i, 7])
        v = float(state_asc[i, 8])
        w = float(state_asc[i, 9])
        V = math.sqrt(u * u + v * v + w * w)

        M = V / a if a > 0.0 else 0.0
        asc_mach[i] = M

        alpha_rad = math.atan2(math.sqrt(v * v + w * w), u) if V > _EPS_V else 0.0
        asc_aoa[i] = alpha_rad * _RAD2DEG

        asc_thrust[i] = thrust_corrected_at(
            p.motor_times, p.motor_thrusts, p.nozzle_area,
            h, p.site_elevation, ti,
        )
        asc_mass[i] = mass_at(
            p.motor_times, p.motor_thrusts, p.m_prop_0,
            p.total_impulse, p.m_dry, ti,
        )
        cg = cg_at(
            p.motor_times, p.motor_thrusts, p.m_prop_0, p.total_impulse,
            p.m_dry, p.cg_dry, p.motor_cg_loaded, ti,
        )
        i_roll, i_lat = inertia_at(
            p.motor_times, p.motor_thrusts, p.m_prop_0, p.total_impulse,
            p.m_dry, p.cg_dry, p.motor_cg_loaded,
            p.I_roll_dry, p.I_lateral_dry,
            p.prop_r_outer, p.prop_r_inner_0, p.prop_length, ti,
        )
        m_dot = mdot_at(p.motor_times, p.motor_thrusts, p.m_prop_0, p.total_impulse, ti)
        asc_cg[i] = cg
        asc_I_roll[i] = i_roll
        asc_I_lateral[i] = i_lat
        asc_mdot[i] = m_dot

        Re = rho * V * p.length / mu if V > _EPS_V and mu > 0.0 else 0.0
        q_rate = float(state_asc[i, 11])
        r_rate = float(state_asc[i, 12])

        power_on = ti <= t_burnout
        _, _, _, _, _, cp_whole = aero_forces_moments(
            p.mach_g, p.re_g, p.alpha_g,
            p.ca_tbl_off, p.ca_tbl_on, power_on,
            p.cn_tbl, p.cp_tbl,
            p.cn_comp, p.cp_comp, p.has_components,
            M, Re, rho, V, p.A_ref,
            u, v, w, q_rate, r_rate, cg,
        )
        asc_sm[i] = (cp_whole - cg) / diameter if diameter > 0.0 else 0.0

        if M >= p.mach_g[0]:
            ca_tbl = p.ca_tbl_on if power_on else p.ca_tbl_off
            ca = ca_at(p.mach_g, p.re_g, p.alpha_g, ca_tbl, M, Re, alpha_rad)
            cn, _ = cn_cp_at(
                p.mach_g, p.re_g, p.alpha_g,
                p.cn_tbl, p.cp_tbl, M, Re, alpha_rad,
            )
            asc_cd[i] = ca * math.cos(alpha_rad) + cn * math.sin(alpha_rad)
        else:
            asc_cd[i] = math.nan

    # --- Descent phase ---
    if t_desc is not None and n_desc > 0:
        desc_pos = state_desc[:n_desc, :3].copy()
        desc_alt = -state_desc[:n_desc, 2].copy()

        desc_mach = np.empty(n_desc, dtype=np.float64)
        desc_sm = np.zeros(n_desc, dtype=np.float64)
        desc_thrust = np.zeros(n_desc, dtype=np.float64)
        desc_mass = np.full(n_desc, p.m_dry, dtype=np.float64)
        desc_aoa = np.full(n_desc, math.nan, dtype=np.float64)
        desc_roll_hz = np.full(n_desc, math.nan, dtype=np.float64)
        desc_cg = np.full(n_desc, p.cg_dry, dtype=np.float64)
        desc_I_roll = np.full(n_desc, p.I_roll_dry, dtype=np.float64)
        desc_I_lateral = np.full(n_desc, p.I_lateral_dry, dtype=np.float64)
        desc_mdot = np.zeros(n_desc, dtype=np.float64)
        desc_cd = np.empty(n_desc, dtype=np.float64)

        for i in range(n_desc):
            h = max(float(desc_alt[i]), 0.0)
            _, _, _, a, _ = isa_at_site(h, p.site_elevation, p.t_offset)
            vD = float(state_desc[i, 5])
            desc_mach[i] = abs(vD) / a if a > 0.0 else 0.0
            # Parachute CD from config (constant per chute, switches at main deploy)
            cda = _parachute_cda(
                h, p.drogue_cda, p.main_cda, p.main_deploy_alt, scenario,
            )
            desc_cd[i] = cda / p.A_ref if p.A_ref > 0.0 else 0.0

        t_desc_trimmed = t_desc[:n_desc]
    else:
        n_desc = 0

    # --- Stitch into unified arrays (skip overlap at phase boundaries) ---
    # Rail → 6DoF: skip first 6DoF point (same as rail exit)
    # 6DoF → Descent: skip first descent point (same as apogee)
    parts_t = [rail_t[:nr]]
    parts_pos = [rail_pos]
    parts_alt = [np.array([max(float(rail_alt[i]), 0.0) for i in range(nr)])]
    parts_mach = [rail_mach]
    parts_aoa = [rail_aoa]
    parts_sm = [rail_sm]
    parts_thrust = [rail_thrust]
    parts_mass = [rail_mass]
    parts_cd = [rail_cd]
    parts_roll_hz = [rail_roll_hz]
    parts_cg = [rail_cg]
    parts_I_roll = [rail_I_roll]
    parts_I_lateral = [rail_I_lateral]
    parts_mdot = [rail_mdot]

    # Skip first ascent point to avoid overlap with last rail point
    if n_asc > 1:
        parts_t.append(t_asc[1:])
        parts_pos.append(asc_pos[1:])
        parts_alt.append(asc_alt[1:])
        parts_mach.append(asc_mach[1:])
        parts_aoa.append(asc_aoa[1:])
        parts_sm.append(asc_sm[1:])
        parts_thrust.append(asc_thrust[1:])
        parts_mass.append(asc_mass[1:])
        parts_cd.append(asc_cd[1:])
        parts_roll_hz.append(asc_roll_hz[1:])
        parts_cg.append(asc_cg[1:])
        parts_I_roll.append(asc_I_roll[1:])
        parts_I_lateral.append(asc_I_lateral[1:])
        parts_mdot.append(asc_mdot[1:])
    elif n_asc == 1:
        # Single-point ascent — still include it
        parts_t.append(t_asc)
        parts_pos.append(asc_pos)
        parts_alt.append(asc_alt)
        parts_mach.append(asc_mach)
        parts_aoa.append(asc_aoa)
        parts_sm.append(asc_sm)
        parts_thrust.append(asc_thrust)
        parts_mass.append(asc_mass)
        parts_cd.append(asc_cd)
        parts_roll_hz.append(asc_roll_hz)
        parts_cg.append(asc_cg)
        parts_I_roll.append(asc_I_roll)
        parts_I_lateral.append(asc_I_lateral)
        parts_mdot.append(asc_mdot)

    if n_desc > 1:
        parts_t.append(t_desc_trimmed[1:])
        parts_pos.append(desc_pos[1:])
        parts_alt.append(desc_alt[1:])
        parts_mach.append(desc_mach[1:])
        parts_aoa.append(desc_aoa[1:])
        parts_sm.append(desc_sm[1:])
        parts_thrust.append(desc_thrust[1:])
        parts_mass.append(desc_mass[1:])
        parts_cd.append(desc_cd[1:])
        parts_roll_hz.append(desc_roll_hz[1:])
        parts_cg.append(desc_cg[1:])
        parts_I_roll.append(desc_I_roll[1:])
        parts_I_lateral.append(desc_I_lateral[1:])
        parts_mdot.append(desc_mdot[1:])

    return TrajectoryProfile(
        time=np.concatenate(parts_t),
        position_ned=np.concatenate(parts_pos, axis=0),
        altitude=np.concatenate(parts_alt),
        mach=np.concatenate(parts_mach),
        aoa_deg=np.concatenate(parts_aoa),
        sm=np.concatenate(parts_sm),
        thrust=np.concatenate(parts_thrust),
        mass=np.concatenate(parts_mass),
        cd=np.concatenate(parts_cd),
        roll_rate_hz=np.concatenate(parts_roll_hz),
        cg=np.concatenate(parts_cg),
        I_roll=np.concatenate(parts_I_roll),
        I_lateral=np.concatenate(parts_I_lateral),
        mdot=np.concatenate(parts_mdot),
    )


# ---------------------------------------------------------------------------
# Top-level trajectory runner (plain Python)
# ---------------------------------------------------------------------------

def run_trajectory(
    params: SimParams,
    scenario: int,
    poly_e: np.ndarray | None = None,
    poly_n: np.ndarray | None = None,
    buffered_ceiling: float = float('inf'),
    keep_profile: bool = False,
) -> FlightSummary | tuple[FlightSummary, TrajectoryProfile]:
    """Run a complete trajectory: rail → 6DoF ascent → descent.

    For ballistic scenarios the 6DoF integrator continues past apogee to
    ground impact.  For parachute scenarios, the 6DoF ends at apogee and
    a simplified 3DoF descent (wind-drift + terminal velocity) follows.

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
    keep_profile : bool
        If *True*, also return a :class:`TrajectoryProfile` with the
        full unified time history.  The default (*False*) skips all
        array allocation for the profile, keeping the MC hot path lean.

    Returns
    -------
    FlightSummary
        Always returned.
    tuple[FlightSummary, TrajectoryProfile]
        When *keep_profile* is *True*.
    """
    p = params
    t_burnout = float(p.motor_times[-1])

    # ---- Phase 1: Launch rail ----
    (V_exit, t_exit, rN_exit, rE_exit, rD_exit,
     rail_t_hist, rail_V_hist, rail_alt_hist, rail_n) = simulate_rail(
        p.rail_azimuth_rad, p.rail_inclination_rad, p.rail_length,
        p.motor_times, p.motor_thrusts, p.nozzle_area, p.impulse_factor,
        p.m_prop_0, p.total_impulse, p.m_dry,
        p.mach_g, p.re_g, p.alpha_g,
        p.ca_tbl_off, p.ca_tbl_on, t_burnout,
        p.A_ref, p.length,
        p.site_elevation, p.t_offset,
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
    is_ballistic = scenario == SCENARIO_BALLISTIC
    (t_hist, state_hist, n_steps,
     max_mach, max_aoa_deg, min_sm_sub, min_sm_sup, peak_alt,
     stab_ok, viol_code) = integrate_sixdof(
        t_exit, state0, t_burnout,
        p.motor_times, p.motor_thrusts, p.m_prop_0, p.total_impulse,
        p.nozzle_area, p.nozzle_position, p.m_dry, p.cg_dry,
        p.motor_cg_loaded,
        p.I_roll_dry, p.I_lateral_dry,
        p.prop_r_outer, p.prop_r_inner_0, p.prop_length,
        p.impulse_factor,
        p.mach_g, p.re_g, p.alpha_g,
        p.ca_tbl_off, p.ca_tbl_on,
        p.cn_tbl, p.cp_tbl,
        p.cn_comp, p.cp_comp, p.has_components, p.cn_alpha_fins,
        p.diameter, p.length, p.A_ref, p.fin_cp_radius,
        p.wind_alt, p.wind_east, p.wind_north,
        p.fin_cant_rad,
        p.site_elevation, p.t_offset,
        p.sm_subsonic_min, p.sm_supersonic_min,
        p.rtol, p.atol,
        terminate_at_apogee=not is_ballistic,
    )

    t_full = t_hist[:n_steps].copy()
    s_full = state_hist[:n_steps].copy()

    # ---- Identify apogee ----
    apogee_alt = peak_alt
    if is_ballistic:
        altitudes = -s_full[:, 2]
        apogee_idx = int(np.argmax(altitudes))
    else:
        apogee_idx = n_steps - 1

    apogee_state = s_full[apogee_idx]
    apogee_t = t_full[apogee_idx]
    apogee_pos = apogee_state[:3].copy()

    # Split at apogee for parachute scenarios; keep full history for ballistic
    if is_ballistic:
        t_asc = t_full
        s_asc = s_full
    else:
        t_asc = t_full[:apogee_idx + 1]
        s_asc = s_full[:apogee_idx + 1]

    # ---- Geofence checks ----
    _ceiling_ok = apogee_alt <= buffered_ceiling
    _footprint_ok = True

    if poly_e is not None and poly_n is not None:
        from geography import all_points_in_polygon
        _footprint_ok = all_points_in_polygon(
            s_full[:, 0], s_full[:, 1], poly_e, poly_n,
        )

    # Helper to build FlightSummary from current local state
    def _make_summary(
        landing_pos: np.ndarray, landing_t: float,
    ) -> FlightSummary:
        return FlightSummary(
            apogee=apogee_alt,
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
            footprint_compliant=_footprint_ok,
            ceiling_compliant=_ceiling_ok,
            stability_compliant=stab_ok,
        )

    # Helper to build profile when requested
    def _maybe_profile(
        t_desc: np.ndarray | None = None,
        state_desc: np.ndarray | None = None,
        n_desc: int = 0,
    ) -> TrajectoryProfile | None:
        if not keep_profile:
            return None
        return _build_profile(
            p,
            rail_t_hist[:rail_n], rail_V_hist[:rail_n],
            rail_alt_hist[:rail_n], rail_n,
            t_asc, s_asc,
            t_desc, state_desc, n_desc,
            scenario,
            t_burnout,
        )

    if is_ballistic:
        # ---- Ballistic: 6DoF ran to ground impact — no 3DoF descent ----
        landing_pos = s_full[n_steps - 1, :3].copy()
        landing_t = t_full[n_steps - 1]
        summary = _make_summary(landing_pos, landing_t)
        profile = _maybe_profile()
        return (summary, profile) if keep_profile else summary

    # ---- Parachute scenarios: 3DoF descent from apogee ----

    if not _ceiling_ok or not _footprint_ok or not stab_ok:
        # Skip descent — sample is already non-compliant.
        summary = _make_summary(apogee_pos, apogee_t)
        profile = _maybe_profile()
        return (summary, profile) if keep_profile else summary

    drogue_cda = p.drogue_cda if p.has_drogue else 0.0
    main_cda = p.main_cda if p.has_main else 0.0
    main_deploy_alt = p.main_deploy_alt

    effective_scenario = scenario
    if scenario == SCENARIO_NOMINAL and main_deploy_alt < 0.0:
        effective_scenario = SCENARIO_PREMATURE_MAIN

    # Extract NED velocity at apogee from the 6DoF state.
    C_nb = quat_to_dcm_nb(
        apogee_state[3], apogee_state[4],
        apogee_state[5], apogee_state[6],
    )
    u_ap, v_ap, w_ap = apogee_state[7], apogee_state[8], apogee_state[9]
    vN_ap = C_nb[0, 0] * u_ap + C_nb[0, 1] * v_ap + C_nb[0, 2] * w_ap
    vE_ap = C_nb[1, 0] * u_ap + C_nb[1, 1] * v_ap + C_nb[1, 2] * w_ap

    descent_state0 = np.array([
        apogee_pos[0], apogee_pos[1], apogee_pos[2],
        vN_ap, vE_ap, 0.0,
    ], dtype=np.float64)
    body_cda = _CD_CYLINDER * p.length * p.diameter

    # ---- Phase 3: parachute descent ----
    if (effective_scenario == SCENARIO_NOMINAL
            and main_deploy_alt > 0.0
            and p.has_main and p.has_drogue):
        # Leg 1: apogee → main deploy altitude (drogue only)
        t1, y1, sm1, n1 = integrate_descent(
            apogee_t, descent_state0,
            p.wind_alt, p.wind_east, p.wind_north,
            p.m_dry, drogue_cda, 0.0, 0.0,
            SCENARIO_DROGUE_ONLY,
            body_cda, p.site_elevation, p.t_offset,
            p.rtol, p.atol,
            stop_alt=main_deploy_alt,
        )
        # Leg 2: main deploy → ground (drogue + main)
        leg2_state = y1[n1 - 1].copy()
        leg2_t0 = t1[n1 - 1]
        t2, y2, sm2, n2 = integrate_descent(
            leg2_t0, leg2_state,
            p.wind_alt, p.wind_east, p.wind_north,
            p.m_dry, drogue_cda, main_cda, main_deploy_alt,
            SCENARIO_PREMATURE_MAIN,
            body_cda, p.site_elevation, p.t_offset,
            p.rtol, p.atol,
        )
        t_desc = np.concatenate((t1[:n1], t2[1:n2]))
        y_desc = np.concatenate((y1[:n1], y2[1:n2]))
        n_desc = n1 + n2 - 1
    else:
        t_desc, y_desc, _, n_desc = integrate_descent(
            apogee_t, descent_state0,
            p.wind_alt, p.wind_east, p.wind_north,
            p.m_dry, drogue_cda, main_cda, main_deploy_alt,
            effective_scenario,
            body_cda, p.site_elevation, p.t_offset,
            p.rtol, p.atol,
        )

    land_idx = n_desc - 1
    landing_pos = y_desc[land_idx, :3].copy()
    landing_t = t_desc[land_idx]

    # ---- Descent footprint check ----
    if poly_e is not None and poly_n is not None:
        from geography import all_points_in_polygon
        if not all_points_in_polygon(
            y_desc[:n_desc, 0], y_desc[:n_desc, 1], poly_e, poly_n,
        ):
            _footprint_ok = False

    summary = _make_summary(landing_pos, landing_t)
    profile = _maybe_profile(t_desc, y_desc, n_desc)
    return (summary, profile) if keep_profile else summary
