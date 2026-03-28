"""Motor model: time-varying thrust, mass, CG, and moments of inertia.

Dry vehicle properties (m_dry, cg_dry, I_roll_dry, I_lateral_dry) are derived
in build_motor_model from the wet vehicle properties and the propellant mass and
inertias provided in vehicle.yaml.  The user never has to specify dry values.

An inside-out burn model is assumed: the propellant CG stays fixed at
``motor_cg_loaded`` throughout the burn, and propellant inertias scale
linearly with remaining propellant fraction.  Instantaneous inertias are
computed via the parallel-axis theorem at each timestep.

Thrust correction for altitude:

    F(h) = F₀ + Aₑ · (p₀ − p_ISA(h))

where Aₑ = π·dₑ²/4 and p₀ = 101 325 Pa.  Use thrust_corrected_at in the
dynamics hot loop; thrust_at returns the raw (sea-level) thrust curve value.

Public API
----------
build_motor_model(motor_data, vehicle_cfg)  →  MotorModel

@njit functions — call directly in the dynamics hot loop:
    thrust_at(times, thrusts, t)                              →  float  [N]
    thrust_corrected_at(times, thrusts, nozzle_area, alt, t)  →  float  [N]
    mdot_at(times, thrusts, m_prop_0, total_impulse, t)       →  float  [kg/s]
    m_prop_at(times, thrusts, m_prop_0, total_impulse, t)     →  float  [kg]
    mass_at(times, thrusts, m_prop_0, total_impulse,
            m_dry, t)                                         →  float  [kg]
    cg_at(times, thrusts, m_prop_0, total_impulse,
          m_dry, cg_dry, motor_cg_loaded, t)                  →  float  [m]
    inertia_at(times, thrusts, m_prop_0, total_impulse,
               m_dry, cg_dry, motor_cg_loaded,
               I_roll_dry, I_lateral_dry,
               prop_I_roll, prop_I_lateral, t)                →  (I_roll, I_lat)

MotorModel bundles all scalars/arrays needed to call the @njit functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numba as nb

from config import MotorData, VehicleConfig


# ---------------------------------------------------------------------------
# Pre-computed bundle
# ---------------------------------------------------------------------------

@dataclass
class MotorModel:
    """All motor data pre-processed and ready for the simulation hot loop.

    Construct via :func:`build_motor_model`.
    """
    # Thrust curve (contiguous float64 arrays)
    times: np.ndarray        # (K,) [s]
    thrusts: np.ndarray      # (K,) [N]  — sea-level values

    # Scalar motor properties
    m_prop_0: float          # initial propellant mass [kg]
    m_casing: float          # motor casing mass = m_motor − m_prop_0 [kg]
    total_impulse: float     # ∫F dt [N·s]
    nozzle_area: float       # π·dₑ²/4 [m²] — for pressure thrust correction

    # Dry vehicle properties (derived from wet + propellant in build_motor_model)
    m_dry: float             # dry vehicle mass [kg]
    cg_dry: float            # dry vehicle CG from nosecone [m]
    motor_cg_loaded: float   # propellant CG (constant, inside-out burn) [m]

    # Inertias
    I_roll_dry: float        # dry roll inertia about roll axis [kg·m²]
    I_lateral_dry: float     # dry lateral inertia about dry CG [kg·m²]
    prop_I_roll: float       # propellant roll inertia about roll axis [kg·m²]
    prop_I_lateral: float    # propellant lateral inertia about propellant CG [kg·m²]


def build_motor_model(motor_data: MotorData, vehicle_cfg: VehicleConfig) -> MotorModel:
    """Construct a MotorModel, deriving dry vehicle properties from wet + propellant.

    Raises
    ------
    ValueError
        If total impulse ≤ 0 or the computed dry mass is non-positive.
    """
    times = np.ascontiguousarray(motor_data.time_s, dtype=np.float64)
    thrusts = np.ascontiguousarray(motor_data.thrust_n, dtype=np.float64)
    total_impulse = float(np.trapz(thrusts, times))

    if total_impulse <= 0:
        raise ValueError("Total impulse must be > 0")

    m_casing = motor_data.m_motor_kg - motor_data.m_prop_kg
    m_prop_0 = motor_data.m_prop_kg
    mass = vehicle_cfg.mass
    geom = vehicle_cfg.geometry

    # --- dry mass
    m_dry = mass.wet_mass - m_prop_0
    if m_dry <= 0:
        raise ValueError(
            f"Computed dry mass {m_dry:.3f} kg must be > 0 "
            f"(wet_mass={mass.wet_mass} kg, m_prop={m_prop_0} kg)"
        )

    # --- dry CG
    # wet_mass·wet_cg = m_dry·cg_dry + m_prop·wet_motor_cg
    cg_dry = (mass.wet_mass * mass.wet_cg - m_prop_0 * mass.wet_motor_cg) / m_dry

    # --- dry roll inertia
    # Roll axis is the symmetry axis, so propellant contribution needs no PAT.
    I_roll_dry = mass.wet_inertia_roll - mass.propellant_I_roll

    # --- dry lateral inertia
    # Step 1: transfer propellant inertia from propellant CG → wet vehicle CG
    d_prop_wet = mass.wet_motor_cg - mass.wet_cg
    I_prop_lat_at_wet_cg = mass.propellant_I_lateral + m_prop_0 * d_prop_wet ** 2
    # Step 2: dry lateral inertia about wet vehicle CG
    I_lat_dry_at_wet_cg = mass.wet_inertia_lateral - I_prop_lat_at_wet_cg
    # Step 3: transfer to dry vehicle CG
    d_dry_wet = cg_dry - mass.wet_cg
    I_lateral_dry = I_lat_dry_at_wet_cg - m_dry * d_dry_wet ** 2

    return MotorModel(
        times=times,
        thrusts=thrusts,
        m_prop_0=m_prop_0,
        m_casing=m_casing,
        total_impulse=total_impulse,
        nozzle_area=geom.nozzle_area,
        m_dry=m_dry,
        cg_dry=cg_dry,
        motor_cg_loaded=mass.wet_motor_cg,
        I_roll_dry=I_roll_dry,
        I_lateral_dry=I_lateral_dry,
        prop_I_roll=mass.propellant_I_roll,
        prop_I_lateral=mass.propellant_I_lateral,
    )


# ---------------------------------------------------------------------------
# Numba hot-loop helpers
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _interp(times: np.ndarray, values: np.ndarray, t: float) -> float:
    """Linear interpolation into a monotone time series; clamped at ends."""
    n = times.shape[0]
    if t <= times[0]:
        return values[0]
    if t >= times[n - 1]:
        return values[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    alpha = (t - times[lo]) / (times[hi] - times[lo])
    return values[lo] + alpha * (values[hi] - values[lo])


@nb.njit(cache=True)
def _isa_pressure(altitude_m: float) -> float:
    """ISA ambient static pressure [Pa] at geometric altitude [m].

    Troposphere (0–11 km) and lower stratosphere (11–20 km).
    Returns sea-level pressure for altitudes ≤ 0.
    """
    T0 = 288.15    # K   sea-level temperature
    p0 = 101325.0  # Pa  sea-level pressure
    L  = 0.0065    # K/m tropospheric lapse rate
    R  = 287.058   # J/(kg·K)
    g  = 9.80665   # m/s²

    if altitude_m <= 0.0:
        return p0
    if altitude_m < 11000.0:
        T = T0 - L * altitude_m
        return p0 * (T / T0) ** (g / (L * R))
    # Lower stratosphere — isothermal at T₁₁ = 216.65 K
    T_11 = T0 - L * 11000.0
    p_11 = p0 * (T_11 / T0) ** (g / (L * R))
    return p_11 * math.exp(-g * (altitude_m - 11000.0) / (R * T_11))


# ---------------------------------------------------------------------------
# Numba hot-loop functions — public API
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def thrust_at(times: np.ndarray, thrusts: np.ndarray, t: float) -> float:
    """Sea-level thrust [N] at time *t* [s]."""
    return _interp(times, thrusts, t)


@nb.njit(cache=True)
def thrust_corrected_at(
    times: np.ndarray,
    thrusts: np.ndarray,
    nozzle_area: float,
    altitude_m: float,
    t: float,
) -> float:
    """Altitude-corrected thrust [N] at time *t* and altitude *altitude_m*.

    The .eng thrust curve is assumed to be measured at sea level.  Thrust
    increases as ambient back-pressure falls:

        F(h) = F₀ + Aₑ · (p₀ − p_ISA(h))
    """
    F0 = _interp(times, thrusts, t)
    delta_p = 101325.0 - _isa_pressure(altitude_m)
    return F0 + nozzle_area * delta_p


@nb.njit(cache=True)
def mdot_at(
    times: np.ndarray,
    thrusts: np.ndarray,
    m_prop_0: float,
    total_impulse: float,
    t: float,
) -> float:
    """Propellant mass flow rate ṁ [kg/s] at time *t*.

    ṁ(t) = m_prop_0 · F(t) / I_total   (§7.3)
    """
    return m_prop_0 * _interp(times, thrusts, t) / total_impulse


@nb.njit(cache=True)
def m_prop_at(
    times: np.ndarray,
    thrusts: np.ndarray,
    m_prop_0: float,
    total_impulse: float,
    t: float,
) -> float:
    """Remaining propellant mass [kg] at time *t*.

    m_prop(t) = m_prop_0 · (1 − I(t) / I_total)
    """
    if t <= times[0]:
        return m_prop_0
    if t >= times[times.shape[0] - 1]:
        return 0.0

    impulse_so_far = 0.0
    n = times.shape[0]
    for i in range(1, n):
        if times[i] <= t:
            impulse_so_far += 0.5 * (thrusts[i - 1] + thrusts[i]) * (times[i] - times[i - 1])
        else:
            f_t = thrusts[i - 1] + (thrusts[i] - thrusts[i - 1]) * (t - times[i - 1]) / (times[i] - times[i - 1])
            impulse_so_far += 0.5 * (thrusts[i - 1] + f_t) * (t - times[i - 1])
            break

    prop_fraction_remaining = 1.0 - impulse_so_far / total_impulse
    if prop_fraction_remaining < 0.0:
        prop_fraction_remaining = 0.0
    return m_prop_0 * prop_fraction_remaining


@nb.njit(cache=True)
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


@nb.njit(cache=True)
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


@nb.njit(cache=True)
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
    prop_I_roll: float,
    prop_I_lateral: float,
    t: float,
) -> tuple[float, float]:
    """Whole-vehicle (I_roll, I_lateral) [kg·m²] at time *t*.

    Inside-out burn: propellant CG stays at ``motor_cg_loaded`` and propellant
    inertias scale linearly with remaining propellant fraction.  The parallel-
    axis theorem is applied to transfer each mass contribution to the
    instantaneous vehicle CG.
    """
    m_p = m_prop_at(times, thrusts, m_prop_0, total_impulse, t)
    prop_frac = m_p / m_prop_0   # 1 at ignition → 0 at burnout

    # Current vehicle CG
    cg_t = (m_dry * cg_dry + m_p * motor_cg_loaded) / (m_dry + m_p)

    # Roll: no PAT needed (propellant is symmetric about roll axis)
    I_roll = I_roll_dry + prop_I_roll * prop_frac

    # Lateral — dry contribution about current CG
    d_dry = cg_dry - cg_t
    I_lat_dry = I_lateral_dry + m_dry * d_dry * d_dry

    # Lateral — propellant contribution about current CG (via PAT from propellant CG)
    d_prop = motor_cg_loaded - cg_t
    I_lat_prop = prop_I_lateral * prop_frac + m_p * d_prop * d_prop

    return I_roll, I_lat_dry + I_lat_prop
