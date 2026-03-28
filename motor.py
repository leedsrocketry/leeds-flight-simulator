"""Motor model: time-varying thrust, propellant mass, and moments of inertia.

The motor knows about itself — thrust curve, propellant consumption, and MoI
evolution during the burn.  It does not know where it sits in the vehicle.

Vehicle CG is assembled in dynamics.py using:

    CG(t) = (m_dry · cg_dry  +  m_prop(t) · motor_cg_loaded)
            ─────────────────────────────────────────────────
                        m_dry + m_prop(t)

where ``motor_cg_loaded`` (from vehicle.yaml) serves as the effective propellant
CG.  This is exact when the motor casing is massless and an excellent
approximation for typical high-mass-fraction solid motors.

Public API
----------
build_motor_model(motor_data, vehicle_cfg)  →  MotorModel

@njit functions — call directly in the dynamics hot loop:
    thrust_at(times, thrusts, t)                          →  float  [N]
    mdot_at(times, thrusts, m_prop_0, total_impulse, t)   →  float  [kg/s]
    m_prop_at(times, thrusts, m_prop_0, total_impulse, t) →  float  [kg]
    inertia_at(times, thrusts, m_prop_0, total_impulse,
               I_R_wet, I_R_dry, I_L_wet, I_L_dry, t)    →  (I_R, I_L) [kg·m²]

MotorModel bundles all arrays/scalars needed to call the @njit functions.
"""

from __future__ import annotations

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
    thrusts: np.ndarray      # (K,) [N]

    # Scalar motor properties
    m_prop_0: float          # initial propellant mass [kg]
    m_casing: float          # casing mass = m_motor_wet − m_prop_0 [kg]
    total_impulse: float     # ∫F dt [N·s], computed from thrust curve

    # Whole-vehicle MoI endpoints for linear interpolation
    I_R_wet: float           # [kg·m²]
    I_R_dry: float           # [kg·m²]
    I_L_wet: float           # [kg·m²]
    I_L_dry: float           # [kg·m²]


def build_motor_model(motor_data: MotorData, vehicle_cfg: VehicleConfig) -> MotorModel:
    """Construct a MotorModel from parsed input data.

    Computes total impulse by trapezoidal integration of the thrust curve.
    """
    times = np.ascontiguousarray(motor_data.time_s, dtype=np.float64)
    thrusts = np.ascontiguousarray(motor_data.thrust_n, dtype=np.float64)
    total_impulse = float(np.trapz(thrusts, times))

    if total_impulse <= 0:
        raise ValueError("Total impulse must be > 0")

    m_casing = motor_data.m_motor_kg - motor_data.m_prop_kg

    return MotorModel(
        times=times,
        thrusts=thrusts,
        m_prop_0=motor_data.m_prop_kg,
        m_casing=m_casing,
        total_impulse=total_impulse,
        I_R_wet=vehicle_cfg.inertia.I_R_wet,
        I_R_dry=vehicle_cfg.inertia.I_R_dry,
        I_L_wet=vehicle_cfg.inertia.I_L_wet,
        I_L_dry=vehicle_cfg.inertia.I_L_dry,
    )


# ---------------------------------------------------------------------------
# Numba hot-loop functions
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
def thrust_at(times: np.ndarray, thrusts: np.ndarray, t: float) -> float:
    """Thrust [N] at time *t* [s].  Zero outside the thrust curve."""
    return _interp(times, thrusts, t)


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

    I(t) is computed by trapezoidal integration of the thrust curve up to *t*.
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
def inertia_at(
    times: np.ndarray,
    thrusts: np.ndarray,
    m_prop_0: float,
    total_impulse: float,
    I_R_wet: float,
    I_R_dry: float,
    I_L_wet: float,
    I_L_dry: float,
    t: float,
) -> tuple[float, float]:
    """Whole-vehicle (I_R, I_L) [kg·m²] at time *t*.

    Linearly interpolated between wet and dry values by propellant fraction
    remaining, per §7.3.
    """
    mp = m_prop_at(times, thrusts, m_prop_0, total_impulse, t)
    prop_fraction = mp / m_prop_0
    I_R = I_R_dry + (I_R_wet - I_R_dry) * prop_fraction
    I_L = I_L_dry + (I_L_wet - I_L_dry) * prop_fraction
    return I_R, I_L
