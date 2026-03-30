"""Propellant model: thrust curve, motor geometry, and propellant-level helpers.

Motor CG is assumed at the geometric centre of the motor, which is flush
with the aft end of the vehicle: ``motor_cg = vehicle_length − motor_length/2``.
The nozzle exit plane is at ``vehicle_length``.

Propellant is modelled as an annular cylinder (outer radius from the .eng
diameter minus optional casing thickness, inner radius from an optional bore
diameter).  During the burn the inner radius grows outward following the mass
flow rate (standard BATES grain assumption).  Propellant moments of inertia
are recomputed from the current annular geometry at each timestep.

Dry vehicle properties (m_dry, cg_dry, I_roll_dry, I_lateral_dry) are derived
in config.py when the ``Vehicle`` dataclass is constructed.

Time-varying vehicle properties (mass, CG, inertia, altitude-corrected thrust)
are computed by functions in dynamics.py.

Public API
----------
Data:
    MotorData                                               — raw .eng parse
    PropellantModel                                         — derived model

Loaders:
    load_motor(path)                                        → MotorData

Builders:
    build_propellant_model(motor_data, vehicle_length,
                           nozzle_area, casing_thickness,
                           propellant_inner_diameter)        → PropellantModel

@njit functions — propellant-level only:
    thrust_at(times, thrusts, t)                            → float  [N]
    mdot_at(times, thrusts, m_prop_0, total_impulse, t)     → float  [kg/s]
    m_prop_at(times, thrusts, m_prop_0, total_impulse, t)   → float  [kg]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numba as nb


# ---------------------------------------------------------------------------
# Motor data (raw parse of .eng file)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorData:
    """Raw data parsed from a RASP .eng file.

    Masses are in kg, time in seconds, thrust in Newtons.
    ``m_motor_kg`` is the total motor mass (casing + propellant) as stated in
    the .eng header (\"total weight\" field).  ``diameter_m`` and ``length_m``
    are the motor's outer diameter and length, converted from the mm values in
    the .eng header.
    """
    name: str
    diameter_m: float         # motor outer diameter [m]
    length_m: float           # motor length [m]
    m_prop_kg: float          # propellant mass [kg]
    m_motor_kg: float         # total motor mass: casing + propellant [kg]
    time_s: np.ndarray        # (K,) thrust curve time points [s]
    thrust_n: np.ndarray      # (K,) thrust values [N]


def load_motor(path: Path | str) -> MotorData:
    """Parse a RASP .eng file and return a MotorData.

    Format expected::

        ; optional comment lines
        Name Diam_mm Length_mm Delays PropMass_kg TotalMass_kg Manufacturer
        time_s  thrust_N
        ...

    Masses are in kg. Thrust in Newtons. The final data point should have
    thrust = 0; if absent it is appended automatically.

    Raises
    ------
    ValueError
        If the file cannot be parsed or contains physically implausible values.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    # Strip comments and blank lines
    data_lines = [l.strip() for l in lines
                  if l.strip() and not l.strip().startswith(";")]

    if len(data_lines) < 2:
        raise ValueError(f"Motor file {path} has no usable data")

    # --- header line
    header = data_lines[0].split()
    if len(header) < 7:
        raise ValueError(
            f"Motor file header must have ≥7 fields, got: {data_lines[0]!r}"
        )
    name = header[0]
    try:
        diameter_m = float(header[1]) / 1000.0
        length_m = float(header[2]) / 1000.0
        m_prop_kg = float(header[4])
        m_motor_kg = float(header[5])
    except ValueError as exc:
        raise ValueError(
            f"Could not parse motor header fields: {data_lines[0]!r}"
        ) from exc

    if diameter_m <= 0:
        raise ValueError(f"Motor diameter must be > 0, got {diameter_m * 1000:.1f} mm")
    if length_m <= 0:
        raise ValueError(f"Motor length must be > 0, got {length_m * 1000:.1f} mm")
    if m_prop_kg <= 0:
        raise ValueError(f"Propellant mass must be > 0, got {m_prop_kg}")
    if m_motor_kg <= m_prop_kg:
        raise ValueError(
            f"Total motor mass ({m_motor_kg} kg) must exceed propellant mass "
            f"({m_prop_kg} kg)"
        )

    # --- thrust curve data points
    times: list[float] = []
    thrusts: list[float] = []
    for line in data_lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t, f = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        times.append(t)
        thrusts.append(f)

    if len(times) < 2:
        raise ValueError(f"Motor file {path} must have ≥2 thrust data points")

    time_arr = np.asarray(times, dtype=np.float64)
    thrust_arr = np.asarray(thrusts, dtype=np.float64)

    if not np.all(np.diff(time_arr) > 0):
        raise ValueError("Thrust curve time points must be strictly increasing")
    if np.any(thrust_arr < 0):
        raise ValueError("Thrust values must be non-negative")

    # Ensure burnout point has thrust = 0
    if thrust_arr[-1] != 0.0:
        time_arr = np.append(time_arr, time_arr[-1])
        thrust_arr = np.append(thrust_arr, 0.0)

    return MotorData(
        name=name,
        diameter_m=diameter_m,
        length_m=length_m,
        m_prop_kg=m_prop_kg,
        m_motor_kg=m_motor_kg,
        time_s=time_arr,
        thrust_n=thrust_arr,
    )


# ---------------------------------------------------------------------------
# Pre-computed bundles
# ---------------------------------------------------------------------------

@dataclass
class PropellantModel:
    """Motor and propellant data derived from the .eng file and vehicle geometry.

    Contains only motor/propellant concerns: thrust curve, masses, CG,
    annular geometry, nozzle properties.  No vehicle-level properties.

    Construct via :func:`build_propellant_model`.
    """
    # Thrust curve (contiguous float64 arrays)
    times: np.ndarray        # (K,) [s]
    thrusts: np.ndarray      # (K,) [N]  — sea-level values

    # Scalar motor properties
    m_prop_0: float          # initial propellant mass [kg]
    m_casing: float          # motor casing mass = m_motor − m_prop_0 [kg]
    total_impulse: float     # ∫F dt [N·s]
    nozzle_area: float       # π·dₑ²/4 [m²] — for pressure thrust correction
    nozzle_position: float   # m from nosecone tip — for jet damping (§8.3.4)
    motor_cg_loaded: float   # propellant CG (constant, inside-out burn) [m]

    # Propellant annular geometry for time-varying inertia
    prop_r_outer: float      # propellant outer radius [m]
    prop_r_inner_0: float    # initial propellant inner radius (bore) [m]
    prop_length: float       # motor/propellant length [m]

    # Initial propellant inertias (about propellant CG)
    I_roll_prop_0: float     # kg·m² — roll, full load
    I_lat_prop_0: float      # kg·m² — lateral about propellant CG, full load


def build_propellant_model(
    motor_data: MotorData,
    vehicle_length: float,
    nozzle_area: float,
    casing_thickness: float | None = None,
    propellant_inner_diameter: float | None = None,
) -> PropellantModel:
    """Build the propellant model from the .eng data and vehicle geometry.

    Parameters
    ----------
    motor_data
        Parsed .eng file data.
    vehicle_length
        Total vehicle length [m] — motor is assumed flush with the aft end.
    nozzle_area
        Nozzle exit area [m²] — for pressure thrust correction.
    casing_thickness
        Motor casing wall thickness [m].  If None, propellant fills the full
        motor diameter.
    propellant_inner_diameter
        Propellant bore diameter [m].  If None, propellant is a solid cylinder.

    Raises
    ------
    ValueError
        If total impulse ≤ 0 or propellant geometry is invalid.
    """
    times = np.ascontiguousarray(motor_data.time_s, dtype=np.float64)
    thrusts = np.ascontiguousarray(motor_data.thrust_n, dtype=np.float64)

    # Prepend (0, 0) if the .eng data doesn't start at t=0, matching
    # RASAero's implicit ignition ramp convention (linear ramp from zero).
    if times[0] > 0.0:
        times   = np.concatenate(([0.0], times))
        thrusts = np.concatenate(([0.0], thrusts))

    total_impulse = float(np.trapz(thrusts, times))
    if total_impulse <= 0:
        raise ValueError("Total impulse must be > 0")

    m_prop_0 = motor_data.m_prop_kg
    m_casing = motor_data.m_motor_kg - m_prop_0
    motor_length = motor_data.length_m
    r_motor = motor_data.diameter_m / 2.0

    # Motor CG at geometric centre, flush with aft end
    motor_cg_loaded = vehicle_length - motor_length / 2.0
    nozzle_position = vehicle_length

    # Propellant annular geometry
    if casing_thickness is not None:
        prop_r_outer = r_motor - casing_thickness
    else:
        prop_r_outer = r_motor

    if propellant_inner_diameter is not None:
        prop_r_inner_0 = propellant_inner_diameter / 2.0
    else:
        prop_r_inner_0 = 0.0

    if prop_r_inner_0 >= prop_r_outer:
        raise ValueError(
            f"Propellant inner radius ({prop_r_inner_0*1000:.1f} mm) must be "
            f"less than outer radius ({prop_r_outer*1000:.1f} mm)"
        )

    # Initial propellant inertias
    r_o2 = prop_r_outer ** 2
    r_i2 = prop_r_inner_0 ** 2
    I_roll_prop_0 = 0.5 * m_prop_0 * (r_o2 + r_i2)
    I_lat_prop_0 = m_prop_0 * (3.0 * (r_o2 + r_i2) + motor_length ** 2) / 12.0

    return PropellantModel(
        times=times,
        thrusts=thrusts,
        m_prop_0=m_prop_0,
        m_casing=m_casing,
        total_impulse=total_impulse,
        nozzle_area=nozzle_area,
        nozzle_position=nozzle_position,
        motor_cg_loaded=motor_cg_loaded,
        prop_r_outer=prop_r_outer,
        prop_r_inner_0=prop_r_inner_0,
        prop_length=motor_length,
        I_roll_prop_0=I_roll_prop_0,
        I_lat_prop_0=I_lat_prop_0,
    )


# ---------------------------------------------------------------------------
# Numba hot-loop helpers
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
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


# ---------------------------------------------------------------------------
# Numba hot-loop functions — propellant-level public API
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def thrust_at(times: np.ndarray, thrusts: np.ndarray, t: float) -> float:
    """Sea-level thrust [N] at time *t* [s]."""
    return _interp(times, thrusts, t)


@nb.njit(cache=True, fastmath=True)
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


@nb.njit(cache=True, fastmath=True)
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
