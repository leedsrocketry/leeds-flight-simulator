"""Trajectory comparison tool — §18.1 of the specification.

Runs a single nominal trajectory with mean inputs and zero wind, then
compares altitude, Mach, stability margin, and mass against a reference
CSV from an external flight simulator.

Supports both the full 6DoF model and the simplified 2DoF point-mass
model (rail → 2DoF ascent → 3DoF descent).

Public API
----------
run_verification(sim_cfg, vehicle_cfg, motor_model, aero_model, dof=6)
    → VerificationResult

Supporting dataclasses:
    QuantityComparison — per-quantity comparison result
    VerificationResult — aggregate result with figure
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

from config import SimulationConfig, VehicleConfig, VerificationConfig
from motor import MotorModel, mass_at, cg_at, thrust_corrected_at
from aerodynamics import AeroModel, aero_forces_moments
from atmosphere import isa
from wind import WindEnsemble
from dynamics import (
    TrajectoryResult,
    run_trajectory,
    simulate_ascent_2dof,
    integrate_descent,
    SCENARIO_NOMINAL,
)
from montecarlo import build_sim_params


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEG2RAD: float = math.pi / 180.0
_G0: float = 9.80665

# Absolute floors for fractional tolerance comparison (avoid false failures
# near zero).  Physically motivated minimum-significance thresholds.
_TOLERANCE_FLOORS: dict[str, float] = {
    "altitude": 1.0,    # metres
    "mach": 0.01,       # dimensionless
    "sm": 0.1,          # calibres
    "mass": 0.1,        # kg
    "thrust": 10.0,     # newtons
}

# Column alias map: quantity → list of substrings to match (case-insensitive).
# First match in a left-to-right scan of CSV headers wins.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "time": ["time"],
    "altitude": ["altitude", "alt", "height"],
    "mach": ["mach"],
    "mass": ["mass", "weight"],
    "sm": ["stability margin", "stability", "sm"],
    "thrust": ["thrust", "force"],
}

# Plot labels for each quantity
_PLOT_LABELS: dict[str, str] = {
    "altitude": "Altitude (m)",
    "mach": "Mach",
    "sm": "Stability Margin (cal)",
    "thrust": "Thrust (N)",
    "mass": "Mass (kg)",
}

# Quantities to compare — 3×2 grid (last cell empty)
_COMPARED_QUANTITIES: list[str] = ["altitude", "mach", "sm", "thrust", "mass"]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QuantityComparison:
    """Result of comparing one quantity against a reference time series."""
    name: str
    ref_time: np.ndarray
    ref_values: np.ndarray
    sim_values: np.ndarray        # interpolated onto ref_time
    tolerance: float              # fractional tolerance
    within_tolerance: np.ndarray  # bool array, True where within band
    passed: bool                  # within exceedance allowance


@dataclass
class VerificationResult:
    """Aggregate result of the trajectory comparison."""
    passed: bool
    comparisons: dict[str, QuantityComparison]
    figure: matplotlib.figure.Figure | None


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _match_column(header: str, aliases: list[str]) -> bool:
    """Return True if *header* (already lowered) contains any alias."""
    for alias in aliases:
        if alias in header:
            return True
    return False


def _load_reference_csv(path: Path) -> dict[str, np.ndarray]:
    """Load a reference trajectory CSV with case-insensitive column matching.

    No unit conversions are applied — values are taken as-is.  The user
    must ensure the CSV uses the same units as the simulator.

    Returns
    -------
    dict mapping quantity name ("time", "altitude", "mach", "mass", "sm")
    to 1-D float64 arrays.

    Raises
    ------
    ValueError
        If a required column cannot be found.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_headers = next(reader)
        rows = list(reader)

    # Build column index: quantity → CSV column index
    headers_lower = [h.strip().lower() for h in raw_headers]
    col_map: dict[str, int] = {}

    for qty, aliases in _COLUMN_ALIASES.items():
        for col_idx, h in enumerate(headers_lower):
            if _match_column(h, aliases):
                col_map[qty] = col_idx
                break

    # Warn about missing optional columns (time is still required)
    missing = [q for q in _COLUMN_ALIASES if q not in col_map]
    if "time" in missing:
        raise ValueError(
            f"Reference CSV {path.name} has no recognisable time column.  "
            f"Headers found: {raw_headers}"
        )
    optional_missing = [q for q in missing if q != "time"]
    if optional_missing:
        warnings.warn(
            f"Reference CSV is missing columns for: "
            f"{', '.join(optional_missing)} — these quantities will be "
            f"skipped in the comparison."
        )

    # Parse numeric data
    data: dict[str, np.ndarray] = {}
    for qty, col_idx in col_map.items():
        values = []
        for row in rows:
            try:
                values.append(float(row[col_idx]))
            except (ValueError, IndexError):
                continue
        data[qty] = np.array(values, dtype=np.float64)

    return data


# ---------------------------------------------------------------------------
# Rail phase reconstruction helper
# ---------------------------------------------------------------------------

def _rail_phase_arrays(
    t_exit: float,
    z_exit: float,
    V_exit: float,
    motor_model: MotorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate time series for the launch rail phase (t=0 to t_exit, exclusive).

    Altitude and speed are linearly interpolated from 0 to their exit values
    (a good approximation for the short rail phase).  Thrust and mass are
    computed exactly from the motor model.

    Returns
    -------
    (t_rail, alt_rail, mach_rail, thrust_rail, mass_rail)
        Each array has length N where N = ceil(t_exit / 0.002), minimum 2.
    """
    mm = motor_model
    N = max(2, math.ceil(t_exit / 0.002))
    t_rail = np.linspace(0.0, t_exit, N + 1)[:-1]  # exclude t_exit (it starts the free-flight arrays)

    alt_rail = np.linspace(0.0, z_exit, N)
    V_rail = np.linspace(0.0, V_exit, N)

    thrust_rail = np.empty(N, dtype=np.float64)
    mass_rail = np.empty(N, dtype=np.float64)
    mach_rail = np.empty(N, dtype=np.float64)

    for i in range(N):
        ti = float(t_rail[i])
        hi = float(alt_rail[i])
        thrust_rail[i] = thrust_corrected_at(mm.times, mm.thrusts, mm.nozzle_area, hi, ti)
        mass_rail[i] = mass_at(mm.times, mm.thrusts, mm.m_prop_0, mm.total_impulse, mm.m_dry, ti)
        _, _, _, a_sound, _ = isa(hi)
        mach_rail[i] = V_rail[i] / a_sound if a_sound > 0.0 else 0.0

    return t_rail, alt_rail, mach_rail, thrust_rail, mass_rail


# ---------------------------------------------------------------------------
# Trajectory quantity extraction — 6DoF
# ---------------------------------------------------------------------------

def _extract_trajectory_quantities_6dof(
    result: TrajectoryResult,
    motor_model: MotorModel,
    aero_model: AeroModel,
    vehicle_cfg: VehicleConfig,
) -> dict[str, np.ndarray]:
    """Extract altitude, Mach, SM, thrust, and mass from a 6DoF trajectory.

    Includes both the ascent and descent phases.

    Returns
    -------
    dict with keys "time", "altitude", "mach", "sm", "thrust", "mass"
    → 1-D arrays.
    """
    # --- Ascent ---
    t_asc = result.t_ascent
    state_asc = result.state_ascent
    n_asc = len(t_asc)

    alt_asc = -state_asc[:, 2]

    mach_asc = np.empty(n_asc, dtype=np.float64)
    thrust_asc = np.empty(n_asc, dtype=np.float64)
    mass_asc = np.empty(n_asc, dtype=np.float64)
    sm_asc = np.empty(n_asc, dtype=np.float64)

    mm = motor_model
    am = aero_model
    geom = vehicle_cfg.geometry
    diameter = geom.diameter

    for i in range(n_asc):
        ti = float(t_asc[i])
        h = max(float(alt_asc[i]), 0.0)

        T, p, rho, a, mu = isa(h)

        u = float(state_asc[i, 7])
        v = float(state_asc[i, 8])
        w = float(state_asc[i, 9])
        V = math.sqrt(u * u + v * v + w * w)

        mach_asc[i] = V / a if a > 0.0 else 0.0

        thrust_asc[i] = thrust_corrected_at(
            mm.times, mm.thrusts, mm.nozzle_area, h, ti,
        )

        mass_asc[i] = mass_at(
            mm.times, mm.thrusts, mm.m_prop_0,
            mm.total_impulse, mm.m_dry, ti,
        )

        cg = cg_at(
            mm.times, mm.thrusts, mm.m_prop_0, mm.total_impulse,
            mm.m_dry, mm.cg_dry, mm.motor_cg_loaded, ti,
        )

        M = mach_asc[i]
        Re = rho * V * geom.length / mu if V > 1.0e-6 and mu > 0.0 else 0.0
        q_rate = float(state_asc[i, 11])
        r_rate = float(state_asc[i, 12])

        _, _, _, _, _, cp_whole = aero_forces_moments(
            am.mach_grid, am.re_grid, am.alpha_grid,
            am.ca_table, am.cn_table, am.cp_table,
            am.cn_comp, am.cp_comp, am.has_components,
            M, Re, rho, V, geom.reference_area,
            u, v, w, q_rate, r_rate, cg,
        )

        sm_asc[i] = (cp_whole - cg) / diameter if diameter > 0.0 else 0.0

    # --- Descent (thrust is zero, mass is dry) ---
    if result.t_descent is not None and result.n_descent > 0:
        n_desc = result.n_descent
        t_desc = result.t_descent[:n_desc]
        state_desc = result.state_descent[:n_desc]
        alt_desc = -state_desc[:, 2]

        mach_desc = np.empty(n_desc, dtype=np.float64)
        thrust_desc = np.zeros(n_desc, dtype=np.float64)
        mass_desc = np.full(n_desc, mm.m_dry, dtype=np.float64)
        sm_desc = np.empty(n_desc, dtype=np.float64)

        for i in range(n_desc):
            h = max(float(alt_desc[i]), 0.0)
            _, _, rho, a, mu = isa(h)

            vN = float(state_desc[i, 3])
            vE = float(state_desc[i, 4])
            vD = float(state_desc[i, 5])
            V = math.sqrt(vN * vN + vE * vE + vD * vD)

            mach_desc[i] = V / a if a > 0.0 else 0.0

            cg = mm.cg_dry
            M = mach_desc[i]
            Re = rho * V * geom.length / mu if V > 1.0e-6 and mu > 0.0 else 0.0

            _, _, _, _, _, cp_whole = aero_forces_moments(
                am.mach_grid, am.re_grid, am.alpha_grid,
                am.ca_table, am.cn_table, am.cp_table,
                am.cn_comp, am.cp_comp, am.has_components,
                M, Re, rho, V, geom.reference_area,
                vN, 0.0, vD, 0.0, 0.0, cg,
            )

            sm_desc[i] = (cp_whole - cg) / diameter if diameter > 0.0 else 0.0

        # Stitch (skip first descent point to avoid overlap at apogee)
        t_full = np.concatenate([t_asc, t_desc[1:]])
        alt_full = np.concatenate([alt_asc, alt_desc[1:]])
        mach_full = np.concatenate([mach_asc, mach_desc[1:]])
        sm_full = np.concatenate([sm_asc, sm_desc[1:]])
        thrust_full = np.concatenate([thrust_asc, thrust_desc[1:]])
        mass_full = np.concatenate([mass_asc, mass_desc[1:]])
    else:
        t_full = t_asc
        alt_full = alt_asc
        mach_full = mach_asc
        sm_full = sm_asc
        thrust_full = thrust_asc
        mass_full = mass_asc

    # --- Prepend launch rail phase (t=0 to t_asc[0]) ---
    # t_asc starts at rail exit; prepend the on-rail period so the LFS time
    # axis aligns with reference simulators that start at ignition (t=0).
    t_exit = float(t_asc[0])
    if t_exit > 0.0:
        u0 = float(state_asc[0, 7])
        v0 = float(state_asc[0, 8])
        w0 = float(state_asc[0, 9])
        V_exit = math.sqrt(u0 * u0 + v0 * v0 + w0 * w0)
        z_exit = float(alt_asc[0])
        t_r, alt_r, mach_r, thrust_r, mass_r = _rail_phase_arrays(t_exit, z_exit, V_exit, mm)
        sm_r = np.full(len(t_r), float(sm_asc[0]))
        t_full    = np.concatenate([t_r,     t_full])
        alt_full  = np.concatenate([alt_r,   alt_full])
        mach_full = np.concatenate([mach_r,  mach_full])
        sm_full   = np.concatenate([sm_r,    sm_full])
        thrust_full = np.concatenate([thrust_r, thrust_full])
        mass_full   = np.concatenate([mass_r,   mass_full])

    return {
        "time": t_full,
        "altitude": alt_full,
        "mach": mach_full,
        "sm": sm_full,
        "thrust": thrust_full,
        "mass": mass_full,
    }


# ---------------------------------------------------------------------------
# Trajectory quantity extraction — 2DoF
# ---------------------------------------------------------------------------

def _extract_trajectory_quantities_2dof(
    t_asc: np.ndarray,
    z_asc: np.ndarray,
    vx_asc: np.ndarray,
    vz_asc: np.ndarray,
    t_desc: np.ndarray,
    state_desc: np.ndarray,
    n_desc: int,
    motor_model: MotorModel,
    aero_model: AeroModel,
    vehicle_cfg: VehicleConfig,
) -> dict[str, np.ndarray]:
    """Extract altitude, Mach, SM, thrust, and mass from a 2DoF ascent + 3DoF descent.

    Returns
    -------
    dict with keys "time", "altitude", "mach", "sm", "thrust", "mass"
    → 1-D arrays.
    """
    mm = motor_model
    am = aero_model
    geom = vehicle_cfg.geometry
    diameter = geom.diameter
    n_asc = len(t_asc)

    # --- Ascent quantities ---
    mach_asc = np.empty(n_asc, dtype=np.float64)
    thrust_asc = np.empty(n_asc, dtype=np.float64)
    mass_asc = np.empty(n_asc, dtype=np.float64)
    sm_asc = np.empty(n_asc, dtype=np.float64)

    for i in range(n_asc):
        ti = float(t_asc[i])
        h = max(float(z_asc[i]), 0.0)
        _, _, rho, a, mu = isa(h)

        V = math.sqrt(float(vx_asc[i])**2 + float(vz_asc[i])**2)
        mach_asc[i] = V / a if a > 0.0 else 0.0

        thrust_asc[i] = thrust_corrected_at(
            mm.times, mm.thrusts, mm.nozzle_area, h, ti,
        )

        mass_asc[i] = mass_at(
            mm.times, mm.thrusts, mm.m_prop_0,
            mm.total_impulse, mm.m_dry, ti,
        )

        # SM at α=0
        cg = cg_at(
            mm.times, mm.thrusts, mm.m_prop_0, mm.total_impulse,
            mm.m_dry, mm.cg_dry, mm.motor_cg_loaded, ti,
        )
        M = mach_asc[i]
        Re = rho * V * geom.length / mu if V > 1.0e-6 and mu > 0.0 else 0.0

        _, _, _, _, _, cp_whole = aero_forces_moments(
            am.mach_grid, am.re_grid, am.alpha_grid,
            am.ca_table, am.cn_table, am.cp_table,
            am.cn_comp, am.cp_comp, am.has_components,
            M, Re, rho, V, geom.reference_area,
            V, 0.0, 0.0, 0.0, 0.0, cg,
        )
        sm_asc[i] = (cp_whole - cg) / diameter if diameter > 0.0 else 0.0

    # --- Descent quantities (thrust is zero, mass is dry) ---
    t_d = t_desc[:n_desc]
    state_d = state_desc[:n_desc]
    alt_desc = -state_d[:, 2]

    mach_desc = np.empty(n_desc, dtype=np.float64)
    thrust_desc = np.zeros(n_desc, dtype=np.float64)
    mass_desc = np.full(n_desc, mm.m_dry, dtype=np.float64)
    sm_desc = np.empty(n_desc, dtype=np.float64)

    for i in range(n_desc):
        h = max(float(alt_desc[i]), 0.0)
        _, _, rho, a, mu = isa(h)

        vN = float(state_d[i, 3])
        vE = float(state_d[i, 4])
        vD = float(state_d[i, 5])
        V = math.sqrt(vN * vN + vE * vE + vD * vD)
        mach_desc[i] = V / a if a > 0.0 else 0.0

        cg = mm.cg_dry
        M = mach_desc[i]
        Re = rho * V * geom.length / mu if V > 1.0e-6 and mu > 0.0 else 0.0

        _, _, _, _, _, cp_whole = aero_forces_moments(
            am.mach_grid, am.re_grid, am.alpha_grid,
            am.ca_table, am.cn_table, am.cp_table,
            am.cn_comp, am.cp_comp, am.has_components,
            M, Re, rho, V, geom.reference_area,
            vN, 0.0, vD, 0.0, 0.0, cg,
        )
        sm_desc[i] = (cp_whole - cg) / diameter if diameter > 0.0 else 0.0

    # Stitch (skip first descent point to avoid overlap at apogee)
    t_full = np.concatenate([t_asc, t_d[1:]])
    alt_full = np.concatenate([z_asc, alt_desc[1:]])
    mach_full = np.concatenate([mach_asc, mach_desc[1:]])
    sm_full = np.concatenate([sm_asc, sm_desc[1:]])
    thrust_full = np.concatenate([thrust_asc, thrust_desc[1:]])
    mass_full = np.concatenate([mass_asc, mass_desc[1:]])

    # --- Prepend launch rail phase (t=0 to t_asc[0]) ---
    t_exit = float(t_asc[0])
    if t_exit > 0.0:
        V_exit = math.sqrt(float(vx_asc[0])**2 + float(vz_asc[0])**2)
        z_exit = float(z_asc[0])
        t_r, alt_r, mach_r, thrust_r, mass_r = _rail_phase_arrays(t_exit, z_exit, V_exit, mm)
        sm_r = np.full(len(t_r), float(sm_asc[0]))
        t_full      = np.concatenate([t_r,      t_full])
        alt_full    = np.concatenate([alt_r,    alt_full])
        mach_full   = np.concatenate([mach_r,   mach_full])
        sm_full     = np.concatenate([sm_r,     sm_full])
        thrust_full = np.concatenate([thrust_r, thrust_full])
        mass_full   = np.concatenate([mass_r,   mass_full])

    return {
        "time": t_full,
        "altitude": alt_full,
        "mach": mach_full,
        "sm": sm_full,
        "thrust": thrust_full,
        "mass": mass_full,
    }


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _compare_quantity(
    name: str,
    ref_time: np.ndarray,
    ref_values: np.ndarray,
    sim_time: np.ndarray,
    sim_values: np.ndarray,
    tolerance: float,
    exceedance_fraction: float = 0.0,
) -> QuantityComparison:
    """Interpolate simulator onto reference timebase and compare.

    Uses fractional tolerance with an absolute floor to handle near-zero
    values gracefully.  A quantity passes if the fraction of points
    outside the tolerance band is ≤ ``exceedance_fraction``.
    """
    # Clip to overlapping time range
    t_max = min(ref_time[-1], sim_time[-1])
    mask = ref_time <= t_max
    t_ref = ref_time[mask]
    v_ref = ref_values[mask]

    # Interpolate simulator to reference time points
    v_sim = np.interp(t_ref, sim_time, sim_values)

    # Fractional tolerance with absolute floor
    floor = _TOLERANCE_FLOORS.get(name, 1.0)
    scale = np.maximum(np.abs(v_ref), floor)
    within = np.abs(v_sim - v_ref) <= tolerance * scale

    n_outside = int(np.sum(~within))
    n_total = len(within)
    passed = (n_outside / n_total) <= exceedance_fraction if n_total > 0 else True

    return QuantityComparison(
        name=name,
        ref_time=t_ref,
        ref_values=v_ref,
        sim_values=v_sim,
        tolerance=tolerance,
        within_tolerance=within,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _build_comparison_figure(
    comparisons: dict[str, QuantityComparison],
) -> matplotlib.figure.Figure:
    """Build a 3×2 comparison figure.

    Five quantities fill positions (0,0), (0,1), (1,0), (1,1), (2,0);
    position (2,1) is left empty.  Reference in grey with tolerance band;
    simulator overlay in green (pass) or red (fail).
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.30, wspace=0.25, left=0.08, right=0.96,
                        top=0.95, bottom=0.07)

    # Flatten to iterate in row-major order
    ax_flat = axes.ravel()

    plotted_quantities = [q for q in _COMPARED_QUANTITIES if q in comparisons]

    for idx, qty_name in enumerate(plotted_quantities):
        ax = ax_flat[idx]
        cmp = comparisons[qty_name]

        # Reference: grey line + tolerance band
        ax.plot(cmp.ref_time, cmp.ref_values,
                color="grey", linewidth=1.5, label="Reference")
        ax.fill_between(
            cmp.ref_time,
            cmp.ref_values * (1.0 - cmp.tolerance),
            cmp.ref_values * (1.0 + cmp.tolerance),
            color="grey", alpha=0.2,
        )

        # Simulator overlay: green if passed, red if failed
        sim_colour = "#2d7a2d" if cmp.passed else "red"
        ax.plot(cmp.ref_time, cmp.sim_values,
                color=sim_colour, linewidth=1.2, label="LFS")

        ax.set_ylabel(_PLOT_LABELS[qty_name])
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)

    # Hide any unused axes
    for idx in range(len(plotted_quantities), len(ax_flat)):
        ax_flat[idx].set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# Zero-wind ensemble
# ---------------------------------------------------------------------------

def _zero_wind_ensemble() -> WindEnsemble:
    """Construct a single-profile WindEnsemble with zero wind everywhere."""
    alt = np.array([0.0, 50000.0], dtype=np.float64)
    zeros = np.zeros(2, dtype=np.float64)
    return WindEnsemble(
        altitude_m=alt,
        wind_east_ms=zeros.reshape(1, 2),
        wind_north_ms=zeros.reshape(1, 2),
        mean_east_ms=zeros,
        mean_north_ms=zeros,
    )


# ---------------------------------------------------------------------------
# Resolve rail angles (handle "auto" before optimisation has run)
# ---------------------------------------------------------------------------

def _resolve_rail_angle(
    value: float | Literal["auto"],
    angle_range: tuple[float, float] | None,
) -> float:
    """Return a numeric angle, using range midpoint when value is 'auto'."""
    if value == "auto":
        if angle_range is None:
            raise ValueError(
                "Rail angle is 'auto' but no range is configured"
            )
        return (angle_range[0] + angle_range[1]) / 2.0
    return float(value)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_verification(
    sim_cfg: SimulationConfig,
    vehicle_cfg: VehicleConfig,
    motor_model: MotorModel,
    aero_model: AeroModel,
    dof: int = 6,
) -> VerificationResult:
    """Run a nominal trajectory and compare against the reference CSV.

    Uses mean input values and zero wind, as specified in §18.1.  When
    rail angles are set to ``"auto"``, the midpoint of the configured
    search range is used (optimisation has not yet run at this point in
    the program flow).

    Parameters
    ----------
    sim_cfg
        Simulation configuration (must have ``verification`` set).
    vehicle_cfg, motor_model, aero_model
        Pre-loaded vehicle/motor/aero models.
    dof : int
        Degrees of freedom for the ascent model: 6 (full 6DoF) or 2
        (point-mass in a vertical plane).

    Returns
    -------
    VerificationResult
        Contains per-quantity comparisons and a matplotlib Figure.
    """
    ver_cfg = sim_cfg.verification
    if ver_cfg is None:
        raise ValueError("No verification config — nothing to compare against")

    warnings.warn(
        "Reference trajectory CSV is assumed to use SI units "
        "(metres, seconds, calibres). There is no unit sanitisation.",
    )

    rail = sim_cfg.launch.rail
    azimuth = _resolve_rail_angle(rail.azimuth, rail.azimuth_range)
    inclination = _resolve_rail_angle(rail.inclination, rail.inclination_range)

    geom = vehicle_cfg.geometry
    zero_wind = _zero_wind_ensemble()

    if dof == 6:
        # --- Full 6DoF trajectory ---
        params = build_sim_params(
            sim_cfg, vehicle_cfg, motor_model, aero_model, zero_wind,
            wind_profile_index=0,
            azimuth_deg=azimuth,
            inclination_deg=inclination,
            impulse_factor=1.0,
            fin_cant_deg=0.0,
        )
        traj = run_trajectory(params, SCENARIO_NOMINAL, None, None, float("inf"))

        sim_data = _extract_trajectory_quantities_6dof(
            traj, motor_model, aero_model, vehicle_cfg,
        )
    elif dof == 2:
        # --- 2DoF ascent + 3DoF nominal descent ---
        t_asc, x_asc, z_asc, vx_asc, vz_asc = simulate_ascent_2dof(
            rail_inclination_rad=inclination * _DEG2RAD,
            rail_length=rail.length,
            motor_times=motor_model.times,
            motor_thrusts=motor_model.thrusts,
            nozzle_area=motor_model.nozzle_area,
            impulse_factor=1.0,
            m_prop_0=motor_model.m_prop_0,
            total_impulse=motor_model.total_impulse,
            m_dry=motor_model.m_dry,
            mach_g=aero_model.mach_grid,
            re_g=aero_model.re_grid,
            alpha_g=aero_model.alpha_grid,
            ca_tbl=aero_model.ca_table,
            A_ref=geom.reference_area,
            ref_length=geom.length,
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        # Set up descent from apogee in NED (azimuth=0: x=North)
        apN = float(x_asc[-1])
        apD = -float(z_asc[-1])
        t_apogee = float(t_asc[-1])

        # Rotate apogee to the configured azimuth
        az_rad = azimuth * _DEG2RAD
        cos_az = math.cos(az_rad)
        sin_az = math.sin(az_rad)
        apN_rot = apN * cos_az
        apE_rot = apN * sin_az

        descent_state0 = np.array([
            apN_rot, apE_rot, apD, 0.0, 0.0, 0.0,
        ], dtype=np.float64)

        # Recovery CdA
        rec = vehicle_cfg.recovery
        has_drogue = rec.drogue is not None
        has_main = rec.main is not None
        drogue_cda = rec.drogue.cd * rec.drogue.area if has_drogue else 0.0
        main_cda = rec.main.cd * rec.main.area if has_main else 0.0
        main_deploy_alt = float(rec.main.threshold) if has_main and rec.main.threshold != "apogee" else -1.0

        zero_wind_alt = np.array([0.0, 50000.0], dtype=np.float64)
        zero_wind_e = np.zeros(2, dtype=np.float64)
        zero_wind_n = np.zeros(2, dtype=np.float64)

        t_desc, y_desc, n_desc = integrate_descent(
            t_apogee, descent_state0,
            zero_wind_alt, zero_wind_e, zero_wind_n,
            aero_model.mach_grid, aero_model.re_grid,
            aero_model.alpha_grid, aero_model.ca_table,
            geom.reference_area, geom.length,
            motor_model.m_dry,
            drogue_cda, main_cda, main_deploy_alt,
            SCENARIO_NOMINAL,
            1.0e-6, 1.0e-6,
        )

        sim_data = _extract_trajectory_quantities_2dof(
            t_asc, z_asc, vx_asc, vz_asc,
            t_desc, y_desc, n_desc,
            motor_model, aero_model, vehicle_cfg,
        )
    else:
        raise ValueError(f"dof must be 2 or 6, got {dof}")

    # --- Load reference CSV ---
    ref_data = _load_reference_csv(ver_cfg.reference_trajectory)

    # --- Compare each quantity ---
    tolerance_map: dict[str, float] = {
        "altitude": ver_cfg.altitude_tolerance,
        "mach": ver_cfg.mach_tolerance,
        "sm": ver_cfg.sm_tolerance,
        "thrust": ver_cfg.thrust_tolerance,
        "mass": ver_cfg.mass_tolerance,
    }

    comparisons: dict[str, QuantityComparison] = {}
    for qty in _COMPARED_QUANTITIES:
        if qty not in ref_data:
            continue
        comparisons[qty] = _compare_quantity(
            name=qty,
            ref_time=ref_data["time"],
            ref_values=ref_data[qty],
            sim_time=sim_data["time"],
            sim_values=sim_data[qty],
            tolerance=tolerance_map[qty],
            exceedance_fraction=ver_cfg.exceedance_fraction,
        )

    all_passed = all(c.passed for c in comparisons.values())

    # --- Build comparison figure ---
    fig = _build_comparison_figure(comparisons)

    return VerificationResult(
        passed=all_passed,
        comparisons=comparisons,
        figure=fig,
    )
