"""Trajectory comparison tool — §18.1 of the specification.

Runs a single nominal trajectory with mean inputs and zero wind, then
compares altitude, Mach, stability margin, and mass against a reference
CSV from an external flight simulator.

Public API
----------
run_verification(sim_cfg, vehicle_cfg, motor_model, aero_model)
    → VerificationResult

Supporting dataclasses:
    QuantityComparison — per-quantity comparison result
    VerificationResult — aggregate result with figure
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

from config import SimulationConfig, VehicleConfig, VerificationConfig
from motor import MotorModel, mass_at, cg_at
from aerodynamics import AeroModel, aero_forces_moments
from atmosphere import isa
from wind import WindEnsemble
from dynamics import (
    TrajectoryResult,
    run_trajectory,
    SCENARIO_NOMINAL,
)
from montecarlo import build_sim_params


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEG2RAD: float = math.pi / 180.0

# Absolute floors for fractional tolerance comparison (avoid false failures
# near zero).  Physically motivated minimum-significance thresholds.
_TOLERANCE_FLOORS: dict[str, float] = {
    "altitude": 1.0,    # metres
    "mach": 0.01,       # dimensionless
    "sm": 0.1,          # calibres
    "mass": 0.1,        # kg
}

# Column alias map: quantity → list of substrings to match (case-insensitive).
# First match in a left-to-right scan of CSV headers wins.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "time": ["time"],
    "altitude": ["altitude", "alt", "height"],
    "mach": ["mach"],
    "mass": ["mass", "weight"],
    "sm": ["stability margin", "stability", "sm"],
}

# Plot labels for each quantity
_PLOT_LABELS: dict[str, str] = {
    "altitude": "Altitude (m)",
    "mach": "Mach",
    "sm": "Stability Margin (cal)",
    "mass": "Mass (kg)",
}

# Quantities to compare (excludes "time" which is the independent variable)
_COMPARED_QUANTITIES: list[str] = ["altitude", "mach", "sm"]


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
    passed: bool                  # all within tolerance


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

    # Check all required columns found
    missing = [q for q in _COLUMN_ALIASES if q not in col_map]
    if missing:
        raise ValueError(
            f"Reference CSV {path.name} is missing columns for: "
            f"{', '.join(missing)}.  Headers found: {raw_headers}"
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
# Trajectory quantity extraction
# ---------------------------------------------------------------------------

def _extract_trajectory_quantities(
    result: TrajectoryResult,
    motor_model: MotorModel,
    aero_model: AeroModel,
    vehicle_cfg: VehicleConfig,
) -> dict[str, np.ndarray]:
    """Extract altitude, Mach, SM, and mass time series from a trajectory.

    Returns
    -------
    dict with keys "time", "altitude", "mach", "sm", "mass" → 1-D arrays.
    """
    t = result.t_ascent
    state = result.state_ascent
    n = len(t)

    # Altitude: -D (NED, D is down)
    altitude = -state[:, 2]

    # Pre-allocate
    mach_arr = np.empty(n, dtype=np.float64)
    mass_arr = np.empty(n, dtype=np.float64)
    sm_arr = np.empty(n, dtype=np.float64)

    mm = motor_model
    am = aero_model
    geom = vehicle_cfg.geometry
    diameter = geom.diameter

    for i in range(n):
        ti = float(t[i])
        h = max(float(altitude[i]), 0.0)

        # Atmosphere
        T, p, rho, a, mu = isa(h)

        # Body-frame velocities (zero wind → these are relative velocities)
        u = float(state[i, 7])
        v = float(state[i, 8])
        w = float(state[i, 9])
        V = math.sqrt(u * u + v * v + w * w)

        # Mach
        mach_arr[i] = V / a if a > 0.0 else 0.0

        # Mass
        mass_arr[i] = mass_at(
            mm.times, mm.thrusts, mm.m_prop_0,
            mm.total_impulse, mm.m_dry, ti,
        )

        # CG and CP → stability margin
        cg = cg_at(
            mm.times, mm.thrusts, mm.m_prop_0, mm.total_impulse,
            mm.m_dry, mm.cg_dry, mm.motor_cg_loaded, ti,
        )

        M = mach_arr[i]
        Re = rho * V * geom.length / mu if V > 1.0e-6 and mu > 0.0 else 0.0
        q_rate = float(state[i, 11])
        r_rate = float(state[i, 12])

        _, _, _, _, _, cp_whole = aero_forces_moments(
            am.mach_grid, am.re_grid, am.alpha_grid,
            am.ca_table, am.cn_table, am.cp_table,
            am.cn_comp, am.cp_comp, am.has_components,
            M, Re, rho, V, geom.reference_area,
            u, v, w, q_rate, r_rate, cg,
        )

        sm_arr[i] = (cp_whole - cg) / diameter if diameter > 0.0 else 0.0

    return {
        "time": t.copy(),
        "altitude": altitude,
        "mach": mach_arr,
        "sm": sm_arr,
        "mass": mass_arr,
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
) -> QuantityComparison:
    """Interpolate simulator onto reference timebase and compare.

    Uses fractional tolerance with an absolute floor to handle near-zero
    values gracefully.
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

    return QuantityComparison(
        name=name,
        ref_time=t_ref,
        ref_values=v_ref,
        sim_values=v_sim,
        tolerance=tolerance,
        within_tolerance=within,
        passed=bool(np.all(within)),
    )


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _build_comparison_figure(
    comparisons: dict[str, QuantityComparison],
) -> matplotlib.figure.Figure:
    """Build a single-column comparison figure with shared x-axis.

    Reference in grey with tolerance band; simulator overlay in green
    (pass) or red (fail).
    """
    n_qty = len(comparisons)
    fig, axes = plt.subplots(
        n_qty, 1, figsize=(10, 3 * n_qty), sharex=True,
    )
    fig.subplots_adjust(hspace=0.15, left=0.12, right=0.96,
                        top=0.92, bottom=0.08)

    # Ensure axes is always iterable (even for a single subplot)
    if n_qty == 1:
        axes = [axes]

    for ax, qty_name in zip(axes, _COMPARED_QUANTITIES):
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
                color=sim_colour, linewidth=1.2, label="Simulator")

        ax.set_ylabel(_PLOT_LABELS[qty_name])
        ax.legend(fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)

    # Only the bottom subplot gets an x-axis label
    axes[-1].set_xlabel("Time (s)")

    # Overall title
    all_passed = all(c.passed for c in comparisons.values())
    title_text = "Trajectory Verification: "
    if all_passed:
        title_text += "PASS"
        title_colour = "#2d7a2d"
    else:
        failed = [c.name for c in comparisons.values() if not c.passed]
        title_text += f"FAIL ({', '.join(failed)})"
        title_colour = "red"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color=title_colour)

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

    Returns
    -------
    VerificationResult
        Contains per-quantity comparisons and a matplotlib Figure.
    """
    ver_cfg = sim_cfg.verification
    if ver_cfg is None:
        raise ValueError("No verification config — nothing to compare against")

    rail = sim_cfg.launch.rail
    azimuth = _resolve_rail_angle(rail.azimuth, rail.azimuth_range)
    inclination = _resolve_rail_angle(rail.inclination, rail.inclination_range)

    # --- Run nominal trajectory with zero wind ---
    zero_wind = _zero_wind_ensemble()
    params = build_sim_params(
        sim_cfg, vehicle_cfg, motor_model, aero_model, zero_wind,
        wind_profile_index=0,
        azimuth_deg=azimuth,
        inclination_deg=inclination,
        impulse_factor=1.0,
        fin_cant_deg=0.0,
    )
    traj = run_trajectory(params, SCENARIO_NOMINAL, None, None, float("inf"))

    # --- Extract simulator quantities ---
    sim_data = _extract_trajectory_quantities(
        traj, motor_model, aero_model, vehicle_cfg,
    )

    # --- Load reference CSV ---
    ref_data = _load_reference_csv(ver_cfg.reference_trajectory)

    # --- Compare each quantity ---
    tolerance_map: dict[str, float] = {
        "altitude": ver_cfg.altitude_tolerance,
        "mach": ver_cfg.mach_tolerance,
        "sm": ver_cfg.sm_tolerance,
        "mass": ver_cfg.mass_tolerance,
    }

    comparisons: dict[str, QuantityComparison] = {}
    for qty in _COMPARED_QUANTITIES:
        comparisons[qty] = _compare_quantity(
            name=qty,
            ref_time=ref_data["time"],
            ref_values=ref_data[qty],
            sim_time=sim_data["time"],
            sim_values=sim_data[qty],
            tolerance=tolerance_map[qty],
        )

    all_passed = all(c.passed for c in comparisons.values())

    # --- Build comparison figure ---
    fig = _build_comparison_figure(comparisons)

    return VerificationResult(
        passed=all_passed,
        comparisons=comparisons,
        figure=fig,
    )
