"""Trajectory comparison tool — §18.1 of the specification.

Runs a single nominal 6DoF trajectory with mean inputs and zero wind,
then compares altitude, Mach, stability margin, and mass against a
reference CSV from an external flight simulator.

Public API
----------
run_verification(sim_cfg, vehicle, propellant, aero_model)
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
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.figure

from config import SimulationConfig, Vehicle, VerificationConfig
from motor import PropellantModel
from aerodynamics import AeroModel
from wind import WindEnsemble
from dynamics import (
    run_trajectory,
    SCENARIO_NOMINAL,
    SCENARIO_BALLISTIC,
)
from montecarlo import build_sim_params


def _nominal_scenario(vehicle: "Vehicle") -> int:
    """Return the nominal descent scenario based on parachute configuration."""
    rec = vehicle.recovery
    if rec.drogue is None and rec.main is None:
        return SCENARIO_BALLISTIC
    return SCENARIO_NOMINAL


# Absolute floors for fractional tolerance comparison (avoid false failures
# near zero).  Physically motivated minimum-significance thresholds.
_TOLERANCE_FLOORS: dict[str, float] = {
    "altitude": 1.0,    # metres
    "mach": 0.01,       # dimensionless
    "sm": 0.1,          # calibres
    "mass": 0.1,        # kg
    "thrust": 10.0,     # newtons
    "cd": 0.01,         # dimensionless
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
    "cd": ["cd", "drag coeff"],
}

# Plot labels for each quantity
_PLOT_LABELS: dict[str, str] = {
    "altitude": "Altitude (m)",
    "mach": "Mach",
    "sm": "Stability Margin (cal)",
    "thrust": "Thrust (N)",
    "mass": "Mass (kg)",
    "cd": "CD",
}

# Quantities to compare — 3×2 grid
_COMPARED_QUANTITIES: list[str] = ["altitude", "mach", "sm", "thrust", "mass", "cd"]


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
    # Use the full reference timebase.  Simulator values beyond its
    # time range are set to NaN so they are not displayed or compared.
    t_ref = ref_time
    v_ref = ref_values

    # Interpolate simulator to reference time points
    v_sim = np.interp(t_ref, sim_time, sim_values)
    beyond = t_ref > sim_time[-1]
    v_sim[beyond] = np.nan

    # Fractional tolerance with absolute floor (NaN points are excluded)
    floor = _TOLERANCE_FLOORS.get(name, 1.0)
    scale = np.maximum(np.abs(v_ref), floor)
    valid = np.isfinite(v_sim)
    within = np.abs(v_sim - v_ref) <= tolerance * scale
    within[~valid] = True  # don't penalise beyond-range points

    n_outside = int(np.sum(~within))
    n_total = int(np.sum(valid))
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

    Six time-series quantities fill all positions in a 3×2 grid, sharing
    a common time x-axis.  Reference in grey with tolerance band;
    simulator overlay in green (pass) or red (fail).
    """
    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(hspace=0.30, wspace=0.25, left=0.08, right=0.96,
                        top=0.95, bottom=0.07)

    gs = fig.add_gridspec(3, 2)
    time_axes: list[matplotlib.axes.Axes] = []
    first_ax: matplotlib.axes.Axes | None = None
    for row in range(3):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col], sharex=first_ax)
            time_axes.append(ax)
            if first_ax is None:
                first_ax = ax

    plotted_quantities = [q for q in _COMPARED_QUANTITIES if q in comparisons]

    for idx, qty_name in enumerate(plotted_quantities):
        if idx >= len(time_axes):
            break
        ax = time_axes[idx]
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
        ax.legend(fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)

    # Hide unused time axes
    for idx in range(len(plotted_quantities), len(time_axes)):
        time_axes[idx].set_visible(False)

    # X-label on the bottom-row axes
    for ax in time_axes[-2:]:
        if ax.get_visible():
            ax.set_xlabel("Time (s)")

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
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    inclination_override: float | None = None,
) -> VerificationResult:
    """Run a nominal 6DoF trajectory and compare against the reference CSV.

    Uses mean input values and zero wind, as specified in §18.1.  When
    rail angles are set to ``"auto"``, the midpoint of the configured
    search range is used (optimisation has not yet run at this point in
    the program flow).

    *inclination_override* (degrees), when not ``None``, takes precedence
    over any value in the config.  Azimuth is always zero for verification.

    Parameters
    ----------
    sim_cfg
        Simulation configuration (must have ``verification`` set).
    vehicle, propellant, aero_model
        Pre-loaded vehicle/propellant/aero models.

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
        "(metres, seconds, calibres). There is no unit sanitisation "
        "and no check that both simulators used the same input parameters.",
    )

    rail = sim_cfg.launch.rail

    # Azimuth is always zero for verification (wind is zero, so heading
    # doesn't matter — simplifies comparison with 2DoF references).
    azimuth = 0.0

    if inclination_override is not None:
        inclination = inclination_override
    elif ver_cfg.inclination is not None:
        inclination = ver_cfg.inclination
    else:
        inclination = _resolve_rail_angle(rail.inclination, rail.inclination_range)

    zero_wind = _zero_wind_ensemble()

    # --- Full trajectory with profile (rail + 6DoF + descent) ---
    params = build_sim_params(
        sim_cfg, vehicle, propellant, aero_model, zero_wind,
        wind_profile_index=0,
        azimuth_deg=azimuth,
        inclination_deg=inclination,
        impulse_factor=1.0,
        fin_cant_deg=0.0,
    )
    scenario = _nominal_scenario(vehicle)
    _, profile = run_trajectory(
        params, scenario, None, None, float("inf"),
        keep_profile=True,
    )

    # Read unified quantities directly from the profile — no recomputation.
    sim_data: dict[str, np.ndarray] = {
        "time": profile.time,
        "altitude": profile.altitude,
        "mach": profile.mach,
        "sm": profile.sm,
        "thrust": profile.thrust,
        "mass": profile.mass,
        "cd": profile.cd,
    }

    # --- Load reference CSV ---
    ref_data = _load_reference_csv(ver_cfg.reference_trajectory)

    # --- Compare each quantity ---
    tolerance_map: dict[str, float] = {
        "altitude": ver_cfg.altitude_tolerance,
        "mach": ver_cfg.mach_tolerance,
        "sm": ver_cfg.sm_tolerance,
        "thrust": ver_cfg.thrust_tolerance,
        "mass": ver_cfg.mass_tolerance,
        "cd": ver_cfg.cd_tolerance,
    }

    # For CD, truncate at apogee (post-apogee CD is parachute drag, not
    # body aero) and strip NaN values (below aero table Mach range).
    ref_apogee = int(np.argmax(ref_data["altitude"])) + 1
    sim_apogee = int(np.argmax(sim_data["altitude"])) + 1

    comparisons: dict[str, QuantityComparison] = {}
    for qty in _COMPARED_QUANTITIES:
        if qty not in ref_data:
            continue
        if qty == "cd":
            ref_t = ref_data["time"][:ref_apogee]
            ref_v = ref_data["cd"][:ref_apogee]
            sim_t = sim_data["time"][:sim_apogee]
            sim_v = sim_data["cd"][:sim_apogee]
            # Drop points outside the valid aero table range: NaN from
            # the simulator (below mach_grid lower bound) and zero from
            # the reference (RASAero reports CD=0 at zero velocity).
            ref_valid = ref_v > 0.0
            ref_t, ref_v = ref_t[ref_valid], ref_v[ref_valid]
            sim_valid = np.isfinite(sim_v)
            sim_t, sim_v = sim_t[sim_valid], sim_v[sim_valid]
        else:
            ref_t = ref_data["time"]
            ref_v = ref_data[qty]
            sim_t = sim_data["time"]
            sim_v = sim_data[qty]
        comparisons[qty] = _compare_quantity(
            name=qty,
            ref_time=ref_t,
            ref_values=ref_v,
            sim_time=sim_t,
            sim_values=sim_v,
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
