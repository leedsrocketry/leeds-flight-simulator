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
    check_stability_compliance,
    SCENARIO_NOMINAL,
    SCENARIO_BALLISTIC,
)
from atmosphere import isa_at_site, compute_t_offset
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
    "altitude": 100.0,  # metres
    "mach": 0.05,        # dimensionless
    "sm": 0.1,          # calibres
    "mass": 0.2,        # kg
    "thrust": 5.0,      # newtons
    "cd": 0.01,         # dimensionless
    "drag": 10.0,        # newtons
    "cg": 0.01,         # metres
    "cp": 0.01,         # metres
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
    "drag": ["drag"],
    "cg": ["cg", "centre of gravity", "center of gravity"],
    "cp": ["cp", "centre of pressure", "center of pressure"],
}

# Plot labels for each quantity
_PLOT_LABELS: dict[str, str] = {
    "altitude": "Altitude (m)",
    "mach": "Mach",
    "sm": "Stability Margin (cal)",
    "thrust": "Thrust (N)",
    "mass": "Mass (kg)",
    "cd": "CD",
    "drag": "Drag (N)",
    "cg": "CG (m from nose)",
    "cp": "CP (m from nose)",
}

# Quantities to compare — 3×2 grid (CD is compared but not plotted; drag
# takes its grid slot since the force is more physically meaningful).
_COMPARED_QUANTITIES: list[str] = ["altitude", "mach", "sm", "thrust", "mass", "drag"]


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

    The tolerance band at each point is the larger of the fractional
    tolerance (``tolerance * |ref|``) and the absolute floor for that
    quantity.  Errors below the floor always pass.  A quantity passes
    overall if the fraction of points outside the band is ≤
    ``exceedance_fraction``.
    """
    # Use the full reference timebase.  Simulator values beyond its
    # time range are set to NaN so they are not displayed or compared.
    t_ref = ref_time
    v_ref = ref_values

    # Interpolate simulator to reference time points
    v_sim = np.interp(t_ref, sim_time, sim_values)
    beyond = t_ref > sim_time[-1]
    v_sim[beyond] = np.nan

    # Band = max(fractional tolerance, absolute floor).
    # The floor is the minimum acceptable absolute error regardless of
    # the reference magnitude — errors below the floor always pass.
    floor = _TOLERANCE_FLOORS.get(name, 1.0)
    band = np.maximum(tolerance * np.abs(v_ref), floor)
    valid = np.isfinite(v_sim)
    within = np.abs(v_sim - v_ref) <= band
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

_ASCENT_ONLY_QUANTITIES = {"sm", "thrust", "mass", "drag"}

_VLINE_KW: dict = dict(color="grey", linewidth=0.9, linestyle="--", zorder=0)
_VLINE_BBOX: dict = dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                         alpha=0.8)


def _build_comparison_figure(
    comparisons: dict[str, QuantityComparison],
    t_apogee_lfs: float,
    t_apogee_ref: float,
) -> matplotlib.figure.Figure:
    """Build a 3×2 comparison figure.

    Top row (altitude, Mach) uses the full flight time base.  Bottom
    four (SM, thrust, mass, drag) use an ascent-only time base trimmed
    to apogee.  CG, CP, and CD comparisons are included in the data
    but not plotted; drag force takes the sixth grid slot instead of CD.
    """
    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(hspace=0.30, wspace=0.25, left=0.08, right=0.96,
                        top=0.95, bottom=0.07)

    gs = fig.add_gridspec(3, 2)

    # Two sharex groups: top row (full flight) and bottom four (ascent)
    ax_top_first: matplotlib.axes.Axes | None = None
    ax_bot_first: matplotlib.axes.Axes | None = None
    time_axes: list[matplotlib.axes.Axes] = []
    for row in range(3):
        for col in range(2):
            idx = row * 2 + col
            if row == 0:
                ax = fig.add_subplot(gs[row, col], sharex=ax_top_first)
                if ax_top_first is None:
                    ax_top_first = ax
            else:
                ax = fig.add_subplot(gs[row, col], sharex=ax_bot_first)
                if ax_bot_first is None:
                    ax_bot_first = ax
            time_axes.append(ax)

    plotted_quantities = [q for q in _COMPARED_QUANTITIES if q in comparisons]

    for idx, qty_name in enumerate(plotted_quantities):
        if idx >= len(time_axes):
            break
        ax = time_axes[idx]
        cmp = comparisons[qty_name]

        # Reference: grey line + tolerance band
        ax.plot(cmp.ref_time, cmp.ref_values,
                color="grey", linewidth=1.5, label="Reference")
        floor = _TOLERANCE_FLOORS.get(qty_name, 1.0)
        band = np.maximum(cmp.tolerance * np.abs(cmp.ref_values), floor)
        ax.fill_between(
            cmp.ref_time,
            cmp.ref_values - band,
            cmp.ref_values + band,
            color="grey", alpha=0.2,
        )

        # Simulator overlay: green if passed, red if failed
        sim_colour = "#2d7a2d" if cmp.passed else "red"
        ax.plot(cmp.ref_time, cmp.sim_values,
                color=sim_colour, linewidth=1.2, label="LFS")

        ax.set_ylabel(_PLOT_LABELS[qty_name])
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(fontsize=8)

        # LFS apogee marker on every subplot
        ax.axvline(t_apogee_lfs, **_VLINE_KW)
        y_bot = ax.get_ylim()[0]
        ax.text(t_apogee_lfs, y_bot, f"Apogee\n{t_apogee_lfs:.1f} s",
                rotation=90, ha="left", va="bottom",
                fontsize=9, color="grey", bbox=_VLINE_BBOX)

        # Reference Apogee marker on altitude subplot only
        if qty_name == "altitude":
            ax.axvline(t_apogee_ref, **_VLINE_KW)
            # Place label above the LFS apogee label to avoid overlap
            y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
            ax.text(t_apogee_ref, y_mid,
                    f"Reference Apogee\n{t_apogee_ref:.1f} s",
                    rotation=90, ha="right", va="center",
                    fontsize=9, color="grey", bbox=_VLINE_BBOX)

    # Hide unused time axes
    for idx in range(len(plotted_quantities), len(time_axes)):
        time_axes[idx].set_visible(False)

    # X-label on the visible bottom-row axes of each group
    for ax in time_axes[-2:]:
        if ax.get_visible():
            ax.set_xlabel("Time (s)")
    # Top-row x-labels only if bottom rows are on a different time base
    if ax_top_first is not None:
        for ax in time_axes[:2]:
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
    summary, profile = run_trajectory(
        params, scenario, None, None, float("inf"),
        keep_profile=True,
    )
    try:
        check_stability_compliance(
            summary, sim_cfg.monte_carlo.acceptance, "Verification",
        )
    except RuntimeError as exc:
        warnings.warn(str(exc))

    # Read unified quantities directly from the profile — no recomputation.
    # Drag force is derived from CD, atmosphere, and reference area.
    site_elev = sim_cfg.site.elevation
    t_offset = (
        compute_t_offset(site_elev, sim_cfg.site.temperature)
        if sim_cfg.site.temperature is not None
        else 0.0
    )
    A_ref = vehicle.geometry.reference_area
    drag = np.empty_like(profile.cd)
    for i in range(len(drag)):
        h = max(float(profile.altitude[i]), 0.0)
        _, _, rho, a, _ = isa_at_site(h, site_elev, t_offset)
        V = float(profile.mach[i]) * a
        drag[i] = 0.5 * rho * V * V * A_ref * float(profile.cd[i])

    sim_data: dict[str, np.ndarray] = {
        "time": profile.time,
        "altitude": profile.altitude,
        "mach": profile.mach,
        "sm": profile.sm,
        "thrust": profile.thrust,
        "mass": profile.mass,
        "cd": profile.cd,
        "drag": drag,
        "cg": profile.cg,
        "cp": profile.cp,
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
        "cd": ver_cfg.drag_tolerance,
        "drag": ver_cfg.drag_tolerance,
        "cg": ver_cfg.cg_tolerance,
        "cp": ver_cfg.cp_tolerance,
    }

    # Apogee times for trimming and markers
    t_apogee_lfs = summary.apogee_time
    t_apogee_ref = float(ref_data["time"][np.argmax(ref_data["altitude"])])

    # Quantities trimmed to ascent (descent values are meaningless)
    _ASCENT_ONLY = {"sm", "thrust", "mass", "drag"}

    comparisons: dict[str, QuantityComparison] = {}
    for qty in _COMPARED_QUANTITIES:
        if qty not in ref_data:
            continue
        ref_t = ref_data["time"]
        ref_v = ref_data[qty]
        sim_t = sim_data["time"]
        sim_v = sim_data[qty]
        if qty in _ASCENT_ONLY:
            ref_mask = ref_t <= t_apogee_lfs
            sim_mask = sim_t <= t_apogee_lfs
            ref_t, ref_v = ref_t[ref_mask], ref_v[ref_mask]
            sim_t, sim_v = sim_t[sim_mask], sim_v[sim_mask]
        comparisons[qty] = _compare_quantity(
            name=qty,
            ref_time=ref_t,
            ref_values=ref_v,
            sim_time=sim_t,
            sim_values=sim_v,
            tolerance=tolerance_map[qty],
            exceedance_fraction=ver_cfg.exceedance_fraction,
        )

    # --- Additional comparisons (CSV dump only, not plotted) ---
    for qty in ("cd", "cg", "cp"):
        if qty not in ref_data or qty not in tolerance_map:
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
    fig = _build_comparison_figure(comparisons, t_apogee_lfs, t_apogee_ref)

    return VerificationResult(
        passed=all_passed,
        comparisons=comparisons,
        figure=fig,
    )
