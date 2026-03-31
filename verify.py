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
from aerodynamics import AeroModel, aero_forces_moments, ca_at, cn_cp_at
from atmosphere import isa
from wind import WindEnsemble
from dynamics import (
    TrajectoryResult,
    run_trajectory,
    simulate_rail,
    SCENARIO_NOMINAL,
    SCENARIO_BALLISTIC,
    mass_at,
    cg_at,
    thrust_corrected_at,
)
from montecarlo import build_sim_params


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEG2RAD: float = math.pi / 180.0
_G0: float = 9.80665


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
# Rail phase reconstruction helper
# ---------------------------------------------------------------------------

def _quantities_from_rail_hist(
    t_hist: np.ndarray,
    V_hist: np.ndarray,
    alt_hist: np.ndarray,
    propellant: PropellantModel,
    aero_model: AeroModel,
    geometry: "VehicleGeometry",
    m_dry: float,
    cg_dry: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute comparison quantities from the actual simulate_rail() history.

    The time, velocity, and altitude arrays come directly from the RK45
    integrator in ``simulate_rail()``, so they reflect the same code path
    used by the Monte Carlo simulation.  Thrust, mass, Mach, SM, and CD
    are derived from the motor and aero models at each recorded time step.

    Returns
    -------
    (t, alt, mach, thrust, mass, sm, cd)
    """
    mm = propellant
    am = aero_model
    N = len(t_hist)
    diameter = geometry.diameter

    mach_arr = np.empty(N, dtype=np.float64)
    thrust_arr = np.empty(N, dtype=np.float64)
    mass_arr = np.empty(N, dtype=np.float64)
    sm_arr = np.empty(N, dtype=np.float64)
    cd_arr = np.empty(N, dtype=np.float64)

    for i in range(N):
        ti = float(t_hist[i])
        hi = float(alt_hist[i])
        Vi = float(V_hist[i])

        _, _, rho, a_sound, mu = isa(hi)
        M = Vi / a_sound if a_sound > 0.0 else 0.0
        mach_arr[i] = M

        thrust_arr[i] = thrust_corrected_at(
            mm.times, mm.thrusts, mm.nozzle_area, hi, ti,
        )
        mass_arr[i] = mass_at(
            mm.times, mm.thrusts, mm.m_prop_0,
            mm.total_impulse, m_dry, ti,
        )

        cg = cg_at(
            mm.times, mm.thrusts, mm.m_prop_0, mm.total_impulse,
            m_dry, cg_dry, mm.motor_cg_loaded, ti,
        )
        Re = rho * Vi * geometry.length / mu if Vi > 1.0e-6 and mu > 0.0 else 0.0

        _, _, _, _, _, cp_whole = aero_forces_moments(
            am.mach_grid, am.re_grid, am.alpha_grid,
            am.ca_table, am.cn_table, am.cp_table,
            am.cn_comp, am.cp_comp, am.has_components,
            M, Re, rho, Vi, geometry.reference_area,
            Vi, 0.0, 0.0, 0.0, 0.0, cg,
        )
        sm_arr[i] = (cp_whole - cg) / diameter if diameter > 0.0 else 0.0

        if M >= am.mach_grid[0]:
            cd_arr[i] = ca_at(
                am.mach_grid, am.re_grid, am.alpha_grid, am.ca_table,
                M, Re, 0.0,
            )
        else:
            cd_arr[i] = math.nan

    return t_hist.copy(), alt_hist.copy(), mach_arr, thrust_arr, mass_arr, sm_arr, cd_arr


# ---------------------------------------------------------------------------
# Trajectory quantity extraction — 6DoF
# ---------------------------------------------------------------------------

def _extract_trajectory_quantities_6dof(
    result: TrajectoryResult,
    propellant: PropellantModel,
    aero_model: AeroModel,
    vehicle: Vehicle,
    rail_t_hist: np.ndarray | None = None,
    rail_V_hist: np.ndarray | None = None,
    rail_alt_hist: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Extract altitude, Mach, SM, thrust, mass, and CD from a 6DoF trajectory.

    Includes both the ascent and descent phases.

    Returns
    -------
    dict with keys "time", "altitude", "mach", "sm", "thrust", "mass", "cd"
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
    cd_asc = np.empty(n_asc, dtype=np.float64)

    mm = propellant
    am = aero_model
    geom = vehicle.geometry
    diameter = geom.diameter
    m_dry = vehicle.m_dry
    cg_dry = vehicle.cg_dry

    for i in range(n_asc):
        ti = float(t_asc[i])
        h = max(float(alt_asc[i]), 0.0)

        T, p, rho, a, mu = isa(h)

        u = float(state_asc[i, 7])
        v = float(state_asc[i, 8])
        w = float(state_asc[i, 9])
        V = math.sqrt(u * u + v * v + w * w)

        M = V / a if a > 0.0 else 0.0
        mach_asc[i] = M

        thrust_asc[i] = thrust_corrected_at(
            mm.times, mm.thrusts, mm.nozzle_area, h, ti,
        )

        mass_asc[i] = mass_at(
            mm.times, mm.thrusts, mm.m_prop_0,
            mm.total_impulse, m_dry, ti,
        )

        cg = cg_at(
            mm.times, mm.thrusts, mm.m_prop_0, mm.total_impulse,
            m_dry, cg_dry, mm.motor_cg_loaded, ti,
        )

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

        # True drag coefficient: CD = CA·cos(α) + CN·sin(α)
        # NaN when below the aero table's Mach range (no valid data).
        alpha_rad = math.atan2(math.sqrt(v * v + w * w), u) if V > 1.0e-6 else 0.0
        if M >= am.mach_grid[0]:
            ca = ca_at(
                am.mach_grid, am.re_grid, am.alpha_grid, am.ca_table,
                M, Re, alpha_rad,
            )
            cn, _ = cn_cp_at(
                am.mach_grid, am.re_grid, am.alpha_grid,
                am.cn_table, am.cp_table, M, Re, alpha_rad,
            )
            cd_asc[i] = ca * math.cos(alpha_rad) + cn * math.sin(alpha_rad)
        else:
            cd_asc[i] = math.nan

    # --- Descent (thrust is zero, mass is dry, under parachute) ---
    if result.t_descent is not None and result.n_descent > 0:
        n_desc = result.n_descent
        t_desc = result.t_descent[:n_desc]
        state_desc = result.state_descent[:n_desc]
        alt_desc = -state_desc[:, 2]

        mach_desc = result.mach_descent[:n_desc]
        thrust_desc = np.zeros(n_desc, dtype=np.float64)
        mass_desc = np.full(n_desc, m_dry, dtype=np.float64)
        sm_desc = result.sm_descent[:n_desc]
        cd_desc = np.zeros(n_desc, dtype=np.float64)

        # Stitch (skip first descent point to avoid overlap at apogee)
        t_full = np.concatenate([t_asc, t_desc[1:]])
        alt_full = np.concatenate([alt_asc, alt_desc[1:]])
        mach_full = np.concatenate([mach_asc, mach_desc[1:]])
        sm_full = np.concatenate([sm_asc, sm_desc[1:]])
        thrust_full = np.concatenate([thrust_asc, thrust_desc[1:]])
        mass_full = np.concatenate([mass_asc, mass_desc[1:]])
        cd_full = np.concatenate([cd_asc, cd_desc[1:]])
    else:
        t_full = t_asc
        alt_full = alt_asc
        mach_full = mach_asc
        sm_full = sm_asc
        thrust_full = thrust_asc
        mass_full = mass_asc
        cd_full = cd_asc

    # --- Prepend launch rail phase (t=0 to t_asc[0]) ---
    # Uses the actual trajectory recorded by simulate_rail() so the
    # comparison exercises the same code path as the Monte Carlo simulation.
    if rail_t_hist is not None and len(rail_t_hist) > 0:
        t_r, alt_r, mach_r, thrust_r, mass_r, sm_r, cd_r = _quantities_from_rail_hist(
            rail_t_hist, rail_V_hist, rail_alt_hist, mm, am, geom,
            m_dry, cg_dry,
        )
        t_full      = np.concatenate([t_r,      t_full])
        alt_full    = np.concatenate([alt_r,    alt_full])
        mach_full   = np.concatenate([mach_r,   mach_full])
        sm_full     = np.concatenate([sm_r,     sm_full])
        thrust_full = np.concatenate([thrust_r, thrust_full])
        mass_full   = np.concatenate([mass_r,   mass_full])
        cd_full     = np.concatenate([cd_r,     cd_full])

    return {
        "time": t_full,
        "altitude": alt_full,
        "mach": mach_full,
        "sm": sm_full,
        "thrust": thrust_full,
        "mass": mass_full,
        "cd": cd_full,
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

    geom = vehicle.geometry
    zero_wind = _zero_wind_ensemble()
    mm = propellant
    m_dry = vehicle.m_dry

    # Run the rail phase once to capture the trajectory history.
    # This uses the same simulate_rail() called inside run_trajectory(),
    # so the comparison exercises the real rail-phase code path.
    _, _, _, _, _, rt_hist, rV_hist, ralt_hist, rn = simulate_rail(
        azimuth * _DEG2RAD, inclination * _DEG2RAD, rail.length,
        mm.times, mm.thrusts, mm.nozzle_area, 1.0,
        mm.m_prop_0, mm.total_impulse, m_dry,
        aero_model.mach_grid, aero_model.re_grid,
        aero_model.alpha_grid, aero_model.ca_table,
        geom.reference_area, geom.length,
        1.0e-6, 1.0e-6,
    )
    rail_t = rt_hist[:rn]
    rail_V = rV_hist[:rn]
    rail_alt = ralt_hist[:rn]

    # --- Full 6DoF trajectory ---
    params = build_sim_params(
        sim_cfg, vehicle, propellant, aero_model, zero_wind,
        wind_profile_index=0,
        azimuth_deg=azimuth,
        inclination_deg=inclination,
        impulse_factor=1.0,
        fin_cant_deg=0.0,
    )
    scenario = _nominal_scenario(vehicle)
    traj = run_trajectory(params, scenario, None, None, float("inf"))

    sim_data = _extract_trajectory_quantities_6dof(
        traj, propellant, aero_model, vehicle,
        rail_t, rail_V, rail_alt,
    )

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
