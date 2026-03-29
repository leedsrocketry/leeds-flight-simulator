"""outputs.py — CSV/YAML serialisation and plot generation (§16).

Public API
----------
create_results_dir      — create/clear fixed results directory
write_samples_csv       — per-sample CSV (§16.1)
write_summary_yaml      — run summary YAML (§16.2)
save_altitude_plot      — altitude-time plot (§16.5)
save_dispersion_plot    — landing dispersion plot (§16.5)
save_replay_3d          — [stub] 3D isometric replay plot (§16.4)
save_replay_plan_view   — [stub] plan-view replay plot (§16.4)
save_replay_altitude    — [stub] altitude-time replay plot (§16.4)
"""

from __future__ import annotations

import csv
import math
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse, PathPatch
import contextily as cx
import numpy as np
from pyproj import Transformer
from scipy.stats import chi2
from shapely.geometry import Polygon as ShapelyPolygon
import yaml

from config import SimulationConfig, SiteConfig
from geography import (
    _lonlat_to_ned,
    load_polygon_ned,
    polygon_to_arrays,
    R_EARTH,
)
from montecarlo import MonteCarloResult, SampleResult, ScenarioStats, SCENARIO_LABELS
from optimisation import OptimisationResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M_TO_FT = 3.28084
M_TO_KM = 1 / 1000
S_TO_MIN = 1 / 60

SCENARIO_KEYS = ["nominal", "ballistic", "drogue_only", "premature_main"]
SCENARIO_COLOURS = {"nominal": "#2d7a2d", "ballistic": "black",
                    "drogue_only": "goldenrod", "premature_main": "red"}
SCENARIO_ALPHA = {"nominal": 1.0, "ballistic": 0.6,
                  "drogue_only": 0.6, "premature_main": 0.6}

VLINE_LABEL_FONTSIZE = 12

# OS Maps tile URL (same as reference script)
_OS_API_KEY = "wGs0Y4WVHmSuoqkPyFdpAlh7FKEvNSx4"
_OS_TILE_STYLE = "Outdoor_3857"
_OS_TILE_URL = (
    f"https://api.os.uk/maps/raster/v1/zxy/"
    f"{_OS_TILE_STYLE}/{{z}}/{{x}}/{{y}}.png?key={_OS_API_KEY}"
)
_MIN_BUFFER_RADIUS_KM = 1.0

# Web Mercator transformer (for basemap overlay only)
_TO_WM = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# ===================================================================
# 1. Results directory
# ===================================================================

def create_results_dir(
    simulation_yaml_path: Path,
    wind_profile_suffix: str | None = None,
    *,
    _clear: bool = True,
) -> Path:
    """Create a ``results/`` directory next to the simulation YAML.

    On each run the contents of ``results/`` are cleared before writing.
    When multiple wind profiles are used, each gets a sub-folder named
    after the ``.npz`` stem::

        results/                         # single wind profile
        results/monday/                  # multi-profile day sub-folder
        results/tuesday/

    The verification plot is always placed directly in ``results/``.

    Parameters
    ----------
    simulation_yaml_path : Path
        Path to the simulation configuration file.
    wind_profile_suffix : str or None
        If not None, a sub-folder is created inside ``results/`` with this
        name.  Typically the stem of the ``.npz`` file (e.g. ``"monday"``).
    _clear : bool
        If True (default), clear existing contents of ``results/`` before
        creating the directory.  Set to False in tests to avoid side-effects.

    Returns
    -------
    Path
        The created results directory (either ``results/`` itself or a
        wind-profile sub-folder within it).
    """
    import shutil

    results_root = simulation_yaml_path.parent / "results"

    if _clear and results_root.exists():
        shutil.rmtree(results_root, ignore_errors=True)

    if wind_profile_suffix is not None:
        results_dir = results_root / wind_profile_suffix
    else:
        results_dir = results_root

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ===================================================================
# 2. Per-sample CSV (§16.1)
# ===================================================================

def write_samples_csv(
    results: list[SampleResult],
    output_dir: Path,
    has_coastline: bool,
    has_monitour: bool,
) -> Path:
    """Write per-sample results to ``samples.csv``.

    Parameters
    ----------
    results : list[SampleResult]
        All sample results from the Monte Carlo run.
    output_dir : Path
        Directory to write the CSV into.
    has_coastline : bool
        Whether to include the ``landing_at_sea`` column.
    has_monitour : bool
        Whether to include the ``in_coverage`` column.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    # Build header — ordered: inputs → flight time → compliance → values → locations
    header = [
        "sample_id", "scenario",
        "azimuth_deg", "inclination_deg", "fin_cant_deg", "impulse_factor",
        "flight_time_s",
    ]
    # Compliance columns (TRUE = compliant in every case)
    if has_coastline:
        header.append("coastline_compliant")
    header.extend(["danger_area_footprint_compliant", "danger_area_ceiling_compliant"])
    if has_monitour:
        header.append("monitour_compliant")
    header.extend([
        "stability_compliant",
        "min_SM_subsonic", "min_SM_supersonic_cal",
        "max_AoA_deg", "max_mach", "apogee_m",
        "landing_lat", "landing_lon",
        "apogee_lat", "apogee_lon",
    ])

    csv_path = output_dir / "samples.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            row: list = [
                r.sample_id, r.scenario,
                r.azimuth_deg, r.inclination_deg, r.fin_cant_deg,
                r.impulse_factor,
                r.flight_time_s,
            ]
            if has_coastline:
                row.append(r.landing_at_sea)
            row.extend([r.in_buffer, r.below_ceiling])
            if has_monitour:
                row.append(r.in_coverage)
            row.extend([
                r.stability_compliant,
                r.min_sm_subsonic, r.min_sm_supersonic,
                r.max_aoa_deg, r.peak_mach, r.apogee_m,
                r.landing_lat, r.landing_lon,
                r.apogee_lat, r.apogee_lon,
            ])
            writer.writerow(row)

    return csv_path


# ===================================================================
# 3. Run summary YAML (§16.2)
# ===================================================================

def write_summary_yaml(
    mc_result: MonteCarloResult,
    sim_cfg: SimulationConfig,
    opt_result: OptimisationResult | None,
    output_dir: Path,
    *,
    simulation_yaml_path: Path | None = None,
    all_warnings: list[str] | None = None,
) -> Path:
    """Write the run summary to ``summary.yaml``.

    Parameters
    ----------
    mc_result : MonteCarloResult
        Aggregate Monte Carlo results.
    sim_cfg : SimulationConfig
        Simulation configuration (for paths and seed).
    opt_result : OptimisationResult or None
        Optimisation diagnostics, if optimisation was run.
    output_dir : Path
        Directory to write the YAML into.
    simulation_yaml_path : Path or None
        Absolute path to the simulation YAML file, recorded in the summary
        so that ``replay`` can locate the original configuration.
    all_warnings : list[str] or None
        All warnings captured during the run (CLI + MC).  Falls back to
        ``mc_result.warnings`` if not provided.

    Returns
    -------
    Path
        Path to the written YAML file.
    """
    warnings_list = all_warnings if all_warnings is not None else mc_result.warnings
    summary: dict = {}

    # metadata
    config_path_str = (
        str(simulation_yaml_path)
        if simulation_yaml_path is not None
        else str(sim_cfg.vehicle.parent / sim_cfg.vehicle.name)
    )
    summary["metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config_path_str,
        "warnings": warnings_list,
    }

    # optimisation (only included when optimisation was performed)
    if opt_result is not None:
        summary["optimisation"] = {
            "azimuth_mean": float(mc_result.azimuth_mean),
            "inclination_mean": float(mc_result.inclination_mean),
            "selected_azimuth": opt_result.selected_azimuth,
            "selected_inclination": opt_result.selected_inclination,
            "phase1_selected_inclination": opt_result.phase1_selected,
            "phase2_feasible_azimuths": opt_result.phase2_feasible,
            "phase3_observations": [
                [az, float(p)] for az, p in opt_result.phase3_observations
            ],
            "phase3_top_candidates": opt_result.phase3_top_candidates,
            "phase4_compliance": {
                int(k): float(v)
                for k, v in opt_result.phase4_compliance.items()
            },
            "phase4_margins": {
                int(k): float(v)
                for k, v in opt_result.phase4_margins.items()
            },
        }

    # scenarios (renamed from scenario_results; removed samples counts and std fields)
    scenarios: dict = {}
    for name, stats in mc_result.scenario_stats.items():
        scenarios[name] = {
            "compliant": stats.n_compliant,
            "non_compliant": stats.n_non_compliant,
            "passed": stats.passed,
            "apogee_m": {
                "mean": float(stats.apogee_mean),
                "min": float(stats.apogee_min),
                "max": float(stats.apogee_max),
            },
            "landing_distance_m": {
                "mean": float(stats.landing_dist_mean),
                "min": float(stats.landing_dist_min),
                "max": float(stats.landing_dist_max),
            },
            "peak_mach": {
                "mean": float(stats.peak_mach_mean),
            },
            "max_aoa_deg": {
                "mean": float(stats.max_aoa_mean),
            },
            "stability_margin": {
                "subsonic_min": float(stats.sm_subsonic_min),
                "supersonic_min": float(stats.sm_supersonic_min),
            },
        }
    summary["scenarios"] = scenarios

    yaml_path = output_dir / "summary.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    return yaml_path


# ===================================================================
# 4. Altitude plot (§16.5)
# ===================================================================

def _break_mark(ax: plt.Axes, side: str) -> None:
    """Draw a diagonal slash at the x-axis break edge."""
    x_frac = 1.0 if side == "right" else 0.0
    disp_x, disp_y = ax.transAxes.transform((x_frac, 0))
    dx, dy = 4, 9
    fig = ax.get_figure()
    fw, fh = fig.get_size_inches() * fig.dpi
    ax.plot(
        [(disp_x - dx) / fw, (disp_x + dx) / fw],
        [(disp_y - dy) / fh, (disp_y + dy) / fh],
        transform=fig.transFigure,
        color=ax.spines["bottom"].get_edgecolor(),
        clip_on=False,
        linewidth=ax.spines["bottom"].get_linewidth(),
        solid_capstyle="round",
    )


def _ft_formatter(x: float, _: float) -> str:
    return f"{int(round(x)):,}"


def _km_formatter(x: float, _: float) -> str:
    return f"{x / M_TO_FT * M_TO_KM:.0f}"


def save_altitude_plot(
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]],
    burnout_time_s: float,
    output_dir: Path,
) -> Path:
    """Generate and save the altitude-time plot.

    Parameters
    ----------
    scenarios : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping of scenario key → (time_s, altitude_m) arrays.
        Only active scenarios need be present.
    burnout_time_s : float
        Motor burnout time in seconds.
    output_dir : Path
        Directory to save the plot into.

    Returns
    -------
    Path
        Path to the saved ``altitude_plot.png``.
    """
    LEFT_FRAC = 0.70
    RIGHT_FRAC = 0.30

    active_keys = [k for k in SCENARIO_KEYS if k in scenarios]

    # Convert to minutes
    scenarios_min = {
        k: (t * S_TO_MIN, alt) for k, (t, alt) in scenarios.items()
    }
    burnout_t_min = burnout_time_s * S_TO_MIN

    # Left panel x-range — all scenarios except premature_main (if it lands late)
    left_keys = [k for k in active_keys if k != "premature_main"]
    if not left_keys:
        left_keys = active_keys
    t_left_max = np.ceil(
        max(scenarios_min[k][0][-1] for k in left_keys) * 1.05 * 10
    ) / 10

    # Right panel
    t_right_span = t_left_max * (RIGHT_FRAC / LEFT_FRAC)
    if "premature_main" in scenarios_min:
        t_pm_land_min = scenarios_min["premature_main"][0][-1]
    else:
        t_pm_land_min = 0.0
    t_right_end = max(
        t_left_max + 1.0 + t_right_span,
        t_pm_land_min * 1.01,
    )
    t_right_start = t_right_end - t_right_span

    # Y limits
    alt_max_ft = (
        max(np.max(scenarios[k][1]) for k in active_keys) * M_TO_FT * 1.08
    )

    # Figure
    fig = plt.figure(figsize=(14, 7))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.10)

    ax_l, ax_r = fig.subplots(
        1, 2, sharey=True,
        gridspec_kw={"width_ratios": [LEFT_FRAC, RIGHT_FRAC], "wspace": 0.04},
    )

    # Draw curves (reversed so Nominal paints on top)
    handles = []
    for key in reversed(active_keys):
        t_min = scenarios_min[key][0]
        alt_ft = scenarios_min[key][1] * M_TO_FT
        colour = SCENARIO_COLOURS.get(key, "grey")
        alpha = SCENARIO_ALPHA.get(key, 0.6)
        kw = dict(color=colour, linewidth=1.8, alpha=alpha)
        line, = ax_l.plot(t_min, alt_ft, **kw)
        ax_r.plot(t_min, alt_ft, **kw)
        handles.insert(0, line)

    # Axis limits
    ax_l.set_xlim(0, t_left_max)
    ax_r.set_xlim(t_right_start, t_right_end)
    ax_l.set_ylim(0, alt_max_ft)

    # Consistent tick spacing
    raw_interval_min = t_left_max / 7
    tick_interval_min = raw_interval_min
    for step in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        if raw_interval_min <= step:
            tick_interval_min = step
            break
    else:
        tick_interval_min = round(raw_interval_min)
    ax_l.xaxis.set_major_locator(mticker.MultipleLocator(tick_interval_min))
    ax_r.xaxis.set_major_locator(mticker.MultipleLocator(tick_interval_min))

    # Spines
    ax_l.spines[["right", "top"]].set_visible(False)
    ax_r.spines[["left", "top", "right"]].set_visible(False)
    ax_r.yaxis.set_visible(False)
    ax_r.tick_params(axis="y", left=False)

    # Break marks
    fig.canvas.draw()
    _break_mark(ax_l, "right")
    _break_mark(ax_r, "left")

    # Primary Y axis: ft
    ax_l.yaxis.set_major_formatter(mticker.FuncFormatter(_ft_formatter))
    ax_l.set_ylabel("Altitude (ft)", labelpad=8)

    # Secondary Y axis: km (offset left spine)
    ax_km = ax_l.twinx()
    ax_km.set_ylim(ax_l.get_ylim())
    ax_km.yaxis.set_ticks_position("left")
    ax_km.yaxis.set_label_position("left")
    ax_km.spines["left"].set_position(("outward", 68))
    ax_km.spines[["right", "top", "bottom"]].set_visible(False)
    ft_ticks = ax_l.yaxis.get_major_locator().tick_values(0, alt_max_ft)
    ft_ticks = ft_ticks[(ft_ticks >= 0) & (ft_ticks <= alt_max_ft)]
    ax_km.set_yticks(ft_ticks)
    ax_km.yaxis.set_major_formatter(mticker.FuncFormatter(_km_formatter))
    ax_km.set_ylabel("Altitude (km)", labelpad=8)

    # Shared x-axis label
    l_pos, r_pos = ax_l.get_position(), ax_r.get_position()
    fig.text(
        (l_pos.x0 + r_pos.x1) / 2, 0.02,
        "Flight Time (min)", ha="center", va="bottom", fontsize=11,
    )

    # Horizontal apogee line
    apogee_ft = max(np.max(scenarios[k][1]) for k in active_keys) * M_TO_FT
    apogee_km = apogee_ft / M_TO_FT * M_TO_KM
    hline_kw = dict(color="grey", linewidth=0.9, linestyle="--", zorder=0)
    ax_l.axhline(apogee_ft, **hline_kw)
    ax_r.axhline(apogee_ft, **hline_kw)

    label_str = f"Apogee ({apogee_ft:,.0f} ft / {apogee_km:.1f} km)"
    l_pos, r_pos = ax_l.get_position(), ax_r.get_position()
    mid_x_fig = (l_pos.x0 + r_pos.x1) / 2
    apogee_axes_frac = apogee_ft / alt_max_ft
    apogee_fig_y = l_pos.y0 + apogee_axes_frac * l_pos.height
    fig.text(
        mid_x_fig, apogee_fig_y, label_str,
        ha="center", va="bottom", fontsize=VLINE_LABEL_FONTSIZE, color="grey",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

    # Vertical dashed lines at events
    def fmt_mmin(t_min: float) -> str:
        total_s = int(round(t_min * 60))
        return f"{total_s // 60}:{total_s % 60:02d}"

    def find_landing_min(t_min: np.ndarray, alt_m: np.ndarray) -> float | None:
        apogee_idx = int(np.argmax(alt_m))
        idx = np.where(alt_m[apogee_idx:] <= 0)[0]
        return float(t_min[apogee_idx + idx[0]]) if len(idx) else None

    apogee_t_min = float(
        scenarios_min[active_keys[0]][0][
            np.argmax(scenarios_min[active_keys[0]][1])
        ]
    )

    landing_labels = {
        "nominal": "Nominal Landing",
        "ballistic": "Ballistic Landing",
        "drogue_only": "Drogue-only Landing",
        "premature_main": "Premature Main Landing",
    }

    vline_events: list[tuple[float, str]] = [
        (burnout_t_min, "Burnout"),
        (apogee_t_min, "Apogee"),
    ]
    for key in active_keys:
        t_land = find_landing_min(*scenarios_min[key])
        if t_land is not None:
            vline_events.append((t_land, landing_labels.get(key, f"{key} Landing")))

    vline_kw = dict(color="grey", linewidth=0.9, linestyle="--", zorder=0)
    bbox_kw = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)

    for idx, (t_min_ev, event_name) in enumerate(vline_events):
        ax_l.axvline(t_min_ev, **vline_kw)
        ax_r.axvline(t_min_ev, **vline_kw)

        ax = (
            ax_l
            if ax_l.get_xlim()[0] <= t_min_ev <= ax_l.get_xlim()[1]
            else ax_r
        )
        y_pos = alt_max_ft * 0.05

        timestamp = fmt_mmin(t_min_ev)
        combined_label = f"{event_name}\n{timestamp}"

        ax.text(
            t_min_ev, y_pos, combined_label,
            rotation=90, ha="left", va="bottom",
            fontsize=VLINE_LABEL_FONTSIZE, color="grey",
            bbox=bbox_kw,
        )

    active_labels = [SCENARIO_LABELS.get(k, k) for k in active_keys]
    ax_r.legend(
        handles=handles, labels=active_labels,
        loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray",
    )

    save_path = output_dir / "altitude_plot.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


# ===================================================================
# 5. Dispersion plot (§16.5)
# ===================================================================

def _ned_km_to_wm(
    north_km: float,
    east_km: float,
    lat0: float,
    lon0: float,
) -> tuple[float, float]:
    """Convert NED km offset from origin to Web Mercator (x, y)."""
    lat0_rad = math.radians(lat0)
    m_per_deg_lat = R_EARTH * math.pi / 180.0
    m_per_deg_lon = R_EARTH * math.cos(lat0_rad) * math.pi / 180.0
    lon = lon0 + (east_km * 1000.0) / m_per_deg_lon
    lat = lat0 + (north_km * 1000.0) / m_per_deg_lat
    return _TO_WM.transform(lon, lat)


def _fit_ellipse_threshold(
    points_ne: np.ndarray,
    threshold: float,
) -> dict:
    """Fit the smallest PCA ellipse containing *threshold* fraction of points.

    Parameters
    ----------
    points_ne : np.ndarray
        (N, 2) array of [north_km, east_km] landing points.
    threshold : float
        Fraction of points the ellipse must contain (e.g. 0.997).

    Returns
    -------
    dict
        Ellipse parameters: center_n, center_e, semi_a, semi_b, angle_deg.
    """
    if points_ne.ndim != 2 or points_ne.shape[1] != 2:
        raise ValueError("points_ne must be (N, 2): [north_km, east_km]")

    north, east = points_ne[:, 0], points_ne[:, 1]
    mean_e, mean_n = east.mean(), north.mean()

    # PCA via covariance eigen-decomposition
    cov = np.cov(np.vstack([east - mean_e, north - mean_n]))
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle_rad = np.arctan2(vecs[1, 0], vecs[0, 0])

    # Project points onto principal axes
    de, dn = east - mean_e, north - mean_n
    proj_a = de * vecs[0, 0] + dn * vecs[1, 0]  # along major axis
    proj_b = de * vecs[0, 1] + dn * vecs[1, 1]  # along minor axis

    # Mahalanobis-like distance: (proj_a / sigma_a)^2 + (proj_b / sigma_b)^2
    sigma_a = np.sqrt(vals[0]) if vals[0] > 0 else 1e-10
    sigma_b = np.sqrt(vals[1]) if vals[1] > 0 else 1e-10
    mahal_sq = (proj_a / sigma_a) ** 2 + (proj_b / sigma_b) ** 2

    # Find the scale factor that contains threshold fraction of points
    sorted_mahal_sq = np.sort(mahal_sq)
    idx = max(0, int(np.ceil(threshold * len(sorted_mahal_sq))) - 1)
    idx = min(idx, len(sorted_mahal_sq) - 1)
    scale = np.sqrt(sorted_mahal_sq[idx])

    return dict(
        center_n=float(mean_n),
        center_e=float(mean_e),
        semi_a=float(scale * sigma_a),
        semi_b=float(scale * sigma_b),
        angle_deg=float(np.degrees(angle_rad)),
    )


def save_dispersion_plot(
    results: list[SampleResult],
    sim_cfg: SimulationConfig,
    compliance_threshold: float,
    output_dir: Path,
    *,
    show_points: bool = False,
) -> Path:
    """Generate and save the landing dispersion plot.

    Parameters
    ----------
    results : list[SampleResult]
        All sample results from the Monte Carlo run.
    sim_cfg : SimulationConfig
        Simulation configuration (for site data).
    compliance_threshold : float
        Fraction of samples the ellipse must contain (from acceptance config).
    output_dir : Path
        Directory to save the plot into.
    show_points : bool
        If True, overlay individual apogee and landing points on the plot.

    Returns
    -------
    Path
        Path to the saved ``dispersion_plot.png``.
    """
    site = sim_cfg.site
    lat0, lon0 = site.latitude, site.longitude

    # --- Group landing points by scenario ---
    scenario_points: dict[str, list[tuple[float, float]]] = {}
    for r in results:
        pts = scenario_points.setdefault(r.scenario, [])
        pts.append((r.landing_north / 1000.0, r.landing_east / 1000.0))

    # --- Determine map extent ---
    all_north_km = [r.landing_north / 1000.0 for r in results]
    all_east_km = [r.landing_east / 1000.0 for r in results]
    margin = 2.0  # km padding
    extent_n = max(all_north_km) + margin
    extent_s = min(all_north_km) - margin
    extent_e = max(all_east_km) + margin
    extent_w = min(all_east_km) - margin

    # Ensure launch site is visible
    extent_n = max(extent_n, margin)
    extent_s = min(extent_s, -margin)
    extent_e = max(extent_e, margin)
    extent_w = min(extent_w, -margin)

    # Expand to include full danger area boundary
    danger_poly_extent = load_polygon_ned(site.danger_area, lat0, lon0)
    da_ext_e, da_ext_n = polygon_to_arrays(danger_poly_extent)
    extent_n = max(extent_n, np.max(da_ext_n) / 1000.0 + margin)
    extent_s = min(extent_s, np.min(da_ext_n) / 1000.0 - margin)
    extent_e = max(extent_e, np.max(da_ext_e) / 1000.0 + margin)
    extent_w = min(extent_w, np.min(da_ext_e) / 1000.0 - margin)

    # Expand to include monitour stations
    for obs in site.monitour_stations:
        ned = _lonlat_to_ned([(obs.longitude, obs.latitude)], lat0, lon0)
        obs_e_km, obs_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        r_km = obs.radius / 1000.0
        extent_n = max(extent_n, obs_n_km + r_km + margin)
        extent_s = min(extent_s, obs_n_km - r_km - margin)
        extent_e = max(extent_e, obs_e_km + r_km + margin)
        extent_w = min(extent_w, obs_e_km - r_km - margin)

    # Expand to include launch site monitour circle
    ls_obs_r_km = site.launch_monitour_radius / 1000.0
    extent_n = max(extent_n, ls_obs_r_km + margin)
    extent_s = min(extent_s, -ls_obs_r_km - margin)
    extent_e = max(extent_e, ls_obs_r_km + margin)
    extent_w = min(extent_w, -ls_obs_r_km - margin)

    grid_spacing_km = 5.0

    # --- Web Mercator helpers ---
    def km_to_wm(n_km: float, e_km: float) -> tuple[float, float]:
        return _ned_km_to_wm(n_km, e_km, lat0, lon0)

    origin_x, origin_y = km_to_wm(0.0, 0.0)
    xmin, ymin = km_to_wm(extent_s, extent_w)
    xmax, ymax = km_to_wm(extent_n, extent_e)

    # --- Figure ---
    ar = (extent_e - extent_w) / (extent_n - extent_s)
    base = 9
    figsize = (base, base / ar) if ar >= 1 else (base * ar, base)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_aspect("equal")
    legend_handles: list = []

    # --- Danger area + buffer ---
    danger_poly = load_polygon_ned(site.danger_area, lat0, lon0)
    buffer_dist = sim_cfg.monte_carlo.acceptance.buffer_distance

    # Convert danger area to Web Mercator for plotting
    da_e, da_n = polygon_to_arrays(danger_poly)
    da_wm = np.array([km_to_wm(n / 1000.0, e / 1000.0) for e, n in zip(da_e, da_n)])
    ax.plot(da_wm[:, 0], da_wm[:, 1], color="red", linewidth=1.0, linestyle="-", zorder=5)

    # Buffer ring — computed in NED, converted to WM only for rendering.
    # Fresh inward buffer without simplification (avoids the distortion that
    # buffer_danger_area's simplify() introduces for the acceptance check).
    smooth_m = _MIN_BUFFER_RADIUS_KM * 1000.0
    inner_ned = danger_poly.buffer(-buffer_dist)
    inner_ned = inner_ned.buffer(-smooth_m).buffer(+smooth_m)

    if not inner_ned.is_empty:
        ring_ned = danger_poly.difference(inner_ned)
        ring_parts = [ring_ned] if ring_ned.geom_type == "Polygon" else list(ring_ned.geoms)
        for part in ring_parts:
            # Convert exterior and interior rings from NED metres to WM
            outer_c = np.array([
                km_to_wm(n / 1000.0, e / 1000.0)
                for e, n in part.exterior.coords
            ])
            interiors = list(part.interiors)
            if interiors:
                inner_c = np.array([
                    km_to_wm(n / 1000.0, e / 1000.0)
                    for e, n in interiors[0].coords
                ])
                verts = np.concatenate([outer_c, inner_c])
                codes = np.concatenate([
                    [MplPath.MOVETO] + [MplPath.LINETO] * (len(outer_c) - 2) + [MplPath.CLOSEPOLY],
                    [MplPath.MOVETO] + [MplPath.LINETO] * (len(inner_c) - 2) + [MplPath.CLOSEPOLY],
                ])
            else:
                verts = outer_c
                codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(outer_c) - 2) + [MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(
                MplPath(verts, codes),
                facecolor="none", edgecolor="red", linewidth=0, zorder=6,
                hatch="....",
            ))

    legend_handles.append(mpatches.Patch(
        facecolor="none", edgecolor="red", linewidth=0,
        hatch="....", label=f"Buffer Zone ({buffer_dist / 1000.0:.0f} km)",
    ))

    # --- Monitour circles (unified into single shape, no edge) ---
    n_circle_pts = 360

    def _obs_circle_wm(
        center_n_km: float, center_e_km: float, radius_km: float,
    ) -> ShapelyPolygon:
        bearings = np.linspace(0, 2 * math.pi, n_circle_pts, endpoint=False)
        wm_pts = [
            km_to_wm(
                center_n_km + radius_km * math.cos(b),
                center_e_km + radius_km * math.sin(b),
            )
            for b in bearings
        ]
        return ShapelyPolygon(wm_pts)

    # Collect all monitour circles as Shapely polygons
    obs_polys: list[ShapelyPolygon] = []

    # Launch site monitour circle
    obs_polys.append(_obs_circle_wm(0.0, 0.0, site.launch_monitour_radius / 1000.0))

    # Configured monitour stations
    for obs in site.monitour_stations:
        ned = _lonlat_to_ned([(obs.longitude, obs.latitude)], lat0, lon0)
        obs_e_km, obs_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        obs_polys.append(_obs_circle_wm(obs_n_km, obs_e_km, obs.radius / 1000.0))

    # Union all circles into a single shape so overlapping areas don't stack
    from shapely.ops import unary_union
    obs_union = unary_union(obs_polys)
    obs_parts = [obs_union] if obs_union.geom_type == "Polygon" else list(obs_union.geoms)
    for part in obs_parts:
        xs, ys = part.exterior.xy
        ax.fill(xs, ys, facecolor="black", edgecolor="none", alpha=0.1, zorder=4)

    legend_handles.append(mpatches.Patch(
        facecolor="black", edgecolor="none", linewidth=0,
        label="Monitored Area", alpha=0.1,
    ))

    # --- Landing point ellipses per scenario ---
    ellipse_styles = ["-", "--", "-.", ":"]
    ellipse_colours = {
        "nominal": "green", "ballistic": "black",
        "drogue_only": "orange", "premature_main": "red",
    }

    for i, (key, pts_list) in enumerate(scenario_points.items()):
        pts_arr = np.array(pts_list)  # (N, 2): [north_km, east_km]
        if len(pts_arr) < 3:
            continue

        el = _fit_ellipse_threshold(pts_arr, compliance_threshold)
        colour = ellipse_colours.get(key, "grey")
        style = ellipse_styles[i % len(ellipse_styles)]
        label = SCENARIO_LABELS.get(key, key)

        ec_wm, nc_wm = km_to_wm(el["center_n"], el["center_e"])
        wm_a = abs(km_to_wm(el["center_n"], el["center_e"] + el["semi_a"])[0] - ec_wm)
        wm_b = abs(km_to_wm(el["center_n"] + el["semi_b"], el["center_e"])[1] - nc_wm)

        ax.add_patch(Ellipse(
            xy=(ec_wm, nc_wm), width=2 * wm_a, height=2 * wm_b,
            angle=el["angle_deg"], edgecolor=colour, facecolor="none",
            linewidth=2, alpha=0.6, zorder=7, linestyle=style,
        ))
        legend_handles.append(mpatches.Patch(
            facecolor="none", edgecolor=colour, alpha=0.6,
            linestyle=style, linewidth=2, label=label,
        ))

    # --- Scatter points (apogee + landing, when --points is set) ---
    if show_points:
        # Landing points — coloured per scenario
        for key, pts_list in scenario_points.items():
            pts_arr = np.array(pts_list)
            colour = ellipse_colours.get(key, "grey")
            wm_pts = np.array([
                km_to_wm(n, e) for n, e in pts_arr
            ])
            ax.scatter(
                wm_pts[:, 0], wm_pts[:, 1],
                s=1, c=colour, alpha=0.4, zorder=8, linewidths=0,
            )
        legend_handles.append(mlines.Line2D(
            [], [], marker="o", color="none",
            markerfacecolor="grey", markeredgecolor="none",
            markersize=3, linestyle="None", label="Landing Points",
        ))

        # Apogee points — uniform grey
        apogee_wm = np.array([
            km_to_wm(r.apogee_north / 1000.0, r.apogee_east / 1000.0)
            for r in results
        ])
        ax.scatter(
            apogee_wm[:, 0], apogee_wm[:, 1],
            s=1, c="blue", alpha=0.3, zorder=8, linewidths=0,
        )
        legend_handles.append(mlines.Line2D(
            [], [], marker="o", color="none",
            markerfacecolor="blue", markeredgecolor="none",
            markersize=3, linestyle="None", label="Apogee Points",
        ))

    # --- Map markers (unique per location) ---
    # Marker pool for monitour stations and map markers
    _marker_pool = ["s", "D", "^", "v", "p", "h", "8", "*"]
    _marker_idx = 0

    # Launch site
    ls_x, ls_y = km_to_wm(0.0, 0.0)
    ax.plot(
        ls_x, ls_y, marker="x", color="black",
        markerfacecolor="none", markeredgecolor="black",
        markersize=8, markeredgewidth=2.5,
        linestyle="None", zorder=10,
    )
    legend_handles.append(mlines.Line2D(
        [], [], marker="x", color="none",
        markerfacecolor="none", markeredgecolor="black",
        markeredgewidth=2.5, markersize=8,
        linestyle="None", label="Launch Site",
    ))

    # Monitour station markers (each gets a unique symbol)
    for obs in site.monitour_stations:
        ned = _lonlat_to_ned([(obs.longitude, obs.latitude)], lat0, lon0)
        obs_e_km, obs_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        mx, my = km_to_wm(obs_n_km, obs_e_km)
        mk_symbol = _marker_pool[_marker_idx % len(_marker_pool)]
        _marker_idx += 1
        ax.plot(
            mx, my, marker=mk_symbol, color="black",
            markerfacecolor="black", markeredgecolor="black",
            markersize=5, markeredgewidth=2.5,
            linestyle="None", zorder=10,
        )
        legend_handles.append(mlines.Line2D(
            [], [], marker=mk_symbol, color="none",
            markerfacecolor="black", markeredgecolor="black",
            markeredgewidth=2.5, markersize=5,
            linestyle="None", label=obs.name,
        ))

    # Configured map markers (each gets a unique symbol)
    for mk in site.map_markers:
        ned = _lonlat_to_ned([(mk.longitude, mk.latitude)], lat0, lon0)
        mk_e_km, mk_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        mx, my = km_to_wm(mk_n_km, mk_e_km)
        mk_symbol = _marker_pool[_marker_idx % len(_marker_pool)]
        _marker_idx += 1
        ax.plot(
            mx, my, marker=mk_symbol, color="black",
            markerfacecolor="black", markeredgecolor="black",
            markersize=5, markeredgewidth=2.5,
            linestyle="None", zorder=10,
        )
        legend_handles.append(mlines.Line2D(
            [], [], marker=mk_symbol, color="none",
            markerfacecolor="black", markeredgecolor="black",
            markeredgewidth=2.5, markersize=5,
            linestyle="None", label=mk.name,
        ))

    # --- Basemap ---
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    tile_cache = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".map_cache",
    )
    os.makedirs(tile_cache, exist_ok=True)
    cx.set_cache_dir(tile_cache)

    fig.canvas.draw()
    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=_OS_TILE_URL, zorder=1, zoom_adjust=0)
    except Exception:
        warnings.warn(
            "Could not fetch OS Maps basemap tiles. "
            "Plot saved without basemap.",
            stacklevel=2,
        )

    # --- Axis ticks in km ---
    wm_per_km_e = km_to_wm(0.0, 1.0)[0] - origin_x
    wm_per_km_n = km_to_wm(1.0, 0.0)[1] - origin_y

    x_ticks_km = np.arange(
        math.ceil(extent_w / grid_spacing_km) * grid_spacing_km,
        extent_e + grid_spacing_km,
        grid_spacing_km,
    )
    y_ticks_km = np.arange(
        math.ceil(extent_s / grid_spacing_km) * grid_spacing_km,
        extent_n + grid_spacing_km,
        grid_spacing_km,
    )

    ax.set_xticks([origin_x + km * wm_per_km_e for km in x_ticks_km])
    ax.set_xticklabels([f"{km:.0f}" for km in x_ticks_km])
    ax.set_yticks([origin_y + km * wm_per_km_n for km in y_ticks_km])
    ax.set_yticklabels([f"{km:.0f}" for km in y_ticks_km])

    ax.set_xlabel("East (km)", fontsize=12, labelpad=8)
    ax.set_ylabel("North (km)", fontsize=12, labelpad=8)
    ax.grid(
        True, which="major", linestyle="--",
        linewidth=0.5, alpha=0.7, zorder=2,
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9).set_zorder(20)

    save_path = output_dir / "dispersion_plot.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


# ===================================================================
# 6. Replay plot stubs (§16.4)
# ===================================================================

def save_replay_3d(
    replayed: list[SampleResult],
    output_dir: Path,
) -> Path:
    """Generate 3D isometric replay plot. Not yet implemented."""
    raise NotImplementedError("Replay 3D isometric plot not yet implemented")


def save_replay_plan_view(
    replayed: list[SampleResult],
    output_dir: Path,
) -> Path:
    """Generate plan-view replay plot. Not yet implemented."""
    raise NotImplementedError("Replay plan view plot not yet implemented")


def save_replay_altitude(
    replayed: list[SampleResult],
    output_dir: Path,
) -> Path:
    """Generate altitude-time replay plot. Not yet implemented."""
    raise NotImplementedError("Replay altitude-time plot not yet implemented")
