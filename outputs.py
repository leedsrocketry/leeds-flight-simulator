"""outputs.py — CSV/YAML serialisation and plot generation (§16).

Public API
----------
create_results_dir      — create/clear fixed results directory
write_samples_csv       — per-sample CSV (§16.1)
write_summary_yaml      — run summary YAML (§16.2)
save_altitude_plot      — altitude-time plot (§16.5)
save_dispersion_plot    — landing dispersion plot (§16.5)
save_replay_3d          — 3D isometric replay plot (§16.4)
save_replay_plan_view   — plan-view replay plot (§16.4)
save_replay_altitude    — altitude-time replay plot (§16.4)
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
    has_monitor: bool,
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
    has_monitor : bool
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
    if has_monitor:
        header.append("monitor_compliant")
    header.extend([
        "stability_compliant",
        "min_sm_subsonic_cal", "min_sm_supersonic_cal",
        "max_aoa_deg", "max_mach", "apogee_m",
        "landing_lat_deg", "landing_lon_deg",
        "landing_north_m", "landing_east_m",
        "apogee_lat_deg", "apogee_lon_deg",
        "apogee_north_m", "apogee_east_m",
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
            if has_monitor:
                row.append(r.in_coverage)
            row.extend([
                r.stability_compliant,
                r.min_sm_subsonic, r.min_sm_supersonic,
                r.max_aoa_deg, r.peak_mach, r.apogee_m,
                r.landing_lat, r.landing_lon,
                r.landing_north, r.landing_east,
                r.apogee_lat, r.apogee_lon,
                r.apogee_north, r.apogee_east,
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
            "inclination_selected": opt_result.inclination_selected,
            "feasible_azimuths": opt_result.narrowing_feasible,
            "azimuth_observations": [
                [az, float(p)] for az, p in opt_result.azimuth_observations
            ],
            "top_candidates": opt_result.azimuth_top_candidates,
            "validation_compliance": {
                int(k): float(v)
                for k, v in opt_result.validation_compliance.items()
            },
            "validation_margins": {
                int(k): float(v)
                for k, v in opt_result.validation_margins.items()
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
    output_dir: Path | None = None,
) -> Path | plt.Figure:
    """Generate and save the altitude-time plot.

    Parameters
    ----------
    scenarios : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping of scenario key → (time_s, altitude_m) arrays.
        Only active scenarios need be present.
    burnout_time_s : float
        Motor burnout time in seconds.
    output_dir : Path or None
        Directory to save the plot into.  If *None*, return the
        ``Figure`` for interactive display.

    Returns
    -------
    Path or Figure
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

    if output_dir is not None:
        save_path = output_dir / "altitude_plot.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return fig


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


def _build_map_axes(
    ax: plt.Axes,
    sim_cfg: SimulationConfig,
    *,
    extra_north_km: np.ndarray | None = None,
    extra_east_km: np.ndarray | None = None,
    show_buffer: bool = True,
    grid_spacing_km: float = 5.0,
) -> tuple:
    """Set up a 2D map axes with danger area, monitors, markers, basemap.

    Draws the danger area boundary (with optional buffer ring), monitor
    coverage circles, launch site cross, monitor station markers, map
    markers, OS Maps basemap tiles, km-labelled grid, and axis labels.

    Parameters
    ----------
    ax : Axes
        A matplotlib 2D axes.
    sim_cfg : SimulationConfig
        Simulation configuration (provides site data and buffer distance).
    extra_north_km, extra_east_km : array-like or None
        Additional points (in km from launch site) to include when
        computing the map extent.
    show_buffer : bool
        If True, draw the hatched buffer ring inside the danger area.
    grid_spacing_km : float
        Grid-line spacing in km.

    Returns
    -------
    (km_to_wm, legend_handles)
        ``km_to_wm(north_km, east_km)`` converts NED km to Web Mercator,
        ``legend_handles`` is a list of artists for the legend.
    """
    from shapely.ops import unary_union

    site = sim_cfg.site
    lat0, lon0 = site.latitude, site.longitude

    def km_to_wm(n_km: float, e_km: float) -> tuple[float, float]:
        return _ned_km_to_wm(n_km, e_km, lat0, lon0)

    # --- Compute map extent ---
    margin = 2.0

    if extra_north_km is not None and len(extra_north_km) > 0:
        extent_n = float(np.max(extra_north_km)) + margin
        extent_s = float(np.min(extra_north_km)) - margin
    else:
        extent_n = margin
        extent_s = -margin

    if extra_east_km is not None and len(extra_east_km) > 0:
        extent_e = float(np.max(extra_east_km)) + margin
        extent_w = float(np.min(extra_east_km)) - margin
    else:
        extent_e = margin
        extent_w = -margin

    # Ensure launch site is visible
    extent_n = max(extent_n, margin)
    extent_s = min(extent_s, -margin)
    extent_e = max(extent_e, margin)
    extent_w = min(extent_w, -margin)

    # Expand to include full danger area boundary
    danger_poly = load_polygon_ned(site.danger_area, lat0, lon0)
    da_e, da_n = polygon_to_arrays(danger_poly)
    extent_n = max(extent_n, np.max(da_n) / 1000.0 + margin)
    extent_s = min(extent_s, np.min(da_n) / 1000.0 - margin)
    extent_e = max(extent_e, np.max(da_e) / 1000.0 + margin)
    extent_w = min(extent_w, np.min(da_e) / 1000.0 - margin)

    # Expand to include monitor stations
    for obs in site.monitor_stations:
        ned = _lonlat_to_ned([(obs.longitude, obs.latitude)], lat0, lon0)
        obs_e_km, obs_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        r_km = obs.radius / 1000.0
        extent_n = max(extent_n, obs_n_km + r_km + margin)
        extent_s = min(extent_s, obs_n_km - r_km - margin)
        extent_e = max(extent_e, obs_e_km + r_km + margin)
        extent_w = min(extent_w, obs_e_km - r_km - margin)

    # Expand to include launch site monitor circle
    ls_obs_r_km = site.launch_monitor_radius / 1000.0
    extent_n = max(extent_n, ls_obs_r_km + margin)
    extent_s = min(extent_s, -ls_obs_r_km - margin)
    extent_e = max(extent_e, ls_obs_r_km + margin)
    extent_w = min(extent_w, -ls_obs_r_km - margin)

    origin_x, origin_y = km_to_wm(0.0, 0.0)
    xmin, ymin = km_to_wm(extent_s, extent_w)
    xmax, ymax = km_to_wm(extent_n, extent_e)

    ax.set_aspect("equal")
    legend_handles: list = []

    # --- Danger area + buffer ---
    buffer_dist = sim_cfg.monte_carlo.acceptance.buffer_distance

    da_wm = np.array([km_to_wm(n / 1000.0, e / 1000.0) for e, n in zip(da_e, da_n)])
    ax.plot(da_wm[:, 0], da_wm[:, 1], color="red", linewidth=1.0, linestyle="-", zorder=5)

    if show_buffer:
        smooth_m = _MIN_BUFFER_RADIUS_KM * 1000.0
        inner_ned = danger_poly.buffer(-buffer_dist)
        inner_ned = inner_ned.buffer(-smooth_m).buffer(+smooth_m)

        if not inner_ned.is_empty:
            ring_ned = danger_poly.difference(inner_ned)
            ring_parts = [ring_ned] if ring_ned.geom_type == "Polygon" else list(ring_ned.geoms)
            for part in ring_parts:
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
            hatch="....", label=f"Buffer Zone ({buffer_dist / 1000.0:.1f} km)",
        ))

    # --- Monitor circles (unified into single shape, no edge) ---
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

    obs_polys: list[ShapelyPolygon] = []
    obs_polys.append(_obs_circle_wm(0.0, 0.0, site.launch_monitor_radius / 1000.0))
    for obs in site.monitor_stations:
        ned = _lonlat_to_ned([(obs.longitude, obs.latitude)], lat0, lon0)
        obs_e_km, obs_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        obs_polys.append(_obs_circle_wm(obs_n_km, obs_e_km, obs.radius / 1000.0))

    obs_union = unary_union(obs_polys)
    obs_parts = [obs_union] if obs_union.geom_type == "Polygon" else list(obs_union.geoms)
    for part in obs_parts:
        xs, ys = part.exterior.xy
        ax.fill(xs, ys, facecolor="black", edgecolor="none", alpha=0.1, zorder=4)

    legend_handles.append(mpatches.Patch(
        facecolor="black", edgecolor="none", linewidth=0,
        label="Monitored Area", alpha=0.1,
    ))

    # --- Map markers ---
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

    # Monitor station markers
    for obs in site.monitor_stations:
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

    # Configured map markers
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

    ax.figure.canvas.draw()
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

    return km_to_wm, legend_handles


def save_dispersion_plot(
    results: list[SampleResult],
    sim_cfg: SimulationConfig,
    compliance_threshold: float,
    output_dir: Path | None = None,
    *,
    show_points: bool = False,
) -> Path | plt.Figure:
    """Generate and save the landing dispersion plot.

    Parameters
    ----------
    results : list[SampleResult]
        All sample results from the Monte Carlo run.
    sim_cfg : SimulationConfig
        Simulation configuration (for site data).
    compliance_threshold : float
        Fraction of samples the ellipse must contain (from acceptance config).
    output_dir : Path or None
        Directory to save the plot into.  If *None*, return the
        ``Figure`` for interactive display.
    show_points : bool
        If True, overlay individual apogee and landing points on the plot.

    Returns
    -------
    Path or Figure
        Path to the saved ``dispersion_plot.png``, or the ``Figure``.
    """
    # --- Group landing points by scenario ---
    scenario_points: dict[str, list[tuple[float, float]]] = {}
    for r in results:
        pts = scenario_points.setdefault(r.scenario, [])
        pts.append((r.landing_north / 1000.0, r.landing_east / 1000.0))

    all_north_km = np.array([r.landing_north / 1000.0 for r in results])
    all_east_km = np.array([r.landing_east / 1000.0 for r in results])

    # --- Build map ---
    ar_n = float(np.ptp(all_north_km)) + 4.0 if len(all_north_km) > 0 else 4.0
    ar_e = float(np.ptp(all_east_km)) + 4.0 if len(all_east_km) > 0 else 4.0
    ar = ar_e / ar_n if ar_n > 0 else 1.0
    base = 9
    figsize = (base, base / ar) if ar >= 1 else (base * ar, base)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    km_to_wm, legend_handles = _build_map_axes(
        ax, sim_cfg,
        extra_north_km=all_north_km,
        extra_east_km=all_east_km,
    )

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

    ax.legend(handles=legend_handles, loc="upper right", fontsize=9).set_zorder(20)

    if output_dir is not None:
        save_path = output_dir / "dispersion_plot.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return fig


# ===================================================================
# 6. Replay plots (§16.4)
# ===================================================================


def _rdp_simplify_3d(
    points: np.ndarray,
    epsilon_km: float,
) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification in 3D.

    Parameters
    ----------
    points
        Shape ``(n, 3)`` array of (north_km, east_km, alt_km) coordinates.
    epsilon_km
        Maximum permitted perpendicular deviation in km.  Points whose
        removal would introduce less than this error are discarded.

    Returns
    -------
    Boolean mask of shape ``(n,)``; ``True`` where a point is kept.
    First and last points are always kept.
    """
    n = len(points)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    if n <= 2:
        return keep

    # Iterative stack-based RDP (avoids Python recursion limit on long runs)
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue

        p0 = points[start]
        p1 = points[end]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)

        interior = points[start + 1 : end]  # shape (m, 3)
        if seg_len < 1e-15:
            # Degenerate: all points collapse to p0
            dists = np.linalg.norm(interior - p0, axis=1)
        else:
            seg_unit = seg / seg_len
            vecs = interior - p0
            proj = vecs @ seg_unit
            perp = vecs - np.outer(proj, seg_unit)
            dists = np.linalg.norm(perp, axis=1)

        max_idx = int(np.argmax(dists))
        if dists[max_idx] > epsilon_km:
            mid = start + 1 + max_idx
            keep[mid] = True
            stack.append((start, mid))
            stack.append((mid, end))

    return keep


def _extract_replay_trajectory(
    sr: SampleResult,
    max_distortion_m: float = 10.0,
    hard_max_points: int = 100,
) -> dict | None:
    """Extract full NED trajectory arrays from a replayed sample.

    Returns a dict with keys ``north_km``, ``east_km``, ``alt_m``,
    ``t_s``, ``colour``, ``label``, ``is_terminated``, or *None* if the
    sample has no trajectory attached.

    The trajectory is simplified using Ramer-Douglas-Peucker in 3-D so
    that no original point deviates more than *max_distortion_m* metres
    from the rendered polyline.  This uses the minimum number of points
    needed to represent each trajectory faithfully — longer or more
    curved paths automatically receive more points than short straight
    ones.  *hard_max_points* caps the output as a safety net.
    """
    if sr.trajectory is None:
        return None

    traj = sr.trajectory

    north_asc = traj.state_ascent[:, 0] / 1000.0
    east_asc = traj.state_ascent[:, 1] / 1000.0
    alt_asc = -traj.state_ascent[:, 2]
    t_asc = traj.t_ascent

    if traj.t_descent is not None and traj.n_descent > 0:
        n = traj.n_descent
        t_desc = traj.t_descent[:n]
        north_desc = traj.state_descent[:n, 0] / 1000.0
        east_desc = traj.state_descent[:n, 1] / 1000.0
        alt_desc = -traj.state_descent[:n, 2]
        # Skip overlapping apogee point in descent arrays
        north_km = np.concatenate([north_asc, north_desc[1:]])
        east_km = np.concatenate([east_asc, east_desc[1:]])
        alt_m = np.concatenate([alt_asc, alt_desc[1:]])
        t_s = np.concatenate([t_asc, t_desc[1:]])
    else:
        north_km = north_asc
        east_km = east_asc
        alt_m = alt_asc
        t_s = t_asc

    # RDP simplification: keep minimum points within distortion budget
    if len(north_km) > 2:
        pts = np.column_stack([north_km, east_km, alt_m / 1000.0])
        mask = _rdp_simplify_3d(pts, epsilon_km=max_distortion_m / 1000.0)
        # Safety cap: if RDP still yields too many points, uniformly thin
        kept = np.where(mask)[0]
        if len(kept) > hard_max_points:
            sub = np.linspace(0, len(kept) - 1, hard_max_points, dtype=int)
            mask = np.zeros(len(north_km), dtype=bool)
            mask[kept[sub]] = True
            mask[0] = True
            mask[-1] = True
        north_km = north_km[mask]
        east_km = east_km[mask]
        alt_m = alt_m[mask]
        t_s = t_s[mask]

    is_terminated = not sr.stability_compliant
    colour = "deeppink" if is_terminated else SCENARIO_COLOURS.get(sr.scenario, "grey")
    label = SCENARIO_LABELS.get(sr.scenario, sr.scenario)

    return {
        "north_km": north_km,
        "east_km": east_km,
        "alt_m": alt_m,
        "t_s": t_s,
        "colour": colour,
        "label": label,
        "is_terminated": is_terminated,
    }


def _replay_line_style(n_lines: int) -> tuple[float, float]:
    """Return (linewidth, alpha) scaled to the number of trajectories.

    For ~50 lines use the old style (lw=1.8, alpha=0.8).
    For ~1000 lines use thin, transparent lines (lw=0.3, alpha=0.08).
    Interpolates log-linearly between those anchors, clamped at both ends.
    """
    import math
    if n_lines <= 50:
        return 1.8, 0.8
    if n_lines >= 1000:
        return 0.3, 0.08
    # Log-linear interpolation between the two anchors
    t = (math.log(n_lines) - math.log(50)) / (math.log(1000) - math.log(50))
    lw = 1.8 + t * (0.3 - 1.8)
    alpha = 0.8 + t * (0.08 - 0.8)
    return lw, alpha


def save_replay_altitude(
    replayed: list[SampleResult],
    sim_cfg: SimulationConfig,
    *,
    output_dir: Path | None = None,
) -> Path | plt.Figure:
    """Generate altitude-time replay plot.

    Parameters
    ----------
    replayed : list[SampleResult]
        Replayed samples with trajectory data attached.
    sim_cfg : SimulationConfig
        Simulation configuration.
    output_dir : Path or None
        If given, save the figure to ``replay_altitude.png`` in this
        directory and return the path.  If *None*, return the
        ``Figure`` for interactive display.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract trajectories and count per legend group
    trajectories: list[dict] = []
    for sr in replayed:
        t = _extract_replay_trajectory(sr)
        if t is not None:
            trajectories.append(t)

    legend_counts: dict[str, int] = {}
    for t in trajectories:
        key = "Terminated" if t["is_terminated"] else t["label"]
        legend_counts[key] = legend_counts.get(key, 0) + 1

    lw, alpha = _replay_line_style(len(trajectories))
    legend_seen: set[str] = set()
    legend_handles: list = []

    for t in trajectories:
        alt_ft = t["alt_m"] * M_TO_FT
        t_min = t["t_s"] * S_TO_MIN

        legend_key = "Terminated" if t["is_terminated"] else t["label"]
        show_label = legend_key not in legend_seen
        legend_seen.add(legend_key)

        ax.plot(
            t_min, alt_ft,
            color=t["colour"],
            linewidth=lw,
            alpha=alpha,
        )
        if show_label:
            legend_handles.append(mlines.Line2D(
                [], [], color=t["colour"], linewidth=2.0,
                label=f"{legend_key} (n={legend_counts[legend_key]})",
            ))

    # Primary y-axis (feet)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ft_formatter))
    ax.set_ylabel("Altitude (ft)", fontsize=11)
    ax.set_xlabel("Flight Time (min)", fontsize=11)

    # Secondary y-axis (km)
    ax_km = ax.twinx()
    ax_km.set_ylim(ax.get_ylim())
    ax_km.yaxis.set_major_formatter(mticker.FuncFormatter(_km_formatter))
    ax_km.set_ylabel("Altitude (km)", fontsize=11)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9, edgecolor="gray")

    fig.tight_layout()

    if output_dir is not None:
        save_path = output_dir / "replay_altitude.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return fig


def save_replay_3d(
    replayed: list[SampleResult],
    sim_cfg: SimulationConfig,
    *,
    output_dir: Path | None = None,
) -> Path | plt.Figure:
    """Generate 3D isometric replay plot (§16.4).

    Trajectories in NED km with altitude in metres.  Danger area, buffer
    ring, monitor coverage, and markers drawn on the z=0 ground plane,
    matching the dispersion/plan-view plot features.  Coloured by descent
    scenario; pink if terminated early.
    """
    from shapely.ops import unary_union

    site = sim_cfg.site
    lat0, lon0 = site.latitude, site.longitude

    # --- Extract trajectories ---
    trajectories: list[dict] = []
    for sr in replayed:
        t = _extract_replay_trajectory(sr)
        if t is not None:
            trajectories.append(t)

    if not trajectories:
        fig = plt.figure(figsize=(10, 8))
        if output_dir is not None:
            save_path = output_dir / "replay_3d.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        return fig

    # --- Figure ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=-60)

    legend_handles: list = []

    # --- Danger area + buffer on ground plane ---
    danger_poly = load_polygon_ned(site.danger_area, lat0, lon0)
    da_e, da_n = polygon_to_arrays(danger_poly)
    da_e_km = da_e / 1000.0
    da_n_km = da_n / 1000.0
    ax.plot(
        da_e_km, da_n_km, zs=0, zdir="z",
        color="red", linewidth=1.0, linestyle="-", zorder=5,
    )

    buffer_dist = sim_cfg.monte_carlo.acceptance.buffer_distance
    smooth_m = _MIN_BUFFER_RADIUS_KM * 1000.0
    inner_ned = danger_poly.buffer(-buffer_dist)
    inner_ned = inner_ned.buffer(-smooth_m).buffer(+smooth_m)
    if not inner_ned.is_empty:
        inner_e, inner_n = polygon_to_arrays(inner_ned)
        ax.plot(
            inner_e / 1000.0, inner_n / 1000.0, zs=0, zdir="z",
            color="red", linewidth=0.6, linestyle=":", alpha=0.6, zorder=5,
        )
        legend_handles.append(mlines.Line2D(
            [], [], color="red", linewidth=0.6, linestyle=":",
            alpha=0.6, label=f"Buffer Zone ({buffer_dist / 1000.0:.1f} km)",
        ))

    # --- Coastline on ground plane (clipped to plot extents) ---
    if site.coastline is not None:
        from shapely.geometry import box as shapely_box

        coastline_poly = load_polygon_ned(site.coastline, lat0, lon0)

        # Compute plot extents from trajectories + danger area
        all_north = np.concatenate([t["north_km"] for t in trajectories])
        all_east = np.concatenate([t["east_km"] for t in trajectories])
        margin = 2.0
        clip_n = max(float(np.max(all_north)), np.max(da_n_km)) + margin
        clip_s = min(float(np.min(all_north)), np.min(da_n_km)) - margin
        clip_e = max(float(np.max(all_east)), np.max(da_e_km)) + margin
        clip_w = min(float(np.min(all_east)), np.min(da_e_km)) - margin

        # Clip polygon (in NED metres) to extent box
        clip_box = shapely_box(
            clip_w * 1000.0, clip_s * 1000.0,
            clip_e * 1000.0, clip_n * 1000.0,
        )
        clipped = coastline_poly.intersection(clip_box)

        def _plot_coastline_geom(geom) -> None:
            if geom.is_empty:
                return
            if geom.geom_type == "Polygon":
                ce, cn = polygon_to_arrays(geom)
                ax.plot(
                    ce / 1000.0, cn / 1000.0, zs=0, zdir="z",
                    color="blue", linewidth=0.8, alpha=0.6, zorder=4,
                )
            elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
                for part in geom.geoms:
                    _plot_coastline_geom(part)

        _plot_coastline_geom(clipped)

    # --- Monitor circles on ground plane ---
    n_circle_pts = 360
    bearings = np.linspace(0, 2 * math.pi, n_circle_pts, endpoint=True)

    def _plot_circle_3d(
        center_n_km: float, center_e_km: float, radius_km: float,
    ) -> None:
        cn = center_n_km + radius_km * np.cos(bearings)
        ce = center_e_km + radius_km * np.sin(bearings)
        ax.plot(
            ce, cn, zs=0, zdir="z",
            color="black", linewidth=0.5, alpha=0.2, zorder=3,
        )

    _plot_circle_3d(0.0, 0.0, site.launch_monitor_radius / 1000.0)
    for obs in site.monitor_stations:
        ned = _lonlat_to_ned([(obs.longitude, obs.latitude)], lat0, lon0)
        obs_e_km, obs_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        _plot_circle_3d(obs_n_km, obs_e_km, obs.radius / 1000.0)

    # --- Launch site marker on ground ---
    ax.plot(
        [0.0], [0.0], [0.0], marker="x", color="black",
        markerfacecolor="none", markeredgecolor="black",
        markersize=8, markeredgewidth=2.5,
        linestyle="None", zorder=10,
    )

    # --- Monitor station and map markers on ground ---
    _marker_pool = ["s", "D", "^", "v", "p", "h", "8", "*"]
    _marker_idx = 0

    legend_handles.append(mlines.Line2D(
        [], [], marker="x", color="none",
        markerfacecolor="none", markeredgecolor="black",
        markeredgewidth=2.5, markersize=8,
        linestyle="None", label="Launch Site",
    ))

    for obs in site.monitor_stations:
        ned = _lonlat_to_ned([(obs.longitude, obs.latitude)], lat0, lon0)
        obs_e_km, obs_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        mk_symbol = _marker_pool[_marker_idx % len(_marker_pool)]
        _marker_idx += 1
        ax.plot(
            [obs_e_km], [obs_n_km], [0.0],
            marker=mk_symbol, color="black",
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

    for mk in site.map_markers:
        ned = _lonlat_to_ned([(mk.longitude, mk.latitude)], lat0, lon0)
        mk_e_km, mk_n_km = ned[0][0] / 1000.0, ned[0][1] / 1000.0
        mk_symbol = _marker_pool[_marker_idx % len(_marker_pool)]
        _marker_idx += 1
        ax.plot(
            [mk_e_km], [mk_n_km], [0.0],
            marker=mk_symbol, color="black",
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

    # --- Count trajectories per legend group ---
    legend_counts: dict[str, int] = {}
    for t in trajectories:
        key = "Terminated" if t["is_terminated"] else t["label"]
        legend_counts[key] = legend_counts.get(key, 0) + 1

    lw, alpha = _replay_line_style(len(trajectories))

    # --- Plot trajectories ---
    legend_seen: set[str] = set()
    for t in trajectories:
        legend_key = "Terminated" if t["is_terminated"] else t["label"]
        show_label = legend_key not in legend_seen
        legend_seen.add(legend_key)

        ax.plot(
            t["east_km"], t["north_km"], t["alt_m"] / 1000.0,
            color=t["colour"], linewidth=lw, alpha=alpha,
        )
        if show_label:
            legend_handles.append(mlines.Line2D(
                [], [], color=t["colour"], linewidth=2.0,
                label=f"{legend_key} (n={legend_counts[legend_key]})",
            ))

    # Equal scale on all three axes (altitude may extend further)
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    z_range = ax.get_zlim()[1] - ax.get_zlim()[0]
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([
        x_range / max_range,
        y_range / max_range,
        z_range / max_range,
    ])

    ax.set_xlabel("East (km)", fontsize=10, labelpad=10)
    ax.set_ylabel("North (km)", fontsize=10, labelpad=10)
    ax.set_zlabel("Altitude (km)", fontsize=10, labelpad=10)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.tight_layout()

    if output_dir is not None:
        save_path = output_dir / "replay_3d.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return fig


def save_replay_plan_view(
    replayed: list[SampleResult],
    sim_cfg: SimulationConfig,
    *,
    output_dir: Path | None = None,
) -> Path | plt.Figure:
    """Generate plan-view replay plot (§16.4).

    2D map with trajectories overlaid, using the same map, danger area,
    monitor zones, and markers as the dispersion plot.  Coloured by
    descent scenario; pink if terminated early.
    """
    # --- Extract trajectories ---
    trajectories: list[dict] = []
    for sr in replayed:
        t = _extract_replay_trajectory(sr)
        if t is not None:
            trajectories.append(t)

    # Collect all trajectory points for extent computation
    if trajectories:
        all_north = np.concatenate([t["north_km"] for t in trajectories])
        all_east = np.concatenate([t["east_km"] for t in trajectories])
    else:
        all_north = np.array([0.0])
        all_east = np.array([0.0])

    # --- Build map ---
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)

    km_to_wm, legend_handles = _build_map_axes(
        ax, sim_cfg,
        extra_north_km=all_north,
        extra_east_km=all_east,
        show_buffer=True,
    )

    # --- Count trajectories per legend group ---
    legend_counts: dict[str, int] = {}
    for t in trajectories:
        key = "Terminated" if t["is_terminated"] else t["label"]
        legend_counts[key] = legend_counts.get(key, 0) + 1

    lw, alpha = _replay_line_style(len(trajectories))

    # --- Plot trajectories ---
    legend_seen: set[str] = set()
    for t in trajectories:
        wm_pts = np.array([
            km_to_wm(n, e) for n, e in zip(t["north_km"], t["east_km"])
        ])

        legend_key = "Terminated" if t["is_terminated"] else t["label"]
        show_label = legend_key not in legend_seen
        legend_seen.add(legend_key)

        ax.plot(
            wm_pts[:, 0], wm_pts[:, 1],
            color=t["colour"], linewidth=lw, alpha=alpha, zorder=8,
        )
        if show_label:
            legend_handles.append(mlines.Line2D(
                [], [], color=t["colour"], linewidth=2.0,
                label=f"{legend_key} (n={legend_counts[legend_key]})",
            ))

    ax.legend(handles=legend_handles, loc="upper right", fontsize=9).set_zorder(20)

    if output_dir is not None:
        save_path = output_dir / "replay_plan_view.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return fig
