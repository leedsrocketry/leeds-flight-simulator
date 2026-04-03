"""Monte Carlo orchestration, parallelism, acceptance checking, and replay.

Specification references: §11, §12, §16, §17.

Public API
----------
Startup / helpers:
    build_sim_params       — construct SimParams from configs + stochastic draws
    generate_sample_draws  — deterministic stochastic draws from SeedSequence

Single-sample runner:
    run_sample             — draw inputs, run trajectory, check acceptance → SampleResult

Orchestration:
    run_monte_carlo        — top-level: all scenarios, parallel, aggregate results.
                             Uses hand-over parallelisation: individual samples from
                             all scenarios are dispatched to a shared process pool so
                             that cores freed by shorter scenarios automatically pick
                             up remaining work from longer ones.
    run_scenario           — run N samples for one scenario sequentially (used by replay)

Replay:
    replay_sample          — reconstruct and re-run a single sample
    replay_non_compliant   — replay all non-compliant samples from a results directory
    REASON_COLS            — mapping of reason keyword → CSV compliance column name
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import (
    SimulationConfig,
    Vehicle,
    load_simulation_config,
    load_vehicle,
)
from atmosphere import compute_t_offset, temperature as _isa_temperature
from motor import PropellantModel
from aerodynamics import AeroModel, build_aero_model
from wind import WindEnsemble, load_wind_ensemble
from dynamics import (
    SimParams,
    FlightSummary,
    TrajectoryProfile,
    check_stability_compliance,
    run_trajectory,
    SCENARIO_MAP,
)
from geography import (
    load_polygon_ned,
    buffer_danger_area,
    polygon_to_arrays,
    prepare_zone,
    check_coastline,
    check_monitor_coverage,
    ned_to_latlon,
    R_EARTH,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEG2RAD: float = math.pi / 180.0
_RAD2DEG: float = 180.0 / math.pi

# Scenario display names (for output tables / CSVs)
SCENARIO_LABELS: dict[str, str] = {
    "nominal": "Nominal",
    "ballistic": "Ballistic",
    "drogue_only": "Drogue-only",
    "premature_main": "Premature Main",
}


# ---------------------------------------------------------------------------
# Sample result
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    """Output of a single Monte Carlo sample, including acceptance checks."""
    sample_id: int
    scenario: str
    run_index: int

    # Trajectory outputs
    apogee_m: float
    apogee_lat: float
    apogee_lon: float
    landing_lat: float
    landing_lon: float
    apogee_north: float         # NED metres (for dispersion plot)
    apogee_east: float          # NED metres
    landing_north: float        # NED metres (for dispersion plot)
    landing_east: float         # NED metres
    flight_time_s: float
    peak_mach: float
    max_aoa_deg: float
    min_sm_subsonic: float     # calibres
    min_sm_supersonic: float   # calibres

    # Acceptance (overall compliance assembled here from FlightSummary + post-sim checks)
    compliant: bool
    footprint_compliant: bool
    ceiling_compliant: bool
    stability_compliant: bool
    landing_at_sea: bool | None    # None if coastline check disabled
    in_coverage: bool | None       # None if monitor check not applicable
    violation_reason: str

    # Stochastic inputs (for CSV / replay)
    wind_profile_index: int
    impulse_factor: float
    azimuth_deg: float
    inclination_deg: float
    fin_cant_deg: float

    # Full TrajectoryProfile kept for replay plots
    trajectory: TrajectoryProfile | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# SimParams builder
# ---------------------------------------------------------------------------

def build_sim_params(
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    wind_profile_index: int,
    azimuth_deg: float,
    inclination_deg: float,
    impulse_factor: float,
    fin_cant_deg: float,
) -> SimParams:
    """Construct a SimParams from configs and per-sample stochastic draws.

    Dry vehicle properties are read from the Vehicle dataclass
    (computed once during vehicle loading).
    """
    geom = vehicle.geometry
    recovery = vehicle.recovery
    acc = sim_cfg.monte_carlo.acceptance

    # Wind profile for this sample (wrapping index)
    n_profiles = wind_ensemble.wind_east_ms.shape[0]
    idx = wind_profile_index % n_profiles

    # Recovery CdA values
    drogue_cda = 0.0
    main_cda = 0.0
    main_deploy_alt = -1.0  # sentinel: deploy at apogee
    has_drogue = recovery.drogue is not None
    has_main = recovery.main is not None

    if has_drogue:
        drogue_cda = recovery.drogue.cd * recovery.drogue.area
    if has_main:
        main_cda = recovery.main.cd * recovery.main.area
        if isinstance(recovery.main.threshold, float):
            main_deploy_alt = recovery.main.threshold

    return SimParams(
        # Motor / propellant + derived dry properties
        motor_times=propellant.times,
        motor_thrusts=propellant.thrusts,
        m_prop_0=propellant.m_prop_0,
        total_impulse=propellant.total_impulse,
        nozzle_area=propellant.nozzle_area,
        nozzle_position=propellant.nozzle_position,
        m_dry=vehicle.m_dry,
        cg_dry=vehicle.cg_dry,
        motor_cg_loaded=propellant.motor_cg_loaded,
        I_roll_dry=vehicle.I_roll_dry,
        I_lateral_dry=vehicle.I_lateral_dry,
        prop_r_outer=propellant.prop_r_outer,
        prop_r_inner_0=propellant.prop_r_inner_0,
        prop_length=propellant.prop_length,
        # Aero
        mach_g=aero_model.mach_grid,
        re_g=aero_model.re_grid,
        alpha_g=aero_model.alpha_grid,
        ca_tbl_off=aero_model.ca_table_off,
        ca_tbl_on=aero_model.ca_table_on,
        cn_tbl=aero_model.cn_table,
        cp_tbl=aero_model.cp_table,
        cn_comp=aero_model.cn_comp,
        cp_comp=aero_model.cp_comp,
        has_components=aero_model.has_components,
        cn_alpha_fins=aero_model.cn_alpha_fins,
        cn_alpha_comp=aero_model.cn_alpha_comp,
        comp_names=aero_model.comp_names,
        # Geometry
        diameter=geom.diameter,
        length=geom.length,
        A_ref=geom.reference_area,
        fin_cp_radius=geom.fin_cp_radius,
        fin_span=geom.fin_span,
        # Wind (single profile)
        wind_alt=wind_ensemble.altitude_m,
        wind_east=wind_ensemble.wind_east_ms[idx],
        wind_north=wind_ensemble.wind_north_ms[idx],
        # Rail
        rail_azimuth_rad=azimuth_deg * _DEG2RAD,
        rail_inclination_rad=inclination_deg * _DEG2RAD,
        rail_length=sim_cfg.launch.rail.length,
        # Stochastic
        fin_cant_rad=fin_cant_deg * _DEG2RAD,
        impulse_factor=impulse_factor,
        # Recovery
        drogue_cda=drogue_cda,
        main_cda=main_cda,
        drogue_cd=recovery.drogue.cd if has_drogue else 0.0,
        main_cd=recovery.main.cd if has_main else 0.0,
        main_deploy_alt=main_deploy_alt,
        has_drogue=has_drogue,
        has_main=has_main,
        # Acceptance thresholds
        sm_subsonic_min=acc.sm_subsonic_min,
        sm_supersonic_min=acc.sm_supersonic_min,
        # Atmosphere site corrections
        site_elevation=sim_cfg.site.elevation,
        t_offset=(
            compute_t_offset(sim_cfg.site.elevation, sim_cfg.site.temperature)
            if sim_cfg.site.temperature is not None
            else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Stochastic draw generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StochasticDraws:
    """Per-sample stochastic input values."""
    wind_profile_index: int
    azimuth_deg: float
    inclination_deg: float
    impulse_factor: float
    fin_cant_deg: float


def generate_sample_draws(
    master_seed: int,
    run_index: int,
    sample_index: int,
    azimuth_mean: float,
    inclination_mean: float,
    uncertainties_cfg,
) -> StochasticDraws:
    """Generate deterministic stochastic draws for one sample.

    Uses ``SeedSequence(master_seed, [run_index, sample_index])`` so that
    any sample can be reproduced without storing trajectory data.
    """
    ss = np.random.SeedSequence(master_seed, spawn_key=(run_index, sample_index))
    rng = np.random.default_rng(ss)

    azimuth = rng.normal(azimuth_mean, uncertainties_cfg.azimuth_sigma)
    inclination = rng.normal(inclination_mean, uncertainties_cfg.inclination_sigma)
    impulse_factor = rng.normal(1.0, uncertainties_cfg.impulse_factor_sigma)
    fin_cant = rng.normal(0.0, uncertainties_cfg.fin_cant_sigma)

    return StochasticDraws(
        wind_profile_index=sample_index,
        azimuth_deg=azimuth,
        inclination_deg=inclination,
        impulse_factor=impulse_factor,
        fin_cant_deg=fin_cant,
    )


# ---------------------------------------------------------------------------
# Single-sample runner
# ---------------------------------------------------------------------------

def run_sample(
    sample_index: int,
    run_index: int,
    scenario_name: str,
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    azimuth_mean: float,
    inclination_mean: float,
    poly_e: np.ndarray | None,
    poly_n: np.ndarray | None,
    buffered_ceiling: float,
    coastline_prepared,
    station_norths: np.ndarray | None,
    station_easts: np.ndarray | None,
    station_radii: np.ndarray | None,
    keep_trajectory: bool = False,
) -> SampleResult:
    """Run a single trajectory sample with full acceptance checking."""
    mc_cfg = sim_cfg.monte_carlo
    acc = mc_cfg.acceptance
    site = sim_cfg.site

    # --- Stochastic draws ---
    draws = generate_sample_draws(
        mc_cfg.seed, run_index, sample_index,
        azimuth_mean, inclination_mean,
        mc_cfg.uncertainties,
    )

    # --- Build SimParams ---
    params = build_sim_params(
        sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
        draws.wind_profile_index,
        draws.azimuth_deg, draws.inclination_deg,
        draws.impulse_factor, draws.fin_cant_deg,
    )

    # --- Run trajectory ---
    scenario_int = SCENARIO_MAP[scenario_name]
    traj_out = run_trajectory(
        params, scenario_int, poly_e, poly_n, buffered_ceiling,
        keep_profile=keep_trajectory,
    )
    if keep_trajectory:
        summary, profile = traj_out
    else:
        summary = traj_out
        profile = None

    # --- Convert positions to lat/lon ---
    apogee_lat, apogee_lon = ned_to_latlon(
        summary.apogee_position[0], summary.apogee_position[1],
        site.latitude, site.longitude,
    )
    landing_lat, landing_lon = ned_to_latlon(
        summary.landing_position[0], summary.landing_position[1],
        site.latitude, site.longitude,
    )

    # --- Extract NED positions ---
    apogee_north = float(summary.apogee_position[0])
    apogee_east = float(summary.apogee_position[1])
    landing_north = float(summary.landing_position[0])
    landing_east = float(summary.landing_position[1])

    # Coastline check (only for configured scenarios)
    landing_at_sea: bool | None = None
    sea_compliant = True
    if coastline_prepared is not None and scenario_name in acc.coastline_check_scenarios:
        landing_at_sea = check_coastline(
            landing_north, landing_east,
            coastline_prepared, site.coastline_mode,
        )
        sea_compliant = landing_at_sea

    # Monitor coverage check (only for configured scenarios)
    in_coverage: bool | None = None
    monitor_compliant = True
    if station_norths is not None and scenario_name in acc.monitor_check_scenarios:
        in_coverage = check_monitor_coverage(
            landing_north, landing_east,
            station_norths, station_easts, station_radii,
        )
        monitor_compliant = in_coverage

    # --- Assemble overall compliance ---
    # FlightSummary provides footprint/ceiling/stability; this layer adds
    # coastline and monitor checks that dynamics.py has no knowledge of.
    dynamics_ok = (
        summary.footprint_compliant
        and summary.ceiling_compliant
        and summary.stability_compliant
    )
    compliant = dynamics_ok and sea_compliant and monitor_compliant

    # Build violation reason string
    violation = ""
    if not summary.stability_compliant:
        violation = "Stability violation"
    elif not summary.footprint_compliant:
        violation = "Trajectory exited buffered danger area"
    elif not summary.ceiling_compliant:
        violation = "Apogee above buffered ceiling"
    elif not sea_compliant:
        violation = "Landing fails coastline check"
    elif not monitor_compliant:
        violation = "Landing outside monitored area"

    return SampleResult(
        sample_id=sample_index,
        scenario=scenario_name,
        run_index=run_index,
        apogee_m=summary.apogee,
        apogee_lat=apogee_lat,
        apogee_lon=apogee_lon,
        landing_lat=landing_lat,
        landing_lon=landing_lon,
        apogee_north=apogee_north,
        apogee_east=apogee_east,
        landing_north=landing_north,
        landing_east=landing_east,
        flight_time_s=summary.flight_time,
        peak_mach=summary.max_mach,
        max_aoa_deg=summary.max_aoa_deg,
        min_sm_subsonic=summary.min_sm_subsonic,
        min_sm_supersonic=summary.min_sm_supersonic,
        compliant=compliant,
        footprint_compliant=summary.footprint_compliant,
        ceiling_compliant=summary.ceiling_compliant,
        stability_compliant=summary.stability_compliant,
        landing_at_sea=landing_at_sea,
        in_coverage=in_coverage,
        violation_reason=violation,
        wind_profile_index=draws.wind_profile_index,
        impulse_factor=draws.impulse_factor,
        azimuth_deg=draws.azimuth_deg,
        inclination_deg=draws.inclination_deg,
        fin_cant_deg=draws.fin_cant_deg,
        trajectory=profile,
    )


# ---------------------------------------------------------------------------
# Scenario runner (worker target for multiprocessing)
# ---------------------------------------------------------------------------

def _prepare_geofence(
    sim_cfg: SimulationConfig,
) -> tuple[
    np.ndarray | None,       # poly_e
    np.ndarray | None,       # poly_n
    float,                   # buffered_ceiling
    object | None,           # coastline_prepared
    np.ndarray | None,       # station_norths
    np.ndarray | None,       # station_easts
    np.ndarray | None,       # station_radii
]:
    """Load and prepare all geofence data from config."""
    site = sim_cfg.site
    acc = sim_cfg.monte_carlo.acceptance

    # Danger area
    danger_poly = load_polygon_ned(site.danger_area, site.latitude, site.longitude)
    buffered_poly = buffer_danger_area(danger_poly, acc.buffer_distance)
    poly_e, poly_n = polygon_to_arrays(buffered_poly)
    buffered_ceiling = site.altitude_ceiling - acc.buffer_distance

    # Coastline
    coastline_prepared = None
    if site.coastline is not None:
        coast_poly = load_polygon_ned(
            site.coastline, site.latitude, site.longitude,
        )
        coastline_prepared = prepare_zone(coast_poly)

    # Monitor stations (including automatic launch-site station)
    station_norths = None
    station_easts = None
    station_radii = None
    if site.monitor_stations or site.launch_monitor_radius > 0:
        lat0_rad = math.radians(site.latitude)
        deg2m_north = R_EARTH * math.pi / 180.0
        deg2m_east = R_EARTH * math.cos(lat0_rad) * math.pi / 180.0

        norths: list[float] = []
        easts: list[float] = []
        radii: list[float] = []

        # Automatic launch-site monitor station
        if site.launch_monitor_radius > 0:
            norths.append(0.0)
            easts.append(0.0)
            radii.append(site.launch_monitor_radius)

        for st in site.monitor_stations:
            norths.append((st.latitude - site.latitude) * deg2m_north)
            easts.append((st.longitude - site.longitude) * deg2m_east)
            radii.append(st.radius)

        station_norths = np.array(norths, dtype=np.float64)
        station_easts = np.array(easts, dtype=np.float64)
        station_radii = np.array(radii, dtype=np.float64)

    return (
        poly_e, poly_n, buffered_ceiling,
        coastline_prepared,
        station_norths, station_easts, station_radii,
    )


def run_scenario(
    scenario_name: str,
    run_index: int,
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    azimuth_mean: float,
    inclination_mean: float,
    progress_queue: mp.Queue | None = None,
) -> list[SampleResult]:
    """Run all samples for a single descent scenario sequentially.

    Used by replay workflows.  The main ``run_monte_carlo`` path now
    dispatches individual samples via ``_run_sample_task`` instead.
    """
    n_samples = sim_cfg.monte_carlo.samples

    # Prepare geofence data (once per process)
    (poly_e, poly_n, buffered_ceiling,
     coastline_prepared,
     station_norths, station_easts, station_radii) = _prepare_geofence(sim_cfg)

    results: list[SampleResult] = []
    for i in range(n_samples):
        sr = run_sample(
            sample_index=i,
            run_index=run_index,
            scenario_name=scenario_name,
            sim_cfg=sim_cfg,
            vehicle=vehicle,
            propellant=propellant,
            aero_model=aero_model,
            wind_ensemble=wind_ensemble,
            azimuth_mean=azimuth_mean,
            inclination_mean=inclination_mean,
            poly_e=poly_e,
            poly_n=poly_n,
            buffered_ceiling=buffered_ceiling,
            coastline_prepared=coastline_prepared,
            station_norths=station_norths,
            station_easts=station_easts,
            station_radii=station_radii,
        )
        results.append(sr)
        if progress_queue is not None:
            progress_queue.put((scenario_name, i + 1))

    return results


# ---------------------------------------------------------------------------
# Scenario statistics
# ---------------------------------------------------------------------------

@dataclass
class ScenarioStats:
    """Aggregate statistics for one descent scenario."""
    scenario: str
    n_samples: int
    n_compliant: int
    n_non_compliant: int
    passed: bool               # compliant fraction >= threshold

    apogee_mean: float
    apogee_std: float
    apogee_min: float
    apogee_max: float

    landing_dist_mean: float   # metres from launch site
    landing_dist_std: float
    landing_dist_min: float
    landing_dist_max: float

    peak_mach_mean: float
    peak_mach_std: float
    max_aoa_mean: float
    max_aoa_std: float

    sm_subsonic_min: float     # worst case across all samples
    sm_supersonic_min: float


def compute_scenario_stats(
    results: list[SampleResult],
    compliance_threshold: float,
) -> ScenarioStats:
    """Compute aggregate statistics for one scenario."""
    n = len(results)
    n_compliant = sum(1 for r in results if r.compliant)

    apogees = np.array([r.apogee_m for r in results])
    distances = np.array([
        math.hypot(r.landing_north, r.landing_east) for r in results
    ])
    machs = np.array([r.peak_mach for r in results])
    aoas = np.array([r.max_aoa_deg for r in results])
    sm_sub = np.array([r.min_sm_subsonic for r in results])
    sm_sup = np.array([r.min_sm_supersonic for r in results])

    return ScenarioStats(
        scenario=results[0].scenario,
        n_samples=n,
        n_compliant=n_compliant,
        n_non_compliant=n - n_compliant,
        passed=n_compliant / n >= compliance_threshold if n > 0 else False,
        apogee_mean=float(np.nanmean(apogees)),
        apogee_std=float(np.nanstd(apogees)),
        apogee_min=float(np.nanmin(apogees)),
        apogee_max=float(np.nanmax(apogees)),
        landing_dist_mean=float(np.nanmean(distances)),
        landing_dist_std=float(np.nanstd(distances)),
        landing_dist_min=float(np.nanmin(distances)),
        landing_dist_max=float(np.nanmax(distances)),
        peak_mach_mean=float(np.nanmean(machs)),
        peak_mach_std=float(np.nanstd(machs)),
        max_aoa_mean=float(np.nanmean(aoas)),
        max_aoa_std=float(np.nanstd(aoas)),
        sm_subsonic_min=float(np.nanmin(sm_sub)),
        sm_supersonic_min=float(np.nanmin(sm_sup)),
    )


# ---------------------------------------------------------------------------
# Top-level MC result
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Aggregate result of a full Monte Carlo analysis."""
    all_results: list[SampleResult]
    scenario_stats: dict[str, ScenarioStats]
    all_passed: bool
    warnings: list[str]
    azimuth_mean: float
    inclination_mean: float
    # Deterministic baseline per scenario: (FlightSummary, time_s, altitude_m).
    baseline_curves: dict[str, tuple[FlightSummary, np.ndarray, np.ndarray]] = field(
        default_factory=dict,
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def load_all_models(
    sim_cfg: SimulationConfig,
) -> tuple[Vehicle, PropellantModel, AeroModel, WindEnsemble]:
    """Load vehicle config and build propellant, aero, and wind models."""
    vehicle, propellant = load_vehicle(sim_cfg.vehicle)
    aero_model = build_aero_model(
        vehicle.aero_tables,
        fins_override=vehicle.fins_aero_table,
    )
    wind_ensemble = load_wind_ensemble(
        sim_cfg.launch.wind_profiles,
        sim_cfg.monte_carlo.samples,
        surface_wind=sim_cfg.launch.surface_wind,
    )
    return vehicle, propellant, aero_model, wind_ensemble


def _run_sample_task(
    scenario_name: str,
    run_index: int,
    sample_index: int,
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    azimuth_mean: float,
    inclination_mean: float,
    geofence: tuple,
) -> SampleResult:
    """Top-level picklable wrapper for a single sample — pool worker target."""
    (poly_e, poly_n, buffered_ceiling,
     coastline_prepared,
     station_norths, station_easts, station_radii) = geofence
    return run_sample(
        sample_index=sample_index,
        run_index=run_index,
        scenario_name=scenario_name,
        sim_cfg=sim_cfg,
        vehicle=vehicle,
        propellant=propellant,
        aero_model=aero_model,
        wind_ensemble=wind_ensemble,
        azimuth_mean=azimuth_mean,
        inclination_mean=inclination_mean,
        poly_e=poly_e,
        poly_n=poly_n,
        buffered_ceiling=buffered_ceiling,
        coastline_prepared=coastline_prepared,
        station_norths=station_norths,
        station_easts=station_easts,
        station_radii=station_radii,
    )


def run_monte_carlo(
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    azimuth_mean: float,
    inclination_mean: float,
    progress_callback=None,
    scenario_done_callback=None,
) -> MonteCarloResult:
    """Run the full Monte Carlo analysis across all active scenarios.

    Individual samples from all scenarios are dispatched across a shared
    process pool so that cores freed by shorter scenarios automatically
    pick up remaining work from longer ones.

    Parameters
    ----------
    sim_cfg, vehicle, propellant, aero_model, wind_ensemble
        Pre-loaded configuration and model objects.
    azimuth_mean, inclination_mean
        Nominal rail angles in degrees (from config or optimisation).
    progress_callback
        Optional callable ``(scenario_name, completed, total)`` for
        progress reporting.
    scenario_done_callback
        Optional callable ``(scenario_name, stats)`` fired when each
        scenario completes and its :class:`ScenarioStats` are available.

    Returns
    -------
    MonteCarloResult
    """
    active = vehicle.recovery.active_scenarios
    warnings_list: list[str] = []
    acc = sim_cfg.monte_carlo.acceptance

    # Warn about configured checks on inactive scenarios
    for s in acc.coastline_check_scenarios:
        if s not in active:
            warnings_list.append(
                f"coastline_check_scenarios includes '{s}' which is not active "
                f"for this vehicle configuration"
            )
    for s in acc.monitor_check_scenarios:
        if s not in active:
            warnings_list.append(
                f"monitor_check_scenarios includes '{s}' which is not active "
                f"for this vehicle configuration"
            )

    n_samples = sim_cfg.monte_carlo.samples

    # Prepare geofence once (same for all scenarios)
    geofence = _prepare_geofence(sim_cfg)

    # --- Dispatch all samples across a shared pool ---
    # Each (scenario, sample_index) pair is an independent task.  Cores
    # freed by shorter scenarios automatically pick up remaining work.
    all_results: list[SampleResult] = []
    scenario_stats: dict[str, ScenarioStats] = {}

    ctx = mp.get_context("spawn")
    n_workers = min(len(active) * n_samples, mp.cpu_count())

    with ctx.Pool(processes=n_workers) as pool:
        # Submit all tasks
        futures: list[tuple[str, mp.pool.AsyncResult]] = []
        for run_idx, scenario_name in enumerate(active):
            for sample_idx in range(n_samples):
                ar = pool.apply_async(
                    _run_sample_task,
                    args=(
                        scenario_name, run_idx, sample_idx,
                        sim_cfg, vehicle, propellant, aero_model,
                        wind_ensemble, azimuth_mean, inclination_mean,
                        geofence,
                    ),
                )
                futures.append((scenario_name, ar))

        # Collect results, reporting progress per scenario
        completed_counts: dict[str, int] = {s: 0 for s in active}
        scenario_results: dict[str, list[SampleResult]] = {s: [] for s in active}
        finished_scenarios: set[str] = set()

        for scenario_name, ar in futures:
            sr = ar.get()
            all_results.append(sr)
            scenario_results[scenario_name].append(sr)
            completed_counts[scenario_name] += 1

            if progress_callback:
                progress_callback(
                    scenario_name,
                    completed_counts[scenario_name],
                    n_samples,
                )

            # Fire scenario-done callback once all samples for it are in
            if (scenario_name not in finished_scenarios
                    and completed_counts[scenario_name] == n_samples):
                finished_scenarios.add(scenario_name)
                stats = compute_scenario_stats(
                    scenario_results[scenario_name],
                    acc.compliance_threshold,
                )
                scenario_stats[scenario_name] = stats
                if scenario_done_callback:
                    scenario_done_callback(scenario_name, stats)

    all_passed = all(s.passed for s in scenario_stats.values())

    # --- Baseline altitude-time curves ---
    # One deterministic trajectory per scenario: all uncertainties zero,
    # mean wind profile.  These are cheap (one run each) and give the
    # altitude plot its reference line without re-running later.
    baseline_curves: dict[str, tuple[FlightSummary, np.ndarray, np.ndarray]] = {}
    for scenario_name in active:
        baseline_params = build_sim_params(
            sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
            wind_profile_index=0,
            azimuth_deg=azimuth_mean,
            inclination_deg=inclination_mean,
            impulse_factor=1.0,
            fin_cant_deg=0.0,
        )
        summary, profile = run_trajectory(
            baseline_params, SCENARIO_MAP[scenario_name],
            keep_profile=True,
        )
        baseline_curves[scenario_name] = (summary, profile.time, profile.altitude)
        check_stability_compliance(
            summary, sim_cfg.monte_carlo.acceptance,
            f"Baseline '{scenario_name}'",
        )

    return MonteCarloResult(
        all_results=all_results,
        scenario_stats=scenario_stats,
        all_passed=all_passed,
        warnings=warnings_list,
        azimuth_mean=azimuth_mean,
        inclination_mean=inclination_mean,
        baseline_curves=baseline_curves,
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_sample(
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    master_seed: int,
    scenario_name: str,
    sample_index: int,
    azimuth_mean: float,
    inclination_mean: float,
) -> SampleResult:
    """Replay a single sample by reconstructing its RNG state.

    Returns a SampleResult with the full TrajectoryProfile attached
    (``keep_trajectory=True``).
    """
    active = vehicle.recovery.active_scenarios
    if scenario_name not in active:
        raise ValueError(
            f"Scenario '{scenario_name}' is not active. "
            f"Active scenarios: {active}"
        )
    run_index = active.index(scenario_name)

    # Prepare geofence
    (poly_e, poly_n, buffered_ceiling,
     coastline_prepared,
     station_norths, station_easts, station_radii) = _prepare_geofence(sim_cfg)

    return run_sample(
        sample_index=sample_index,
        run_index=run_index,
        scenario_name=scenario_name,
        sim_cfg=sim_cfg,
        vehicle=vehicle,
        propellant=propellant,
        aero_model=aero_model,
        wind_ensemble=wind_ensemble,
        azimuth_mean=azimuth_mean,
        inclination_mean=inclination_mean,
        poly_e=poly_e,
        poly_n=poly_n,
        buffered_ceiling=buffered_ceiling,
        coastline_prepared=coastline_prepared,
        station_norths=station_norths,
        station_easts=station_easts,
        station_radii=station_radii,
        keep_trajectory=True,
    )


# Maps --reason keyword → samples.csv compliance column name.
# Used by replay_non_compliant and exposed for CLI validation.
REASON_COLS: dict[str, str] = {
    "footprint":  "danger_area_footprint_compliant",
    "ceiling":    "danger_area_ceiling_compliant",
    "stability":  "stability_compliant",
    "coastline":  "coastline_compliant",
    "monitor":    "monitor_compliant",
}


def replay_non_compliant(
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    master_seed: int,
    azimuth_mean: float,
    inclination_mean: float,
    samples_csv_path: Path,
    reasons: tuple[str, ...] = (),
    scenario_filter: str | None = None,
) -> list[SampleResult]:
    """Replay all non-compliant samples from a previous run.

    Reads the samples CSV to find non-compliant (sample_id, scenario) pairs,
    then replays each one with full trajectory data.

    Parameters
    ----------
    reasons
        Optional filter: only replay samples whose non-compliance matches at
        least one of the given reason keywords.  Valid keywords are the keys
        of :data:`REASON_COLS`.  When empty (default) all non-compliant
        samples are replayed.
    scenario_filter
        Optional scenario name to restrict replay to a single scenario.
    """
    import csv

    def _is_false(val: str) -> bool:
        return val.strip().lower() in ("false", "0", "no")

    pairs: list[tuple[str, int]] = []
    with open(samples_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["scenario"].strip()
            if scenario_filter is not None and scenario != scenario_filter:
                continue

            # A sample is non-compliant if any _compliant column is False
            compliance_cols = [c for c in row if c.endswith("_compliant")]
            is_non_compliant = any(_is_false(row[c]) for c in compliance_cols)
            if not is_non_compliant:
                continue

            # Optional reason filter: at least one requested column must be False
            if reasons:
                matched = any(
                    _is_false(row[REASON_COLS[r]])
                    for r in reasons
                    if REASON_COLS[r] in row
                )
                if not matched:
                    continue

            active = vehicle.recovery.active_scenarios
            if scenario in active:
                pairs.append((scenario, int(row["sample_id"])))

    results: list[SampleResult] = []
    for scen, sample_idx in pairs:
        sr = replay_sample(
            sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
            master_seed, scen, sample_idx,
            azimuth_mean, inclination_mean,
        )
        results.append(sr)

    return results


def replay_compliant(
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    master_seed: int,
    azimuth_mean: float,
    inclination_mean: float,
    samples_csv_path: Path,
    scenario_filter: str | None = None,
) -> list[SampleResult]:
    """Replay all compliant samples from a previous run.

    Reads the samples CSV to find fully-compliant (sample_id, scenario) pairs,
    then replays each one with full trajectory data.

    Parameters
    ----------
    scenario_filter
        Optional scenario name to restrict replay to a single scenario.
    """
    import csv

    pairs: list[tuple[str, int]] = []
    with open(samples_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["scenario"].strip()
            if scenario_filter is not None and scenario != scenario_filter:
                continue

            compliance_cols = [
                c for c in row if c.endswith("_compliant")
            ]
            is_compliant = all(
                row[c].strip().lower() not in ("false", "0", "no")
                for c in compliance_cols
            )
            if is_compliant:
                active = vehicle.recovery.active_scenarios
                if scenario in active:
                    pairs.append((scenario, int(row["sample_id"])))

    results: list[SampleResult] = []
    for scen, sample_idx in pairs:
        sr = replay_sample(
            sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
            master_seed, scen, sample_idx,
            azimuth_mean, inclination_mean,
        )
        results.append(sr)

    return results
