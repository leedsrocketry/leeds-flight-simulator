"""Automatic launch rail azimuth and inclination optimisation.

Selects optimal integer launch azimuth and inclination, maximising the
probability that all active descent scenarios remain within the buffered
danger area.

Steps
-----
1. Inclination selection   — deterministic 6DoF sweeps, zero wind
2. Azimuth narrowing       — analytical wind-drift filter
3. Azimuth optimisation    — 1-D Bayesian optimisation, GP + UCB
4. Candidate validation    — full-uncertainty MC

Public API
----------
run_optimisation(sim_cfg, vehicle, propellant, aero_model,
                 wind_ensemble, progress_callback) → OptimisationResult
"""

from __future__ import annotations

import math
import multiprocessing as mp
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from shapely.geometry import Point, Polygon

from atmosphere import isa
from config import SimulationConfig, Vehicle
from motor import PropellantModel
from aerodynamics import AeroModel
from wind import WindEnsemble, interpolate_wind
from dynamics import (
    SimParams,
    run_trajectory,
    integrate_descent,
    SCENARIO_MAP,
    SCENARIO_BALLISTIC,
    SCENARIO_NOMINAL,
    SCENARIO_PREMATURE_MAIN,
    SCENARIO_DROGUE_ONLY,
)
from montecarlo import build_sim_params
from geography import (
    load_polygon_ned,
    buffer_danger_area,
    polygon_to_arrays,
    _point_in_polygon,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_G0: float = 9.80665
_DEG2RAD: float = math.pi / 180.0
_RAD2DEG: float = 180.0 / math.pi


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


def _run_6dof_apogee(
    inclination_deg: float,
    sim_cfg: "SimulationConfig",
    vehicle: "Vehicle",
    propellant: "PropellantModel",
    aero_model: "AeroModel",
) -> tuple[float, float, float, float]:
    """Run a deterministic 6DoF trajectory and return apogee (N, E, D, t).

    Uses azimuth=0, zero wind, impulse_factor=1, no fin cant.
    """
    zero_wind = _zero_wind_ensemble()
    params = build_sim_params(
        sim_cfg, vehicle, propellant, aero_model, zero_wind,
        wind_profile_index=0,
        azimuth_deg=0.0,
        inclination_deg=inclination_deg,
        impulse_factor=1.0,
        fin_cant_deg=0.0,
    )
    traj = run_trajectory(params, SCENARIO_BALLISTIC, None, None, float("inf"))
    ap = traj.apogee_position
    return float(ap[0]), float(ap[1]), float(ap[2]), traj.apogee_time


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptimisationResult:
    """Output of the full optimisation routine."""
    selected_azimuth: int          # degrees
    selected_inclination: int      # degrees

    # Inclination selection diagnostics
    inclination_apogees: dict[int, tuple[float, float, float]]       # inc → (N, E, D)
    inclination_ballistic_landings: dict[int, tuple[float, float]]   # inc → (N, E)
    inclination_selected: int

    # Azimuth narrowing diagnostics
    narrowing_feasible: list[int]     # surviving azimuth candidates
    narrowing_total_candidates: int

    # Azimuth optimisation diagnostics
    azimuth_observations: list[tuple[int, float]]  # (az, p_success) pairs
    azimuth_top_candidates: list[int]

    # Candidate validation diagnostics
    validation_compliance: dict[int, float]    # az → compliance fraction
    validation_margins: dict[int, float]       # az → margin in metres


# ---------------------------------------------------------------------------
# Worst-drift scenario helpers
# ---------------------------------------------------------------------------

def _worst_drift_scenario(vehicle: Vehicle) -> str:
    """Return the active scenario with the most descent drift.

    Premature-main drifts most (both chutes from apogee).  If not active
    (main deploys at apogee), nominal drifts most (drogue → main).  Then
    drogue-only.  Ballistic drifts least.
    """
    active = vehicle.recovery.active_scenarios
    if "premature_main" in active:
        return "premature_main"
    if "nominal" in active and vehicle.recovery.main is not None:
        return "nominal"
    if "drogue_only" in active:
        return "drogue_only"
    return "ballistic"


def _worst_drift_cda(vehicle: Vehicle) -> float:
    """Return the effective CdA for the worst-drift scenario.

    For premature-main: drogue + main (both deploy at apogee).
    For nominal with altitude-triggered main: drogue CdA above main deploy,
    main CdA below — we use main CdA as the conservative (slower descent,
    more drift) estimate for the analytical filter.
    """
    recovery = vehicle.recovery
    scenario = _worst_drift_scenario(vehicle)
    drogue_cda = (recovery.drogue.cd * recovery.drogue.area
                  if recovery.drogue is not None else 0.0)
    main_cda = (recovery.main.cd * recovery.main.area
                if recovery.main is not None else 0.0)

    if scenario == "premature_main":
        return drogue_cda + main_cda
    if scenario == "nominal":
        # Conservative: use main CdA (slower descent = more drift)
        return main_cda if main_cda > 0.0 else drogue_cda
    if scenario == "drogue_only":
        return drogue_cda
    # ballistic — no parachute; use a small CdA from body drag
    # (not meaningful for drift, but avoids division by zero)
    return 0.0


def _worst_drift_scenario_int(vehicle: Vehicle) -> int:
    """Return the integer scenario code for the worst-drift scenario."""
    return SCENARIO_MAP[_worst_drift_scenario(vehicle)]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotate_apogee(
    apogee_N: float, apogee_E: float,
    azimuth_rad: float, base_azimuth_rad: float = 0.0,
) -> tuple[float, float]:
    """Rotate an apogee NE position from *base_azimuth* to *azimuth*.

    The deterministic ascent is run at ``base_azimuth`` (typically 0°).
    For a different azimuth, the horizontal displacement rotates by the
    difference angle.
    """
    delta = azimuth_rad - base_azimuth_rad
    cos_d = math.cos(delta)
    sin_d = math.sin(delta)
    return (
        apogee_N * cos_d - apogee_E * sin_d,
        apogee_N * sin_d + apogee_E * cos_d,
    )


def _signed_distance_to_boundary(
    north: float, east: float, polygon: Polygon,
) -> float:
    """Signed distance from a point to the polygon boundary.

    Positive → inside, negative → outside.
    """
    pt = Point(east, north)  # Shapely: (x, y) = (east, north)
    dist = polygon.exterior.distance(pt)
    if polygon.contains(pt):
        return dist
    return -dist


# ---------------------------------------------------------------------------
# Wind drift calculation
# ---------------------------------------------------------------------------

def _compute_wind_drift(
    apogee_alt: float,
    wind_alt: np.ndarray,
    wind_east: np.ndarray,
    wind_north: np.ndarray,
    cda: float,
    m_dry: float,
    dz: float = 50.0,
) -> tuple[float, float]:
    """Compute analytical wind drift from apogee to ground.

    Integrates ``v_wind(z) / v_descent(z) * dz`` from ground up to apogee,
    where ``v_descent(z) = sqrt(2 * m * g / (rho(z) * CdA))``.

    Parameters
    ----------
    apogee_alt : float
        Apogee altitude in metres AGL.
    wind_alt, wind_east, wind_north : np.ndarray
        Wind profile arrays (mean or single profile).
    cda : float
        Effective CdA for the worst-drift scenario.
    m_dry : float
        Post-burnout vehicle mass in kg.
    dz : float
        Altitude step in metres.

    Returns
    -------
    (drift_N, drift_E) : tuple[float, float]
        Accumulated wind drift in NED metres.
    """
    if cda <= 0.0 or apogee_alt <= 0.0:
        return 0.0, 0.0

    drift_n = 0.0
    drift_e = 0.0
    n_steps = max(1, int(apogee_alt / dz))
    actual_dz = apogee_alt / n_steps

    for i in range(n_steps):
        z = apogee_alt - (i + 0.5) * actual_dz  # midpoint altitude
        if z < 0.0:
            z = 0.0

        _, _, rho, _, _ = isa(z)
        v_descent = math.sqrt(2.0 * m_dry * _G0 / (rho * cda))

        v_wn, v_we = interpolate_wind(wind_alt, wind_east, wind_north, z)

        dt = actual_dz / v_descent
        drift_n += v_wn * dt
        drift_e += v_we * dt

    return drift_n, drift_e


# ---------------------------------------------------------------------------
# Inclination selection
# ---------------------------------------------------------------------------

def select_inclination(
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    poly_e: np.ndarray,
    poly_n: np.ndarray,
) -> tuple[int, dict[int, tuple[float, float, float]], dict[int, tuple[float, float]], dict[int, float]]:
    """Select the steepest safe inclination.

    Returns
    -------
    (selected_inclination, apogee_positions, ballistic_landings, apogee_times)
    """
    rail_cfg = sim_cfg.launch.rail
    inc_range = rail_cfg.inclination_range
    assert inc_range is not None
    candidates = list(range(int(inc_range[0]), int(inc_range[1]) + 1))

    exclusion_r = sim_cfg.site.ballistic_exclusion_radius

    apogee_positions: dict[int, tuple[float, float, float]] = {}
    ballistic_landings: dict[int, tuple[float, float]] = {}
    apogee_times: dict[int, float] = {}
    valid: list[int] = []

    for idx, inc in enumerate(candidates):
        # 6DoF ascent, no wind, no uncertainty
        apN, apE, apD, t_ap = _run_6dof_apogee(
            float(inc), sim_cfg, vehicle, propellant, aero_model,
        )
        apogee_positions[inc] = (apN, apE, apD)
        apogee_times[inc] = t_ap

        # Ballistic descent from apogee with no wind — landing point is
        # approximately below the apogee.
        ballistic_landings[inc] = (apN, apE)

        # Check exclusion radius and containment
        dist = math.hypot(apN, apE)
        inside = _point_in_polygon(apE, apN, poly_e, poly_n)

        if dist >= exclusion_r and inside:
            valid.append(inc)

    if not valid:
        # No inclination satisfies both constraints.  Fall back to the
        # steepest candidate (least horizontal drift) so downstream phases
        # can still evaluate and report honest compliance.
        warnings.warn(
            f"No inclination in range [{inc_range[0]}, {inc_range[1]}] satisfies "
            f"both the ballistic exclusion radius ({exclusion_r} m) and the "
            f"buffered danger area constraint. Continuing with the steepest "
            f"candidate ({max(candidates)}°)."
        )
        valid = candidates

    selected = max(valid)
    return selected, apogee_positions, ballistic_landings, apogee_times


# ---------------------------------------------------------------------------
# Azimuth bound narrowing
# ---------------------------------------------------------------------------

def narrow_azimuth_bounds(
    selected_inclination: int,
    apogee_positions: dict[int, tuple[float, float, float]],
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    wind_ensemble: WindEnsemble,
    buffered_polygon: Polygon,
) -> list[int]:
    """Discard azimuths whose landing centroid falls outside the buffered
    danger area.

    Returns a list of feasible integer azimuth candidates.
    """
    az_range = sim_cfg.launch.rail.azimuth_range
    assert az_range is not None
    az_min, az_max = int(az_range[0]), int(az_range[1])

    # Handle wrap-around (e.g. [350, 10])
    if az_min <= az_max:
        candidates = list(range(az_min, az_max + 1))
    else:
        candidates = list(range(az_min, 360)) + list(range(0, az_max + 1))

    apN0, apE0, apD0 = apogee_positions[selected_inclination]
    apogee_alt = -apD0  # D is negative altitude

    cda = _worst_drift_cda(vehicle)
    m_dry = vehicle.m_dry

    feasible: list[int] = []
    # Track signed distance for every candidate so we can fall back to
    # the closest ones when none are strictly inside the buffered area.
    candidate_distances: list[tuple[int, float]] = []

    for idx, az in enumerate(candidates):
        # Rotate apogee from az=0 to this candidate
        rot_N, rot_E = _rotate_apogee(apN0, apE0, az * _DEG2RAD, 0.0)

        # Compute wind drift using mean profile
        drift_N, drift_E = _compute_wind_drift(
            apogee_alt,
            wind_ensemble.altitude_m,
            wind_ensemble.mean_east_ms,
            wind_ensemble.mean_north_ms,
            cda, m_dry,
        )

        centroid_N = rot_N + drift_N
        centroid_E = rot_E + drift_E

        dist = _signed_distance_to_boundary(centroid_N, centroid_E, buffered_polygon)
        candidate_distances.append((az, dist))

        if dist >= 0.0:
            feasible.append(az)

    if not feasible:
        # No azimuth lands inside the buffered area.  Rather than aborting,
        # return the candidates closest to the boundary so the downstream
        # phases can evaluate them properly and report honest compliance.
        candidate_distances.sort(key=lambda x: x[1], reverse=True)
        n_fallback = min(len(candidate_distances), max(3, len(candidates) // 10))
        feasible = [az for az, _ in candidate_distances[:n_fallback]]
        best_dist = candidate_distances[0][1]
        warnings.warn(
            f"No azimuth in range [{az_min}, {az_max}] produces a landing "
            f"centroid inside the buffered danger area (closest miss: "
            f"{-best_dist:.0f} m outside). Continuing with the {n_fallback} "
            f"least-infeasible candidates."
        )

    return feasible


# ---------------------------------------------------------------------------
# Azimuth evaluation helpers
# ---------------------------------------------------------------------------

def _run_descent_single(
    apogee_N: float, apogee_E: float, apogee_D: float, t_apogee: float,
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    m_dry: float,
    drogue_cda: float, main_cda: float, main_deploy_alt: float,
    scenario: int,
) -> tuple[float, float]:
    """Run a single parachute descent from apogee and return (landing_N, landing_E)."""
    state0 = np.array([
        apogee_N, apogee_E, apogee_D,
    ], dtype=np.float64)

    t_desc, y_desc, _, n_desc = integrate_descent(
        t_apogee, state0,
        wind_alt, wind_east, wind_north,
        m_dry,
        drogue_cda, main_cda, main_deploy_alt,
        scenario,
        1.0e-6, 1.0e-6,
    )
    return float(y_desc[n_desc - 1, 0]), float(y_desc[n_desc - 1, 1])


def _descent_worker(args: tuple) -> tuple[float, float]:
    """Multiprocessing worker for a single descent sim."""
    return _run_descent_single(*args)


def _run_descent_batch(
    apogee_N: float, apogee_E: float, apogee_D: float, t_apogee: float,
    wind_ensemble: WindEnsemble,
    vehicle: Vehicle,
    scenario_int: int,
    n_sims: int,
    pool: mp.pool.Pool | None = None,
) -> np.ndarray:
    """Run *n_sims* parachute descent sims from a known apogee, varying wind.

    Returns (n_sims, 2) array of [landing_N, landing_E].
    """
    m_dry = vehicle.m_dry
    recovery = vehicle.recovery
    drogue_cda = (recovery.drogue.cd * recovery.drogue.area
                  if recovery.drogue is not None else 0.0)
    main_cda = (recovery.main.cd * recovery.main.area
                if recovery.main is not None else 0.0)
    main_deploy_alt = -1.0  # sentinel: deploy at apogee
    if recovery.main is not None and isinstance(recovery.main.threshold, float):
        main_deploy_alt = recovery.main.threshold

    n_profiles = wind_ensemble.wind_east_ms.shape[0]

    args_list = []
    for i in range(n_sims):
        idx = i % n_profiles
        args_list.append((
            apogee_N, apogee_E, apogee_D, t_apogee,
            wind_ensemble.altitude_m,
            wind_ensemble.wind_east_ms[idx],
            wind_ensemble.wind_north_ms[idx],
            m_dry,
            drogue_cda, main_cda, main_deploy_alt,
            scenario_int,
        ))

    if pool is not None:
        results = pool.map(_descent_worker, args_list)
    else:
        results = [_run_descent_single(*a) for a in args_list]

    return np.array(results, dtype=np.float64)


def _evaluate_azimuth(
    azimuth_deg: int,
    apogee_N0: float, apogee_E0: float, apogee_D: float, t_apogee: float,
    wind_ensemble: WindEnsemble,
    vehicle: Vehicle,
    poly_e: np.ndarray,
    poly_n: np.ndarray,
    scenario_int: int,
    n_sims: int,
    pool: mp.pool.Pool | None = None,
) -> float:
    """Evaluate a single azimuth: run descent batch, return p_success."""
    rot_N, rot_E = _rotate_apogee(apogee_N0, apogee_E0, azimuth_deg * _DEG2RAD, 0.0)

    landings = _run_descent_batch(
        rot_N, rot_E, apogee_D, t_apogee,
        wind_ensemble, vehicle,
        scenario_int, n_sims, pool,
    )

    n_inside = 0
    for j in range(landings.shape[0]):
        if _point_in_polygon(landings[j, 1], landings[j, 0], poly_e, poly_n):
            n_inside += 1

    return n_inside / n_sims


# ---------------------------------------------------------------------------
# Azimuth optimisation (GP + UCB)
# ---------------------------------------------------------------------------

def optimise_azimuth(
    feasible_azimuths: list[int],
    selected_inclination: int,
    apogee_positions: dict[int, tuple[float, float, float]],
    t_apogee: float,
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    poly_e: np.ndarray,
    poly_n: np.ndarray,
) -> tuple[list[int], list[tuple[int, float]]]:
    """Bayesian optimisation over feasible azimuths.

    Returns (top_candidates, observations) where observations is a list
    of (azimuth, p_success) pairs from all evaluations.
    """
    scenario_int = _worst_drift_scenario_int(vehicle)
    apN0, apE0, apD0 = apogee_positions[selected_inclination]
    SIMS_PER_ITER = 150
    MAX_ITER = 20

    # If <= 3 candidates, evaluate all directly — no BO needed
    if len(feasible_azimuths) <= 3:
        observations: list[tuple[int, float]] = []
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(mp.cpu_count(), 4)) as pool:
            for i, az in enumerate(feasible_azimuths):
                p = _evaluate_azimuth(
                    az, apN0, apE0, apD0, t_apogee,
                    wind_ensemble, vehicle,
                    poly_e, poly_n, scenario_int, SIMS_PER_ITER, pool,
                )
                observations.append((az, p))

        observations.sort(key=lambda x: x[1], reverse=True)
        return [az for az, _ in observations], observations

    # --- Bayesian optimisation ---
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from scipy.stats import norm
    # Initial evaluation points: evenly spaced across the feasible range
    n_init = min(5, len(feasible_azimuths))
    init_indices = np.unique(
        np.linspace(0, len(feasible_azimuths) - 1, n_init, dtype=int)
    )

    X_obs: list[float] = []
    Y_obs: list[float] = []
    observations = []
    evaluated: set[int] = set()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(mp.cpu_count(), 4)) as pool:
        # Evaluate initial points
        for idx in init_indices:
            az = feasible_azimuths[idx]
            if az in evaluated:
                continue
            p = _evaluate_azimuth(
                az, apN0, apE0, apD0, t_apogee,
                wind_ensemble, vehicle,
                poly_e, poly_n, scenario_int, SIMS_PER_ITER, pool,
            )
            X_obs.append(float(az))
            Y_obs.append(p)
            observations.append((az, p))
            evaluated.add(az)

        # BO loop
        for iteration in range(MAX_ITER):
            X_arr = np.array(X_obs).reshape(-1, 1)
            Y_arr = np.array(Y_obs)

            # Noise variance: p(1-p)/n
            alpha_noise = np.array([
                max(p * (1.0 - p) / SIMS_PER_ITER, 1e-8) for p in Y_obs
            ])

            kernel = RBF(length_scale=30.0, length_scale_bounds=(5.0, 180.0))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha_noise,
                n_restarts_optimizer=3,
                normalize_y=True,
            )
            gp.fit(X_arr, Y_arr)

            # Evaluate UCB at all unevaluated feasible azimuths
            unevaluated = [az for az in feasible_azimuths if az not in evaluated]
            if not unevaluated:
                break

            X_cand = np.array(unevaluated).reshape(-1, 1)
            mu, sigma = gp.predict(X_cand, return_std=True)

            # UCB acquisition
            kappa = 2.0
            ucb = mu + kappa * sigma
            best_idx = int(np.argmax(ucb))
            next_az = unevaluated[best_idx]

            # Check expected improvement for termination
            f_best = max(Y_obs)
            z_vals = np.where(sigma > 1e-10, (mu - f_best) / sigma, 0.0)
            ei = np.where(
                sigma > 1e-10,
                (mu - f_best) * norm.cdf(z_vals) + sigma * norm.pdf(z_vals),
                0.0,
            )
            if np.max(ei) < 0.05:
                break

            # Evaluate next candidate
            p = _evaluate_azimuth(
                next_az, apN0, apE0, apD0, t_apogee,
                wind_ensemble, vehicle,
                poly_e, poly_n, scenario_int, SIMS_PER_ITER, pool,
            )
            X_obs.append(float(next_az))
            Y_obs.append(p)
            observations.append((next_az, p))
            evaluated.add(next_az)

    # Select top 3 by GP posterior mean (or raw observations if GP not fitted)
    if len(X_obs) >= 2:
        X_all = np.array([float(az) for az in feasible_azimuths]).reshape(-1, 1)
        X_fit = np.array(X_obs).reshape(-1, 1)
        Y_fit = np.array(Y_obs)
        alpha_noise = np.array([
            max(p * (1.0 - p) / SIMS_PER_ITER, 1e-8) for p in Y_obs
        ])
        kernel = RBF(length_scale=30.0, length_scale_bounds=(5.0, 180.0))
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha_noise,
            n_restarts_optimizer=3, normalize_y=True,
        )
        gp.fit(X_fit, Y_fit)
        mu_all = gp.predict(X_all)
        ranked = np.argsort(mu_all)[::-1]
        top = [feasible_azimuths[int(i)] for i in ranked[:3]]
    else:
        observations.sort(key=lambda x: x[1], reverse=True)
        top = [az for az, _ in observations[:3]]

    return top, observations


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------

def _validation_worker(args: tuple) -> tuple[int, list[tuple[float, float]]]:
    """Worker that runs N full-uncertainty sims for one azimuth candidate.

    Returns (azimuth, list_of_(landing_N, landing_E)).
    """
    (azimuth_deg, selected_inclination, sim_cfg, vehicle,
     propellant, aero_model, wind_ensemble, n_sims, scenario_name) = args

    from montecarlo import run_sample, _prepare_geofence

    (poly_e, poly_n, buffered_ceiling,
     coastline_prepared,
     station_norths, station_easts, station_radii) = _prepare_geofence(sim_cfg)

    landings: list[tuple[float, float]] = []
    for i in range(n_sims):
        sr = run_sample(
            sample_index=i,
            run_index=0,
            scenario_name=scenario_name,
            sim_cfg=sim_cfg,
            vehicle=vehicle,
            propellant=propellant,
            aero_model=aero_model,
            wind_ensemble=wind_ensemble,
            azimuth_mean=float(azimuth_deg),
            inclination_mean=float(selected_inclination),
            poly_e=poly_e,
            poly_n=poly_n,
            buffered_ceiling=buffered_ceiling,
            coastline_prepared=coastline_prepared,
            station_norths=station_norths,
            station_easts=station_easts,
            station_radii=station_radii,
        )
        landings.append((sr.landing_north, sr.landing_east))

    return azimuth_deg, landings


def validate_candidates(
    top_candidates: list[int],
    selected_inclination: int,
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    buffered_polygon: Polygon,
    poly_e: np.ndarray,
    poly_n: np.ndarray,
) -> tuple[int, dict[int, float], dict[int, float]]:
    """Validate top candidates with full-uncertainty MC.

    Returns (optimal_azimuth, compliance_fractions, margins).
    """
    SIMS_PER_CANDIDATE = 500
    scenario_name = _worst_drift_scenario(vehicle)
    compliance_threshold = sim_cfg.monte_carlo.acceptance.compliance_threshold

    args_list = [
        (az, selected_inclination, sim_cfg, vehicle,
         propellant, aero_model, wind_ensemble,
         SIMS_PER_CANDIDATE, scenario_name)
        for az in top_candidates
    ]

    # Run candidates in parallel (one process per candidate)
    ctx = mp.get_context("spawn")
    n_procs = min(len(top_candidates), mp.cpu_count())
    with ctx.Pool(processes=n_procs) as pool:
        results = pool.map(_validation_worker, args_list)

    compliance: dict[int, float] = {}
    margins: dict[int, float] = {}

    for az, landings in results:
        landing_arr = np.array(landings, dtype=np.float64)
        n_total = landing_arr.shape[0]

        # Compute signed distances
        distances = np.array([
            _signed_distance_to_boundary(
                landing_arr[j, 0], landing_arr[j, 1], buffered_polygon,
            )
            for j in range(n_total)
        ])

        n_inside = int(np.sum(distances >= 0.0))
        compliance[az] = n_inside / n_total

        # Margin: signed distance at the compliance_threshold percentile
        # (sorted ascending — the threshold-percentile worst case)
        sorted_dists = np.sort(distances)
        pct_idx = max(0, int(math.ceil((1.0 - compliance_threshold) * n_total)) - 1)
        margins[az] = float(sorted_dists[pct_idx])

    # Select candidate with greatest margin
    optimal = max(top_candidates, key=lambda az: margins[az])
    return optimal, compliance, margins


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_optimisation(
    sim_cfg: SimulationConfig,
    vehicle: Vehicle,
    propellant: PropellantModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    progress_callback: Callable[[int], None] | None = None,
) -> OptimisationResult:
    """Run the full optimisation routine.

    *progress_callback(step)* is called after each of the four stages
    completes (step 1–4).  The caller should set up a bar with total=4.
    """
    rail = sim_cfg.launch.rail
    inc_is_auto = rail.inclination == "auto"
    az_is_auto = rail.azimuth == "auto"

    def _notify(step: int) -> None:
        if progress_callback is not None:
            progress_callback(step)

    # Load and buffer the danger area
    site = sim_cfg.site
    acc = sim_cfg.monte_carlo.acceptance
    danger_poly = load_polygon_ned(site.danger_area, site.latitude, site.longitude)
    buffered_poly = buffer_danger_area(danger_poly, acc.buffer_distance)
    poly_e, poly_n = polygon_to_arrays(buffered_poly)

    # --- Inclination selection ---
    if inc_is_auto:
        selected_inc, apogee_positions, ballistic_landings, apogee_times = (
            select_inclination(
                sim_cfg, vehicle, propellant, aero_model,
                poly_e, poly_n,
            )
        )
    else:
        selected_inc = int(rail.inclination)
        # Still need a 6DoF ascent at this inclination for azimuth steps
        apN, apE, apD, t_ap = _run_6dof_apogee(
            float(selected_inc), sim_cfg, vehicle, propellant, aero_model,
        )
        apogee_positions = {selected_inc: (apN, apE, apD)}
        apogee_times = {selected_inc: t_ap}
        ballistic_landings = {}
    _notify(1)

    t_apogee = apogee_times[selected_inc]

    # --- Azimuth selection ---
    if az_is_auto:
        # Narrowing
        feasible = narrow_azimuth_bounds(
            selected_inc, apogee_positions,
            sim_cfg, vehicle, propellant, wind_ensemble,
            buffered_poly,
        )
        _notify(2)

        # Optimisation
        top_candidates, az_obs = optimise_azimuth(
            feasible, selected_inc, apogee_positions, t_apogee,
            sim_cfg, vehicle, propellant, aero_model,
            wind_ensemble, poly_e, poly_n,
        )
        _notify(3)

        # Validation
        optimal_az, val_compliance, val_margins = validate_candidates(
            top_candidates, selected_inc,
            sim_cfg, vehicle, propellant, aero_model,
            wind_ensemble, buffered_poly, poly_e, poly_n,
        )
        _notify(4)
        selected_az = optimal_az
    else:
        selected_az = int(rail.azimuth)
        feasible = []
        az_obs = []
        top_candidates = []
        val_compliance = {}
        val_margins = {}

    return OptimisationResult(
        selected_azimuth=selected_az,
        selected_inclination=selected_inc,
        inclination_apogees=apogee_positions,
        inclination_ballistic_landings=ballistic_landings,
        inclination_selected=selected_inc,
        narrowing_feasible=feasible,
        narrowing_total_candidates=len(feasible),
        azimuth_observations=az_obs,
        azimuth_top_candidates=top_candidates,
        validation_compliance=val_compliance,
        validation_margins=val_margins,
    )
