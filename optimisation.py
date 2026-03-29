"""Automatic launch rail azimuth and inclination optimisation.

Four-phase routine (specification §13) to select optimal integer launch
azimuth and inclination, maximising the probability that all active descent
scenarios remain within the buffered danger area.

Phase 1 — Inclination selection (deterministic 3DoF sweeps)
Phase 2 — Azimuth bound narrowing (analytical wind-drift filter)
Phase 3 — Azimuth optimisation (1-D Bayesian optimisation, GP + UCB)
Phase 4 — Candidate validation (full-uncertainty MC)

Public API
----------
run_optimisation(sim_cfg, vehicle_cfg, motor_model, aero_model,
                 wind_ensemble, progress_callback) → OptimisationResult
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from shapely.geometry import Point, Polygon

from atmosphere import isa, density
from config import SimulationConfig, VehicleConfig
from motor import MotorModel
from aerodynamics import AeroModel
from wind import WindEnsemble, interpolate_wind
from dynamics import (
    SimParams,
    simulate_ascent_3dof,
    integrate_descent,
    SCENARIO_MAP,
    SCENARIO_BALLISTIC,
    SCENARIO_NOMINAL,
    SCENARIO_PREMATURE_MAIN,
    SCENARIO_DROGUE_ONLY,
)
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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptimisationResult:
    """Output of the full optimisation routine."""
    selected_azimuth: int          # degrees
    selected_inclination: int      # degrees

    # Phase 1 diagnostics
    phase1_apogees: dict[int, tuple[float, float, float]]       # inc → (N, E, D)
    phase1_ballistic_landings: dict[int, tuple[float, float]]   # inc → (N, E)
    phase1_selected: int

    # Phase 2 diagnostics
    phase2_feasible: list[int]     # surviving azimuth candidates
    phase2_total_candidates: int

    # Phase 3 diagnostics
    phase3_observations: list[tuple[int, float]]  # (az, p_success) pairs
    phase3_top_candidates: list[int]

    # Phase 4 diagnostics
    phase4_compliance: dict[int, float]    # az → compliance fraction
    phase4_margins: dict[int, float]       # az → margin in metres


# ---------------------------------------------------------------------------
# Worst-drift scenario helpers
# ---------------------------------------------------------------------------

def _worst_drift_scenario(vehicle_cfg: VehicleConfig) -> str:
    """Return the active scenario with the most descent drift.

    Premature-main drifts most (both chutes from apogee).  If not active
    (main deploys at apogee), nominal drifts most (drogue → main).  Then
    drogue-only.  Ballistic drifts least.
    """
    active = vehicle_cfg.recovery.active_scenarios
    if "premature_main" in active:
        return "premature_main"
    if "nominal" in active and vehicle_cfg.recovery.main is not None:
        return "nominal"
    if "drogue_only" in active:
        return "drogue_only"
    return "ballistic"


def _worst_drift_cda(vehicle_cfg: VehicleConfig) -> float:
    """Return the effective CdA for the worst-drift scenario.

    For premature-main: drogue + main (both deploy at apogee).
    For nominal with altitude-triggered main: drogue CdA above main deploy,
    main CdA below — we use main CdA as the conservative (slower descent,
    more drift) estimate for the analytical filter.
    """
    recovery = vehicle_cfg.recovery
    scenario = _worst_drift_scenario(vehicle_cfg)
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


def _worst_drift_scenario_int(vehicle_cfg: VehicleConfig) -> int:
    """Return the integer scenario code for the worst-drift scenario."""
    return SCENARIO_MAP[_worst_drift_scenario(vehicle_cfg)]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotate_apogee(
    apogee_N: float, apogee_E: float,
    azimuth_rad: float, base_azimuth_rad: float = 0.0,
) -> tuple[float, float]:
    """Rotate an apogee NE position from *base_azimuth* to *azimuth*.

    The 3DoF ascent is run at ``base_azimuth`` (typically 0°).  For a
    different azimuth, the horizontal displacement rotates by the
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
# Phase 2: wind drift calculation
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
# Phase 1: Inclination selection
# ---------------------------------------------------------------------------

def select_inclination(
    sim_cfg: SimulationConfig,
    vehicle_cfg: VehicleConfig,
    motor_model: MotorModel,
    aero_model: AeroModel,
    poly_e: np.ndarray,
    poly_n: np.ndarray,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[int, dict[int, tuple[float, float, float]], dict[int, tuple[float, float]], dict[int, float]]:
    """Phase 1: select the steepest safe inclination.

    Returns
    -------
    (selected_inclination, apogee_positions, ballistic_landings, apogee_times)
    """
    rail_cfg = sim_cfg.launch.rail
    inc_range = rail_cfg.inclination_range
    assert inc_range is not None
    candidates = list(range(int(inc_range[0]), int(inc_range[1]) + 1))

    geom = vehicle_cfg.geometry
    exclusion_r = sim_cfg.site.ballistic_exclusion_radius

    apogee_positions: dict[int, tuple[float, float, float]] = {}
    ballistic_landings: dict[int, tuple[float, float]] = {}
    apogee_times: dict[int, float] = {}
    valid: list[int] = []

    for idx, inc in enumerate(candidates):
        # 3DoF ascent at azimuth=0, no wind, no uncertainty
        apogee_alt, apN, apE, apD, t_ap, V_ap = simulate_ascent_3dof(
            rail_azimuth_rad=0.0,
            rail_inclination_rad=inc * _DEG2RAD,
            rail_length=rail_cfg.length,
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
        apogee_positions[inc] = (apN, apE, apD)
        apogee_times[inc] = t_ap

        # Ballistic descent from apogee — run integrate_descent to get
        # actual landing point (accounts for gravity during fall).
        descent_state0 = np.array([
            apN, apE, apD, 0.0, 0.0, 0.0,
        ], dtype=np.float64)
        # No wind for Phase 1 — zero arrays
        zero_wind_alt = np.array([0.0, 50000.0], dtype=np.float64)
        zero_wind_e = np.zeros(2, dtype=np.float64)
        zero_wind_n = np.zeros(2, dtype=np.float64)

        t_desc, y_desc, n_desc = integrate_descent(
            t_ap, descent_state0,
            zero_wind_alt, zero_wind_e, zero_wind_n,
            aero_model.mach_grid, aero_model.re_grid,
            aero_model.alpha_grid, aero_model.ca_table,
            geom.reference_area, geom.length,
            motor_model.m_dry,
            0.0, 0.0, -1.0,  # no parachutes, sentinel deploy alt
            SCENARIO_BALLISTIC,
            1.0e-6, 1.0e-6,
        )

        land_N = float(y_desc[n_desc - 1, 0])
        land_E = float(y_desc[n_desc - 1, 1])
        ballistic_landings[inc] = (land_N, land_E)

        # Check exclusion radius and containment
        dist = math.hypot(land_N, land_E)
        inside = _point_in_polygon(land_E, land_N, poly_e, poly_n)

        if dist >= exclusion_r and inside:
            valid.append(inc)

        if progress_callback is not None:
            progress_callback("Phase 1: Inclination", idx + 1, len(candidates))

    if not valid:
        raise ValueError(
            f"No inclination in range [{inc_range[0]}, {inc_range[1]}] satisfies "
            f"both the ballistic exclusion radius ({exclusion_r} m) and the "
            f"buffered danger area constraint. Consider widening the range or "
            f"reducing the exclusion radius."
        )

    selected = max(valid)
    return selected, apogee_positions, ballistic_landings, apogee_times


# ---------------------------------------------------------------------------
# Phase 2: Azimuth bound narrowing
# ---------------------------------------------------------------------------

def narrow_azimuth_bounds(
    selected_inclination: int,
    apogee_positions: dict[int, tuple[float, float, float]],
    sim_cfg: SimulationConfig,
    vehicle_cfg: VehicleConfig,
    motor_model: MotorModel,
    wind_ensemble: WindEnsemble,
    buffered_polygon: Polygon,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> list[int]:
    """Phase 2: discard azimuths whose landing centroid falls outside the
    buffered danger area.

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

    cda = _worst_drift_cda(vehicle_cfg)
    m_dry = motor_model.m_dry

    feasible: list[int] = []
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

        if buffered_polygon.contains(Point(centroid_E, centroid_N)):
            feasible.append(az)

    if progress_callback is not None:
        progress_callback("Phase 2: Narrowing", 1, 1)

    if not feasible:
        raise ValueError(
            f"No azimuth in range [{az_min}, {az_max}] produces a landing "
            f"centroid inside the buffered danger area. Wind conditions may be "
            f"too strong or the danger area too small."
        )

    return feasible


# ---------------------------------------------------------------------------
# Phase 3: Bayesian optimisation helpers
# ---------------------------------------------------------------------------

def _run_descent_single(
    apogee_N: float, apogee_E: float, apogee_D: float, t_apogee: float,
    wind_alt: np.ndarray, wind_east: np.ndarray, wind_north: np.ndarray,
    mach_g: np.ndarray, re_g: np.ndarray, alpha_g: np.ndarray,
    ca_tbl: np.ndarray,
    A_ref: float, ref_length: float,
    m_dry: float,
    drogue_cda: float, main_cda: float, main_deploy_alt: float,
    scenario: int,
) -> tuple[float, float]:
    """Run a single descent from known apogee and return (landing_N, landing_E)."""
    state0 = np.array([
        apogee_N, apogee_E, apogee_D, 0.0, 0.0, 0.0,
    ], dtype=np.float64)

    t_desc, y_desc, n_desc = integrate_descent(
        t_apogee, state0,
        wind_alt, wind_east, wind_north,
        mach_g, re_g, alpha_g, ca_tbl,
        A_ref, ref_length,
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
    aero_model: AeroModel,
    vehicle_cfg: VehicleConfig,
    motor_model: MotorModel,
    scenario_int: int,
    n_sims: int,
    pool: mp.pool.Pool | None = None,
) -> np.ndarray:
    """Run *n_sims* descent sims from a known apogee, varying wind profile.

    Returns (n_sims, 2) array of [landing_N, landing_E].
    """
    recovery = vehicle_cfg.recovery
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
            aero_model.mach_grid, aero_model.re_grid,
            aero_model.alpha_grid, aero_model.ca_table,
            vehicle_cfg.geometry.reference_area, vehicle_cfg.geometry.length,
            motor_model.m_dry,
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
    aero_model: AeroModel,
    vehicle_cfg: VehicleConfig,
    motor_model: MotorModel,
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
        wind_ensemble, aero_model, vehicle_cfg, motor_model,
        scenario_int, n_sims, pool,
    )

    n_inside = 0
    for j in range(landings.shape[0]):
        if _point_in_polygon(landings[j, 1], landings[j, 0], poly_e, poly_n):
            n_inside += 1

    return n_inside / n_sims


# ---------------------------------------------------------------------------
# Phase 3: Azimuth optimisation (GP + UCB)
# ---------------------------------------------------------------------------

def optimise_azimuth(
    feasible_azimuths: list[int],
    selected_inclination: int,
    apogee_positions: dict[int, tuple[float, float, float]],
    t_apogee: float,
    sim_cfg: SimulationConfig,
    vehicle_cfg: VehicleConfig,
    motor_model: MotorModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    poly_e: np.ndarray,
    poly_n: np.ndarray,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[list[int], list[tuple[int, float]]]:
    """Phase 3: Bayesian optimisation over feasible azimuths.

    Returns (top_candidates, observations) where observations is a list
    of (azimuth, p_success) pairs from all evaluations.
    """
    scenario_int = _worst_drift_scenario_int(vehicle_cfg)
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
                    wind_ensemble, aero_model, vehicle_cfg, motor_model,
                    poly_e, poly_n, scenario_int, SIMS_PER_ITER, pool,
                )
                observations.append((az, p))
                if progress_callback:
                    progress_callback("Phase 3: Evaluation", i + 1, len(feasible_azimuths))

        observations.sort(key=lambda x: x[1], reverse=True)
        return [az for az, _ in observations], observations

    # --- Bayesian optimisation ---
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from scipy.stats import norm
    from scipy.stats.qmc import Sobol

    # Initial evaluation points via Sobol
    n_init = min(5, len(feasible_azimuths))
    sobol = Sobol(d=1, scramble=True, seed=sim_cfg.monte_carlo.seed)
    sobol_points = sobol.random(n_init).flatten()
    init_indices = np.unique(np.clip(
        (sobol_points * len(feasible_azimuths)).astype(int),
        0, len(feasible_azimuths) - 1,
    ))

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
                wind_ensemble, aero_model, vehicle_cfg, motor_model,
                poly_e, poly_n, scenario_int, SIMS_PER_ITER, pool,
            )
            X_obs.append(float(az))
            Y_obs.append(p)
            observations.append((az, p))
            evaluated.add(az)

        if progress_callback:
            progress_callback("Phase 3: BO init", len(evaluated), MAX_ITER + len(init_indices))

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
                wind_ensemble, aero_model, vehicle_cfg, motor_model,
                poly_e, poly_n, scenario_int, SIMS_PER_ITER, pool,
            )
            X_obs.append(float(next_az))
            Y_obs.append(p)
            observations.append((next_az, p))
            evaluated.add(next_az)

            if progress_callback:
                progress_callback(
                    "Phase 3: BO iteration",
                    len(init_indices) + iteration + 1,
                    MAX_ITER + len(init_indices),
                )

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
# Phase 4: Candidate validation
# ---------------------------------------------------------------------------

def _validation_worker(args: tuple) -> tuple[int, list[tuple[float, float]]]:
    """Worker that runs N full-uncertainty sims for one azimuth candidate.

    Returns (azimuth, list_of_(landing_N, landing_E)).
    """
    (azimuth_deg, selected_inclination, sim_cfg, vehicle_cfg,
     motor_model, aero_model, wind_ensemble, n_sims, scenario_name) = args

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
            vehicle_cfg=vehicle_cfg,
            motor_model=motor_model,
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
    vehicle_cfg: VehicleConfig,
    motor_model: MotorModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    buffered_polygon: Polygon,
    poly_e: np.ndarray,
    poly_n: np.ndarray,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[int, dict[int, float], dict[int, float]]:
    """Phase 4: validate top candidates with full-uncertainty MC.

    Returns (optimal_azimuth, compliance_fractions, margins).
    """
    SIMS_PER_CANDIDATE = 500
    scenario_name = _worst_drift_scenario(vehicle_cfg)
    compliance_threshold = sim_cfg.monte_carlo.acceptance.compliance_threshold

    args_list = [
        (az, selected_inclination, sim_cfg, vehicle_cfg,
         motor_model, aero_model, wind_ensemble,
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

    if progress_callback:
        progress_callback("Phase 4: Validation", 1, 1)

    # Select candidate with greatest margin
    optimal = max(top_candidates, key=lambda az: margins[az])
    return optimal, compliance, margins


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_optimisation(
    sim_cfg: SimulationConfig,
    vehicle_cfg: VehicleConfig,
    motor_model: MotorModel,
    aero_model: AeroModel,
    wind_ensemble: WindEnsemble,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> OptimisationResult:
    """Run the full optimisation routine.

    Checks which parameters are ``"auto"`` and runs only the required phases.
    """
    rail = sim_cfg.launch.rail
    inc_is_auto = rail.inclination == "auto"
    az_is_auto = rail.azimuth == "auto"

    # Load and buffer the danger area
    site = sim_cfg.site
    acc = sim_cfg.monte_carlo.acceptance
    danger_poly = load_polygon_ned(site.danger_area, site.latitude, site.longitude)
    buffered_poly = buffer_danger_area(danger_poly, acc.buffer_distance)
    poly_e, poly_n = polygon_to_arrays(buffered_poly)

    # --- Phase 1: Inclination ---
    if inc_is_auto:
        selected_inc, apogee_positions, ballistic_landings, apogee_times = (
            select_inclination(
                sim_cfg, vehicle_cfg, motor_model, aero_model,
                poly_e, poly_n, progress_callback,
            )
        )
    else:
        selected_inc = int(rail.inclination)
        # Still need a 3DoF ascent at this inclination for Phases 2-4
        geom = vehicle_cfg.geometry
        apogee_alt, apN, apE, apD, t_ap, V_ap = simulate_ascent_3dof(
            rail_azimuth_rad=0.0,
            rail_inclination_rad=selected_inc * _DEG2RAD,
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
        apogee_positions = {selected_inc: (apN, apE, apD)}
        apogee_times = {selected_inc: t_ap}
        ballistic_landings = {}

    t_apogee = apogee_times[selected_inc]

    # --- Phases 2-4: Azimuth ---
    if az_is_auto:
        # Phase 2
        feasible = narrow_azimuth_bounds(
            selected_inc, apogee_positions,
            sim_cfg, vehicle_cfg, motor_model, wind_ensemble,
            buffered_poly, progress_callback,
        )

        # Phase 3
        top_candidates, phase3_obs = optimise_azimuth(
            feasible, selected_inc, apogee_positions, t_apogee,
            sim_cfg, vehicle_cfg, motor_model, aero_model,
            wind_ensemble, poly_e, poly_n, progress_callback,
        )

        # Phase 4
        optimal_az, phase4_compliance, phase4_margins = validate_candidates(
            top_candidates, selected_inc,
            sim_cfg, vehicle_cfg, motor_model, aero_model,
            wind_ensemble, buffered_poly, poly_e, poly_n,
            progress_callback,
        )
        selected_az = optimal_az
    else:
        selected_az = int(rail.azimuth)
        feasible = []
        phase3_obs = []
        top_candidates = []
        phase4_compliance = {}
        phase4_margins = {}

    return OptimisationResult(
        selected_azimuth=selected_az,
        selected_inclination=selected_inc,
        phase1_apogees=apogee_positions,
        phase1_ballistic_landings=ballistic_landings,
        phase1_selected=selected_inc,
        phase2_feasible=feasible,
        phase2_total_candidates=len(feasible),
        phase3_observations=phase3_obs,
        phase3_top_candidates=top_candidates,
        phase4_compliance=phase4_compliance,
        phase4_margins=phase4_margins,
    )
