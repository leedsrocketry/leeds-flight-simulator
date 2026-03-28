"""Configuration loader: YAML files → dataclasses.

Two public functions:
    load_simulation_config(path)  →  SimulationConfig
    load_vehicle_config(path)     →  VehicleConfig

Paths inside SimulationConfig are resolved relative to the simulation.yaml
file itself; paths inside VehicleConfig are resolved relative to vehicle.yaml.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Shared geographic sub-types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationStation:
    name: str
    latitude: float   # degrees
    longitude: float  # degrees
    radius: float     # metres


@dataclass(frozen=True)
class MapMarker:
    name: str
    latitude: float   # degrees
    longitude: float  # degrees


# ---------------------------------------------------------------------------
# Simulation config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteConfig:
    latitude: float                                # degrees
    longitude: float                               # degrees
    ballistic_exclusion_radius: float              # metres — minimum ballistic landing distance
                                                   # from launch site; used during inclination
                                                   # optimisation (§13.2)
    launch_observation_radius: float               # metres — radius of the automatic launch site
                                                   # observation station added to every LOS check
    altitude_ceiling: float                        # metres
    danger_area: Path
    coastline: Path | None                         # None → sea-landing check disabled
    observation_stations: tuple[ObservationStation, ...]
    map_markers: tuple[MapMarker, ...]


@dataclass(frozen=True)
class RailConfig:
    azimuth: float | Literal["auto"]              # degrees clockwise from North, or "auto"
    azimuth_range: tuple[float, float] | None     # [min, max] integer search range;
                                                   # required when azimuth == "auto"
    inclination: float | Literal["auto"]          # degrees from horizontal, or "auto"
    inclination_range: tuple[float, float] | None  # [min, max] integer search range;
                                                   # required when inclination == "auto"
    length: float                                  # metres


@dataclass(frozen=True)
class SurfaceWindConfig:
    speed_ms: float        # m/s
    bearing_deg: float     # degrees clockwise from North
    blend_height_m: float  # metres AGL — always required when section is present


@dataclass(frozen=True)
class LaunchConfig:
    rail: RailConfig
    wind_profiles: Path
    surface_wind: SurfaceWindConfig | None  # None → surface override disabled


@dataclass(frozen=True)
class UncertaintiesConfig:
    azimuth_sigma: float          # degrees
    inclination_sigma: float      # degrees
    fin_cant_sigma: float         # degrees
    impulse_factor_sigma: float   # fractional (e.g. 0.067 for 6.7 %)


@dataclass(frozen=True)
class AcceptanceConfig:
    compliance_threshold: float        # fractional (0.0–1.0) of landings inside danger area
    buffer_distance: float             # metres inward from danger area boundary
    sm_transition_mach: float          # Mach number dividing subsonic / supersonic SM check
    sm_subsonic_min: float             # calibres (M < sm_transition_mach)
    sm_supersonic_min: float           # calibres (M >= sm_transition_mach)
    aoa_max: float                     # degrees
    sm_aoa_threshold: float            # degrees: SM check applies when AoA < this
    sea_check_scenarios: tuple[str, ...]  # scenarios checked for sea landing
    los_check_scenarios: tuple[str, ...]  # scenarios checked for observation-station LOS


@dataclass(frozen=True)
class MonteCarloConfig:
    samples: int
    seed: int
    uncertainties: UncertaintiesConfig
    acceptance: AcceptanceConfig


@dataclass(frozen=True)
class SimulationConfig:
    vehicle: Path           # resolved path to vehicle.yaml
    site: SiteConfig
    launch: LaunchConfig
    monte_carlo: MonteCarloConfig


# ---------------------------------------------------------------------------
# Vehicle config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleGeometry:
    diameter: float         # m — reference diameter
    length: float           # m — total length
    nozzle_position: float  # m from nosecone tip — nozzle exit plane
    nozzle_diameter: float  # m — nozzle exit diameter (for thrust correction)
    fin_cp_radius: float    # m — fin CP spanwise distance from centreline

    @property
    def reference_area(self) -> float:
        """π·d²/4 [m²] — derived from diameter."""
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def nozzle_area(self) -> float:
        """π·dₑ²/4 [m²] — nozzle exit area for pressure thrust correction."""
        return math.pi * self.nozzle_diameter ** 2 / 4.0


@dataclass(frozen=True)
class VehicleMass:
    wet_mass: float              # kg — total mass at launch
    wet_cg: float                # m from nosecone tip — wet vehicle CG
    wet_motor_cg: float          # m from nosecone tip — loaded motor CG
                                 # used as propellant CG (inside-out burn model)
    propellant_inertia_roll: float     # kg·m² — propellant roll inertia about roll axis
    propellant_inertia_lateral: float  # kg·m² — propellant lateral inertia about propellant CG
    wet_inertia_lateral: float   # kg·m² — wet vehicle lateral inertia about wet CG
    wet_inertia_roll: float      # kg·m² — wet vehicle roll inertia about roll axis


@dataclass(frozen=True)
class ParachuteConfig:
    """Configuration for a single parachute stage."""
    cd: float                                  # drag coefficient
    area: float                                # m² — reference area
    threshold: float | Literal["apogee"]       # deploy altitude [m AGL] or "apogee"


@dataclass(frozen=True)
class VehicleRecovery:
    """Recovery system configuration.

    Valid configurations (drogue without main is not permitted):

        both drogue and main  — all up to four scenarios may be active
        main only             — ``drogue_only`` never generated
        neither               — only ``nominal`` generated

    Active descent scenarios derived from this configuration (§9):

        ``nominal``        — always
        ``ballistic``      — at least one parachute configured
        ``drogue_only``    — both drogue AND main configured
        ``premature_main`` — main configured AND ``main.threshold`` is numeric
    """
    drogue: ParachuteConfig | None  # None → no drogue stage
    main: ParachuteConfig | None    # None → no main stage

    @property
    def active_scenarios(self) -> tuple[str, ...]:
        """Return the tuple of active descent scenario names for this recovery config."""
        scenarios: list[str] = ["nominal"]
        if self.drogue is not None or self.main is not None:
            scenarios.append("ballistic")
        if self.drogue is not None and self.main is not None:
            scenarios.append("drogue_only")
        if self.main is not None and isinstance(self.main.threshold, float):
            scenarios.append("premature_main")
        return tuple(scenarios)


@dataclass(frozen=True)
class VehicleConfig:
    geometry: VehicleGeometry
    mass: VehicleMass
    recovery: VehicleRecovery
    motor: Path               # resolved path to .eng file
    aero_tables: Path         # resolved path to aero tables directory
    fins_aero_table: Path | None  # resolved path to fins component CSV, or None to
                                  # use the filename heuristic in aerodynamics.py

    @property
    def reference_area(self) -> float:
        """Convenience accessor — delegates to geometry.reference_area."""
        return self.geometry.reference_area


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _parse_auto(value: object) -> float | Literal["auto"]:
    if value == "auto":
        return "auto"
    return float(value)


def _parse_threshold(value: object) -> float | Literal["apogee"]:
    if value == "apogee":
        return "apogee"
    return float(value)


def _parse_range(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    return (float(value[0]), float(value[1]))


def load_simulation_config(path: Path | str) -> SimulationConfig:
    """Parse *path* (simulation.yaml) and return a SimulationConfig.

    All file paths are resolved relative to the directory containing the
    simulation.yaml file.
    """
    path = Path(path).resolve()
    base_dir = path.parent

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    def _resolve(p: str) -> Path:
        return (base_dir / p).resolve()

    # -- vehicle path
    vehicle = _resolve(str(raw["vehicle"]))

    # -- site
    site_raw = raw["site"]
    stations_raw = site_raw.get("observation_stations") or []
    markers_raw = site_raw.get("map_markers") or []
    coastline_raw = site_raw.get("coastline")
    coastline: Path | None = None if coastline_raw is None else _resolve(coastline_raw)

    site = SiteConfig(
        latitude=float(site_raw["latitude"]),
        longitude=float(site_raw["longitude"]),
        ballistic_exclusion_radius=float(site_raw["ballistic_exclusion_radius"]),
        launch_observation_radius=float(site_raw["launch_observation_radius"]),
        altitude_ceiling=float(site_raw["altitude_ceiling"]),
        danger_area=_resolve(site_raw["danger_area"]),
        coastline=coastline,
        observation_stations=tuple(
            ObservationStation(
                name=str(s["name"]),
                latitude=float(s["latitude"]),
                longitude=float(s["longitude"]),
                radius=float(s["radius"]),
            )
            for s in stations_raw
        ),
        map_markers=tuple(
            MapMarker(
                name=str(m["name"]),
                latitude=float(m["latitude"]),
                longitude=float(m["longitude"]),
            )
            for m in markers_raw
        ),
    )

    # -- launch
    launch_raw = raw["launch"]
    rail_raw = launch_raw["rail"]
    sw_raw = launch_raw.get("surface_wind")

    surface_wind: SurfaceWindConfig | None
    if sw_raw is not None:
        surface_wind = SurfaceWindConfig(
            speed_ms=float(sw_raw["speed_ms"]),
            bearing_deg=float(sw_raw["bearing_deg"]),
            blend_height_m=float(sw_raw["blend_height_m"]),
        )
    else:
        surface_wind = None

    rail_azimuth = _parse_auto(rail_raw["azimuth"])
    rail_inclination = _parse_auto(rail_raw["inclination"])
    azimuth_range = _parse_range(rail_raw.get("azimuth_range"))
    inclination_range = _parse_range(rail_raw.get("inclination_range"))

    if rail_azimuth == "auto" and azimuth_range is None:
        raise ValueError(
            "launch.rail.azimuth_range is required when azimuth is 'auto'"
        )
    if rail_inclination == "auto" and inclination_range is None:
        raise ValueError(
            "launch.rail.inclination_range is required when inclination is 'auto'"
        )

    launch = LaunchConfig(
        rail=RailConfig(
            azimuth=rail_azimuth,
            azimuth_range=azimuth_range,
            inclination=rail_inclination,
            inclination_range=inclination_range,
            length=float(rail_raw["length"]),
        ),
        wind_profiles=_resolve(launch_raw["wind_profiles"]),
        surface_wind=surface_wind,
    )

    # -- monte_carlo
    mc_raw = raw["monte_carlo"]
    unc_raw = mc_raw["uncertainties"]
    acc_raw = mc_raw["acceptance"]

    monte_carlo = MonteCarloConfig(
        samples=int(mc_raw["samples"]),
        seed=int(mc_raw["seed"]),
        uncertainties=UncertaintiesConfig(
            azimuth_sigma=float(unc_raw["azimuth_sigma"]),
            inclination_sigma=float(unc_raw["inclination_sigma"]),
            fin_cant_sigma=float(unc_raw["fin_cant_sigma"]),
            impulse_factor_sigma=float(unc_raw["impulse_factor_sigma"]),
        ),
        acceptance=AcceptanceConfig(
            compliance_threshold=float(acc_raw["compliance_threshold"]),
            buffer_distance=float(acc_raw["buffer_distance"]),
            sm_transition_mach=float(acc_raw["sm_transition_mach"]),
            sm_subsonic_min=float(acc_raw["sm_subsonic_min"]),
            sm_supersonic_min=float(acc_raw["sm_supersonic_min"]),
            aoa_max=float(acc_raw["aoa_max"]),
            sm_aoa_threshold=float(acc_raw["sm_aoa_threshold"]),
            sea_check_scenarios=tuple(
                str(s) for s in (acc_raw.get("sea_check_scenarios") or [])
            ),
            los_check_scenarios=tuple(
                str(s) for s in (acc_raw.get("los_check_scenarios") or [])
            ),
        ),
    )

    return SimulationConfig(
        vehicle=vehicle,
        site=site,
        launch=launch,
        monte_carlo=monte_carlo,
    )


def load_vehicle_config(path: Path | str) -> VehicleConfig:
    """Parse *path* (vehicle.yaml) and return a VehicleConfig.

    All file paths are resolved relative to the directory containing the
    vehicle.yaml file.
    """
    path = Path(path).resolve()
    base_dir = path.parent

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    def _resolve(p: str) -> Path:
        return (base_dir / p).resolve()

    geom = raw["geometry"]
    mass = raw["mass"]
    recovery_raw = raw.get("recovery") or {}

    def _parse_chute(key: str) -> ParachuteConfig | None:
        r = recovery_raw.get(key)
        if r is None:
            return None
        return ParachuteConfig(
            cd=float(r["cd"]),
            area=float(r["area"]),
            threshold=_parse_threshold(r["threshold"]),
        )

    drogue = _parse_chute("drogue")
    main = _parse_chute("main")
    if drogue is not None and main is None:
        raise ValueError(
            "Recovery configuration error: a drogue parachute is configured but "
            "no main parachute is present. Valid configurations are: both drogue "
            "and main, main only, or neither."
        )

    fins_raw = raw.get("fins_aero_table")
    fins_aero_table: Path | None = _resolve(fins_raw) if fins_raw is not None else None

    return VehicleConfig(
        geometry=VehicleGeometry(
            diameter=float(geom["diameter"]),
            length=float(geom["length"]),
            nozzle_position=float(geom["nozzle_position"]),
            nozzle_diameter=float(geom["nozzle_diameter"]),
            fin_cp_radius=float(geom["fin_cp_radius"]),
        ),
        mass=VehicleMass(
            wet_mass=float(mass["wet_mass"]),
            wet_cg=float(mass["wet_cg"]),
            wet_motor_cg=float(mass["wet_motor_cg"]),
            propellant_inertia_roll=float(mass["propellant_inertia_roll"]),
            propellant_inertia_lateral=float(mass["propellant_inertia_lateral"]),
            wet_inertia_lateral=float(mass["wet_inertia_lateral"]),
            wet_inertia_roll=float(mass["wet_inertia_roll"]),
        ),
        recovery=VehicleRecovery(
            drogue=drogue,
            main=main,
        ),
        motor=_resolve(raw["motor"]),
        aero_tables=_resolve(raw["aero_tables"]),
        fins_aero_table=fins_aero_table,
    )


# ---------------------------------------------------------------------------
# Motor data dataclass  (raw parsed output of load_motor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorData:
    """Raw data parsed from a RASP .eng file.

    Masses are in kg, time in seconds, thrust in Newtons.
    ``m_motor_kg`` is the total motor mass (casing + propellant) as stated in
    the .eng header (\"total weight\" field).
    """
    name: str
    m_prop_kg: float          # propellant mass [kg]
    m_motor_kg: float         # total motor mass: casing + propellant [kg]
    time_s: np.ndarray        # (K,) thrust curve time points [s]
    thrust_n: np.ndarray      # (K,) thrust values [N]


# ---------------------------------------------------------------------------
# Motor loader
# ---------------------------------------------------------------------------

def load_motor(path: Path | str) -> MotorData:
    """Parse a RASP .eng file and return a MotorData.

    Format expected::

        ; optional comment lines
        Name Diam_mm Length_mm Delays PropMass_kg TotalMass_kg Manufacturer
        time_s  thrust_N
        ...

    Masses are in kg. Thrust in Newtons. The final data point should have
    thrust = 0; if absent it is appended automatically.

    Raises
    ------
    ValueError
        If the file cannot be parsed or contains physically implausible values.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    # Strip comments and blank lines
    data_lines = [l.strip() for l in lines
                  if l.strip() and not l.strip().startswith(";")]

    if len(data_lines) < 2:
        raise ValueError(f"Motor file {path} has no usable data")

    # --- header line
    header = data_lines[0].split()
    if len(header) < 7:
        raise ValueError(
            f"Motor file header must have ≥7 fields, got: {data_lines[0]!r}"
        )
    name = header[0]
    try:
        m_prop_kg = float(header[4])
        m_motor_kg = float(header[5])
    except ValueError as exc:
        raise ValueError(
            f"Could not parse motor masses from header: {data_lines[0]!r}"
        ) from exc

    if m_prop_kg <= 0:
        raise ValueError(f"Propellant mass must be > 0, got {m_prop_kg}")
    if m_motor_kg <= m_prop_kg:
        raise ValueError(
            f"Total motor mass ({m_motor_kg} kg) must exceed propellant mass "
            f"({m_prop_kg} kg)"
        )

    # --- thrust curve data points
    times: list[float] = []
    thrusts: list[float] = []
    for line in data_lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t, f = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        times.append(t)
        thrusts.append(f)

    if len(times) < 2:
        raise ValueError(f"Motor file {path} must have ≥2 thrust data points")

    time_arr = np.asarray(times, dtype=np.float64)
    thrust_arr = np.asarray(thrusts, dtype=np.float64)

    if not np.all(np.diff(time_arr) > 0):
        raise ValueError("Thrust curve time points must be strictly increasing")
    if np.any(thrust_arr < 0):
        raise ValueError("Thrust values must be non-negative")

    # Ensure burnout point has thrust = 0
    if thrust_arr[-1] != 0.0:
        time_arr = np.append(time_arr, time_arr[-1])
        thrust_arr = np.append(thrust_arr, 0.0)

    return MotorData(
        name=name,
        m_prop_kg=m_prop_kg,
        m_motor_kg=m_motor_kg,
        time_s=time_arr,
        thrust_n=thrust_arr,
    )
