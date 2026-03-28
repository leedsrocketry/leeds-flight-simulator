"""Configuration loader: YAML files → dataclasses.

Two public functions:
    load_simulation_config(path)  →  SimulationConfig
    load_vehicle_config(path)     →  VehicleConfig

All paths inside SimulationConfig are resolved relative to the simulation.yaml
file itself, so the caller can pass an absolute or relative path and the
referenced input files will be found correctly.
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
    min_safe_radius: float                         # metres
    observation_stations: tuple[ObservationStation, ...]
    map_markers: tuple[MapMarker, ...]


@dataclass(frozen=True)
class RailConfig:
    azimuth: float | Literal["auto"]      # degrees clockwise from North, or "auto"
    inclination: float | Literal["auto"]  # degrees from horizontal, or "auto"
    length: float                         # metres


@dataclass(frozen=True)
class SurfaceWindConfig:
    speed_ms: float        # m/s
    bearing_deg: float     # degrees clockwise from North
    blend_height_m: float  # metres AGL — always required when section is present


@dataclass(frozen=True)
class LaunchConfig:
    rail: RailConfig
    surface_wind: SurfaceWindConfig | None  # None → surface override disabled


@dataclass(frozen=True)
class UncertaintiesConfig:
    azimuth_sigma: float          # degrees
    inclination_sigma: float      # degrees
    fin_cant_sigma: float         # degrees
    impulse_factor_sigma: float   # fractional (e.g. 0.067 for 6.7 %)


@dataclass(frozen=True)
class AcceptanceConfig:
    compliance_threshold: float   # percent of landings inside danger area
    buffer_distance: float        # metres inward from danger area boundary
    altitude_ceiling: float       # metres
    sm_transition_mach: float     # Mach number dividing subsonic / supersonic SM check
    sm_subsonic_min: float        # calibres (M < sm_transition_mach)
    sm_supersonic_min: float      # calibres (M >= sm_transition_mach)
    aoa_max: float                # degrees
    sm_aoa_threshold: float       # degrees: SM check applies when AoA < this


@dataclass(frozen=True)
class MonteCarloConfig:
    samples: int
    seed: int
    uncertainties: UncertaintiesConfig
    acceptance: AcceptanceConfig


@dataclass(frozen=True)
class PathsConfig:
    vehicle: Path
    motor: Path
    aero_tables: Path
    wind_profiles: Path
    danger_area: Path
    coastline: Path | None  # None → sea-landing check disabled


@dataclass(frozen=True)
class SimulationConfig:
    site: SiteConfig
    launch: LaunchConfig
    monte_carlo: MonteCarloConfig
    paths: PathsConfig


# ---------------------------------------------------------------------------
# Vehicle config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleGeometry:
    diameter: float  # m
    length: float    # m

    @property
    def reference_area(self) -> float:
        """π·d²/4 [m²] — derived from diameter."""
        return math.pi * self.diameter ** 2 / 4.0


@dataclass(frozen=True)
class VehicleMass:
    wet: float              # kg — mass at launch (airframe + loaded motor)
    dry: float              # kg — mass after burnout (airframe + empty casing)
    cg_dry: float           # m from nosecone tip (dry vehicle incl. empty casing)
    motor_cg_loaded: float  # m from nosecone tip — used in dynamics.py CG assembly:
                            #   CG(t) = (m_dry·cg_dry + m_prop(t)·motor_cg_loaded)
                            #           / (m_dry + m_prop(t))


@dataclass(frozen=True)
class VehicleInertia:
    I_R_wet: float  # kg·m²  roll axis, wet
    I_R_dry: float  # kg·m²  roll axis, dry
    I_L_wet: float  # kg·m²  lateral axis, wet
    I_L_dry: float  # kg·m²  lateral axis, dry


@dataclass(frozen=True)
class VehicleNozzle:
    exit: float  # m from nosecone tip


@dataclass(frozen=True)
class VehicleRecovery:
    CdA_drogue: float           # m²
    CdA_main: float             # m²
    deploy_altitude_agl: float  # m above launch site


@dataclass(frozen=True)
class VehicleRoll:
    r_fin: float  # m — fin CP spanwise distance from centreline


@dataclass(frozen=True)
class VehicleConfig:
    geometry: VehicleGeometry
    mass: VehicleMass
    inertia: VehicleInertia
    nozzle: VehicleNozzle
    recovery: VehicleRecovery
    roll: VehicleRoll

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


def load_simulation_config(path: Path | str) -> SimulationConfig:
    """Parse *path* (simulation.yaml) and return a SimulationConfig.

    All file paths in the ``paths`` section are resolved relative to the
    directory containing the simulation.yaml file.
    """
    path = Path(path).resolve()
    base_dir = path.parent

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    def _resolve(p: str) -> Path:
        return (base_dir / p).resolve()

    # -- site
    site_raw = raw["site"]
    stations_raw = site_raw.get("observation_stations") or []
    markers_raw = site_raw.get("map_markers") or []

    site = SiteConfig(
        latitude=float(site_raw["latitude"]),
        longitude=float(site_raw["longitude"]),
        min_safe_radius=float(site_raw["min_safe_radius"]),
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

    launch = LaunchConfig(
        rail=RailConfig(
            azimuth=_parse_auto(rail_raw["azimuth"]),
            inclination=_parse_auto(rail_raw["inclination"]),
            length=float(rail_raw["length"]),
        ),
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
            altitude_ceiling=float(acc_raw["altitude_ceiling"]),
            sm_transition_mach=float(acc_raw["sm_transition_mach"]),
            sm_subsonic_min=float(acc_raw["sm_subsonic_min"]),
            sm_supersonic_min=float(acc_raw["sm_supersonic_min"]),
            aoa_max=float(acc_raw["aoa_max"]),
            sm_aoa_threshold=float(acc_raw["sm_aoa_threshold"]),
        ),
    )

    # -- paths
    paths_raw = raw["paths"]
    coastline_raw = paths_raw.get("coastline")
    coastline: Path | None = None if coastline_raw is None else _resolve(coastline_raw)

    paths = PathsConfig(
        vehicle=_resolve(paths_raw["vehicle"]),
        motor=_resolve(paths_raw["motor"]),
        aero_tables=_resolve(paths_raw["aero_tables"]),
        wind_profiles=_resolve(paths_raw["wind_profiles"]),
        danger_area=_resolve(paths_raw["danger_area"]),
        coastline=coastline,
    )

    return SimulationConfig(site=site, launch=launch, monte_carlo=monte_carlo, paths=paths)


def load_vehicle_config(path: Path | str) -> VehicleConfig:
    """Parse *path* (vehicle.yaml) and return a VehicleConfig."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    geom = raw["geometry"]
    mass = raw["mass"]
    inertia = raw["inertia"]
    nozzle = raw["nozzle"]
    recovery = raw["recovery"]
    roll = raw["roll"]

    return VehicleConfig(
        geometry=VehicleGeometry(
            diameter=float(geom["diameter"]),
            length=float(geom["length"]),
        ),
        mass=VehicleMass(
            wet=float(mass["wet"]),
            dry=float(mass["dry"]),
            cg_dry=float(mass["cg_dry"]),
            motor_cg_loaded=float(mass["motor_cg_loaded"]),
        ),
        inertia=VehicleInertia(
            I_R_wet=float(inertia["I_R_wet"]),
            I_R_dry=float(inertia["I_R_dry"]),
            I_L_wet=float(inertia["I_L_wet"]),
            I_L_dry=float(inertia["I_L_dry"]),
        ),
        nozzle=VehicleNozzle(
            exit=float(nozzle["exit"]),
        ),
        recovery=VehicleRecovery(
            CdA_drogue=float(recovery["CdA_drogue"]),
            CdA_main=float(recovery["CdA_main"]),
            deploy_altitude_agl=float(recovery["deploy_altitude_agl"]),
        ),
        roll=VehicleRoll(
            r_fin=float(roll["r_fin"]),
        ),
    )


# ---------------------------------------------------------------------------
# Motor data dataclass  (raw parsed output of load_motor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorData:
    """Raw data parsed from a RASP .eng file.

    Masses are in kg, time in seconds, thrust in Newtons.
    ``m_motor_kg`` is the total motor mass (casing + propellant) as stated in
    the .eng header ("total weight" field).
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
