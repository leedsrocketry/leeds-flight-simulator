"""Configuration loader: YAML files → dataclasses.

Two public functions:
    load_simulation_config(path)  →  SimulationConfig
    load_vehicle_config(path)     →  VehicleConfig

All paths inside SimulationConfig are resolved relative to the simulation.yaml
file itself, so the caller can pass an absolute or relative path and the
referenced input files will be found correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


# ---------------------------------------------------------------------------
# Simulation config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaunchSiteConfig:
    latitude: float   # degrees
    longitude: float  # degrees


@dataclass(frozen=True)
class LaunchRailConfig:
    azimuth: float | Literal["auto"]      # degrees clockwise from North
    inclination: float | Literal["auto"]  # degrees from horizontal


@dataclass(frozen=True)
class MCConfig:
    num_samples: int
    master_seed: int


@dataclass(frozen=True)
class DistributionsConfig:
    azimuth_sigma: float      # degrees
    inclination_sigma: float  # degrees
    fin_cant_sigma: float     # degrees
    impulse_factor_sigma: float  # percent


@dataclass(frozen=True)
class AcceptanceConfig:
    compliance_threshold: float  # percent
    buffer_distance: float       # metres inward
    altitude_ceiling: float      # metres
    sm_subsonic_min: float       # calibres (M < 0.91)
    sm_supersonic_min: float     # calibres (M >= 0.91)
    aoa_max: float               # degrees
    sm_aoa_threshold: float      # degrees


@dataclass(frozen=True)
class OptimisationConfig:
    min_safe_radius: float  # metres


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


@dataclass(frozen=True)
class PathsConfig:
    vehicle: Path
    motor: Path
    aero_dir: Path
    wind_profiles: Path
    danger_area: Path
    coastline: Path | None  # None → sea-landing check disabled


@dataclass(frozen=True)
class SurfaceOverrideConfig:
    speed_ms: float    # m/s
    bearing_deg: float  # degrees clockwise from North
    blend_height_m: float | None  # metres AGL; None → override disabled


@dataclass(frozen=True)
class SimulationConfig:
    launch_site: LaunchSiteConfig
    launch_rail: LaunchRailConfig
    mc: MCConfig
    distributions: DistributionsConfig
    acceptance: AcceptanceConfig
    optimisation: OptimisationConfig
    observation_stations: tuple[ObservationStation, ...]
    map_markers: tuple[MapMarker, ...]
    paths: PathsConfig
    surface_override: SurfaceOverrideConfig


# ---------------------------------------------------------------------------
# Vehicle config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleConfig:
    # Geometry
    diameter: float          # m
    length: float            # m
    reference_area: float    # m²
    launch_rail_length: float  # m

    # Mass
    wet_mass: float          # kg
    dry_mass: float          # kg
    cg_dry: float            # m from nosecone tip
    cg_propellant: float     # m from nosecone tip

    # Moments of inertia
    I_R_wet: float           # kg·m²  roll, wet
    I_R_dry: float           # kg·m²  roll, dry
    I_L_wet: float           # kg·m²  lateral, wet
    I_L_dry: float           # kg·m²  lateral, dry

    # Nozzle
    nozzle_exit: float       # m from nosecone tip

    # Recovery
    CdA_drogue: float        # m²
    CdA_main: float          # m²
    deploy_altitude_agl: float  # m above launch site

    # Roll model
    r_fin: float             # m — fin CP spanwise distance from centreline


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

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

    ls = raw["launch_site"]
    lr = raw["launch_rail"]
    mc = raw["mc"]
    dist = raw["distributions"]
    acc = raw["acceptance"]
    opt = raw.get("optimisation", {})
    stations_raw = raw.get("observation_stations") or []
    markers_raw = raw.get("map_markers") or []
    paths_raw = raw["paths"]
    so = raw.get("surface_override", {})

    def _parse_auto(value: object) -> float | Literal["auto"]:
        if value == "auto":
            return "auto"
        return float(value)

    coastline_raw = paths_raw.get("coastline")
    coastline: Path | None = (
        None if coastline_raw is None else _resolve(coastline_raw)
    )

    blend_raw = so.get("blend_height_m")
    blend: float | None = (
        None if blend_raw is None else float(blend_raw)
    )

    return SimulationConfig(
        launch_site=LaunchSiteConfig(
            latitude=float(ls["latitude"]),
            longitude=float(ls["longitude"]),
        ),
        launch_rail=LaunchRailConfig(
            azimuth=_parse_auto(lr["azimuth"]),
            inclination=_parse_auto(lr["inclination"]),
        ),
        mc=MCConfig(
            num_samples=int(mc["num_samples"]),
            master_seed=int(mc["master_seed"]),
        ),
        distributions=DistributionsConfig(
            azimuth_sigma=float(dist["azimuth_sigma"]),
            inclination_sigma=float(dist["inclination_sigma"]),
            fin_cant_sigma=float(dist["fin_cant_sigma"]),
            impulse_factor_sigma=float(dist["impulse_factor_sigma"]),
        ),
        acceptance=AcceptanceConfig(
            compliance_threshold=float(acc["compliance_threshold"]),
            buffer_distance=float(acc["buffer_distance"]),
            altitude_ceiling=float(acc["altitude_ceiling"]),
            sm_subsonic_min=float(acc["sm_subsonic_min"]),
            sm_supersonic_min=float(acc["sm_supersonic_min"]),
            aoa_max=float(acc["aoa_max"]),
            sm_aoa_threshold=float(acc["sm_aoa_threshold"]),
        ),
        optimisation=OptimisationConfig(
            min_safe_radius=float(opt.get("min_safe_radius", 0.0)),
        ),
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
        paths=PathsConfig(
            vehicle=_resolve(paths_raw["vehicle"]),
            motor=_resolve(paths_raw["motor"]),
            aero_dir=_resolve(paths_raw["aero_dir"]),
            wind_profiles=_resolve(paths_raw["wind_profiles"]),
            danger_area=_resolve(paths_raw["danger_area"]),
            coastline=coastline,
        ),
        surface_override=SurfaceOverrideConfig(
            speed_ms=float(so.get("speed_ms", 0.0)),
            bearing_deg=float(so.get("bearing_deg", 0.0)),
            blend_height_m=blend,
        ),
    )


def load_vehicle_config(path: Path | str) -> VehicleConfig:
    """Parse *path* (vehicle.yaml) and return a VehicleConfig."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return VehicleConfig(
        diameter=float(raw["diameter"]),
        length=float(raw["length"]),
        reference_area=float(raw["reference_area"]),
        launch_rail_length=float(raw["launch_rail_length"]),
        wet_mass=float(raw["wet_mass"]),
        dry_mass=float(raw["dry_mass"]),
        cg_dry=float(raw["cg_dry"]),
        cg_propellant=float(raw["cg_propellant"]),
        I_R_wet=float(raw["I_R_wet"]),
        I_R_dry=float(raw["I_R_dry"]),
        I_L_wet=float(raw["I_L_wet"]),
        I_L_dry=float(raw["I_L_dry"]),
        nozzle_exit=float(raw["nozzle_exit"]),
        CdA_drogue=float(raw["CdA_drogue"]),
        CdA_main=float(raw["CdA_main"]),
        deploy_altitude_agl=float(raw["deploy_altitude_agl"]),
        r_fin=float(raw["r_fin"]),
    )
