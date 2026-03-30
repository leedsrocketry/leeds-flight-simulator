"""Configuration loader: YAML files → dataclasses.

Public functions:
    load_simulation_config(path)                    →  SimulationConfig
    load_vehicle(path, propellant)                  →  Vehicle

Paths inside SimulationConfig are resolved relative to the simulation.yaml
file itself; paths inside Vehicle are resolved relative to vehicle.yaml.

The ``Vehicle`` dataclass holds both user-specified wet properties and
derived dry properties.  Dry properties are computed once during loading
by subtracting the propellant contribution (from the ``PropellantModel``)
from the wet values.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from motor import PropellantModel, MotorData, load_motor, build_propellant_model


# ---------------------------------------------------------------------------
# Shared geographic sub-types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitourStation:
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
    launch_monitour_radius: float               # metres — radius of the automatic monitour
                                                   # station at the launch site
    altitude_ceiling: float                        # metres
    danger_area: Path
    coastline: Path | None                         # None → sea-landing check disabled
    coastline_mode: str                            # "sea" or "land" (§14.2)
    monitour_stations: tuple[MonitourStation, ...]
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
    monitour_check_scenarios: tuple[str, ...]  # scenarios checked for monitour station coverage


@dataclass(frozen=True)
class MonteCarloConfig:
    samples: int
    seed: int
    uncertainties: UncertaintiesConfig
    acceptance: AcceptanceConfig


@dataclass(frozen=True)
class VerificationConfig:
    reference_trajectory: Path  # resolved path to reference CSV
    altitude_tolerance: float   # fractional tolerance on altitude
    mach_tolerance: float       # fractional tolerance on Mach number
    sm_tolerance: float         # fractional tolerance on static margin
    mass_tolerance: float       # fractional tolerance on vehicle mass
    thrust_tolerance: float     # fractional tolerance on thrust
    exceedance_fraction: float  # fraction of points allowed outside tolerance (0 = strict)
    azimuth: float | None       # degrees — override launch.rail.azimuth for verification
    inclination: float | None   # degrees — override launch.rail.inclination for verification


@dataclass(frozen=True)
class SimulationConfig:
    vehicle: Path           # resolved path to vehicle.yaml
    site: SiteConfig
    launch: LaunchConfig
    monte_carlo: MonteCarloConfig
    verification: VerificationConfig | None  # None → trajectory comparison skipped


# ---------------------------------------------------------------------------
# Vehicle config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleGeometry:
    diameter: float         # m — reference diameter
    length: float           # m — total length
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

    @property
    def nozzle_position(self) -> float:
        """Nozzle exit plane [m] — assumed flush with the aft end."""
        return self.length


@dataclass(frozen=True)
class ParachuteConfig:
    """Configuration for a single parachute stage."""
    cd: float                                  # drag coefficient
    diameter: float                            # m — reference diameter
    threshold: float | Literal["apogee"]       # deploy altitude [m AGL] or "apogee"

    @property
    def area(self) -> float:
        """π·d²/4 [m²] — reference area derived from diameter."""
        return math.pi * self.diameter ** 2 / 4.0


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
class Vehicle:
    """Complete vehicle description: geometry, mass (wet + dry), recovery, paths.

    Dry mass properties are derived from the wet values and the propellant
    model during construction (see :func:`load_vehicle`).
    """
    geometry: VehicleGeometry
    recovery: VehicleRecovery
    motor: Path               # resolved path to .eng file
    aero_tables: Path         # resolved path to aero tables directory
    fins_aero_table: Path | None  # resolved path to fins component CSV, or None to
                                  # use the filename heuristic in aerodynamics.py

    # Wet mass properties (user-specified)
    wet_mass: float              # kg — total mass at launch
    wet_cg: float                # m from nosecone tip — wet vehicle CG
    wet_inertia_lateral: float   # kg·m² — wet vehicle lateral inertia about wet CG
    wet_inertia_roll: float      # kg·m² — wet vehicle roll inertia about roll axis

    # Derived dry properties (computed once from wet − propellant)
    m_dry: float                 # kg
    cg_dry: float                # m from nosecone tip
    I_roll_dry: float            # kg·m² — roll inertia about roll axis
    I_lateral_dry: float         # kg·m² — lateral inertia about dry CG

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
    stations_raw = site_raw.get("monitour_stations") or []
    markers_raw = site_raw.get("map_markers") or []
    coastline_raw = site_raw.get("coastline")
    coastline: Path | None = None if coastline_raw is None else _resolve(coastline_raw)
    coastline_mode_raw = str(site_raw.get("coastline_mode", "sea"))
    if coastline_mode_raw not in ("sea", "land"):
        raise ValueError(
            f"coastline_mode must be 'sea' or 'land', got: {coastline_mode_raw!r}"
        )

    site = SiteConfig(
        latitude=float(site_raw["latitude"]),
        longitude=float(site_raw["longitude"]),
        ballistic_exclusion_radius=float(site_raw["ballistic_exclusion_radius"]),
        launch_monitour_radius=float(site_raw["launch_monitour_radius"]),
        altitude_ceiling=float(site_raw["altitude_ceiling"]),
        danger_area=_resolve(site_raw["danger_area"]),
        coastline=coastline,
        coastline_mode=coastline_mode_raw,
        monitour_stations=tuple(
            MonitourStation(
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

    _wp = _resolve(launch_raw["wind_profiles"])
    if not _wp.exists():
        raise ValueError(f"wind_profiles path does not exist: {_wp}")
    if _wp.is_file() and _wp.suffix.lower() != ".npz":
        raise ValueError(
            f"wind_profiles must be a .npz file or a directory of .npz files, "
            f"got: {_wp}"
        )

    launch = LaunchConfig(
        rail=RailConfig(
            azimuth=rail_azimuth,
            azimuth_range=azimuth_range,
            inclination=rail_inclination,
            inclination_range=inclination_range,
            length=float(rail_raw["length"]),
        ),
        wind_profiles=_wp,
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
            monitour_check_scenarios=tuple(
                str(s) for s in (acc_raw.get("monitour_check_scenarios") or [])
            ),
        ),
    )

    # -- verification (optional)
    ver_raw = raw.get("verification")
    verification: VerificationConfig | None
    if ver_raw is not None:
        verification = VerificationConfig(
            reference_trajectory=_resolve(str(ver_raw["reference_trajectory"])),
            altitude_tolerance=float(ver_raw["altitude_tolerance"]),
            mach_tolerance=float(ver_raw["mach_tolerance"]),
            sm_tolerance=float(ver_raw["sm_tolerance"]),
            mass_tolerance=float(ver_raw["mass_tolerance"]),
            thrust_tolerance=float(ver_raw.get("thrust_tolerance", ver_raw["altitude_tolerance"])),
            exceedance_fraction=float(ver_raw.get("exceedance_fraction", 0.0)),
            azimuth=float(ver_raw["azimuth"]) if "azimuth" in ver_raw else None,
            inclination=float(ver_raw["inclination"]) if "inclination" in ver_raw else None,
        )
    else:
        verification = None

    return SimulationConfig(
        vehicle=vehicle,
        site=site,
        launch=launch,
        monte_carlo=monte_carlo,
        verification=verification,
    )


def _derive_dry_properties(
    propellant: PropellantModel,
    wet_mass: float,
    wet_cg: float,
    wet_inertia_roll: float,
    wet_inertia_lateral: float,
) -> tuple[float, float, float, float]:
    """Derive dry vehicle properties by subtracting propellant from wet.

    Returns ``(m_dry, cg_dry, I_roll_dry, I_lateral_dry)``.
    """
    m_prop_0 = propellant.m_prop_0
    motor_cg = propellant.motor_cg_loaded

    # --- dry mass
    m_dry = wet_mass - m_prop_0
    if m_dry <= 0:
        raise ValueError(
            f"Computed dry mass {m_dry:.3f} kg must be > 0 "
            f"(wet_mass={wet_mass} kg, m_prop={m_prop_0} kg)"
        )

    # --- dry CG
    cg_dry = (wet_mass * wet_cg - m_prop_0 * motor_cg) / m_dry

    # --- dry roll inertia (no PAT needed on symmetry axis)
    I_roll_dry = wet_inertia_roll - propellant.I_roll_prop_0

    # --- dry lateral inertia
    # Step 1: transfer propellant inertia from propellant CG → wet vehicle CG
    d_prop_wet = motor_cg - wet_cg
    I_prop_lat_at_wet_cg = propellant.I_lat_prop_0 + m_prop_0 * d_prop_wet ** 2
    # Step 2: dry lateral inertia about wet vehicle CG
    I_lat_dry_at_wet_cg = wet_inertia_lateral - I_prop_lat_at_wet_cg
    # Step 3: transfer to dry vehicle CG
    d_dry_wet = cg_dry - wet_cg
    I_lateral_dry = I_lat_dry_at_wet_cg - m_dry * d_dry_wet ** 2

    return m_dry, cg_dry, I_roll_dry, I_lateral_dry


def load_vehicle(path: Path | str) -> tuple[Vehicle, PropellantModel]:
    """Parse *path* (vehicle.yaml), load the motor, and return Vehicle + PropellantModel.

    All file paths are resolved relative to the directory containing the
    vehicle.yaml file.  The motor .eng file is loaded and a propellant model
    built automatically.  Dry mass properties are derived from the wet values
    and the propellant model.

    Returns
    -------
    (vehicle, propellant)
    """
    path = Path(path).resolve()
    base_dir = path.parent

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    def _resolve(p: str) -> Path:
        return (base_dir / p).resolve()

    geom_raw = raw["geometry"]
    mass = raw["mass"]
    recovery_raw = raw.get("recovery") or {}

    def _parse_chute(key: str) -> ParachuteConfig | None:
        r = recovery_raw.get(key)
        if r is None:
            return None
        return ParachuteConfig(
            cd=float(r["cd"]),
            diameter=float(r["diameter"]),
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

    geometry = VehicleGeometry(
        diameter=float(geom_raw["diameter"]),
        length=float(geom_raw["length"]),
        nozzle_diameter=float(geom_raw["nozzle_diameter"]),
        fin_cp_radius=float(geom_raw["fin_cp_radius"]),
    )

    # --- Load motor and build propellant model ---
    motor_path = _resolve(raw["motor"])
    motor_data = load_motor(motor_path)

    prop_inner_raw = mass.get("propellant_inner_diameter")
    prop_outer_raw = mass.get("propellant_outer_diameter")
    if prop_inner_raw is None:
        warnings.warn(
            "No propellant_inner_diameter specified; assuming solid cylinder. "
            "Propellant roll inertia will be underestimated for hollow grains."
        )
    if prop_outer_raw is None:
        warnings.warn(
            "No propellant_outer_diameter specified; assuming propellant fills "
            "the full motor diameter (no casing, liner, or insulator)."
        )

    propellant = build_propellant_model(
        motor_data,
        vehicle_length=geometry.length,
        nozzle_area=geometry.nozzle_area,
        propellant_outer_diameter=float(prop_outer_raw) if prop_outer_raw is not None else None,
        propellant_inner_diameter=float(prop_inner_raw) if prop_inner_raw is not None else None,
    )

    # --- Derive dry properties ---
    wet_mass_val = float(mass["wet_mass"])
    wet_cg_val = float(mass["wet_cg"])
    wet_inertia_lateral_val = float(mass["wet_inertia_lateral"])
    wet_inertia_roll_val = float(mass["wet_inertia_roll"])

    m_dry, cg_dry, I_roll_dry, I_lateral_dry = _derive_dry_properties(
        propellant,
        wet_mass_val,
        wet_cg_val,
        wet_inertia_roll_val,
        wet_inertia_lateral_val,
    )

    vehicle = Vehicle(
        geometry=geometry,
        recovery=VehicleRecovery(
            drogue=drogue,
            main=main,
        ),
        motor=motor_path,
        aero_tables=_resolve(raw["aero_tables"]),
        fins_aero_table=fins_aero_table,
        wet_mass=wet_mass_val,
        wet_cg=wet_cg_val,
        wet_inertia_lateral=wet_inertia_lateral_val,
        wet_inertia_roll=wet_inertia_roll_val,
        m_dry=m_dry,
        cg_dry=cg_dry,
        I_roll_dry=I_roll_dry,
        I_lateral_dry=I_lateral_dry,
    )

    return vehicle, propellant
