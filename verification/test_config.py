"""Tests for config.py — YAML loading and dataclass construction."""

import math
import textwrap
from pathlib import Path

import pytest

from config import (
    load_simulation_config,
    load_vehicle_config,
    SimulationConfig,
    VehicleConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


_MINIMAL_SIM_YAML = """
    vehicle:
      config: "vehicle.yaml"
      motor: "motor.eng"
      aero_tables: "aero_tables"
    site:
      latitude: 58.61
      longitude: -4.94
      min_safe_radius: 500
      altitude_ceiling: 16764
      danger_area: "danger_area.geojson"
      observation_stations:
        - name: "Station A"
          latitude: 58.40
          longitude: -4.76
          radius: 10000
      map_markers:
        - name: "Pad"
          latitude: 58.61
          longitude: -4.94
    launch:
      rail:
        azimuth: 45.0
        inclination: 87.0
        length: 4.0
      wind_profiles: "wind_profiles.npz"
      surface_wind:
        speed_ms: 5.0
        bearing_deg: 270.0
        blend_height_m: 300
    monte_carlo:
      samples: 500
      seed: 99
      uncertainties:
        azimuth_sigma: 1.0
        inclination_sigma: 0.5
        fin_cant_sigma: 0.02
        impulse_factor_sigma: 0.067
      acceptance:
        compliance_threshold: 0.997
        buffer_distance: 1000
        sm_transition_mach: 0.91
        sm_subsonic_min: 1.0
        sm_supersonic_min: 2.0
        aoa_max: 12.0
        sm_aoa_threshold: 5.0
    """

_VEHICLE_YAML = """
    geometry:
      diameter: 0.130
      length: 2.6
      nozzle_position: 2.55
      nozzle_diameter: 0.08
      fin_cp_radius: 0.095
    mass:
      wet_mass: 26.5
      wet_cg: 1.15
      wet_motor_cg: 1.82
      propellant_I_roll: 0.05
      propellant_I_lateral: 0.8
      wet_inertia_lateral: 5.2
      wet_inertia_roll: 0.012
    recovery:
      drogue_cd: 2.0
      drogue_area: 0.15
      drogue_threshold: apogee
      main_cd: 2.0
      main_area: 2.8
      main_threshold: 305
    """


# ---------------------------------------------------------------------------
# SimulationConfig tests
# ---------------------------------------------------------------------------

def test_simulation_config_loads(tmp_path):
    p = _write(tmp_path, "simulation.yaml", _MINIMAL_SIM_YAML)
    cfg = load_simulation_config(p)
    assert isinstance(cfg, SimulationConfig)


def test_vehicle_files(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.vehicle.config.is_absolute()
    assert cfg.vehicle.config == (tmp_path / "vehicle.yaml").resolve()
    assert cfg.vehicle.aero_tables == (tmp_path / "aero_tables").resolve()


def test_site_values(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.site.latitude == pytest.approx(58.61)
    assert cfg.site.longitude == pytest.approx(-4.94)
    assert cfg.site.min_safe_radius == pytest.approx(500.0)
    assert cfg.site.altitude_ceiling == pytest.approx(16764.0)


def test_site_danger_area_resolved(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.site.danger_area == (tmp_path / "danger_area.geojson").resolve()


def test_coastline_omitted(tmp_path):
    lines = [l for l in _MINIMAL_SIM_YAML.splitlines() if "coastline" not in l]
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", "\n".join(lines)))
    assert cfg.site.coastline is None


def test_launch_rail_numeric(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.launch.rail.azimuth == pytest.approx(45.0)
    assert cfg.launch.rail.inclination == pytest.approx(87.0)
    assert cfg.launch.rail.length == pytest.approx(4.0)


def test_launch_rail_auto(tmp_path):
    content = _MINIMAL_SIM_YAML.replace("azimuth: 45.0", 'azimuth: "auto"')
    content = content.replace("inclination: 87.0", 'inclination: "auto"')
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", content))
    assert cfg.launch.rail.azimuth == "auto"
    assert cfg.launch.rail.inclination == "auto"


def test_launch_wind_profiles_resolved(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.launch.wind_profiles == (tmp_path / "wind_profiles.npz").resolve()


def test_monte_carlo_values(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.monte_carlo.samples == 500
    assert cfg.monte_carlo.seed == 99


def test_uncertainties(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    unc = cfg.monte_carlo.uncertainties
    assert unc.impulse_factor_sigma == pytest.approx(0.067)
    assert unc.fin_cant_sigma == pytest.approx(0.02)


def test_acceptance(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    acc = cfg.monte_carlo.acceptance
    assert acc.compliance_threshold == pytest.approx(0.997)
    assert acc.sm_transition_mach == pytest.approx(0.91)
    assert acc.sm_subsonic_min == pytest.approx(1.0)
    assert acc.sm_supersonic_min == pytest.approx(2.0)
    assert acc.aoa_max == pytest.approx(12.0)


def test_observation_stations(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert len(cfg.site.observation_stations) == 1
    assert cfg.site.observation_stations[0].name == "Station A"
    assert cfg.site.observation_stations[0].radius == pytest.approx(10000)


def test_map_markers(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert len(cfg.site.map_markers) == 1
    assert cfg.site.map_markers[0].name == "Pad"


def test_surface_wind_values(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    sw = cfg.launch.surface_wind
    assert sw is not None
    assert sw.speed_ms == pytest.approx(5.0)
    assert sw.bearing_deg == pytest.approx(270.0)
    assert sw.blend_height_m == pytest.approx(300.0)


def test_surface_wind_omitted(tmp_path):
    """Omitting the surface_wind section gives surface_wind=None."""
    lines = [l for l in _MINIMAL_SIM_YAML.splitlines()
             if not any(k in l for k in ("surface_wind", "speed_ms",
                                         "bearing_deg", "blend_height_m"))]
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", "\n".join(lines)))
    assert cfg.launch.surface_wind is None


def test_config_is_frozen(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    with pytest.raises(Exception):
        cfg.monte_carlo.samples = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VehicleConfig tests
# ---------------------------------------------------------------------------

def test_vehicle_config_loads(tmp_path):
    p = _write(tmp_path, "vehicle.yaml", _VEHICLE_YAML)
    v = load_vehicle_config(p)
    assert isinstance(v, VehicleConfig)


def test_vehicle_geometry(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.geometry.diameter == pytest.approx(0.130)
    assert v.geometry.length == pytest.approx(2.6)
    assert v.geometry.nozzle_position == pytest.approx(2.55)
    assert v.geometry.nozzle_diameter == pytest.approx(0.08)
    assert v.geometry.fin_cp_radius == pytest.approx(0.095)


def test_reference_area_computed(tmp_path):
    """reference_area is π·d²/4 — derived from diameter, not read from YAML."""
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    expected = math.pi * 0.130 ** 2 / 4.0
    assert v.reference_area == pytest.approx(expected, rel=1e-6)
    assert v.geometry.reference_area == pytest.approx(expected, rel=1e-6)


def test_nozzle_area_computed(tmp_path):
    """nozzle_area is π·dₑ²/4 — derived from nozzle_diameter."""
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    expected = math.pi * 0.08 ** 2 / 4.0
    assert v.geometry.nozzle_area == pytest.approx(expected, rel=1e-6)


def test_vehicle_mass(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.mass.wet_mass == pytest.approx(26.5)
    assert v.mass.wet_cg == pytest.approx(1.15)
    assert v.mass.wet_motor_cg == pytest.approx(1.82)


def test_vehicle_inertia(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.mass.wet_inertia_roll == pytest.approx(0.012)
    assert v.mass.wet_inertia_lateral == pytest.approx(5.2)
    assert v.mass.propellant_I_roll == pytest.approx(0.05)
    assert v.mass.propellant_I_lateral == pytest.approx(0.8)


def test_vehicle_recovery(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.recovery.drogue_cd == pytest.approx(2.0)
    assert v.recovery.drogue_area == pytest.approx(0.15)
    assert v.recovery.drogue_threshold == "apogee"
    assert v.recovery.main_cd == pytest.approx(2.0)
    assert v.recovery.main_area == pytest.approx(2.8)
    assert v.recovery.main_threshold == pytest.approx(305.0)


def test_recovery_numeric_threshold(tmp_path):
    """main_threshold=305 parses as a float; drogue_threshold='apogee' as literal."""
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert isinstance(v.recovery.main_threshold, float)
    assert v.recovery.drogue_threshold == "apogee"


def test_vehicle_is_frozen(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    with pytest.raises(Exception):
        v.geometry.diameter = 0.2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Round-trip test against the real input files
# ---------------------------------------------------------------------------

def test_real_simulation_yaml_loads():
    """The committed input/simulation.yaml must parse without errors."""
    real = Path(__file__).parent.parent / "input" / "simulation.yaml"
    if not real.exists():
        pytest.skip("input/simulation.yaml not present")
    cfg = load_simulation_config(real)
    assert cfg.site.latitude == pytest.approx(58.61047)


def test_real_vehicle_yaml_loads():
    """The committed input/vehicle.yaml must parse without errors."""
    real = Path(__file__).parent.parent / "input" / "vehicle.yaml"
    if not real.exists():
        pytest.skip("input/vehicle.yaml not present")
    v = load_vehicle_config(real)
    assert v.geometry.diameter == pytest.approx(0.130)
