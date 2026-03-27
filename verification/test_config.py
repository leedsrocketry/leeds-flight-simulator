"""Tests for config.py — YAML loading and dataclass construction."""

import textwrap
from pathlib import Path

import pytest
import yaml

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
    launch_site:
      latitude: 58.61
      longitude: -4.94
    launch_rail:
      azimuth: 45.0
      inclination: 87.0
    mc:
      num_samples: 500
      master_seed: 99
    distributions:
      azimuth_sigma: 1.0
      inclination_sigma: 0.5
      fin_cant_sigma: 0.02
      impulse_factor_sigma: 6.7
    acceptance:
      compliance_threshold: 99.7
      buffer_distance: 1000
      altitude_ceiling: 16764
      sm_subsonic_min: 1.0
      sm_supersonic_min: 2.0
      aoa_max: 12.0
      sm_aoa_threshold: 5.0
    optimisation:
      min_safe_radius: 500
    observation_stations:
      - name: "Station A"
        latitude: 58.40
        longitude: -4.76
        radius: 10000
    map_markers:
      - name: "Pad"
        latitude: 58.61
        longitude: -4.94
    paths:
      vehicle: "vehicle.yaml"
      motor: "motor.eng"
      aero_dir: "aero_tables/"
      wind_profiles: "wind_profiles.npz"
      danger_area: "danger_area.geojson"
      coastline: "coastline.geojson"
    surface_override:
      speed_ms: 5.0
      bearing_deg: 270.0
      blend_height_m: 300
    """

_VEHICLE_YAML = """
    diameter: 0.130
    length: 2.6
    reference_area: 0.01327
    launch_rail_length: 4.0
    wet_mass: 26.5
    dry_mass: 14.7
    cg_dry: 1.15
    cg_propellant: 2.10
    I_R_wet: 0.012
    I_R_dry: 0.008
    I_L_wet: 5.2
    I_L_dry: 3.1
    nozzle_exit: 2.55
    CdA_drogue: 0.15
    CdA_main: 2.8
    deploy_altitude_agl: 305
    r_fin: 0.095
    """


# ---------------------------------------------------------------------------
# SimulationConfig tests
# ---------------------------------------------------------------------------

def test_simulation_config_loads(tmp_path):
    p = _write(tmp_path, "simulation.yaml", _MINIMAL_SIM_YAML)
    cfg = load_simulation_config(p)
    assert isinstance(cfg, SimulationConfig)


def test_launch_site_values(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.launch_site.latitude == pytest.approx(58.61)
    assert cfg.launch_site.longitude == pytest.approx(-4.94)


def test_launch_rail_numeric(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.launch_rail.azimuth == pytest.approx(45.0)
    assert cfg.launch_rail.inclination == pytest.approx(87.0)


def test_launch_rail_auto(tmp_path):
    content = _MINIMAL_SIM_YAML.replace("azimuth: 45.0", 'azimuth: "auto"')
    content = content.replace("inclination: 87.0", 'inclination: "auto"')
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", content))
    assert cfg.launch_rail.azimuth == "auto"
    assert cfg.launch_rail.inclination == "auto"


def test_mc_values(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.mc.num_samples == 500
    assert cfg.mc.master_seed == 99


def test_distributions(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.distributions.impulse_factor_sigma == pytest.approx(6.7)
    assert cfg.distributions.fin_cant_sigma == pytest.approx(0.02)


def test_acceptance(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.acceptance.compliance_threshold == pytest.approx(99.7)
    assert cfg.acceptance.sm_subsonic_min == pytest.approx(1.0)
    assert cfg.acceptance.sm_supersonic_min == pytest.approx(2.0)
    assert cfg.acceptance.aoa_max == pytest.approx(12.0)


def test_observation_stations(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert len(cfg.observation_stations) == 1
    assert cfg.observation_stations[0].name == "Station A"
    assert cfg.observation_stations[0].radius == pytest.approx(10000)


def test_map_markers(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert len(cfg.map_markers) == 1
    assert cfg.map_markers[0].name == "Pad"


def test_paths_resolved(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    # Paths should be absolute and resolve relative to the yaml file's directory
    assert cfg.paths.vehicle.is_absolute()
    assert cfg.paths.vehicle == (tmp_path / "vehicle.yaml").resolve()


def test_coastline_omitted(tmp_path):
    lines = [l for l in _MINIMAL_SIM_YAML.splitlines() if "coastline" not in l]
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", "\n".join(lines)))
    assert cfg.paths.coastline is None


def test_surface_override_values(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.surface_override.speed_ms == pytest.approx(5.0)
    assert cfg.surface_override.bearing_deg == pytest.approx(270.0)
    assert cfg.surface_override.blend_height_m == pytest.approx(300.0)


def test_surface_override_blend_omitted(tmp_path):
    lines = [l for l in _MINIMAL_SIM_YAML.splitlines() if "blend_height_m" not in l]
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", "\n".join(lines)))
    assert cfg.surface_override.blend_height_m is None


def test_config_is_frozen(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    with pytest.raises(Exception):
        cfg.mc.num_samples = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VehicleConfig tests
# ---------------------------------------------------------------------------

def test_vehicle_config_loads(tmp_path):
    p = _write(tmp_path, "vehicle.yaml", _VEHICLE_YAML)
    v = load_vehicle_config(p)
    assert isinstance(v, VehicleConfig)


def test_vehicle_geometry(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.diameter == pytest.approx(0.130)
    assert v.reference_area == pytest.approx(0.01327)
    assert v.launch_rail_length == pytest.approx(4.0)


def test_vehicle_mass(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.wet_mass == pytest.approx(26.5)
    assert v.dry_mass == pytest.approx(14.7)
    assert v.cg_dry == pytest.approx(1.15)
    assert v.cg_propellant == pytest.approx(2.10)


def test_vehicle_moi(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.I_R_wet == pytest.approx(0.012)
    assert v.I_L_dry == pytest.approx(3.1)


def test_vehicle_recovery(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.CdA_drogue == pytest.approx(0.15)
    assert v.CdA_main == pytest.approx(2.8)
    assert v.deploy_altitude_agl == pytest.approx(305.0)


def test_vehicle_roll(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.r_fin == pytest.approx(0.095)


def test_vehicle_is_frozen(tmp_path):
    v = load_vehicle_config(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    with pytest.raises(Exception):
        v.diameter = 0.2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Round-trip test against the real input/simulation.yaml
# ---------------------------------------------------------------------------

def test_real_simulation_yaml_loads():
    """The committed input/simulation.yaml must parse without errors."""
    real = Path(__file__).parent.parent / "input" / "simulation.yaml"
    if not real.exists():
        pytest.skip("input/simulation.yaml not present")
    cfg = load_simulation_config(real)
    assert cfg.launch_site.latitude == pytest.approx(58.61047)


def test_real_vehicle_yaml_loads():
    """The committed input/vehicle.yaml must parse without errors."""
    real = Path(__file__).parent.parent / "input" / "vehicle.yaml"
    if not real.exists():
        pytest.skip("input/vehicle.yaml not present")
    v = load_vehicle_config(real)
    assert v.diameter == pytest.approx(0.130)
