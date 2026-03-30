"""Tests for config.py — YAML loading and dataclass construction."""

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from config import (
    load_simulation_config,
    load_vehicle,
    SimulationConfig,
    Vehicle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STUB_ENG = """\
; Stub motor for config tests — small propellant mass to stay consistent
; with _VEHICLE_YAML wet values (26.5 kg, 5.2 kg·m² lateral inertia)
M100 54 200 0 0.5 1.0 TestMfr
0.0    0.0
0.1   200.0
1.0   200.0
2.0     0.0
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    # Create a stub wind_profiles.npz so wind_profiles path validation passes
    stub = tmp_path / "wind_profiles.npz"
    if not stub.exists():
        np.savez(stub, altitude_m=np.array([0.0]), wind_east_ms=np.zeros((1, 1)),
                 wind_north_ms=np.zeros((1, 1)))
    # Create a stub motor.eng so load_vehicle can parse it
    eng = tmp_path / "motor.eng"
    if not eng.exists():
        eng.write_text(_STUB_ENG, encoding="utf-8")
    return p


_MINIMAL_SIM_YAML = """
    vehicle: "vehicle.yaml"
    site:
      latitude: 58.61
      longitude: -4.94
      ballistic_exclusion_radius: 500
      launch_monitor_radius: 200
      altitude_ceiling: 16764
      danger_area: "danger_area.geojson"
      monitor_stations:
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
        coastline_check_scenarios:
          - nominal
          - ballistic
        monitor_check_scenarios:
          - ballistic
          - drogue_only
    """

_VEHICLE_YAML = """
    motor: "motor.eng"
    aero_tables: "aero_tables"
    geometry:
      diameter: 0.130
      length: 2.6
      nozzle_diameter: 0.08
      fin_cp_radius: 0.095
    mass:
      wet_mass: 26.5
      wet_cg: 1.15
      wet_inertia_lateral: 5.2
      wet_inertia_roll: 0.012
      propellant_inner_diameter: 0.030
      propellant_outer_diameter: 0.050
    recovery:
      drogue:
        cd: 2.0
        diameter: 0.437019
        threshold: apogee
      main:
        cd: 2.0
        diameter: 1.888139
        threshold: 305
    """


# ---------------------------------------------------------------------------
# SimulationConfig tests
# ---------------------------------------------------------------------------

def test_simulation_config_loads(tmp_path):
    p = _write(tmp_path, "simulation.yaml", _MINIMAL_SIM_YAML)
    cfg = load_simulation_config(p)
    assert isinstance(cfg, SimulationConfig)


def test_vehicle_path(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.vehicle.is_absolute()
    assert cfg.vehicle == (tmp_path / "vehicle.yaml").resolve()


def test_site_values(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.site.latitude == pytest.approx(58.61)
    assert cfg.site.longitude == pytest.approx(-4.94)
    assert cfg.site.ballistic_exclusion_radius == pytest.approx(500.0)
    assert cfg.site.launch_monitor_radius == pytest.approx(200.0)
    assert cfg.site.altitude_ceiling == pytest.approx(16764.0)


def test_site_danger_area_resolved(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.site.danger_area == (tmp_path / "danger_area.geojson").resolve()


def test_coastline_omitted(tmp_path):
    lines = [l for l in _MINIMAL_SIM_YAML.splitlines()
             if "coastline:" not in l and "coastline_mode" not in l]
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", "\n".join(lines)))
    assert cfg.site.coastline is None


def test_launch_rail_numeric(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert cfg.launch.rail.azimuth == pytest.approx(45.0)
    assert cfg.launch.rail.inclination == pytest.approx(87.0)
    assert cfg.launch.rail.length == pytest.approx(4.0)
    assert cfg.launch.rail.azimuth_range is None
    assert cfg.launch.rail.inclination_range is None


def test_launch_rail_auto(tmp_path):
    content = _MINIMAL_SIM_YAML.replace("azimuth: 45.0", 'azimuth: "auto"')
    content = content.replace("inclination: 87.0", 'inclination: "auto"')
    content = content.replace(
        "        length: 4.0",
        "        azimuth_range: [-90, 90]\n        inclination_range: [85, 90]\n        length: 4.0",
    )
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", content))
    assert cfg.launch.rail.azimuth == "auto"
    assert cfg.launch.rail.inclination == "auto"
    assert cfg.launch.rail.azimuth_range == pytest.approx((-90.0, 90.0))
    assert cfg.launch.rail.inclination_range == pytest.approx((85.0, 90.0))


def test_launch_rail_auto_requires_azimuth_range(tmp_path):
    """azimuth='auto' without azimuth_range must raise ValueError."""
    content = _MINIMAL_SIM_YAML.replace("azimuth: 45.0", 'azimuth: "auto"')
    with pytest.raises(ValueError, match="azimuth_range"):
        load_simulation_config(_write(tmp_path, "s.yaml", content))


def test_launch_rail_auto_requires_inclination_range(tmp_path):
    """inclination='auto' without inclination_range must raise ValueError."""
    content = _MINIMAL_SIM_YAML.replace("inclination: 87.0", 'inclination: "auto"')
    with pytest.raises(ValueError, match="inclination_range"):
        load_simulation_config(_write(tmp_path, "s.yaml", content))


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


def test_monitor_stations(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    assert len(cfg.site.monitor_stations) == 1
    assert cfg.site.monitor_stations[0].name == "Station A"
    assert cfg.site.monitor_stations[0].radius == pytest.approx(10000)


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


def test_acceptance_scenario_lists(tmp_path):
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", _MINIMAL_SIM_YAML))
    acc = cfg.monte_carlo.acceptance
    assert "nominal" in acc.coastline_check_scenarios
    assert "ballistic" in acc.coastline_check_scenarios
    assert "ballistic" in acc.monitor_check_scenarios
    assert "drogue_only" in acc.monitor_check_scenarios


def test_acceptance_scenario_lists_empty_when_omitted(tmp_path):
    """Omitting scenario lists gives empty tuples — no check runs."""
    lines = [l for l in _MINIMAL_SIM_YAML.splitlines()
             if not any(k in l for k in ("sea_check", "monitor_check", "- nominal",
                                         "- ballistic", "- drogue_only"))]
    cfg = load_simulation_config(_write(tmp_path, "s.yaml", "\n".join(lines)))
    assert cfg.monte_carlo.acceptance.coastline_check_scenarios == ()
    assert cfg.monte_carlo.acceptance.monitor_check_scenarios == ()


# ---------------------------------------------------------------------------
# Vehicle tests
# ---------------------------------------------------------------------------

def test_vehicle_config_loads(tmp_path):
    p = _write(tmp_path, "vehicle.yaml", _VEHICLE_YAML)
    v, _ = load_vehicle(p)
    assert isinstance(v, Vehicle)


def test_vehicle_file_paths(tmp_path):
    """motor and aero_tables are resolved to absolute paths."""
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.motor.is_absolute()
    assert v.motor == (tmp_path / "motor.eng").resolve()
    assert v.aero_tables.is_absolute()
    assert v.aero_tables == (tmp_path / "aero_tables").resolve()


def test_fins_aero_table_absent(tmp_path):
    """fins_aero_table defaults to None when not specified."""
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.fins_aero_table is None


def test_fins_aero_table_specified(tmp_path):
    """fins_aero_table is resolved to an absolute path when specified."""
    yaml_with_fins = _VEHICLE_YAML.replace(
        'motor: "motor.eng"',
        'motor: "motor.eng"\n    fins_aero_table: "aero_tables/fins.csv"',
    )
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", yaml_with_fins))
    assert v.fins_aero_table is not None
    assert v.fins_aero_table.is_absolute()
    assert v.fins_aero_table == (tmp_path / "aero_tables" / "fins.csv").resolve()


def test_vehicle_geometry(tmp_path):
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.geometry.diameter == pytest.approx(0.130)
    assert v.geometry.length == pytest.approx(2.6)
    assert v.geometry.nozzle_position == pytest.approx(2.6)  # = length (flush aft)
    assert v.geometry.nozzle_diameter == pytest.approx(0.08)
    assert v.geometry.fin_cp_radius == pytest.approx(0.095)


def test_reference_area_computed(tmp_path):
    """reference_area is π·d²/4 — derived from diameter, not read from YAML."""
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    expected = math.pi * 0.130 ** 2 / 4.0
    assert v.reference_area == pytest.approx(expected, rel=1e-6)
    assert v.geometry.reference_area == pytest.approx(expected, rel=1e-6)


def test_nozzle_area_computed(tmp_path):
    """nozzle_area is π·dₑ²/4 — derived from nozzle_diameter."""
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    expected = math.pi * 0.08 ** 2 / 4.0
    assert v.geometry.nozzle_area == pytest.approx(expected, rel=1e-6)


def test_vehicle_mass(tmp_path):
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.wet_mass == pytest.approx(26.5)
    assert v.wet_cg == pytest.approx(1.15)
    assert v.m_dry > 0.0
    assert v.cg_dry > 0.0


def test_vehicle_inertia(tmp_path):
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.wet_inertia_roll == pytest.approx(0.012)
    assert v.wet_inertia_lateral == pytest.approx(5.2)
    assert v.I_roll_dry > 0.0
    assert v.I_lateral_dry > 0.0


def test_vehicle_recovery(tmp_path):
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert v.recovery.drogue is not None
    assert v.recovery.drogue.cd == pytest.approx(2.0)
    assert v.recovery.drogue.diameter == pytest.approx(0.437019)
    assert v.recovery.drogue.area == pytest.approx(0.15, rel=1e-4)
    assert v.recovery.drogue.threshold == "apogee"
    assert v.recovery.main is not None
    assert v.recovery.main.cd == pytest.approx(2.0)
    assert v.recovery.main.diameter == pytest.approx(1.888139)
    assert v.recovery.main.area == pytest.approx(2.8, rel=1e-4)
    assert v.recovery.main.threshold == pytest.approx(305.0)


def test_recovery_numeric_threshold(tmp_path):
    """main threshold=305 parses as float; drogue threshold='apogee' as literal."""
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert isinstance(v.recovery.main.threshold, float)
    assert v.recovery.drogue.threshold == "apogee"


def test_recovery_drogue_optional(tmp_path):
    """Omitting the drogue key gives drogue=None."""
    yaml_no_drogue = _VEHICLE_YAML.replace(
        "    recovery:\n"
        "      drogue:\n"
        "        cd: 2.0\n"
        "        diameter: 0.437019\n"
        "        threshold: apogee\n",
        "    recovery:\n",
    )
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", yaml_no_drogue))
    assert v.recovery.drogue is None
    assert v.recovery.main is not None


def test_recovery_drogue_without_main_raises(tmp_path):
    """Drogue present but main absent is an invalid configuration."""
    yaml_no_main = _VEHICLE_YAML.replace(
        "      main:\n"
        "        cd: 2.0\n"
        "        diameter: 1.888139\n"
        "        threshold: 305\n",
        "",
    )
    with pytest.raises(ValueError, match="drogue"):
        load_vehicle(_write(tmp_path, "v.yaml", yaml_no_main))


def test_recovery_main_only(tmp_path):
    """Omitting the drogue gives main-only vehicle."""
    yaml_main_only = _VEHICLE_YAML.replace(
        "    recovery:\n"
        "      drogue:\n"
        "        cd: 2.0\n"
        "        diameter: 0.437019\n"
        "        threshold: apogee\n",
        "    recovery:\n",
    )
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", yaml_main_only))
    assert v.recovery.drogue is None
    assert v.recovery.main is not None


def test_recovery_no_chutes(tmp_path):
    """Omitting both parachutes gives drogue=None, main=None."""
    yaml_no_chutes = _VEHICLE_YAML.replace(
        "    recovery:\n"
        "      drogue:\n"
        "        cd: 2.0\n"
        "        diameter: 0.437019\n"
        "        threshold: apogee\n"
        "      main:\n"
        "        cd: 2.0\n"
        "        diameter: 1.888139\n"
        "        threshold: 305\n",
        "    recovery:\n",
    )
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", yaml_no_chutes))
    assert v.recovery.drogue is None
    assert v.recovery.main is None


def test_vehicle_is_frozen(tmp_path):
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    with pytest.raises(Exception):
        v.geometry.diameter = 0.2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# active_scenarios property tests
# ---------------------------------------------------------------------------

def test_active_scenarios_both_numeric_main(tmp_path):
    """Both drogue and main with numeric main threshold → all four scenarios."""
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", _VEHICLE_YAML))
    assert set(v.recovery.active_scenarios) == {
        "nominal", "ballistic", "drogue_only", "premature_main"
    }


def test_active_scenarios_both_apogee_main(tmp_path):
    """Both drogue and main with apogee main threshold → no premature_main."""
    yaml_apogee_main = _VEHICLE_YAML.replace("threshold: 305", "threshold: apogee")
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", yaml_apogee_main))
    assert set(v.recovery.active_scenarios) == {
        "nominal", "ballistic", "drogue_only"
    }


def test_active_scenarios_main_only_numeric(tmp_path):
    """Main only with numeric threshold → nominal, ballistic, premature_main."""
    yaml_main_only = _VEHICLE_YAML.replace(
        "    recovery:\n"
        "      drogue:\n"
        "        cd: 2.0\n"
        "        diameter: 0.437019\n"
        "        threshold: apogee\n",
        "    recovery:\n",
    )
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", yaml_main_only))
    assert set(v.recovery.active_scenarios) == {
        "nominal", "ballistic", "premature_main"
    }


def test_active_scenarios_no_chutes(tmp_path):
    """No parachutes → nominal only."""
    yaml_no_chutes = _VEHICLE_YAML.replace(
        "    recovery:\n"
        "      drogue:\n"
        "        cd: 2.0\n"
        "        diameter: 0.437019\n"
        "        threshold: apogee\n"
        "      main:\n"
        "        cd: 2.0\n"
        "        diameter: 1.888139\n"
        "        threshold: 305\n",
        "    recovery:\n",
    )
    v, _ = load_vehicle(_write(tmp_path, "v.yaml", yaml_no_chutes))
    assert v.recovery.active_scenarios == ("nominal",)


# ---------------------------------------------------------------------------
# Round-trip test against the real input files
# ---------------------------------------------------------------------------

def test_real_simulation_yaml_loads(sim_yaml_path):
    """The committed simulation YAML must parse without errors."""
    cfg = load_simulation_config(sim_yaml_path)
    assert cfg.site.latitude == pytest.approx(58.61047)


def test_real_vehicle_yaml_loads(vehicle_yaml_path):
    """The committed vehicle YAML must parse without errors."""
    v, _ = load_vehicle(vehicle_yaml_path)
    assert v.geometry.diameter == pytest.approx(0.130)
