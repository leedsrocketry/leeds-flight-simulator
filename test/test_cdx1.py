"""Tests for cdx1 module — YAML loader and comparison with auto inclination."""
from __future__ import annotations

import textwrap

import pytest
import yaml

from cdx1 import load_yaml_for_diff, build_comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yamls(tmp_path, rail_inclination, verification_inclination=None):
    """Write minimal simulation and vehicle YAMLs, return (sim_path, veh_path)."""
    veh = {
        "geometry": {
            "length": 2.0,
            "diameter": 0.1,
            "nozzle_diameter": 0.03,
        },
        "mass": {
            "wet_mass": 20.0,
            "wet_cg": 1.0,
        },
        "recovery": {
            "drogue": {"cd": 1.5, "diameter": 0.5, "threshold": "apogee"},
            "main": {"cd": 2.2, "diameter": 1.2, "threshold": 300.0},
        },
        "motor": "test_motor.eng",
    }
    sim = {
        "launch": {
            "rail": {
                "inclination": rail_inclination,
                "length": 5.0,
            },
        },
    }
    if verification_inclination is not None:
        sim["verification"] = {"inclination": verification_inclination}

    veh_path = tmp_path / "vehicle.yaml"
    sim_path = tmp_path / "simulation.yaml"
    with open(veh_path, "w") as f:
        yaml.dump(veh, f)
    with open(sim_path, "w") as f:
        yaml.dump(sim, f)
    return sim_path, veh_path


def _dummy_cdx1():
    """Return a minimal CDX1 dict matching parse_cdx1 output."""
    return {
        "total_length_m": 2.0,
        "diameter_m": 0.1,
        "nozzle_diameter_m": 0.03,
        "launch_mass_kg": 20.0,
        "cg_m": 1.0,
        "rod_angle_deg": 5.0,   # → inclination = 85°
        "rod_length_m": 5.0,
        "temperature_K": 288.15,
        "pressure_Pa": 101325.0,
        "altitude_m": 0.0,
        "drogue_cd": 1.5,
        "drogue_diameter_m": 0.5,
        "main_cd": 2.2,
        "main_diameter_m": 1.2,
        "main_deploy_alt_m": 300.0,
        "drogue_deploy": "Apogee",
        "main_deploy": "Altitude",
        "motor_name": "test_motor",
        "motor_matched": True,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAutoInclination:
    """load_yaml_for_diff and build_comparison handle 'auto' inclination."""

    def test_auto_rail_inclination_skips_row(self, tmp_path):
        sim_path, veh_path = _write_yamls(tmp_path, rail_inclination="auto")
        yaml_cfg = load_yaml_for_diff(sim_path, veh_path)
        assert yaml_cfg["rail_inclination_deg"] is None

        rows = build_comparison(_dummy_cdx1(), yaml_cfg)
        labels = [r.label for r in rows]
        assert "Inclination (deg)" not in labels

    def test_numeric_inclination_still_compared(self, tmp_path):
        sim_path, veh_path = _write_yamls(tmp_path, rail_inclination=85.0)
        yaml_cfg = load_yaml_for_diff(sim_path, veh_path)
        assert yaml_cfg["rail_inclination_deg"] == 85.0

        rows = build_comparison(_dummy_cdx1(), yaml_cfg)
        labels = [r.label for r in rows]
        assert "Inclination (deg)" in labels

    def test_verification_overrides_auto_rail(self, tmp_path):
        """Numeric verification inclination takes precedence over auto rail."""
        sim_path, veh_path = _write_yamls(
            tmp_path, rail_inclination="auto", verification_inclination=82.0,
        )
        yaml_cfg = load_yaml_for_diff(sim_path, veh_path)
        assert yaml_cfg["rail_inclination_deg"] == 82.0
