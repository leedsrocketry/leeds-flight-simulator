"""Shared test configuration and fixtures.

Adds the project root to sys.path (replacing pyproject.toml's
``pythonpath`` setting) and provides paths to example simulation data
so that individual test modules don't hardcode directory layouts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is importable (replaces pyproject.toml pythonpath)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Example simulation data directory
# ---------------------------------------------------------------------------
EXAMPLE_SIM_DIR = PROJECT_ROOT / "simulations" / "g2b2-safety-case"


@pytest.fixture(scope="session")
def example_sim_dir() -> Path:
    """Path to the example simulation data directory."""
    return EXAMPLE_SIM_DIR


@pytest.fixture(scope="session")
def sim_yaml_path() -> Path:
    """Path to the example simulation YAML, skipping if absent."""
    p = EXAMPLE_SIM_DIR / "cape-wrath.yaml"
    if not p.exists():
        pytest.skip("cape-wrath.yaml not present")
    return p


@pytest.fixture(scope="session")
def vehicle_yaml_path() -> Path:
    """Path to the example vehicle YAML, skipping if absent."""
    p = EXAMPLE_SIM_DIR / "g2b2.yaml"
    if not p.exists():
        pytest.skip("g2b2.yaml not present")
    return p


@pytest.fixture(scope="session")
def motor_path() -> Path:
    """Path to the example motor file, skipping if absent."""
    p = EXAMPLE_SIM_DIR / "o3400.eng"
    if not p.exists():
        pytest.skip("o3400.eng not present")
    return p


@pytest.fixture(scope="session")
def aero_dir() -> Path:
    """Path to the example aero tables directory, skipping if absent/empty."""
    p = EXAMPLE_SIM_DIR / "aero_tables"
    if not p.exists() or not list(p.glob("*.csv")):
        pytest.skip("aero_tables not present or empty")
    return p


@pytest.fixture(scope="session")
def d802_path() -> Path:
    """Path to the D802 danger area GeoJSON, skipping if absent."""
    p = EXAMPLE_SIM_DIR / "d802.geojson"
    if not p.exists():
        pytest.skip("d802.geojson not present")
    return p


@pytest.fixture(scope="session")
def coastline_path() -> Path:
    """Path to the coastline GeoJSON, skipping if absent."""
    p = EXAMPLE_SIM_DIR / "coastline.geojson"
    if not p.exists():
        pytest.skip("coastline.geojson not present")
    return p
