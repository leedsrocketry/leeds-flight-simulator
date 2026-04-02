"""Shared test configuration and fixtures.

Adds the project root to sys.path (replacing pyproject.toml's
``pythonpath`` setting) and provides paths to example simulation data
so that individual test modules don't hardcode directory layouts.

The simulation data directory defaults to the ``sim_data`` value in
``pytest.ini`` (relative to the project root) and can be overridden
with ``--sim-data``::

    python -m pytest --sim-data ../simulations/cases/g2b2-cape-wrath
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Ensure the project root is importable (replaces pyproject.toml pythonpath)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# --sim-data CLI option
# ---------------------------------------------------------------------------

# Module-level default for tests that import EXAMPLE_SIM_DIR directly.
# Fixtures use the configurable --sim-data / pytest.ini value instead.
EXAMPLE_SIM_DIR = PROJECT_ROOT.parent / "simulations" / "cases" / "g2b2-cape-wrath"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini("sim_data", "Path to simulation data directory (relative to project root)")
    parser.addoption(
        "--sim-data",
        action="store",
        default=None,
        help="Path to simulation data directory (overrides pytest.ini sim_data)",
    )


def _resolve_sim_dir(config: pytest.Config) -> Path:
    """Return the resolved simulation data directory."""
    cli_value = config.getoption("--sim-data")
    if cli_value is not None:
        return Path(cli_value).resolve()
    ini_value = config.getini("sim_data")
    if ini_value:
        return (PROJECT_ROOT / ini_value).resolve()
    return PROJECT_ROOT.parent / "simulations" / "cases" / "g2b2-cape-wrath"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def example_sim_dir(pytestconfig: pytest.Config) -> Path:
    """Path to the simulation data directory."""
    return _resolve_sim_dir(pytestconfig)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return the top-level dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def sim_yaml_path(example_sim_dir: Path) -> Path:
    """Path to the example simulation YAML, skipping if absent."""
    p = example_sim_dir / "config.yaml"
    if not p.exists():
        pytest.skip("config.yaml not present")
    return p


@pytest.fixture(scope="session")
def vehicle_yaml_path(sim_yaml_path: Path) -> Path:
    """Path to the vehicle YAML referenced by the simulation config."""
    cfg = _load_yaml(sim_yaml_path)
    p = (sim_yaml_path.parent / cfg["vehicle"]).resolve()
    if not p.exists():
        pytest.skip(f"vehicle YAML not present: {p}")
    return p


@pytest.fixture(scope="session")
def motor_path(vehicle_yaml_path: Path) -> Path:
    """Path to the motor file referenced by the vehicle config."""
    vcfg = _load_yaml(vehicle_yaml_path)
    p = (vehicle_yaml_path.parent / vcfg["motor"]).resolve()
    if not p.exists():
        pytest.skip(f"motor file not present: {p}")
    return p


@pytest.fixture(scope="session")
def aero_dir(vehicle_yaml_path: Path) -> Path:
    """Path to the aero tables directory referenced by the vehicle config."""
    vcfg = _load_yaml(vehicle_yaml_path)
    p = (vehicle_yaml_path.parent / vcfg["aero_tables"]).resolve()
    if not p.exists() or not list(p.glob("*.csv")):
        pytest.skip("aero-tables not present or empty")
    return p


@pytest.fixture(scope="session")
def d802_path(example_sim_dir: Path) -> Path:
    """Path to the D802 danger area GeoJSON, skipping if absent."""
    p = example_sim_dir / "d802.geojson"
    if not p.exists():
        pytest.skip("d802.geojson not present")
    return p


@pytest.fixture(scope="session")
def coastline_path(example_sim_dir: Path) -> Path:
    """Path to the coastline GeoJSON, skipping if absent."""
    p = example_sim_dir / "coastline.geojson"
    if not p.exists():
        pytest.skip("coastline.geojson not present")
    return p
