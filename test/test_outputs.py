"""Tests for outputs.py — CSV/YAML serialisation and plot generation."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from montecarlo import SampleResult, ScenarioStats, MonteCarloResult, SCENARIO_LABELS
from optimisation import OptimisationResult
from config import (
    SimulationConfig, SiteConfig, LaunchConfig, RailConfig,
    MonteCarloConfig, UncertaintiesConfig, AcceptanceConfig,
    MonitourStation, MapMarker,
)
from outputs import (
    create_results_dir,
    write_samples_csv,
    write_summary_yaml,
    save_altitude_plot,
    save_dispersion_plot,
    save_replay_3d,
    save_replay_plan_view,
    save_replay_altitude,
    _fit_ellipse_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sample(
    sample_id: int = 0,
    scenario: str = "nominal",
    compliant: bool = True,
    landing_at_sea: bool | None = None,
    in_coverage: bool | None = None,
    landing_north: float = 1000.0,
    landing_east: float = 500.0,
) -> SampleResult:
    """Create a minimal SampleResult for testing."""
    return SampleResult(
        sample_id=sample_id,
        scenario=scenario,
        run_index=0,
        apogee_m=15000.0,
        apogee_lat=58.75,
        apogee_lon=-4.80,
        apogee_north=2000.0,
        apogee_east=300.0,
        landing_lat=58.62,
        landing_lon=-4.93,
        landing_north=landing_north,
        landing_east=landing_east,
        flight_time_s=300.0,
        peak_mach=2.1,
        peak_altitude_ft=49212.0,
        max_aoa_deg=3.5,
        min_sm_subsonic=2.0,
        min_sm_supersonic=1.5,
        compliant=compliant,
        in_buffer=True,
        below_ceiling=True,
        stability_compliant=True,
        landing_at_sea=landing_at_sea,
        in_coverage=in_coverage,
        violation_reason="",
        wind_profile_index=0,
        impulse_factor=1.0,
        azimuth_deg=315.0,
        inclination_deg=85.0,
        fin_cant_deg=0.5,
    )


def _make_scenario_stats(
    scenario: str = "nominal",
    n_samples: int = 100,
    n_compliant: int = 98,
) -> ScenarioStats:
    return ScenarioStats(
        scenario=scenario,
        n_samples=n_samples,
        n_compliant=n_compliant,
        n_non_compliant=n_samples - n_compliant,
        passed=True,
        apogee_mean=15000.0,
        apogee_std=200.0,
        apogee_min=14500.0,
        apogee_max=15500.0,
        landing_dist_mean=2000.0,
        landing_dist_std=300.0,
        landing_dist_min=1500.0,
        landing_dist_max=2500.0,
        peak_mach_mean=2.1,
        peak_mach_std=0.05,
        max_aoa_mean=3.5,
        max_aoa_std=0.8,
        sm_subsonic_min=1.8,
        sm_supersonic_min=1.2,
    )


def _make_mc_result(
    scenarios: list[str] | None = None,
    n_samples_per: int = 10,
) -> MonteCarloResult:
    if scenarios is None:
        scenarios = ["nominal", "ballistic"]

    all_results: list[SampleResult] = []
    rng = np.random.default_rng(42)
    for sc in scenarios:
        for i in range(n_samples_per):
            all_results.append(_make_sample(
                sample_id=i,
                scenario=sc,
                landing_north=1000.0 + rng.normal(0, 200),
                landing_east=500.0 + rng.normal(0, 100),
            ))

    stats = {sc: _make_scenario_stats(sc, n_samples_per) for sc in scenarios}

    return MonteCarloResult(
        all_results=all_results,
        scenario_stats=stats,
        all_passed=True,
        warnings=["test warning"],
        azimuth_mean=315.0,
        inclination_mean=85.0,
    )


def _make_sim_cfg(tmp_path: Path) -> SimulationConfig:
    """Create a minimal SimulationConfig for testing."""
    # Create a stub danger area GeoJSON
    geojson_path = tmp_path / "danger_area.geojson"
    geojson_path.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        '"geometry":{"type":"Polygon","coordinates":[[[-5.0,58.5],'
        '[-4.5,58.5],[-4.5,58.8],[-5.0,58.8],[-5.0,58.5]]]},'
        '"properties":{}}]}',
        encoding="utf-8",
    )

    return SimulationConfig(
        vehicle=tmp_path / "vehicle.yaml",
        site=SiteConfig(
            latitude=58.6105,
            longitude=-4.9435,
            ballistic_exclusion_radius=500.0,
            launch_monitour_radius=200.0,
            altitude_ceiling=16764.0,
            danger_area=geojson_path,
            coastline=None,
            coastline_mode="sea",
            monitour_stations=(
                MonitourStation(
                    name="Visibility Coverage",
                    latitude=58.6105,
                    longitude=-4.9435,
                    radius=5000.0,
                ),
            ),
            map_markers=(
                MapMarker(name="Durness", latitude=58.5687, longitude=-4.7476),
            ),
        ),
        launch=LaunchConfig(
            rail=RailConfig(
                azimuth=315.0,
                azimuth_range=None,
                inclination=85.0,
                inclination_range=None,
                length=6.0,
            ),
            wind_profiles=tmp_path / "wind.npz",
            surface_wind=None,
        ),
        monte_carlo=MonteCarloConfig(
            samples=100,
            seed=42,
            uncertainties=UncertaintiesConfig(
                azimuth_sigma=2.0,
                inclination_sigma=1.0,
                fin_cant_sigma=0.3,
                impulse_factor_sigma=0.067,
            ),
            acceptance=AcceptanceConfig(
                compliance_threshold=0.997,
                buffer_distance=1000.0,
                sm_transition_mach=0.8,
                sm_subsonic_min=1.5,
                sm_supersonic_min=1.0,
                aoa_max=15.0,
                sm_aoa_threshold=5.0,
                sea_check_scenarios=(),
                monitour_check_scenarios=(),
            ),
        ),
        verification=None,
    )


def _make_opt_result() -> OptimisationResult:
    return OptimisationResult(
        selected_azimuth=315,
        selected_inclination=85,
        phase1_apogees={85: (100.0, 50.0, -15000.0)},
        phase1_ballistic_landings={85: (2000.0, 1000.0)},
        phase1_selected=85,
        phase2_feasible=[310, 315, 320],
        phase2_total_candidates=36,
        phase3_observations=[(310, 0.95), (315, 0.99), (320, 0.92)],
        phase3_top_candidates=[315, 310],
        phase4_compliance={315: 0.998, 310: 0.995},
        phase4_margins={315: 150.0, 310: 80.0},
    )


# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------

class TestCreateResultsDir:
    def test_creates_directory(self, tmp_path: Path):
        sim_yaml = tmp_path / "simulation.yaml"
        sim_yaml.touch()
        results_dir = create_results_dir(sim_yaml, _clear=False)
        assert results_dir.exists()
        assert results_dir.is_dir()
        assert results_dir.name == "results"
        assert results_dir.parent == tmp_path

    def test_clears_existing_contents(self, tmp_path: Path):
        sim_yaml = tmp_path / "simulation.yaml"
        sim_yaml.touch()
        # Pre-populate results/
        old_results = tmp_path / "results"
        old_results.mkdir()
        (old_results / "old_file.txt").write_text("stale")
        results_dir = create_results_dir(sim_yaml, _clear=True)
        assert results_dir.exists()
        assert not (results_dir / "old_file.txt").exists()

    def test_wind_profile_subfolder(self, tmp_path: Path):
        """When multiple wind profiles are used, a sub-folder is created."""
        sim_yaml = tmp_path / "simulation.yaml"
        sim_yaml.touch()
        results_dir = create_results_dir(sim_yaml, wind_profile_suffix="day1", _clear=False)
        assert results_dir.name == "day1"
        assert results_dir.parent.name == "results"
        assert results_dir.exists()

    def test_no_subfolder_when_none(self, tmp_path: Path):
        sim_yaml = tmp_path / "simulation.yaml"
        sim_yaml.touch()
        results_dir = create_results_dir(sim_yaml, wind_profile_suffix=None, _clear=False)
        assert results_dir.name == "results"


# ---------------------------------------------------------------------------
# Per-sample CSV
# ---------------------------------------------------------------------------

class TestWriteSamplesCsv:
    def test_basic_columns(self, tmp_path: Path):
        results = [_make_sample(i) for i in range(3)]
        path = write_samples_csv(results, tmp_path, has_coastline=False, has_monitour=False)
        assert path.exists()
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "sample_id" in header
        assert "scenario" in header
        assert "fin_cant_deg" in header
        assert "coastline_compliant" not in header
        assert "monitour_compliant" not in header
        assert "compliant" not in header
        # Removed columns should not appear
        assert "wind_profile_index" not in header
        assert "peak_altitude_ft" not in header

    def test_coastline_column_present(self, tmp_path: Path):
        results = [_make_sample(0, landing_at_sea=False)]
        path = write_samples_csv(results, tmp_path, has_coastline=True, has_monitour=False)
        with open(path) as f:
            header = next(csv.reader(f))
        assert "coastline_compliant" in header

    def test_monitour_column_present(self, tmp_path: Path):
        results = [_make_sample(0, in_coverage=True)]
        path = write_samples_csv(results, tmp_path, has_coastline=False, has_monitour=True)
        with open(path) as f:
            header = next(csv.reader(f))
        assert "monitour_compliant" in header

    def test_row_count(self, tmp_path: Path):
        n = 5
        results = [_make_sample(i) for i in range(n)]
        path = write_samples_csv(results, tmp_path, has_coastline=False, has_monitour=False)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == n + 1  # header + data

    def test_column_order_matches_spec(self, tmp_path: Path):
        """Verify columns: inputs → flight time → compliance → values → locations."""
        results = [_make_sample(0, landing_at_sea=True, in_coverage=True)]
        path = write_samples_csv(results, tmp_path, has_coastline=True, has_monitour=True)
        with open(path) as f:
            header = next(csv.reader(f))
        # Inputs first
        assert header[0] == "sample_id"
        assert header[1] == "scenario"
        assert header[2] == "azimuth_deg"
        # Flight time before compliance columns
        ft_idx = header.index("flight_time_s")
        stab_idx = header.index("stability_compliant")
        assert ft_idx < stab_idx
        # Locations last
        assert header[-1] == "apogee_lon"

    def test_bool_values_written(self, tmp_path: Path):
        results = [_make_sample(0, compliant=True)]
        path = write_samples_csv(results, tmp_path, has_coastline=False, has_monitour=False)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
        stab_idx = header.index("stability_compliant")
        assert row[stab_idx] == "True"


# ---------------------------------------------------------------------------
# Summary YAML
# ---------------------------------------------------------------------------

class TestWriteSummaryYaml:
    def test_basic_structure(self, tmp_path: Path):
        mc = _make_mc_result()
        sim_cfg = _make_sim_cfg(tmp_path)
        path = write_summary_yaml(mc, sim_cfg, None, tmp_path)
        assert path.exists()
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "metadata" in data
        assert "scenarios" in data
        assert "warnings" in data["metadata"]
        assert "config" in data["metadata"]

    def test_no_optimisation(self, tmp_path: Path):
        mc = _make_mc_result()
        sim_cfg = _make_sim_cfg(tmp_path)
        path = write_summary_yaml(mc, sim_cfg, None, tmp_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "optimisation" not in data

    def test_with_optimisation(self, tmp_path: Path):
        mc = _make_mc_result()
        sim_cfg = _make_sim_cfg(tmp_path)
        opt = _make_opt_result()
        path = write_summary_yaml(mc, sim_cfg, opt, tmp_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        opt_data = data["optimisation"]
        assert opt_data["selected_azimuth"] == 315
        assert opt_data["selected_inclination"] == 85
        assert len(opt_data["phase2_feasible_azimuths"]) == 3
        assert "azimuth_mean" in opt_data
        assert "inclination_mean" in opt_data

    def test_scenario_stats(self, tmp_path: Path):
        mc = _make_mc_result(scenarios=["nominal"])
        sim_cfg = _make_sim_cfg(tmp_path)
        path = write_summary_yaml(mc, sim_cfg, None, tmp_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        sc = data["scenarios"]["nominal"]
        assert "compliant" in sc
        assert "apogee_m" in sc
        assert "mean" in sc["apogee_m"]
        assert "stability_margin" in sc
        assert "subsonic_min" in sc["stability_margin"]

    def test_warnings_from_all_warnings(self, tmp_path: Path):
        mc = _make_mc_result()
        sim_cfg = _make_sim_cfg(tmp_path)
        all_w = ["cli warning", "test warning"]
        path = write_summary_yaml(mc, sim_cfg, None, tmp_path, all_warnings=all_w)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["metadata"]["warnings"] == ["cli warning", "test warning"]

    def test_warnings_fallback_to_mc(self, tmp_path: Path):
        mc = _make_mc_result()
        sim_cfg = _make_sim_cfg(tmp_path)
        path = write_summary_yaml(mc, sim_cfg, None, tmp_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "test warning" in data["metadata"]["warnings"]


# ---------------------------------------------------------------------------
# Altitude plot
# ---------------------------------------------------------------------------

class TestSaveAltitudePlot:
    def test_generates_file(self, tmp_path: Path):
        """Plot file is created and non-empty."""
        import matplotlib
        matplotlib.use("Agg")

        t = np.linspace(0, 300, 200)
        alt = 15000.0 * np.sin(np.pi * t / 120.0)
        alt[t > 120] = np.maximum(15000.0 - 50.0 * (t[t > 120] - 120.0), 0)

        scenarios = {
            "nominal": (t.copy(), alt.copy()),
            "ballistic": (t[:150].copy(), alt[:150].copy()),
        }
        path = save_altitude_plot(scenarios, 6.2, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 1000  # non-trivial PNG

    def test_filename(self, tmp_path: Path):
        import matplotlib
        matplotlib.use("Agg")

        t = np.linspace(0, 300, 100)
        alt = 10000.0 * np.sin(np.pi * t / 100.0)
        alt = np.maximum(alt, 0)
        scenarios = {"nominal": (t, alt)}
        path = save_altitude_plot(scenarios, 5.0, tmp_path)
        assert path.name == "altitude_plot.png"


# ---------------------------------------------------------------------------
# Dispersion plot
# ---------------------------------------------------------------------------

class TestSaveDispersionPlot:
    def test_generates_file(self, tmp_path: Path):
        """Plot file is created and non-empty."""
        import matplotlib
        matplotlib.use("Agg")

        mc = _make_mc_result(scenarios=["nominal", "ballistic"], n_samples_per=20)
        sim_cfg = _make_sim_cfg(tmp_path)
        path = save_dispersion_plot(
            mc.all_results, sim_cfg, 0.997, tmp_path,
        )
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_filename(self, tmp_path: Path):
        import matplotlib
        matplotlib.use("Agg")

        mc = _make_mc_result(n_samples_per=10)
        sim_cfg = _make_sim_cfg(tmp_path)
        path = save_dispersion_plot(
            mc.all_results, sim_cfg, 0.95, tmp_path,
        )
        assert path.name == "dispersion_plot.png"


# ---------------------------------------------------------------------------
# Ellipse fitting
# ---------------------------------------------------------------------------

class TestFitEllipseThreshold:
    def test_contains_threshold_fraction(self):
        """Fitted ellipse should contain at least the threshold fraction."""
        rng = np.random.default_rng(0)
        points = rng.normal(size=(500, 2))
        threshold = 0.95
        el = _fit_ellipse_threshold(points, threshold)

        # Check containment: project points and count inside
        de = points[:, 1] - el["center_e"]
        dn = points[:, 0] - el["center_n"]
        angle = math.radians(el["angle_deg"])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        proj_a = de * cos_a + dn * sin_a
        proj_b = -de * sin_a + dn * cos_a
        inside = (proj_a / el["semi_a"]) ** 2 + (proj_b / el["semi_b"]) ** 2 <= 1.0
        frac = inside.sum() / len(points)
        assert frac >= threshold - 0.01  # allow tiny rounding

    def test_higher_threshold_larger_ellipse(self):
        """A higher threshold should produce a larger or equal ellipse."""
        rng = np.random.default_rng(1)
        points = rng.normal(size=(200, 2))
        el_low = _fit_ellipse_threshold(points, 0.90)
        el_high = _fit_ellipse_threshold(points, 0.99)
        area_low = el_low["semi_a"] * el_low["semi_b"]
        area_high = el_high["semi_a"] * el_high["semi_b"]
        assert area_high >= area_low

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            _fit_ellipse_threshold(np.array([1.0, 2.0]), 0.95)


# ---------------------------------------------------------------------------
# Replay stubs
# ---------------------------------------------------------------------------

class TestReplayStubs:
    def test_replay_3d_not_implemented(self):
        with pytest.raises(NotImplementedError):
            save_replay_3d([], Path("."))

    def test_replay_plan_view_not_implemented(self):
        with pytest.raises(NotImplementedError):
            save_replay_plan_view([], Path("."))

    def test_replay_altitude_not_implemented(self):
        with pytest.raises(NotImplementedError):
            save_replay_altitude([], Path("."))


# ---------------------------------------------------------------------------
# Test point generator (moved from reference script)
# ---------------------------------------------------------------------------

def generate_test_points(
    offset_east_km: float,
    offset_north_km: float,
    spread_km: float = 0.2,
    n_points: int = 100,
    spread_ratio: float = 1.0,
    angle_deg: float = 0.0,
    rng_seed: int | None = None,
) -> np.ndarray:
    """Gaussian point cloud centred at (offset_north_km, offset_east_km).

    Returns (N, 2) array [north_km, east_km].
    """
    rng = np.random.default_rng(rng_seed)
    local = rng.standard_normal((n_points, 2)) * np.array(
        [spread_km, spread_km * spread_ratio]
    )
    theta = np.radians(angle_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    rotated = local @ rot.T
    return np.column_stack([
        rotated[:, 1] + offset_north_km,
        rotated[:, 0] + offset_east_km,
    ])
