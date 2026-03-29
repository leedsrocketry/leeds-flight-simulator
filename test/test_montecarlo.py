"""Tests for montecarlo.py — sample draw reproducibility and scenario statistics.

Scope
-----
- generate_sample_draws determinism and independence
- compute_scenario_stats correctness (mean, std, min, max, pass/fail)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from montecarlo import (
    SampleResult,
    StochasticDraws,
    compute_scenario_stats,
    generate_sample_draws,
)
from config import UncertaintiesConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def uncertainties() -> UncertaintiesConfig:
    return UncertaintiesConfig(
        azimuth_sigma=1.0,
        inclination_sigma=0.5,
        fin_cant_sigma=0.02,
        impulse_factor_sigma=0.067,
    )


# ===========================================================================
# 1. generate_sample_draws — reproducibility
# ===========================================================================

class TestGenerateSampleDraws:
    def test_same_inputs_same_outputs(self, uncertainties):
        """Identical (seed, run, sample) must produce identical draws."""
        a = generate_sample_draws(42, 0, 0, 0.0, 87.0, uncertainties)
        b = generate_sample_draws(42, 0, 0, 0.0, 87.0, uncertainties)
        assert a == b

    def test_different_sample_index(self, uncertainties):
        """Different sample indices must produce different draws."""
        a = generate_sample_draws(42, 0, 0, 0.0, 87.0, uncertainties)
        b = generate_sample_draws(42, 0, 1, 0.0, 87.0, uncertainties)
        assert a != b

    def test_different_run_index(self, uncertainties):
        """Different run indices must produce different draws."""
        a = generate_sample_draws(42, 0, 0, 0.0, 87.0, uncertainties)
        b = generate_sample_draws(42, 1, 0, 0.0, 87.0, uncertainties)
        assert a != b

    def test_different_seed(self, uncertainties):
        """Different master seeds must produce different draws."""
        a = generate_sample_draws(42, 0, 0, 0.0, 87.0, uncertainties)
        b = generate_sample_draws(99, 0, 0, 0.0, 87.0, uncertainties)
        assert a != b

    def test_wind_profile_index_equals_sample_index(self, uncertainties):
        """Wind profile index should equal the sample index."""
        for i in [0, 1, 50, 999]:
            draws = generate_sample_draws(42, 0, i, 0.0, 87.0, uncertainties)
            assert draws.wind_profile_index == i

    def test_mean_values_centred(self, uncertainties):
        """Over many samples, draws should be centred on the mean values."""
        az_mean, inc_mean = 45.0, 87.0
        n = 5000
        azimuths = []
        inclinations = []
        impulses = []
        cants = []
        for i in range(n):
            d = generate_sample_draws(42, 0, i, az_mean, inc_mean, uncertainties)
            azimuths.append(d.azimuth_deg)
            inclinations.append(d.inclination_deg)
            impulses.append(d.impulse_factor)
            cants.append(d.fin_cant_deg)

        assert np.mean(azimuths) == pytest.approx(az_mean, abs=0.1)
        assert np.mean(inclinations) == pytest.approx(inc_mean, abs=0.05)
        assert np.mean(impulses) == pytest.approx(1.0, abs=0.01)
        assert np.mean(cants) == pytest.approx(0.0, abs=0.005)

    def test_sigma_values(self, uncertainties):
        """Standard deviations should match the configured sigmas."""
        n = 5000
        azimuths = []
        inclinations = []
        impulses = []
        cants = []
        for i in range(n):
            d = generate_sample_draws(42, 0, i, 0.0, 87.0, uncertainties)
            azimuths.append(d.azimuth_deg)
            inclinations.append(d.inclination_deg)
            impulses.append(d.impulse_factor)
            cants.append(d.fin_cant_deg)

        assert np.std(azimuths) == pytest.approx(1.0, rel=0.1)
        assert np.std(inclinations) == pytest.approx(0.5, rel=0.1)
        assert np.std(impulses) == pytest.approx(0.067, rel=0.1)
        assert np.std(cants) == pytest.approx(0.02, rel=0.1)

    def test_order_independence(self, uncertainties):
        """Drawing sample 500 directly must equal drawing it after others."""
        direct = generate_sample_draws(42, 0, 500, 0.0, 87.0, uncertainties)
        # Draw 0..499 first (each is independent via SeedSequence)
        for i in range(500):
            generate_sample_draws(42, 0, i, 0.0, 87.0, uncertainties)
        after_others = generate_sample_draws(42, 0, 500, 0.0, 87.0, uncertainties)
        assert direct == after_others


# ===========================================================================
# 2. compute_scenario_stats
# ===========================================================================

def _make_sample_result(
    sample_id: int = 0,
    compliant: bool = True,
    apogee_m: float = 15000.0,
    landing_north: float = 100.0,
    landing_east: float = 200.0,
    peak_mach: float = 2.5,
    max_aoa_deg: float = 3.0,
    min_sm_subsonic: float = 2.0,
    min_sm_supersonic: float = 3.0,
    **kwargs,
) -> SampleResult:
    """Create a SampleResult with sensible defaults for testing."""
    defaults = dict(
        sample_id=sample_id,
        scenario="nominal",
        run_index=0,
        apogee_m=apogee_m,
        apogee_lat=58.61,
        apogee_lon=-4.94,
        apogee_north=2000.0,
        apogee_east=300.0,
        landing_lat=58.62,
        landing_lon=-4.93,
        landing_north=landing_north,
        landing_east=landing_east,
        flight_time_s=300.0,
        peak_mach=peak_mach,
        peak_altitude_ft=apogee_m * 3.28084,
        max_aoa_deg=max_aoa_deg,
        min_sm_subsonic=min_sm_subsonic,
        min_sm_supersonic=min_sm_supersonic,
        compliant=compliant,
        in_buffer=True,
        below_ceiling=True,
        stability_compliant=True,
        landing_at_sea=None,
        in_coverage=None,
        violation_reason="",
        wind_profile_index=sample_id,
        impulse_factor=1.0,
        azimuth_deg=0.0,
        inclination_deg=87.0,
        fin_cant_deg=0.0,
    )
    defaults.update(kwargs)
    return SampleResult(**defaults)


class TestComputeScenarioStats:
    def test_all_compliant(self):
        """All samples compliant → passed is True."""
        results = [_make_sample_result(sample_id=i) for i in range(100)]
        stats = compute_scenario_stats(results, compliance_threshold=0.99)
        assert stats.n_samples == 100
        assert stats.n_compliant == 100
        assert stats.n_non_compliant == 0
        assert stats.passed is True

    def test_all_non_compliant(self):
        """All samples non-compliant → passed is False."""
        results = [
            _make_sample_result(sample_id=i, compliant=False)
            for i in range(100)
        ]
        stats = compute_scenario_stats(results, compliance_threshold=0.99)
        assert stats.n_compliant == 0
        assert stats.n_non_compliant == 100
        assert stats.passed is False

    def test_threshold_exact_boundary(self):
        """Exactly at the threshold boundary → passed is True."""
        # 997 out of 1000 = 0.997 exactly
        results = (
            [_make_sample_result(sample_id=i, compliant=True) for i in range(997)]
            + [_make_sample_result(sample_id=i + 997, compliant=False) for i in range(3)]
        )
        stats = compute_scenario_stats(results, compliance_threshold=0.997)
        assert stats.passed is True

    def test_threshold_just_below(self):
        """One fewer compliant than needed → passed is False."""
        # 996 out of 1000 = 0.996 < 0.997
        results = (
            [_make_sample_result(sample_id=i, compliant=True) for i in range(996)]
            + [_make_sample_result(sample_id=i + 996, compliant=False) for i in range(4)]
        )
        stats = compute_scenario_stats(results, compliance_threshold=0.997)
        assert stats.passed is False

    def test_apogee_statistics(self):
        """Mean, std, min, max of apogee are computed correctly."""
        apogees = [14000.0, 15000.0, 16000.0]
        results = [
            _make_sample_result(sample_id=i, apogee_m=a)
            for i, a in enumerate(apogees)
        ]
        stats = compute_scenario_stats(results, compliance_threshold=0.0)
        assert stats.apogee_mean == pytest.approx(np.mean(apogees))
        assert stats.apogee_std == pytest.approx(np.std(apogees))
        assert stats.apogee_min == pytest.approx(14000.0)
        assert stats.apogee_max == pytest.approx(16000.0)

    def test_landing_distance(self):
        """Landing distance is Euclidean from (0, 0)."""
        results = [
            _make_sample_result(sample_id=0, landing_north=300.0, landing_east=400.0),
        ]
        stats = compute_scenario_stats(results, compliance_threshold=0.0)
        assert stats.landing_dist_mean == pytest.approx(500.0)

    def test_worst_case_stability_margins(self):
        """sm_subsonic_min / sm_supersonic_min track the worst sample."""
        results = [
            _make_sample_result(sample_id=0, min_sm_subsonic=2.5, min_sm_supersonic=4.0),
            _make_sample_result(sample_id=1, min_sm_subsonic=1.1, min_sm_supersonic=2.8),
            _make_sample_result(sample_id=2, min_sm_subsonic=1.8, min_sm_supersonic=3.5),
        ]
        stats = compute_scenario_stats(results, compliance_threshold=0.0)
        assert stats.sm_subsonic_min == pytest.approx(1.1)
        assert stats.sm_supersonic_min == pytest.approx(2.8)

    def test_scenario_name_propagated(self):
        """The scenario name comes from the first result."""
        results = [_make_sample_result(scenario="ballistic")]
        stats = compute_scenario_stats(results, compliance_threshold=0.0)
        assert stats.scenario == "ballistic"
