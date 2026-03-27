"""Tests for wind.py — .npz loader, surface override, and interpolation."""

from pathlib import Path

import numpy as np
import pytest

from config import SurfaceOverrideConfig
from wind import WindEnsemble, load_wind_ensemble, interpolate_wind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npz(tmp_path: Path, N: int = 10, M: int = 20, seed: int = 0) -> Path:
    """Write a valid synthetic wind .npz to tmp_path and return its path."""
    rng = np.random.default_rng(seed)
    altitude_m = np.linspace(0.0, 10_000.0, M)
    wind_east_ms = rng.uniform(-10.0, 10.0, (N, M))
    wind_north_ms = rng.uniform(-10.0, 10.0, (N, M))
    p = tmp_path / "wind.npz"
    np.savez(p, altitude_m=altitude_m, wind_east_ms=wind_east_ms,
             wind_north_ms=wind_north_ms)
    return p


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def test_loads_successfully(tmp_path):
    p = _make_npz(tmp_path)
    ens = load_wind_ensemble(p, num_samples=10)
    assert isinstance(ens, WindEnsemble)


def test_arrays_have_correct_shapes(tmp_path):
    p = _make_npz(tmp_path, N=15, M=30)
    ens = load_wind_ensemble(p, num_samples=10)
    assert ens.altitude_m.shape == (30,)
    assert ens.wind_east_ms.shape == (15, 30)
    assert ens.wind_north_ms.shape == (15, 30)
    assert ens.mean_east_ms.shape == (30,)
    assert ens.mean_north_ms.shape == (30,)


def test_arrays_are_float64(tmp_path):
    p = _make_npz(tmp_path)
    ens = load_wind_ensemble(p, num_samples=5)
    assert ens.altitude_m.dtype == np.float64
    assert ens.wind_east_ms.dtype == np.float64


def test_too_few_profiles_raises(tmp_path):
    p = _make_npz(tmp_path, N=5)
    with pytest.raises(ValueError, match="5 profiles"):
        load_wind_ensemble(p, num_samples=10)


def test_missing_key_raises(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, altitude_m=np.array([0.0, 1000.0]))
    with pytest.raises(ValueError, match="missing array"):
        load_wind_ensemble(p, num_samples=1)


def test_non_monotonic_altitude_raises(tmp_path):
    p = tmp_path / "bad.npz"
    alt = np.array([0.0, 500.0, 300.0, 1000.0])  # not monotonic
    east = np.ones((5, 4))
    north = np.ones((5, 4))
    np.savez(p, altitude_m=alt, wind_east_ms=east, wind_north_ms=north)
    with pytest.raises(ValueError, match="monotonically"):
        load_wind_ensemble(p, num_samples=5)


def test_mismatched_shape_raises(tmp_path):
    p = tmp_path / "bad.npz"
    alt = np.linspace(0, 1000, 10)
    east = np.ones((5, 10))
    north = np.ones((5, 8))   # wrong M
    np.savez(p, altitude_m=alt, wind_east_ms=east, wind_north_ms=north)
    with pytest.raises(ValueError):
        load_wind_ensemble(p, num_samples=5)


# ---------------------------------------------------------------------------
# Mean profile
# ---------------------------------------------------------------------------

def test_mean_profile_correct(tmp_path):
    rng = np.random.default_rng(42)
    M = 20
    alt = np.linspace(0, 5000, M)
    east = rng.uniform(-5, 5, (8, M))
    north = rng.uniform(-5, 5, (8, M))
    p = tmp_path / "w.npz"
    np.savez(p, altitude_m=alt, wind_east_ms=east, wind_north_ms=north)

    ens = load_wind_ensemble(p, num_samples=8)
    np.testing.assert_allclose(ens.mean_east_ms, east.mean(axis=0))
    np.testing.assert_allclose(ens.mean_north_ms, north.mean(axis=0))


# ---------------------------------------------------------------------------
# Surface override
# ---------------------------------------------------------------------------

def _override(speed: float, bearing: float, blend: float) -> SurfaceOverrideConfig:
    return SurfaceOverrideConfig(speed_ms=speed, bearing_deg=bearing,
                                 blend_height_m=blend)


def test_override_disabled_when_blend_none(tmp_path):
    """No override applied when blend_height_m is None."""
    p = _make_npz(tmp_path, seed=7)
    ens_no_ov = load_wind_ensemble(p, num_samples=5)
    cfg = SurfaceOverrideConfig(speed_ms=10.0, bearing_deg=90.0, blend_height_m=None)
    ens_ov = load_wind_ensemble(p, num_samples=5, surface_override=cfg)
    np.testing.assert_array_equal(ens_no_ov.wind_east_ms, ens_ov.wind_east_ms)


def test_override_at_zero_altitude(tmp_path):
    """At h=0, all profiles should equal the override vector."""
    M = 10
    alt = np.linspace(0.0, 5000.0, M)
    east = np.zeros((5, M))
    north = np.zeros((5, M))
    p = tmp_path / "w.npz"
    np.savez(p, altitude_m=alt, wind_east_ms=east, wind_north_ms=north)

    # bearing=0 (north): east component = 0, north component = speed
    cfg = _override(speed=8.0, bearing=0.0, blend=1000.0)
    ens = load_wind_ensemble(p, num_samples=5, surface_override=cfg)

    np.testing.assert_allclose(ens.wind_east_ms[:, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(ens.wind_north_ms[:, 0], 8.0, atol=1e-10)


def test_override_vector_east(tmp_path):
    """bearing=90° → purely eastward override vector."""
    M = 5
    alt = np.linspace(0.0, 1000.0, M)
    east = np.zeros((3, M))
    north = np.zeros((3, M))
    p = tmp_path / "w.npz"
    np.savez(p, altitude_m=alt, wind_east_ms=east, wind_north_ms=north)

    cfg = _override(speed=10.0, bearing=90.0, blend=500.0)
    ens = load_wind_ensemble(p, num_samples=3, surface_override=cfg)

    np.testing.assert_allclose(ens.wind_east_ms[:, 0], 10.0, atol=1e-10)
    np.testing.assert_allclose(ens.wind_north_ms[:, 0], 0.0, atol=1e-10)


def test_override_unchanged_above_blend(tmp_path):
    """Altitudes at or above blend_height_m must not be modified."""
    M = 20
    alt = np.linspace(0.0, 10_000.0, M)
    rng = np.random.default_rng(3)
    east = rng.uniform(-5, 5, (4, M))
    north = rng.uniform(-5, 5, (4, M))
    p = tmp_path / "w.npz"
    np.savez(p, altitude_m=alt, wind_east_ms=east, wind_north_ms=north)

    blend = 2000.0
    cfg = _override(speed=5.0, bearing=45.0, blend=blend)
    ens = load_wind_ensemble(p, num_samples=4, surface_override=cfg)

    above = alt >= blend
    np.testing.assert_array_equal(ens.wind_east_ms[:, above], east[:, above])
    np.testing.assert_array_equal(ens.wind_north_ms[:, above], north[:, above])


def test_override_blend_is_linear(tmp_path):
    """At the midpoint of the blend zone the mix should be 50/50."""
    # Use zero base wind so blended = (1-alpha)*override
    M = 3
    blend = 1000.0
    alt = np.array([0.0, blend / 2, blend])
    east = np.zeros((1, M))
    north = np.zeros((1, M))
    p = tmp_path / "w.npz"
    np.savez(p, altitude_m=alt, wind_east_ms=east, wind_north_ms=north)

    cfg = _override(speed=10.0, bearing=90.0, blend=blend)  # ov_east=10, ov_north=0
    ens = load_wind_ensemble(p, num_samples=1, surface_override=cfg)

    # At h=500 (alpha=0.5): blended_east = 0.5*10 + 0.5*0 = 5.0
    np.testing.assert_allclose(ens.wind_east_ms[0, 1], 5.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def test_interpolation_exact_grid_point():
    """Interpolation at a grid point must return that point's values."""
    alt = np.array([0.0, 1000.0, 2000.0, 5000.0])
    east = np.array([1.0, 3.0, 5.0, 9.0])
    north = np.array([0.0, -1.0, -2.0, -4.0])

    v_north, v_east = interpolate_wind(alt, east, north, 1000.0)
    assert v_east == pytest.approx(3.0)
    assert v_north == pytest.approx(-1.0)


def test_interpolation_midpoint():
    """Midpoint between two grid values should be the average."""
    alt = np.array([0.0, 1000.0, 2000.0])
    east = np.array([0.0, 10.0, 20.0])
    north = np.array([4.0, 8.0, 12.0])

    v_north, v_east = interpolate_wind(alt, east, north, 500.0)
    assert v_east == pytest.approx(5.0)
    assert v_north == pytest.approx(6.0)


def test_interpolation_below_grid_held_constant():
    """Below the lowest grid point, return the lowest value."""
    alt = np.array([100.0, 500.0, 1000.0])
    east = np.array([3.0, 5.0, 7.0])
    north = np.array([1.0, 2.0, 3.0])

    v_north, v_east = interpolate_wind(alt, east, north, 0.0)
    assert v_east == pytest.approx(3.0)
    assert v_north == pytest.approx(1.0)

    v_north2, v_east2 = interpolate_wind(alt, east, north, -500.0)
    assert v_east2 == pytest.approx(3.0)


def test_interpolation_above_grid_held_constant():
    """Above the highest grid point, return the highest value."""
    alt = np.array([0.0, 1000.0, 5000.0])
    east = np.array([1.0, 2.0, 6.0])
    north = np.array([0.0, 1.0, 3.0])

    v_north, v_east = interpolate_wind(alt, east, north, 8000.0)
    assert v_east == pytest.approx(6.0)
    assert v_north == pytest.approx(3.0)


def test_interpolation_returns_ned_order():
    """Return value is (v_north, v_east) — NED convention."""
    alt = np.array([0.0, 1000.0])
    east = np.array([7.0, 7.0])
    north = np.array([3.0, 3.0])
    v_north, v_east = interpolate_wind(alt, east, north, 500.0)
    assert v_north == pytest.approx(3.0)
    assert v_east == pytest.approx(7.0)
