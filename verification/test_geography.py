"""Tests for geography.py — GeoJSON loading, buffering, containment, coastline.

Scope
-----
- GeoJSON loading and NED projection (against example data)
- Inward buffering, simplification, MultiPolygon handling
- Ray-casting point-in-polygon (Numba)
- all_points_in_polygon trajectory check
- Coastline check (sea / land modes)
- Observation-station coverage
- polygon_to_arrays round-trip
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from geography import (
    _load_geojson_polygon,
    _lonlat_to_ned,
    _point_in_polygon,
    all_points_in_polygon,
    buffer_danger_area,
    check_coastline,
    check_observation_coverage,
    load_polygon_ned,
    polygon_to_arrays,
    prepare_zone,
)

# ---------------------------------------------------------------------------
# Paths to example data
# ---------------------------------------------------------------------------
EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "example"
D802_PATH = EXAMPLE_DIR / "d802.geojson"
COASTLINE_PATH = EXAMPLE_DIR / "coastline.geojson"

# Launch site from example/simulation.yaml
LAUNCH_LAT = 58.6104700
LAUNCH_LON = -4.9434804


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_geojson(tmp_path: Path, name: str, coords: list) -> Path:
    """Write a minimal GeoJSON FeatureCollection to a temporary file."""
    data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {},
        }],
    }
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def _simple_square_geojson(tmp_path: Path, half_deg: float = 0.1) -> Path:
    """Create a GeoJSON square centred on the launch site."""
    coords = [
        [LAUNCH_LON - half_deg, LAUNCH_LAT - half_deg],
        [LAUNCH_LON + half_deg, LAUNCH_LAT - half_deg],
        [LAUNCH_LON + half_deg, LAUNCH_LAT + half_deg],
        [LAUNCH_LON - half_deg, LAUNCH_LAT + half_deg],
        [LAUNCH_LON - half_deg, LAUNCH_LAT - half_deg],
    ]
    return _write_geojson(tmp_path, "square.geojson", coords)


# ===========================================================================
# 1. GeoJSON loading & NED projection
# ===========================================================================

class TestGeoJSONLoading:
    def test_load_d802(self):
        """D802 danger area loads as a valid polygon with many vertices."""
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        assert poly.is_valid
        assert len(poly.exterior.coords) > 10

    def test_launch_site_near_origin(self):
        """Launch site projects to approximately (0, 0) in NED."""
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        centroid = poly.centroid
        # Launch site should be inside the danger area, and the centroid
        # should be within a reasonable distance of the origin.
        assert abs(centroid.x) < 20_000  # east, metres
        assert abs(centroid.y) < 20_000  # north, metres

    def test_launch_site_inside_d802(self):
        """The launch site (origin) should be inside D802."""
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        from shapely.geometry import Point
        assert poly.contains(Point(0.0, 0.0))

    def test_load_coastline(self):
        """Coastline GeoJSON loads as a valid polygon."""
        poly = load_polygon_ned(COASTLINE_PATH, LAUNCH_LAT, LAUNCH_LON)
        assert poly.is_valid
        assert len(poly.exterior.coords) > 10

    def test_invalid_geojson_no_features(self, tmp_path):
        data = {"type": "FeatureCollection", "features": []}
        p = tmp_path / "empty.geojson"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="no features"):
            load_polygon_ned(p, LAUNCH_LAT, LAUNCH_LON)

    def test_invalid_geojson_wrong_type(self, tmp_path):
        data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                "properties": {},
            }],
        }
        p = tmp_path / "line.geojson"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Expected a Polygon"):
            load_polygon_ned(p, LAUNCH_LAT, LAUNCH_LON)


# ===========================================================================
# 2. Buffering
# ===========================================================================

class TestBuffering:
    def test_zero_buffer_unchanged(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        buffered = buffer_danger_area(poly, 0.0)
        assert buffered.equals(poly)

    def test_buffer_reduces_area(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        buffered = buffer_danger_area(poly, 1000.0)
        assert buffered.area < poly.area

    def test_buffer_still_valid_polygon(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        buffered = buffer_danger_area(poly, 1000.0)
        assert isinstance(buffered, Polygon)
        assert buffered.is_valid

    def test_simplify_reduces_vertices(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        buffered = buffer_danger_area(poly, 1000.0)
        # Raw buffer typically adds many vertices; simplify should reduce them.
        raw_buffered = poly.buffer(-1000.0)
        if raw_buffered.geom_type == "MultiPolygon":
            raw_buffered = max(raw_buffered.geoms, key=lambda g: g.area)
        assert len(buffered.exterior.coords) <= len(raw_buffered.exterior.coords)

    def test_excessive_buffer_raises(self):
        """A buffer larger than the polygon should raise ValueError."""
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        with pytest.raises(ValueError, match="collapsed"):
            buffer_danger_area(poly, 1_000_000.0)

    def test_multipolygon_keeps_largest(self, tmp_path):
        """If buffering produces a MultiPolygon, the largest part is kept."""
        # Create a dumbbell shape: two squares connected by a thin bridge.
        # An inward buffer will sever the bridge, producing a MultiPolygon.
        coords = [
            [0, 0], [100, 0], [100, 80], [60, 80], [60, 90],
            [100, 90], [100, 200], [0, 200], [0, 90], [40, 90],
            [40, 80], [0, 80], [0, 0],
        ]
        poly = Polygon(coords)
        # Use a buffer that severs the 20 m wide bridge
        buffered = buffer_danger_area(poly, 15.0)
        assert isinstance(buffered, Polygon)


# ===========================================================================
# 3. Ray-casting point-in-polygon
# ===========================================================================

class TestPointInPolygon:
    @pytest.fixture()
    def d802_arrays(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        return polygon_to_arrays(poly)

    @pytest.fixture()
    def buffered_arrays(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        buffered = buffer_danger_area(poly, 1000.0)
        return polygon_to_arrays(buffered)

    def test_origin_inside(self, d802_arrays):
        """Launch site (0, 0) is inside D802."""
        pe, pn = d802_arrays
        assert _point_in_polygon(0.0, 0.0, pe, pn) is True

    def test_far_away_outside(self, d802_arrays):
        """A point far from the danger area is outside."""
        pe, pn = d802_arrays
        assert _point_in_polygon(1e6, 1e6, pe, pn) is False

    def test_origin_inside_buffered(self, buffered_arrays):
        """Launch site is still inside after a 1 km buffer."""
        pe, pn = buffered_arrays
        assert _point_in_polygon(0.0, 0.0, pe, pn) is True


# ===========================================================================
# 4. all_points_in_polygon (trajectory check)
# ===========================================================================

class TestAllPointsInPolygon:
    @pytest.fixture()
    def buffered_arrays(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        buffered = buffer_danger_area(poly, 1000.0)
        return polygon_to_arrays(buffered)

    def test_vertical_trajectory_inside(self, buffered_arrays):
        """A trajectory staying near the launch site is inside."""
        pe, pn = buffered_arrays
        north = np.zeros(100)
        east = np.zeros(100)
        assert all_points_in_polygon(north, east, pe, pn) is True

    def test_drifting_trajectory_outside(self, buffered_arrays):
        """A trajectory drifting far east exits the danger area."""
        pe, pn = buffered_arrays
        north = np.zeros(100)
        east = np.linspace(0, 1e6, 100)
        assert all_points_in_polygon(north, east, pe, pn) is False

    def test_single_point_inside(self, buffered_arrays):
        """Single-element arrays work correctly."""
        pe, pn = buffered_arrays
        north = np.array([0.0])
        east = np.array([0.0])
        assert all_points_in_polygon(north, east, pe, pn) is True


# ===========================================================================
# 5. Coastline check
# ===========================================================================

class TestCoastlineCheck:
    @pytest.fixture()
    def coastline_prepared(self):
        poly = load_polygon_ned(COASTLINE_PATH, LAUNCH_LAT, LAUNCH_LON)
        return prepare_zone(poly)

    def test_sea_mode_offshore(self, coastline_prepared):
        """A point far offshore (north-west into the sea) passes in sea mode."""
        # The launch site at Cape Wrath is near the coast.  A point far
        # to the north-west should be at sea (outside the land polygon).
        assert check_coastline(5000.0, -15000.0, coastline_prepared, "sea")

    def test_sea_mode_onshore(self, coastline_prepared):
        """A point well inland fails in sea mode."""
        # ~20 km south-east of launch site — should be on land.
        assert not check_coastline(-20000.0, 10000.0, coastline_prepared, "sea")

    def test_land_mode_inverts(self, coastline_prepared):
        """Land mode inverts the sea-mode result."""
        result_sea = check_coastline(-20000.0, 10000.0, coastline_prepared, "sea")
        result_land = check_coastline(-20000.0, 10000.0, coastline_prepared, "land")
        assert result_sea != result_land

    def test_invalid_mode_raises(self, coastline_prepared):
        with pytest.raises(ValueError, match="Unknown coastline_mode"):
            check_coastline(0.0, 0.0, coastline_prepared, "invalid")


# ===========================================================================
# 6. Observation coverage
# ===========================================================================

class TestObservationCoverage:
    def test_within_radius(self):
        sn = np.array([0.0])
        se = np.array([0.0])
        sr = np.array([5000.0])
        assert check_observation_coverage(100.0, 100.0, sn, se, sr)

    def test_outside_radius(self):
        sn = np.array([0.0])
        se = np.array([0.0])
        sr = np.array([100.0])
        assert not check_observation_coverage(5000.0, 5000.0, sn, se, sr)

    def test_multiple_stations_one_in_range(self):
        sn = np.array([0.0, 10000.0])
        se = np.array([0.0, 0.0])
        sr = np.array([100.0, 500.0])
        # Close to second station
        assert check_observation_coverage(10000.0, 200.0, sn, se, sr)

    def test_multiple_stations_none_in_range(self):
        sn = np.array([0.0, 10000.0])
        se = np.array([0.0, 0.0])
        sr = np.array([100.0, 100.0])
        assert not check_observation_coverage(50000.0, 50000.0, sn, se, sr)


# ===========================================================================
# 7. polygon_to_arrays round-trip
# ===========================================================================

class TestPolygonToArrays:
    def test_dtype_and_shape(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        east, north = polygon_to_arrays(poly)
        assert east.dtype == np.float64
        assert north.dtype == np.float64
        assert east.ndim == 1
        assert north.ndim == 1
        assert east.shape == north.shape

    def test_vertex_count_matches(self):
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        east, north = polygon_to_arrays(poly)
        assert east.shape[0] == len(poly.exterior.coords)

    def test_closed_ring(self):
        """First and last vertices should be identical (closed ring)."""
        poly = load_polygon_ned(D802_PATH, LAUNCH_LAT, LAUNCH_LON)
        east, north = polygon_to_arrays(poly)
        assert east[0] == east[-1]
        assert north[0] == north[-1]
