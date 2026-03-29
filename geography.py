"""Geospatial utilities: projections, polygons, and containment checking.

Loads danger-area and coastline polygons from GeoJSON, projects them to
NED metres, buffers the danger area inward, and provides fast containment
queries for trajectory points.

Specification references: §3.1 (flat-earth), §11.1 (acceptance), §14 (geometry).

Public API
----------
Coordinate conversion:
    ned_to_latlon         — NED metres → WGS-84 (lat, lon) degrees

Startup (Shapely, called once):
    load_polygon_ned      — GeoJSON → Shapely Polygon in NED metres
    buffer_danger_area    — inward buffer, simplify, largest-polygon extraction
    polygon_to_arrays     — extract exterior ring as NumPy arrays for Numba
    prepare_zone          — wrap Polygon in PreparedGeometry for point queries

Numba containment (called between integration phases):
    all_points_in_polygon — check every trajectory point, early exit

Post-sim queries (called once per sample):
    check_coastline            — landing point vs coastline polygon
    check_observation_coverage — landing point vs station radii
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import numba as nb
from shapely.geometry import Polygon, Point, MultiPolygon, shape
from shapely.prepared import prep as prepare_geometry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_EARTH = 6_371_000.0  # metres


# ---------------------------------------------------------------------------
# Startup helpers (Shapely)
# ---------------------------------------------------------------------------

def _load_geojson_polygon(path: Path) -> list[tuple[float, float]]:
    """Parse a GeoJSON FeatureCollection and return the first Polygon ring.

    Returns a list of (longitude, latitude) tuples forming the exterior ring.
    Raises ``ValueError`` if the file does not contain a valid Polygon.
    """
    with open(path) as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError(f"GeoJSON has no features: {path}")
        geom = features[0].get("geometry")
    elif data.get("type") == "Feature":
        geom = data.get("geometry")
    else:
        geom = data

    if geom is None or geom.get("type") != "Polygon":
        raise ValueError(
            f"Expected a Polygon geometry in {path}, "
            f"got {geom.get('type') if geom else None}"
        )

    coords = geom["coordinates"][0]  # exterior ring
    return [(lon, lat) for lon, lat in coords]


def _lonlat_to_ned(
    coords: list[tuple[float, float]],
    lat0: float,
    lon0: float,
) -> list[tuple[float, float]]:
    """Convert (lon, lat) degree pairs to (east, north) metres.

    Uses the flat-earth approximation from specification §3.1.
    Returns coordinates in Shapely's (x, y) = (east, north) order.
    """
    lat0_rad = math.radians(lat0)
    deg2m_north = R_EARTH * math.pi / 180.0
    deg2m_east = R_EARTH * math.cos(lat0_rad) * math.pi / 180.0

    ned = []
    for lon, lat in coords:
        north = (lat - lat0) * deg2m_north
        east = (lon - lon0) * deg2m_east
        ned.append((east, north))
    return ned


def ned_to_latlon(
    north: float, east: float,
    lat0: float, lon0: float,
) -> tuple[float, float]:
    """Convert NED metres to WGS-84 (latitude, longitude) degrees.

    Inverse of the flat-earth approximation (§3.1).

    Parameters
    ----------
    north, east : float
        Position in NED metres relative to the launch site.
    lat0, lon0 : float
        Launch-site latitude and longitude in degrees — the NED origin.

    Returns
    -------
    (latitude, longitude) : tuple[float, float]
        WGS-84 degrees.
    """
    lat0_rad = math.radians(lat0)
    deg2m_north = R_EARTH * math.pi / 180.0
    deg2m_east = R_EARTH * math.cos(lat0_rad) * math.pi / 180.0
    lat = lat0 + north / deg2m_north
    lon = lon0 + east / deg2m_east
    return lat, lon


def load_polygon_ned(
    path: Path, lat0: float, lon0: float,
) -> Polygon:
    """Load a GeoJSON file and return its polygon in NED metres.

    Parameters
    ----------
    path : Path
        Path to a GeoJSON file containing a FeatureCollection or Feature
        with a Polygon geometry.
    lat0, lon0 : float
        Launch-site latitude and longitude in degrees — the NED origin.

    Returns
    -------
    shapely.geometry.Polygon
        Polygon with coordinates in (east, north) metres.
    """
    lonlat_ring = _load_geojson_polygon(path)
    ned_ring = _lonlat_to_ned(lonlat_ring, lat0, lon0)
    return Polygon(ned_ring)


def buffer_danger_area(polygon: Polygon, buffer_distance: float) -> Polygon:
    """Apply an inward buffer, simplify, and extract the largest polygon.

    Parameters
    ----------
    polygon : Polygon
        Danger-area polygon in NED metres.
    buffer_distance : float
        Inward buffer in metres (from ``AcceptanceConfig.buffer_distance``).
        Also used as the simplification tolerance.  If zero, the polygon
        is returned unchanged.

    Returns
    -------
    Polygon
        The buffered and simplified danger-area polygon.
    """
    if buffer_distance == 0.0:
        return polygon

    buffered = polygon.buffer(-buffer_distance)
    buffered = buffered.simplify(buffer_distance)

    if buffered.is_empty:
        raise ValueError(
            f"Inward buffer of {buffer_distance} m collapsed the danger area "
            f"to an empty geometry — buffer_distance is too large."
        )

    if isinstance(buffered, MultiPolygon):
        buffered = max(buffered.geoms, key=lambda g: g.area)

    if not isinstance(buffered, Polygon):
        raise ValueError(
            f"Buffering produced a {type(buffered).__name__}, expected Polygon."
        )

    return buffered


def polygon_to_arrays(
    polygon: Polygon,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the exterior ring as contiguous float64 arrays.

    Returns
    -------
    (east, north) : tuple[np.ndarray, np.ndarray]
        1-D arrays of the polygon's exterior ring coordinates, suitable
        for passing to the Numba ``all_points_in_polygon`` function.
    """
    coords = np.asarray(polygon.exterior.coords, dtype=np.float64)
    east = np.ascontiguousarray(coords[:, 0])
    north = np.ascontiguousarray(coords[:, 1])
    return east, north


def prepare_zone(polygon: Polygon):
    """Wrap a polygon in a Shapely PreparedGeometry for fast point queries.

    Used for coastline containment checks (not the hot Numba path).
    """
    return prepare_geometry(polygon)


# ---------------------------------------------------------------------------
# Numba containment functions
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _point_in_polygon(
    ex: float, ny: float,
    poly_e: np.ndarray, poly_n: np.ndarray,
) -> bool:
    """Ray-casting point-in-polygon test.

    Casts a ray in the +east direction from (ex, ny) and counts edge
    crossings.  Odd count → inside.

    Parameters
    ----------
    ex, ny : float
        Point coordinates (east, north) in metres.
    poly_e, poly_n : np.ndarray
        Polygon exterior ring (east, north) coordinate arrays.
        The ring must be closed (first vertex == last vertex).
    """
    n = poly_e.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        ni = poly_n[i]
        nj = poly_n[j]
        if ((ni > ny) != (nj > ny)) and (
            ex < (poly_e[j] - poly_e[i]) * (ny - ni) / (nj - ni) + poly_e[i]
        ):
            inside = not inside
        j = i
    return inside


@nb.njit(cache=True)
def all_points_in_polygon(
    north: np.ndarray, east: np.ndarray,
    poly_e: np.ndarray, poly_n: np.ndarray,
) -> bool:
    """Check that every trajectory point lies inside the polygon.

    Returns ``False`` as soon as the first outside point is found.

    Parameters
    ----------
    north, east : np.ndarray
        1-D arrays of trajectory positions in NED metres.
    poly_e, poly_n : np.ndarray
        Buffered danger-area exterior ring arrays (from ``polygon_to_arrays``).
    """
    for k in range(north.shape[0]):
        if not _point_in_polygon(east[k], north[k], poly_e, poly_n):
            return False
    return True


# ---------------------------------------------------------------------------
# Post-sim queries
# ---------------------------------------------------------------------------

def check_coastline(
    north: float, east: float,
    coastline_prepared, mode: str,
) -> bool:
    """Check whether a landing point satisfies the coastline constraint.

    Parameters
    ----------
    north, east : float
        Landing position in NED metres.
    coastline_prepared : PreparedGeometry
        Coastline polygon wrapped via ``prepare_zone``.
    mode : str
        ``"sea"`` — landing must be **outside** the polygon (at sea).
        ``"land"`` — landing must be **inside** the polygon (on land).

    Returns
    -------
    bool
        ``True`` if the landing point satisfies the constraint.
    """
    pt = Point(east, north)
    inside = coastline_prepared.contains(pt)
    if mode == "sea":
        return not inside
    elif mode == "land":
        return inside
    else:
        raise ValueError(f"Unknown coastline_mode: {mode!r}")


def check_observation_coverage(
    north: float, east: float,
    station_norths: np.ndarray,
    station_easts: np.ndarray,
    station_radii: np.ndarray,
) -> bool:
    """Check that the landing point is within range of at least one station.

    Uses Euclidean distance, which is valid in the NED metre coordinate
    system.  The station arrays should include the automatic launch-site
    observation station.

    Parameters
    ----------
    north, east : float
        Landing position in NED metres.
    station_norths, station_easts, station_radii : np.ndarray
        1-D arrays for all observation stations (including launch site).
    """
    dn = station_norths - north
    de = station_easts - east
    dist_sq = dn * dn + de * de
    radii_sq = station_radii * station_radii
    return bool(np.any(dist_sq <= radii_sq))
