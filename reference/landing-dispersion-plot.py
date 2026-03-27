import os
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.path import Path
from matplotlib.patches import Ellipse, Polygon as MplPolygon, PathPatch, Circle
import contextily as cx
from pyproj import Transformer
from scipy.stats import chi2
from shapely.geometry import Polygon as ShapelyPolygon

# ── OS Maps API ───────────────────────────────────────────────────────────────
OS_API_KEY   = "wGs0Y4WVHmSuoqkPyFdpAlh7FKEvNSx4"
OS_TILE_STYLE = "Outdoor_3857"  # Outdoor_3857 | Light_3857 | Road_3857

# ── Locations (lat, lon WGS84) ────────────────────────────────────────────────
LAUNCH_SITE   = (58.6104700, -4.9434804)
DURNESS       = (58.56874,   -4.74763)
RANGE_CONTROL = (58.60215,   -4.77592)
ORIGIN_LAT, ORIGIN_LON = LAUNCH_SITE

# ── Map extent (km from launch site, true ground distance) ───────────────────
EXTENT_E  =  30.0
EXTENT_W  = -10.0
EXTENT_N  =  16.0
EXTENT_S  = -10.0
GRID_SPACING_KM = 5.0
MIN_BUFFER_RADIUS_KM = 1.0

# ── EGD802 Cape Wrath — boundary vertices (lat, lon) ─────────────────────────
# Used geojson.io to trace coastline waypoints between the SE corner and COAST_S
_EGD802_BOUNDARY = [
    (58.7500, -4.5000),   # NE
    (58.5764, -4.5000),   # SE — start of coastline
    (58.576313, -4.500711),
    (58.575786, -4.502286),
    (58.576521, -4.506808),
    (58.577092, -4.510241),
    (58.578318, -4.512252),
    (58.578890, -4.515555),
    (58.577095, -4.522738),
    (58.577421, -4.524930),
    (58.579783, -4.526434),
    (58.578967, -4.533327),
    (58.581749, -4.533588),
    (58.580929, -4.536891),
    (58.578967, -4.537075),
    (58.578978, -4.540396),
    (58.581335, -4.542945),
    (58.580520, -4.545803),
    (58.578168, -4.546326),
    (58.576877, -4.553206),
    (58.578655, -4.557980),
    (58.578020, -4.565301),
    (58.576089, -4.571325),
    (58.580217, -4.576297),
    (58.579652, -4.581411),
    (58.578269, -4.586567),
    (58.576402, -4.592189),
    (58.574137, -4.591024),
    (58.572859, -4.596212),
    (58.569240, -4.599718),
    (58.562822, -4.599162),
    (58.559676, -4.598252),
    (58.557428, -4.601904),
    (58.555171, -4.603887),
    (58.552424, -4.602667),
    (58.548637, -4.598103),
    (58.546192, -4.595404),
    (58.538141, -4.594798),
    (58.537258, -4.595700),
    (58.534438, -4.593735),
    (58.533554, -4.597502),
    (58.533152, -4.601435),
    (58.531375, -4.602658),
    (58.526772, -4.602984),
    (58.526281, -4.606032),
    (58.523762, -4.610624),
    (58.523861, -4.618961),
    (58.523860, -4.627359),
    (58.522968, -4.627669),
    (58.520295, -4.633545),
    (58.520700, -4.634940),
    (58.521349, -4.638507),
    (58.521188, -4.640986),
    (58.516583, -4.649305),
    (58.511134, -4.652772),
    (58.508938, -4.658874),
    (58.507150, -4.661246),
    (58.505124, -4.660645),
    (58.498456, -4.661240),
    (58.499737, -4.668096),
    (58.495711, -4.670042),
    (58.496799, -4.661898),
    (58.494860, -4.661287),
    (58.489761, -4.662911),
    (58.486120, -4.661960),
    (58.483218, -4.663882),
    (58.482092, -4.666964),
    (58.483545, -4.673179),
    (58.482902, -4.677781),
    (58.477500, -4.684181),
    (58.477497, -4.687369),
    (58.477492, -4.692091),
    (58.478057, -4.694994),
    (58.477083, -4.695759),
    (58.474814, -4.694994),
    (58.469929, -4.698810),
    (58.467893, -4.702775),
    (58.465128, -4.708003),
    (58.464888, -4.712886),
    (58.463112, -4.715495),
    (58.463844, -4.723278),
    (58.458033, -4.726378),
    (58.451618, -4.731060),
    (58.446930, -4.738270),
    (58.449430, -4.749367),
    (58.448866, -4.760035),
    (58.450867, -4.758632),
    (58.451420, -4.755041),
    (58.453843, -4.752064),
    (58.456819, -4.753641),
    (58.459748, -4.752388),
    (58.459216, -4.749072),
    (58.458041, -4.744341),
    (58.462390, -4.746063),
    (58.467823, -4.741122),
    (58.475020, -4.729210),
    (58.481756, -4.723195),
    (58.489796, -4.716911),
    (58.497230, -4.710927),
    (58.500101, -4.704297),
    (58.504691, -4.693750),
    (58.508053, -4.691227),
    (58.508059, -4.687888),
    (58.512626, -4.682259),
    (58.515210, -4.683363),
    (58.525576, -4.669705),
    (58.525888, -4.667412),
    (58.524364, -4.665271),
    (58.524926, -4.662966),
    (58.527584, -4.660497),
    (58.530942, -4.655428),
    (58.534512, -4.651327),
    (58.540476, -4.653058),
    (58.544204, -4.654316),
    (58.546524, -4.657331),
    (58.549840, -4.657000),
    (58.551167, -4.653174),
    (58.552410, -4.656672),
    (58.551415, -4.659856),
    (58.552659, -4.663188),
    (58.551166, -4.663511),
    (58.550337, -4.666532),
    (58.551415, -4.667640),
    (58.551166, -4.673360),
    (58.552493, -4.680506),
    (58.552907, -4.678440),
    (58.554068, -4.680185),
    (58.555810, -4.685426),
    (58.556972, -4.690986),
    (58.556442, -4.695321),
    (58.555980, -4.702485),
    (58.557731, -4.704524),
    (58.559404, -4.707047),
    (58.560916, -4.709897),
    (58.563270, -4.709422),
    (58.563939, -4.706560),
    (58.566462, -4.704435),
    (58.566871, -4.708163),
    (58.569020, -4.714050),
    (58.570146, -4.717191),
    (58.570454, -4.720926),
    (58.570149, -4.726628),
    (58.568721, -4.731346),
    (58.566273, -4.734493),
    (58.569848, -4.740955),
    (58.570766, -4.740365),
    (58.573114, -4.742716),
    (58.575363, -4.742124),
    (58.575669, -4.744673),
    (58.579966, -4.740546),
    (58.582627, -4.742505),
    (58.583651, -4.741915),
    (58.584469, -4.743681),
    (58.583239, -4.745056),
    (58.582624, -4.751922),
    (58.584058, -4.755453),
    (58.587128, -4.754865),
    (58.587844, -4.762518),
    (58.588765, -4.764480),
    (58.590401, -4.766442),
    (58.592242, -4.765265),
    (58.593571, -4.767227),
    (58.596025, -4.766246),
    (58.597967, -4.765069),
    (58.599808, -4.770170),
    (58.601546, -4.771348),
    (58.604919, -4.768993),
    (58.604510, -4.770759),
    (58.604306, -4.774487),
    (58.601750, -4.778804),
    (58.600217, -4.779981),
    (58.601239, -4.784298),
    (58.599501, -4.786260),
    (58.600421, -4.789792),
    (58.600012, -4.792736),
    (58.599297, -4.790970),
    (58.597456, -4.788026),
    (58.595514, -4.787045),
    (58.594696, -4.779785),
    (58.591935, -4.777038),
    (58.588356, -4.771348),
    (58.585697, -4.769385),
    (58.583780, -4.765864),
    (58.579773, -4.765869),
    (58.576796, -4.768030),
    (58.576796, -4.771758),
    (58.577617, -4.776074),
    (58.577621, -4.782943),
    (58.577214, -4.788243),
    (58.577507, -4.792353),
    (58.576152, -4.795662),
    (58.578406, -4.795472),
    (58.580428, -4.798783),
    (58.579297, -4.799558),
    (58.577572, -4.797610),
    (58.573475, -4.798573),
    (58.570201, -4.799341),
    (58.568253, -4.801082),
    (58.565910, -4.802650),
    (58.564074, -4.805576),
    (58.560714, -4.804423),
    (58.558985, -4.798779),
    (58.555922, -4.796445),
    (58.553468, -4.789608),
    (58.552621, -4.782329),
    (58.550840, -4.777941),
    (58.548123, -4.778359),
    (58.544969, -4.778987),
    (58.541268, -4.781080),
    (58.538101, -4.783793),
    (58.536467, -4.788087),
    (58.5333, -4.7911),    # COAST_S — end of coastline
    (58.5333, -5.0000),   # SW
    (58.7500, -5.0000),   # NW
]
MIN_BUFFER_RADIUS_KM = 1.0
_CHI2_95      = np.sqrt(chi2.ppf(0.95, df=2))
_TO_WM        = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(ORIGIN_LAT))
_OS_TILE_URL  = (
    f"https://api.os.uk/maps/raster/v1/zxy/"
    f"{OS_TILE_STYLE}/{{z}}/{{x}}/{{y}}.png?key={OS_API_KEY}"
)
_TILE_CACHE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map-tile-cache")

def _km_to_wm(north_km: float, east_km: float) -> tuple[float, float]:
    """True-ground km offset from origin → Web Mercator (x, y)."""
    return _TO_WM.transform(
        ORIGIN_LON + (east_km  * 1000.0) / _M_PER_DEG_LON,
        ORIGIN_LAT + (north_km * 1000.0) / _M_PER_DEG_LAT,
    )

def _latlon_to_km(lat: float, lon: float) -> tuple[float, float]:
    """WGS84 → (north_km, east_km) relative to origin."""
    return (
        (lat - ORIGIN_LAT) * _M_PER_DEG_LAT / 1000.0,
        (lon - ORIGIN_LON) * _M_PER_DEG_LON / 1000.0,
    )

def _extent_wm() -> tuple[float, float, float, float]:
    """Map corners in Web Mercator: (xmin, xmax, ymin, ymax)."""
    xmin, ymin = _km_to_wm(EXTENT_S, EXTENT_W)
    xmax, ymax = _km_to_wm(EXTENT_N, EXTENT_E)
    return xmin, xmax, ymin, ymax

class NEDMap:
    """
    Accumulates NE point sets and 95%-confidence PCA ellipses,
    then renders them on an OS Maps basemap.

    Axes: X → km East, Y ↑ km North  (true ground distance from launch site).
    """
    _COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def __init__(self):
        self._datasets:     list[dict] = []
        self._danger_areas: list[dict] = []
        self._markers:      list[dict] = []
        self._circles:      list[dict] = []
        self._colour_idx = 0
        os.makedirs(_TILE_CACHE, exist_ok=True)
        cx.set_cache_dir(_TILE_CACHE)
    
    def add_to_map(self, points_ne: np.ndarray, label: str,
                   colour: str | None = None) -> None:
        """Add a (N, 2) [north_km, east_km] point cloud and its fitted ellipse."""
        points_ne = np.asarray(points_ne, dtype=float)
        if colour is None:
            colour = self._COLOURS[self._colour_idx % len(self._COLOURS)]
            self._colour_idx += 1
        self._datasets.append(dict(
            points=points_ne, ellipse=self._fit_ellipse(points_ne),
            label=label, colour=colour,
        ))

    def add_marker(self, lat: float, lon: float, label: str,
                   marker: str = "o", colour: str = "black",
                   size: float = 10.0, filled: bool = True) -> None:
        """Add a map marker at a WGS84 position."""
        north_km, east_km = _latlon_to_km(lat, lon)
        self._markers.append(dict(
            north_km=north_km, east_km=east_km, label=label,
            marker=marker, colour=colour, size=size, filled=filled,
        ))

    def add_circle(self, lat: float, lon: float, radius_km: float,
                   label: str, colour: str = "black",
                   fill_alpha: float = 0.1) -> None:
        """
        Add a circle of a given radius centred on a WGS84 position.

        The circle is drawn as a true ground-distance radius by sampling
        points around the circumference and projecting each to Web Mercator,
        so it remains accurate despite the Mercator distortion at this latitude.

        Parameters
        ----------
        lat, lon      : Centre of the circle in WGS84 decimal degrees.
        radius_km     : Radius in kilometres (true ground distance).
        label         : Legend label.
        colour        : Edge and fill colour.
        fill_alpha    : Opacity of the filled interior (0 = transparent).
        """
        north_km, east_km = _latlon_to_km(lat, lon)
        self._circles.append(dict(
            north_km=north_km, east_km=east_km,
            radius_km=radius_km, label=f"{label} ({radius_km} km)",
            colour=colour, fill_alpha=fill_alpha,
        ))

    def add_danger_area(self, ref_code: str, offset_km: float) -> None:
        boundaries = {"EGD802": _EGD802_BOUNDARY}
        if ref_code not in boundaries:
            raise ValueError(f"Unknown danger area '{ref_code}'. "
                            f"Supported: {list(boundaries)}")
        self._danger_areas.append(dict(
            latlon=boundaries[ref_code], offset_km=offset_km,
            buffer_label=f"Buffer Zone ({offset_km:.0f} km)", ref_code=ref_code,
        ))

    def display(self) -> None:
        """Render everything on the OS basemap."""
        origin_x, origin_y = _km_to_wm(0.0, 0.0)
        xmin, xmax, ymin, ymax = _extent_wm()

        ar   = (EXTENT_E - EXTENT_W) / (EXTENT_N - EXTENT_S)
        BASE = 9
        fig, ax = plt.subplots(
            figsize=((BASE, BASE / ar) if ar >= 1 else (BASE * ar, BASE)),
            constrained_layout=True,
        )
        ax.set_aspect("equal")
        legend_handles = []

        self._draw_danger_areas(ax, legend_handles)
        self._draw_circles(ax, legend_handles)
        self._draw_ellipses(ax, legend_handles)
        self._draw_markers(ax, legend_handles)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        fig.canvas.draw()
        cx.add_basemap(ax, crs="EPSG:3857", source=_OS_TILE_URL, zorder=1, zoom_adjust=0)

        wm_per_km_e = _km_to_wm(0.0, 1.0)[0] - origin_x
        wm_per_km_n = _km_to_wm(1.0, 0.0)[1] - origin_y

        # Ticks anchored at 0,0 and spaced every GRID_SPACING_KM
        x_ticks_km = np.arange(
            math.ceil(EXTENT_W / GRID_SPACING_KM) * GRID_SPACING_KM,
            EXTENT_E + GRID_SPACING_KM,
            GRID_SPACING_KM,
        )
        y_ticks_km = np.arange(
            math.ceil(EXTENT_S / GRID_SPACING_KM) * GRID_SPACING_KM,
            EXTENT_N + GRID_SPACING_KM,
            GRID_SPACING_KM,
        )

        ax.set_xticks([origin_x + km * wm_per_km_e for km in x_ticks_km])
        ax.set_xticklabels([f"{km:.0f}" for km in x_ticks_km])
        ax.set_yticks([origin_y + km * wm_per_km_n for km in y_ticks_km])
        ax.set_yticklabels([f"{km:.0f}" for km in y_ticks_km])

        ax.set_xlabel("East (km)", fontsize=12, labelpad=8)
        ax.set_ylabel("North (km)", fontsize=12, labelpad=8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7, zorder=2)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
        plt.show()
    
    def _fit_ellipse(self, points_ne: np.ndarray) -> dict:
        if points_ne.ndim != 2 or points_ne.shape[1] != 2:
            raise ValueError("points_ne must be (N, 2): [north_km, east_km]")
        north, east = points_ne[:, 0], points_ne[:, 1]
        vals, vecs  = np.linalg.eigh(np.cov(np.vstack([east, north])))
        order       = np.argsort(vals)[::-1]
        vals, vecs  = vals[order], vecs[:, order]
        return dict(
            center_n  = north.mean(),
            center_e  = east.mean(),
            semi_a    = _CHI2_95 * np.sqrt(vals[0]),
            semi_b    = _CHI2_95 * np.sqrt(vals[1]),
            angle_deg = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])),
        )

    def _draw_danger_areas(self, ax, handles):
        for da in self._danger_areas:
            wm_outer   = np.array(
                [_TO_WM.transform(lon, lat) for lat, lon in da["latlon"]]
            )
            outer_poly = ShapelyPolygon(wm_outer)
            offset_wm  = da["offset_km"] * 1000.0 / math.cos(math.radians(ORIGIN_LAT))
            smooth_wm  = MIN_BUFFER_RADIUS_KM * 1000.0 / math.cos(math.radians(ORIGIN_LAT))
            inner_poly = outer_poly.buffer(-offset_wm)
            inner_poly = inner_poly.buffer(-smooth_wm).buffer(+smooth_wm)

            if inner_poly.is_empty:
                warnings.warn(f"Buffer offset {da['offset_km']} km too large "
                            f"for '{da['ref_code']}' — skipping.", stacklevel=2)
                continue

            # Buffer ring fill with hole using Path
            ring = outer_poly.difference(inner_poly)
            for part in ([ring] if ring.geom_type == "Polygon" else list(ring.geoms)):
                outer_c = np.array(part.exterior.coords)
                inner_c = np.array(list(part.interiors)[0].coords)
                verts = np.concatenate([outer_c, inner_c])
                codes = np.concatenate([
                    [Path.MOVETO] + [Path.LINETO] * (len(outer_c) - 2) + [Path.CLOSEPOLY],
                    [Path.MOVETO] + [Path.LINETO] * (len(inner_c) - 2) + [Path.CLOSEPOLY],
                ])
                ax.add_patch(PathPatch(
                    Path(verts, codes),
                    facecolor="none", edgecolor="red", linewidth=0, zorder=3,
                    hatch="....",
                ))

            # Outer edge
            ax.plot(*outer_poly.exterior.xy,
                    color="red", linewidth=1.0, linestyle="-", zorder=5)

            handles.append(mpatches.Patch(
                facecolor="none", edgecolor="red", linewidth=0,
                hatch="....", label=da["buffer_label"]
            ))

    def _draw_circles(self, ax, handles):
        """Draw all registered circles in Web Mercator space."""
        N_PTS = 360  # number of points used to approximate the circumference
        for circ in self._circles:
            # Build the circumference by stepping around bearings, computing
            # each point's true-ground offset, then projecting to Web Mercator.
            bearings = np.linspace(0, 2 * math.pi, N_PTS, endpoint=False)
            wm_pts = np.array([
                _km_to_wm(
                    circ["north_km"] + circ["radius_km"] * math.cos(b),
                    circ["east_km"]  + circ["radius_km"] * math.sin(b),
                )
                for b in bearings
            ])
            # Close the ring
            wm_pts = np.vstack([wm_pts, wm_pts[0]])

            ax.fill(wm_pts[:, 0], wm_pts[:, 1],
                    color=circ["colour"], alpha=circ["fill_alpha"],
                    zorder=4)
            
            handles.append(mpatches.Patch(
                    facecolor=circ["colour"], edgecolor="none", linewidth=0,
                    label=circ["label"], alpha=circ["fill_alpha"]
            ))

    def _draw_ellipses(self, ax, handles):
        ellipse_styles = ["-", "--", "-.", ":"]

        for i, ds in enumerate(self._datasets):
            el, c = ds["ellipse"], ds["colour"]
            style = ellipse_styles[i % len(ellipse_styles)]
            ec, nc = _km_to_wm(el["center_n"], el["center_e"])
            wm_a = abs(_km_to_wm(el["center_n"], el["center_e"] + el["semi_a"])[0] - ec)
            wm_b = abs(_km_to_wm(el["center_n"] + el["semi_b"], el["center_e"])[1] - nc)
            for alpha, fc in [(0.15, c), (1.0, "none")]:
                ax.add_patch(Ellipse(
                    xy=(ec, nc), width=2*wm_a, height=2*wm_b,
                    angle=el["angle_deg"], edgecolor=c, facecolor="none",
                    linewidth=2, alpha=alpha, zorder=7, linestyle=style,
                ))
            handles.append(
                mpatches.Patch(facecolor="none", edgecolor=c, alpha=0.6,
                            linestyle=style, linewidth=2, label=ds["label"]))
    
    def _draw_markers(self, ax, handles):
        for mk in self._markers:
            mx, my = _km_to_wm(mk["north_km"], mk["east_km"])
            fc = mk["colour"] if mk["filled"] else "none"
            ax.plot(mx, my, marker=mk["marker"], color=mk["colour"],
                    markerfacecolor=fc, markeredgecolor=mk["colour"],
                    markersize=mk["size"], markeredgewidth=2.5,
                    linestyle="None", zorder=10)
            handles.append(mlines.Line2D(
                [], [], marker=mk["marker"], color="none",
                markerfacecolor=fc, markeredgecolor=mk["colour"],
                markeredgewidth=2.5, markersize=mk["size"],
                linestyle="None", label=mk["label"],
            ))

def generate_test_points(
    offset_east_km: float, offset_north_km: float,
    spread_km: float = 0.2, n_points: int = 100,
    spread_ratio: float = 1.0, angle_deg: float = 0.0,
    rng_seed: int | None = None,
) -> np.ndarray:
    """Gaussian point cloud centred at (offset_north_km, offset_east_km).
    Returns (N, 2) array [north_km, east_km]."""
    rng     = np.random.default_rng(rng_seed)
    local   = rng.standard_normal((n_points, 2)) * \
              np.array([spread_km, spread_km * spread_ratio])
    theta   = np.radians(angle_deg)
    R       = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
    rotated = local @ R.T
    return np.column_stack([rotated[:, 1] + offset_north_km,
                            rotated[:, 0] + offset_east_km])


# NOTES:
# 1. Danger area info here: https://www.aurora.nats.co.uk/htmlAIP/Publications/2026-02-19-AIRAC/html/index-en-GB.html
# 2. Airspace map here: https://www.aurora.nats.co.uk/htmlAIP/Publications/2023-09-07-AIRAC/graphics/359608.pdf
# 3. Used this to trace the coastline (required for some danger areas): https://geojson.io/
# 4. All data is BS atm.
if __name__ == "__main__":
    clouds = [
        generate_test_points( 0.25,  1, spread_km=0.1, spread_ratio=0.4, angle_deg= 30,  n_points=120, rng_seed=0),
        generate_test_points( 0.75,  2.5, spread_km=0.4,  spread_ratio=0.9, angle_deg=  0,  n_points=80,  rng_seed=1),
        generate_test_points( 0.5,  2.0, spread_km=0.2,  spread_ratio=0.9, angle_deg=  0,  n_points=80,  rng_seed=1),
        generate_test_points( 3.0,  8.2, spread_km=0.5,  spread_ratio=0.8, angle_deg=-45,  n_points=500, rng_seed=2),
    ]
    labels  = ["Ballistic Descent", "Nominal Descent", "Drogue Descent", "Main Descent"]
    colours = ["black", "green", "orange", "red"]

    ned_map = NEDMap()

    ned_map.add_danger_area("EGD802", offset_km=1)
    ned_map.add_marker(*LAUNCH_SITE,   label="Launch Site",        marker="x", colour="black", size=8, filled=False)
    ned_map.add_marker(*RANGE_CONTROL, label="MOD Range Control",  marker="s", colour="black", size=5)
    ned_map.add_marker(*DURNESS,       label="Durness",            marker="o", colour="black", size=5)

    ned_map.add_circle(*LAUNCH_SITE,   radius_km=5,  label="Visibility Coverage", colour="purple", fill_alpha=0.1)
    ned_map.add_circle(*RANGE_CONTROL, radius_km=10, label="Marine Radar Coverage", colour="black", fill_alpha=0.1)

    for cloud, label, colour in zip(clouds, labels, colours):
        ned_map.add_to_map(cloud, label=label, colour=colour)

    ned_map.display()