"""Wind model: .npz ensemble loader, surface wind override, and runtime interpolation.

Public API
----------
list_wind_profile_files(path)                          →  list[Path]
load_wind_ensemble(path, num_samples, surface_wind)    →  WindEnsemble
interpolate_wind(altitude_m, wind_east, wind_north, h) →  (v_north, v_east)

WindEnsemble holds the full profile array (N × M) and the mean profile (M,).
interpolate_wind is Numba @njit compiled; pass a single profile row at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numba as nb

from config import SurfaceWindConfig


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class WindEnsemble:
    """Loaded and pre-processed wind profile ensemble.

    Attributes
    ----------
    altitude_m : (M,) float64
        Altitude grid in metres AGL, monotonically increasing.
    wind_east_ms : (N, M) float64
        Eastward wind component for each profile at each altitude.
        Surface override (if any) has already been applied.
    wind_north_ms : (N, M) float64
        Northward wind component. Same layout as wind_east_ms.
    mean_east_ms : (M,) float64
        Arithmetic mean eastward wind across all N profiles (for optimisation).
    mean_north_ms : (M,) float64
        Arithmetic mean northward wind across all N profiles.
    """
    altitude_m: np.ndarray
    wind_east_ms: np.ndarray
    wind_north_ms: np.ndarray
    mean_east_ms: np.ndarray
    mean_north_ms: np.ndarray


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def list_wind_profile_files(path: Path | str) -> list[Path]:
    """Return a sorted list of .npz wind profile files to process.

    Parameters
    ----------
    path:
        Either a path to a single ``.npz`` file, or a directory containing
        one or more ``.npz`` files.

    Returns
    -------
    list[Path]
        Single-element list when *path* is a file.
        Sorted list of all ``.npz`` files in the directory otherwise.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist, or is a directory containing no ``.npz`` files.
    ValueError
        If *path* is a file but does not have a ``.npz`` extension.
    """
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file, got: {p}")
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in directory: {p}")
        return files
    raise FileNotFoundError(f"Wind profiles path not found: {p}")


def load_wind_ensemble(
    path: Path | str,
    num_samples: int,
    surface_wind: SurfaceWindConfig | None = None,
) -> WindEnsemble:
    """Load wind profiles from a .npz file and return a WindEnsemble.

    Parameters
    ----------
    path:
        Path to the .npz file.
    num_samples:
        Required minimum number of profiles (N >= num_samples).
    surface_wind:
        If provided, replace the lower portion of every profile with the
        surface wind vector, blended linearly up to ``blend_height_m``.
        Pass ``None`` to leave all profiles unmodified.

    Raises
    ------
    ValueError
        If the .npz arrays have the wrong shape or N < num_samples.
    """
    data = np.load(path)

    # --- validate keys
    for key in ("altitude_m", "wind_east_ms", "wind_north_ms"):
        if key not in data:
            raise ValueError(f"wind profiles .npz missing array '{key}'")

    altitude_m: np.ndarray = np.ascontiguousarray(data["altitude_m"], dtype=np.float64)
    wind_east_ms: np.ndarray = np.ascontiguousarray(data["wind_east_ms"], dtype=np.float64)
    wind_north_ms: np.ndarray = np.ascontiguousarray(data["wind_north_ms"], dtype=np.float64)

    # --- validate shapes
    if altitude_m.ndim != 1:
        raise ValueError(f"altitude_m must be 1-D, got shape {altitude_m.shape}")
    M = altitude_m.shape[0]

    if wind_east_ms.ndim != 2 or wind_east_ms.shape[1] != M:
        raise ValueError(
            f"wind_east_ms must be (N, {M}), got {wind_east_ms.shape}"
        )
    if wind_north_ms.shape != wind_east_ms.shape:
        raise ValueError(
            f"wind_north_ms shape {wind_north_ms.shape} != wind_east_ms shape {wind_east_ms.shape}"
        )
    N = wind_east_ms.shape[0]
    if N < num_samples:
        raise ValueError(
            f"wind profile ensemble has {N} profiles but num_samples={num_samples}"
        )

    # --- validate monotonicity
    if not np.all(np.diff(altitude_m) > 0):
        raise ValueError("altitude_m must be monotonically increasing")

    # --- apply surface wind override (modifies copies, not original arrays)
    if surface_wind is not None:
        wind_east_ms, wind_north_ms = _apply_surface_wind(
            altitude_m, wind_east_ms, wind_north_ms, surface_wind
        )

    mean_east_ms = np.ascontiguousarray(wind_east_ms.mean(axis=0))
    mean_north_ms = np.ascontiguousarray(wind_north_ms.mean(axis=0))

    return WindEnsemble(
        altitude_m=altitude_m,
        wind_east_ms=wind_east_ms,
        wind_north_ms=wind_north_ms,
        mean_east_ms=mean_east_ms,
        mean_north_ms=mean_north_ms,
    )


def _apply_surface_wind(
    altitude_m: np.ndarray,
    wind_east_ms: np.ndarray,
    wind_north_ms: np.ndarray,
    cfg: SurfaceWindConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return new east/north arrays with the surface wind blended in."""
    bearing_rad = np.radians(cfg.bearing_deg)
    ov_east = cfg.speed_ms * np.sin(bearing_rad)
    ov_north = cfg.speed_ms * np.cos(bearing_rad)
    blend_h = cfg.blend_height_m  # guaranteed not None by caller

    east = wind_east_ms.copy()
    north = wind_north_ms.copy()

    for j, h in enumerate(altitude_m):
        if h <= 0.0:
            east[:, j] = ov_east
            north[:, j] = ov_north
        elif h < blend_h:
            alpha = h / blend_h          # 0 → 1 as h goes 0 → blend_h
            east[:, j] = (1.0 - alpha) * ov_east + alpha * east[:, j]
            north[:, j] = (1.0 - alpha) * ov_north + alpha * north[:, j]
        # else: h >= blend_h → unchanged

    return east, north


# ---------------------------------------------------------------------------
# Numba-compiled runtime interpolation
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def interpolate_wind(
    altitude_m: np.ndarray,
    wind_east: np.ndarray,
    wind_north: np.ndarray,
    h: float,
) -> tuple[float, float]:
    """Linearly interpolate wind at altitude *h* from a single profile.

    Parameters
    ----------
    altitude_m : (M,) float64
        Monotonically increasing altitude grid [m AGL].
    wind_east : (M,) float64
        Eastward wind component for this profile [m/s].
    wind_north : (M,) float64
        Northward wind component for this profile [m/s].
    h : float
        Query altitude [m AGL].

    Returns
    -------
    (v_north, v_east) : tuple[float, float]
        Wind components in NED convention [m/s].
        Below the lowest grid point the lowest value is returned (held constant).
    """
    M = altitude_m.shape[0]

    # Below the grid — hold constant at lowest altitude
    if h <= altitude_m[0]:
        return wind_north[0], wind_east[0]

    # Above the grid — hold constant at highest altitude
    if h >= altitude_m[M - 1]:
        return wind_north[M - 1], wind_east[M - 1]

    # Binary search for the bracketing interval
    lo = 0
    hi = M - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if altitude_m[mid] <= h:
            lo = mid
        else:
            hi = mid

    t = (h - altitude_m[lo]) / (altitude_m[hi] - altitude_m[lo])
    v_east = wind_east[lo] + t * (wind_east[hi] - wind_east[lo])
    v_north = wind_north[lo] + t * (wind_north[hi] - wind_north[lo])
    return v_north, v_east
