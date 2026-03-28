"""Aerodynamic model: table lookup for force/moment coefficients.

Loads per-component RASAero II Aeroplot CSVs, fits C_Nα, assembles
whole-vehicle coefficient tables on a regular grid, and exposes
Numba @njit hot-loop lookup functions.

CSV format (one header row, then data)::

    Mach,Reynolds,AoA_deg,CA,CN,CP_m

``CP_m`` is in metres from the nosecone tip.  Reynolds is the full
Reynolds number (not ×10⁶).

Multiple files → per-component mode (used for roll and pitch/yaw damping).
One file → whole-vehicle mode; roll and damping tables are zeroed and a
warning is issued.

File naming: the fin component must have ``fin`` (case-insensitive) anywhere
in the filename stem.

Public API
----------
build_aero_model(aero_dir)  →  AeroModel

@njit functions — call directly in the dynamics hot loop:
    ca_at(mach_g, re_g, alpha_g, ca_tbl, M, Re, alpha_rad)
        → float [−]

    cn_cp_at(mach_g, re_g, alpha_g, cn_tbl, cp_tbl, M, Re, alpha_rad)
        → (CN [−], CP [m from nosecone])

    damping_sum_at(mach_g, re_g, cna_s, cna_cp_s, cna_cp2_s, M, Re, cg)
        → float  [Σᵢ C_Nα_i · (CP_i − CG)², m²/rad]

    cn_alpha_fins_at(mach_g, re_g, cna_fins, M, Re)
        → float  [C_Nα of fin component, 1/rad]

AeroModel bundles all arrays needed by the @njit functions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numba as nb


# ---------------------------------------------------------------------------
# Pre-computed bundle
# ---------------------------------------------------------------------------

@dataclass
class AeroModel:
    """All aerodynamic data pre-processed and ready for the simulation hot loop.

    Construct via :func:`build_aero_model`.
    """
    # Regular (structured) grid axes — sorted 1-D float64 arrays
    mach_grid: np.ndarray     # (NM,)  Mach number
    re_grid: np.ndarray       # (NR,)  Reynolds number
    alpha_grid: np.ndarray    # (NA,)  AoA in degrees

    # Whole-vehicle 3-D tables [NM, NR, NA]
    ca_table: np.ndarray      # axial force coefficient C_A
    cn_table: np.ndarray      # normal force coefficient C_N
    cp_table: np.ndarray      # centre of pressure from nosecone tip [m]

    # Per-component 2-D damping sums [NM, NR]
    # Damping sum = Σᵢ C_Nα_i · (CP_i − CG)²
    # Decomposed as  C − 2·CG·B + CG²·A  to avoid recomputing at each CG:
    cna_sum: np.ndarray       # A = Σᵢ C_Nα_i           [1/rad]
    cna_cp_sum: np.ndarray    # B = Σᵢ C_Nα_i · CP_i     [m/rad]
    cna_cp2_sum: np.ndarray   # C = Σᵢ C_Nα_i · CP_i²    [m²/rad]

    # Fin-only 2-D C_Nα [NM, NR] — for roll torques (Barrowman)
    cn_alpha_fins: np.ndarray  # [1/rad]

    # True when per-component data were loaded; False → single whole-vehicle file
    has_components: bool


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_aero_model(aero_dir: Path | str) -> AeroModel:
    """Load all ``*.csv`` files from *aero_dir* and return an :class:`AeroModel`.

    Parameters
    ----------
    aero_dir:
        Directory containing RASAero II Aeroplot CSV files.

    Raises
    ------
    FileNotFoundError
        If *aero_dir* contains no ``*.csv`` files.
    ValueError
        If a CSV cannot be parsed or has unexpected columns.
    """
    aero_dir = Path(aero_dir)
    csv_files = sorted(aero_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {aero_dir}")

    if len(csv_files) == 1:
        warnings.warn(
            f"Only one aeroplot CSV found ({csv_files[0].name}). "
            "Treating as whole-vehicle data. "
            "Roll torques and pitch/yaw damping will not be computed.",
            UserWarning,
            stacklevel=2,
        )
        return _build_single(csv_files[0])

    return _build_components(csv_files)


# ---------------------------------------------------------------------------
# Internal helpers — build
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> np.ndarray:
    """Load a CSV → contiguous (N, 6) float64: Mach, Re, AoA_deg, CA, CN, CP_m."""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Cannot read aeroplot CSV {path}: {exc}") from exc
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 6:
        raise ValueError(
            f"{path.name}: expected ≥6 columns "
            f"(Mach,Reynolds,AoA_deg,CA,CN,CP_m), got {data.shape[1]}"
        )
    return np.ascontiguousarray(data[:, :6], dtype=np.float64)


def _unique_axes(
    *datasets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted union of Mach, Re, and AoA_deg values across all datasets."""
    mach  = np.unique(np.concatenate([d[:, 0] for d in datasets]))
    re    = np.unique(np.concatenate([d[:, 1] for d in datasets]))
    alpha = np.unique(np.concatenate([d[:, 2] for d in datasets]))
    return (
        np.ascontiguousarray(mach,  dtype=np.float64),
        np.ascontiguousarray(re,    dtype=np.float64),
        np.ascontiguousarray(alpha, dtype=np.float64),
    )


def _resample_3d(
    src: np.ndarray,
    mach_g: np.ndarray,
    re_g: np.ndarray,
    alpha_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample CA, CN, CP from *src* onto the structured target grid.

    *src* must be a structured (complete Cartesian product) grid — this is
    always the case for RASAero II Aeroplot output.  Extrapolation is clamped
    to the source grid boundary (nearest-value).

    Parameters
    ----------
    src:
        (N, 6) source data: Mach, Re, AoA_deg, CA, CN, CP_m.
    mach_g, re_g, alpha_g:
        Target grid axes (sorted 1-D arrays).

    Returns
    -------
    ca, cn, cp:
        Each of shape ``(NM, NR, NA)``, C-contiguous float64.
    """
    m_src = np.unique(src[:, 0])
    r_src = np.unique(src[:, 1])
    a_src = np.unique(src[:, 2])
    NM_s, NR_s, NA_s = len(m_src), len(r_src), len(a_src)

    if NM_s * NR_s * NA_s != len(src):
        raise ValueError(
            f"Aeroplot data is not a structured grid: expected "
            f"{NM_s}×{NR_s}×{NA_s}={NM_s * NR_s * NA_s} rows, got {len(src)}.  "
            "Check the CSV for missing or duplicate (Mach, Reynolds, AoA) rows."
        )

    # Sort rows lexicographically (Mach, Re, AoA) and reshape to 3-D tables
    idx = np.lexsort((src[:, 2], src[:, 1], src[:, 0]))
    s = src[idx]

    out: list[np.ndarray] = []
    for col in (3, 4, 5):  # CA, CN, CP
        tbl = s[:, col].reshape(NM_s, NR_s, NA_s)
        out.append(_trilinear(m_src, r_src, a_src, tbl, mach_g, re_g, alpha_g))

    return out[0], out[1], out[2]


def _trilinear(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    tbl: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """Trilinear interpolation of *tbl* [NX, NY, NZ] onto the grid xi × yi × zi.

    Clamped at source boundaries.  Handles singleton axes (single grid value).

    Returns
    -------
    np.ndarray
        Shape ``(len(xi), len(yi), len(zi))``, C-contiguous float64.
    """
    def _idx_t(ax: np.ndarray, query: np.ndarray):
        if len(ax) == 1:
            return np.zeros(query.size, dtype=int), np.zeros(query.size)
        i = np.searchsorted(ax, query, side="right") - 1
        i = np.clip(i, 0, len(ax) - 2)
        t = np.clip((query - ax[i]) / (ax[i + 1] - ax[i]), 0.0, 1.0)
        return i, t

    mg, rg, ag = np.meshgrid(xi, yi, zi, indexing="ij")
    im, tm = _idx_t(xg, mg.ravel())
    ir, tr = _idx_t(yg, rg.ravel())
    ia, ta = _idx_t(zg, ag.ravel())

    im1 = np.minimum(im + 1, tbl.shape[0] - 1)
    ir1 = np.minimum(ir + 1, tbl.shape[1] - 1)
    ia1 = np.minimum(ia + 1, tbl.shape[2] - 1)

    result = (
          tbl[im,  ir,  ia ] * (1 - tm) * (1 - tr) * (1 - ta)
        + tbl[im1, ir,  ia ] *       tm * (1 - tr) * (1 - ta)
        + tbl[im,  ir1, ia ] * (1 - tm) *       tr * (1 - ta)
        + tbl[im1, ir1, ia ] *       tm *       tr * (1 - ta)
        + tbl[im,  ir,  ia1] * (1 - tm) * (1 - tr) *       ta
        + tbl[im1, ir,  ia1] *       tm * (1 - tr) *       ta
        + tbl[im,  ir1, ia1] * (1 - tm) *       tr *       ta
        + tbl[im1, ir1, ia1] *       tm *       tr *       ta
    )
    return np.ascontiguousarray(result.reshape(len(xi), len(yi), len(zi)), dtype=np.float64)


def _fit_cna_cp(
    cn_tbl: np.ndarray,   # [NM, NR, NA]
    cp_tbl: np.ndarray,   # [NM, NR, NA]
    alpha_g: np.ndarray,  # degrees, (NA,)
) -> tuple[np.ndarray, np.ndarray]:
    """Fit C_Nα [1/rad] and linear-regime CP [m] at every (M, Re) grid point.

    C_Nα is the least-squares slope of CN vs α through the origin using only
    data points where AoA ≤ 5°.  CP is the CN-weighted mean CP at those AoA.

    Returns
    -------
    cna : (NM, NR) float64
    cp_lin : (NM, NR) float64
    """
    NM, NR, _ = cn_tbl.shape
    fit_mask = alpha_g <= 5.0
    alpha_rad = np.deg2rad(alpha_g[fit_mask])

    if not fit_mask.any():
        # No data at α ≤ 5° — fall back to full range
        fit_mask = np.ones(len(alpha_g), dtype=bool)
        alpha_rad = np.deg2rad(alpha_g)

    denom_a = float(np.dot(alpha_rad, alpha_rad))

    cna    = np.empty((NM, NR), dtype=np.float64)
    cp_lin = np.empty((NM, NR), dtype=np.float64)

    for im in range(NM):
        for ir in range(NR):
            cn_sub = cn_tbl[im, ir, fit_mask]
            cp_sub = cp_tbl[im, ir, fit_mask]

            # C_Nα: least-squares slope CN = C_Nα · α through origin
            cna[im, ir] = (
                float(np.dot(alpha_rad, cn_sub) / denom_a)
                if denom_a > 1e-30 else 0.0
            )

            # CP at linear regime: CN-weighted mean
            w = np.abs(cn_sub)
            wsum = float(w.sum())
            cp_lin[im, ir] = (
                float((w * cp_sub).sum() / wsum)
                if wsum > 1e-30 else float(cp_sub.mean())
            )

    return (
        np.ascontiguousarray(cna,    dtype=np.float64),
        np.ascontiguousarray(cp_lin, dtype=np.float64),
    )


def _build_single(path: Path) -> AeroModel:
    src = _read_csv(path)
    mach_g, re_g, alpha_g = _unique_axes(src)
    ca, cn, cp = _resample_3d(src, mach_g, re_g, alpha_g)
    NM, NR = len(mach_g), len(re_g)
    zeros = np.zeros((NM, NR), dtype=np.float64)
    return AeroModel(
        mach_grid=mach_g, re_grid=re_g, alpha_grid=alpha_g,
        ca_table=ca, cn_table=cn, cp_table=cp,
        cna_sum=zeros, cna_cp_sum=zeros, cna_cp2_sum=zeros,
        cn_alpha_fins=zeros,
        has_components=False,
    )


def _build_components(csv_files: list[Path]) -> AeroModel:
    fins_paths = [p for p in csv_files if "fin" in p.stem.lower()]
    if not fins_paths:
        warnings.warn(
            "No file with 'fin' in its name found among aeroplot CSVs. "
            "Roll torques will not be computed.",
            UserWarning,
            stacklevel=3,
        )

    datasets = {p: _read_csv(p) for p in csv_files}
    mach_g, re_g, alpha_g = _unique_axes(*datasets.values())

    NM, NR, NA = len(mach_g), len(re_g), len(alpha_g)
    ca_tot = np.zeros((NM, NR, NA), dtype=np.float64)
    cn_tot = np.zeros((NM, NR, NA), dtype=np.float64)
    cp_num = np.zeros((NM, NR, NA), dtype=np.float64)  # Σ CN_i · CP_i

    cna_sum   = np.zeros((NM, NR), dtype=np.float64)
    cna_cp_s  = np.zeros((NM, NR), dtype=np.float64)
    cna_cp2_s = np.zeros((NM, NR), dtype=np.float64)
    cna_fins  = np.zeros((NM, NR), dtype=np.float64)

    for path, src in datasets.items():
        ca_i, cn_i, cp_i = _resample_3d(src, mach_g, re_g, alpha_g)
        ca_tot += ca_i
        cn_tot += cn_i
        cp_num += cn_i * cp_i

        cna_i, cp_lin_i = _fit_cna_cp(cn_i, cp_i, alpha_g)
        cna_sum   += cna_i
        cna_cp_s  += cna_i * cp_lin_i
        cna_cp2_s += cna_i * cp_lin_i ** 2

        if path in fins_paths:
            cna_fins += cna_i

    # Whole-vehicle CP: moment balance Σ(CN_i · CP_i) / Σ(CN_i)
    # At near-zero CN (α ≈ 0), fall back to C_Nα-weighted linear CP
    cna_sum_safe = np.where(cna_sum > 1e-30, cna_sum, 1.0)
    cp_lin_whole = (cna_cp_s / cna_sum_safe)[:, :, np.newaxis]  # broadcast to 3D

    cn_safe = np.where(cn_tot > 1e-9, cn_tot, np.nan)
    cp_tot = np.where(cn_tot > 1e-9, cp_num / cn_safe, cp_lin_whole)

    return AeroModel(
        mach_grid=mach_g, re_grid=re_g, alpha_grid=alpha_g,
        ca_table=np.ascontiguousarray(ca_tot, dtype=np.float64),
        cn_table=np.ascontiguousarray(cn_tot, dtype=np.float64),
        cp_table=np.ascontiguousarray(cp_tot, dtype=np.float64),
        cna_sum=np.ascontiguousarray(cna_sum,   dtype=np.float64),
        cna_cp_sum=np.ascontiguousarray(cna_cp_s,  dtype=np.float64),
        cna_cp2_sum=np.ascontiguousarray(cna_cp2_s, dtype=np.float64),
        cn_alpha_fins=np.ascontiguousarray(cna_fins, dtype=np.float64),
        has_components=True,
    )


# ---------------------------------------------------------------------------
# Numba hot-loop helpers
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _bisect(arr: np.ndarray, x: float) -> int:
    """Return lo such that arr[lo] ≤ x < arr[lo+1]; clamped to [0, n−2].

    Returns 0 for singleton arrays (n == 1).
    """
    n = arr.shape[0]
    if n == 1:
        return 0
    if x <= arr[0]:
        return 0
    if x >= arr[n - 1]:
        return n - 2
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if arr[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@nb.njit(cache=True)
def _frac(lo: float, hi: float, x: float) -> float:
    """Interpolation fraction; 0.0 when lo == hi (singleton axis), clamped [0,1]."""
    d = hi - lo
    if d == 0.0:
        return 0.0
    t = (x - lo) / d
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return t


@nb.njit(cache=True)
def _interp3(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    tbl: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> float:
    """Trilinear interpolation on a structured 3-D table ``tbl[NX, NY, NZ]``.

    Clamped at boundaries; handles singleton axes (single grid value).
    """
    ix = _bisect(xg, x)
    iy = _bisect(yg, y)
    iz = _bisect(zg, z)
    ix1 = ix + 1 if ix + 1 < xg.shape[0] else ix
    iy1 = iy + 1 if iy + 1 < yg.shape[0] else iy
    iz1 = iz + 1 if iz + 1 < zg.shape[0] else iz
    tx = _frac(xg[ix], xg[ix1], x)
    ty = _frac(yg[iy], yg[iy1], y)
    tz = _frac(zg[iz], zg[iz1], z)
    return (
        tbl[ix,  iy,  iz ] * (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
      + tbl[ix1, iy,  iz ] *        tx  * (1.0 - ty) * (1.0 - tz)
      + tbl[ix,  iy1, iz ] * (1.0 - tx) *        ty  * (1.0 - tz)
      + tbl[ix1, iy1, iz ] *        tx  *        ty  * (1.0 - tz)
      + tbl[ix,  iy,  iz1] * (1.0 - tx) * (1.0 - ty) *        tz
      + tbl[ix1, iy,  iz1] *        tx  * (1.0 - ty) *        tz
      + tbl[ix,  iy1, iz1] * (1.0 - tx) *        ty  *        tz
      + tbl[ix1, iy1, iz1] *        tx  *        ty  *        tz
    )


@nb.njit(cache=True)
def _interp2(
    xg: np.ndarray,
    yg: np.ndarray,
    tbl: np.ndarray,
    x: float,
    y: float,
) -> float:
    """Bilinear interpolation on a structured 2-D table ``tbl[NX, NY]``.

    Clamped at boundaries; handles singleton axes.
    """
    ix = _bisect(xg, x)
    iy = _bisect(yg, y)
    ix1 = ix + 1 if ix + 1 < xg.shape[0] else ix
    iy1 = iy + 1 if iy + 1 < yg.shape[0] else iy
    tx = _frac(xg[ix], xg[ix1], x)
    ty = _frac(yg[iy], yg[iy1], y)
    return (
        tbl[ix,  iy ] * (1.0 - tx) * (1.0 - ty)
      + tbl[ix1, iy ] *        tx  * (1.0 - ty)
      + tbl[ix,  iy1] * (1.0 - tx) *        ty
      + tbl[ix1, iy1] *        tx  *        ty
    )


# ---------------------------------------------------------------------------
# Numba hot-loop functions
# ---------------------------------------------------------------------------

_RAD_TO_DEG: float = 180.0 / 3.141592653589793


@nb.njit(cache=True)
def ca_at(
    mach_g: np.ndarray,
    re_g: np.ndarray,
    alpha_g: np.ndarray,
    ca_tbl: np.ndarray,
    M: float,
    Re: float,
    alpha_rad: float,
) -> float:
    """Axial force coefficient C_A at (M, Re, α).

    Parameters
    ----------
    alpha_rad:
        Total angle of attack in radians (non-negative).
    """
    return _interp3(mach_g, re_g, alpha_g, ca_tbl, M, Re, alpha_rad * _RAD_TO_DEG)


@nb.njit(cache=True)
def cn_cp_at(
    mach_g: np.ndarray,
    re_g: np.ndarray,
    alpha_g: np.ndarray,
    cn_tbl: np.ndarray,
    cp_tbl: np.ndarray,
    M: float,
    Re: float,
    alpha_rad: float,
) -> tuple[float, float]:
    """Normal force coefficient C_N and centre of pressure CP [m] at (M, Re, α).

    Parameters
    ----------
    alpha_rad:
        Total angle of attack in radians (non-negative).

    Returns
    -------
    (C_N, CP_m)
    """
    alpha_deg = alpha_rad * _RAD_TO_DEG
    cn = _interp3(mach_g, re_g, alpha_g, cn_tbl, M, Re, alpha_deg)
    cp = _interp3(mach_g, re_g, alpha_g, cp_tbl, M, Re, alpha_deg)
    return cn, cp


@nb.njit(cache=True)
def damping_sum_at(
    mach_g: np.ndarray,
    re_g: np.ndarray,
    cna_s: np.ndarray,
    cna_cp_s: np.ndarray,
    cna_cp2_s: np.ndarray,
    M: float,
    Re: float,
    cg: float,
) -> float:
    """Pitch/yaw damping sum Σᵢ C_Nα_i · (CP_i − CG)² [m²/rad] at (M, Re, CG).

    Multiply by ½ V A_ref to obtain C₂A.
    """
    A = _interp2(mach_g, re_g, cna_s,    M, Re)
    B = _interp2(mach_g, re_g, cna_cp_s, M, Re)
    C = _interp2(mach_g, re_g, cna_cp2_s, M, Re)
    val = C - 2.0 * cg * B + cg * cg * A
    return val if val > 0.0 else 0.0


@nb.njit(cache=True)
def cn_alpha_fins_at(
    mach_g: np.ndarray,
    re_g: np.ndarray,
    cna_fins: np.ndarray,
    M: float,
    Re: float,
) -> float:
    """Fin-component C_Nα [1/rad] at (M, Re).  Used for roll torques."""
    return _interp2(mach_g, re_g, cna_fins, M, Re)
