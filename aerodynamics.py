"""Aerodynamic model: table lookup for force/moment coefficients.

Loads per-component RASAero II Aeroplot CSVs, assembles per-component and
whole-vehicle tables on a regular grid, and exposes Numba @njit hot-loop
functions.

CSV format (one header row, then data).  8-column (recommended)::

    Mach,Reynolds,AoA_deg,CA_off,CA_on,CN,CP_m,CN_alpha_per_rad

or 7-column (no CN_alpha_per_rad)::

    Mach,Reynolds,AoA_deg,CA_off,CA_on,CN,CP_m

``CP_m`` is in metres from the nosecone tip.  Reynolds is the full
Reynolds number (not ×10⁶).

Multiple files → per-component mode.  Forces and moments are computed
using local angles of attack at each component's aerodynamic centre,
which naturally captures both restoring and damping effects.
One file → whole-vehicle mode; per-component computation and roll are
disabled and a warning is issued.

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

    aero_forces_moments(
        mach_g, re_g, alpha_g,
        ca_tbl_off, ca_tbl_on, power_on,
        cn_tbl, cp_tbl, cn_comp, cp_comp, has_components, cn_alpha_comp,
        M, Re, rho, V, A_ref,
        u_rel, v_rel, w_rel, q_rate, r_rate, cg,
    ) → (F_x, F_y, F_z, tau_pitch, tau_yaw, cp_whole)

AeroModel bundles all arrays needed by the @njit functions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import math

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
    ca_table_off: np.ndarray  # axial force coefficient C_A, power-off (always the sum)
    ca_table_on: np.ndarray   # axial force coefficient C_A, power-on  (always the sum)
    cn_table: np.ndarray      # C_N — used in single-file fallback; zeros in per-component mode
    cp_table: np.ndarray      # C_P — used in single-file fallback; zeros in per-component mode

    # Per-component 4-D tables [N_comp, NM, NR, NA]
    # N_comp = 0 dummy shape in single-file mode (has_components=False)
    cn_comp: np.ndarray       # per-component normal force coefficient
    cp_comp: np.ndarray       # per-component centre of pressure [m from nosecone]

    # Per-component 3-D C_Nα [N_comp, NM, NR] — for damping and roll torques
    cn_alpha_comp: np.ndarray  # [1/rad]

    # Index into cn_alpha_comp for the fin component (-1 if none found)
    fin_comp_idx: int

    # Component names (stems of the CSV files, in the same order as cn_comp axis 0)
    comp_names: list[str]

    # True when per-component data were loaded; False → single whole-vehicle file
    has_components: bool


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_aero_model(
    aero_dir: Path | str,
    fins_override: Path | None = None,
) -> AeroModel:
    """Load aeroplot CSV(s) and return an :class:`AeroModel`.

    Parameters
    ----------
    aero_dir:
        Either a path to a single RASAero II Aeroplot CSV file, or a
        directory containing one or more such files.  Passing a single file
        directly is equivalent to a directory containing only that file
        (whole-vehicle mode, with a warning).
    fins_override:
        Explicit path to the CSV file for the fins component.  When provided,
        this file is used as the fins component instead of the filename
        heuristic (looking for ``"fin"`` in the stem).  Has no effect in
        whole-vehicle mode (single file).

    Raises
    ------
    FileNotFoundError
        If *aero_dir* is a directory that contains no ``*.csv`` files.
    ValueError
        If a CSV cannot be parsed or has unexpected columns.
    """
    aero_path = Path(aero_dir)

    if aero_path.is_file():
        warnings.warn(
            f"Single aeroplot CSV passed directly ({aero_path.name}). "
            "Treating as whole-vehicle data. "
            "Roll torques and pitch/yaw damping will not be computed.",
            UserWarning,
            stacklevel=2,
        )
        return _build_single(aero_path)

    csv_files = sorted(aero_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {aero_path}")

    if len(csv_files) == 1:
        warnings.warn(
            f"Only one aeroplot CSV found ({csv_files[0].name}). "
            "Treating as whole-vehicle data. "
            "Roll torques and pitch/yaw damping will not be computed.",
            UserWarning,
            stacklevel=2,
        )
        return _build_single(csv_files[0])

    return _build_components(csv_files, fins_override=fins_override)


# ---------------------------------------------------------------------------
# Internal helpers — build
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> np.ndarray:
    """Load a CSV → contiguous (N, 8) float64.

    Column layout (indices 0–7):
        0 Mach, 1 Reynolds, 2 AoA_deg, 3 CA_off, 4 CA_on, 5 CN, 6 CP_m,
        7 CN_alpha_per_rad

    Supported input layouts:

    * **8-column**: ``Mach,Reynolds,AoA_deg,CA_off,CA_on,CN,CP_m,CN_alpha_per_rad``
      — standard per-component output from ``pyrasaero convert``.
    * **7-column**: ``Mach,Reynolds,AoA_deg,CA_off,CA_on,CN,CP_m``
      — legacy format; column 7 (CN_alpha_per_rad) is set to NaN.
    """
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Cannot read aeroplot CSV {path}: {exc}") from exc
    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[1] >= 8:
        # 8-column format: Mach, Re, AoA_deg, CA_off, CA_on, CN, CP_m, CN_alpha_per_rad
        data = data[:, :8]
    elif data.shape[1] >= 7:
        # 7-column format: no CN_alpha_per_rad — append NaN column
        d7 = data[:, :7]
        data = np.empty((d7.shape[0], 8), dtype=np.float64)
        data[:, :7] = d7
        data[:, 7] = math.nan
    else:
        raise ValueError(
            f"{path.name}: expected ≥7 columns "
            f"(Mach,Reynolds,AoA_deg,CA_off,CA_on,CN,CP_m), got {data.shape[1]}"
        )

    return np.ascontiguousarray(data, dtype=np.float64)


def _complete_grid(src: np.ndarray) -> np.ndarray:
    """Fill missing rows so that the data forms a complete Cartesian grid.

    Groups by (col0, col1) and fills any missing col2 values by copying
    the nearest existing row.  Handles truncated RASAero II exports.
    """
    a2_all = np.unique(src[:, 2])
    expected_per_group = len(a2_all)

    # Group rows by (col0, col1) and find incomplete groups.
    # Use rank-based compound key to avoid floating-point collisions
    # when col0 and col1 span very different magnitudes.
    c0_ranks = np.searchsorted(np.unique(src[:, 0]), src[:, 0])
    n_r = len(np.unique(src[:, 1]))
    keys = c0_ranks * (n_r + 1) + np.searchsorted(np.unique(src[:, 1]), src[:, 1])
    unique_keys, counts = np.unique(keys, return_counts=True)
    incomplete = unique_keys[counts < expected_per_group]

    if len(incomplete) == 0:
        return src

    fill_rows: list[np.ndarray] = []
    for key in incomplete:
        mask = keys == key
        group = src[mask]
        existing_a2 = set(group[:, 2].tolist())
        for v2 in a2_all:
            if v2 not in existing_a2:
                nearest_idx = np.argmin(np.abs(group[:, 2] - v2))
                row = group[nearest_idx].copy()
                row[2] = v2
                fill_rows.append(row)

    if fill_rows:
        src = np.vstack([src] + fill_rows)
    return src


def _collapse_covarying_re(src: np.ndarray) -> np.ndarray:
    """If Reynolds covaries 1-to-1 with Mach, collapse Re to a constant.

    Returns a (possibly modified copy of) *src* with column 1 set to 0.0
    when each unique Mach maps to exactly one unique Re value, i.e. the
    data is effectively a 2-D (Mach × AoA) grid.

    Also fills any missing grid rows beforehand via :func:`_complete_grid`.
    """
    src = _complete_grid(src)
    m_vals = np.unique(src[:, 0])
    r_vals = np.unique(src[:, 1])
    a_vals = np.unique(src[:, 2])
    if len(m_vals) == len(r_vals) and len(m_vals) * len(a_vals) == len(src):
        src = src.copy()
        src[:, 1] = 0.0
    return src


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample CA_off, CA_on, CN, CP from *src* onto the structured target grid.

    *src* must be a structured (complete Cartesian product) grid — this is
    always the case for RASAero II Aeroplot output.  Extrapolation is clamped
    to the source grid boundary (nearest-value).

    Parameters
    ----------
    src:
        (N, 8) source data: Mach, Re, AoA_deg, CA_off, CA_on, CN, CP_m,
        CN_alpha_per_rad.  Only columns 3–6 are used; column 7 is ignored.
    mach_g, re_g, alpha_g:
        Target grid axes (sorted 1-D arrays).

    Returns
    -------
    ca_off, ca_on, cn, cp:
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
    for col in (3, 4, 5, 6):  # CA_off, CA_on, CN, CP
        tbl = s[:, col].reshape(NM_s, NR_s, NA_s)
        out.append(_trilinear(m_src, r_src, a_src, tbl, mach_g, re_g, alpha_g))

    return out[0], out[1], out[2], out[3]


def _resample_cna_2d(
    src: np.ndarray,
    mach_g: np.ndarray,
    re_g: np.ndarray,
) -> np.ndarray:
    """Build a (NM, NR) C_Nα table from the per-component source data.

    Reads CN_alpha_per_rad from column 7 of *src*.  Values are averaged over
    AoA ≤ 5° for each (Mach, Re) cell, then bilinearly interpolated onto the
    target grid via :func:`_trilinear` with a singleton alpha axis.

    Returns a zero array if column 7 is all NaN (legacy 7-column files).
    """
    cna_col = src[:, 7]
    if np.all(np.isnan(cna_col)):
        return np.zeros((len(mach_g), len(re_g)), dtype=np.float64)

    m_src = np.unique(src[:, 0])
    r_src = np.unique(src[:, 1])
    a_src = np.unique(src[:, 2])
    NM_s, NR_s, NA_s = len(m_src), len(r_src), len(a_src)

    idx_sort = np.lexsort((src[:, 2], src[:, 1], src[:, 0]))
    s = src[idx_sort]
    cna_3d = s[:, 7].reshape(NM_s, NR_s, NA_s)

    # Average over alpha ≤ 5° (linear regime)
    alpha_mask = a_src <= 5.0
    if not alpha_mask.any():
        alpha_mask = np.ones(NA_s, dtype=bool)
    cna_2d = np.mean(cna_3d[:, :, alpha_mask], axis=2)

    # Bilinear interpolation via _trilinear with singleton alpha axis
    cna_target = _trilinear(
        m_src, r_src, np.array([0.0]),
        cna_2d[:, :, np.newaxis],
        mach_g, re_g, np.array([0.0]),
    )
    return np.ascontiguousarray(cna_target[:, :, 0], dtype=np.float64)


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
    src = _collapse_covarying_re(_read_csv(path))
    mach_g, re_g, alpha_g = _unique_axes(src)
    ca_off, ca_on, cn, cp = _resample_3d(src, mach_g, re_g, alpha_g)
    NM, NR, NA = len(mach_g), len(re_g), len(alpha_g)
    zeros_2d = np.zeros((NM, NR), dtype=np.float64)
    # Dummy per-component arrays — shape (1, NM, NR, NA), never used when has_components=False
    dummy = np.zeros((1, NM, NR, NA), dtype=np.float64)
    dummy_cna = np.zeros((1, NM, NR), dtype=np.float64)
    return AeroModel(
        mach_grid=mach_g, re_grid=re_g, alpha_grid=alpha_g,
        ca_table_off=ca_off, ca_table_on=ca_on,
        cn_table=cn, cp_table=cp,
        cn_comp=dummy, cp_comp=dummy,
        cn_alpha_comp=dummy_cna,
        fin_comp_idx=-1,
        comp_names=[],
        has_components=False,
    )


def _build_components(
    csv_files: list[Path],
    fins_override: Path | None = None,
) -> AeroModel:
    if fins_override is not None:
        fins_paths = [fins_override]
    else:
        fins_paths = [p for p in csv_files if "fin" in p.stem.lower()]
    if not fins_paths:
        warnings.warn(
            "No file with 'fin' in its name found among aeroplot CSVs. "
            "Roll torques will not be computed.",
            UserWarning,
            stacklevel=3,
        )

    datasets = {p: _collapse_covarying_re(_read_csv(p)) for p in csv_files}
    mach_g, re_g, alpha_g = _unique_axes(*datasets.values())

    NM, NR, NA = len(mach_g), len(re_g), len(alpha_g)
    ca_off_tot = np.zeros((NM, NR, NA), dtype=np.float64)
    ca_on_tot  = np.zeros((NM, NR, NA), dtype=np.float64)

    cn_comp_list: list[np.ndarray] = []
    cp_comp_list: list[np.ndarray] = []
    cna_comp_list: list[np.ndarray] = []
    comp_names: list[str] = []
    fin_comp_idx = -1

    for path, src in datasets.items():
        ca_off_i, ca_on_i, cn_i, cp_i = _resample_3d(src, mach_g, re_g, alpha_g)
        ca_off_tot += ca_off_i
        ca_on_tot  += ca_on_i
        cn_comp_list.append(np.ascontiguousarray(cn_i, dtype=np.float64))
        cp_comp_list.append(np.ascontiguousarray(cp_i, dtype=np.float64))

        # C_Nα from CSV column 7 (averaged over linear AoA regime)
        cna_i = _resample_cna_2d(src, mach_g, re_g)
        cna_comp_list.append(np.ascontiguousarray(cna_i, dtype=np.float64))
        comp_names.append(path.stem)

        if path in fins_paths:
            fin_comp_idx = len(comp_names) - 1

    # Stack per-component tables: shape (N_comp, NM, NR, NA)
    cn_comp = np.ascontiguousarray(np.stack(cn_comp_list, axis=0), dtype=np.float64)
    cp_comp = np.ascontiguousarray(np.stack(cp_comp_list, axis=0), dtype=np.float64)
    # Per-component C_Nα: shape (N_comp, NM, NR)
    cn_alpha_comp = np.ascontiguousarray(np.stack(cna_comp_list, axis=0), dtype=np.float64)

    # Whole-vehicle CN and CP tables are not stored in per-component mode (§6.3).
    zeros_3d = np.zeros((NM, NR, NA), dtype=np.float64)

    return AeroModel(
        mach_grid=mach_g, re_grid=re_g, alpha_grid=alpha_g,
        ca_table_off=np.ascontiguousarray(ca_off_tot, dtype=np.float64),
        ca_table_on=np.ascontiguousarray(ca_on_tot, dtype=np.float64),
        cn_table=zeros_3d,
        cp_table=zeros_3d,
        cn_comp=cn_comp,
        cp_comp=cp_comp,
        cn_alpha_comp=cn_alpha_comp,
        fin_comp_idx=fin_comp_idx,
        comp_names=comp_names,
        has_components=True,
    )


# ---------------------------------------------------------------------------
# Numba hot-loop helpers
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
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


@nb.njit(cache=True, fastmath=True)
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


@nb.njit(cache=True, fastmath=True)
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


@nb.njit(cache=True, fastmath=True)
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


@nb.njit(cache=True, fastmath=True)
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


@nb.njit(cache=True, fastmath=True)
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


@nb.njit(cache=True, fastmath=True)
def aero_forces_moments(
    mach_g: np.ndarray,
    re_g: np.ndarray,
    alpha_g: np.ndarray,
    ca_tbl_off: np.ndarray,
    ca_tbl_on: np.ndarray,
    power_on: bool,
    cn_tbl: np.ndarray,
    cp_tbl: np.ndarray,
    cn_comp: np.ndarray,
    cp_comp: np.ndarray,
    has_components: bool,
    cn_alpha_comp: np.ndarray,
    M: float,
    Re: float,
    rho: float,
    V: float,
    A_ref: float,
    u_rel: float,
    v_rel: float,
    w_rel: float,
    q_rate: float,
    r_rate: float,
    cg: float,
) -> tuple[float, float, float, float, float, float]:
    """Body-frame aerodynamic forces [N] and pitch/yaw moments [N·m].

    Implements §6.4 of the specification using a hybrid approach:

    **Restoring forces** are computed by looking up each component's C_N and
    C_P at the **vehicle (bulk) angle of attack**.  The per-component tables
    are derived by differencing cumulative assemblies from an aerodynamic code
    that runs all components simultaneously at a common vehicle AoA.  The
    tabulated values encode inter-component interference at whole-vehicle flow
    conditions, so querying them at a per-component local AoA would misuse the
    data.

    **Pitch/yaw damping** is computed analytically using the Mandell linearised
    formulation.  When the rocket pitches at rate *q*, each component at
    position X_CPj experiences a crossflow perturbation
    ``delta_alpha_j = q * (X_CPj - X_CG) / V``.  The resulting damping moment
    is::

        delta_tau = -0.5 * rho * V * A_ref * q
                    * Sigma_j C_Nalpha_j * (X_CPj - X_CG)^2

    For a statically stable rocket the net sum is dominated by fin surfaces
    and the damping is stabilising.  Components with negative C_Nalpha
    (e.g. boattails) contribute anti-damping.

    Parameters
    ----------
    cn_comp, cp_comp:
        Per-component 4-D tables [N_comp, NM, NR, NA].
    cn_alpha_comp:
        Per-component C_Nalpha 3-D tables [N_comp, NM, NR] used for the
        analytical damping term.
    has_components:
        True  → per-component mode (restoring + damping).
        False → whole-vehicle fallback (restoring only).
    u_rel, v_rel, w_rel:
        Body-frame velocity of vehicle relative to wind [m/s].
    q_rate, r_rate:
        Body-frame pitch and yaw angular rates [rad/s].
    cg:
        Centre of gravity from nosecone tip [m].

    Returns
    -------
    (F_x, F_y, F_z, tau_pitch, tau_yaw, cp_whole)
        Forces in N, moments in N·m.
        ``cp_whole`` is the moment-balanced whole-vehicle CP [m from nosecone
        tip] — equals ``cg`` when total C_N ≈ 0.
    """
    EPS = 1.0e-6

    # Bulk lateral speed and AoA — used for axial force and all lookups
    V_lat_bulk = (v_rel * v_rel + w_rel * w_rel) ** 0.5
    alpha_bulk_deg = math.atan2(V_lat_bulk, u_rel) * _RAD_TO_DEG

    # Axial force — whole-vehicle C_A at bulk AoA (§6.4)
    ca_tbl = ca_tbl_on if power_on else ca_tbl_off
    C_A = _interp3(mach_g, re_g, alpha_g, ca_tbl, M, Re, alpha_bulk_deg)
    F_x = -0.5 * rho * V * V * A_ref * C_A

    q_dyn = 0.5 * rho * V * V * A_ref

    if has_components:
        # Per-component restoring forces + analytical damping (§6.4)
        F_y = 0.0
        F_z = 0.0
        tau_pitch = 0.0
        tau_yaw = 0.0
        cn_sum = 0.0
        cn_cp_sum = 0.0
        damp_sum = 0.0  # Σ C_Nα_i · arm_i²

        # Force direction from bulk crossflow
        if V_lat_bulk > EPS:
            sin_y_bulk = -v_rel / V_lat_bulk
            sin_z_bulk = -w_rel / V_lat_bulk
        else:
            sin_y_bulk = 0.0
            sin_z_bulk = 0.0

        n_comp = cn_comp.shape[0]
        for i in range(n_comp):
            # C_N and CP at vehicle (bulk) AoA
            cn_i = _interp3(mach_g, re_g, alpha_g, cn_comp[i], M, Re,
                            alpha_bulk_deg)
            cp_i = _interp3(mach_g, re_g, alpha_g, cp_comp[i], M, Re,
                            alpha_bulk_deg)
            arm_i = cp_i - cg

            # Restoring force: per-component magnitude, bulk direction
            F_N_i = q_dyn * cn_i
            F_yi = F_N_i * sin_y_bulk
            F_zi = F_N_i * sin_z_bulk
            F_y += F_yi
            F_z += F_zi

            # Restoring moment about CG
            tau_pitch += arm_i * F_zi
            tau_yaw   -= arm_i * F_yi

            # Damping coefficient: C_Nα_i · arm_i²
            cna_i = _interp2(mach_g, re_g, cn_alpha_comp[i], M, Re)
            damp_sum += cna_i * arm_i * arm_i

            # Accumulate for whole-vehicle CP (§6.3 item 3)
            cn_sum    += cn_i
            cn_cp_sum += cn_i * cp_i

        # Analytical pitch/yaw damping (Mandell §18.2)
        damp_coeff = -0.5 * rho * V * A_ref * damp_sum
        tau_pitch += damp_coeff * q_rate
        tau_yaw  += damp_coeff * r_rate

        if cn_sum > 1.0e-9:
            cp_whole = cn_cp_sum / cn_sum
        else:
            cp_whole = cg  # undefined at zero AoA; return CG (zero margin)

        return F_x, F_y, F_z, tau_pitch, tau_yaw, cp_whole

    else:
        # Single-file fallback — restoring only, no pitch/yaw damping (§6.4)
        CN = _interp3(mach_g, re_g, alpha_g, cn_tbl, M, Re, alpha_bulk_deg)
        CP = _interp3(mach_g, re_g, alpha_g, cp_tbl, M, Re, alpha_bulk_deg)

        if V_lat_bulk > EPS:
            sin_y = -v_rel / V_lat_bulk
            sin_z = -w_rel / V_lat_bulk
        else:
            sin_y = 0.0
            sin_z = 0.0

        F_N = q_dyn * CN
        F_y = F_N * sin_y
        F_z = F_N * sin_z

        arm = CP - cg
        tau_pitch =  arm * F_z
        tau_yaw   = -arm * F_y

        return F_x, F_y, F_z, tau_pitch, tau_yaw, CP


@nb.njit(cache=True, fastmath=True)
def cn_alpha_comp_at(
    mach_g: np.ndarray,
    re_g: np.ndarray,
    cna_comp: np.ndarray,
    M: float,
    Re: float,
    comp_idx: int,
) -> float:
    """Per-component C_Nα [1/rad] at (M, Re) for component *comp_idx*.

    Used for damping post-processing.
    """
    return _interp2(mach_g, re_g, cna_comp[comp_idx], M, Re)
