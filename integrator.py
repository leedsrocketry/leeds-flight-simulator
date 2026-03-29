"""Dormand-Prince RK4(5) adaptive step-size utilities.

Provides the Butcher tableau constants and step-size control helpers for an
embedded Runge-Kutta method with FSAL (First Same As Last) property.
The integration loops themselves live in ``dynamics.py``, where they call
phase-specific derivative functions directly.

Reference: Dormand & Prince (1980); Press et al., *Numerical Recipes* (2007), §17.2.

Public API
----------
Module-level constants:
    DP_C  — nodes (7,)
    DP_A  — stage coefficient rows, stored as individual 1-D arrays DP_A1..DP_A5
    DP_B  — 5th-order weights (7,)
    DP_E  — error weights B − B* (7,)

@njit helper functions:
    error_norm(y, y_new, y_err, rtol, atol) → float
    optimal_step_factor(err_norm) → float
    clamp_step(h_new, h_min, h_max) → float
"""

from __future__ import annotations

import numpy as np
import numba as nb

# ---------------------------------------------------------------------------
# Dormand-Prince Butcher tableau
# ---------------------------------------------------------------------------
# Nodes c_i
DP_C = np.array([
    0.0,
    1.0 / 5.0,
    3.0 / 10.0,
    4.0 / 5.0,
    8.0 / 9.0,
    1.0,
    1.0,
], dtype=np.float64)

# Stage coefficients a_{i,j} — stored as individual rows to avoid a large
# sparse 2-D array.  Row i has i entries (lower-triangular).
DP_A1 = np.array([1.0 / 5.0], dtype=np.float64)
DP_A2 = np.array([3.0 / 40.0, 9.0 / 40.0], dtype=np.float64)
DP_A3 = np.array([44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0], dtype=np.float64)
DP_A4 = np.array([
    19372.0 / 6561.0, -25360.0 / 2187.0,
    64448.0 / 6561.0, -212.0 / 729.0,
], dtype=np.float64)
DP_A5 = np.array([
    9017.0 / 3168.0, -355.0 / 33.0,
    46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0,
], dtype=np.float64)

# 5th-order weights b_i  (7 stages, FSAL)
DP_B = np.array([
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
], dtype=np.float64)

# 4th-order embedded weights b*_i
_DP_BS = np.array([
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
], dtype=np.float64)

# Error weights e_i = b_i − b*_i  (used to compute the error estimate)
DP_E = np.ascontiguousarray(DP_B - _DP_BS, dtype=np.float64)


# ---------------------------------------------------------------------------
# Step-size control helpers
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def error_norm(
    y: np.ndarray,
    y_new: np.ndarray,
    y_err: np.ndarray,
    rtol: float,
    atol: float,
) -> float:
    """RMS error norm for adaptive step-size control.

    err = sqrt( (1/n) * sum_i ( y_err_i / (atol + rtol * max(|y_i|, |y_new_i|)) )^2 )

    Returns a value <= 1.0 when the step is acceptable.
    """
    n = y.shape[0]
    s = 0.0
    for i in range(n):
        scale = atol + rtol * max(abs(y[i]), abs(y_new[i]))
        s += (y_err[i] / scale) ** 2
    return (s / n) ** 0.5


@nb.njit(cache=True, fastmath=True)
def optimal_step_factor(err_norm_val: float) -> float:
    """Optimal step-size scaling factor from the error norm.

    factor = 0.9 * err^(-1/5)  for 5th-order method, clamped to [0.2, 5.0].
    Returns 5.0 when err_norm_val is effectively zero (step far too small).
    """
    if err_norm_val < 1.0e-10:
        return 5.0
    factor = 0.9 * err_norm_val ** (-0.2)
    if factor < 0.2:
        return 0.2
    if factor > 5.0:
        return 5.0
    return factor


@nb.njit(cache=True, fastmath=True)
def clamp_step(h: float, h_min: float, h_max: float) -> float:
    """Clamp step size to [h_min, h_max]."""
    if h < h_min:
        return h_min
    if h > h_max:
        return h_max
    return h
