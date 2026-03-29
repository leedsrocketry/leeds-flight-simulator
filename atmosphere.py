"""ISA atmosphere model (ISO 2533:1975).

All public functions are Numba @njit compiled and accept scalar float64 altitude
in metres (geopotential, AGL). They return SI units throughout.
"""

import numba as nb
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G0: float = 9.80665          # m/s²  standard gravity
M_AIR: float = 0.0289644     # kg/mol  molar mass of dry air
R_STAR: float = 8.31447      # J/(mol·K)  universal gas constant
GAMMA: float = 1.4           # —  ratio of specific heats for air

# Sutherland viscosity reference values
_T_REF_SUTH: float = 273.15  # K
_MU_REF_SUTH: float = 1.716e-5  # Pa·s
_S_SUTH: float = 110.4       # K  Sutherland constant

# ---------------------------------------------------------------------------
# ISA layer table  (base altitude m, base temperature K, lapse rate K/m,
#                   base pressure Pa — computed analytically at import time)
# ---------------------------------------------------------------------------
#   Layer           h_b       T_b      L
_LAYERS = np.array([
    [0.0,        288.15,  -0.0065],   # Troposphere
    [11_000.0,   216.65,   0.0],      # Tropopause
    [20_000.0,   216.65,  +0.001],    # Stratosphere-1
], dtype=np.float64)


def _compute_base_pressures() -> np.ndarray:
    """Compute layer base pressures from the layer table at import time."""
    p = np.empty(len(_LAYERS), dtype=np.float64)
    p[0] = 101_325.0
    for i in range(1, len(_LAYERS)):
        h_b, T_b, L = _LAYERS[i - 1]
        h1 = _LAYERS[i, 0]
        dh = h1 - h_b
        if L != 0.0:
            T1 = T_b + L * dh
            p[i] = p[i - 1] * (T1 / T_b) ** (-G0 * M_AIR / (R_STAR * L))
        else:
            p[i] = p[i - 1] * np.exp(-G0 * M_AIR * dh / (R_STAR * T_b))
    return p


_BASE_PRESSURES: np.ndarray = _compute_base_pressures()

# Flatten to plain arrays for Numba (no structured dtype in @njit)
_H_BASE = np.ascontiguousarray(_LAYERS[:, 0])
_T_BASE = np.ascontiguousarray(_LAYERS[:, 1])
_L_BASE = np.ascontiguousarray(_LAYERS[:, 2])
_P_BASE = np.ascontiguousarray(_BASE_PRESSURES)

# ---------------------------------------------------------------------------
# Numba-compiled core
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _layer_index(h: float) -> int:
    """Return the index of the ISA layer containing altitude *h*."""
    idx = 0
    for i in range(1, 3):
        if h >= _H_BASE[i]:
            idx = i
    return idx


@nb.njit(cache=True, fastmath=True)
def temperature(h: float) -> float:
    """Static temperature [K] at geopotential altitude *h* [m]."""
    i = _layer_index(h)
    return _T_BASE[i] + _L_BASE[i] * (h - _H_BASE[i])


@nb.njit(cache=True, fastmath=True)
def pressure(h: float) -> float:
    """Static pressure [Pa] at geopotential altitude *h* [m]."""
    i = _layer_index(h)
    T_b = _T_BASE[i]
    L = _L_BASE[i]
    p_b = _P_BASE[i]
    dh = h - _H_BASE[i]
    if L != 0.0:
        T = T_b + L * dh
        return p_b * (T / T_b) ** (-G0 * M_AIR / (R_STAR * L))
    else:
        return p_b * np.exp(-G0 * M_AIR * dh / (R_STAR * T_b))


@nb.njit(cache=True, fastmath=True)
def density(h: float) -> float:
    """Air density [kg/m³] at geopotential altitude *h* [m]."""
    T = temperature(h)
    p = pressure(h)
    return p * M_AIR / (R_STAR * T)


@nb.njit(cache=True, fastmath=True)
def speed_of_sound(h: float) -> float:
    """Speed of sound [m/s] at geopotential altitude *h* [m]."""
    T = temperature(h)
    return (GAMMA * R_STAR * T / M_AIR) ** 0.5


@nb.njit(cache=True, fastmath=True)
def dynamic_viscosity(h: float) -> float:
    """Dynamic viscosity [Pa·s] via Sutherland's law at altitude *h* [m]."""
    T = temperature(h)
    return (
        _MU_REF_SUTH
        * (T / _T_REF_SUTH) ** 1.5
        * (_T_REF_SUTH + _S_SUTH)
        / (T + _S_SUTH)
    )


@nb.njit(cache=True, fastmath=True)
def isa(h: float) -> tuple[float, float, float, float, float]:
    """Return (T [K], p [Pa], rho [kg/m³], a [m/s], mu [Pa·s]) at altitude *h* [m]."""
    i = _layer_index(h)
    T_b = _T_BASE[i]
    L = _L_BASE[i]
    p_b = _P_BASE[i]
    dh = h - _H_BASE[i]

    T = T_b + L * dh

    if L != 0.0:
        p = p_b * (T / T_b) ** (-G0 * M_AIR / (R_STAR * L))
    else:
        p = p_b * np.exp(-G0 * M_AIR * dh / (R_STAR * T_b))

    rho = p * M_AIR / (R_STAR * T)
    a = (GAMMA * R_STAR * T / M_AIR) ** 0.5
    mu = (
        _MU_REF_SUTH
        * (T / _T_REF_SUTH) ** 1.5
        * (_T_REF_SUTH + _S_SUTH)
        / (T + _S_SUTH)
    )
    return T, p, rho, a, mu
