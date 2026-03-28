# Leeds Flight Simulator (LFS) — Specification
---

## 1 Purpose & Scope

This document specifies a six-degree-of-freedom (6DoF) Monte Carlo (MC) flight simulator for single-stage, passively stabilised, axisymmetric sounding rockets. The simulator generates flight safety analysis evidence suitable for a CAA large rocket permission safety case:

- Trajectory containment within a configurable inward buffer of a danger area, under all credible descent failure modes.
- Landing points outside the coastline polygon, if a coastline is provided.
- Ballistic and drogue-only landings within configured observation coverage zones.
- Aerodynamic stability throughout powered and coasting flight.
- Automated launch azimuth and inclination optimisation when these parameters are set to `"auto"`.
- Launch-day go/no-go decision support via re-analysis conditioned on current weather data.

The simulator covers launch rail exit to landing. It does not model ground handling, ignition transients, or post-landing dynamics. It must run entirely offline.

## 2 Notation

| Symbol | Definition |
|--------|-----------|
| **NED** | North-East-Down earth-fixed frame |
| **Body frame** | x nosecone, y starboard, z ventral; origin at CG |
| **q** = [q₀, q₁, q₂, q₃] | Attitude quaternion (scalar-first), body → NED |
| **ω** = [p, q, r] | Body-frame angular velocity (roll, pitch, yaw) |
| **v_b** = [u, v, w] | Body-frame translational velocity |
| V | Airspeed magnitude (m/s) |
| α | Angle of attack = atan2(w_rel, u_rel) |
| β | Sideslip = asin(v_rel / V) |
| M, Re | Mach number, Reynolds number |
| C_A, C_N | Axial and normal force coefficients |
| C_Nα | Normal force slope ∂C_N/∂α (per rad), used for damping only |
| CP | Centre of pressure from nosecone tip (m) |
| CG | Centre of gravity from nosecone tip (m) |
| SM | Static margin = (CP − CG) / d (calibres) |
| d | Reference diameter (from vehicle config) |
| A_ref | Reference area = πd²/4 |
| I_R | Roll moment of inertia (= I_xx) (kg·m²) |
| I_L | Lateral moment of inertia (= I_yy = I_zz) (kg·m²) |
| C₂, C₂A, C₂R | Damping coefficient: total, aerodynamic, propulsive (N·m·s) |

The vehicle is axisymmetric: I_yy = I_zz = I_L. Aerodynamic coefficients depend only on total AoA magnitude, not roll orientation.


## 3 Coordinate Systems

### 3.1 NED Frame

Origin at the launch rail exit point (lat/lon from `simulation.yaml`). Flat-Earth approximation. NED → geodetic conversion:

```
lat = lat₀ + north / R_Earth
lon = lon₀ + east / (R_Earth · cos(lat₀))
```

R_Earth = 6,371,000 m.

### 3.2 Body Frame

Right-handed, fixed to vehicle. x toward nosecone, y starboard, z ventral. Origin at CG (shifts during burn).

### 3.3 Aerodynamic Angles

```
v_wind_b = C_bn · [v_wind_N, v_wind_E, 0]
v_rel    = v_b − v_wind_b = [u_rel, v_rel, w_rel]
V        = |v_rel|
α = atan2(w_rel, u_rel)
β = asin(v_rel / V)
```

### 3.4 Attitude

Unit quaternion q (scalar-first). DCM:

```
        ⎡ 1−2(q₂²+q₃²)   2(q₁q₂−q₀q₃)   2(q₁q₃+q₀q₂) ⎤
C_nb =  ⎢ 2(q₁q₂+q₀q₃)   1−2(q₁²+q₃²)   2(q₂q₃−q₀q₁) ⎥
        ⎣ 2(q₁q₃−q₀q₂)   2(q₂q₃+q₀q₁)   1−2(q₁²+q₂²) ⎦
```

C_bn = C_nbᵀ. Renormalised after every integration step.

<!-- Ref: Diebel, "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors" (2006) -->


## 4 Atmospheric Model

### 4.1 ISA (ISO 2533:1975)

| Layer | h_b (m) | T_b (K) | L (K/m) |
|-------|---------|---------|---------|
| Troposphere | 0 | 288.15 | −0.0065 |
| Tropopause | 11,000 | 216.65 | 0.0 |
| Stratosphere 1 | 20,000 | 216.65 | +0.001 |

Constants: g₀ = 9.80665 m/s², M_air = 0.0289644 kg/mol, R* = 8.31447 J/(mol·K), γ_air = 1.4.

```
T(h) = T_b + L(h − h_b)
```

Pressure (L ≠ 0): p = p_b[T/T_b]^(−g₀M_air/(R*L)).  
Pressure (L = 0): p = p_b·exp(−g₀M_air(h−h_b)/(R*T_b)).  
Density: ρ = pM_air/(R*T).  
Speed of sound: a = √(γ_air R*T/M_air).  
Viscosity (Sutherland): μ = 1.716×10⁻⁵·(T/273.15)^1.5·(273.15+110.4)/(T+110.4) Pa·s.

All ISA functions Numba-compiled. Geopotential correction not applied.

### 4.2 Gravity

Constant g = g₀, NED down.


## 5 Wind Model

### 5.1 Input Format

Wind profiles are supplied as an external NumPy `.npz` file specified in `simulation.yaml`. The file contains a pre-generated ensemble of perturbed wind profiles — one per Monte Carlo sample. The simulator has no knowledge of how the profiles were generated; all source-specific logic (EarthGRAM, GFS, ECMWF, radiosonde, perturbation modelling) is handled by a separate wind profile generator tool.

The `.npz` file must contain the following arrays:

| Array key | Shape | Description |
|-----------|-------|-------------|
| `altitude_m` | `(M,)` | Altitude grid in metres AGL, monotonically increasing |
| `wind_east_ms` | `(N, M)` | Eastward wind component per profile per altitude (m/s) |
| `wind_north_ms` | `(N, M)` | Northward wind component per profile per altitude (m/s) |

`N` must be ≥ `num_samples` in `simulation.yaml`. Sample `i` uses profile `i`. The mean wind profile (used in optimisation Phase 2, §13.3) is computed at load time as the arithmetic mean across all `N` profiles.

> **Planned behaviour (not yet implemented):** `wind_profiles` in `simulation.yaml` may also point to a directory of `.npz` files. The full analysis runs independently for each file; output folders are suffixed with the wind profile filename when more than one is present.

### 5.2 Surface Wind Override

When a `surface_wind` sub-section is present under `launch` in `simulation.yaml`, a user-specified surface wind (speed in m/s, bearing in degrees clockwise from north) replaces the lower portion of each profile up to a configurable blend height. Omit the `surface_wind` section entirely to disable the override.

```
if h ≤ 0:              wind = override_vector
elif h < blend_height:  wind = (1 − h/blend_height)·override_vector + (h/blend_height)·profile(h)
else:                   wind = profile(h)
```

The override vector is derived from bearing and speed: eastward = speed·sin(bearing), northward = speed·cos(bearing).

### 5.3 Runtime Interpolation

During integration, wind at the current altitude is linearly interpolated from the profile assigned to the current sample. Below the lowest grid altitude, held constant. Expressed in NED as [v_north, v_east, 0].


## 6 Aerodynamic Model

### 6.1 Data Source

Aeroplot CSV files specified via `aero_tables` in `vehicle.yaml`. The value may be either a **single CSV file** (whole-vehicle mode) or a **directory of CSV files** (per-component mode). Per-component data is required for C₂ damping (§8.3.5) and roll torques (§6.5). If only a single file is provided (whether passed directly or as the sole file in a directory), assume it covers the full vehicle, disable roll and damping assessment, and warn the user.

### 6.2 CSV Format

```
Mach, Reynolds, AoA_deg, CA, CN, CP_m
```

CP_m in metres from nosecone tip. Grid may be irregular in all three dimensions.

### 6.3 Table Construction

At startup:

1. **Per-component:** C_Nα(M, Re) and CP(M, Re) for C₂ and roll. C_Nα derived by least-squares fit through origin for α ≤ 5°.
2. **Whole-vehicle:** C_A(M, Re, α), C_N(M, Re, α), CP(M, Re, α) by summing components.

Irregular tables resampled onto a regular grid at startup for Numba-compatible `np.searchsorted` interpolation. Clamped at boundaries.

### 6.4 Forces

Body-frame aerodynamic forces at the actual current α (no linearisation):

```
F_aero_x = −½ρV²A_ref C_A(M,Re,α)
F_aero_y =  ½ρV²A_ref C_N(M,Re,α)·(−v_rel/√(v_rel²+w_rel²))
F_aero_z =  ½ρV²A_ref C_N(M,Re,α)·(−w_rel/√(v_rel²+w_rel²))
```

Normal force direction factors distribute C_N into yaw (y) and pitch (z) planes. Normal force vanishes at zero AoA.

### 6.5 Roll (Barrowman)

RASAero assumes zero roll. Roll torques computed analytically:

**Cant forcing:** τ_cant = ½ρV²A_ref C_lδ δ, where C_lδ = (C_Nα)_fins·(r_fin/d).  
**Damping:** τ_damp = −½ρVA_ref d C_lp(ω_z d/2V), where C_lp = −(C_Nα)_fins·(r_fin/d)²·(4/3).

<!-- Ref: Barrowman (1967), §6; Niskanen (2009), §3.4 -->

Roll dynamics: I_R dω_z/dt = τ_cant + τ_damp.


## 7 Propulsion

### 7.1 Thrust Curve

Provided as a **.eng file** (RASP format). Parsed for time-thrust pairs, propellant mass and mass of the motor loaded and ready for flight. Linear interpolation; zero propellant mass and thrust after final point.

### 7.2 Impulse Scaling

F(t) = F_nominal(t) · k_impulse, where k_impulse ~ Normal(1.0, whatever is in the simulation input parameters).

### 7.3 Mass Properties

The user supplies wet vehicle properties and propellant properties in `vehicle.yaml`. Dry properties are derived at startup and never specified by the user.

**Dry mass and CG** (derived in `motor.py`):

```
m_dry  = wet_mass − m_prop_0
cg_dry = (wet_mass · wet_cg − m_prop_0 · wet_motor_cg) / m_dry
```

where `m_prop_0` is the propellant mass from the `.eng` header and `wet_motor_cg` (the loaded motor CG from `vehicle.yaml`) is used as the fixed propellant CG throughout the burn (inside-out burn model — exact when the casing is massless, an excellent approximation for high-mass-fraction solid motors).

**Dry inertias** (derived via parallel-axis theorem):

```
I_roll_dry = wet_inertia_roll − propellant_inertia_roll

I_lateral_dry = wet_inertia_lateral
              − (propellant_inertia_lateral + m_prop_0 · (wet_motor_cg − wet_cg)²)
              − m_dry · (cg_dry − wet_cg)²
```

**Time-varying mass, CG, and inertia** (evaluated at each integration timestep in `motor.py`):

```
m_prop(t)  = m_prop_0 · (1 − ∫₀ᵗF(τ)dτ / I_total)
ṁ(t)       = m_prop_0 · F(t) / I_total
m_total(t) = m_dry + m_prop(t)

CG(t) = (m_dry · cg_dry  +  m_prop(t) · wet_motor_cg) / m_total(t)
```

Inertias at each timestep are computed via the parallel-axis theorem, transferring the dry and propellant contributions to the instantaneous CG:

```
I_roll(t) = I_roll_dry + propellant_inertia_roll · f(t)

I_lateral(t) = (I_lateral_dry + m_dry · (cg_dry − CG(t))²)
             + (propellant_inertia_lateral · f(t) + m_prop(t) · (wet_motor_cg − CG(t))²)
```

where `f(t) = m_prop(t) / m_prop_0`.

### 7.4 Nozzle and Thrust Altitude Correction

`nozzle_position` (distance from nosecone tip, metres) is the deterministic input for C₂R. `nozzle_diameter` gives the nozzle exit area Aₑ = π·dₑ²/4, used for altitude thrust correction:

```
F(h) = F₀(t) + Aₑ · (p₀ − p_ISA(h))
```

where p₀ = 101 325 Pa and p_ISA(h) is the ISA ambient pressure (§4.1). The `.eng` thrust curve is assumed to be measured at sea level. `thrust_corrected_at` is used in the dynamics hot loop; `thrust_at` returns the raw sea-level value.

### 7.5 Thrust Misalignment

Not modelled.


## 8 Equations of Motion

### 8.1 Phases

1. **Launch rail** — Constrained 1D translation, no rotation.
2. **Free flight (6DoF)** — Full translation + rotation.
3. **Descent (3DoF)** — Point-mass under drag and gravity.

### 8.2 Launch Rail Phase

Launch rail axis (azimuth ψ, inclination θ, both stochastic):

```
ê_rail = [cosθ cosψ, cosθ sinψ, −sinθ]
F_along = F_thrust − ½ρV²A_ref C_A(M,Re,0) − mg sinθ
```

Vehicle accelerates until CG travels the launch rail length (from `simulation.yaml`, `launch.rail.length`). Phase 2 initial conditions: v_b = [V_exit, 0, 0], quaternion from launch rail orientation, ω = 0.

### 8.3 Free Flight (6DoF)

#### 8.3.1 State Vector (13 components)

```
x = [r_N, r_E, r_D, q₀, q₁, q₂, q₃, u, v, w, p, q, r]
```

#### 8.3.2 Translational Dynamics

<!-- Ref: Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics", 3rd ed. (2014), Ch. 4 -->

```
m·(dv_b/dt + ω×v_b) = F_aero_b + [F(t),0,0] + C_bn·[0,0,mg₀]
```

Aerodynamic forces from §6.4.

#### 8.3.3 Rotational Dynamics

<!-- Ref: Zipfel (2014), Ch. 5; Sheridan et al. (2014), §3 -->

```
I·(dω/dt) + ω×(I·ω) = τ_aero
```

I = diag(I_R, I_L, I_L), time-varying during burn. Expanded:

```
I_R dp/dt = τ_roll
I_L dq/dt = (I_L−I_R)rp + τ_pitch
I_L dr/dt = (I_R−I_L)pq + τ_yaw
```

#### 8.3.4 Aerodynamic Moments

**Restoring (nonlinear):**

```
M_restore = ½ρV²A_ref C_N(M,Re,α)·(CP(M,Re,α) − CG(t))
τ_pitch_restore = M_restore·(−w_rel/√(v_rel²+w_rel²))
τ_yaw_restore   = M_restore·(−v_rel/√(v_rel²+w_rel²))
```

**Damping (Mandell per-component):**

<!-- Ref: Mandell, Caporaso, Bengen, "Topics in Advanced Model Rocketry" (1973), Eq. 97, 99, 101 -->

```
C₂A = ½VA_ref Σᵢ(C_Nα)ᵢ·(CPᵢ − CG)²
C₂R = ṁ(t)·(L_ne − CG)²                  (burn only)
C₂  = C₂A + C₂R  (thrust) or C₂A (coast)

τ_pitch_damp = −C₂·q
τ_yaw_damp   = −C₂·r
```

**Roll:** §6.5.

**Totals:**

```
τ_roll  = τ_cant + τ_damp
τ_pitch = τ_pitch_restore + τ_pitch_damp
τ_yaw   = τ_yaw_restore + τ_yaw_damp
```

#### 8.3.5 Kinematics

```
dr_e/dt = C_nb·v_b
dq/dt   = ½Ω(ω)·q
```

Quaternion renormalised after each step.

#### 8.3.6 Ascent Termination

Terminates on:

1. **Apogee:** vertical velocity sign change.
2. **Stability/AoA violation:** criteria from §11.1 violated — sample flagged non-compliant, terminated.

The configurable AoA threshold (`sm_aoa_threshold` in `simulation.yaml`) in the SM check naturally excludes launch rail-exit and apogee transients.

### 8.4 Descent (3DoF)

Point-mass: dv/dt = F_drag/m + g, dr/dt = v.

```
F_drag = −½ρ|v_rel|²·CdA_eff·(v_rel/|v_rel|)
```

**`ballistic`:** CdA = A_ref·C_A(M,Re,0) — varies with speed/altitude (lawn-dart, α≈0).
**`drogue_only`:** CdA = CdA_drogue.
**`premature_main`:** CdA = CdA_drogue + CdA_main (both open at apogee).
**`nominal`:** CdA = CdA_drogue → CdA_drogue + CdA_main at configured deployment altitude.

Drogue drag always adds to main when main is deployed. Main deploys at the configured deployment altitude above launch site elevation (flight computers zero at power-on). Descent terminates at r_D ≥ 0.

### 8.5 Ascent (3DoF)

A reduced 3-DOF ascent mode is available for use by the optimisation routine (§13). In this mode, roll dynamics are disabled, the vehicle is treated as a point mass with axial drag only, and α = 0 is assumed throughout. The state vector reduces to position and scalar velocity along the launch rail/flight axis. This mode is not used for the main Monte Carlo analysis.


## 9 Descent Scenarios

| Scenario | Drogue | Main | CdA_eff |
|----------|--------|------|---------|
| `nominal` | Apogee | Configured deploy alt | CdA_drogue → CdA_drogue+CdA_main |
| `ballistic` | No | No | A_ref·C_A(M,Re,0) |
| `drogue_only` | Apogee | Fails | CdA_drogue |
| `premature_main` | Apogee | Apogee | CdA_drogue+CdA_main |

### 9.1 Valid Recovery Configurations

A drogue parachute without a main parachute is not a valid configuration (loading `vehicle.yaml` will raise an error). The three valid configurations and their active scenarios are:

| Recovery config | `nominal` | `ballistic` | `drogue_only` | `premature_main` |
|----------------|-----------|-------------|---------------|-----------------|
| Both, `main.threshold` = numeric | ✓ | ✓ | ✓ | ✓ |
| Both, `main.threshold` = `"apogee"` | ✓ | ✓ | ✓ | — |
| Main only, `main.threshold` = numeric | ✓ | ✓ | — | ✓ |
| Main only, `main.threshold` = `"apogee"` | ✓ | ✓ | — | — |
| No parachutes | ✓ | — | — | — |

### 9.2 Scenario Generation Rules

| Scenario | Condition |
|----------|-----------|
| `nominal` | Always |
| `ballistic` | At least one parachute is configured |
| `drogue_only` | Both drogue **and** main are configured |
| `premature_main` | Main is configured **and** `main.threshold` is a numeric altitude (not `"apogee"`) |

`premature_main` is suppressed when `main.threshold = "apogee"` because apogee deployment is the earliest possible main deployment — no earlier failure mode exists.

### 9.3 Inactive Scenario Filtering and Warnings

`sea_check_scenarios` and `los_check_scenarios` in `simulation.yaml` are user-supplied lists and may name scenarios that are inactive for the current vehicle configuration (e.g. `drogue_only` listed but no drogue configured). The simulator must not raise an error in this case. Instead:

1. At startup, compute the set of active scenarios via `VehicleRecovery.active_scenarios` (§9.2, implemented in `config.py`).
2. For each check list, silently drop any scenario name that is not in the active set — only the intersection is checked.
3. Emit a warning to the simulation output for each dropped scenario, e.g.:

   ```
   WARNING: 'drogue_only' in sea_check_scenarios is not an active scenario for
   this vehicle configuration and will be skipped.
   ```

Warnings are emitted once per run, before the Monte Carlo loop begins. They do not affect the compliance result.

> **Planned behaviour (not yet implemented):** all warnings are blocking — the simulator pauses and prompts the user to acknowledge before continuing. `--no-warn` (§17.2) suppresses the interactive prompt; warnings still appear in the log and results summary.


## 10 Numerical Integration

### 10.1 Method

Adaptive Dormand-Prince (RK45) with embedded error control, Numba `@njit` compiled. Perferably an existing implementation.

<!-- Ref: Dormand & Prince (1980); Press et al., "Numerical Recipes" (2007), §17.2 -->

### 10.2 Tolerances

| Parameter | Default |
|-----------|---------|
| rtol, atol | 1×10⁻⁶ |
| Min step | 1×10⁻⁴ s |
| Max step (powered/coast/descent) | 0.05 / 0.1 / 1.0 s |

### 10.3 Auto-Calibration

At the beging of execution, run ~20 samples at tight tolerance (1×10⁻⁹), re-run at progressively looser tolerances, use and report the loosest that maintains acceptable output deviation.

## 11 Acceptance Criteria & Compliance

### 11.1 Per-Sample

A sample is compliant iff **all** of:

**Stability & AoA** — during powered/coasting flight, whenever AoA < `sm_aoa_threshold`: SM ≥ `sm_subsonic_min` cal below `sm_transition_mach` Mach, or SM ≥ `sm_supersonic_min` cal at or above it. AoA must not exceed `aoa_max` at any point. Violation terminates the sample (§8.3.6). All thresholds configurable in `simulation.yaml`. The Mach transition threshold definition varies between tools; the default value follows the RASAero II convention.

<!-- Ref: RASAero manual; Mandell et al. (1973), p. 85 -->

**Containment** — full trajectory within the buffered danger area: landing point inside the buffered danger area footprint, and peak altitude below the altitude ceiling (§14).

**Coastline check** — if a coastline file is provided, the landing point must satisfy the configured `coastline_mode` (§14.2). If no coastline file is provided, this check is skipped.

**Observation coverage** — landing within the configured radius of at least one observation station. Applied only to scenarios listed in `los_check_scenarios` (§9).

### 11.2 Run-Level

Pass iff ≥ `compliance_threshold` fraction compliant (configurable, default 0.997). All active scenario runs must pass (§9).


## 12 Monte Carlo Framework

### 12.1 Stochastic Inputs

| Parameter | Distribution | Parameters |
|-----------|-------------|------------|
| Wind profile | Indexed | Profile `i` from `.npz` ensemble for sample `i` |
| Launch rail azimuth | Normal | μ, σ from config |
| Launch rail inclination | Normal | μ, σ from config |
| Impulse factor | Normal | μ, σ from config |
| Fin cant | Normal | μ, σ from config |

Normal distributions are not truncated.

### 12.2 Reproducibility

Each sample's random draws are deterministic from (master_seed, run_index, sample_index) via `SeedSequence`. Wind profile assignment is deterministic by sample index. Any sample can be replayed exactly without storing trajectory data.

### 12.3 Structure

One run per active scenario (§9) × N samples (default 1,000). Parallel across runs via `multiprocessing`. Between one and four scenarios are active depending on which parachutes are configured.

### 12.4 Automatic Optimisation Trigger

If `azimuth` or `inclination` in the `launch.rail` config section is set to `"auto"`, the optimisation routine (§13) runs before the main Monte Carlo analysis. The routine selects optimal integer values for whichever parameters are set to `"auto"`, then the main MC analysis proceeds using those values as the nominal μ. See §13 for details.


## 13 Inclination/Azimuth Optimisation

### 13.1 Overview

A four-phase routine to select optimal integer launch azimuth and inclination, maximising the probability that all active descent scenarios remain within the buffered danger area. Uses a deterministic worst-case ascent, analytical pre-filtering, and 1D Bayesian optimisation over azimuth. Implemented in `optimisation.py`.

The routine runs automatically when `azimuth` and/or `inclination` are set to `"auto"` in `simulation.yaml` under `launch.rail` (§12.4). If only inclination is `"auto"`, only Phase 1 runs and the provided azimuth is used directly. If only azimuth is `"auto"`, the provided inclination is used and Phases 2–4 run. If both are `"auto"`, all four phases run.

Once the optimisation completes, the standard Monte Carlo analysis (§12) runs with the determined values as the nominal azimuth and inclination. The acceptance criteria and thresholds are the same as for any other run (§11).

### 13.2 Phase 1 — Inclination Selection

**Function:** `select_inclination()`

**Inputs:**

| Parameter | Value |
|-----------|-------|
| Inclination candidates | Integer values from `inclination_range` in `simulation.yaml` |
| Azimuth | 0° fixed (arbitrary, no wind) |
| Total impulse | Maximum (deterministic) |
| Wind | Disabled |
| Fin cant | 0° |
| Simulator mode | 3-DOF ascent (§8.5, roll dynamics disabled) |

**Process:**

1. Run one deterministic simulation per inclination candidate.
2. Record ballistic landing point and apogee position `(x, y, z)` for each.
3. Select the maximum inclination satisfying both:
   - Ballistic landing point **outside** `ballistic_exclusion_radius` from launch site (from `simulation.yaml`)
   - Ballistic landing point **inside** buffered danger area footprint

**Outputs:**

- `selected_inclination` — fixed for all subsequent phases
- `apogee_positions[inclination]` — dict of apogee `(x, y, z)` keyed by inclination candidate

### 13.3 Phase 2 — Azimuth Bound Narrowing

**Function:** `narrow_azimuth_bounds(selected_inclination, apogee_positions)`

**Inputs:**

| Parameter | Source |
|-----------|--------|
| `selected_inclination` | Phase 1 |
| `apogee_positions[selected_inclination]` | Phase 1 |
| Mean wind profile | Arithmetic mean of the `.npz` ensemble, computed at load time |
| Premature-main descent rate profile `v_descent(z)` | Derived from main parachute Cd and vehicle mass |
| Azimuth candidates | −90° to +90°, integer (180 values) |
| Buffered danger area boundary | From geometry |

**Process:**

For each azimuth candidate:

1. Rotate `apogee_positions[selected_inclination]` by azimuth to obtain apogee position.
2. Compute wind drift vector:

$$\vec{d} = \sum_{z} \frac{\vec{v}_{wind}(z)}{v_{descent}(z)} \cdot \Delta z$$

3. Compute `landing_centroid = apogee_position + drift_vector`.
4. Discard candidate if centroid falls outside buffered danger area.

**Outputs:**

- `feasible_azimuths` — list of surviving integer azimuth candidates

### 13.4 Phase 3 — Azimuth Optimisation

**Function:** `optimise_azimuth(feasible_azimuths, selected_inclination, apogee_positions)`

**Inputs:**

| Parameter | Value |
|-----------|-------|
| Search space | `feasible_azimuths` |
| Scenario | Premature main only |
| Ascent | Deterministic — max total impulse, `apogee_positions[selected_inclination]` reused |
| Active uncertainty | Wind profile perturbations only |
| Frozen uncertainties | Launch rail angles, total impulse (max), fin cant |
| Sims per iteration | 150 |
| Max iterations | 20 |

**Process:**

1. Initialise Gaussian Process surrogate over `feasible_azimuths`.
2. Select 3–5 initial evaluation points via Sobol sequence.
3. For each BO iteration:
   - Run 150 premature-main Monte Carlo sims at proposed azimuth.
   - Compute `p_success = fraction of landing points inside buffered danger area`.
   - Update GP with observation `(azimuth, p_success)` and known noise `σ² = p(1−p)/n`.
   - Select next candidate via Upper Confidence Bound (UCB) acquisition function.
4. Terminate when expected improvement < 0.5% or 20 iterations reached.

**Outputs:**

- `top_candidates` — top 3 azimuths by GP posterior mean `p_success`
- `gp_model` — surrogate model retained for diagnostics

### 13.5 Phase 4 — Candidate Validation

**Function:** `validate_candidates(top_candidates, selected_inclination)`

**Inputs:**

| Parameter | Value |
|-----------|-------|
| Candidates | `top_candidates` (3 azimuths) |
| Scenario | Premature main only |
| Active uncertainties | Wind, total impulse, inclination, azimuth, fin cant (full set) |
| Sims per candidate | 500 |

**Process:**

1. Run 500 premature-main sims per candidate with the full uncertainty set.
2. Compute 99.7th percentile landing contour per candidate.
3. Select `optimal_azimuth` as the candidate with greatest margin inside the buffered danger area boundary.

**Outputs:**

- `optimal_azimuth`
- Per-candidate landing distributions and 99.7th percentile contours (diagnostics)

### 13.6 Simulation Budget

| Phase | Method | Sims | Notes |
|-------|--------|------|-------|
| 1 — Inclination selection | `select_inclination` | 6 | Deterministic, 3-DOF |
| 2 — Azimuth narrowing | `narrow_azimuth_bounds` | 0 | Analytical only |
| 3 — BO optimisation | `optimise_azimuth` | ~3,000 | Wind uncertainty only |
| 4 — Candidate validation | `validate_candidates` | ~1,500 | Full uncertainty |
| **Total (optimisation only)** | | **~4,500** | |

After optimisation, the standard MC analysis runs as normal, bringing the combined total to ~8,500 sims.


## 14 Geometry

### 14.1 Danger Area & Buffered Danger Area

The **danger area** is a 3D volume defined by a footprint polygon (from `danger_area.geojson`) and an altitude ceiling (from `simulation.yaml`). Its purpose is to define the **buffered danger area**: the danger area footprint with a configurable inward buffer applied. Setting the buffer to zero makes the buffered danger area identical to the danger area.

Buffer computed at startup via Shapely `buffer(-distance)`, then `simplify(~50 m)` to smooth, then largest polygon kept if MultiPolygon results. The altitude ceiling applies identically to both.

All trajectory containment checks (§11.1) and optimisation containment checks (§13) are evaluated against the buffered danger area.

### 14.2 Coastline

Optional. GeoJSON polygon of the on-land area. If the `coastline` key is omitted from the `site` section of `simulation.yaml`, the coastline compliance check (§11.1) is skipped.

The direction of the check is controlled by `coastline_mode` in `simulation.yaml`:

| `coastline_mode` | Pass condition |
|-----------------|----------------|
| `"sea"` (default) | Landing point is **outside** the polygon (at sea) |
| `"land"` | Landing point is **inside** the polygon (on land) |

> **Not yet implemented.** `coastline_mode` is a planned parameter; only `"sea"` behaviour is currently available.

### 14.3 Observation Stations

Optional. Coordinates and radii in `simulation.yaml`.

### 14.4 Miscellaenous Output Map Markers
Optional. Coordinates in `simulation.yaml`. Added to map figure(s).

## 15 Input Files

| File | Format | Content |
|------|--------|---------|
| `simulation.yaml` | YAML | See §15.1 |
| `vehicle.yaml` | YAML | See §15.2 |
| `motor.eng` | .eng | Thrust curve, propellant mass, motor mass (path in `vehicle.yaml`) |
| `aero_tables/` or `aero.csv` | CSV | Aeroplot coefficient tables — directory or single file (path in `vehicle.yaml`; §6.1) |
| `wind_profiles.npz` | NumPy `.npz` | Wind profile ensemble (§5.1) |
| `danger_area.geojson` | GeoJSON | Danger area footprint polygon |
| `coastline.geojson` | GeoJSON | On-land polygon (optional — omit `coastline` key in `simulation.yaml` to disable; §14.2) |

### 15.1 Simulation Config (`simulation.yaml`)

The file is divided into four top-level sections. All file paths are resolved relative to the directory containing `simulation.yaml`.

```yaml
vehicle: "vehicle.yaml"           # path to vehicle.yaml (motor and aero_tables paths are in vehicle.yaml)

site:
  latitude: 58.6104700            # degrees
  longitude: -4.9434804           # degrees
  ballistic_exclusion_radius: 500 # metres — minimum ballistic landing distance from launch
                                  # site; used when inclination is "auto" (§13.2)
  launch_observation_radius: 200  # metres — radius of the automatic launch site observation
                                  # station added to every LOS check (§14.3)
  altitude_ceiling: 16764         # metres (55,000 ft)
  danger_area: "danger_area.geojson"
  # coastline: "coastline.geojson"  # omit to disable coastline check
  # coastline_mode: "sea"           # "sea" (default) or "land" — see §14.2
  observation_stations:
    - name: "MOD Range Control"
      latitude: 58.40
      longitude: -4.76
      radius: 10000               # metres
  map_markers:
    - name: "Durness"
      latitude: 58.40
      longitude: -4.76

launch:
  rail:
    azimuth: "auto"               # degrees clockwise from North, or "auto"
    azimuth_range: [-90, 90]      # [min, max] integer search range — required when azimuth is "auto"
    inclination: "auto"           # degrees from horizontal, or "auto"
    inclination_range: [85, 90]   # [min, max] integer search range — required when inclination is "auto"
    length: 4.0                   # metres
  wind_profiles: "wind_profiles.npz"
  surface_wind:                   # omit this section entirely to disable surface override
    speed_ms: 5.0                 # m/s
    bearing_deg: 270.0            # degrees clockwise from North
    blend_height_m: 300           # metres AGL

monte_carlo:
  samples: 1000                   # per scenario
  seed: 42
  uncertainties:
    azimuth_sigma: 1.0            # degrees
    inclination_sigma: 0.5        # degrees
    fin_cant_sigma: 0.02          # degrees
    impulse_factor_sigma: 0.067   # fraction of total impulse (e.g. 0.067 = 6.7%)
  acceptance:
    compliance_threshold: 0.997   # fraction (e.g. 0.997 = 99.7%)
    buffer_distance: 1000         # metres inward from danger area boundary
    sm_transition_mach: 0.91      # Mach dividing subsonic / supersonic SM check (§11.1);
                                  # definition varies between tools — default follows RASAero II
    sm_subsonic_min: 1.0          # calibres (M < sm_transition_mach)
    sm_supersonic_min: 2.0        # calibres (M >= sm_transition_mach)
    aoa_max: 12.0                 # degrees
    sm_aoa_threshold: 5.0         # degrees: SM check applies when AoA < this
    sea_check_scenarios:          # scenarios subject to coastline check (§9.3, §14.2)
      - nominal
      - ballistic
      - drogue_only
      - premature_main
    los_check_scenarios:          # landing must be visible from at least one station (§9.3)
      - ballistic
      - drogue_only

# verification:                       # optional — single-trajectory pre-run comparison (§18.1)
#   reference_trajectory: "ref.csv"   # CSV from any flight simulator (not yet implemented)
#   altitude_tol_m: 50
#   mach_tol: 0.05
#   sm_tol_cal: 0.3
#   mass_tol_kg: 0.1
#   inertia_tol_pct: 5.0
```

### 15.2 Vehicle Config (`vehicle.yaml`)

All distances are from the nosecone tip in metres. Dry mass properties are derived automatically from wet + propellant — the user never specifies dry values.

```yaml
motor: "motor.eng"          # path to RASP .eng file (relative to this file)
aero_tables: "aero_tables"  # path to aeroplot CSV file or directory of CSV files (§6.1)
# fins_aero_table: "aero_tables/fins.csv"  # optional explicit fins component path;
                                            # if omitted, file containing "fin" in stem is used

geometry:
  diameter: 0.130           # m — reference diameter (A_ref = π·d²/4 is derived)
  length: 2.6               # m — total vehicle length
  nozzle_position: 2.55     # m — nozzle exit plane from nosecone tip (for C₂R)
  nozzle_diameter: 0.08     # m — nozzle exit diameter (for thrust altitude correction)
  fin_cp_radius: 0.095      # m — fin CP spanwise distance from centreline (for roll)

mass:
  wet_mass: 26.5            # kg — total launch mass (vehicle + propellant)
  wet_cg: 1.15              # m — wet vehicle CG from nosecone tip
  wet_motor_cg: 1.82        # m — loaded motor CG from nosecone tip; used as
                            #     fixed propellant CG (inside-out burn model)
  propellant_inertia_roll: 0.05     # kg·m² — propellant roll inertia about roll axis
  propellant_inertia_lateral: 0.8  # kg·m² — propellant lateral inertia about propellant CG
  wet_inertia_lateral: 5.2         # kg·m² — wet vehicle lateral inertia about wet CG
  wet_inertia_roll: 0.012          # kg·m² — wet vehicle roll inertia about roll axis

recovery:
  drogue:                   # omit this section for no drogue (main-only is valid)
    cd: 2.0                 # drag coefficient
    area: 0.15              # m² — reference area (CdA_drogue = cd × area)
    threshold: apogee       # deploy at apogee, or numeric altitude in m AGL
  main:                     # omit this section for no main parachute
    cd: 2.0
    area: 2.8               # m²
    threshold: 305          # m AGL (or "apogee")
```

**Validation:** a drogue without a main is not a valid configuration and will raise an error at load time.


## 16 Outputs

### 16.1 Per-Sample CSV

One row per sample. Columns: sample_id, scenario, compliant, apogee_m, apogee_lat/lon, landing_lat/lon, landing_at_sea (or none if no coastline), in_buffer, below_ceiling, in_coverage, stability_compliant, min_SM_subsonic, min_SM_supersonic_cal, max_AoA_deg, peak_mach, peak_altitude_ft, flight_time_s, wind_profile_index, impulse_factor, azimuth_deg, inclination_deg, fin_cant_deg.

### 16.2 Run Summary

YAML: per-scenario compliant/non-compliant counts, pass/fail, statistics (mean/std/min/max apogee, landing distance, margins) and any warnings and or error messages. When optimisation was run, the summary includes the selected azimuth, inclination, and per-phase diagnostics.

### 16.3 Dispersion Data

Landing lat/lon CSV for all samples.

### 16.4 Replay

No full trajectories saved by default. Replay any sample by (master_seed, run_index, sample_index). Option to auto-replay all non-compliant samples. Replay opens two figures automatically (all replayed samples overlaid on shared figures):

1. **3D isometric** — trajectory in NED space with map on the ground plane, matching `dispersion_plot.png` style. Coloured by descent scenario; black if terminated early due to stability/AoA violation.
2. **Altitude vs time** — full flight from rail exit to landing.

### 16.5 Saved Plots

`dispersion_plot.png` and `altitude_plot.png`. Visual style from existing plotting scripts (to be provided to implementer). All outputs saved to `./results/<timestamp>/`.


## 17 Command-Line Interface

### 17.1 Technology

All interaction via the command line. No graphical UI. Progress bars, tables, and status output use `rich`. Colour/highlight for pass/fail verdicts. Plots saved to disk via `matplotlib` (no interactive display required).

### 17.2 Commands

All commands are run from inside the project directory (the git repo root, which
is also the Python package root). `__main__.py` is the entry point.

```
python . run example/simulation.yaml
python . run example/simulation.yaml --no-warn
python . replay results/<timestamp>/summary.yaml --seed 42 --run 3 --sample 117
python . replay results/<timestamp>/summary.yaml --non-compliant
```

`run` is the primary command. First argument is always the simulation configuration file. If `azimuth` or `inclination` is `"auto"`, optimisation runs first, with phase progress displayed, before the main MC analysis.

`--no-warn` suppresses the interactive blocking prompt for warnings; warnings still appear in the log and results summary. *(Not yet implemented — see §9.3.)*

`replay` is the simulation replay command. First argument is always the simulation results summary file. The data comes from the same directory as this file. Replay is implemented as a function inside `montecarlo.py` and called directly by the CLI.

### 17.3 Run Output

**Progress:** `rich` progress bar per scenario. Format: `Nominal ████████░░ 342/1000 — ~01:12 remaining`. During optimisation, a separate progress display shows the current phase and iteration.

**Results table:** Printed to terminal after completion using `rich.table`:

```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Scenario         ┃ Samples ┃ Compliant ┃ Non-Compliant ┃ ≥ 99.7% ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Nominal          │    1000 │      1000 │             0 │  PASS   │
│ Ballistic        │    1000 │      1000 │             0 │  PASS   │
│ Drogue-only      │    1000 │       998 │             2 │  PASS   │
│ Premature main   │    1000 │       997 │             3 │  PASS   │
├─────────────────┼─────────┼───────────┼───────────────┼─────────┤
│ Total            │    4000 │      3995 │             5 │         │
└─────────────────┴─────────┴───────────┴───────────────┴─────────┘
```

PASS rendered in green, FAIL in red.

**Verdict:** Printed below the table. "ALL ACCEPTANCE CRITERIA MET" (green) or "ACCEPTANCE CRITERIA NOT MET" (red) with specific failure reasons listed.

**Output path:** Final line prints the results directory path.


## 18 Verification

### 18.1 Trajectory Comparison Tool

> **Not yet implemented.**

Before the main MC run, an optional single-trajectory comparison against any external flight simulator output. Configured via a `verification:` block in `simulation.yaml` (§15.1). The reference CSV may come from any tool; columns are matched case-insensitively. Quantities compared: altitude, Mach, stability margin, mass, lateral inertia vs time. Tolerance bands are configurable per-quantity.

Reference data plotted in grey with the configured tolerance band; simulator output overlaid in green (all within tolerance) or red (any out of tolerance). Pass/fail printed to console and recorded in `summary.yaml`. On failure, the figure opens and the user is prompted whether to continue.

Default tolerances (configurable in `simulation.yaml`):

| Quantity | Default tolerance |
|----------|------------------|
| Altitude | ± 50 m |
| Mach | ± 0.05 |
| Static margin | ± 0.3 cal |
| Mass | ± 0.1 kg |
| Lateral inertia | ± 5 % |

### 18.2 Unit Tests

ISA vs published tables. Quaternion/DCM/frame transforms. Launch rail exit velocity (analytical). Terminal velocity (analytical). Aero interpolation spot checks. C₂A/C₂R hand calculations. .eng parser. AoA computation. Wind `.npz` loader (shape validation, interpolation).

### 18.3 Integrator Convergence

10× tighter tolerance → < 0.5% apogee change, < 0.1% landing change (100 samples).

### 18.4 Statistical Stability

5 seeds × 1,000 samples: mean stable to < 2%, std to < 5%.

### 18.5 Dynamics

Small-perturbation damping rate matches exp(−C₂t/(2I_L)). Oscillation frequency ≈ √(½V²A_ref C_Nα(CP−CG)/I_L). Roll → coning. Large-AoA restoring uses nonlinear tables.


## 19 Performance

| Metric | Target |
|--------|--------|
| Standard MC suite (4×1,000) | < 3 min |
| Optimisation + MC suite | < 5 min |
| Hardware | 2020-era laptop, 4-core, 8 GB |
| JIT warmup | < 30 s |

Strategy: Numba `@njit` hot loops, contiguous arrays, parallelism, adaptive stepping.


## 20 Architecture

The git repo root is the Python package root. All modules are imported directly
(e.g. `import atmosphere`). Run with `python . run …` from inside the directory.

```
leeds-flight-simulator/      ← git repo root = package root
├── __main__.py              # CLI entry point (click)
├── cli.py                   # Command definitions, rich output
├── config.py                # YAML → dataclasses; load_motor / MotorData; active_scenarios
├── atmosphere.py            # ISA (Numba)
├── wind.py                  # .npz loader, surface_wind blending, interpolation
├── aerodynamics.py          # Aero tables, C_Nα, forces, roll torques (Barrowman)
├── motor.py                 # Motor physics: thrust/mass/CG/MoI @njit functions
├── dynamics.py              # 6DoF + 3DoF derivatives (Numba), launch rail phase, descent CdA
├── integrator.py            # State integrator to actualy perform the simulations
├── geometry.py              # Polygons, buffer, containment
├── montecarlo.py            # MC orchestration, parallelism, acceptance checking, replay
├── optimisation.py          # Inclination/azimuth optimisation (§13)
├── outputs.py               # CSV, YAML serialisation, plot generation
├── verification/
│   ├── test_config.py       # config.py — YAML loading and dataclass construction
│   ├── test_isa.py          # ISA vs published tables
│   ├── test_aero_interp.py  # aero table interpolation spot checks
│   ├── test_c2_damping.py   # C₂A/C₂R hand calculations
│   ├── test_eng_parser.py   # .eng parser and motor physics
│   ├── test_wind_loader.py  # .npz loading, surface wind blending
│   ├── test_frames.py       # quaternion/DCM/frame transforms
│   ├── test_launch_rail.py  # launch rail exit velocity (analytical)
│   ├── test_descent.py      # terminal descent (analytical)
│   ├── test_dynamics.py     # 6DoF dynamics
│   └── compare_rasaero.py   # full trajectory comparison vs RASAero
└── example/
    ├── vehicle.yaml
    ├── motor.eng
    ├── aero_tables/
    ├── wind_profiles.npz
    ├── danger_area.geojson
    ├── coastline.geojson
    └── simulation.yaml
```