# Leeds Flight Simulator (LFS)

A six-degree-of-freedom Monte Carlo flight simulator for single-stage, passively stabilised, axisymmetric sounding rockets. Generates flight safety analysis evidence suitable for a CAA large rocket permission safety case under Article 96 of the Air Navigation Order 2016.

The simulator covers launch rail exit to landing. It evaluates between one and four descent scenarios (depending on the vehicle's recovery system configuration) and checks trajectory containment, sea-landing exclusion, observation coverage, and aerodynamic stability against configurable acceptance criteria. It can automatically optimise launch azimuth and inclination when these are set to `"auto"`.

The simulator runs entirely offline. The only network access is to fetch base map tiles for the dispersion plot, which are cached locally after the first download.

---

## Background

This tool was originally developed for the Gryphon II Block II (G2B2) launch by the Leeds University Rocketry Association (LURA) -- a supersonic sounding rocket targeting the UKRA amateur altitude record, launching from Cape Wrath, Scotland. The safety case for that launch is available here:

> **G2B2 Safety Case:** [https://leedsrocketry.co.uk/g2b2-safety-case](https://leedsrocketry.co.uk/g2b2-safety-case)
> <!-- TODO: confirm URL -->

Although built for G2B2, the simulator is entirely generic. Any team launching a passively stabilised, axisymmetric rocket under a CAA permission (or similar regulatory framework) can use it by supplying their own configuration files.


## Installation

### Prerequisites

- Python 3.10 or later
- Python packages: `numpy`, `numba`, `scipy`, `shapely`, `pyyaml`, `matplotlib`, `rich`, `click`, `scikit-optimize`

### Setup

```
git clone https://github.com/leedsrocketry/leeds-flight-simulator.git
cd leeds-flight-simulator
pip install numpy numba scipy shapely pyyaml matplotlib rich click scikit-optimize
```

Verify:

```
python . --help
```


## Quick Start

```
python . run example/simulation.yaml
```

This runs all active descent scenarios and prints a results table to the terminal:

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

ALL ACCEPTANCE CRITERIA MET

Results saved to: ./results/20260620_093000/
```

If `azimuth` or `inclination` is set to `"auto"` in `simulation.yaml`, the optimisation routine runs first, then the main Monte Carlo analysis proceeds using the optimised values.


## Key Concepts

### Monte Carlo Analysis

Rather than simulating a single trajectory, the simulator runs many trajectories with randomised conditions -- different wind profiles, motor impulse, launch rail alignment, and fin cant -- producing a cloud of landing points. The `compliance_threshold` in `simulation.yaml` determines whether the flight is safe.

### Descent Scenarios

Every sample shares the same ascent. At apogee, the sample follows one of up to four descent branches:

| Scenario | Description | Significance |
|----------|-------------|--------------|
| `nominal` | Drogue at apogee (if present), main at deployment altitude | Expected case |
| `ballistic` | No parachutes | Worst-case impact energy, minimal drift |
| `drogue_only` | Drogue deploys, main fails | Moderate drift, high descent rate |
| `premature_main` | Main opens at apogee | Maximum drift -- bounding case for containment |

Which scenarios are active depends on the vehicle's recovery configuration:

| Recovery config | `nominal` | `ballistic` | `drogue_only` | `premature_main` |
|----------------|-----------|-------------|---------------|-----------------|
| Both drogue and main, `main.threshold` numeric | ✓ | ✓ | ✓ | ✓ |
| Both drogue and main, `main.threshold` = `"apogee"` | ✓ | ✓ | ✓ | — |
| Main only, `main.threshold` numeric | ✓ | ✓ | — | ✓ |
| Main only, `main.threshold` = `"apogee"` | ✓ | ✓ | — | — |
| No parachutes | ✓ | — | — | — |

`premature_main` is suppressed when `main.threshold = "apogee"` because apogee deployment is the earliest possible deployment — no earlier failure mode exists.

Each active scenario is a separate run. Scenarios listed in `sea_check_scenarios` or `los_check_scenarios` that are not active for the current vehicle are silently skipped with a warning.

### Wind Profiles

Wind data is supplied as a NumPy `.npz` file containing a pre-generated ensemble of perturbed wind profiles. The simulator has no knowledge of how the profiles were generated; all source-specific logic (EarthGRAM, GFS, ECMWF, radiosonde, perturbation modelling) is handled by a separate wind profile generator tool.

The `.npz` file must contain:

| Array key | Shape | Description |
|-----------|-------|-------------|
| `altitude_m` | `(M,)` | Altitude grid in metres AGL, monotonically increasing |
| `wind_east_ms` | `(N, M)` | Eastward wind component per profile (m/s) |
| `wind_north_ms` | `(N, M)` | Northward wind component per profile (m/s) |

`N` must be >= `samples` under `monte_carlo` in `simulation.yaml`. Sample `i` uses profile `i`.


## Input Files

Example input files are provided in `example/`. The values in those files are justified in the G2B2 safety case (see Background).

### `vehicle.yaml`

Defines the vehicle's physical properties. All distances are in metres from the nosecone tip. All masses are in kg. All distances are in meters. Dry mass properties are derived automatically from the wet properties and propellant data — you do not need to specify them.

The aerodynamic reference area is defined as A_ref = pi * d**2 / 4, where d is the reference diameter. This is the same as RASAero.

The rocket's overall length is used as the reference length for Reynolds number calculation. This is the same as RASAero.

See `example/vehicle.yaml` for a fully annotated example.

### `motor.eng`

Standard RASP/RockSim `.eng` file. Downloadable from [thrustcurve.org](https://www.thrustcurve.org) for most certified motors.

### Aerodynamic tables (`aero_tables`)

The `aero_tables` field in `vehicle.yaml` can point to either a **single CSV file** or a **directory of CSV files**:

- **Single file** — treated as full-vehicle data; per-component force and moment computation is disabled (forces act at the whole-vehicle CP). A warning is issued.
- **Directory** — one CSV per aerodynamic component (nosecone, body tube, fin set, boattail, etc.), enabling full 6-DoF simulation with per-component local angle-of-attack forces and moments, pitch/yaw damping, and roll torques.

Each CSV must contain:

```
Mach, Reynolds, AoA_deg, CA, CN, CP_m
```

CP_m is in metres from the nosecone tip. The grid need not be uniformly spaced.

### `danger_area.geojson`

A GeoJSON polygon defining the danger area footprint. **Coordinates are `[longitude, latitude]` per the GeoJSON spec.**

Useful tools:
- NATS Aeronautical Information Publication: [nats.aero/ais/aip](http://www.nats.aero/ais/aip) — for tracing danger area boundaries from official charts
- Online GeoJSON editor: [geojson.io](https://geojson.io) — for drawing and editing polygons

### `coastline.geojson`

Optional. A GeoJSON polygon delineating the **on-land** area. The direction of the compliance check is set by `coastline_mode` under `site` in `simulation.yaml`:

| `coastline_mode` | Pass condition |
|-----------------|----------------|
| `"sea"` (default) | Landing point is **outside** the polygon (at sea) |
| `"land"` | Landing point is **inside** the polygon (on land) |

Omit the `coastline` key from the `site` section of `simulation.yaml` to disable the check entirely.

### `simulation.yaml`

Main simulation configuration. All file paths are resolved relative to the directory containing `simulation.yaml`. See `example/simulation.yaml` for a fully annotated example with comments for every parameter.


## Acceptance Criteria

A sample is compliant if **all** of the following hold:

- **Stability and AoA** -- during powered and coasting flight, whenever AoA < `sm_aoa_threshold`: static margin >= `sm_subsonic_min` calibres below `sm_transition_mach` Mach, or >= `sm_supersonic_min` calibres at or above it. AoA must not exceed `aoa_max` at any point. Violation terminates the sample immediately. This is the same approach as RASAero, with the addition of a maximum AoA check.
- **Containment** -- landing point inside the buffered danger area and peak altitude below the buffered altitude ceiling.
- **Coastline check** -- if a coastline file is provided, the landing point must satisfy the configured `coastline_mode` (see `coastline.geojson` above).
- **Observation coverage** -- landing within the configured radius of at least one observation station. Applied only to scenarios listed in `los_check_scenarios`.

A run passes if >= `compliance_threshold` fraction of samples are compliant. All active scenario runs must pass.


## Optimisation

When `azimuth` and/or `inclination` are set to `"auto"`, a four-phase optimisation routine runs before the main Monte Carlo analysis:

1. **Phase 1 -- Inclination selection.** Runs deterministic 3-DoF (translation only) simulations at each integer inclination in `inclination_range` (no wind). Selects the steepest inclination (to maximise apogee) whose ballistic landing point is both outside `ballistic_exclusion_radius` from the launch site and inside the buffered danger area.

2. **Phase 2 -- Azimuth narrowing.** Analytically filters integer azimuths in `azimuth_range` by estimating wind drift from the mean wind profile, discarding any azimuth whose estimated premature-main landing centroid falls outside the buffered danger area.

3. **Phase 3 -- Azimuth optimisation.** Uses Bayesian optimisation (Gaussian Process with UCB acquisition) over the surviving azimuths. Each iteration runs premature-main Monte Carlo simulations with wind uncertainty to estimate the containment probability.

4. **Phase 4 -- Candidate validation.** Validates the top candidate azimuths with the full uncertainty set (wind, impulse, launch angles, fin cant). Selects the azimuth with the greatest containment margin.

If only inclination is `"auto"`, only Phase 1 runs. If only azimuth is `"auto"`, the provided inclination is used and Phases 2--4 run.


## Surface Wind Override

When a `surface_wind` sub-section is present under `launch` in `simulation.yaml`, a user-specified surface wind replaces the lower portion of each wind profile. The override is specified as a speed (m/s) and bearing (degrees clockwise from North) and blends linearly into the profile wind between ground level and `blend_height_m`. Omit the `surface_wind` section entirely to disable the override. This is useful on launch day when you have an anemometer reading at the pad but are using forecast or balloon data for the upper atmosphere.


## Multi-Day Wind

`wind_profiles` in `simulation.yaml` can point to either a single `.npz` file or a directory of `.npz` files:

```yaml
launch:
  wind_profiles: "wind_profiles/"   # directory — one run per file inside
  # wind_profiles: "wind.npz"       # single file — current behaviour
```

When a directory is given, the simulator runs the full analysis independently for each `.npz` file found inside it. Output folders are suffixed with the wind profile filename to keep results from each day separate, e.g. `results/20260620_093000_day1/`, `results/20260620_093000_day2/`. This allows a multi-day wind campaign to be assessed in a single invocation.


## Replaying Samples

Every sample is deterministic given the master seed, run index, and sample index. No full trajectory data is stored.

To replay a specific sample:

```
python . replay results/<timestamp>/summary.yaml --seed 42 --run 3 --sample 117
```

To automatically replay all non-compliant samples from a completed run:

```
python . replay results/<timestamp>/summary.yaml --non-compliant
```

Replay outputs a detailed time history (position, velocity, attitude, Mach, AoA, stability margin, damping coefficients, forces, moments) as a CSV, and opens two figures automatically. If multiple samples are replayed in one invocation, all trajectories are overlaid on the same figures:

1. **3D isometric** — trajectory in NED space with the map overlaid on the ground plane, matching the `dispersion_plot.png` style. Coloured by descent scenario; black if the sample was terminated early due to a stability or AoA violation.
2. **Altitude vs time** — full flight from rail exit to landing.


## Output Files

Results are saved to `./results/<timestamp>/`:

| File | Contents |
|------|----------|
| `summary.yaml` | Run metadata, pass/fail, per-scenario statistics, optimisation diagnostics (if applicable) |
| `samples.csv` | One row per sample with all compliance details and stochastic inputs |
| `dispersion.csv` | Landing lat/lon for all samples |
| `dispersion_plot.png` | Landing points colour-coded by scenario, with danger area, buffer, coastline, observation circles, and launch site overlaid |
| `altitude_plot.png` | Mean altitude profile for each scenario |


## Tolerance Auto-Calibration

At the start of execution, the simulator runs a small run of samples at very tight integrator tolerances, then re-runs at progressively looser settings. It automatically selects and reports the loosest tolerance that maintains acceptable output deviation.

## Warnings

Warnings are raised when the simulator detects a configuration that is unusual but not an error (e.g. a single aero table is provided, falling back to 3-DoF mode). By default, warnings are **blocking**: the simulator pauses and prompts you to acknowledge each one before continuing.

```
WARNING: Only one aeroplot CSV found. Per-component force and moment computation disabled; forces will act at the whole-vehicle CP.
Press Enter to continue, or Ctrl-C to abort.
```

Warnings still appear in the run log and results summary regardless.

To suppress the interactive prompt (e.g. in automated pipelines), pass `--no-warn`:

```
python . run example/simulation.yaml --no-warn
```

Warnings remain in the log and summary; only the interactive pause is suppressed.


## Verification

Before relying on the simulator for a safety case, verify it against an independent tool.

**Unit tests:**

```
python -m pytest verification/
```

Checks ISA against published tables, quaternion maths, launch rail exit velocity, terminal descent, aero interpolation and per-component local AoA damping, the `.eng` parser, AoA computation, and wind `.npz` loading.

**Trajectory verification tool:**

Before the main MC run, you can supply a reference CSV from any other flight simulator for a single-trajectory comparison. Add the path under `verification` in `simulation.yaml`:

```yaml
verification:
  reference_trajectory: "rasaero_output.csv"  # optional — omit to skip
  # Tolerance bands (all configurable):
  altitude_tol_m: 50
  mach_tol: 0.05
  sm_tol_cal: 0.3
  mass_tol_kg: 0.1
  inertia_tol_pct: 5.0
```

The reference CSV must contain columns for time and at least one of: altitude, Mach, stability margin, mass, lateral inertia. Column names are matched case-insensitively; any column not present is skipped.

The tool plots each quantity with the reference in grey and the simulator output overlaid. If all quantities are within the configured tolerance bands, the output is rendered in green and the run continues automatically. If any quantity falls outside tolerance, the output is rendered in red, the figure is opened for inspection, and you are asked whether to proceed.

Pass/fail is also printed to the console and recorded in `summary.yaml`.

## Troubleshooting

**Simulation takes much longer than expected** -- Check that Numba is installed and working. The first run includes JIT compilation overhead (~30 s). If every run is slow, Numba may be falling back to interpreted mode; check for warnings.

**All samples are non-compliant** -- Check that your launch rail azimuth points the trajectory into the danger area. Check that the danger area GeoJSON coordinates are `[longitude, latitude]`, not reversed. Check the vehicle's static stability margin.

**Dispersion plot looks wrong or empty** -- Ensure the danger area and coastline GeoJSON files cover the area where the rocket actually lands. GeoJSON uses `[lon, lat]` coordinate order.


## Operational Workflow

A typical campaign uses the simulator at three stages:

**Safety case (months before)** -- Run with climatological wind profiles representing the full spread of weather for the planned launch month. This forms the basis for the safety case submitted to the CAA.

**Operations planning (days before)** -- Run with forecast-derived wind profiles (GFS, ECMWF, or similar). If the analysis fails, conditions may not be suitable for launch.

**Launch day go/no-go (hours before)** -- Run with radiosonde-derived wind profiles from the launch site. Enable the surface wind override with the current anemometer reading. The launch director uses this result alongside all other go/no-go criteria to make the final call. Re-run with fresh data if conditions change.


## Contact

For questions, bug reports, or contributions:
- **Toby Thomson** -- el21tbt@leeds.ac.uk, me@tobythomson.co.uk
- **LURA Team** -- launch@leedsrocketry.co.uk


## Licence

<!-- TODO: Add licence information -->


## References

- Mandell, G. K., Caporaso, G., and Bengen, W. P. (1973). *Topics in Advanced Model Rocketry*. MIT Press.
- Barrowman, J. S. (1967). *The Practical Calculation of the Aerodynamic Characteristics of Slender Finned Vehicles*. MSc thesis, The Catholic University of America.
- Niskanen, S. (2009). *Development of an Open Source model rocket simulation software*. MSc thesis, Helsinki University of Technology.
- Zipfel, P. H. (2014). *Modeling and Simulation of Aerospace Vehicle Dynamics*, 3rd ed. AIAA.
- Dormand, J. R. and Prince, P. J. (1980). A family of embedded Runge-Kutta formulae. *Journal of Computational and Applied Mathematics*, 6(1), 19--26.
