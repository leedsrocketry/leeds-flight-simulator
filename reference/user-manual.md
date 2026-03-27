# Leeds Flight Simulator (LFS)

A six-degree-of-freedom Monte Carlo flight simulator for single-stage, passively stabilised, axisymmetric sounding rockets. Generates flight safety analysis evidence suitable for a CAA large rocket permission safety case under Article 96 of the Air Navigation Order 2016.

The simulator covers launch rail exit to landing. It evaluates four descent failure scenarios and checks trajectory containment, sea-landing exclusion, observation coverage, and aerodynamic stability against configurable acceptance criteria. It can automatically optimise launch azimuth and inclination when these are set to `"auto"`.

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
- RASAero II (for generating aerodynamic coefficient tables -- not needed at runtime)

### Setup

```
git clone https://github.com/leedsrocketry/leeds-flight-simulator.git
cd leeds-flight-simulator
pip install numpy numba scipy shapely pyyaml matplotlib rich click scikit-optimize
```

Verify:

```
python -m leeds-flight-simulator --help
```

The simulator runs entirely offline. No network access is required.


## Quick Start

```
python -m leeds-flight-simulator run input/simulation.yaml
```

This runs all four descent scenarios (4,000 samples by default) and prints a results table to the terminal:

```
+-----------------+---------+-----------+---------------+---------+
| Scenario        | Samples | Compliant | Non-Compliant | >= 99.7%|
+-----------------+---------+-----------+---------------+---------+
| Nominal         |    1000 |      1000 |             0 |  PASS   |
| Ballistic       |    1000 |      1000 |             0 |  PASS   |
| Drogue-only     |    1000 |       998 |             2 |  PASS   |
| Premature main  |    1000 |       997 |             3 |  PASS   |
+-----------------+---------+-----------+---------------+---------+
| Total           |    4000 |      3995 |             5 |         |
+-----------------+---------+-----------+---------------+---------+

ALL ACCEPTANCE CRITERIA MET

Results saved to: ./results/20260620_093000/
```

PASS is rendered in green, FAIL in red. If any scenario fails, specific reasons are listed below the verdict.

If `azimuth` or `inclination` is set to `"auto"` in `simulation.yaml`, the optimisation routine runs first (with its own progress display), then the main Monte Carlo analysis proceeds using the optimised values.

On a 2020-era laptop, a standard run completes in under 3 minutes. Optimisation adds roughly 2 minutes.


## Key Concepts

### Monte Carlo Analysis

Rather than simulating a single trajectory, the simulator runs thousands of trajectories with randomised conditions -- different wind profiles, motor impulse, launch rail alignment, and fin cant -- producing a cloud of landing points. The acceptance criteria (default: 99.7% of samples compliant) determine whether the flight is safe.

### Descent Scenarios

Every sample shares the same ascent. At apogee, the sample follows one of four descent branches:

| Scenario | Description | Significance |
|----------|-------------|--------------|
| Nominal | Drogue at apogee, main at deployment altitude | Expected case |
| Ballistic | No parachutes | Worst-case impact energy, minimal drift |
| Drogue-only | Drogue deploys, main fails | Moderate drift, high descent rate |
| Premature main | Main opens at apogee | Maximum drift -- bounding case for containment |

Each scenario runs as a separate batch (default: 1,000 samples each).

### Wind Profiles

Wind data is supplied as a NumPy `.npz` file containing a pre-generated ensemble of perturbed wind profiles. The simulator has no knowledge of how the profiles were generated; all source-specific logic (EarthGRAM, GFS, ECMWF, radiosonde, perturbation modelling) is handled by a separate wind profile generator tool.

The `.npz` file must contain:

| Array key | Shape | Description |
|-----------|-------|-------------|
| `altitude_m` | `(M,)` | Altitude grid in metres AGL, monotonically increasing |
| `wind_east_ms` | `(N, M)` | Eastward wind component per profile (m/s) |
| `wind_north_ms` | `(N, M)` | Northward wind component per profile (m/s) |

`N` must be >= `num_samples` in `simulation.yaml`. Sample `i` uses profile `i`.


## Input Files

The simulator requires six input files. Example files are provided in `input/`.

### `vehicle.yaml`

Defines the vehicle's physical properties. All distances are in metres from the nosecone tip.

```yaml
# Vehicle geometry
diameter: 0.130          # Reference diameter (m)
length: 2.6              # Total length (m)
reference_area: 0.01327  # Pi * d^2 / 4 (m^2)
launch_rail_length: 4.0  # Launch rail length (m)

# Mass properties
wet_mass: 26.5           # Mass at launch (kg)
dry_mass: 14.7           # Mass after burnout (kg)
cg_dry: 1.15             # Dry CG from nosecone (m)
cg_propellant: 2.10      # Propellant CG from nosecone (m)

# Moments of inertia (kg*m^2)
I_R_wet: 0.012           # Roll, wet
I_R_dry: 0.008           # Roll, dry
I_L_wet: 5.2             # Lateral, wet
I_L_dry: 3.1             # Lateral, dry

# Nozzle
nozzle_exit: 2.55        # Nozzle exit from nosecone (m)

# Recovery
CdA_drogue: 0.15         # Drogue drag area (m^2)
CdA_main: 2.8            # Main drag area (m^2)
deploy_altitude_agl: 305 # Main deploy altitude above launch site (m)

# Roll model (Barrowman)
r_fin: 0.095             # Fin CP spanwise distance from centreline (m)
```

### `motor.eng`

Standard RASP/RockSim `.eng` file. Downloadable from [thrustcurve.org](https://www.thrustcurve.org) for most certified motors.

### `aero_tables/` Directory

One CSV per aerodynamic component, exported from RASAero II Aeroplots. Each file must contain:

```
Mach, Reynolds, AoA_deg, CA, CN, CP_m
```

CP_m is in metres from the nosecone tip. The grid need not be uniformly spaced. Typical components: nosecone, body tube, fin set, boattail. Per-component data is required for pitch/yaw damping and roll torque computation. If only one file is provided, the simulator treats it as full-vehicle data and disables damping and roll assessment (with a warning).

### `danger_area.geojson`

A GeoJSON polygon defining the danger area footprint. Coordinates are `[longitude, latitude]` per the GeoJSON spec. Create this from NATS airspace charts or by tracing the boundary in a GIS tool like QGIS.

### `coastline.geojson`

Optional GeoJSON polygon delineating land from sea. Set the path to `none` in `simulation.yaml` to disable the sea-landing compliance check.

### `simulation.yaml`

Main simulation configuration:

```yaml
# Launch site
launch_site:
  latitude: 58.6104700
  longitude: -4.9434804

# Launch rail orientation (nominal).
# Both fields required. Set to "auto" for optimisation.
launch_rail:
  azimuth: "auto"         # degrees clockwise from North, or "auto"
  inclination: "auto"     # degrees from horizontal, or "auto"

# Monte Carlo
mc:
  num_samples: 1000       # per scenario
  master_seed: 42

# Stochastic distributions
distributions:
  azimuth_sigma: 1.0            # degrees
  inclination_sigma: 0.5        # degrees
  fin_cant_sigma: 0.02          # degrees
  impulse_factor_sigma: 6.7     # percent

# Acceptance criteria
acceptance:
  compliance_threshold: 99.7  # percent
  buffer_distance: 1000       # metres inward from danger area
  altitude_ceiling: 16764     # metres (55,000 ft)
  sm_subsonic_min: 1.0        # calibres (M < 0.91)
  sm_supersonic_min: 2.0      # calibres (M >= 0.91)
  aoa_max: 12.0               # degrees
  sm_aoa_threshold: 5.0       # degrees: SM check applies when AoA < this

# Optimisation (required only when azimuth or inclination is "auto")
optimisation:
  min_safe_radius: 500        # metres -- only used when inclination is "auto"

# Observation stations
observation_stations:
  - name: "Range control"
    latitude: 58.40
    longitude: -4.76
    radius: 10000
  - name: "Cliff station"
    latitude: 58.60
    longitude: -4.95
    radius: 5000

# Map markers
map_markers:
  - name: "Durness"
    latitude: 58.40
    longitude: -4.76

# Input file paths
paths:
  vehicle: "input/vehicle.yaml"
  motor: "input/motor.eng"
  aero_dir: "input/aero_tables/"
  wind_profiles: "input/wind_profiles.npz"
  danger_area: "input/danger_area.geojson"
  coastline: "input/coastline.geojson"  # none to disable sea-landing check

# Surface wind override
surface_override:
  speed_ms: 5.0             # m/s
  bearing_deg: 270.0        # degrees clockwise from North
  blend_height_m: 300       # metres AGL; none = override disabled
```


## Acceptance Criteria

A sample is compliant if **all** of the following hold:

- **Stability and AoA** -- during powered and coasting flight, whenever AoA < `sm_aoa_threshold`: static margin >= `sm_subsonic_min` calibres (M < 0.91) or >= `sm_supersonic_min` calibres (M >= 0.91). AoA must not exceed `aoa_max` at any point. Violation terminates the sample immediately.
- **Containment** -- landing point inside the buffered danger area and peak altitude below the altitude ceiling.
- **Sea landing** -- if a coastline file is provided, landing point must be outside the coastline polygon (i.e. on land).
- **Observation coverage** (ballistic and drogue-only scenarios only) -- landing within the configured radius of at least one observation station.

A run passes if >= `compliance_threshold` percent of samples are compliant. All four scenario runs must pass.


## Optimisation

When `azimuth` and/or `inclination` are set to `"auto"`, a four-phase optimisation routine runs before the main Monte Carlo analysis:

1. **Phase 1 -- Inclination selection.** Runs deterministic 3-DOF simulations at integer inclinations from 85 to 90 degrees (no wind). Selects the steepest inclination whose ballistic landing point is both outside `min_safe_radius` from the launch site and inside the buffered danger area.

2. **Phase 2 -- Azimuth narrowing.** Analytically filters integer azimuths from -90 to +90 degrees by estimating wind drift from the mean wind profile and discarding any azimuth whose estimated premature-main landing centroid falls outside the buffered danger area.

3. **Phase 3 -- Azimuth optimisation.** Uses Bayesian optimisation (Gaussian Process with UCB acquisition) over the surviving azimuths. Each iteration runs 150 premature-main Monte Carlo simulations with wind uncertainty to estimate the containment probability. Terminates when expected improvement drops below 0.5% or after 20 iterations.

4. **Phase 4 -- Candidate validation.** Runs 500 premature-main simulations per top-3 candidate azimuth with the full uncertainty set (wind, impulse, launch angles, fin cant). Selects the azimuth with the greatest containment margin.

If only inclination is `"auto"`, only Phase 1 runs. If only azimuth is `"auto"`, the provided inclination is used and Phases 2--4 run.

The total optimisation budget is roughly 4,500 simulations, bringing the combined total (with the main 4,000-sample MC analysis) to approximately 8,500 simulations.


## Surface Wind Override

When `surface_override` is configured in `simulation.yaml`, a user-specified surface wind replaces the lower portion of each wind profile. The override is specified as a speed (m/s) and bearing (degrees clockwise from North) and blends linearly into the profile wind between ground level and `blend_height_m`. Set `blend_height_m` to `none` to disable. This is useful on launch day when you have an anemometer reading at the pad but are using forecast or balloon data for the upper atmosphere.


## Replaying Samples

Every sample is deterministic given the master seed, run index, and sample index. No full trajectory data is stored.

To replay a specific sample:

```
python -m leeds-flight-simulator replay results/<timestamp>/summary.yaml --seed 42 --run 3 --sample 117
```

To automatically replay all non-compliant samples from a completed run:

```
python -m leeds-flight-simulator replay results/<timestamp>/summary.yaml --non-compliant
```

Replay outputs a detailed time history (position, velocity, attitude, Mach, AoA, stability margin, damping coefficients, forces, moments) as a CSV.


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

At the start of execution, the simulator runs a small batch of samples at very tight integrator tolerances (1e-9), then re-runs at progressively looser settings. It automatically selects and reports the loosest tolerance that maintains acceptable output deviation. The default tolerances (1e-6) are a reasonable starting point if calibration is not needed.


## Verification

Before relying on the simulator for a safety case, verify it against an independent tool. The `verification/` directory contains scripts for this.

**RASAero comparison:** `compare_rasaero.py` runs a single deterministic trajectory and compares apogee, peak Mach, time to apogee, and stability margin against RASAero. Agreement within 5% is expected.

**Unit tests:**

```
python -m pytest verification/
```

Checks ISA against published tables, quaternion maths, launch rail exit velocity, terminal descent, aero interpolation, damping coefficients, the `.eng` parser, AoA computation, and wind `.npz` loading.


## Troubleshooting

**Simulation takes much longer than expected** -- Check that Numba is installed and working. The first run includes JIT compilation overhead (~30 s). If every run is slow, Numba may be falling back to interpreted mode; check for warnings.

**All samples are non-compliant** -- Check that your launch rail azimuth points the trajectory into the danger area. Check that the danger area GeoJSON coordinates are `[longitude, latitude]`, not reversed. Check the vehicle's static stability margin.

**Dispersion plot looks wrong or empty** -- Ensure the danger area and coastline GeoJSON files cover the area where the rocket actually lands. GeoJSON uses `[lon, lat]` coordinate order.


## Operational Workflow

A typical campaign uses the simulator at three stages:

**Safety case (months before)** -- Run with climatological wind profiles representing the full spread of weather for the planned launch month. This forms the basis for the safety case submitted to the CAA.

**Operations planning (days before)** -- Run with forecast-derived wind profiles (GFS, ECMWF, or similar). If the analysis fails, conditions may not be suitable for launch.

**Launch day go/no-go (hours before)** -- Run with radiosonde-derived wind profiles from the launch site. Enable the surface wind override with the current anemometer reading. The launch director uses this result alongside all other go/no-go criteria to make the final call. Re-run with fresh data if conditions change.


## Project Structure

```
leeds-flight-simulator/
  __main__.py              # CLI entry point (click)
  cli.py                   # Command definitions, rich output
  config.py                # YAML -> dataclasses
  atmosphere.py            # ISA (Numba)
  wind.py                  # .npz loader, surface override, interpolation
  aerodynamics.py          # Aero tables, forces, roll torques (Barrowman)
  propulsion.py            # .eng parser, mass/CG/MoI
  dynamics.py              # 6DoF + 3DoF derivatives (Numba)
  integrator.py            # Adaptive RK45 (Numba)
  recovery.py              # Descent scenarios, CdA switching
  geometry.py              # Polygons, buffer, containment
  montecarlo.py            # MC orchestration, parallelism, acceptance
  optimisation.py          # Inclination/azimuth optimisation
  outputs.py               # CSV, YAML, plot generation
  replay.py                # Single-sample replay
  verification/
    test_isa.py
    test_frames.py
    test_launch_rail.py
    test_descent.py
    test_aero_interp.py
    test_c2_damping.py
    test_eng_parser.py
    test_dynamics.py
    test_wind_loader.py
    compare_rasaero.py
  input/
    vehicle.yaml
    motor.eng
    aero_tables/
    wind_profiles.npz
    danger_area.geojson
    coastline.geojson
    simulation.yaml
```


## Contact

For questions, bug reports, or contributions:
- **Toby Thomson** -- el21tbt@leeds.ac.uk, me@tobythomson.co.uk
- **LURA Team** -- launch@leedsrocketry.co.uk


## Licence and Acknowledgements

<!-- TODO: Add licence information -->

Aerodynamic coefficient generation relies on RASAero II by Rogers Aeroscience.

The pitch/yaw damping model follows the per-component formulation described by Mandell, Caporaso, and Bengen in *Topics in Advanced Model Rocketry* (1973), based on Barrowman's aerodynamic analysis.