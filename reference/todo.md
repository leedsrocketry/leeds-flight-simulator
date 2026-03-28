# TODO

## Design Changes

- [ ] **Multi-day wind**: Accept a single `.npz` file or a directory of `.npz` files under `wind_profiles/`. Run independently for each file; suffix the output folder name with the wind profile filename when more than one is provided. *(Update user manual.)*

- [ ] **Verification tool**: Accept an optional flight simulator output CSV as an input file. Before the main MC run, compare altitude, stability margin, Mach, mass, and inertia vs time — reference data in grey with a configurable tolerance band, our output overlaid in green (pass) or red (fail). Print pass/fail to console. If there are failures, open the figure and ask the user whether to proceed; otherwise continue automatically. Tolerance parameters go in `simulation.yaml`. Must work with any flight simulator, not just RASAero. *(Document in user manual.)*

- [ ] **Warnings**: Make warnings blocking — prompt the user to acknowledge each one before continuing. `--no-warn` suppresses the prompt; warnings still appear in the log and the results summary. *(Document in user manual.)*

- [ ] **Replay plots**: Three figures that open automatically — (1) 3D isometric with the map on the ground plane, trajectory coloured by descent scenario (black if terminated early due to instability); (2) plan view matching `dispersion_plot.png` style with trajectory trace; (3) altitude vs time.

- [x] **Optimisation search ranges**: Replace hardcoded inclination (85–90°) and azimuth (−90° to +90°) ranges with configurable `simulation.yaml` parameters, required when the corresponding value is `"auto"`. *(Update spec and `simulation.yaml`.)*

- [x] **Vehicle file paths**: Move `motor` and `aero_tables` paths from `simulation.yaml` into `vehicle.yaml`, so all vehicle-related inputs are in one place.

- [x] **Fins aero table**: Specify which aero CSV corresponds to the fins in `vehicle.yaml`, rather than relying on the filename containing `"fin"`.

- [x] **Rename `input/` → `example/`**: The example inputs directory should be called `example/`, not `input/`.

- [x] **Launch site observation station**: Automatically add the launch site as an observation station. Add a `launch_observation_radius` field to `site` in `simulation.yaml` for its radius. At the same time, rename `min_safe_radius` to `ballistic_exclusion_radius` to make clear that these are two distinct concepts: one is for LOS coverage, the other is the minimum acceptable ballistic landing distance used during inclination optimisation. *(Config fields and example YAMLs updated; runtime logic in `montecarlo.py`.)*

- [ ] **Sea/land criterion**: Consider making the sea-landing check direction configurable (land-only vs sea-only).


## Specification

- [ ] Remove explicit SM transition Mach threshold from the text; say it is configurable in `simulation.yaml` because the definition varies between tools. Add a comment to `simulation.yaml` noting the default follows RASAero's definition.

- [ ] Update `ballistic_exclusion_radius` name and `launch_observation_radius` field throughout.


## User Manual

**Remove:**
- [ ] Simulation and iteration counts
- [ ] Details of which uncertainties are reintroduced in the optimisation routine
- [ ] Mentions of RASAero (except the verification tool section)
- [ ] Hardcoded default parameter values
- [ ] Colour specifications
- [ ] Inline copies of `simulation.yaml` and `vehicle.yaml` — direct users to `example/` instead
- [ ] RASAero acknowledgement

**Add / update:**
- [ ] References section
- [ ] Note that the simulator runs fully offline, except for fetching base map tiles for plotting (which are cached locally)
- [ ] Active descent scenario table by recovery configuration (copy from spec §9)
- [ ] Point to `example/` for example inputs; note that justification for those values is in the safety case
- [ ] Update input file count and directory name to reflect `example/` rename
- [ ] Note that the reference area is defined as π·d²/4, consistent with RASAero
- [ ] Note that the simulator falls back to 3DoF when only one aero table is provided
- [ ] Links to a NATS airspace chart viewer and an online GeoJSON editor
- [ ] Clarify that the coastline GeoJSON should be the on-land polygon (the area the rocket must not land in)
- [ ] Verification tool: how to use it and how to interpret the output *(once implemented)*
- [ ] Warnings: how they work and how to dismiss them *(once implemented)*
- [ ] Multi-day wind: how to provide multiple wind profiles *(once implemented)*


## Input Files

**`simulation.yaml`:**
- [ ] Add a comment for every parameter: what it does, its unit, and its allowable range
- [ ] Show optional parameters as commented-out entries with brief explanations
- [x] Rename `min_safe_radius` → `ballistic_exclusion_radius`; add `launch_observation_radius`
- [x] Add optimisation search range fields *(see Design Changes)*
- [ ] Add a comment for every parameter: what it does, its unit, and its allowable range
- [ ] Show optional parameters as commented-out entries with brief explanations
- [ ] Add verification tolerance parameters *(see Design Changes)*

**`vehicle.yaml`:**
- [x] Add `motor` and `aero_tables` path fields *(see Design Changes)*
- [x] Add fins aero table field *(see Design Changes)*
