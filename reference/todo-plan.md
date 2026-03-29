# Implementation Plan

Ordered to settle existing features first: Organisation > UX > UI > Output > Documentation > Backend.

---

## Phase 1 — Organisation

Renames and restructure. No logic changes.

### 1.1 Rename `verification/` to `test/`
- Rename the directory.
- Update `pyproject.toml` test paths (if any).
- Update any imports or path references in code/README.
- Run `python -m pytest test/` to confirm tests still pass.

### 1.2 Rename `example/` to `simulations/g2b2-safety-case/`
- Create `simulations/` parent directory and move contents.
- Update all relative-path references: README, any hardcoded example paths.
- Update `.gitignore` if it references `example/`.

### 1.3 Rename `simulation.yaml` to `cape-wrath.yaml`
- Rename the file inside the newly-moved `simulations/g2b2-safety-case/`.
- Update README quick-start command and any other docs that reference `simulation.yaml`.

### 1.4 Commit checkpoint
- Run full test suite, then commit: "Reorganise directory structure".

### 1.5 Add `test/conftest.py` and remove `pyproject.toml`
- Create `test/conftest.py` with:
  - `sys.path` setup for project root (replaces `pyproject.toml`'s `pythonpath` setting).
  - `EXAMPLE_SIM_DIR` constant pointing to `simulations/g2b2-safety-case/`.
  - Session-scoped fixtures for common example data paths (`sim_yaml_path`, `vehicle_yaml_path`, `motor_path`, `aero_dir`, `d802_path`, `coastline_path`), each skipping if absent.
- Update test files to use conftest fixtures/constants instead of hardcoded paths.
- Delete `pyproject.toml` (its only purpose was `pythonpath`).
- Run full test suite, then commit.

---

## Phase 2 — UX

Changes to how the tool behaves (file I/O, CLI arguments), not how it looks.

### 2.1 Overwrite results in a fixed `results/` folder
Currently `create_results_dir()` in `outputs.py` creates a timestamped folder each run.

- Change `create_results_dir()` to:
  1. Place a `results/` folder alongside the simulation config YAML (i.e. `simulations/g2b2-safety-case/results/`).
  2. On each run, **clear** all contents of `results/` before writing.
  3. When multiple wind profiles (multi-day): create a subfolder per day named after the `.npz` stem (e.g. `results/monday/`). Verification plot stays in `results/`.
  4. When only one wind profile: write everything directly into `results/` (no subfolder).
- Update `cli.py` to pass through the new logic and remove timestamp generation.
- Update `verify.py` / `cli.py` so the verification plot is saved into `results/`.

### 2.2 Add `results/` to `.gitignore`
- Append `results/` to the project `.gitignore`.

### 2.3 Rename `--no-warn` to `--non-blocking`
- The flag suppresses interactive prompts, not the warnings themselves. `--non-blocking` is more accurate.
- Update `cli.py` Click option name, all references in code, and README.

### 2.4 Implement abbreviated CLI flags
- Add short aliases for existing flags: `-n` for `--non-blocking`, `-q` for `--no-popup`.
- These are just extra Click option names — minimal code change.

### 2.5 Commit checkpoint
- Run full test suite, then commit: "Rework results output and CLI flags".

---

## Phase 3 — UI

Changes to what the user sees in the terminal. All in `cli.py`.

### 3.1 Verification result: warning instead of red error
- In `cli.py` `run()`, when verification fails, replace `console.print("[red]Verification FAILED[/]")` with a `warnings.warn(...)` call so it flows through the existing warning hook (yellow, blocking/non-blocking).

### 3.2 PASS/FAIL on right-hand side of progress bars in reverse style
- After each scenario's progress bar completes, append the scenario's pass/fail result to the progress bar description using Rich's `[reverse]` markup (e.g. `[black on green] PASS [/]` / `[bold red] FAIL [/]`).
- This means the result is shown inline on the progress bar row, right-aligned.

### 3.3 Monotone progress bars
- Remove any colour-change logic when a bar hits 100%. Use a single neutral colour (e.g. white/default) for the bar fill throughout, so completion doesn't look like a pass.

### 3.4 Ellipses colour matching
- Ensure any "..." in status messages inherits the same style as surrounding text (no separate colour).

### 3.5 Pre-simulation status message
- Before entering the Monte Carlo progress section, print a status line: `"Running Monte Carlo analysis..."` (same `[bold]` style as the other status messages like "Loading configuration and models...").

### 3.6 Remove results table, warnings summary, and acceptance summary
- Delete `_print_results_table()`, `_print_verdict()`, and `_print_warnings()` calls from the `run()` command.
- The per-scenario PASS/FAIL is now shown on the progress bars (step 3.2), so the table is redundant.
- Warnings were already shown inline when they occurred.

### 3.7 Warnings in yellow bordered boxes with vertical spacing
- Replace inline yellow text warnings with Rich `Panel` objects: yellow border, warning text inside.
- Add a blank line after each warning panel for spacing.

### 3.8 Warnings when optimisation/verification are skipped
- If `sim_cfg.verification is None`, emit `warnings.warn("Verification section not configured — verification skipped.")`.
- If azimuth and inclination are both fixed (not "auto"), emit `warnings.warn("Azimuth and inclination are fixed - optimisation skipped.")`.

### 3.9 Warning when acceptance scenarios are inactive
- Compare the scenarios listed in `simulation.yaml` acceptance criteria against `vehicle_cfg.recovery.active_scenarios`.
- If any configured acceptance scenario is not active, emit a warning listing which ones are missing.

### 3.10 Commit checkpoint
- Run full test suite, then commit: "Overhaul CLI UI: progress bars, warnings, status messages".

---

## Phase 4 — Output

Changes to `summary.yaml`, `samples.csv`, and plots in `outputs.py`.

### 4.1 `summary.yaml` cleanup
All changes in `write_summary_yaml()`:
1. Rename `run_details` key to `details`.
2. Remove fields already present in the simulation/vehicle config YAMLs (identify which by diffing summary output against config inputs).
3. Rename `azimuth_inclination` section to `optimisation`, keeping only `azimuth_mean` and `inclination_mean`.
4. Remove all `samples` count fields (redundant with samples.csv).
5. Remove the `overall` section.
6. Rename `scenario_results` to `scenarios`.
7. Move the `warnings` list into the `details` section.
8. Remove all `std` fields from scenario statistics.

### 4.2 `samples.csv` cleanup
All changes in `write_samples_csv()`:
1. Remove `wind_profile_index` and `peak_altitude_ft` columns.
2. Rename boolean compliance columns to self-describing names where `FALSE` = non-compliant. Be careful to respect the changing compliance requirements per scenario. This is best handled with rohobust logic in one instance that the rest of the program relies on, rather than reimplementing the logic in several places:
   - `landing_at_sea` -> `sea_landing_compliant` (invert if needed so FALSE = bad)
   - `in_buffer` -> `buffer_compliant` (invert: currently TRUE means inside buffer = bad)
   - `below_ceiling` -> `ceiling_compliant`
   - `in_coverage` -> `coverage_compliant`
3. Reorder columns left-to-right: **inputs** (sample_id, scenario, seed info, azimuth, inclination, fin_cant, impulse_factor) -> **flight time** -> **compliance results** (all `_compliant` columns) -> **calculated compliance values** (SM, AoA, mach, apogee) -> **locations** (lat/lon).

### 4.3 Verification plot: remove title
- In `verify.py`, find the `set_title()` or `suptitle()` call on the verification figure and remove it.

### 4.4 Dispersion plot: buffer zone visibility
- Investigate whether the buffer zone polygon is being drawn correctly in `save_dispersion_plot()`. Likely a z-order or alpha issue — ensure it renders visibly above the basemap.

### 4.5 Dispersion plot: unique markers for every location
- Ensure each observation station and map location gets a distinct marker symbol/colour.
- Add a marker for MOD Range Control (currently missing).

### 4.6 Dispersion plot: observation circle styling
- Change all observation circles to the same colour: black with transparency.
- Remove per-station circle colours (e.g. purple for MOD Range Control).
- Consolidate legend to a single "Observation Coverage" entry.

### 4.7 Dispersion plot: launch site observation circle
- The launch site is an observation station. Ensure it gets an observation coverage circle in addition to its "X" marker.

### 4.8 Dispersion plot: buffer distance from config + ensure full danger area in view
- Confirm the buffer zone legend label pulls its distance from `sim_cfg.monte_carlo.acceptance.buffer_m` rather than being hardcoded.
- Adjust the plot extent calculation to ensure the full danger area boundary (not just landing points) is always visible.

### 4.9 Commit checkpoint
- Run full test suite, then commit: "Clean up output formats and dispersion plot".

---

## Phase 5 — Documentation

### 5.1 Update README figures
- Change the README to use output figures from `simulations/g2b2-safety-case/results/` as the embedded images.
- Add placeholders for additional screenshots (to be taken later).

### 5.2 Update README to reflect structural changes
- Update all paths: `example/` -> `simulations/g2b2-safety-case/`, `simulation.yaml` -> `cape-wrath.yaml`, `verification/` -> `test/`.
- Update CLI flag names: `--no-warn` -> `--non-blocking`, add `-n`/`-q` abbreviations.
- Update output directory description (no more timestamped folders).
- Remove references to the results table / acceptance summary (now removed from CLI).

### 5.3 Commit checkpoint
- Run full test suite, then commit: "Update documentation for new structure and UI".

---

## Phase 6 — Backend

New features and physics fixes. Each is independent and can be committed separately.

### 6.1 Fix 3DoF ascent for optimisation/verification
- Review `simulate_ascent_3dof()` in `dynamics.py`. The todo suggests it may need to be 4DoF (3 translational + 1 rotational?) and may not be implemented correctly.
- Compare against specification section 8.5.
- Fix or upgrade as needed; re-run optimisation to verify behaviour is sensible.

### 6.2 Implement replay plots
- The trajectory result history issue is blocking this. First, resolve whatever prevents `TrajectoryResult` from storing full time-history data for replay.
- Then implement the three stubs in `outputs.py`: `save_replay_3d()`, `save_replay_plan_view()`, `save_replay_altitude()`.
- Match the visual style of the reference plotting scripts.

### 6.3 Add inclination field to verification config
- Add an optional `inclination` field to the `verification` section of `simulation.yaml` schema in `config.py`.
- Use it in `verify.py` when running the reference trajectory comparison.

### 6.4 Move vehicle CG/MOI calculations
- If `motor.py` currently contains vehicle-level CG and MOI methods, move them to `dynamics.py`.
- Keep motor-specific evolving properties (motor CG, motor inertia vs. time) in `motor.py`.

### 6.5 Fix failing tests
- Run `python -m pytest test/` and fix whatever is broken.
- Audit test quality: ensure tests aren't trivially always-passing (e.g. `assert True`).

### 6.6 Double-check simulation inputs
- Review `config.py` parsing and `g2b2.yaml` values against real vehicle data.

### 6.7 Test multiple aero tables
- Manually test with per-component aero CSVs (not just whole-vehicle).
- Fix any issues found.

### 6.8 Verify 3DoF descent treats vehicle as point mass
- In `integrate_descent()` in `dynamics.py`, confirm the vehicle moves purely with the local wind vector (no residual body-frame aerodynamics).
- Fix if not.

### 6.9 Commit after each fix
- Each backend item is independent — commit separately after tests pass.

---

## Execution notes

- Run `python -m pytest` after every step, as per CLAUDE.md.
- Commit after each phase (or sub-phase where noted).
- Phases 1-5 should not break any physics — they are restructuring, UI, and output formatting only.
- Phase 6 items touch physics and should each be tested carefully.