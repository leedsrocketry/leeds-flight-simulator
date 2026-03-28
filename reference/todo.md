# TODO

## Design Changes

- [ ] **Multi-day wind**: Accept a single `.npz` file or a directory of `.npz` files under `wind_profiles/`. Run independently for each file; suffix the output folder name with the wind profile filename when more than one is provided. *(User manual updated with planned interface.)*

- [ ] **Verification tool**: Accept an optional flight simulator output CSV as an input file. Before the main MC run, compare altitude, stability margin, Mach, mass, and inertia vs time — reference data in grey with a configurable tolerance band, our output overlaid in green (pass) or red (fail). Print pass/fail to console. If there are failures, open the figure and ask the user whether to proceed; otherwise continue automatically. Tolerance parameters go in `simulation.yaml`. Must work with any flight simulator, not just RASAero. *(User manual updated with planned interface; add tolerance parameters to `simulation.yaml` once implemented.)*

- [ ] **Warnings**: Make warnings blocking — prompt the user to acknowledge each one before continuing. `--no-warn` suppresses the prompt; warnings still appear in the log and the results summary. *(User manual updated with planned interface.)*

- [ ] **Replay plots**: Two figures that open automatically (shared if multiple samples requested) — (1) 3D isometric with the map on the ground plane, trajectory coloured by descent scenario (black if terminated early due to instability), matching `dispersion_plot.png`; (2) altitude vs time. *(Documented in user manual; runtime implementation pending.)*

- [ ] **Sea/land criterion**: Make the coastline check direction configurable via `coastline_mode: "sea" | "land"` in `simulation.yaml`. *(Documented in user manual, spec, and example YAML as planned interface.)*
