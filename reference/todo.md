# TODO:

## Backend
[x] update user manual to match code base
[x] implement main and cli
[ ] fix 3 dof ascent for optimisation and verification routines (I think it should maybe be 4 and is not implemented properly)
[ ] deal with trajectory result history issue blocking the implementation of replay plots generation in outputs.py
[ ] implement the replay plots methods
[ ] add a field to verification section in simulation.yaml for the inclination?
[ ] Move the vehicle cg and moi calculation methods from motor.py to dynamics.py (if even there in the first place?) Leave the methods for calculating the evolving motor properties there.
[ ] Fix the failing tests
[ ] Check the tests give good coverage and haven't been made to always pass
[ ] Double-check simulation inputs
[ ] Test multiple aero tables functionality
[ ] Check that the 3 dof descent portion of the flight is treating the vehicle as a point mass, always moving in the direction and at the speed of the local wind vector.

## Organisation
[ ] Rename verification/ to test/
[ ] Rename example/ to simulations/g2b2-safety-case/
[ ] Rename simulation.yaml to cape-wrath.yaml
[ ] Remove reference/ folder when done with it

## Documentation
[ ] Change README to use outputs from example/ as the figures (+ some additional screenshots I'll take later)
[ ] Update docs to point to felect changes in structure

## UI
[ ] Remove the red error message when verification dosen't pass. Just throw a warning.
[ ] Place PASS/FAIL on the right hand side of each scenario progress bar in "reverse" style
[ ] Make the progres bar stuff monotone (changing colours when 100% progress makes it seem like all the results passed as opposed to the simulation is just complete)
[ ] Make the elipses "..." the same colour as the text in the cli status messages
[ ] Add a CLI status message (same format as the others) before running the simulations
[ ] Remove the PASS/FAIL table, warnings summary and acceptance summary text from the CLI.
[ ] Warnings in yellow bordered boxes with some vertical spacing between them and whatever comes next
[ ] Throw warnings when optimisation and verification are skipped over
[ ] Throw a warning when some of the scenarios mentioned in simulation.yaml acceptance criteria are not active.

## Output
[ ] Remove fields from summary.yaml which are covered in the simulation or vehicle setup yamls
[ ] Rename "azimuth_inclination" section in summary.yaml to "optimisation", keeping just azimuth_mean and inclination_mean
[ ] Remove "samples" fields from summary.yaml
[ ] Rename "run_details" to "details" in summary.yaml
[ ] Remove the "overall" section from summary.yaml
[ ] Rename "scenario_results" to "scenarios" in summary.yaml
[ ] Move the "warnings" section into the newly-renamed "details" section
[ ] Remove "wind_profile_index" and "peak_altitude_ft" columns from samples.csv
[ ] Rename the "landing_at_sea", "in_buffer", "below_ceiling" and "in_coverage" samples.csv to names such as "stability_compliant" i.e. all the use has to do is look for the column(s) containing "FALSE" to known where the complicance check fell down (currently they have to work out if the result is supposed to be TRUE or FALSE for each line based on the scenario)
[ ] Reorder the columns in samples.csv to a more sensible order that flows left to right from inputs to outputs. Should be inputs, flight time, compliance results, calculated compliance values and then locations
[ ] Remove tile from the verifiy.py plot
[ ] Remove the "std" fields from summary.yaml (not nessecerily normal and we don't want to lull the user into thinking it always is)
[ ] Buffer zone not visible on dispersion plot?
[ ] Add a unique marker for every observation station and map location (missing one for MOD Range Control atm)
[ ] Change MOD Range Control observation circle to black-transparent, from purple. Colour all observation circles the same and include a single legend item for "Observation Coverage" this will explain what all the observation circles are doing.
[ ] Observation circle missing at launch site? Launch site is an observation station, however recieves only one marker and always the "X" as is currently the case.
[ ] Buffer Zone legend distance should be driven by simulation config yaml, if not already. Ensure the full danger area is in view on the dispersion plot.

## UX
[ ] Don't create a new simulation output folder with the timestamp every time the simulation is run, overwrite the results that are already there, call the folder "results/". Create a sub folder for each day simulated with the name of that day's ind profile. verification plot is in the results/ folder, while everything else is in that day's sub folder. If only one day then don't bother with the sub folder. Each time the simulation is rerun, clear the contents of the results/ folder. Create the results/ folder in the same place as currently.
[ ] Add the results/ directory to .gitignore
[ ] Better name for --no-warn. The warnings still generated, it's just non-blocking.
[ ] Implement abbreviated flags that can also be used for the non-blocking warnings and pop-up arguments