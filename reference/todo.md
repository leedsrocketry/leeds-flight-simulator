# TODO:

## Simulator Debugging
[ ] Thrust correct? <-- about to check...
[ ] Inputs the same for RASAero and LFS? <-- will write a helper script for this
[ ] Using power on and power off CA!?
[ ] Inertia correct? <-- I don't think so, but shouldn't affect 2 dof
[ ] Motor CG correct? <--- this is probably where the stability error comes from
[ ] Stability after apogee should be zero?

## Feature Add
[x] update user manual to match code base
[x] implement main and cli

[x] fix 3 dof ascent for optimisation and verification routines (I think it should maybe be 4 and is not implemented properly)

[ ] add a field to verification section in simulation.yaml for the inclination?

[ ] deal with trajectory result history issue blocking the implementation of replay plots generation in outputs.py
[ ] implement the replay plots methods

[ ] Move the vehicle cg and moi calculation methods from motor.py to dynamics.py (if even there in the first place?) Leave the methods for calculating the evolving motor properties there.

[ ] Fix the failing tests
[ ] Check the tests give good coverage and haven't been made to always pass
[ ] Double-check simulation inputs
[ ] Test multiple aero tables functionality
[ ] Check that the 3 dof descent portion of the flight is treating the vehicle as a point mass, always moving in the direction and at the speed of the local wind vector.

## Organisation
[ ] Remove reference/ folder when done with it

## Documentation

## UI

## Output

## UX
