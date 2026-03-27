# Leeds Flight Simulator

## Context
Read reference/specification.md for the full technical specification.
Read reference/user-manual.md for user-facing requirements.
Reference plotting scripts are in reference/ — match their visual style.

## Rules
- Python 3.10+. Use type hints. Use enviroment.
- Hot-loop physics code must be Numba @njit compatible (no Python objects, no scipy in the inner loop).
- British English throughout.
- Run `python -m pytest` after implementing each module to check nothing is broken.
- Commit after each working module.

## Architecture
See §19 of reference/specification.md for the module layout.