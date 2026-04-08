#!/usr/bin/env python
"""Generate all documentation figures from a simulation config.

Usage
-----
    python doc/generate_figures.py <simulation.yaml>

Runs the verify, run, and replay commands with ``-q`` and copies the
resulting figures into ``doc/``.  Requires a valid simulation config
with wind profiles and a verification reference trajectory.

The Monte Carlo is run with the default sample count from the config.
To produce representative figures quickly, temporarily lower
``monte_carlo.samples`` in the config (e.g. to 100).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent

# Figures produced by each command, mapped to their output locations
# relative to the results directory.
RUN_FIGURES = [
    "altitude_plot.png",
    "dispersion_plot.png",
    "damping.png",
    "damping_breakdown.png",
]

VERIFY_FIGURES = [
    "verification_plot.png",
]

REPLAY_FIGURES = [
    "replay_3d.png",
    "replay_plan_view.png",
    "replay_altitude.png",
    "replay_aoa.png",
    "replay_roll_rate.png",
]


def _run(args: list[str]) -> None:
    """Run a CLI command, forwarding output and raising on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {' '.join(args)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(args, cwd=str(DOC_DIR.parent))
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    results_dir = config_path.parent / "results"
    config_str = str(config_path)

    # 1. Verify
    _run([sys.executable, ".", "verify", config_str, "-q"])

    # 2. Run (produces altitude, dispersion, damping plots)
    _run([sys.executable, ".", "run", config_str, "-q", "-p"])

    # 3. Replay non-compliant samples (produces replay plots)
    summary_path = results_dir / "summary.yaml"
    if summary_path.exists():
        _run([sys.executable, ".", "replay", str(summary_path),
              "--non-compliant", "-q"])

    # Copy figures to doc/
    copied = []
    missing = []
    for name in RUN_FIGURES + VERIFY_FIGURES + REPLAY_FIGURES:
        src = results_dir / name
        if src.exists():
            shutil.copy2(src, DOC_DIR / name)
            copied.append(name)
        else:
            missing.append(name)

    print(f"\nCopied {len(copied)} figures to {DOC_DIR}/")
    for name in copied:
        print(f"  {name}")
    if missing:
        print(f"\n{len(missing)} figures not generated (may require "
              f"per-component aero tables or non-compliant samples):")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
