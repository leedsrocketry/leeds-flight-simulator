#!/usr/bin/env python
"""Generate all documentation figures from a simulation config.

Usage
-----
    python doc/generate_figures.py <simulation.yaml>

Runs verify, damping, and a single-sample replay (ballistic sample 0)
with ``-q`` and copies the resulting figures into ``doc/``.  Ballistic
sample 0 is used because it is always active regardless of vehicle
recovery configuration and produces all plot features without needing
a full Monte Carlo run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent

VERIFY_FIGURES = [
    "verification_plot.png",
]

DAMPING_FIGURES = [
    "damping.png",
    "damping_breakdown.png",
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

    # 1. Verify (produces verification_plot.png)
    _run([sys.executable, ".", "verify", config_str, "-q"])

    # 2. Damping (produces damping.png, damping_breakdown.png)
    _run([sys.executable, ".", "damping", config_str, "-q"])

    # 3. Run with 1 sample, ballistic only (produces altitude_plot.png,
    #    dispersion_plot.png, and provides summary.yaml for replay)
    _run([sys.executable, ".", "run", config_str,
          "-q", "-p", "-s", "ballistic"])

    # 4. Replay ballistic sample 0 (produces all replay plots)
    summary_path = results_dir / "summary.yaml"
    _run([sys.executable, ".", "replay", str(summary_path),
          "--scenario", "ballistic", "--sample", "0", "-q"])

    # Copy figures to doc/
    all_figures = VERIFY_FIGURES + DAMPING_FIGURES + REPLAY_FIGURES + [
        "altitude_plot.png",
        "dispersion_plot.png",
    ]
    copied = []
    missing = []
    for name in all_figures:
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
              f"per-component aero tables):")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
