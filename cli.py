"""Command-line interface — ``run`` and ``replay`` commands (§17).

Provides the Click CLI group invoked by ``__main__.py``.  Uses ``rich``
for progress bars, warning panels, and status output.
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

import click
import numpy as np
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
)
from rich.spinner import Spinner
from rich.text import Text

from config import load_simulation_config
from montecarlo import (
    SCENARIO_LABELS,
    build_sim_params,
    load_all_models,
    replay_compliant,
    replay_non_compliant,
    replay_sample,
    run_monte_carlo,
)
from optimisation import run_optimisation
from outputs import (
    create_results_dir,
    save_altitude_plot,
    save_dispersion_plot,
    save_replay_3d,
    save_replay_altitude,
    save_replay_plan_view,
    write_samples_csv,
    write_summary_yaml,
)
from verify import run_verification
from wind import load_wind_ensemble
from dynamics import SCENARIO_MAP, run_trajectory

console = Console()


# ---------------------------------------------------------------------------
# Custom Rich columns
# ---------------------------------------------------------------------------

class _ElapsedColumn(ProgressColumn):
    """Elapsed time shown as ``mm:ss`` in white (no hours).

    Reads ``_start`` and ``_finish`` from ``task.fields`` (set by
    ``_RunDisplay``) so each bar tracks its own wall-clock interval
    independently of when the Rich task was created.
    """

    def render(self, task: Task) -> Text:
        import time as _time
        start = task.fields.get("_start")
        finish = task.fields.get("_finish")
        if start is None:
            return Text("00:00", style="white")
        if finish is not None:
            elapsed = finish - start
        else:
            elapsed = _time.monotonic() - start
        total_secs = int(elapsed)
        mins, secs = divmod(total_secs, 60)
        return Text(f"{mins:02d}:{secs:02d}", style="white")


def _progress_columns():
    """Return the standard column tuple shared by all progress bars."""
    return (
        TextColumn("[bold]{task.description:<30}"),
        BarColumn(bar_width=40, complete_style="white", finished_style="white"),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("—"),
        _ElapsedColumn(),
    )


# ---------------------------------------------------------------------------
# Live display manager
# ---------------------------------------------------------------------------

class _RunDisplay:
    """Composite live display: spinner + warnings panel + optional progress.

    The spinner sits at the top.  Warnings accumulate as bullet points in
    a single yellow-bordered panel beneath it.  Progress bars (when active)
    appear below the warnings panel.
    """

    def __init__(self, con: Console) -> None:
        self._console = con
        self._warnings: list[str] = []
        self._spinner = Spinner("line", text="Initialising...")
        self._progress = Progress(*_progress_columns(), auto_refresh=False)
        self._live = Live(
            self._build(), console=con, refresh_per_second=12,
        )

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._live.start()

    def stop(self) -> None:
        self._live.stop()

    # -- status ----------------------------------------------------------------

    def update_status(self, text: str) -> None:
        self._spinner.update(text=text, style="default")
        self._refresh()

    # -- warnings --------------------------------------------------------------

    def add_warning(self, text: str) -> None:
        """Append a warning to the panel (non-blocking)."""
        self._warnings.append(text)
        self._refresh()

    # -- errors ----------------------------------------------------------------

    def abort(self, message: str) -> None:
        """Display a red error panel and exit."""
        self._live.stop()
        self._console.print(Panel(
            message, border_style="red", title="ERROR", title_align="left",
        ))
        sys.exit(1)

    # -- progress bars ---------------------------------------------------------

    @property
    def progress(self) -> Progress:
        """Single shared :class:`Progress` for the entire run."""
        return self._progress

    def start_task(self, task_id) -> None:
        """Record the start time for a task's elapsed column."""
        import time
        self._progress.update(task_id, _start=time.monotonic())
        self._refresh()

    def finish_task(self, task_id, **kwargs) -> None:
        """Mark a task complete, freeze its elapsed time, and refresh."""
        import time
        self._progress.update(
            task_id,
            completed=self._progress.tasks[task_id].total,
            _finish=time.monotonic(),
            **kwargs,
        )
        self._live.update(self._build())
        self._live.refresh()

    # -- internals -------------------------------------------------------------

    def _build(self) -> Group:
        parts: list = [Text(), self._spinner, Text()]
        if self._warnings:
            bullet_list = "\n".join(f"• {w}" for w in self._warnings)
            parts.append(Panel(
                bullet_list,
                border_style="yellow",
                title="WARNINGS",
                title_align="left",
            ))
        if self._progress.tasks:
            parts.append(Text())
            parts.append(self._progress)
            parts.append(Text())
        return Group(*parts)

    def _refresh(self) -> None:
        self._live.update(self._build())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_results(results_root: Path, display: _RunDisplay) -> None:
    """Remove the results directory, aborting if any file is locked."""
    if not results_root.exists():
        return

    import stat
    import time

    def _force_remove(func, path, exc_info):  # noqa: ANN001
        """Clear read-only flag and retry — standard Windows rmtree fix."""
        os.chmod(path, stat.S_IWRITE)
        func(path)

    try:
        shutil.rmtree(results_root, onerror=_force_remove)
    except OSError:
        # Stale file handles on Windows can take a moment to release
        time.sleep(0.5)
        try:
            shutil.rmtree(results_root, onerror=_force_remove)
        except OSError as exc:
            display.abort(
                f"Could not clear results directory — a file may be locked "
                f"by another program.\n\n{exc}"
            )



def _generate_altitude_curves(
    sim_cfg,
    vehicle,
    propellant,
    aero_model,
    wind_ensemble,
    azimuth_mean: float,
    inclination_mean: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run one mean-input trajectory per scenario and return altitude curves.

    Returns ``{scenario_key: (time_s, altitude_m)}`` for each active
    descent scenario, using mean stochastic inputs and the first wind
    profile (index 0).
    """
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for scenario in vehicle.recovery.active_scenarios:
        params = build_sim_params(
            sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
            wind_profile_index=0,
            azimuth_deg=azimuth_mean,
            inclination_deg=inclination_mean,
            impulse_factor=1.0,
            fin_cant_deg=0.0,
        )
        traj = run_trajectory(params, SCENARIO_MAP[scenario])

        # Ascent portion
        t_asc = traj.t_ascent
        alt_asc = -traj.state_ascent[:, 2]  # NED Down → altitude

        if traj.t_descent is not None and traj.n_descent > 0:
            t_desc = traj.t_descent[:traj.n_descent]
            alt_desc = -traj.state_descent[:traj.n_descent, 2]
            # Skip overlapping apogee point in descent
            t_full = np.concatenate([t_asc, t_desc[1:]])
            alt_full = np.concatenate([alt_asc, alt_desc[1:]])
        else:
            t_full = t_asc
            alt_full = alt_asc

        curves[scenario] = (t_full, alt_full)

    return curves


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.group()
def main():
    """Leeds Flight Simulator — 6DoF Monte Carlo sounding rocket analysis."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("-q", "--no-popup", is_flag=True, help="Do not open figures after execution.")
@click.option("-p", "--points", is_flag=True, help="Draw apogee and landing points on the dispersion plot.")
def run(config_path: Path, no_popup: bool, points: bool) -> None:
    """Run a Monte Carlo flight safety analysis."""
    import matplotlib.pyplot as plt

    config_path = Path(config_path).resolve()
    sim_cfg = load_simulation_config(config_path)
    all_warnings: list[str] = []

    display = _RunDisplay(console)

    # Route all warnings through the live display
    def _showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: object = None,
        line: str | None = None,
    ) -> None:
        text = str(message)
        all_warnings.append(text)
        display.add_warning(text)

    warnings.showwarning = _showwarning

    # --- Detect wind profile mode ---
    wind_path = sim_cfg.launch.wind_profiles
    if wind_path.is_dir():
        npz_files = sorted(wind_path.glob("*.npz"))
        if not npz_files:
            console.print(f"[red]Error:[/] No .npz files found in {wind_path}")
            sys.exit(1)
    else:
        npz_files = [wind_path]

    display.start()
    display.update_status("Loading configuration and models...")
    vehicle, propellant, aero_model, wind_ensemble = load_all_models(sim_cfg)

    # --- Clear and create results directory ---
    results_root = config_path.parent / "results"
    _clear_results(results_root, display)
    results_root.mkdir(parents=True, exist_ok=True)

    # --- Verification (runs once, before optimisation) ---
    figure_paths: list[Path] = []
    verification_figure: plt.Figure | None = None
    progress = display.progress
    if sim_cfg.verification is not None:
        display.update_status("Running trajectory verification...")
        ver_task = progress.add_task("Verification", total=1)
        display.start_task(ver_task)

        ver_result = run_verification(sim_cfg, vehicle, propellant, aero_model)
        if ver_result.figure is not None:
            if no_popup:
                ver_fig_path = results_root / "verification_plot.png"
                ver_result.figure.savefig(ver_fig_path, dpi=150, bbox_inches="tight")
                plt.close(ver_result.figure)
                figure_paths.append(ver_fig_path)
            else:
                verification_figure = ver_result.figure

        if ver_result.passed:
            verdict = "[black on green] PASS [/]"
        else:
            verdict = "[white on red] FAIL [/]"
        display.finish_task(ver_task,
                            description=f"{'Verification':<16} {verdict}")
    else:
        warnings.warn("Verification section not configured — verification skipped.")

    # --- Optimisation (runs once) ---
    opt_result = None
    rail = sim_cfg.launch.rail
    az_is_auto = rail.azimuth == "auto"
    inc_is_auto = rail.inclination == "auto"

    if az_is_auto or inc_is_auto:
        display.update_status("Running optimisation...")
        opt_task = progress.add_task("Optimisation", total=100)
        display.start_task(opt_task)

        def _opt_callback(phase_name: str, completed: int, total: int) -> None:
            progress.update(opt_task, description=f"{phase_name:<30}",
                            completed=completed, total=total)

        try:
            opt_result = run_optimisation(
                sim_cfg, vehicle, propellant, aero_model,
                wind_ensemble, _opt_callback,
            )
        except ValueError as exc:
            display.abort(f"Optimisation failed: {exc}")

        verdict = "[black on green] PASS [/]"
        display.finish_task(opt_task, description=f"{'Optimisation':<16} {verdict}")
        display.update_status(
            f"Optimisation complete — azimuth: {opt_result.selected_azimuth}°, "
            f"inclination: {opt_result.selected_inclination}°"
        )
    else:
        warnings.warn("Azimuth and inclination are fixed — optimisation skipped.")

    # Resolve final rail angles
    azimuth_mean = (
        float(opt_result.selected_azimuth) if az_is_auto
        else float(rail.azimuth)
    )
    inclination_mean = (
        float(opt_result.selected_inclination) if inc_is_auto
        else float(rail.inclination)
    )

    # --- Per wind-profile analysis loop ---
    multi_wind = len(npz_files) > 1

    for npz_path in npz_files:
        wind_suffix: str | None = None

        if multi_wind:
            wind_suffix = npz_path.stem
            display.update_status(f"Loading wind profile: {wind_suffix}...")
            wind_ensemble = load_wind_ensemble(
                npz_path,
                sim_cfg.monte_carlo.samples,
                surface_wind=sim_cfg.launch.surface_wind,
            )

        # --- Monte Carlo ---
        active_scenarios = vehicle.recovery.active_scenarios
        n_samples = sim_cfg.monte_carlo.samples

        display.update_status("Running Monte Carlo analysis...")
        tasks = {}
        for scenario in active_scenarios:
            label = SCENARIO_LABELS.get(scenario, scenario)
            tasks[scenario] = progress.add_task(label, total=n_samples)
            display.start_task(tasks[scenario])

        def _mc_callback(scenario_name: str, completed: int, total: int) -> None:
            if scenario_name in tasks:
                progress.update(tasks[scenario_name], completed=completed)

        def _scenario_done(scenario_name: str, stats) -> None:
            if scenario_name in tasks:
                label = SCENARIO_LABELS.get(scenario_name, scenario_name)
                if stats.passed:
                    verdict = "[black on green] PASS [/]"
                else:
                    verdict = "[white on red] FAIL [/]"
                display.finish_task(
                    tasks[scenario_name],
                    description=f"{label:<16} {verdict}",
                )

        mc_result = run_monte_carlo(
            sim_cfg, vehicle, propellant, aero_model,
            wind_ensemble, azimuth_mean, inclination_mean,
            progress_callback=_mc_callback,
            scenario_done_callback=_scenario_done,
        )

        # MC warnings
        for w in mc_result.warnings:
            warnings.warn(w)

        # --- Plots and outputs ---
        display.update_status("Generating plots...")
        altitude_data = _generate_altitude_curves(
            sim_cfg, vehicle, propellant, aero_model,
            wind_ensemble, azimuth_mean, inclination_mean,
        )
        burnout_time = float(propellant.times[-1])

        results_dir = create_results_dir(config_path, wind_suffix, _clear=False)

        has_coastline = sim_cfg.site.coastline is not None
        has_monitor = (
            bool(sim_cfg.site.monitor_stations)
            or sim_cfg.site.launch_monitor_radius > 0
        )

        display.update_status("Writing results...")
        write_samples_csv(mc_result.all_results, results_dir, has_coastline, has_monitor)
        write_summary_yaml(
            mc_result, sim_cfg, opt_result, results_dir,
            simulation_yaml_path=config_path,
            all_warnings=all_warnings,
        )

        plot_dir = results_dir if no_popup else None
        alt_result = save_altitude_plot(altitude_data, burnout_time, plot_dir)
        disp_result = save_dispersion_plot(
            mc_result.all_results, sim_cfg,
            sim_cfg.monte_carlo.acceptance.compliance_threshold,
            plot_dir,
            show_points=points,
        )
        if isinstance(alt_result, Path):
            figure_paths.append(alt_result)
        if isinstance(disp_result, Path):
            figure_paths.append(disp_result)

    display.stop()
    console.print(f"Results saved to: [bold]{results_dir}[/]\n")

    if no_popup and figure_paths:
        for p in figure_paths:
            console.print(f"  Saved: {p}")
    elif not no_popup:
        plt.show()


@main.command()
@click.argument("summary_path", type=click.Path(exists=True, path_type=Path))
@click.option("--seed", type=int, default=None, help="Override master seed.")
@click.option("--run", "run_index", type=int, default=None,
              help="Run (scenario) index. Active scenarios are a subset of: "
                   + ", ".join(f"{i}={name}" for i, name in enumerate(SCENARIO_LABELS))
                   + ".")
@click.option("--sample", "sample_index", type=int, default=None, help="Sample index.")
@click.option("--non-compliant", is_flag=True, help="Replay all non-compliant samples.")
@click.option("--compliant", is_flag=True, help="Replay all compliant samples.")
@click.option("-q", "--no-popup", is_flag=True, help="Save figures to disk instead of interactive display.")
def replay(
    summary_path: Path,
    seed: int | None,
    run_index: int | None,
    sample_index: int | None,
    non_compliant: bool,
    compliant: bool,
    no_popup: bool,
) -> None:
    """Replay specific samples from a completed run."""
    summary_path = Path(summary_path).resolve()
    summary_dir = summary_path.parent

    # --- Parse summary ---
    with open(summary_path, encoding="utf-8") as f:
        summary = yaml.safe_load(f)

    sim_config_path = Path(summary["metadata"]["config"]).resolve()

    # --- Validate options ---
    single_replay = run_index is not None and sample_index is not None
    if not single_replay and not non_compliant and not compliant:
        console.print(
            "[red]Error:[/] Specify either --run and --sample for a single "
            "replay, --non-compliant, or --compliant."
        )
        sys.exit(1)
    if non_compliant and compliant:
        console.print(
            "[red]Error:[/] --non-compliant and --compliant are mutually exclusive."
        )
        sys.exit(1)

    # --- Load config and models ---
    console.print("[bold]Loading configuration and models...[/]")
    sim_cfg = load_simulation_config(sim_config_path)

    # Seed from CLI override or simulation config
    master_seed = seed if seed is not None else sim_cfg.monte_carlo.seed

    # Azimuth/inclination from optimisation results if available, else config
    opt_section = summary.get("optimisation")
    if opt_section is not None:
        azimuth_mean = float(opt_section["azimuth_mean"])
        inclination_mean = float(opt_section["inclination_mean"])
    else:
        rail = sim_cfg.launch.rail
        azimuth_mean = float(rail.azimuth)
        inclination_mean = float(rail.inclination)
    vehicle, propellant, aero_model, wind_ensemble = load_all_models(sim_cfg)

    # --- Replay ---
    if single_replay:
        console.print(
            f"Replaying seed={master_seed}, run={run_index}, sample={sample_index}..."
        )
        results = [
            replay_sample(
                sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
                master_seed, run_index, sample_index,
                azimuth_mean, inclination_mean,
            )
        ]
    else:
        samples_csv = summary_dir / "samples.csv"
        if not samples_csv.exists():
            console.print(f"[red]Error:[/] samples.csv not found in {summary_dir}")
            sys.exit(1)
        if compliant:
            console.print("Replaying all compliant samples...")
            results = replay_compliant(
                sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
                master_seed, azimuth_mean, inclination_mean, samples_csv,
            )
        else:
            console.print("Replaying all non-compliant samples...")
            results = replay_non_compliant(
                sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
                master_seed, azimuth_mean, inclination_mean, samples_csv,
            )

    if not results:
        msg = "compliant" if compliant else "non-compliant"
        console.print(f"[green]No {msg} samples to replay.[/]")
        return

    console.print(f"Replayed {len(results)} sample(s).")

    # --- Print summary per replayed sample ---
    for sr in results:
        label = SCENARIO_LABELS.get(sr.scenario, sr.scenario)
        status = "[green]Compliant[/]" if sr.compliant else f"[red]Non-compliant: {sr.violation_reason}[/]"
        console.print(
            f"  Sample {sr.sample_id} ({label}): {status}"
        )

    # --- Generate replay figures ---
    import matplotlib.pyplot as plt

    out_dir = summary_dir if no_popup else None
    figure_paths: list[Path] = []
    for save_fn, name in [
        (save_replay_3d, "3D isometric"),
        (save_replay_plan_view, "plan view"),
        (save_replay_altitude, "altitude-time"),
    ]:
        try:
            result = save_fn(results, sim_cfg, output_dir=out_dir)
            if isinstance(result, Path):
                figure_paths.append(result)
        except NotImplementedError:
            console.print(f"[yellow]Warning:[/yellow] {name} replay plot not yet implemented.")

    if no_popup and figure_paths:
        for p in figure_paths:
            console.print(f"  Saved: {p}")
    elif not no_popup:
        plt.show()


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("-i", "--inclination", type=float, default=None,
              help="Launch rail inclination (degrees). Overrides config value.")
@click.option("--dump-csv", type=click.Path(path_type=Path), default=None,
              help="Write comparison data to a CSV file.")
@click.option("-q", "--no-popup", is_flag=True,
              help="Do not open figures after execution.")
def verify(config_path: Path, inclination: float | None,
           dump_csv: Path | None, no_popup: bool) -> None:
    """Compare a single trajectory against a reference CSV."""
    import matplotlib.pyplot as plt

    config_path = Path(config_path).resolve()
    sim_cfg = load_simulation_config(config_path)

    if sim_cfg.verification is None:
        console.print(Panel(
            "No verification section in config.",
            border_style="red", title="ERROR", title_align="left",
        ))
        sys.exit(1)

    display = _RunDisplay(console)

    def _showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: object = None,
        line: str | None = None,
    ) -> None:
        display.add_warning(str(message))

    warnings.showwarning = _showwarning

    display.start()

    display.update_status("Loading configuration and models...")
    vehicle, propellant, aero_model, _ = load_all_models(sim_cfg)

    display.update_status("Running 6DoF verification trajectory...")
    ver_result = run_verification(
        sim_cfg, vehicle, propellant, aero_model,
        inclination_override=inclination,
    )

    display.update_status("Done.")
    display.stop()

    # --- Dump comparison data to CSV ---
    if dump_csv is not None:
        import csv
        dump_path = Path(dump_csv).resolve()
        # Quantities may span different time ranges (e.g. CD is
        # truncated at apogee), so write each on its own timebase.
        qty_names = [q for q in ver_result.comparisons if q != "cd"]
        first = ver_result.comparisons[qty_names[0]]

        with open(dump_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["time_s"]
            for q in qty_names:
                header.extend([f"ref_{q}", f"lfs_{q}", f"err_{q}"])
            writer.writerow(header)

            for i in range(len(first.ref_time)):
                row: list[object] = [f"{first.ref_time[i]:.6f}"]
                for q in qty_names:
                    c = ver_result.comparisons[q]
                    row.append(f"{c.ref_values[i]:.6f}")
                    row.append(f"{c.sim_values[i]:.6f}")
                    row.append(f"{c.sim_values[i] - c.ref_values[i]:.6f}")
                writer.writerow(row)

        # CD comparison has a shorter timebase; append as a second file
        if "cd" in ver_result.comparisons:
            cd_cmp = ver_result.comparisons["cd"]
            cd_path = dump_path.with_stem(dump_path.stem + "_cd")
            with open(cd_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", "ref_cd", "lfs_cd", "err_cd"])
                for i in range(len(cd_cmp.ref_time)):
                    writer.writerow([
                        f"{cd_cmp.ref_time[i]:.6f}",
                        f"{cd_cmp.ref_values[i]:.6f}",
                        f"{cd_cmp.sim_values[i]:.6f}",
                        f"{cd_cmp.sim_values[i] - cd_cmp.ref_values[i]:.6f}",
                    ])

        console.print(f"\nComparison data written to: {dump_path}")

    if ver_result.figure is not None:
        if no_popup:
            fig_path = config_path.parent / "results" / "verification_plot.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            ver_result.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
            console.print(f"\nFigure saved to: {fig_path}")
        else:
            plt.show()


# ── diff ──────────────────────────────────────────────────────────────────

@main.command("diff")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.argument("cdx1_path", type=click.Path(exists=True, path_type=Path))
@click.option("-t", "--threshold", type=float, default=0.05,
              help="Acceptance threshold as a fraction (default 0.05 = 5%).")
@click.option("-m", "--motor", type=str, default=None,
              help="Motor name substring for CDX1 simulation entry.")
@click.option("-f", "--force", is_flag=True,
              help="Update YAML config files to match CDX1 values.")
def diff_cmd(config_path: Path, cdx1_path: Path,
             threshold: float, motor: str | None, force: bool) -> None:
    """Compare a RASAero CDX1 file against the LFS YAML configuration."""
    from rich.table import Table

    from cdx1 import (
        parse_cdx1, load_yaml_for_diff, build_comparison,
        sort_rows, apply_force_updates,
        _sig3,
    )
    from config import load_simulation_config

    config_path = Path(config_path).resolve()
    cdx1_path = Path(cdx1_path).resolve()

    sim_cfg = load_simulation_config(config_path)
    veh_path = sim_cfg.vehicle

    # Auto-detect motor from vehicle YAML if not specified
    if motor is None:
        motor_stem = Path(veh_path).stem  # e.g. "o3400" from "o3400.eng"
        # Read motor field from vehicle YAML directly
        with open(veh_path, encoding="utf-8") as f:
            veh_raw = yaml.safe_load(f)
        motor_stem = Path(veh_raw["motor"]).stem
        motor_hint = motor_stem
    else:
        motor_hint = motor

    old_showwarning = warnings.showwarning
    caught_warnings: list[str] = []

    def _catch(message, category, *args, **kwargs):
        caught_warnings.append(str(message))

    warnings.showwarning = _catch

    cdx1_data = parse_cdx1(cdx1_path, motor_hint=motor_hint)
    yaml_data = load_yaml_for_diff(config_path, veh_path)
    rows = build_comparison(cdx1_data, yaml_data, threshold=threshold)
    sorted_rows = sort_rows(rows)

    warnings.showwarning = old_showwarning

    # Print warnings
    for w in caught_warnings:
        console.print(Panel(w, border_style="yellow", title="WARNING",
                            title_align="left"))

    # Build Rich table
    table = Table(show_header=True, header_style="bold",
                  show_lines=False, pad_edge=False)
    table.add_column("Parameter", style="white", min_width=22)
    table.add_column("CDX1", justify="right")
    table.add_column("YAML", justify="right")
    table.add_column("Diff (%)", justify="right")
    table.add_column("", justify="center")  # PASS/FAIL

    for row in sorted_rows:
        if isinstance(row.cdx1_val, float):
            cdx1_str = _sig3(row.cdx1_val)
            yaml_str = _sig3(row.yaml_val) if isinstance(row.yaml_val, float) else str(row.yaml_val)
        else:
            cdx1_str = str(row.cdx1_val)
            yaml_str = str(row.yaml_val)

        if row.diff_pct is not None:
            diff_str = f"{row.diff_pct:.2f}"
        else:
            diff_str = "—"

        if row.passed:
            verdict = Text(" PASS ", style="black on green")
        else:
            verdict = Text(" FAIL ", style="white on red")
        table.add_row(row.label, cdx1_str, yaml_str, diff_str, verdict)

    console.print()
    console.print(f"  Threshold: {threshold * 100:.0f}%")
    console.print(table)

    # Force-update
    if force:
        n_veh, n_sim = apply_force_updates(sorted_rows, config_path, veh_path)
        total = n_veh + n_sim
        if total > 0:
            console.print(
                f"\nUpdated {total} value(s) "
                f"({n_veh} in {veh_path.name}, {n_sim} in {config_path.name})."
            )
        else:
            console.print("\nAll values within threshold — nothing to update.")
