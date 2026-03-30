"""Command-line interface — ``run`` and ``replay`` commands (§17).

Provides the Click CLI group invoked by ``__main__.py``.  Uses ``rich``
for progress bars, warning panels, and status output.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
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


def _open_figures(paths: list[Path]) -> None:
    """Open figure files with the system's default image viewer."""
    system = platform.system()
    for p in paths:
        if not p.exists():
            continue
        if system == "Windows":
            os.startfile(str(p))  # noqa: S606
        elif system == "Darwin":
            subprocess.Popen(["open", str(p)])  # noqa: S603
        else:
            subprocess.Popen(["xdg-open", str(p)])  # noqa: S603


def _generate_altitude_curves(
    sim_cfg,
    vehicle_cfg,
    motor_model,
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

    for scenario in vehicle_cfg.recovery.active_scenarios:
        params = build_sim_params(
            sim_cfg, vehicle_cfg, motor_model, aero_model, wind_ensemble,
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
    vehicle_cfg, motor_model, aero_model, wind_ensemble = load_all_models(sim_cfg)

    # --- Clear and create results directory ---
    results_root = config_path.parent / "results"
    _clear_results(results_root, display)
    results_root.mkdir(parents=True, exist_ok=True)

    # --- Verification (runs once, before optimisation) ---
    verification_figure_path: Path | None = None
    progress = display.progress
    if sim_cfg.verification is not None:
        display.update_status("Running trajectory verification...")
        ver_task = progress.add_task("Verification", total=1)
        display.start_task(ver_task)

        ver_result = run_verification(sim_cfg, vehicle_cfg, motor_model, aero_model)
        if ver_result.figure is not None:
            ver_fig_path = results_root / "verification_plot.png"
            ver_result.figure.savefig(ver_fig_path, dpi=150, bbox_inches="tight")
            verification_figure_path = ver_fig_path

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
                sim_cfg, vehicle_cfg, motor_model, aero_model,
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
    figure_paths: list[Path] = []
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
        active_scenarios = vehicle_cfg.recovery.active_scenarios
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
            sim_cfg, vehicle_cfg, motor_model, aero_model,
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
            sim_cfg, vehicle_cfg, motor_model, aero_model,
            wind_ensemble, azimuth_mean, inclination_mean,
        )
        burnout_time = float(motor_model.times[-1])

        results_dir = create_results_dir(config_path, wind_suffix, _clear=False)

        has_coastline = sim_cfg.site.coastline is not None
        has_monitour = (
            bool(sim_cfg.site.monitour_stations)
            or sim_cfg.site.launch_monitour_radius > 0
        )

        display.update_status("Writing results...")
        write_samples_csv(mc_result.all_results, results_dir, has_coastline, has_monitour)
        write_summary_yaml(
            mc_result, sim_cfg, opt_result, results_dir,
            simulation_yaml_path=config_path,
            all_warnings=all_warnings,
        )
        alt_path = save_altitude_plot(altitude_data, burnout_time, results_dir)
        disp_path = save_dispersion_plot(
            mc_result.all_results, sim_cfg,
            sim_cfg.monte_carlo.acceptance.compliance_threshold,
            results_dir,
            show_points=points,
        )
        figure_paths.extend([alt_path, disp_path])

    display.stop()
    console.print(f"Results saved to: [bold]{results_dir}[/]\n")

    # --- Open figures ---
    if not no_popup:
        if verification_figure_path is not None:
            figure_paths.insert(0, verification_figure_path)
        _open_figures(figure_paths)


@main.command()
@click.argument("summary_path", type=click.Path(exists=True, path_type=Path))
@click.option("--seed", type=int, default=None, help="Override master seed.")
@click.option("--run", "run_index", type=int, default=None, help="Run (scenario) index.")
@click.option("--sample", "sample_index", type=int, default=None, help="Sample index.")
@click.option("--non-compliant", is_flag=True, help="Replay all non-compliant samples.")
def replay(
    summary_path: Path,
    seed: int | None,
    run_index: int | None,
    sample_index: int | None,
    non_compliant: bool,
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
    if not single_replay and not non_compliant:
        console.print(
            "[red]Error:[/] Specify either --run and --sample for a single "
            "replay, or --non-compliant to replay all non-compliant samples."
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
    vehicle_cfg, motor_model, aero_model, wind_ensemble = load_all_models(sim_cfg)

    # --- Replay ---
    if single_replay:
        console.print(
            f"Replaying seed={master_seed}, run={run_index}, sample={sample_index}..."
        )
        results = [
            replay_sample(
                sim_cfg, vehicle_cfg, motor_model, aero_model, wind_ensemble,
                master_seed, run_index, sample_index,
                azimuth_mean, inclination_mean,
            )
        ]
    else:
        samples_csv = summary_dir / "samples.csv"
        if not samples_csv.exists():
            console.print(f"[red]Error:[/] samples.csv not found in {summary_dir}")
            sys.exit(1)
        console.print("Replaying all non-compliant samples...")
        results = replay_non_compliant(
            sim_cfg, vehicle_cfg, motor_model, aero_model, wind_ensemble,
            master_seed, azimuth_mean, inclination_mean, samples_csv,
        )

    if not results:
        console.print("[green]No samples to replay (all compliant).[/]")
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
    figure_paths: list[Path] = []
    for save_fn, name in [
        (save_replay_3d, "3D isometric"),
        (save_replay_plan_view, "plan view"),
        (save_replay_altitude, "altitude-time"),
    ]:
        try:
            fig_path = save_fn(results, summary_dir)
            figure_paths.append(fig_path)
        except NotImplementedError:
            console.print(f"[yellow]Warning:[/yellow] {name} replay plot not yet implemented.")

    if figure_paths:
        _open_figures(figure_paths)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("-d", "--dof", type=click.Choice(["2", "6"]), default="6",
              help="Ascent model: 6 (full 6DoF) or 2 (point-mass).")
@click.option("-a", "--azimuth", type=float, default=None,
              help="Launch rail azimuth (degrees). Overrides config value.")
@click.option("-i", "--inclination", type=float, default=None,
              help="Launch rail inclination (degrees). Overrides config value.")
@click.option("--dump-csv", type=click.Path(path_type=Path), default=None,
              help="Write comparison data to a CSV file.")
@click.option("-q", "--no-popup", is_flag=True,
              help="Do not open figures after execution.")
def verify(config_path: Path, dof: str, azimuth: float | None,
           inclination: float | None, dump_csv: Path | None,
           no_popup: bool) -> None:
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

    if dof == "2":
        display.add_warning(
            "2DoF verification uses a simplified point-mass model, not the "
            "6DoF code path used by Monte Carlo. Use --dof 6 to verify the "
            "production simulation."
        )

    display.update_status("Loading configuration and models...")
    vehicle_cfg, motor_model, aero_model, _ = load_all_models(sim_cfg)

    display.update_status(f"Running {dof}DoF verification trajectory...")
    ver_result = run_verification(
        sim_cfg, vehicle_cfg, motor_model, aero_model, dof=int(dof),
        azimuth_override=azimuth, inclination_override=inclination,
    )

    display.update_status("Done.")
    display.stop()

    # --- Print per-quantity summary table ---
    from rich.table import Table
    table = Table(title="Verification Summary", show_lines=False)
    table.add_column("Quantity", style="bold")
    table.add_column("Result", justify="centre")
    table.add_column("Max |err|")
    table.add_column("RMS err")
    table.add_column("Mean bias")

    import numpy as np
    for qty_name, cmp in ver_result.comparisons.items():
        err = cmp.sim_values - cmp.ref_values
        max_abs = float(np.max(np.abs(err)))
        rms = float(np.sqrt(np.mean(err ** 2)))
        mean_bias = float(np.mean(err))

        n_fail = int(np.sum(~cmp.within_tolerance))
        n_total = len(cmp.within_tolerance)
        if cmp.passed:
            if n_fail == 0:
                result_str = "[bold green]PASS[/bold green]"
            else:
                pct = 100.0 * n_fail / n_total
                result_str = f"[bold green]PASS[/bold green] ({pct:.2f}%)"
        else:
            result_str = f"[bold red]FAIL[/bold red] ({n_fail} pts)"

        table.add_row(
            cmp.name.title(),
            result_str,
            f"{max_abs:.4g}",
            f"{rms:.4g}",
            f"{mean_bias:+.4g}",
        )

    console.print()
    console.print(table)

    overall = "[bold green]PASS[/bold green]" if ver_result.passed else "[bold red]FAIL[/bold red]"
    console.print(f"\nOverall: {overall}")

    # --- Dump comparison data to CSV ---
    if dump_csv is not None:
        import csv
        dump_path = Path(dump_csv).resolve()
        # All comparisons share the same ref_time (interpolated to reference timebase)
        qty_names = list(ver_result.comparisons.keys())
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

        console.print(f"\nComparison data written to: {dump_path}")

    if ver_result.figure is not None:
        fig_path = config_path.parent / "results" / "verification_plot.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        ver_result.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
        console.print(f"\nFigure saved to: {fig_path}")
        if not no_popup:
            plt.show()
