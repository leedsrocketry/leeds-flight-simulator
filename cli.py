"""Command-line interface — ``run`` and ``replay`` commands (§17).

Provides the Click CLI group invoked by ``__main__.py``.  Uses ``rich``
for progress bars, results tables, and coloured verdict output.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import click
import numpy as np
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from config import load_simulation_config
from montecarlo import (
    SCENARIO_LABELS,
    MonteCarloResult,
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
# Helpers
# ---------------------------------------------------------------------------

def _warn_blocking(message: str, no_warn: bool) -> None:
    """Print a warning; block for acknowledgement unless *no_warn*."""
    console.print(f"[yellow]WARNING:[/yellow] {message}")
    if not no_warn:
        try:
            input("Press Enter to continue, or Ctrl-C to abort. ")
        except KeyboardInterrupt:
            console.print("\nAborted.")
            sys.exit(1)


def _install_warning_hook(
    all_warnings: list[str], no_warn: bool,
) -> None:
    """Override :func:`warnings.showwarning` so that *UserWarning*s are
    displayed via :func:`_warn_blocking` and collected into *all_warnings*.
    """
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
        _warn_blocking(text, no_warn)

    warnings.showwarning = _showwarning


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


def _print_results_table(mc_result: MonteCarloResult) -> None:
    """Print the per-scenario results table (§17.3)."""
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Scenario", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("Compliant", justify="right")
    table.add_column("Non-Compliant", justify="right")
    table.add_column("Accepted?", justify="center")

    total_samples = 0
    total_compliant = 0
    total_non_compliant = 0

    for name, stats in mc_result.scenario_stats.items():
        label = SCENARIO_LABELS.get(name, name)
        accepted = "[black on green] PASS [/]" if stats.passed else "[bold red]FAIL[/]"
        table.add_row(
            label,
            str(stats.n_samples),
            str(stats.n_compliant),
            str(stats.n_non_compliant),
            accepted,
        )
        total_samples += stats.n_samples
        total_compliant += stats.n_compliant
        total_non_compliant += stats.n_non_compliant

    table.add_section()
    table.add_row(
        "Total",
        str(total_samples),
        str(total_compliant),
        str(total_non_compliant),
        "",
    )

    console.print(table)


def _print_verdict(mc_result: MonteCarloResult) -> None:
    """Print the overall pass/fail verdict."""
    console.print()
    if mc_result.all_passed:
        console.print("[bold green]ALL ACCEPTANCE CRITERIA MET[/]")
    else:
        console.print("[bold red]ACCEPTANCE CRITERIA NOT MET[/]")
        for name, stats in mc_result.scenario_stats.items():
            if not stats.passed:
                label = SCENARIO_LABELS.get(name, name)
                frac = stats.n_compliant / stats.n_samples if stats.n_samples > 0 else 0.0
                console.print(
                    f"  [red]{label}:[/] {stats.n_compliant}/{stats.n_samples} "
                    f"compliant ({frac:.1%})"
                )


def _print_warnings(warnings: list[str]) -> None:
    """Print any collected warnings."""
    if warnings:
        console.print()
        for w in warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.group()
def main():
    """Leeds Flight Simulator — 6DoF Monte Carlo sounding rocket analysis."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--no-warn", is_flag=True, help="Suppress blocking warning prompts.")
@click.option("--no-popup", is_flag=True, help="Do not open figures after execution.")
def run(config_path: Path, no_warn: bool, no_popup: bool) -> None:
    """Run a Monte Carlo flight safety analysis."""
    config_path = Path(config_path).resolve()
    sim_cfg = load_simulation_config(config_path)
    all_warnings: list[str] = []
    _install_warning_hook(all_warnings, no_warn)

    # --- Detect wind profile mode (single file or directory) ---
    wind_path = sim_cfg.launch.wind_profiles
    if wind_path.is_dir():
        npz_files = sorted(wind_path.glob("*.npz"))
        if not npz_files:
            console.print(f"[red]Error:[/] No .npz files found in {wind_path}")
            sys.exit(1)
    else:
        npz_files = [wind_path]

    # --- Load models (using first wind profile for verification/optimisation) ---
    console.print("[bold]Loading configuration and models...[/]")
    vehicle_cfg, motor_model, aero_model, wind_ensemble = load_all_models(sim_cfg)

    # --- Verification (runs once, before optimisation) ---
    verification_figure_path: Path | None = None
    if sim_cfg.verification is not None:
        console.print("[bold]Running trajectory verification...[/]")
        ver_result = run_verification(sim_cfg, vehicle_cfg, motor_model, aero_model)
        if ver_result.passed:
            console.print("[green]Verification PASSED[/]")
        else:
            console.print("[red]Verification FAILED[/]")
            if ver_result.figure is not None:
                ver_fig_path = config_path.parent / "verification_plot.png"
                ver_result.figure.savefig(ver_fig_path, dpi=150, bbox_inches="tight")
                verification_figure_path = ver_fig_path
                if not no_popup:
                    _open_figures([ver_fig_path])
            msg = "Trajectory verification failed — see verification_plot.png"
            all_warnings.append(msg)
            _warn_blocking(msg, no_warn)

    # --- Optimisation (runs once) ---
    opt_result = None
    rail = sim_cfg.launch.rail
    az_is_auto = rail.azimuth == "auto"
    inc_is_auto = rail.inclination == "auto"

    if az_is_auto or inc_is_auto:
        console.print("[bold]Running optimisation...[/]")
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("—"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            opt_task = progress.add_task("Optimisation", total=100)

            def _opt_callback(phase_name: str, completed: int, total: int) -> None:
                progress.update(opt_task, description=phase_name,
                                completed=completed, total=total)

            try:
                opt_result = run_optimisation(
                    sim_cfg, vehicle_cfg, motor_model, aero_model,
                    wind_ensemble, _opt_callback,
                )
            except ValueError as exc:
                console.print(f"\n[bold red]Optimisation failed:[/] {exc}")
                sys.exit(1)

        console.print(
            f"  Selected azimuth: [cyan]{opt_result.selected_azimuth}°[/], "
            f"inclination: [cyan]{opt_result.selected_inclination}°[/]"
        )

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
            console.print(f"\n[bold]Wind profile: {wind_suffix}[/]")
            wind_ensemble = load_wind_ensemble(
                npz_path,
                sim_cfg.monte_carlo.samples,
                surface_wind=sim_cfg.launch.surface_wind,
            )

        # --- Monte Carlo run with progress ---
        active_scenarios = vehicle_cfg.recovery.active_scenarios
        n_samples = sim_cfg.monte_carlo.samples

        with Progress(
            TextColumn("[bold]{task.description:<16}"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("—"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            tasks = {}
            for scenario in active_scenarios:
                label = SCENARIO_LABELS.get(scenario, scenario)
                tasks[scenario] = progress.add_task(label, total=n_samples)

            def _mc_callback(scenario_name: str, completed: int, total: int) -> None:
                if scenario_name in tasks:
                    progress.update(tasks[scenario_name], completed=completed)

            mc_result = run_monte_carlo(
                sim_cfg, vehicle_cfg, motor_model, aero_model,
                wind_ensemble, azimuth_mean, inclination_mean,
                progress_callback=_mc_callback,
            )

        # Merge MC warnings
        all_warnings.extend(mc_result.warnings)

        # --- Generate altitude plot data ---
        altitude_data = _generate_altitude_curves(
            sim_cfg, vehicle_cfg, motor_model, aero_model,
            wind_ensemble, azimuth_mean, inclination_mean,
        )
        burnout_time = float(motor_model.times[-1])

        # --- Create results directory and write outputs ---
        results_dir = create_results_dir(config_path, wind_suffix)

        has_coastline = sim_cfg.site.coastline is not None
        has_coverage = (
            bool(sim_cfg.site.observation_stations)
            or sim_cfg.site.launch_observation_radius > 0
        )

        write_samples_csv(mc_result.all_results, results_dir, has_coastline, has_coverage)
        write_summary_yaml(
            mc_result, sim_cfg, opt_result, results_dir,
            simulation_yaml_path=config_path,
        )
        alt_path = save_altitude_plot(altitude_data, burnout_time, results_dir)
        disp_path = save_dispersion_plot(
            mc_result.all_results, sim_cfg,
            sim_cfg.monte_carlo.acceptance.compliance_threshold,
            results_dir,
        )
        figure_paths.extend([alt_path, disp_path])

        # --- Print results ---
        console.print()
        _print_results_table(mc_result)
        _print_verdict(mc_result)
        _print_warnings(all_warnings)
        console.print(f"\nResults saved to: [bold]{results_dir}[/]")

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

    sim_config_path = Path(summary["run_details"]["simulation_config"]).resolve()
    master_seed = int(summary["run_details"]["master_seed"])
    azimuth_mean = float(summary["azimuth_inclination"]["azimuth_mean"])
    inclination_mean = float(summary["azimuth_inclination"]["inclination_mean"])

    if seed is not None:
        master_seed = seed

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
