"""Command-line interface — ``run``, ``replay``, ``verify``, and ``diff`` (§17).

Provides the Click CLI group invoked by ``__main__.py``.  Uses ``rich``
for progress bars, warning panels, and status output.

See the **Command conventions** comment block (above ``@click.group``)
for the patterns every command must follow.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import click
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
    REASON_COLS,
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
    save_replay_aoa,
    save_damping_breakdown,
    save_damping,
    save_replay_plan_view,
    save_replay_roll_rate,
    ReplayPicker,
    write_samples_csv,
    write_summary_yaml,
)
from verify import run_verification
from wind import load_wind_ensemble

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
            parts.append(Text())
        if self._progress.tasks:
            parts.append(self._progress)
        return Group(*parts)

    def _refresh(self) -> None:
        self._live.update(self._build())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _QuietGroup(click.Group):
    """Suppress Click's ``Aborted!`` and style uncaught exceptions."""

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except KeyboardInterrupt:
            raise SystemExit(130)
        except (click.exceptions.Exit, click.Abort, click.ClickException):
            raise
        except Exception as exc:
            console.print(Panel(
                f"{type(exc).__name__}: {exc}",
                border_style="red", title="ERROR", title_align="left",
            ))
            raise SystemExit(1)


def _start_warning_capture(
    display: _RunDisplay | None = None,
) -> tuple[list[str], object]:
    """Begin routing ``warnings.warn()`` to *display* (if provided).

    Returns ``(collected_warnings, original_handler)``.  Call
    ``_stop_warning_capture(original_handler)`` when finished.
    """
    collected: list[str] = []
    original = warnings.showwarning

    def _hook(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: object = None,
        line: str | None = None,
    ) -> None:
        text = str(message)
        collected.append(text)
        if display is not None:
            display.add_warning(text)

    warnings.showwarning = _hook
    return collected, original


def _stop_warning_capture(original: object) -> None:
    """Restore the warning handler returned by ``_start_warning_capture``."""
    warnings.showwarning = original  # type: ignore[assignment]


def _error_exit(message: str, display: _RunDisplay | None = None) -> None:
    """Print a red ERROR panel and terminate the process.

    Stops *display* first if one is running.  All error exits in every
    command should go through this function so the format is consistent.
    """
    if display is not None:
        display.stop()
    console.print(Panel(
        message, border_style="red", title="ERROR", title_align="left",
    ))
    raise SystemExit(1)


def _print_warnings(warnings_list: list[str]) -> None:
    """Print collected warnings as a yellow WARNINGS panel.

    Used by commands that do not use a live display (e.g. ``diff``).
    Commands that *do* use ``_RunDisplay`` show warnings inline via
    ``_start_warning_capture(display)`` — no need to call this afterwards.
    """
    if not warnings_list:
        return
    bullet_list = "\n".join(f"• {w}" for w in warnings_list)
    console.print(Panel(
        bullet_list,
        border_style="yellow",
        title="WARNINGS",
        title_align="left",
    ))


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
            _error_exit(
                f"Could not clear results directory — a file may be locked "
                f"by another program.\n\n{exc}",
                display,
            )



def _tile_figures() -> None:
    """Arrange all open matplotlib figures in a grid filling the screen."""
    import matplotlib.pyplot as plt
    figs = [plt.figure(n) for n in plt.get_fignums()]
    n = len(figs)
    if n == 0:
        return
    try:
        screen = figs[0].canvas.manager.window.screen()
        geom = screen.availableGeometry()
        sx, sy, sw, sh = geom.x(), geom.y(), geom.width(), geom.height()
    except Exception:
        return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cw = sw // cols
    ch = sh // rows
    for i, fig in enumerate(figs):
        r, c = divmod(i, cols)
        try:
            win = fig.canvas.manager.window
            win.setGeometry(sx + c * cw, sy + r * ch, cw, ch)
            win.showMinimized()
            win.showNormal()
            win.raise_()
            win.activateWindow()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Command conventions (read before adding or editing commands)
# ---------------------------------------------------------------------------
#
# Every ``@main.command()`` must follow these patterns so the CLI behaves
# consistently.  Helpers above exist to enforce them — use them rather than
# rolling your own.
#
# Warnings
#     w, orig = _start_warning_capture()           # before config loading
#     sim_cfg = load_simulation_config(...)
#     display = _RunDisplay(console)
#     _stop_warning_capture(orig)                   # stop displayless capture
#     for w in early: display.add_warning(w)        # replay into display
#     w, orig = _start_warning_capture(display)     # restart with display
#     ...
#     _stop_warning_capture(orig)
#   Start capture *before* ``load_simulation_config`` so model-loading
#   warnings are collected.  Replay them into the display when ready.
#   Always restore before ``display.stop()``.
#   For commands without a display, call ``_print_warnings(w)`` afterwards.
#
# Error exit
#     _error_exit("message", display)             # display is optional
#   Prints a red ERROR panel and terminates.  Never use bare
#   ``console.print("[red]Error:...")``, ``sys.exit``, or ``raise
#   SystemExit`` directly — always go through ``_error_exit`` so every
#   error looks the same.
#
#   ``_QuietGroup`` provides a catch-all: any uncaught exception from
#   any command is displayed as a red ERROR panel.  Per-command
#   ``try/except`` blocks are only needed for recovery (e.g. skip a
#   plot and continue) — not for styling.
#
# Vertical spacing
#   • After ``display.stop()``, print a blank ``console.print()`` to
#     separate the Rich output from any summary text.
#   • End the command's output with ``console.print()`` for a trailing
#     blank line before the shell prompt.
#   • Never embed ``\n`` in ``console.print()`` strings for spacing —
#     use a separate ``console.print()`` call instead.
#
# Keyboard interrupt
#   ``_QuietGroup`` suppresses Click's default "Aborted!" message.
#   Individual commands do not need to catch ``KeyboardInterrupt``.
#

@click.group(cls=_QuietGroup)
def main():
    """Leeds Flight Simulator — 6DoF Monte Carlo sounding rocket analysis."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("-q", "--no-popup", is_flag=True, help="Do not open figures after execution.")
@click.option("-p", "--points", is_flag=True, help="Draw apogee and landing points on the dispersion plot.")
@click.option("--no-termination", is_flag=True,
              help="Disable early termination on stability/AoA violations.")
@click.option("-s", "--scenario", "scenarios", multiple=True,
              type=click.Choice(list(SCENARIO_LABELS), case_sensitive=False),
              help="Scenario(s) to simulate and plot.  May be repeated.  "
                   "Default: all active scenarios.")
def run(config_path: Path, no_popup: bool, points: bool, no_termination: bool,
        scenarios: tuple[str, ...]) -> None:
    """Run a Monte Carlo flight safety analysis."""
    import matplotlib.pyplot as plt

    config_path = Path(config_path).resolve()
    all_warnings, _orig_warn = _start_warning_capture()
    sim_cfg = load_simulation_config(config_path)

    if no_termination:
        acc = sim_cfg.monte_carlo.acceptance
        acc = replace(acc, sm_subsonic_min=-1e9, sm_supersonic_min=-1e9)
        mc = replace(sim_cfg.monte_carlo, acceptance=acc)
        sim_cfg = replace(sim_cfg, monte_carlo=mc)
    display = _RunDisplay(console)
    _stop_warning_capture(_orig_warn)
    for w in all_warnings:
        display.add_warning(w)
    all_warnings, _orig_warn = _start_warning_capture(display)

    # --- Detect wind profile mode ---
    wind_path = sim_cfg.launch.wind_profiles
    if wind_path.is_dir():
        npz_files = sorted(wind_path.glob("*.npz"))
        if not npz_files:
            _error_exit(f"No .npz files found in {wind_path}")
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

        try:
            ver_result = run_verification(sim_cfg, vehicle, propellant, aero_model)
        except RuntimeError as exc:
            _error_exit(str(exc), display)
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
        warnings.warn("Verification section not configured - verification skipped.")

    # --- Optimisation (runs once) ---
    opt_result = None
    rail = sim_cfg.launch.rail
    az_is_auto = rail.azimuth == "auto"
    inc_is_auto = rail.inclination == "auto"

    if az_is_auto or inc_is_auto:
        display.update_status("Running optimisation...")
        opt_task = progress.add_task("Optimisation", total=4)
        display.start_task(opt_task)

        def _opt_callback(step: int) -> None:
            progress.update(opt_task, completed=step)

        try:
            opt_result = run_optimisation(
                sim_cfg, vehicle, propellant, aero_model,
                wind_ensemble, _opt_callback,
            )
        except ValueError as exc:
            _error_exit(f"Optimisation failed: {exc}", display)

        verdict = "[black on green] PASS [/]"
        display.finish_task(opt_task, description=f"{'Optimisation':<16} {verdict}")
        display.update_status(
            f"Optimisation complete — azimuth: {opt_result.selected_azimuth}°, "
            f"inclination: {opt_result.selected_inclination}°"
        )
    else:
        warnings.warn("Azimuth and inclination are fixed - optimisation skipped.")

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
        filter_scenarios = scenarios if scenarios else None
        display_scenarios = filter_scenarios if filter_scenarios else active_scenarios
        n_samples = sim_cfg.monte_carlo.samples

        display.update_status("Running Monte Carlo analysis...")
        tasks = {}
        for scenario in display_scenarios:
            label = SCENARIO_LABELS.get(scenario, scenario)
            tasks[scenario] = progress.add_task(label, total=n_samples)
            display.start_task(tasks[scenario])

        def _mc_callback(scenario_name: str, completed: int, total: int) -> None:
            if scenario_name in tasks:
                progress.update(tasks[scenario_name], completed=completed)

        def _scenario_done(scenario_name: str, stats) -> None:
            if scenario_name in tasks:
                label = SCENARIO_LABELS.get(scenario_name, scenario_name)
                total = stats.n_compliant + stats.n_non_compliant
                if stats.passed:
                    verdict = f"[black on green] PASS {stats.n_compliant}/{total} [/]"
                else:
                    verdict = f"[white on red] FAIL {stats.n_non_compliant}/{total} [/]"
                display.finish_task(
                    tasks[scenario_name],
                    description=f"{label:<16} {verdict}",
                )

        try:
            mc_result = run_monte_carlo(
                sim_cfg, vehicle, propellant, aero_model,
                wind_ensemble, azimuth_mean, inclination_mean,
                progress_callback=_mc_callback,
                scenario_done_callback=_scenario_done,
                filter_scenarios=filter_scenarios,
            )
        except RuntimeError as exc:
            _error_exit(str(exc), display)

        # MC warnings
        for w in mc_result.warnings:
            warnings.warn(w)

        # --- Plots and outputs ---
        display.update_status("Generating plots...")
        altitude_data = mc_result.baseline_curves
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

        # --- Damping assessment (from nominal baseline) ---
        if "nominal" in mc_result.baseline_curves:
            nominal_profile = mc_result.baseline_curves["nominal"][3]
            if nominal_profile.zeta is not None:
                from montecarlo import SampleResult
                nominal_sr = SampleResult(
                    sample_id=0, scenario="nominal", run_index=0,
                    apogee_m=0.0, apogee_lat=0.0, apogee_lon=0.0,
                    landing_lat=0.0, landing_lon=0.0,
                    apogee_north=0.0, apogee_east=0.0,
                    landing_north=0.0, landing_east=0.0,
                    flight_time_s=0.0, peak_mach=0.0, max_aoa_deg=0.0,
                    min_sm_subsonic=0.0, min_sm_supersonic=0.0,
                    compliant=True, footprint_compliant=True,
                    ceiling_compliant=True, stability_compliant=True,
                    landing_at_sea=None, in_coverage=None,
                    violation_reason="",
                    wind_profile_index=0, impulse_factor=1.0,
                    azimuth_deg=azimuth_mean,
                    inclination_deg=inclination_mean,
                    fin_cant_deg=0.0,
                    trajectory=nominal_profile,
                )
                damping_results = [nominal_sr]
                for save_fn, name in [
                    (save_damping, "damping"),
                    (save_damping_breakdown, "damping breakdown"),
                ]:
                    try:
                        result = save_fn(damping_results, sim_cfg,
                                         output_dir=plot_dir)
                        if isinstance(result, Path):
                            figure_paths.append(result)
                    except NotImplementedError as exc:
                        display.add_warning(f"{name} plot skipped — {exc}")

    _stop_warning_capture(_orig_warn)
    display.stop()
    console.print()

    if opt_result is not None:
        console.print(
            f"Optimised azimuth:     [white]{opt_result.selected_azimuth}°[/]"
        )
        console.print(
            f"Optimised inclination: [white]{opt_result.selected_inclination}°[/]"
        )
        console.print()

    console.print(f"Results saved to: [bold]{results_dir}[/]")
    console.print()

    if not no_popup:
        _tile_figures()
        plt.show()


@main.command()
@click.argument("summary_path", type=click.Path(exists=True, path_type=Path))
@click.option("--seed", type=int, default=None, help="Override master seed.")
@click.option("--scenario", "scenario_name", type=click.Choice(
    list(SCENARIO_LABELS), case_sensitive=False), default=None,
    help="Scenario name.  Required with --sample; optional filter for "
         "--compliant / --non-compliant.")
@click.option("--sample", "sample_index", type=int, default=None, help="Sample index.")
@click.option("--non-compliant", is_flag=True, help="Replay all non-compliant samples.")
@click.option(
    "--reason", "reasons", multiple=True,
    type=click.Choice(list(REASON_COLS), case_sensitive=False),
    help=(
        "Filter --non-compliant to a specific violation type; may be repeated. "
        "footprint: trajectory exited danger area; "
        "ceiling: apogee above altitude ceiling; "
        "stability: static margin violation; "
        "coastline: landing at sea; "
        "monitor: outside monitored area. "
        "Default: all non-compliant reasons."
    ),
)
@click.option("--compliant", is_flag=True, help="Replay all compliant samples.")
@click.option("-q", "--no-popup", is_flag=True, help="Save figures to disk instead of interactive display.")
def replay(
    summary_path: Path,
    seed: int | None,
    scenario_name: str | None,
    sample_index: int | None,
    non_compliant: bool,
    reasons: tuple[str, ...],
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
    single_replay = scenario_name is not None and sample_index is not None
    if not single_replay and not non_compliant and not compliant:
        _error_exit(
            "Specify either --scenario and --sample for a single "
            "replay, --non-compliant, or --compliant."
        )
    if non_compliant and compliant:
        _error_exit("--non-compliant and --compliant are mutually exclusive.")
    if reasons and not non_compliant:
        _error_exit("--reason requires --non-compliant.")
    if sample_index is not None and scenario_name is None:
        _error_exit("--sample requires --scenario.")

    # --- Live display (same style as run command) ---
    early_warnings, _early_orig = _start_warning_capture()
    display = _RunDisplay(console)
    _stop_warning_capture(_early_orig)
    for w in early_warnings:
        display.add_warning(w)
    _, _orig_warn = _start_warning_capture(display)

    try:
        display.start()

        # --- Load config and models ---
        display.update_status("Loading configuration and models...")
        sim_cfg = load_simulation_config(sim_config_path)

        master_seed = seed if seed is not None else sim_cfg.monte_carlo.seed

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
            display.update_status(
                f"Replaying seed={master_seed}, scenario={scenario_name}, "
                f"sample={sample_index}..."
            )
            results = [
                replay_sample(
                    sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
                    master_seed, scenario_name, sample_index,
                    azimuth_mean, inclination_mean,
                )
            ]
        else:
            samples_csv = summary_dir / "samples.csv"
            if not samples_csv.exists():
                _error_exit(f"samples.csv not found in {summary_dir}", display)
            if compliant:
                display.update_status("Replaying compliant samples...")
                results = replay_compliant(
                    sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
                    master_seed, azimuth_mean, inclination_mean, samples_csv,
                    scenario_filter=scenario_name,
                )
            else:
                filters: list[str] = []
                if scenario_name:
                    filters.append(f"scenario={scenario_name}")
                if reasons:
                    filters.append(f"reason={', '.join(reasons)}")
                if filters:
                    display.update_status(
                        f"Replaying non-compliant samples "
                        f"({'; '.join(filters)})..."
                    )
                else:
                    display.update_status("Replaying all non-compliant samples...")
                results = replay_non_compliant(
                    sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
                    master_seed, azimuth_mean, inclination_mean, samples_csv,
                    reasons=reasons,
                    scenario_filter=scenario_name,
                )

        # --- Check for compliance mismatches vs original run ---
        samples_csv_check = summary_dir / "samples.csv"
        if samples_csv_check.exists():
            import csv as _csv
            orig: dict[tuple[int, str], bool] = {}
            with open(samples_csv_check, newline="", encoding="utf-8") as _f:
                for _row in _csv.DictReader(_f):
                    _key = (int(_row["sample_id"]), _row["scenario"].strip())
                    _cols = [c for c in _row if c.endswith("_compliant")]
                    _ok = all(_row[c].strip().lower() not in ("false", "0", "no")
                             for c in _cols)
                    orig[_key] = _ok
            n_mismatch = sum(
                1 for sr in results
                if (sr.sample_id, sr.scenario) in orig
                and sr.compliant != orig[(sr.sample_id, sr.scenario)]
            )
            if n_mismatch > 0:
                warnings.warn(
                    f"{n_mismatch} of {len(results)} replayed samples have "
                    f"different compliance results to the original run. "
                    f"This is expected if the simulation config has changed "
                    f"since the run (e.g. integrator tolerances)."
                )

        if not results:
            display.stop()
            console.print()
            msg = "compliant" if compliant else "non-compliant"
            console.print(f"[green]No {msg} samples to replay.[/]")
            console.print()
            return

        # --- Compute damping on replayed trajectories (for roll rate limit) ---
        if aero_model.has_components:
            from damping import compute_damping
            for sr in results:
                if sr.trajectory is not None and sr.trajectory.zeta is None:
                    params = build_sim_params(
                        sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
                        sr.wind_profile_index, sr.azimuth_deg,
                        sr.inclination_deg, sr.impulse_factor,
                        sr.fin_cant_deg,
                    )
                    compute_damping(sr.trajectory, params)

        display.update_status("Generating plots...")

        # --- Generate replay figures ---
        import matplotlib.pyplot as plt

        out_dir = summary_dir if no_popup else None
        figure_paths: list[Path] = []
        replay_figures: list[plt.Figure] = []
        replay_plots: list[tuple] = [
            (save_replay_3d, "3D isometric"),
            (save_replay_plan_view, "plan view"),
            (save_replay_altitude, "altitude-time"),
            (save_replay_aoa, "angle of attack"),
        ]
        if aero_model.has_components:
            replay_plots.append((save_replay_roll_rate, "roll rate"))
        for save_fn, name in replay_plots:
            try:
                result = save_fn(results, sim_cfg, output_dir=out_dir)
                if isinstance(result, Path):
                    figure_paths.append(result)
                else:
                    replay_figures.append(result)
            except NotImplementedError as exc:
                display.add_warning(f"{name} plot skipped — {exc}")

        display.stop()
        console.print()

        if not no_popup:
            if replay_figures:
                _picker = ReplayPicker(replay_figures, results)  # noqa: F841
            _tile_figures()
        plt.show()

    finally:
        _stop_warning_capture(_orig_warn)


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
    early_warnings, _early_orig = _start_warning_capture()
    sim_cfg = load_simulation_config(config_path)

    if sim_cfg.verification is None:
        _stop_warning_capture(_early_orig)
        _print_warnings(early_warnings)
        _error_exit("No verification section in config.")

    display = _RunDisplay(console)
    _stop_warning_capture(_early_orig)
    for w in early_warnings:
        display.add_warning(w)
    _, _orig_warn = _start_warning_capture(display)

    display.start()

    display.update_status("Loading configuration and models...")
    vehicle, propellant, aero_model, _ = load_all_models(sim_cfg)

    display.update_status("Running 6DoF verification trajectory...")
    try:
        ver_result = run_verification(
            sim_cfg, vehicle, propellant, aero_model,
            inclination_override=inclination,
        )
    except RuntimeError as exc:
        _error_exit(str(exc), display)

    display.update_status("Done.")
    _stop_warning_capture(_orig_warn)
    display.stop()
    console.print()

    # --- Dump comparison data to CSV ---
    if dump_csv is not None:
        import csv
        dump_path = Path(dump_csv).resolve()
        qty_names = list(ver_result.comparisons.keys())
        # Use the longest timebase (full-flight quantities) for the row count
        max_len = max(len(c.ref_time) for c in ver_result.comparisons.values())
        longest = max(ver_result.comparisons.values(), key=lambda c: len(c.ref_time))

        with open(dump_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["time_s"]
            for q in qty_names:
                header.extend([f"ref_{q}", f"lfs_{q}", f"err_{q}"])
            writer.writerow(header)

            for i in range(max_len):
                row: list[object] = [f"{longest.ref_time[i]:.6f}"]
                for q in qty_names:
                    c = ver_result.comparisons[q]
                    if i < len(c.ref_time):
                        row.append(f"{c.ref_values[i]:.6f}")
                        row.append(f"{c.sim_values[i]:.6f}")
                        row.append(f"{c.sim_values[i] - c.ref_values[i]:.6f}")
                    else:
                        row.extend(["", "", ""])
                writer.writerow(row)

        console.print(f"Comparison data written to: {dump_path}")

    if ver_result.figure is not None:
        if no_popup:
            fig_path = config_path.parent / "results" / "verification_plot.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            ver_result.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
            console.print(f"Figure saved to: {fig_path}")
        else:
            _tile_figures()
        plt.show()

    console.print()


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("-q", "--no-popup", is_flag=True,
              help="Do not open figures after execution.")
def damping(config_path: Path, no_popup: bool) -> None:
    """Run a mean-wind nominal trajectory and assess pitch/yaw damping."""
    import matplotlib.pyplot as plt

    config_path = Path(config_path).resolve()
    early_warnings, _early_orig = _start_warning_capture()
    sim_cfg = load_simulation_config(config_path)

    display = _RunDisplay(console)
    _stop_warning_capture(_early_orig)
    for w in early_warnings:
        display.add_warning(w)
    _, _orig_warn = _start_warning_capture(display)

    display.start()

    display.update_status("Loading configuration and models...")
    vehicle, propellant, aero_model, wind_ensemble = load_all_models(sim_cfg)

    if not aero_model.has_components:
        _error_exit(
            "Damping assessment requires per-component aero tables "
            "(directory of CSVs, not a single file).",
            display,
        )

    # Resolve rail angles (fixed values — no optimisation)
    rail = sim_cfg.launch.rail
    azimuth_deg = 0.0 if rail.azimuth == "auto" else float(rail.azimuth)
    inclination_deg = 85.0 if rail.inclination == "auto" else float(rail.inclination)

    display.update_status("Running mean-wind nominal trajectory...")
    from dynamics import run_trajectory, SCENARIO_NOMINAL
    from damping import compute_damping
    from montecarlo import SampleResult

    params = build_sim_params(
        sim_cfg, vehicle, propellant, aero_model, wind_ensemble,
        wind_profile_index=0,
        azimuth_deg=azimuth_deg,
        inclination_deg=inclination_deg,
        impulse_factor=1.0,
        fin_cant_deg=0.0,
        wind_east_override=wind_ensemble.mean_east_ms,
        wind_north_override=wind_ensemble.mean_north_ms,
    )
    summary, profile = run_trajectory(params, SCENARIO_NOMINAL, keep_profile=True)

    display.update_status("Computing damping quantities...")
    compute_damping(profile, params)

    display.update_status("Generating plots...")

    nominal_sr = SampleResult(
        sample_id=0, scenario="nominal", run_index=0,
        apogee_m=summary.apogee, apogee_lat=0.0, apogee_lon=0.0,
        landing_lat=0.0, landing_lon=0.0,
        apogee_north=0.0, apogee_east=0.0,
        landing_north=0.0, landing_east=0.0,
        flight_time_s=0.0, peak_mach=summary.max_mach,
        max_aoa_deg=summary.max_aoa_deg,
        min_sm_subsonic=summary.min_sm_subsonic,
        min_sm_supersonic=summary.min_sm_supersonic,
        compliant=True, footprint_compliant=True,
        ceiling_compliant=True, stability_compliant=True,
        landing_at_sea=None, in_coverage=None,
        violation_reason="",
        wind_profile_index=0, impulse_factor=1.0,
        azimuth_deg=azimuth_deg,
        inclination_deg=inclination_deg,
        fin_cant_deg=0.0,
        trajectory=profile,
    )
    results = [nominal_sr]

    results_dir = config_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir = results_dir if no_popup else None

    figure_paths: list[Path] = []
    for save_fn, name in [
        (save_damping, "damping"),
        (save_damping_breakdown, "damping breakdown"),
    ]:
        try:
            result = save_fn(results, sim_cfg, output_dir=out_dir)
            if isinstance(result, Path):
                figure_paths.append(result)
        except NotImplementedError as exc:
            display.add_warning(f"{name} plot skipped — {exc}")

    _stop_warning_capture(_orig_warn)
    display.stop()
    console.print()

    if figure_paths:
        for fp in figure_paths:
            console.print(f"Saved: {fp}")
    elif not no_popup:
        _tile_figures()
        plt.show()

    console.print()
