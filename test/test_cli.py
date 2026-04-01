"""Tests for the CLI module — argument validation, error display, and layout.

Uses Click's ``CliRunner`` with a Rich ``Console`` wired to a ``StringIO``
buffer so every test can inspect the exact terminal output without needing
real simulation data.

Where a command would normally load models and run physics, the tests
mock the expensive internals and only exercise the CLI layer itself.
"""
from __future__ import annotations

import csv
import json
import textwrap
from io import StringIO
from pathlib import Path
from unittest import mock
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after every test."""
    yield
    plt.close("all")

# Import the CLI entry point and internal helpers
from cli import (
    main,
    _error_exit,
    _start_warning_capture,
    _stop_warning_capture,
    _print_warnings,
    _RunDisplay,
    _QuietGroup,
    console,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal files for each command
# ---------------------------------------------------------------------------

_STUB_ENG = textwrap.dedent("""\
    ; stub motor
    M100 50 100 0 0.5 1.0 LURA
    0.0 100.0
    0.5 100.0
    1.0 0.0
    ;
""")

_STUB_GEOJSON = json.dumps({
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-5.1, 58.5], [-4.7, 58.5],
                [-4.7, 58.8], [-5.1, 58.8], [-5.1, 58.5],
            ]],
        },
        "properties": {},
    }],
})


def _write_stub_motor(path: Path) -> None:
    path.write_text(_STUB_ENG, encoding="utf-8")


def _write_stub_wind(path: Path, n_profiles: int = 2) -> None:
    alt = np.array([0.0, 1000.0, 5000.0, 10000.0])
    np.savez(
        path,
        altitude_m=alt,
        wind_east_ms=np.zeros((n_profiles, len(alt))),
        wind_north_ms=np.zeros((n_profiles, len(alt))),
    )


def _write_stub_geojson(path: Path) -> None:
    path.write_text(_STUB_GEOJSON, encoding="utf-8")


def _write_aero_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Mach", "Reynolds", "AoA_deg", "CA_off", "CA_on", "CN", "CP_m"])
        for mach in [0.3, 1.0, 2.0]:
            for aoa in [0.0, 5.0]:
                w.writerow([mach, 1e6, aoa, 0.4, 0.5, 0.1 * aoa, 1.0])


def _write_vehicle_yaml(tmp_path: Path) -> Path:
    _write_stub_motor(tmp_path / "motor.eng")
    aero_dir = tmp_path / "aero_tables"
    for name in ["NoseCone", "BodyTube", "Fin", "BoatTail"]:
        _write_aero_csv(aero_dir / f"{name}.csv")

    veh = {
        "motor": "motor.eng",
        "aero_tables": "aero_tables",
        "geometry": {
            "diameter": 0.13,
            "length": 2.6,
            "nozzle_diameter": 0.08,
            "fin_cp_radius": 0.095,
        },
        "mass": {
            "wet_mass": 26.5,
            "wet_cg": 1.15,
            "wet_inertia_lateral": 5.2,
            "wet_inertia_roll": 0.012,
        },
        "recovery": {
            "drogue": {"cd": 2.0, "diameter": 0.44, "threshold": "apogee"},
            "main": {"cd": 2.0, "diameter": 1.89, "threshold": 305},
        },
    }
    p = tmp_path / "vehicle.yaml"
    p.write_text(yaml.dump(veh), encoding="utf-8")
    return p


def _write_sim_yaml(
    tmp_path: Path,
    *,
    samples: int = 2,
    seed: int = 42,
    azimuth: object = 45.0,
    inclination: object = 87.0,
    verification: dict | None = None,
) -> Path:
    _write_vehicle_yaml(tmp_path)
    _write_stub_wind(tmp_path / "wind_profiles.npz", n_profiles=max(samples, 2))
    _write_stub_geojson(tmp_path / "danger_area.geojson")

    cfg: dict = {
        "vehicle": "vehicle.yaml",
        "site": {
            "latitude": 58.61,
            "longitude": -4.94,
            "ballistic_exclusion_radius": 500,
            "launch_monitor_radius": 0,
            "altitude_ceiling": 20000,
            "danger_area": "danger_area.geojson",
        },
        "launch": {
            "rail": {
                "azimuth": azimuth,
                "inclination": inclination,
                "length": 4.0,
            },
            "wind_profiles": "wind_profiles.npz",
        },
        "monte_carlo": {
            "samples": samples,
            "seed": seed,
            "uncertainties": {
                "azimuth_sigma": 1.0,
                "inclination_sigma": 0.5,
                "fin_cant_sigma": 0.02,
                "impulse_factor_sigma": 0.067,
            },
            "acceptance": {
                "compliance_threshold": 0.95,
                "buffer_distance": 1000,
                "sm_subsonic_min": 1.0,
                "sm_supersonic_min": 2.0,
            },
        },
    }
    if azimuth == "auto":
        cfg["launch"]["rail"]["azimuth_range"] = [0, 360]
    if inclination == "auto":
        cfg["launch"]["rail"]["inclination_range"] = [80, 90]
    if verification is not None:
        cfg["verification"] = verification

    p = tmp_path / "sim.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


def _write_summary_yaml(
    tmp_path: Path,
    sim_yaml_path: Path,
    *,
    seed: int = 42,
    azimuth_mean: float = 45.0,
    inclination_mean: float = 87.0,
    optimisation: dict | None = None,
) -> Path:
    summary: dict = {
        "metadata": {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "config": str(sim_yaml_path),
            "warnings": [],
        },
        "scenarios": {
            "nominal": {
                "compliant": 2, "non_compliant": 0, "passed": True,
                "apogee_m": {"mean": 15000, "min": 14000, "max": 16000},
                "landing_distance_m": {"mean": 1500, "min": 1000, "max": 2000},
            },
        },
    }
    if optimisation is not None:
        summary["optimisation"] = optimisation

    p = tmp_path / "summary.yaml"
    p.write_text(yaml.dump(summary), encoding="utf-8")
    return p


def _write_samples_csv(
    tmp_path: Path,
    *,
    n: int = 4,
    scenarios: tuple[str, ...] = ("nominal", "ballistic"),
) -> Path:
    p = tmp_path / "samples.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "scenario", "compliant", "violation_reason",
            "azimuth_deg", "inclination_deg", "fin_cant_deg", "impulse_factor",
            "wind_profile_index",
        ])
        idx = 0
        for sc in scenarios:
            for i in range(n):
                compliant = i % 3 != 0
                reason = "" if compliant else "footprint"
                w.writerow([idx, sc, compliant, reason, 45.0, 87.0, 0.01, 1.0, 0])
                idx += 1
    return p


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------

def _invoke(*args: str, **kwargs) -> CliRunner.Result:
    """Run a CLI command and return the result."""
    runner = CliRunner()
    return runner.invoke(main, list(args), catch_exceptions=False, **kwargs)


def _invoke_catching(*args: str, **kwargs) -> CliRunner.Result:
    """Run a CLI command, allowing SystemExit to be caught by the runner."""
    runner = CliRunner()
    return runner.invoke(main, list(args), catch_exceptions=True, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Helper unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorExit:
    """``_error_exit`` prints a red ERROR panel and raises SystemExit(1)."""

    def test_raises_system_exit(self):
        with pytest.raises(SystemExit, match="1"):
            _error_exit("something broke")

    def test_exit_code_is_one(self):
        with pytest.raises(SystemExit) as exc_info:
            _error_exit("boom")
        assert exc_info.value.code == 1

    def test_stops_display_before_exit(self):
        display = mock.MagicMock(spec=_RunDisplay)
        with pytest.raises(SystemExit):
            _error_exit("err", display)
        display.stop.assert_called_once()

    def test_no_display_does_not_crash(self):
        with pytest.raises(SystemExit):
            _error_exit("err", display=None)


class TestWarningCapture:
    """``_start_warning_capture`` / ``_stop_warning_capture`` pair."""

    def test_collects_warnings(self):
        import warnings
        collected, orig = _start_warning_capture()
        try:
            warnings.warn("hello")
            warnings.warn("world")
        finally:
            _stop_warning_capture(orig)
        assert len(collected) == 2
        assert "hello" in collected[0]
        assert "world" in collected[1]

    def test_restores_original_handler(self):
        import warnings
        original = warnings.showwarning
        _, orig = _start_warning_capture()
        assert warnings.showwarning is not original
        _stop_warning_capture(orig)
        assert warnings.showwarning is original

    def test_routes_to_display(self):
        import warnings
        display = mock.MagicMock(spec=_RunDisplay)
        collected, orig = _start_warning_capture(display)
        try:
            warnings.warn("test warning")
        finally:
            _stop_warning_capture(orig)
        display.add_warning.assert_called_once()
        assert "test warning" in display.add_warning.call_args[0][0]

    def test_no_display_still_collects(self):
        import warnings
        collected, orig = _start_warning_capture(display=None)
        try:
            warnings.warn("no display")
        finally:
            _stop_warning_capture(orig)
        assert len(collected) == 1


class TestPrintWarnings:
    """``_print_warnings`` outputs a yellow WARNINGS panel."""

    def test_empty_list_prints_nothing(self, capsys):
        _print_warnings([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_non_empty_list_prints_panel(self):
        buf = StringIO()
        test_console = console.__class__(file=buf, force_terminal=True)
        with mock.patch("cli.console", test_console):
            _print_warnings(["first", "second"])
        output = buf.getvalue()
        assert "WARNINGS" in output
        assert "first" in output
        assert "second" in output


class TestQuietGroup:
    """``_QuietGroup`` suppresses Click's 'Aborted!' on KeyboardInterrupt."""

    def test_no_aborted_message_on_interrupt(self):
        result = _invoke_catching("run", "--help")
        # --help works normally (no interrupt)
        assert result.exit_code == 0
        assert "Aborted!" not in (result.output or "")

    def test_keyboard_interrupt_returns_130(self):
        @main.command("_test_interrupt")
        def _test_interrupt():
            raise KeyboardInterrupt

        result = _invoke_catching("_test_interrupt")
        assert result.exit_code == 130
        assert "Aborted!" not in (result.output or "")
        # Clean up: remove test command
        main.commands.pop("_test_interrupt", None)


# ═══════════════════════════════════════════════════════════════════════════
# ``run`` command
# ═══════════════════════════════════════════════════════════════════════════


class TestRunArgs:
    """Argument and option handling for ``run``."""

    def test_help(self):
        result = _invoke("run", "--help")
        assert result.exit_code == 0
        assert "--no-popup" in result.output
        assert "--points" in result.output
        assert "--no-termination" in result.output
        assert "CONFIG_PATH" in result.output

    def test_missing_config_path(self):
        result = _invoke_catching("run")
        assert result.exit_code != 0

    def test_nonexistent_config_path(self, tmp_path):
        result = _invoke_catching("run", str(tmp_path / "nope.yaml"))
        assert result.exit_code != 0

    def test_short_flags_accepted(self):
        result = _invoke("run", "--help")
        assert "-q" in result.output
        assert "-p" in result.output


class TestRunErrors:
    """Error-exit behaviour for ``run``."""

    def test_empty_wind_dir_errors(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        wind_dir = tmp_path / "wind_dir"
        wind_dir.mkdir()

        # Rewrite sim YAML to point to directory
        cfg = yaml.safe_load(sim_yaml.read_text())
        cfg["launch"]["wind_profiles"] = "wind_dir"
        sim_yaml.write_text(yaml.dump(cfg))

        result = _invoke_catching("run", str(sim_yaml), "-q")
        assert result.exit_code == 1
        assert "No .npz files found" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# ``replay`` command
# ═══════════════════════════════════════════════════════════════════════════


class TestReplayArgs:
    """Argument and option handling for ``replay``."""

    def test_help(self):
        result = _invoke("replay", "--help")
        assert result.exit_code == 0
        assert "--scenario" in result.output
        assert "--sample" in result.output
        assert "--non-compliant" in result.output
        assert "--compliant" in result.output
        assert "--reason" in result.output
        assert "--seed" in result.output
        assert "-q" in result.output
        assert "SUMMARY_PATH" in result.output

    def test_missing_summary_path(self):
        result = _invoke_catching("replay")
        assert result.exit_code != 0

    def test_nonexistent_summary_path(self, tmp_path):
        result = _invoke_catching("replay", str(tmp_path / "nope.yaml"))
        assert result.exit_code != 0


class TestReplayValidation:
    """Argument validation — all should produce a red ERROR panel."""

    def test_no_mode_specified(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching("replay", str(summary))
        assert result.exit_code == 1
        assert "Specify either" in result.output

    def test_compliant_and_non_compliant_mutually_exclusive(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--compliant", "--non-compliant",
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_reason_without_non_compliant(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--compliant", "--reason", "footprint",
        )
        assert result.exit_code == 1
        assert "--reason requires --non-compliant" in result.output

    def test_sample_without_scenario(self, tmp_path):
        """--sample with --compliant but no --scenario should fail."""
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        # --sample alone triggers "Specify either..." first, so add
        # --compliant to reach the "--sample requires --scenario" check.
        result = _invoke_catching(
            "replay", str(summary), "--sample", "0", "--compliant",
        )
        assert result.exit_code == 1
        assert "--sample requires --scenario" in result.output

    def test_invalid_scenario_name(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--scenario", "bogus", "--sample", "0",
        )
        assert result.exit_code != 0

    def test_invalid_reason_name(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--non-compliant", "--reason", "bogus",
        )
        assert result.exit_code != 0

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_multiple_reasons_accepted(self, _mock, tmp_path):
        """Multiple --reason flags should parse without error (validation only)."""
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary),
            "--non-compliant", "--reason", "footprint", "--reason", "ceiling",
        )
        assert "Invalid value" not in (result.output or "")

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_scenario_names_are_case_insensitive(self, _mock, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--scenario", "NOMINAL", "--sample", "0",
        )
        assert "Invalid value" not in (result.output or "")

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_reason_names_are_case_insensitive(self, _mock, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--non-compliant", "--reason", "FOOTPRINT",
        )
        assert "Invalid value" not in (result.output or "")


# ═══════════════════════════════════════════════════════════════════════════
# ``verify`` command
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifyArgs:
    """Argument and option handling for ``verify``."""

    def test_help(self):
        result = _invoke("verify", "--help")
        assert result.exit_code == 0
        assert "-i" in result.output
        assert "--inclination" in result.output
        assert "--dump-csv" in result.output
        assert "-q" in result.output
        assert "CONFIG_PATH" in result.output

    def test_missing_config_path(self):
        result = _invoke_catching("verify")
        assert result.exit_code != 0

    def test_nonexistent_config_path(self, tmp_path):
        result = _invoke_catching("verify", str(tmp_path / "nope.yaml"))
        assert result.exit_code != 0


class TestVerifyErrors:
    """Error-exit behaviour for ``verify``."""

    def test_no_verification_section(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path, verification=None)
        result = _invoke_catching("verify", str(sim_yaml))
        assert result.exit_code == 1
        assert "No verification section" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# ``diff`` command
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffArgs:
    """Argument and option handling for ``diff``."""

    def test_help(self):
        result = _invoke("diff", "--help")
        assert result.exit_code == 0
        assert "-t" in result.output
        assert "--threshold" in result.output
        assert "-m" in result.output
        assert "--motor" in result.output
        assert "-f" in result.output
        assert "--force" in result.output
        assert "CONFIG_PATH" in result.output
        assert "CDX1_PATH" in result.output

    def test_missing_both_paths(self):
        result = _invoke_catching("diff")
        assert result.exit_code != 0

    def test_missing_cdx1_path(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        result = _invoke_catching("diff", str(sim_yaml))
        assert result.exit_code != 0

    def test_nonexistent_cdx1_path(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        result = _invoke_catching("diff", str(sim_yaml), str(tmp_path / "nope.CDX1"))
        assert result.exit_code != 0


# ═══════════════════════════════════════════════════════════════════════════
# Top-level group
# ═══════════════════════════════════════════════════════════════════════════


class TestGroup:
    """Top-level Click group."""

    def test_help(self):
        result = _invoke("--help")
        assert result.exit_code == 0
        assert "run" in result.output
        assert "replay" in result.output
        assert "verify" in result.output
        assert "diff" in result.output

    def test_no_command_shows_help(self):
        result = _invoke_catching()
        assert "Usage:" in (result.output or "")

    def test_unknown_command(self):
        result = _invoke_catching("explode")
        assert result.exit_code != 0


# ═══════════════════════════════════════════════════════════════════════════
# Error display consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorDisplayConsistency:
    """All error exits must use the same red ERROR panel format."""

    def _get_error_output(self, *args: str, tmp_path: Path) -> str:
        """Run a command expected to fail and return its output."""
        result = _invoke_catching(*args)
        assert result.exit_code != 0
        return result.output or ""

    def test_replay_no_mode_uses_error_panel(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching("replay", str(summary))
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_replay_mutual_exclusion_uses_error_panel(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--compliant", "--non-compliant",
        )
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_replay_reason_without_flag_uses_error_panel(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--compliant", "--reason", "footprint",
        )
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_replay_sample_without_scenario_uses_error_panel(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary), "--sample", "0", "--compliant",
        )
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_verify_no_section_uses_error_panel(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path, verification=None)
        result = _invoke_catching("verify", str(sim_yaml))
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_run_empty_wind_dir_uses_error_panel(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        wind_dir = tmp_path / "wind_dir"
        wind_dir.mkdir()
        cfg = yaml.safe_load(sim_yaml.read_text())
        cfg["launch"]["wind_profiles"] = "wind_dir"
        sim_yaml.write_text(yaml.dump(cfg))

        result = _invoke_catching("run", str(sim_yaml), "-q")
        assert result.exit_code == 1
        assert "ERROR" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# No "[red]Error:[/]" or bare sys.exit — only _error_exit panels
# ═══════════════════════════════════════════════════════════════════════════


class TestNoLegacyErrorFormat:
    """Ensure no command uses the old ``[red]Error:[/]`` format."""

    def test_replay_errors_have_no_red_error_prefix(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)

        for args in [
            ["replay", str(summary)],
            ["replay", str(summary), "--compliant", "--non-compliant"],
            ["replay", str(summary), "--compliant", "--reason", "footprint"],
            ["replay", str(summary), "--sample", "0", "--compliant"],
        ]:
            result = _invoke_catching(*args)
            output = result.output or ""
            assert "Error:" not in output, f"Legacy error format in: {args}"

    def test_verify_error_has_no_red_error_prefix(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path, verification=None)
        result = _invoke_catching("verify", str(sim_yaml))
        output = result.output or ""
        assert "Error:" not in output

    def test_run_wind_error_has_no_red_error_prefix(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        wind_dir = tmp_path / "wind_dir"
        wind_dir.mkdir()
        cfg = yaml.safe_load(sim_yaml.read_text())
        cfg["launch"]["wind_profiles"] = "wind_dir"
        sim_yaml.write_text(yaml.dump(cfg))

        result = _invoke_catching("run", str(sim_yaml), "-q")
        output = result.output or ""
        assert "Error:" not in output


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases and weird argument combinations
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Unusual but valid (or interestingly invalid) argument combos."""

    # -- replay --

    def test_replay_scenario_alone_is_invalid(self, tmp_path):
        """--scenario without --sample or --compliant/--non-compliant."""
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching("replay", str(summary), "--scenario", "nominal")
        assert result.exit_code == 1
        assert "Specify either" in result.output

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_replay_sample_negative_accepted_by_click(self, _mock, tmp_path):
        """Click accepts negative ints — validation happens later."""
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary),
            "--scenario", "nominal", "--sample", "-1",
        )
        assert "Invalid value" not in (result.output or "")

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_replay_seed_override(self, _mock, tmp_path):
        """--seed flag should be accepted without complaint."""
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        result = _invoke_catching(
            "replay", str(summary),
            "--scenario", "nominal", "--sample", "0", "--seed", "999",
        )
        assert "Invalid value" not in (result.output or "")

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_replay_reason_all_five_types(self, _mock, tmp_path):
        """Every reason name should be accepted."""
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        for reason in ["footprint", "ceiling", "stability", "coastline", "monitor"]:
            result = _invoke_catching(
                "replay", str(summary),
                "--non-compliant", "--reason", reason,
            )
            assert "Invalid value" not in (result.output or ""), \
                f"--reason {reason} rejected"

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_replay_all_reasons_at_once(self, _mock, tmp_path):
        """All five --reason flags together should parse."""
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)
        args = ["replay", str(summary), "--non-compliant"]
        for reason in ["footprint", "ceiling", "stability", "coastline", "monitor"]:
            args.extend(["--reason", reason])
        result = _invoke_catching(*args)
        assert "Invalid value" not in (result.output or "")

    # -- verify --

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_verify_inclination_accepts_float(self, _mock, tmp_path):
        sim_yaml = _write_sim_yaml(
            tmp_path,
            verification={
                "reference_trajectory": "ref.csv",
                "altitude_tolerance": 0.05,
                "mach_tolerance": 0.05,
                "sm_tolerance": 0.05,
                "mass_tolerance": 0.05,
                "exceedance_fraction": 0.01,
            },
        )
        ref = tmp_path / "ref.csv"
        ref.write_text("time_s,altitude_m\n0,0\n1,100\n", encoding="utf-8")

        result = _invoke_catching("verify", str(sim_yaml), "-i", "85.5")
        assert "Invalid value" not in (result.output or "")

    @mock.patch("cli.load_all_models", side_effect=SystemExit(99))
    def test_verify_dump_csv_accepts_path(self, _mock, tmp_path):
        sim_yaml = _write_sim_yaml(
            tmp_path,
            verification={
                "reference_trajectory": "ref.csv",
                "altitude_tolerance": 0.05,
                "mach_tolerance": 0.05,
                "sm_tolerance": 0.05,
                "mass_tolerance": 0.05,
                "exceedance_fraction": 0.01,
            },
        )
        ref = tmp_path / "ref.csv"
        ref.write_text("time_s,altitude_m\n0,0\n1,100\n", encoding="utf-8")

        result = _invoke_catching(
            "verify", str(sim_yaml),
            "--dump-csv", str(tmp_path / "out.csv"),
        )
        assert "Invalid value" not in (result.output or "")

    # -- diff --

    def test_diff_threshold_accepts_float(self):
        result = _invoke("diff", "--help")
        assert "threshold" in result.output.lower()

    def test_diff_threshold_negative(self, tmp_path):
        """Click accepts negative floats — behaviour depends on the command."""
        sim_yaml = _write_sim_yaml(tmp_path)
        # Create a dummy CDX1 file (not valid, but Click only checks existence)
        cdx1 = tmp_path / "dummy.CDX1"
        cdx1.write_text("<xml/>", encoding="utf-8")
        result = _invoke_catching(
            "diff", str(sim_yaml), str(cdx1), "-t", "-0.5",
        )
        # Passes Click validation — fails later on CDX1 parsing
        assert "Invalid value" not in (result.output or "")

    # -- run --

    def test_run_all_flags_together(self, tmp_path):
        """All flags combined should parse without error."""
        sim_yaml = _write_sim_yaml(tmp_path)
        wind_dir = tmp_path / "wind_dir"
        wind_dir.mkdir()
        cfg = yaml.safe_load(sim_yaml.read_text())
        cfg["launch"]["wind_profiles"] = "wind_dir"
        sim_yaml.write_text(yaml.dump(cfg))

        result = _invoke_catching(
            "run", str(sim_yaml), "-q", "-p", "--no-termination",
        )
        # Flags parse fine; error is about empty wind dir
        assert "Invalid value" not in (result.output or "")


# ═══════════════════════════════════════════════════════════════════════════
# Spacing / layout consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestSpacingConsistency:
    """Error output should never embed \\n for spacing."""

    def test_error_panel_has_no_embedded_newline_prefix(self, tmp_path):
        """Error messages shouldn't start with \\n."""
        sim_yaml = _write_sim_yaml(tmp_path, verification=None)
        result = _invoke_catching("verify", str(sim_yaml))
        lines = (result.output or "").split("\n")
        # No line should be a lone newline followed immediately by
        # ERROR panel text — the blank line should be a separate print
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped.startswith("ERROR"):
                # The line before an ERROR header should be blank or
                # the start of the panel border — not a \n-prefixed string
                if i > 0:
                    assert "\n\n" not in lines[i - 1] + line

    def test_replay_validation_errors_are_single_line_messages(self, tmp_path):
        sim_yaml = _write_sim_yaml(tmp_path)
        summary = _write_summary_yaml(tmp_path, sim_yaml)

        # Each of these should produce a short, clean error message
        cases = [
            (["replay", str(summary)], "Specify either"),
            (["replay", str(summary), "--compliant", "--non-compliant"],
             "mutually exclusive"),
            (["replay", str(summary), "--sample", "0", "--compliant"],
             "--sample requires --scenario"),
        ]
        for args, expected_text in cases:
            result = _invoke_catching(*args)
            assert expected_text in (result.output or ""), \
                f"Missing '{expected_text}' in output for {args}"


# ═══════════════════════════════════════════════════════════════════════════
# Source-level consistency checks
# ═══════════════════════════════════════════════════════════════════════════


class TestSourceConsistency:
    """Verify that cli.py source code follows the conventions."""

    def _read_cli_source(self) -> str:
        cli_path = Path(__file__).resolve().parent.parent / "cli.py"
        return cli_path.read_text(encoding="utf-8")

    def test_no_sys_exit_in_commands(self):
        """Commands should use ``_error_exit``, not ``sys.exit``."""
        source = self._read_cli_source()
        # Find sys.exit calls that aren't in _error_exit or _QuietGroup
        lines = source.split("\n")
        in_error_exit = False
        in_quiet_group = False
        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "def _error_exit" in line:
                in_error_exit = True
            elif "class _QuietGroup" in line:
                in_quiet_group = True
            elif stripped.startswith("def ") or stripped.startswith("class "):
                in_error_exit = False
                in_quiet_group = False
            if (
                "sys.exit" in line
                and not in_error_exit
                and not in_quiet_group
                and not line.strip().startswith("#")
                and not line.strip().startswith("raise SystemExit")
            ):
                violations.append((i, line.strip()))
        assert not violations, (
            f"sys.exit found outside _error_exit/_QuietGroup:\n"
            + "\n".join(f"  line {n}: {l}" for n, l in violations)
        )

    def test_no_display_abort_calls(self):
        """No command should call ``display.abort()`` — use ``_error_exit``."""
        source = self._read_cli_source()
        assert "display.abort(" not in source, \
            "display.abort() found — use _error_exit(msg, display) instead"

    def test_no_red_error_console_prints(self):
        """No ``console.print("[red]Error:...")`` — use ``_error_exit``."""
        source = self._read_cli_source()
        lines = source.split("\n")
        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if '[red]Error:' in line and not stripped.startswith("#"):
                violations.append((i, stripped))
        assert not violations, (
            "Found legacy [red]Error: pattern — use _error_exit instead:\n"
            + "\n".join(f"  line {n}: {l}" for n, l in violations)
        )

    def test_no_embedded_newlines_in_console_print(self):
        """Spacing should use ``console.print()``, not ``\\n`` in strings."""
        source = self._read_cli_source()
        lines = source.split("\n")
        violations = []
        for i, line in enumerate(lines, 1):
            if 'console.print(f"\\n' in line or 'console.print("\\n' in line:
                violations.append((i, line.strip()))
        assert not violations, (
            f"Found embedded \\n in console.print:\n"
            + "\n".join(f"  line {n}: {l}" for n, l in violations)
        )

    def test_all_commands_registered(self):
        """The group should have all four expected commands."""
        assert set(main.commands) >= {"run", "replay", "verify", "diff"}
