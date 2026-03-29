"""Tests for the trajectory comparison tool (verify.py, §18.1).

Tests:
- Reference CSV loading: column matching, aliases, case insensitivity, missing columns
- Comparison logic: within/outside tolerance, near-zero handling, interpolation
- Plot generation: axes count, pass/fail colours
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from verify import (
    QuantityComparison,
    VerificationResult,
    _load_reference_csv,
    _compare_quantity,
    _build_comparison_figure,
    _match_column,
    _COMPARED_QUANTITIES,
)


# ---------------------------------------------------------------------------
# CSV loading tests
# ---------------------------------------------------------------------------

class TestLoadReferenceCsv:
    """Tests for _load_reference_csv."""

    def _write_csv(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "ref.csv"
        p.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
        return p

    def test_basic_rasaero_format(self, tmp_path: Path) -> None:
        """Columns from real RASAero output are matched and values pass through."""
        csv_path = self._write_csv(tmp_path, """\
            Time (sec),Mach Number,Weight (lb),Stability Margin (cal),Altitude (ft),Extra Col
            0.0,0.0,53.0,2.12,0.0,999
            1.0,0.5,52.0,2.10,100.0,999
            2.0,1.0,51.0,2.08,400.0,999
        """)
        data = _load_reference_csv(csv_path)

        assert set(data.keys()) == {"time", "altitude", "mach", "mass", "sm"}
        np.testing.assert_array_equal(data["time"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(data["mach"], [0.0, 0.5, 1.0])
        # No unit conversion — values pass through as-is
        np.testing.assert_array_equal(data["mass"], [53.0, 52.0, 51.0])
        np.testing.assert_array_equal(data["altitude"], [0.0, 100.0, 400.0])
        np.testing.assert_array_equal(data["sm"], [2.12, 2.10, 2.08])

    def test_case_insensitive(self, tmp_path: Path) -> None:
        """Mixed-case headers are matched correctly."""
        csv_path = self._write_csv(tmp_path, """\
            TIME (SEC),MACH NUMBER,Weight (LB),STABILITY MARGIN (cal),ALTITUDE (FT)
            0.0,0.0,53.0,2.12,0.0
        """)
        data = _load_reference_csv(csv_path)
        assert len(data["time"]) == 1

    def test_alternative_aliases(self, tmp_path: Path) -> None:
        """Alternative column names (aliases) are matched."""
        csv_path = self._write_csv(tmp_path, """\
            time,mach,mass,sm,alt
            0.0,0.0,53.0,2.12,0.0
            1.0,0.5,52.0,2.10,100.0
        """)
        data = _load_reference_csv(csv_path)
        assert len(data["altitude"]) == 2
        assert len(data["mass"]) == 2
        assert len(data["sm"]) == 2

    def test_height_alias(self, tmp_path: Path) -> None:
        """'height' is accepted as an alias for altitude."""
        csv_path = self._write_csv(tmp_path, """\
            time,mach,mass,stability,height
            0.0,0.0,53.0,2.12,0.0
        """)
        data = _load_reference_csv(csv_path)
        assert "altitude" in data

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        """ValueError raised when a required column is missing."""
        csv_path = self._write_csv(tmp_path, """\
            time,altitude,mass,sm
            0.0,0.0,53.0,2.12
        """)
        with pytest.raises(ValueError, match="mach"):
            _load_reference_csv(csv_path)

    def test_multiple_missing_columns(self, tmp_path: Path) -> None:
        """All missing columns are listed in the error."""
        csv_path = self._write_csv(tmp_path, """\
            time,altitude
            0.0,0.0
        """)
        with pytest.raises(ValueError, match="mach"):
            _load_reference_csv(csv_path)

    def test_first_match_wins(self, tmp_path: Path) -> None:
        """When two columns match the same alias, the first one wins."""
        csv_path = self._write_csv(tmp_path, """\
            time,mach speed,mach number,mass,sm,altitude
            0.0,0.1,0.2,53.0,2.12,0.0
        """)
        data = _load_reference_csv(csv_path)
        # "mach speed" appears first → should be the one matched
        assert data["mach"][0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Column alias matching
# ---------------------------------------------------------------------------

class TestMatchColumn:
    def test_substring_match(self) -> None:
        assert _match_column("stability margin (cal)", ["stability margin"])
        assert _match_column("mach number", ["mach"])

    def test_no_match(self) -> None:
        assert not _match_column("velocity", ["mach"])

    def test_partial_alias(self) -> None:
        assert _match_column("altitude (ft)", ["alt"])


# ---------------------------------------------------------------------------
# Comparison logic tests
# ---------------------------------------------------------------------------

class TestCompareQuantity:
    def test_within_tolerance(self) -> None:
        """All points within 5% tolerance → passed."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        ref = np.array([100.0, 200.0, 300.0, 400.0])
        sim = ref * 1.03  # 3% off, within 5% tolerance

        cmp = _compare_quantity("altitude", t, ref, t, sim, tolerance=0.05)

        assert cmp.passed is True
        assert np.all(cmp.within_tolerance)
        assert cmp.name == "altitude"
        assert cmp.tolerance == 0.05

    def test_outside_tolerance(self) -> None:
        """One point exceeds tolerance → not passed."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        ref = np.array([100.0, 200.0, 300.0, 400.0])
        sim = ref.copy()
        sim[2] = ref[2] * 1.10  # 10% off at index 2

        cmp = _compare_quantity("altitude", t, ref, t, sim, tolerance=0.05)

        assert cmp.passed is False
        assert cmp.within_tolerance[0] is np.True_
        assert cmp.within_tolerance[2] is np.False_

    def test_near_zero_values(self) -> None:
        """Values near zero use absolute floor, avoiding false failures."""
        t = np.array([0.0, 1.0, 2.0])
        ref = np.array([0.0, 0.001, 0.0])
        # Small absolute difference but huge relative difference
        sim = np.array([0.005, 0.006, 0.005])

        # With 5% tolerance and floor of 0.01 for Mach:
        # scale = max(|ref|, 0.01) = [0.01, 0.01, 0.01]
        # allowed = 0.05 * 0.01 = 0.0005
        # |sim - ref| = [0.005, 0.005, 0.005] > 0.0005 → all fail
        cmp = _compare_quantity("mach", t, ref, t, sim, tolerance=0.05)
        assert not cmp.passed

        # But with a larger tolerance it should pass
        cmp2 = _compare_quantity("mach", t, ref, t, sim, tolerance=1.0)
        assert cmp2.passed

    def test_interpolation(self) -> None:
        """Simulator at irregular timesteps is interpolated to reference grid."""
        ref_t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ref_v = np.array([0.0, 100.0, 200.0, 300.0, 400.0])

        # Simulator has fewer, irregular points
        sim_t = np.array([0.0, 2.5, 5.0])
        sim_v = np.array([0.0, 250.0, 500.0])  # linear: v = 100 * t

        cmp = _compare_quantity("altitude", ref_t, ref_v, sim_t, sim_v, tolerance=0.05)

        # np.interp should give [0, 100, 200, 300, 400] — exactly matching ref
        np.testing.assert_allclose(cmp.sim_values, ref_v, atol=1e-10)
        assert cmp.passed is True

    def test_time_clipping(self) -> None:
        """Only the overlapping time range is compared."""
        ref_t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ref_v = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])

        # Simulator ends earlier at t=3
        sim_t = np.array([0.0, 1.0, 2.0, 3.0])
        sim_v = np.array([0.0, 100.0, 200.0, 300.0])

        cmp = _compare_quantity("altitude", ref_t, ref_v, sim_t, sim_v, tolerance=0.05)

        # Should only compare up to t=3 (4 points)
        assert len(cmp.ref_time) == 4
        assert cmp.ref_time[-1] == 3.0
        assert cmp.passed is True


# ---------------------------------------------------------------------------
# Plot generation tests
# ---------------------------------------------------------------------------

class TestPlotGeneration:
    def _make_comparison(
        self, name: str, passed: bool,
    ) -> QuantityComparison:
        t = np.linspace(0, 10, 50)
        ref = np.sin(t) + 5.0
        sim = ref * (1.01 if passed else 1.5)
        within = np.abs(sim - ref) <= 0.05 * np.maximum(np.abs(ref), 1.0)
        return QuantityComparison(
            name=name,
            ref_time=t,
            ref_values=ref,
            sim_values=sim,
            tolerance=0.05,
            within_tolerance=within,
            passed=passed,
        )

    def test_figure_has_correct_axes(self) -> None:
        """The comparison figure should have one axis per compared quantity."""
        comparisons = {
            qty: self._make_comparison(qty, True)
            for qty in _COMPARED_QUANTITIES
        }
        fig = _build_comparison_figure(comparisons)

        assert len(fig.axes) == len(_COMPARED_QUANTITIES)
        plt.close(fig)

    def test_pass_colour_is_green(self) -> None:
        """When all quantities pass, simulator lines should be green."""
        comparisons = {
            qty: self._make_comparison(qty, True)
            for qty in _COMPARED_QUANTITIES
        }
        fig = _build_comparison_figure(comparisons)

        # Check the suptitle colour
        suptitle = fig._suptitle
        assert "PASS" in suptitle.get_text()
        assert suptitle.get_color() == "#2d7a2d"
        plt.close(fig)

    def test_fail_colour_is_red(self) -> None:
        """When a quantity fails, its overlay is red and title says FAIL."""
        comparisons = {
            qty: self._make_comparison(qty, qty != "mach")
            for qty in _COMPARED_QUANTITIES
        }
        fig = _build_comparison_figure(comparisons)

        suptitle = fig._suptitle
        assert "FAIL" in suptitle.get_text()
        assert "mach" in suptitle.get_text()
        assert suptitle.get_color() == "red"
        plt.close(fig)
