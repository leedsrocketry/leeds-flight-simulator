"""Parse RASAero CDX1 files and compare against LFS YAML configuration.

Provides CDX1 (XML) parsing, YAML field extraction, and a structured
comparison with percentage mismatch computation.  Used by the ``diff``
CLI command.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from atmosphere import temperature as _isa_temperature, pressure as _isa_pressure


# ---------------------------------------------------------------------------
# Unit conversions (CDX1 → SI)
# ---------------------------------------------------------------------------

def _in_to_m(val: float) -> float:
    return val * 0.0254

def _ft_to_m(val: float) -> float:
    return val * 0.3048

def _lb_to_kg(val: float) -> float:
    return val * 0.45359237

def _f_to_k(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0 + 273.15

def _inhg_to_pa(inhg: float) -> float:
    return inhg * 3386.389


# ---------------------------------------------------------------------------
# CDX1 parser
# ---------------------------------------------------------------------------

def parse_cdx1(path: Path, motor_hint: str | None = None) -> dict:
    """Extract vehicle, launch, recovery, and motor config from a CDX1 file.

    Parameters
    ----------
    motor_hint
        Case-insensitive substring matched against ``SustainerEngine`` to
        select the simulation entry.  When *None*, the first entry is used.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    design = root.find("RocketDesign")
    nose = design.find("NoseCone")
    body = design.find("BodyTube")
    boattail = design.find("BoatTail")
    fin = boattail.find("Fin")

    site = root.find("LaunchSite")
    recovery = root.find("Recovery")

    # Select simulation entry
    sim_list = root.find("SimulationList")
    sim = None
    motor_matched = False
    if motor_hint is not None:
        hint_lower = motor_hint.lower()
        for s in sim_list.findall("Simulation"):
            if hint_lower in s.findtext("SustainerEngine", "").lower():
                sim = s
                motor_matched = True
                break

    if sim is None:
        all_sims = sim_list.findall("Simulation")
        if not all_sims:
            raise ValueError(f"No <Simulation> entries in {path.name}")
        sim = all_sims[0]
        if motor_hint is not None:
            warnings.warn(
                f"No CDX1 simulation matching '{motor_hint}'; "
                f"using first entry: "
                f"{sim.findtext('SustainerEngine', '').strip()}"
            )

    # Geometry (inches → metres)
    nose_len = float(nose.findtext("Length"))
    body_len = float(body.findtext("Length"))
    bt_len = float(boattail.findtext("Length"))
    diameter = float(nose.findtext("Diameter"))
    bt_rear_diam = float(boattail.findtext("RearDiameter"))
    total_length_in = nose_len + body_len + bt_len

    # Recovery — parachute diameters in inches, deploy altitude in feet
    drogue_diam_in = float(recovery.findtext("Size1"))
    main_diam_in = float(recovery.findtext("Size2"))
    main_alt_ft = float(recovery.findtext("Altitude2"))

    return {
        "total_length_m": _in_to_m(total_length_in),
        "diameter_m": _in_to_m(diameter),
        "boattail_rear_diameter_m": _in_to_m(bt_rear_diam),
        "nozzle_diameter_m": _in_to_m(float(sim.findtext("SustainerNozzleDiameter"))),
        "launch_mass_kg": _lb_to_kg(float(sim.findtext("SustainerLaunchWt"))),
        "cg_m": _in_to_m(float(sim.findtext("SustainerCG"))),
        "rod_angle_deg": float(site.findtext("RodAngle")),
        "rod_length_m": _ft_to_m(float(site.findtext("RodLength"))),
        "temperature_K": _f_to_k(float(site.findtext("Temperature"))),
        "pressure_Pa": _inhg_to_pa(float(site.findtext("Pressure"))),
        "altitude_m": float(site.findtext("Altitude")),
        "drogue_cd": float(recovery.findtext("CD1")),
        "drogue_diameter_m": _in_to_m(drogue_diam_in),
        "main_cd": float(recovery.findtext("CD2")),
        "main_diameter_m": _in_to_m(main_diam_in),
        "main_deploy_alt_m": _ft_to_m(main_alt_ft),
        "drogue_deploy": recovery.findtext("EventType1"),
        "main_deploy": recovery.findtext("EventType2"),
        "motor_name": sim.findtext("SustainerEngine").strip(),
        "motor_matched": motor_matched,
    }


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_yaml_for_diff(sim_path: Path, veh_path: Path) -> dict:
    """Load fields from the simulation and vehicle YAMLs for comparison."""
    with open(sim_path, encoding="utf-8") as f:
        sim = yaml.safe_load(f)
    with open(veh_path, encoding="utf-8") as f:
        veh = yaml.safe_load(f)

    geom = veh["geometry"]
    mass = veh["mass"]
    rec = veh.get("recovery", {})
    rail = sim["launch"]["rail"]
    ver = sim.get("verification") or {}

    drogue = rec.get("drogue") or {}
    main = rec.get("main") or {}

    drogue_deploy = str(drogue.get("threshold", "")) if drogue else ""
    main_deploy = str(main.get("threshold", "")) if main else ""

    # Normalise deployment style to CDX1 convention
    drogue_deploy_style = "Apogee" if drogue_deploy == "apogee" else "Altitude"
    main_deploy_style = "Apogee" if main_deploy == "apogee" else "Altitude"

    main_deploy_alt = float(main_deploy) if main_deploy not in ("apogee", "") else None

    # Prefer verification inclination (matches verify command precedence).
    # Rail inclination may be "auto" — treat as None so the comparison is skipped.
    ver_incl = ver.get("inclination")
    if ver_incl is not None:
        rail_inclination = ver_incl
    else:
        raw = rail["inclination"]
        rail_inclination = None if raw == "auto" else raw

    return {
        "total_length_m": geom["length"],
        "diameter_m": geom["diameter"],
        "nozzle_diameter_m": geom["nozzle_diameter"],
        "launch_mass_kg": mass["wet_mass"],
        "cg_m": mass["wet_cg"],
        "rail_inclination_deg": rail_inclination,
        "rail_length_m": rail["length"],
        "drogue_cd": drogue.get("cd"),
        "drogue_diameter_m": drogue.get("diameter"),
        "main_cd": main.get("cd"),
        "main_diameter_m": main.get("diameter"),
        "main_deploy_alt_m": main_deploy_alt,
        "drogue_deploy": drogue_deploy_style if drogue else "",
        "main_deploy": main_deploy_style if main else "",
        "motor_file": veh["motor"],
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    """One row of the CDX1-vs-YAML comparison table."""
    label: str
    cdx1_val: float | str
    yaml_val: float | str
    diff_pct: float | None   # None for string-only comparisons
    passed: bool
    # YAML key path for --force updates; None = informational only
    yaml_key: str | None = None
    yaml_file: Literal["vehicle", "sim"] | None = None


def _sig3(val: float) -> str:
    """Format a float to 3 significant figures."""
    if val == 0.0:
        return "0.00"
    return f"{val:.3g}"


def _pct(a: float, b: float) -> float:
    """Percentage difference between two numbers."""
    denom = max(abs(a), abs(b), 1e-12)
    return 100.0 * abs(a - b) / denom


def build_comparison(cdx1: dict, yaml_cfg: dict,
                     threshold: float = 0.05) -> list[ComparisonRow]:
    """Build structured comparison rows from parsed CDX1 and YAML data.

    *threshold* is the fractional tolerance (0.05 = 5%).
    """
    thr_pct = threshold * 100.0
    rows: list[ComparisonRow] = []

    def num(label: str, cdx1_key: str, yaml_key: str,
            cdx1_val: float | None = None, yaml_val: float | None = None,
            force_key: str | None = None,
            force_file: Literal["vehicle", "sim"] | None = None) -> None:
        c = cdx1_val if cdx1_val is not None else cdx1.get(cdx1_key)
        y = yaml_val if yaml_val is not None else yaml_cfg.get(yaml_key)
        if c is None or y is None:
            return
        c, y = float(c), float(y)
        pct = _pct(c, y)
        rows.append(ComparisonRow(
            label=label, cdx1_val=c, yaml_val=y,
            diff_pct=pct, passed=pct <= thr_pct,
            yaml_key=force_key, yaml_file=force_file,
        ))

    def string(label: str, cdx1_val: str, yaml_val: str) -> None:
        rows.append(ComparisonRow(
            label=label, cdx1_val=cdx1_val, yaml_val=yaml_val,
            diff_pct=None, passed=cdx1_val == yaml_val,
        ))

    # RASAero rod_angle=0 means vertical (90 deg from horizontal)
    rasaero_inclination = 90.0 - cdx1["rod_angle_deg"]

    # --- Geometry ---
    num("Total length (m)", "total_length_m", "total_length_m",
        force_key="geometry.length", force_file="vehicle")
    num("Diameter (m)", "diameter_m", "diameter_m",
        force_key="geometry.diameter", force_file="vehicle")
    num("Nozzle diameter (m)", "nozzle_diameter_m", "nozzle_diameter_m",
        force_key="geometry.nozzle_diameter", force_file="vehicle")

    # --- Mass ---
    num("Launch mass (kg)", "launch_mass_kg", "launch_mass_kg",
        force_key="mass.wet_mass", force_file="vehicle")
    num("CG from nose (m)", "cg_m", "cg_m",
        force_key="mass.wet_cg", force_file="vehicle")

    # --- Launch ---
    num("Inclination (deg)", "", "",
        cdx1_val=rasaero_inclination,
        yaml_val=yaml_cfg["rail_inclination_deg"],
        force_key="launch.rail.inclination", force_file="sim")
    num("Rail length (m)", "rod_length_m", "rail_length_m",
        force_key="launch.rail.length", force_file="sim")

    # --- Atmosphere (informational — ISA-derived, not updateable) ---
    num("Temperature (K)", "", "",
        cdx1_val=cdx1["temperature_K"],
        yaml_val=float(_isa_temperature(cdx1["altitude_m"])))
    num("Pressure (Pa)", "", "",
        cdx1_val=cdx1["pressure_Pa"],
        yaml_val=float(_isa_pressure(cdx1["altitude_m"])))

    # --- Recovery ---
    num("Drogue Cd", "drogue_cd", "drogue_cd",
        force_key="recovery.drogue.cd", force_file="vehicle")
    num("Drogue diameter (m)", "drogue_diameter_m", "drogue_diameter_m",
        force_key="recovery.drogue.diameter", force_file="vehicle")
    num("Main Cd", "main_cd", "main_cd",
        force_key="recovery.main.cd", force_file="vehicle")
    num("Main diameter (m)", "main_diameter_m", "main_diameter_m",
        force_key="recovery.main.diameter", force_file="vehicle")
    num("Main deploy alt (m)", "main_deploy_alt_m", "main_deploy_alt_m",
        force_key="recovery.main.threshold", force_file="vehicle")

    # --- Strings (informational) ---
    string("Drogue deployment", cdx1["drogue_deploy"], yaml_cfg["drogue_deploy"])
    string("Main deployment", cdx1["main_deploy"], yaml_cfg["main_deploy"])
    rows.append(ComparisonRow(
        label="Motor", cdx1_val=cdx1["motor_name"],
        yaml_val=yaml_cfg["motor_file"],
        diff_pct=None, passed=cdx1["motor_matched"],
    ))

    return rows


def sort_rows(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    """Sort rows by descending mismatch percentage (strings last)."""
    def key(r: ComparisonRow) -> float:
        if r.diff_pct is None:
            return 0.0  # strings sort last
        return -r.diff_pct
    return sorted(rows, key=key)


# ---------------------------------------------------------------------------
# Force-update YAML files
# ---------------------------------------------------------------------------

def _set_nested(data, dotted_key: str, new_val: float) -> None:
    """Set a value in a ruamel.yaml CommentedMap by dotted key path."""
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = round(new_val, 10)


def apply_force_updates(
    rows: list[ComparisonRow],
    sim_path: Path,
    veh_path: Path,
) -> tuple[int, int]:
    """Write CDX1 values into the YAML files for rows that failed.

    Uses ruamel.yaml for round-trip editing that preserves comments and
    formatting.  Returns (n_vehicle_updates, n_sim_updates).
    """
    from ruamel.yaml import YAML

    ryaml = YAML()
    ryaml.preserve_quotes = True

    # Collect updates grouped by file
    veh_updates: list[tuple[str, float]] = []
    sim_updates: list[tuple[str, float]] = []

    for row in rows:
        if row.passed or row.yaml_key is None:
            continue
        cdx1_val = float(row.cdx1_val)
        if row.yaml_file == "vehicle":
            veh_updates.append((row.yaml_key, cdx1_val))
        elif row.yaml_file == "sim":
            sim_updates.append((row.yaml_key, cdx1_val))

    if veh_updates:
        with open(veh_path, encoding="utf-8") as f:
            data = ryaml.load(f)
        for key, val in veh_updates:
            _set_nested(data, key, val)
        with open(veh_path, "w", encoding="utf-8") as f:
            ryaml.dump(data, f)

    if sim_updates:
        with open(sim_path, encoding="utf-8") as f:
            data = ryaml.load(f)
        for key, val in sim_updates:
            _set_nested(data, key, val)
        with open(sim_path, "w", encoding="utf-8") as f:
            ryaml.dump(data, f)

    return len(veh_updates), len(sim_updates)
