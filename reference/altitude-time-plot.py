"""
flight_plot.py — Rocket flight profile comparison plot.

To switch between fake and real data, comment/uncomment the two lines in main().
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M_TO_FT = 3.28084
M_TO_KM = 1 / 1000
S_TO_MIN = 1 / 60

SCENARIO_KEYS   = ["nominal", "ballistic", "drogue_only", "premature_main"]
SCENARIO_LABELS = ["Nominal", "Ballistic", "Drogue-only", "Premature Main"]
SCENARIO_COLORS = ["#2d7a2d", "black", "goldenrod", "red"]
SCENARIO_ALPHA  = [1.0, 0.6, 0.6, 0.6]

# ---------------------------------------------------------------------------
# Font size for ALL dashed-line labels (timestamps, event names, apogee label)
# ---------------------------------------------------------------------------
VLINE_LABEL_FONTSIZE = 12

# ---------------------------------------------------------------------------
# Fake data generation
# ---------------------------------------------------------------------------
def generate_fake_data(
    apogee_m: float = 15300.0,
    drogue_rate_ms: float = 45.0,
    main_rate_ms: float = 8.0,
    main_deploy_alt_m: float = 300.0,
    dt: float = 1.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Generate synthetic altitude vs. time data for four flight scenarios.
    All descents begin at apogee; ascent is a shared cosine ramp.
    """
    avg_ascent_speed = 250.0  # m/s
    t_apogee = apogee_m / avg_ascent_speed

    t_asc = np.arange(0, t_apogee + dt, dt)
    alt_asc = apogee_m * (1 - np.cos(np.pi * t_asc / t_apogee)) / 2

    def concat(t_desc, alt_desc):
        return (
            np.concatenate([t_asc,   t_apogee + t_desc[1:]]),
            np.concatenate([alt_asc, alt_desc[1:]]),
        )

    # Ballistic
    g = 9.81
    t_b = np.arange(0, np.sqrt(2 * apogee_m / g) + dt, dt)
    t_bal, alt_bal = concat(t_b, np.maximum(apogee_m - 0.5 * g * t_b**2, 0))

    # Drogue only
    t_d = np.arange(0, apogee_m / drogue_rate_ms + dt, dt)
    t_dro, alt_dro = concat(t_d, np.maximum(apogee_m - drogue_rate_ms * t_d, 0))

    # Premature main (main deploys at apogee)
    t_m = np.arange(0, apogee_m / main_rate_ms + dt, dt)
    t_pre, alt_pre = concat(t_m, np.maximum(apogee_m - main_rate_ms * t_m, 0))

    # Nominal: drogue to main_deploy_alt_m, then main to ground
    t_phase1 = (apogee_m - main_deploy_alt_m) / drogue_rate_ms
    t_p1 = np.arange(0, t_phase1 + dt, dt)
    alt_p1 = apogee_m - drogue_rate_ms * t_p1
    t_phase2 = main_deploy_alt_m / main_rate_ms
    t_p2 = np.arange(0, t_phase2 + dt, dt)
    alt_p2 = np.maximum(main_deploy_alt_m - main_rate_ms * t_p2, 0)
    t_nom, alt_nom = concat(
        np.concatenate([t_p1, t_phase1 + t_p2[1:]]),
        np.concatenate([alt_p1, alt_p2[1:]]),
    )

    return {
        "nominal":        (t_nom,  alt_nom),
        "ballistic":      (t_bal,  alt_bal),
        "drogue_only":    (t_dro,  alt_dro),
        "premature_main": (t_pre,  alt_pre),
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def load_scenario_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column CSV (time_s, altitude_m)."""
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def load_all_csvs(directory: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load nominal/ballistic/drogue_only/premature_main CSVs from a directory."""
    scenarios = {}
    for key in SCENARIO_KEYS:
        path = os.path.join(directory, f"{key}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CSV: {path}")
        scenarios[key] = load_scenario_csv(path)
    return scenarios


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _break_mark(ax, side: str):
    """
    Single diagonal slash at the x-axis break edge.
    Drawn in display (pixel) space so both sides are identical in size and angle.
    """
    # Anchor: edge of this axis at y=0, in display coords
    x_frac = 1.0 if side == "right" else 0.0
    disp_x, disp_y = ax.transAxes.transform((x_frac, 0))
    dx, dy = 4, 9   # half-extents in pixels — small, fixed, same on both sides
    fig = ax.get_figure()
    # Convert pixel offsets to figure-fraction for plotting
    fw, fh = fig.get_size_inches() * fig.dpi
    ax.plot(
        [(disp_x - dx) / fw, (disp_x + dx) / fw],
        [(disp_y - dy) / fh, (disp_y + dy) / fh],
        transform=fig.transFigure,
        color=ax.spines["bottom"].get_edgecolor(),
        clip_on=False,
        linewidth=ax.spines["bottom"].get_linewidth(),
        solid_capstyle="round",
    )


def _ft_formatter(x, _):
    return f"{int(round(x)):,}"


def _km_formatter(x, _):
    return f"{x / M_TO_FT * M_TO_KM:.0f}"


# ---------------------------------------------------------------------------
# Main plot builder
# ---------------------------------------------------------------------------
def build_plot(
    burnout_t_s: float,
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str | None = None,
):
    LEFT_FRAC  = 0.70
    RIGHT_FRAC = 0.30

    # Convert all scenario time arrays from seconds to minutes for plotting
    scenarios_min = {
        k: (t * S_TO_MIN, alt) for k, (t, alt) in scenarios.items()
    }
    burnout_t_min = burnout_t_s * S_TO_MIN

    # --- Left panel x-range (minutes) ---
    left_keys  = ["nominal", "ballistic", "drogue_only"]
    t_left_max = np.ceil(max(scenarios_min[k][0][-1] for k in left_keys) * 1.05 * 10) / 10

    # --- Right panel: same scale, gap auto-sized so premature_main landing is visible ---
    t_right_span  = t_left_max * (RIGHT_FRAC / LEFT_FRAC)
    t_pm_land_min = scenarios_min["premature_main"][0][-1]
    t_right_end   = max(
        t_left_max + 1.0 + t_right_span,   # default: at least 1 min gap
        t_pm_land_min * 1.01,
    )
    t_right_start = t_right_end - t_right_span

    # --- Y limits ---
    alt_max_ft = max(np.max(scenarios[k][1]) for k in SCENARIO_KEYS) * M_TO_FT * 1.08

    # --- Figure ---
    fig = plt.figure(figsize=(14, 7))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.10)

    ax_l, ax_r = fig.subplots(
        1, 2, sharey=True,
        gridspec_kw={"width_ratios": [LEFT_FRAC, RIGHT_FRAC], "wspace": 0.04},
    )

    # --- Draw curves (reversed so Nominal paints on top) ---
    handles = []
    for i in reversed(range(len(SCENARIO_KEYS))):
        t_min  = scenarios_min[SCENARIO_KEYS[i]][0]
        alt_ft = scenarios_min[SCENARIO_KEYS[i]][1] * M_TO_FT
        kw = dict(color=SCENARIO_COLORS[i], linewidth=1.8, alpha=SCENARIO_ALPHA[i])
        line, = ax_l.plot(t_min, alt_ft, **kw)
        ax_r.plot(t_min, alt_ft, **kw)
        handles.insert(0, line)

    # --- Axis limits ---
    ax_l.set_xlim(0, t_left_max)
    ax_r.set_xlim(t_right_start, t_right_end)
    ax_l.set_ylim(0, alt_max_ft)

    # --- Consistent tick spacing on both panels (in minutes) ---
    raw_interval_min = t_left_max / 7
    for step in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        if raw_interval_min <= step:
            tick_interval_min = step
            break
    else:
        tick_interval_min = round(raw_interval_min)
    ax_l.xaxis.set_major_locator(mticker.MultipleLocator(tick_interval_min))
    ax_r.xaxis.set_major_locator(mticker.MultipleLocator(tick_interval_min))

    # --- Spines ---
    ax_l.spines[["right", "top"]].set_visible(False)
    ax_r.spines[["left",  "top", "right"]].set_visible(False)
    ax_r.yaxis.set_visible(False)
    ax_r.tick_params(axis="y", left=False)

    # --- Break marks: called late so transAxes pixel positions are finalised ---
    fig.canvas.draw()   # force layout so transform coords are accurate
    _break_mark(ax_l, "right")
    _break_mark(ax_r, "left")

    # --- Primary Y axis: ft ---
    ax_l.yaxis.set_major_formatter(mticker.FuncFormatter(_ft_formatter))
    ax_l.set_ylabel("Altitude (ft)", labelpad=8)

    # --- Secondary Y axis: km (offset left spine) ---
    ax_km = ax_l.twinx()
    ax_km.set_ylim(ax_l.get_ylim())
    ax_km.yaxis.set_ticks_position("left")
    ax_km.yaxis.set_label_position("left")
    ax_km.spines["left"].set_position(("outward", 68))
    ax_km.spines[["right", "top", "bottom"]].set_visible(False)
    ft_ticks = ax_l.yaxis.get_major_locator().tick_values(0, alt_max_ft)
    ft_ticks = ft_ticks[(ft_ticks >= 0) & (ft_ticks <= alt_max_ft)]
    ax_km.set_yticks(ft_ticks)
    ax_km.yaxis.set_major_formatter(mticker.FuncFormatter(_km_formatter))
    ax_km.set_ylabel("Altitude (km)", labelpad=8)

    # --- Shared x-axis label centred across both panels ---
    l_pos, r_pos = ax_l.get_position(), ax_r.get_position()
    fig.text((l_pos.x0 + r_pos.x1) / 2, 0.02,
             "Flight Time (min)", ha="center", va="bottom", fontsize=11)

    # --- Horizontal apogee line spanning both panels + centred label ---
    apogee_ft = max(np.max(scenarios[k][1]) for k in SCENARIO_KEYS) * M_TO_FT
    apogee_km = apogee_ft / M_TO_FT * M_TO_KM
    hline_kw  = dict(color="grey", linewidth=0.9, linestyle="--", zorder=0)
    ax_l.axhline(apogee_ft, **hline_kw)
    ax_r.axhline(apogee_ft, **hline_kw)

    # Label centred across both panels, sitting just above the dashed line
    label_str = f"Apogee ({apogee_ft:,.0f} ft / {apogee_km:.1f} km)"
    l_pos, r_pos = ax_l.get_position(), ax_r.get_position()
    mid_x_fig = (l_pos.x0 + r_pos.x1) / 2
    # Convert apogee_ft to figure-y coordinate
    apogee_axes_frac = (apogee_ft - 0) / alt_max_ft        # fraction within axes
    apogee_fig_y = l_pos.y0 + apogee_axes_frac * l_pos.height
    fig.text(mid_x_fig, apogee_fig_y, label_str,
             ha="center", va="bottom", fontsize=VLINE_LABEL_FONTSIZE, color="grey",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # --- Vertical dashed lines at burnout, apogee time and each landing time ---
    def fmt_mmin(t_min: float) -> str:
        """Format a time in minutes as m:ss (e.g. 1:07)."""
        total_s = int(round(t_min * 60))
        return f"{total_s // 60}:{total_s % 60:02d}"

    def find_landing_min(t_min, alt_m) -> float:
        """Return time (min) of landing: first sample at altitude=0 after apogee."""
        apogee_idx = int(np.argmax(alt_m))
        idx = np.where(alt_m[apogee_idx:] <= 0)[0]
        return t_min[apogee_idx + idx[0]] if len(idx) else None

    apogee_t_min = scenarios_min["nominal"][0][np.argmax(scenarios_min["nominal"][1])]
    vline_events = [
        (burnout_t_min, "Burnout"),
        (apogee_t_min, "Apogee"),
        *[
            (find_landing_min(*scenarios_min[k]), lbl)
            for k, lbl in zip(
                SCENARIO_KEYS,
                ["Nominal Landing", "Ballistic Landing", "Drogue-only Landing", "Premature Main Landing"],
            )
            if find_landing_min(*scenarios_min[k]) is not None
        ],
    ]

    # Vertical label positions staggered to reduce overlap: alternate between
    # lower-third and mid-height of the axes.
    label_y_fracs = [0.05]

    vline_kw = dict(color="grey", linewidth=0.9, linestyle="--", zorder=0)
    bbox_kw  = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)

    for idx, (t_min_ev, event_name) in enumerate(vline_events):
        ax_l.axvline(t_min_ev, **vline_kw)
        ax_r.axvline(t_min_ev, **vline_kw)

        ax = ax_l if ax_l.get_xlim()[0] <= t_min_ev <= ax_l.get_xlim()[1] else ax_r
        y_frac = label_y_fracs[idx % len(label_y_fracs)]
        y_pos  = alt_max_ft * y_frac

        timestamp = fmt_mmin(t_min_ev)
        combined_label = f"{event_name}\n{timestamp}"

        ax.text(
            t_min_ev, y_pos, combined_label,
            rotation=90, ha="left", va="bottom",
            fontsize=VLINE_LABEL_FONTSIZE, color="grey",
            bbox=bbox_kw,
        )

    ax_r.legend(handles=handles, labels=SCENARIO_LABELS,
                loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Entry point — comment/uncomment one line to switch data source
# ---------------------------------------------------------------------------
def main():
    scenarios = generate_fake_data()
    # scenarios = load_all_csvs("path/to/csv/dir")

    build_plot(6.2, scenarios, save_path=None)


if __name__ == "__main__":
    main()