"""
F1 25 telemetry analyzer
========================
Turns a CSV from f1_logger.py into the engineer overlay:
speed / delta / throttle / brake / RPM / gear vs lap distance,
comparing any two laps (default: your two fastest).

Usage:
    python -m workspace.f1_analyze f1_session_20260714_193000.csv
    python -m workspace.f1_analyze f1_session_xxx.csv --laps 3 5

The web UI imports load / lap_summary / build_figure / render_png from here.
"""

import io
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RENDER_DPI = 130          # for the inline web chart
DEFAULT_COLORS = ("#3671C6", "#FF8000")   # lap1 (blue), lap2 (orange)


def load(csv_path):
    df = pd.read_csv(csv_path)
    # keep only rows where the car is actually on a lap
    df = df[(df["lap"] > 0) & (df["lap_distance_m"] >= 0)].copy()
    return df


def lap_summary(df):
    """Best available lap time per lap: max running lap_time_ms seen on that lap."""
    out = []
    for lap, g in df.groupby("lap"):
        out.append({"lap": int(lap),
                    "time_s": g["lap_time_ms"].max() / 1000.0,
                    "max_dist": g["lap_distance_m"].max(),
                    "samples": len(g)})
    s = pd.DataFrame(out)
    if s.empty:
        return s
    # a "complete" lap covers nearly the full track length seen in the file
    track_len = s["max_dist"].max()
    s["complete"] = s["max_dist"] > 0.98 * track_len
    return s.sort_values("time_s")


def _select_pass(g):
    """One lap number can hold several physical passes over the track when the
    game's lap counter doesn't advance (flashbacks, session resets, telemetry
    glitches). Plotting all passes sorted by distance interleaves them into a
    filled band, so split on the big backward distance jumps (start/finish
    crossings) and keep the single pass that achieved the lap's max running
    time — i.e. the pass the lap summary reports."""
    g = g.reset_index(drop=True)
    d = g["lap_distance_m"].values
    resets = np.where(np.diff(d) < -1000.0)[0] + 1
    if len(resets) == 0:
        return g
    bounds = [0, *resets.tolist(), len(g)]
    passes = [g.iloc[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    best_idx = int(g["lap_time_ms"].values.argmax())
    for p in passes:
        if p.index[0] <= best_idx <= p.index[-1]:
            return p
    return max(passes, key=len)


def _forward_only(g):
    """Keep only forward-progress samples: each point must exceed the furthest
    distance reached so far. A mid-lap rewind/flashback makes distance jump
    backwards and re-cover ground already driven; sorting those by distance
    overlays two drives at the same point (a filled band). Dropping the rewound
    samples leaves one strictly-increasing, single-line trace."""
    d = g["lap_distance_m"].values
    prev_peak = np.concatenate(([-np.inf], np.maximum.accumulate(d)[:-1]))
    return g[d > prev_peak]


def get_lap(df, lap_num):
    g = df[df["lap"] == lap_num]
    g = _select_pass(g)          # pick the right pass across full-lap resets
    g = _forward_only(g)         # then drop sub-lap rewinds within that pass
    return g


def delta_time(lap_a, lap_b):
    """
    Real delta: both laps logged (running lap time) at each distance.
    Interpolate lap B's time onto lap A's distance points and subtract.
    Positive = B behind A at that point.
    """
    common_d = lap_a["lap_distance_m"].values
    t_a = lap_a["lap_time_ms"].values / 1000.0
    t_b = np.interp(common_d, lap_b["lap_distance_m"].values,
                    lap_b["lap_time_ms"].values / 1000.0)
    return common_d, t_b - t_a


def pick_laps(summary, laps=None):
    """Return (lap1, lap2) either from an explicit request or the two fastest
    complete laps. Raises ValueError if there aren't enough complete laps."""
    if laps:
        return int(laps[0]), int(laps[1])
    complete = summary[summary["complete"]]
    if len(complete) < 2:
        raise ValueError("Need at least two complete laps to compare. "
                         "Pass explicit laps to force a comparison.")
    return int(complete["lap"].iloc[0]), int(complete["lap"].iloc[1])


def build_figure(df, lap1_n, lap2_n, color1=DEFAULT_COLORS[0], color2=DEFAULT_COLORS[1]):
    """Build the 6-panel engineer overlay comparing two laps.
    color1 / color2 are the line colors for lap1 / lap2.
    Returns (figure, info) where info has the net delta at the line."""
    A = get_lap(df, lap1_n)
    B = get_lap(df, lap2_n)
    d, delta = delta_time(A, B)

    fig, ax = plt.subplots(6, 1, figsize=(14, 15), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.6, 1, 0.7, 1.1, 0.8]})

    ax[0].plot(A["lap_distance_m"], A["speed_kmh"], label=f"Lap {lap1_n}", color=color1)
    ax[0].plot(B["lap_distance_m"], B["speed_kmh"], label=f"Lap {lap2_n}", color=color2)
    ax[0].set_ylabel("Speed (km/h)")
    ax[0].legend(loc="lower left")
    ax[0].set_title(f"Lap {lap1_n} vs Lap {lap2_n} - your own telemetry")

    ax[1].plot(d, delta, color="purple")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel(f"Delta (s)\n+ = Lap {lap2_n} behind")
    net = float(delta[-1]) if len(delta) else 0.0
    ax[1].annotate(f"net at line: {net:+.3f}s", xy=(0.99, 0.06),
                   xycoords="axes fraction", ha="right",
                   bbox=dict(boxstyle="round", fc="lavender"))

    ax[2].plot(A["lap_distance_m"], A["throttle"] * 100, color=color1)
    ax[2].plot(B["lap_distance_m"], B["throttle"] * 100, color=color2)
    ax[2].set_ylabel("Throttle %")

    ax[3].plot(A["lap_distance_m"], A["brake"] * 100, color=color1)
    ax[3].plot(B["lap_distance_m"], B["brake"] * 100, color=color2)
    ax[3].set_ylabel("Brake %")   # console F1 gives full analog brake data

    ax[4].plot(A["lap_distance_m"], A["rpm"], color=color1)
    ax[4].plot(B["lap_distance_m"], B["rpm"], color=color2)
    ax[4].set_ylabel("RPM")

    ax[5].plot(A["lap_distance_m"], A["gear"], color=color1, drawstyle="steps-post")
    ax[5].plot(B["lap_distance_m"], B["gear"], color=color2, drawstyle="steps-post")
    ax[5].set_ylabel("Gear")
    ax[5].set_xlabel("Lap distance (m)")

    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    return fig, {"net_delta": net}


def render_png(df, lap1_n, lap2_n, color1=DEFAULT_COLORS[0], color2=DEFAULT_COLORS[1]):
    """Render the overlay to PNG bytes (used by the web UI). Non-interactive."""
    fig, info = build_figure(df, lap1_n, lap2_n, color1=color1, color2=color2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=RENDER_DPI, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--laps", nargs=2, type=int, default=None,
                    help="two lap numbers to compare (default: two fastest complete)")
    args = ap.parse_args()

    df = load(args.csv)
    if df.empty:
        sys.exit("No lap data in this file - did the logger run while you drove?")

    summary = lap_summary(df)
    print("\nLaps in this session:")
    print(summary.to_string(index=False))

    try:
        lap1_n, lap2_n = pick_laps(summary, args.laps)
    except ValueError as e:
        sys.exit(str(e))

    fig, info = build_figure(df, lap1_n, lap2_n)
    out = args.csv.replace(".csv", f"_lap{lap1_n}_vs_lap{lap2_n}.png")
    fig.savefig(out, dpi=150)
    print(f"\nNet delta at line: {info['net_delta']:+.3f}s")
    print(f"Chart saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
