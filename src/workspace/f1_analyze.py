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

# Corner positions as a fraction of lap distance (turn -> frac), per circuit.
CORNER_FRAC = {1: 0.0731, 2: 0.0998, 3: 0.1459, 4: 0.3267, 5: 0.3728, 6: 0.469,
               7: 0.5035, 8: 0.5495, 9: 0.5686, 10: 0.6261, 11: 0.6872,
               12: 0.7449, 13: 0.787, 14: 0.8451, 15: 0.9307}   # Interlagos
YAS_FRAC = {1: 0.0833, 2: 0.0883, 3: 0.1072, 4: 0.153, 5: 0.2777, 6: 0.5068,
            7: 0.5186, 8: 0.5235, 9: 0.7102, 10: 0.7242, 11: 0.8246, 12: 0.833,
            13: 0.8379, 14: 0.8523, 15: 0.8754, 16: 0.9731}   # Yas Marina
BARCA_FRAC = {1: 0.172, 2: 0.18, 3: 0.188, 4: 0.377, 5: 0.46, 6: 0.538,
              7: 0.547, 8: 0.558, 9: 0.636, 10: 0.752, 11: 0.808, 12: 0.82,
              13: 0.883, 14: 0.941}   # Barcelona, calibrated to braking zones from telemetry
MELB_FRAC = {1: 0.0747, 2: 0.0872, 3: 0.2113, 4: 0.2367, 5: 0.2702, 6: 0.3594,
             7: 0.3663, 8: 0.4163, 9: 0.621, 10: 0.6596, 11: 0.7885, 12: 0.8366,
             13: 0.8857, 14: 0.9041}   # Albert Park (calibrated: apexes on curvature peaks)
SHANGHAI_FRAC = {1: 0.1083, 2: 0.1208, 3: 0.1458, 4: 0.1708, 5: 0.225,
                 6: 0.2708, 7: 0.35, 8: 0.4125, 9: 0.4458, 10: 0.4667,
                 11: 0.5542, 12: 0.575, 13: 0.6, 14: 0.8583, 15: 0.8792,
                 16: 0.925}   # Shanghai (arc-length; recalibrate from laps)
SUZUKA_FRAC = {1: 0.1231, 2: 0.1462, 3: 0.1923, 4: 0.2115, 5: 0.2423,
               6: 0.2692, 7: 0.3038, 8: 0.3962, 9: 0.4269, 10: 0.4808,
               11: 0.5077, 12: 0.5692, 13: 0.6615, 14: 0.6885, 15: 0.8615,
               16: 0.9269, 17: 0.9423, 18: 0.9577}   # Suzuka (arc-length)
BAHRAIN_FRAC = {1: 0.1333, 2: 0.1542, 3: 0.175, 4: 0.2833, 5: 0.35,
                6: 0.3583, 7: 0.3833, 8: 0.4167, 9: 0.4833, 10: 0.5042,
                11: 0.6375, 12: 0.7083, 13: 0.7625, 14: 0.9,
                15: 0.925}   # Bahrain (rough; recalibrate)
SAUDI_FRAC = {1: 0.0667, 2: 0.075, 3: 0.1042, 4: 0.1542, 5: 0.1708, 6: 0.2,
              7: 0.2125, 8: 0.2208, 9: 0.25, 10: 0.2667, 11: 0.275, 12: 0.3,
              13: 0.3583, 14: 0.4292, 15: 0.475, 16: 0.4875, 17: 0.5083,
              18: 0.5292, 19: 0.5708, 20: 0.6042, 21: 0.6458, 22: 0.6875,
              23: 0.7042, 24: 0.7208, 25: 0.7708, 26: 0.8375,
              27: 0.8917}   # Jeddah (rough; recalibrate from laps)
CORNER_FRAC_BY_TRACK = {"br-1940": CORNER_FRAC, "ae-2009": YAS_FRAC,
                        "es-1991": BARCA_FRAC, "au-1953": MELB_FRAC,
                        "cn-2004": SHANGHAI_FRAC, "jp-1962": SUZUKA_FRAC,
                        "bh-2002": BAHRAIN_FRAC, "sa-2021": SAUDI_FRAC}


def corners_for(track):
    """Corner fractions for a circuit id, defaulting to Interlagos."""
    return CORNER_FRAC_BY_TRACK.get(track, CORNER_FRAC)


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
    """Keep the most recent pass over each point: a sample survives only if its
    distance is less than every distance that comes after it. A rewind/flashback
    re-covers ground already driven, so this drops the aborted earlier pass —
    including the slow-down that triggered the flashback — and keeps the clean
    re-driven line, leaving one strictly-increasing trace with no seam spike."""
    d = g["lap_distance_m"].values
    later_min = np.minimum.accumulate(d[::-1])[::-1]          # later_min[i] = min(d[i:])
    next_min = np.concatenate((later_min[1:], [np.inf]))      # min of everything after i
    return g[d < next_min]


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


def _smooth(series, n, tight=False):
    """Take the integer-quantisation 'stairs' out of a channel with a light centred
    rolling mean, sized to the lap's sample count so corner detail is kept. `tight`
    uses a smaller window for on/off channels (throttle/brake) to preserve edges."""
    w = min(41, max(5, n // (600 if tight else 200)))
    return series.rolling(w, center=True, min_periods=1).mean()


def build_figure(sources, laps, colors=DEFAULT_COLORS, corners=None, labels=None):
    """Build the 6-panel engineer overlay comparing two laps.
    sources is (dfA, dfB) — the same dataframe twice for an intra-session compare,
    or two different sessions' dataframes to compare a lap across sessions.
    laps is (lapA, lapB); colors is (lapA, lapB) line colors; corners maps turn ->
    lap fraction; labels are the (lapA, lapB) legend names. Returns (figure, info)."""
    color1, color2 = colors
    corners = corners if corners is not None else CORNER_FRAC
    dfa, dfb = sources
    lap1_n, lap2_n = laps
    la, lb = labels if labels else (f"Lap {lap1_n}", f"Lap {lap2_n}")
    A = get_lap(dfa, lap1_n)
    B = get_lap(dfb, lap2_n)
    if len(A) < 2 or len(B) < 2:
        raise ValueError("Not enough data for one of the selected laps.")
    d, delta = delta_time(A, B)

    fig, ax = plt.subplots(6, 1, figsize=(20, 15), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.6, 1, 0.7, 1.1, 0.8]})

    na, nb = len(A), len(B)
    ax[0].plot(A["lap_distance_m"], _smooth(A["speed_kmh"], na), label=la, color=color1)
    ax[0].plot(B["lap_distance_m"], _smooth(B["speed_kmh"], nb), label=lb, color=color2)
    ax[0].set_ylabel("Speed (km/h)")
    ax[0].legend(loc="lower left")
    ax[0].set_title(f"Lap {lap1_n} vs Lap {lap2_n} - your own telemetry", pad=38)

    ax[1].plot(d, delta, color="purple")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel("Delta (s)")
    # sign key with the (possibly long) lap/session name — centred above the panel
    # so it reads at any length and never collides with an axis label.
    ax[1].annotate(f"+ = {lb} behind", xy=(0.5, 1.0), xycoords="axes fraction",
                   ha="center", va="bottom", fontsize=9, color="0.4")
    net = float(delta[-1]) if len(delta) else 0.0
    ax[1].annotate(f"net at line: {net:+.3f}s", xy=(0.99, 0.06),
                   xycoords="axes fraction", ha="right",
                   bbox=dict(boxstyle="round", fc="lavender"))

    ax[2].plot(A["lap_distance_m"], _smooth(A["throttle"], na, tight=True) * 100, color=color1)
    ax[2].plot(B["lap_distance_m"], _smooth(B["throttle"], nb, tight=True) * 100, color=color2)
    ax[2].set_ylabel("Throttle %")

    ax[3].plot(A["lap_distance_m"], _smooth(A["brake"], na, tight=True) * 100, color=color1)
    ax[3].plot(B["lap_distance_m"], _smooth(B["brake"], nb, tight=True) * 100, color=color2)
    ax[3].set_ylabel("Brake %")   # console F1 gives full analog brake data

    ax[4].plot(A["lap_distance_m"], _smooth(A["rpm"], na), color=color1)
    ax[4].plot(B["lap_distance_m"], _smooth(B["rpm"], nb), color=color2)
    ax[4].set_ylabel("RPM")

    ax[5].plot(A["lap_distance_m"], A["gear"], color=color1, drawstyle="steps-post")
    ax[5].plot(B["lap_distance_m"], B["gear"], color=color2, drawstyle="steps-post")
    ax[5].set_ylabel("Gear")
    ax[5].set_xlabel("Lap distance (m)")

    for a in ax:
        a.grid(alpha=0.25)
    # turn markers: a dashed line at each corner and its own T# label. Every turn is
    # labelled individually (never merged) on ONE horizontal row; a label that would
    # collide with the previous one is slid sideways just enough to stay clear, while
    # its dashed line stays on the true corner position.
    track_len = float(A["lap_distance_m"].max())
    items = sorted(corners.items(), key=lambda kv: kv[1])
    for _, frac in items:
        for a in ax:
            a.axvline(frac * track_len, color="0.5", lw=0.6, ls=(0, (3, 3)), alpha=0.35)
    min_sep = track_len * 0.011           # smallest gap between adjacent labels
    last_x = -1e9
    for turn, frac in items:
        x = max(frac * track_len, last_x + min_sep)
        ax[0].annotate(f"T{turn}", xy=(x, 1.02), xycoords=("data", "axes fraction"),
                       ha="center", va="bottom", fontsize=7, color="0.4")
        last_x = x
    fig.tight_layout()
    return fig, {"net_delta": net}


def render_png(sources, laps, colors=DEFAULT_COLORS, corners=None, labels=None):
    """Render the overlay to PNG bytes (used by the web UI). Non-interactive."""
    fig, info = build_figure(sources, laps, colors, corners, labels)
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

    fig, info = build_figure((df, df), (lap1_n, lap2_n))
    out = args.csv.replace(".csv", f"_lap{lap1_n}_vs_lap{lap2_n}.png")
    fig.savefig(out, dpi=150)
    print(f"\nNet delta at line: {info['net_delta']:+.3f}s")
    print(f"Chart saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
