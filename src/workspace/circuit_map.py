"""
Circuit map renderer
====================
Draws a track map from a *coordinate data file* (``circuits/<id>.json``) — the
layout, not captured telemetry. The layout data lives entirely in those files;
this module only knows how to read and draw them, so adding a circuit means
adding a file, never editing code.

What it draws:
    * the centreline outline, normalised and scaled to fit (aspect preserved)
    * the start/finish line and numbered corners (from the file's `labels`)
    * optionally, the trace colour-coded by a telemetry channel
      (throttle / brake / speed / gear) from a reference lap

Usage:
    f1-map <circuit-id>                                   # outline only
    f1-map <circuit-id> --csv session.csv --lap 5 --channel speed
    f1-map --list

The web UI imports load_circuit / render_map / render_png as needed.
"""

import io
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless: f1-map saves PNGs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

RENDER_DPI = 130

# channel -> (csv column, axis label, colormap, value scale). vmin/vmax are
# fixed where the channel has a natural range, else taken from the lap.
CHANNELS = {
    "speed":    {"col": "speed_kmh", "label": "Speed (km/h)", "cmap": "viridis",
                 "scale": 1.0, "vmin": None, "vmax": None},
    "throttle": {"col": "throttle",  "label": "Throttle %",   "cmap": "YlGn",
                 "scale": 100.0, "vmin": 0, "vmax": 100},
    "brake":    {"col": "brake",     "label": "Brake %",      "cmap": "OrRd",
                 "scale": 100.0, "vmin": 0, "vmax": 100},
    "gear":     {"col": "gear",      "label": "Gear",         "cmap": "turbo",
                 "scale": 1.0, "vmin": 1, "vmax": 8},
}


class CircuitNotFound(Exception):
    """Raised when no layout file exists for a requested circuit."""


# --------------------------------------------------------------------------
# Data loading (layout files are pure data, kept out of this module)
# --------------------------------------------------------------------------
def circuits_dir():
    """Where layout files live. Defaults to the packaged `circuits/` folder;
    override with WORKSPACE_CIRCUITS_DIR (mirrors WORKSPACE_SESSIONS_DIR)."""
    env = os.environ.get("WORKSPACE_CIRCUITS_DIR")
    return Path(env) if env else Path(__file__).parent / "circuits"


def available_circuits():
    """Sorted (id, name) of every layout file found, for listings and errors."""
    out = []
    for p in sorted(circuits_dir().glob("*.json")):
        if p.stem == "TEMPLATE":
            continue
        try:
            d = json.loads(p.read_text())
            out.append((d.get("id", p.stem), d.get("name", p.stem)))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_circuit(circuit):
    """Load a layout by id (looked up in the circuits dir) or by file path.
    Raises CircuitNotFound with a helpful message if it doesn't exist."""
    p = Path(circuit)
    if p.suffix == ".json" and p.exists():
        return json.loads(p.read_text())
    path = circuits_dir() / f"{circuit}.json"
    if not path.exists():
        have = ", ".join(cid for cid, _ in available_circuits()) or "(none)"
        raise CircuitNotFound(
            f"No layout file for '{circuit}'. Available: {have}. "
            f"Add one at {circuits_dir()}/{circuit}.json (see TEMPLATE.json).")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def normalize(points, width, height, pad=40):
    """Scale a list of X/Y points to fit a width x height box, preserving aspect
    ratio and centring the result. Y is flipped so the map renders upright
    regardless of the source coordinate system."""
    pts = np.asarray(points, dtype=float)
    mn = pts.min(axis=0)
    span = pts.max(axis=0) - mn
    span[span == 0] = 1.0
    s = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    ox = pad + (width - 2 * pad - span[0] * s) / 2
    oy = pad + (height - 2 * pad - span[1] * s) / 2
    out = (pts - mn) * s + [ox, oy]
    out[:, 1] = height - out[:, 1]            # flip for upright display
    return out


def _channel_along_line(circuit, lap_df, channel):
    """Interpolate a telemetry channel onto every centreline point by arc length.
    Returns values already multiplied by the channel's display scale."""
    line = np.asarray(circuit["centerline"], dtype=float)
    seg = np.sqrt((np.diff(np.vstack([line, line[:1]]), axis=0) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])[:-1]
    frac = cum / cum[-1] if cum[-1] else cum
    track_len = float(lap_df["lap_distance_m"].max()) or 1.0
    dist = frac * track_len
    d = lap_df["lap_distance_m"].values
    vals = np.interp(dist, d, lap_df[CHANNELS[channel]["col"]].values)
    return vals * CHANNELS[channel]["scale"]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _outward(P, idx, off):
    """A point `off` pixels out from centreline[idx], on the side away from the
    track centroid — used to place labels clear of the line."""
    n = len(P)
    cx, cy = P[:, 0].mean(), P[:, 1].mean()
    a, b = P[(idx - 3) % n], P[(idx + 3) % n]
    nx, ny = -(b[1] - a[1]), (b[0] - a[0])
    m = np.hypot(nx, ny) or 1.0
    nx, ny = nx / m, ny / m
    if nx * (P[idx][0] - cx) + ny * (P[idx][1] - cy) < 0:
        nx, ny = -nx, -ny
    return P[idx][0] + nx * off, P[idx][1] + ny * off


def _annotate(ax, P, circuit):
    """Draw the start/finish line, numbered corners, and optional overlays."""
    n = len(P)
    labels = circuit.get("labels", {})

    sf = labels.get("start_finish", 0) % n
    a, b = P[(sf - 1) % n], P[(sf + 1) % n]
    tx, ty = (b - a) / (np.hypot(*(b - a)) or 1)
    s0 = P[sf]
    ax.plot([s0[0] + ty * 13, s0[0] - ty * 13],
            [s0[1] - tx * 13, s0[1] + tx * 13], color="black", lw=3, zorder=5)
    lx, ly = _outward(P, sf, 24)
    ax.annotate("S/F", (lx, ly), ha="center", va="center", fontsize=9,
                fontweight="bold", color="black")

    for c in labels.get("corners", []):
        lx, ly = _outward(P, c["index"] % n, 22)
        ax.annotate(str(c["n"]), (lx, ly), ha="center", va="center", fontsize=8,
                    color="0.25", bbox=dict(boxstyle="circle,pad=0.15", fc="white",
                                            ec="0.6", lw=0.6))

    ov = circuit.get("overlays", {})
    if "speed_trap" in ov:
        ax.plot(*P[ov["speed_trap"] % n], "o", color="#e83ab4", ms=8, zorder=6)
    for i in ov.get("drs_detection", []):
        ax.plot(*P[i % n], "o", color="#39d353", ms=8, zorder=6)


def render_map(circuit, lap_df=None, channel=None, lap_label=None, size=(1000, 640)):
    """Render a circuit map to a matplotlib Figure.

    circuit  : layout dict from load_circuit().
    lap_df   : a single lap's telemetry (already pass-selected, e.g. from
               f1_analyze.get_lap). If None, only the outline is drawn.
    channel  : one of CHANNELS ('speed'/'throttle'/'brake'/'gear') to colour by.
    """
    width, height = size
    P = normalize(circuit["centerline"], width, height)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.set_aspect("equal")
    ax.axis("off")

    closed = np.vstack([P, P[:1]])
    segs = np.stack([closed[:-1], closed[1:]], axis=1)

    coloured = lap_df is not None and channel in CHANNELS
    if coloured:
        vals = _channel_along_line(circuit, lap_df, channel)
        spec = CHANNELS[channel]
        vmin = spec["vmin"] if spec["vmin"] is not None else float(vals.min())
        vmax = spec["vmax"] if spec["vmax"] is not None else float(vals.max())
        # grey base so the full outline reads even where a segment is faint
        ax.add_collection(LineCollection(segs, colors="0.85", linewidths=9,
                                         capstyle="round"))
        lc = LineCollection(segs, cmap=spec["cmap"],
                            norm=plt.Normalize(vmin, vmax), capstyle="round")
        lc.set_array(vals)
        lc.set_linewidth(6)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02).set_label(spec["label"])
    else:
        ax.add_collection(LineCollection(segs, colors="0.2", linewidths=6,
                                         capstyle="round"))

    _annotate(ax, P, circuit)

    # flag emoji omitted: matplotlib's default font has no regional-indicator glyphs
    title = f"{circuit['name']} — {circuit.get('country','')}"
    sub = f"{circuit.get('length_m','?')} m · {circuit.get('turns','?')} turns"
    if coloured:
        sub += f" · {lap_label or 'lap'} coloured by {channel}"
    else:
        sub += " · layout"
    ax.set_title(f"{title}\n{sub}", fontsize=13, pad=16)

    # LineCollections don't drive autoscale, and corner/S-F labels sit outside the
    # line — expand the view so nothing near the edges gets clipped.
    ax.autoscale_view()
    ax.margins(0.07)
    fig.tight_layout()
    return fig


def render_png(circuit, lap_df=None, channel=None, lap_label=None):
    """Render to PNG bytes (used by the web UI / anything non-interactive)."""
    fig = render_map(circuit, lap_df, channel, lap_label)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=RENDER_DPI, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _reference_lap(csv_path, lap_num):
    """Load a single lap's telemetry from a session CSV, picking the fastest
    complete lap if lap_num is None. Exits with a clear message on any problem."""
    from workspace import f1_analyze       # lazy: outline rendering needs no pandas
    df = f1_analyze.load(csv_path)
    if df.empty:
        sys.exit("No lap data in that CSV.")
    if lap_num is None:
        complete = f1_analyze.lap_summary(df)
        complete = complete[complete["complete"]]
        if complete.empty:
            sys.exit("No complete lap in that CSV; pass --lap to force one.")
        lap_num = int(complete["lap"].iloc[0])
    lap_df = f1_analyze.get_lap(df, lap_num)
    if len(lap_df) < 2:
        sys.exit(f"Lap {lap_num} has too little data to colour the map.")
    return lap_df, f"Lap {lap_num}"


def main():
    ap = argparse.ArgumentParser(
        description="Render an F1 circuit map from coordinate data, optionally "
                    "colour-coded by a reference lap's telemetry.")
    ap.add_argument("circuit", nargs="?", help="circuit id (e.g. mc-1929) or a .json path")
    ap.add_argument("--list", action="store_true", help="list available circuits and exit")
    ap.add_argument("--csv", help="session CSV with a reference lap to colour by")
    ap.add_argument("--lap", type=int, help="lap number (default: fastest complete lap)")
    ap.add_argument("--channel", choices=list(CHANNELS), default="speed",
                    help="telemetry channel to colour by (default: speed)")
    ap.add_argument("--out", help="output PNG path (default: <id>_map.png)")
    args = ap.parse_args()

    if args.list:
        rows = available_circuits()
        if not rows:
            print(f"No circuits found in {circuits_dir()}")
            return
        print("Available circuits:")
        for cid, name in rows:
            print(f"  {cid:10}  {name}")
        return

    if not args.circuit:
        ap.error("give a circuit id (or --list). e.g. f1-map mc-1929")

    try:
        circuit = load_circuit(args.circuit)
    except CircuitNotFound as e:
        sys.exit(str(e))

    lap_df = lap_label = None
    if args.csv:
        lap_df, lap_label = _reference_lap(args.csv, args.lap)

    fig = render_map(circuit, lap_df, args.channel if lap_df is not None else None,
                     lap_label)
    out = args.out or f"{circuit['id']}_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    how = f"coloured by {args.channel} ({lap_label})" if lap_df is not None else "outline"
    print(f"Circuit map saved: {out}  ({circuit['name']}, {how})")


if __name__ == "__main__":
    main()
