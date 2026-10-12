"""Tests for the coordinate-driven circuit map renderer."""
import socket
import time

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from workspace import circuit_map, f1_analyze, f1_sim
from workspace.recorder import Recorder

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_all_seeded_circuits_are_valid():
    ids = [cid for cid, _ in circuit_map.available_circuits()]
    assert "mc-1929" in ids and "br-1940" in ids
    assert "TEMPLATE" not in ids                      # template is excluded
    for cid in ids:
        c = circuit_map.load_circuit(cid)
        n = len(c["centerline"])
        assert n > 10
        assert all(len(p) == 2 for p in c["centerline"])
        labels = c["labels"]
        assert 0 <= labels["start_finish"] < n
        for corner in labels["corners"]:
            assert 0 <= corner["index"] < n           # every corner indexes a real point
        assert len(labels["corners"]) == c["turns"]   # matches declared turn count


def test_normalize_preserves_aspect_ratio_and_fits():
    # a wide rectangle; scale must be uniform on both axes and stay inside the box
    pts = [[0, 0], [100, 0], [100, 20], [0, 20]]
    out = circuit_map.normalize(pts, width=200, height=200, pad=10)
    assert out[:, 0].min() >= 10 - 1e-6 and out[:, 0].max() <= 190 + 1e-6
    assert out[:, 1].min() >= 10 - 1e-6 and out[:, 1].max() <= 190 + 1e-6
    sx = (out[:, 0].max() - out[:, 0].min()) / 100
    sy = (out[:, 1].max() - out[:, 1].min()) / 20
    assert sx == pytest.approx(sy)                     # aspect ratio preserved


def test_render_outline_only_png():
    c = circuit_map.load_circuit("mc-1929")
    png = circuit_map.render_png(c)                    # no lap data -> outline alone
    assert png[:8] == PNG_SIG


def test_missing_circuit_raises_with_listing():
    with pytest.raises(circuit_map.CircuitNotFound) as e:
        circuit_map.load_circuit("no-such-track")
    msg = str(e.value)
    assert "no-such-track" in msg and "Available" in msg


def test_render_coloured_by_each_channel(tmp_path):
    port = _free_udp_port()
    rec = Recorder(tmp_path)
    rec.start(port=port, track="mc-1929")
    f1_sim.simulate(dest=("127.0.0.1", port), laps=3, hz=240, warmup=0.3)
    time.sleep(0.3)
    rec.stop()

    csv = next(tmp_path.glob("f1_session_*.csv"))
    df = f1_analyze.load(csv)
    summary = f1_analyze.lap_summary(df)
    lap = int(summary[summary["complete"]]["lap"].iloc[0])
    lap_df = f1_analyze.get_lap(df, lap)

    c = circuit_map.load_circuit("mc-1929")
    for channel in ("speed", "throttle", "brake", "gear"):
        png = circuit_map.render_png(c, lap_df, channel, f"Lap {lap}")
        assert png[:8] == PNG_SIG


def test_channel_mapping_covers_full_lap():
    """Every centreline point gets a telemetry value interpolated by arc length."""
    import pandas as pd
    c = circuit_map.load_circuit("mc-1929")
    n = len(c["centerline"])
    df = pd.DataFrame({
        "lap_distance_m": np.linspace(0, 3337, 200),
        "speed_kmh": np.linspace(80, 300, 200),
        "throttle": [1.0] * 200, "brake": [0.0] * 200, "gear": [5] * 200,
    })
    vals = circuit_map._channel_along_line(c, df, "speed")
    assert len(vals) == n
    assert vals.min() >= 80 - 1 and vals.max() <= 300 + 1
