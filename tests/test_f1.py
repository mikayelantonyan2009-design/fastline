"""End-to-end-ish tests that need no PS5: they drive the synthetic UDP emitter
through the recorder, then analyze the resulting CSV."""

import socket
import time

import matplotlib
matplotlib.use("Agg")

from workspace import f1_analyze, f1_sim
from workspace.recorder import Recorder


def _free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_record_then_analyze(tmp_path):
    port = _free_udp_port()
    rec = Recorder(tmp_path)
    assert rec.start(port=port) is True
    assert rec.start(port=port) is False          # already recording

    # feed synthetic laps to the recorder
    f1_sim.simulate(dest=("127.0.0.1", port), laps=3, hz=240, warmup=0.3)
    time.sleep(0.3)                               # let the loop drain
    status = rec.stop()
    assert status["recording"] is False
    assert status["rows"] > 0

    csvs = list(tmp_path.glob("f1_session_*.csv"))
    assert len(csvs) == 1

    df = f1_analyze.load(csvs[0])
    assert not df.empty
    summary = f1_analyze.lap_summary(df)
    assert (summary["complete"]).sum() >= 2       # at least two full laps

    lap1, lap2 = f1_analyze.pick_laps(summary)
    png, info = f1_analyze.render_png(df, lap1, lap2)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"          # valid PNG signature
    assert "net_delta" in info


def test_session_lock_ignores_other_source(tmp_path):
    """Two sources on the same port (e.g. a real PS5 + the demo) must NOT thrash
    the recorder into many files — it locks to the first session and ignores the
    rest, producing a single analyzable CSV."""
    import threading

    port = _free_udp_port()
    rec = Recorder(tmp_path)
    rec.start(port=port)

    # a second, competing session streaming to the same port
    intruder = threading.Thread(
        target=f1_sim.simulate,
        kwargs={"dest": ("127.0.0.1", port), "laps": 3, "hz": 120,
                "warmup": 0.35, "session_uid": 0xDEADBEEF},
        daemon=True,
    )
    intruder.start()
    f1_sim.simulate(dest=("127.0.0.1", port), laps=3, hz=240, warmup=0.3)  # "real"
    intruder.join(timeout=10)
    time.sleep(0.3)
    status = rec.stop()

    # exactly one file (the bug produced dozens), and it stays analyzable
    csvs = list(tmp_path.glob("f1_session_*.csv"))
    assert len(csvs) == 1
    assert status["ignored"] > 0                  # the other source was dropped
    df = f1_analyze.load(csvs[0])
    summary = f1_analyze.lap_summary(df)
    assert (summary["complete"]).sum() >= 2


def test_get_lap_drops_rewind_overlap():
    """A mid-lap flashback (distance jumps backward < a full lap) must not leave
    two overlapping drives at the same distance — get_lap keeps forward progress
    only, so the plotted distance is strictly increasing (no filled band)."""
    import numpy as np
    import pandas as pd

    # drive 0..700, flash back to 0, then complete 0..1000
    dist = list(range(0, 701, 50)) + list(range(0, 1001, 50))
    n = len(dist)
    df = pd.DataFrame({
        "lap": [1] * n,
        "lap_distance_m": dist,
        "lap_time_ms": list(range(0, n * 100, 100)),
        "speed_kmh": [200] * n, "throttle": [1.0] * n, "brake": [0.0] * n,
        "gear": [7] * n, "rpm": [10000] * n,
        "session_time": [i * 0.1 for i in range(n)],
        "frame": list(range(n)), "drs": [0] * n,
    })
    d = f1_analyze.get_lap(df, 1)["lap_distance_m"].values
    assert np.all(np.diff(d) > 0)     # strictly increasing, no overlap/fill
    assert d.max() == 1000            # keeps the full completed pass


def test_lap_trace_endpoint(tmp_path, monkeypatch):
    """The track-view endpoint returns aligned per-sample arrays for a lap."""
    monkeypatch.setenv("WORKSPACE_SESSIONS_DIR", str(tmp_path))
    port = _free_udp_port()
    rec = Recorder(tmp_path)
    rec.start(port=port)
    f1_sim.simulate(dest=("127.0.0.1", port), laps=3, hz=240, warmup=0.3)
    time.sleep(0.3)
    rec.stop()

    from workspace.web.app import create_app
    client = create_app().test_client()
    name = client.get("/api/sessions").get_json()[0]["name"]
    laps = client.get(f"/api/sessions/{name}/laps").get_json()["laps"]
    lap = next(lp["lap"] for lp in laps if lp["complete"])

    tr = client.get(f"/api/sessions/{name}/lap/{lap}/trace").get_json()
    n = len(tr["dist"])
    assert n > 10
    keys = ("speed", "throttle", "brake", "gear", "rpm", "time")
    assert all(len(tr[k]) == n for k in keys)
    assert tr["time"][-1] >= tr["time"][0]     # lap time runs forward
    assert tr["track_len_m"] > 1000
    assert max(tr["speed"]) > 200
    assert client.get(f"/api/sessions/{name}/lap/999/trace").status_code == 400
    assert client.get("/api/sessions/nope.csv/lap/1/trace").status_code == 404


def test_pick_laps_needs_two_complete():
    import pandas as pd
    summary = pd.DataFrame([{"lap": 1, "time_s": 90.0, "complete": True},
                            {"lap": 2, "time_s": 91.0, "complete": False}])
    try:
        f1_analyze.pick_laps(summary)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_session_remembers_track(tmp_path, monkeypatch):
    """Recording with a chosen circuit writes a .track sidecar, and the sessions
    listing surfaces it so the viewer can reopen with the right map."""
    monkeypatch.setenv("WORKSPACE_SESSIONS_DIR", str(tmp_path))
    port = _free_udp_port()
    rec = Recorder(tmp_path)
    assert rec.start(port=port, track="ae-2009") is True
    f1_sim.simulate(dest=("127.0.0.1", port), laps=3, hz=240, warmup=0.3)
    time.sleep(0.3)
    rec.stop()

    csvs = list(tmp_path.glob("f1_session_*.csv"))
    assert len(csvs) == 1
    meta = csvs[0].with_suffix(".track")
    assert meta.is_file() and meta.read_text() == "ae-2009"

    from workspace.web.app import create_app
    client = create_app().test_client()
    row = next(s for s in client.get("/api/sessions").get_json()
               if s["name"] == csvs[0].name)
    assert row["track"] == "ae-2009"


def test_record_endpoint_saves_track(tmp_path, monkeypatch):
    """Full path: /api/record/start with a track -> record -> stop -> the session
    lists that track (this is exactly what the track picker drives)."""
    monkeypatch.setenv("WORKSPACE_SESSIONS_DIR", str(tmp_path))
    from workspace.web.app import create_app
    client = create_app().test_client()
    port = _free_udp_port()
    assert client.post("/api/record/start",
                       json={"port": port, "track": "ae-2009"}).status_code == 200
    f1_sim.simulate(dest=("127.0.0.1", port), laps=3, hz=240, warmup=0.3)
    time.sleep(0.3)
    client.post("/api/record/stop")

    csvs = list(tmp_path.glob("f1_session_*.csv"))
    assert len(csvs) == 1
    assert csvs[0].with_suffix(".track").read_text() == "ae-2009"
    row = next(s for s in client.get("/api/sessions").get_json()
               if s["name"] == csvs[0].name)
    assert row["track"] == "ae-2009"


def test_record_start_ignores_bad_track(tmp_path, monkeypatch):
    """A malformed track id is dropped, not persisted — recording still starts."""
    monkeypatch.setenv("WORKSPACE_SESSIONS_DIR", str(tmp_path))
    from workspace.web.app import create_app
    client = create_app().test_client()
    r = client.post("/api/record/start",
                    json={"port": _free_udp_port(), "track": "../evil"})
    assert r.status_code == 200
    client.post("/api/record/stop")
    assert not list(tmp_path.glob("*.track"))


def test_session_rename(tmp_path, monkeypatch):
    """A friendly name can be saved for a session, shows in the listing, and an
    empty name clears it; unknown sessions 404."""
    monkeypatch.setenv("WORKSPACE_SESSIONS_DIR", str(tmp_path))
    port = _free_udp_port()
    rec = Recorder(tmp_path)
    rec.start(port=port)
    f1_sim.simulate(dest=("127.0.0.1", port), laps=3, hz=240, warmup=0.3)
    time.sleep(0.3)
    rec.stop()

    from workspace.web.app import create_app
    client = create_app().test_client()
    name = client.get("/api/sessions").get_json()[0]["name"]

    r = client.post(f"/api/sessions/{name}/name", json={"name": "Quali run"})
    assert r.status_code == 200 and r.get_json()["label"] == "Quali run"
    row = next(s for s in client.get("/api/sessions").get_json() if s["name"] == name)
    assert row["label"] == "Quali run"

    client.post(f"/api/sessions/{name}/name", json={"name": ""})   # clear
    row = next(s for s in client.get("/api/sessions").get_json() if s["name"] == name)
    assert row["label"] is None
    assert client.post("/api/sessions/nope.csv/name",
                       json={"name": "x"}).status_code == 404


def test_session_remembers_year(tmp_path, monkeypatch):
    """The chosen car-year is saved with the session and surfaced in the list;
    an unsupported year is dropped."""
    monkeypatch.setenv("WORKSPACE_SESSIONS_DIR", str(tmp_path))
    port = _free_udp_port()
    rec = Recorder(tmp_path)
    assert rec.start(port=port, track="ae-2009", year=2026) is True
    f1_sim.simulate(dest=("127.0.0.1", port), laps=3, hz=240, warmup=0.3)
    time.sleep(0.3)
    rec.stop()

    csvs = list(tmp_path.glob("f1_session_*.csv"))
    assert len(csvs) == 1
    assert csvs[0].with_suffix(".year").read_text() == "2026"

    from workspace.web.app import create_app, _clean_year
    assert _clean_year("2025") == 2025 and _clean_year(1999) is None
    client = create_app().test_client()
    row = next(s for s in client.get("/api/sessions").get_json()
               if s["name"] == csvs[0].name)
    assert row["year"] == 2026 and row["track"] == "ae-2009"
