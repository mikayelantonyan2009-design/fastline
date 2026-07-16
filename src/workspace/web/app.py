"""Flask web UI: start/stop a recording session, browse sessions, and view the
engineer-overlay chart for any two laps inline in the browser."""

import base64
import os
import re
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless rendering, before pyplot is imported

from flask import Flask, jsonify, request, send_from_directory

from .. import f1_analyze, f1_sim
from ..f1_logger import PORT
from ..recorder import Recorder

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _clean_colors(colors):
    """Keep only valid #rrggbb values; ignore anything else so a bad payload
    just falls back to the default line colors."""
    out = {}
    if isinstance(colors, dict):
        for key in ("color1", "color2"):
            val = colors.get(key)
            if isinstance(val, str) and _HEX.match(val):
                out[key] = val
    return out


def sessions_dir():
    return Path(os.environ.get("WORKSPACE_SESSIONS_DIR",
                               Path.cwd() / "sessions")).resolve()


def _safe_csv(name):
    """Resolve a session filename to a path inside sessions_dir (no traversal)."""
    d = sessions_dir()
    p = (d / name).resolve()
    if p.parent != d or p.suffix != ".csv" or not p.is_file():
        return None
    return p


def create_app():
    app = Flask(__name__)
    here = Path(__file__).parent
    recorder = Recorder(sessions_dir())

    @app.get("/")
    def index():
        return send_from_directory(here / "templates", "index.html")

    # ---- recording control -------------------------------------------------
    @app.post("/api/record/start")
    def record_start():
        port = int(request.json.get("port", PORT)) if request.is_json else PORT
        ok = recorder.start(port=port)
        if not ok:
            return jsonify(error="Already recording"), 409
        return jsonify(recorder.status())

    @app.post("/api/record/stop")
    def record_stop():
        return jsonify(recorder.stop())

    @app.get("/api/record/status")
    def record_status():
        return jsonify(recorder.status())

    @app.post("/api/record/demo")
    def record_demo():
        """Fire the synthetic emitter at the recorder's port (no PS5 needed)."""
        st = recorder.status()
        if not st["recording"]:
            return jsonify(error="Start recording first"), 409
        laps = int(request.json.get("laps", 3)) if request.is_json else 3
        threading.Thread(
            target=f1_sim.simulate,
            kwargs={"port": st["port"], "laps": laps, "hz": 120},
            daemon=True,
        ).start()
        return jsonify(ok=True, laps=laps)

    # ---- sessions & analysis ----------------------------------------------
    @app.get("/api/sessions")
    def list_sessions():
        d = sessions_dir()
        out = []
        for p in sorted(d.glob("f1_session_*.csv"), reverse=True):
            stat = p.stat()
            out.append({"name": p.name, "size": stat.st_size,
                        "modified": int(stat.st_mtime)})
        return jsonify(out)

    @app.get("/api/sessions/<name>/laps")
    def session_laps(name):
        p = _safe_csv(name)
        if p is None:
            return jsonify(error="No such session"), 404
        df = f1_analyze.load(p)
        if df.empty:
            return jsonify(laps=[])
        summary = f1_analyze.lap_summary(df)
        laps = [
            {"lap": int(r.lap), "time_s": round(float(r.time_s), 3),
             "samples": int(r.samples), "complete": bool(r.complete)}
            for r in summary.itertuples()
        ]
        return jsonify(laps=laps)

    @app.post("/api/sessions/<name>/analyze")
    def analyze(name):
        p = _safe_csv(name)
        if p is None:
            return jsonify(error="No such session"), 404
        df = f1_analyze.load(p)
        if df.empty:
            return jsonify(error="No lap data in this session"), 400
        summary = f1_analyze.lap_summary(df)
        body = request.json or {}
        laps = body.get("laps")
        colors = _clean_colors(body.get("colors"))
        try:
            lap1, lap2 = f1_analyze.pick_laps(summary, laps)
            png, info = f1_analyze.render_png(df, lap1, lap2, **colors)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(
            lap1=lap1, lap2=lap2,
            net_delta=round(info["net_delta"], 3),
            image="data:image/png;base64," + base64.b64encode(png).decode(),
        )

    return app


def main():
    app = create_app()
    port = int(os.environ.get("WORKSPACE_WEB_PORT", 5000))
    print(f"F1 telemetry web UI -> http://127.0.0.1:{port}")
    print(f"Sessions dir: {sessions_dir()}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
