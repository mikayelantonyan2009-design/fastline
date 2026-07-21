"""Background-thread wrapper around f1_logger.record_session so the web UI can
start / stop recording and poll live status without blocking."""

import threading
import time
from pathlib import Path

from . import f1_logger
from .f1_logger import LiveStatus


def _idle_status():
    return {
        "recording": False,
        "session_file": None,
        "current_lap": 0,
        "last_lap_ms": 0,
        "last_completed_lap": 0,
        "last_completed_ms": 0,
        "rows": 0,
        "packets": 0,
        "ignored": 0,
        "message": "Idle",
        "started_at": None,
        "port": f1_logger.PORT,
    }


class Recorder:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._thread = None
        self._stop = None
        self._track = None
        self._year = None
        self._lock = threading.Lock()
        self._status = LiveStatus(_idle_status())

    def start(self, port=f1_logger.PORT, track=None, year=None):
        with self._lock:
            if self._status.get("recording"):
                return False
            self._track = track
            self._year = year
            self._status = LiveStatus({**_idle_status(), "recording": True,
                                       "message": "Waiting for packets…",
                                       "started_at": time.time(), "port": port})
            self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(port,), daemon=True)
        self._thread.start()
        return True

    def _write_track_meta(self):
        """Persist which circuit and car-year this session was recorded at, as
        sidecars next to the CSV, so the viewer can pick the right map later."""
        fname = self._status.get("session_file")
        if not fname:
            return
        base = self.out_dir / fname
        try:
            if self._track:
                base.with_suffix(".track").write_text(self._track)
            if self._year:
                base.with_suffix(".year").write_text(str(self._year))
        except OSError:
            pass

    def _run(self, port):
        try:
            f1_logger.record_session(self.out_dir, self._stop, port=port,
                                     status=self._status, log=lambda *_: None)
        except Exception as exc:  # surface bind errors etc. to the UI
            self._status.update(message=f"Error: {exc}")
        finally:
            self._write_track_meta()
            msg = self._status.get("message", "")
            self._status.update(recording=False,
                                message=msg if msg.startswith("Error") else "Stopped")

    def stop(self):
        with self._lock:
            stop = self._stop
        if stop is not None:
            stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return self.status()

    def status(self):
        snap = self._status.snapshot()
        started = snap.get("started_at")
        snap["elapsed_s"] = round(time.time() - started, 1) if started else 0
        return snap
