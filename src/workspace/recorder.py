"""Background-thread wrapper around f1_logger.record_session so the web UI can
start / stop recording and poll live status without blocking."""

import threading
import time
from pathlib import Path

from . import f1_logger


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
        self._lock = threading.Lock()
        self._status = _idle_status()

    def start(self, port=f1_logger.PORT):
        with self._lock:
            if self._status["recording"]:
                return False
            self._stop = threading.Event()
            self._status = _idle_status()
            self._status.update(recording=True, message="Waiting for packets…",
                                started_at=time.time(), port=port)
        self._thread = threading.Thread(target=self._run, args=(port,), daemon=True)
        self._thread.start()
        return True

    def _run(self, port):
        try:
            f1_logger.record_session(
                self.out_dir, self._stop, port=port,
                status=self._status, lock=self._lock, log=lambda *_: None,
            )
        except Exception as exc:  # surface bind errors etc. to the UI
            with self._lock:
                self._status["message"] = f"Error: {exc}"
        finally:
            with self._lock:
                self._status["recording"] = False
                if not self._status["message"].startswith("Error"):
                    self._status["message"] = "Stopped"

    def stop(self):
        with self._lock:
            stop = self._stop
        if stop is not None:
            stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return self.status()

    def status(self):
        with self._lock:
            snap = dict(self._status)
        if snap["started_at"]:
            snap["elapsed_s"] = round(time.time() - snap["started_at"], 1)
        else:
            snap["elapsed_s"] = 0
        return snap
