"""
F1 25 -> CSV telemetry logger
=============================
Catches the game's UDP telemetry packets and writes one CSV per session.

Two ways to use it:
    * CLI:  python -m workspace.f1_logger   (or the `f1-logger` command)
    * Programmatically: call record_session(out_dir, stop_event, status=...)
      from a thread (this is what the web UI does).

Requires: Python 3.8+, standard library only.

In-game settings (Settings > Telemetry Settings):
    UDP Telemetry     : On
    UDP Broadcast Mode: Off
    UDP IP Address    : <this machine's IP, e.g. 192.168.1.34>
    UDP Port          : 20777
    UDP Send Rate     : 60Hz
    UDP Format        : 2025   (IMPORTANT - this script parses the 2025 format)
"""

import socket
import struct
import csv
import time
import threading
from pathlib import Path

PORT = 20777
HEADER_SIZE = 29           # PacketHeader size in the 2025 format
LAPDATA_TRAILING = 2       # trailing bytes after the 22 car blocks in LapData
NUM_CARS = 22
TEL_STRIDE = 60            # CarTelemetryData size, stable in the 2025 format
RECV_TIMEOUT = 1.0         # so the loop can notice stop_event promptly

CSV_HEADER = ["session_time", "frame", "lap", "lap_distance_m", "lap_time_ms",
              "speed_kmh", "throttle", "brake", "gear", "rpm", "drs"]


class LiveStatus:
    """Thread-safe status dict shared between the recording thread and readers."""

    def __init__(self, initial=None):
        self._lock = threading.Lock()
        self._data = dict(initial or {})

    def update(self, **kw):
        with self._lock:
            self._data.update(kw)

    def snapshot(self):
        with self._lock:
            return dict(self._data)

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)


def parse_header(data):
    """PacketHeader (2025 format), little-endian, packed."""
    (packet_format, game_year, major, minor, packet_version, packet_id,
     session_uid, session_time, frame_id, overall_frame,
     player_idx, secondary_idx) = struct.unpack_from("<HBBBBBQfIIBB", data, 0)
    return {
        "format": packet_format, "id": packet_id, "session_uid": session_uid,
        "session_time": session_time, "frame": frame_id, "player": player_idx,
    }


class _Session:
    """One recording: owns the CSV file, the running lap state, and counters.
    Locked to the first session-ID seen so a second source on the same port
    can't fragment the output."""

    def __init__(self, out_dir, status, log):
        self.out_dir = Path(out_dir)
        self.status = status
        self.log = log
        self.uid = None
        self.file = None
        self.writer = None
        self.paths = []
        self.rows = 0
        self.packets = 0
        self.ignored = 0
        self.lap_state = {"lap": 0, "lap_distance": 0.0, "lap_time_ms": 0}
        self.last_lap_num = 0
        self._warned_stride = False
        self._warned_other = False

    def accept(self, h):
        """Return True if this packet belongs to the locked session."""
        if self.uid is None:
            self.uid = h["session_uid"]
            self._open()
            return True
        if h["session_uid"] != self.uid:
            self.ignored += 1
            if not self._warned_other:
                self.log("Ignoring telemetry from another session/source on "
                         "this port; locked to the first session seen.")
                self._warned_other = True
            return False
        return True

    def _open(self):
        fname = time.strftime("f1_session_%Y%m%d_%H%M%S.csv")
        path = self.out_dir / fname
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(CSV_HEADER)
        self.paths.append(path)
        self.log(f"New session -> logging to {path}")
        self.status.update(session_file=fname, message="Recording…")

    def on_lap_data(self, data, h):
        """Packet 2: lap number, distance, running lap time."""
        stride = (len(data) - HEADER_SIZE - LAPDATA_TRAILING) / NUM_CARS
        if stride != int(stride):
            if not self._warned_stride:
                self.log(f"WARNING: unexpected LapData size ({len(data)} bytes). "
                         "Game update may have changed the format.")
                self._warned_stride = True
            return
        base = HEADER_SIZE + int(stride) * h["player"]
        last_lap_ms = struct.unpack_from("<I", data, base + 0)[0]
        cur_lap_ms = struct.unpack_from("<I", data, base + 4)[0]
        distance = struct.unpack_from("<f", data, base + 20)[0]
        cur_lap_num = struct.unpack_from("<B", data, base + 33)[0]

        self.lap_state = {"lap": cur_lap_num, "lap_distance": distance,
                          "lap_time_ms": cur_lap_ms}
        self.status.update(current_lap=cur_lap_num, last_lap_ms=cur_lap_ms,
                           packets=self.packets)
        if cur_lap_num != self.last_lap_num and self.last_lap_num and last_lap_ms > 0:
            m, s = divmod(last_lap_ms / 1000.0, 60)
            self.log(f"  Lap {self.last_lap_num} complete: {int(m)}:{s:06.3f}")
            self.status.update(last_completed_lap=self.last_lap_num,
                               last_completed_ms=last_lap_ms)
        self.last_lap_num = cur_lap_num

    def on_telemetry(self, data, h):
        """Packet 6: speed, pedals, gear, RPM."""
        if self.writer is None:
            return
        base = HEADER_SIZE + TEL_STRIDE * h["player"]
        speed, throttle, steer, brake, clutch, gear, rpm, drs = \
            struct.unpack_from("<HfffBbHB", data, base)
        self.writer.writerow([f"{h['session_time']:.3f}", h["frame"],
                              self.lap_state["lap"],
                              f"{self.lap_state['lap_distance']:.1f}",
                              self.lap_state["lap_time_ms"],
                              speed, f"{throttle:.3f}", f"{brake:.3f}",
                              gear, rpm, drs])
        self.rows += 1
        if self.rows % 30 == 0:
            self.status.update(rows=self.rows, packets=self.packets)

    def close(self):
        if self.file:
            self.file.close()
            self.log("CSV saved.")


def record_session(out_dir, stop_event, *, port=PORT, status=None, log=print):
    """Listen for telemetry until stop_event is set. Returns the list of CSV
    paths written. `status` is an optional LiveStatus updated live for the UI."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if status is None:
        status = LiveStatus()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(RECV_TIMEOUT)
    log(f"Listening on UDP port {port} ... start driving in F1 25.")

    sess = _Session(out_dir, status, log)
    warned_format = False
    try:
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                status.update(message="Waiting for packets…")
                continue
            if len(data) < HEADER_SIZE:
                continue

            sess.packets += 1
            h = parse_header(data)
            if h["format"] != 2025 and not warned_format:
                log(f"WARNING: packet format is {h['format']}, expected 2025. "
                    "Set 'UDP Format' to 2025 in the telemetry settings.")
                warned_format = True

            if not sess.accept(h):
                if sess.ignored % 30 == 0:
                    status.update(ignored=sess.ignored)
                continue
            if h["id"] == 2:
                sess.on_lap_data(data, h)
            elif h["id"] == 6:
                sess.on_telemetry(data, h)
    finally:
        status.update(rows=sess.rows, packets=sess.packets, ignored=sess.ignored)
        sess.close()
        sock.close()
    return sess.paths


def main():
    stop = threading.Event()
    try:
        record_session(Path.cwd(), stop)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped. Run f1-analyze on the CSV.")


if __name__ == "__main__":
    main()
