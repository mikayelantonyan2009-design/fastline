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
    UDP Send Rate     : 60Hz   (more resolution = better analysis)
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

CSV_HEADER = ["session_time", "frame", "lap", "lap_distance_m", "lap_time_ms",
              "speed_kmh", "throttle", "brake", "gear", "rpm", "drs"]


# ---------------------------------------------------------------- header
def parse_header(data):
    """PacketHeader (2025 format), little-endian, packed."""
    (packet_format, game_year, major, minor, packet_version, packet_id,
     session_uid, session_time, frame_id, overall_frame,
     player_idx, secondary_idx) = struct.unpack_from("<HBBBBBQfIIBB", data, 0)
    return {
        "format": packet_format, "id": packet_id, "session_uid": session_uid,
        "session_time": session_time, "frame": frame_id, "player": player_idx,
    }


def _update(status, lock, **kw):
    """Thread-safe merge into the shared status dict (no-op if status is None)."""
    if status is None:
        return
    if lock is not None:
        with lock:
            status.update(kw)
    else:
        status.update(kw)


# ---------------------------------------------------------- core loop
def record_session(out_dir, stop_event, *, port=PORT, timeout=1.0,
                   status=None, lock=None, log=print):
    """Listen for telemetry until stop_event is set. Returns the list of CSV
    paths written (one per in-game session seen).

    out_dir     : directory to write CSV files into.
    stop_event  : threading.Event; the loop exits promptly once it is set.
    status      : optional mutable dict updated live (used by the web UI).
    lock        : optional threading.Lock guarding `status`.
    log         : callable for human-readable progress lines.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(timeout)
    log(f"Listening on UDP port {port} ... start driving in F1 25.")

    files_written = []
    csv_file = None
    writer = None
    locked_uid = None          # first session we see; everything else is ignored
    warned_format = False
    warned_stride = False
    warned_other = False
    rows = 0
    packets = 0
    ignored = 0

    lap_state = {"lap": 0, "lap_distance": 0.0, "lap_time_ms": 0}
    last_lap_num = 0

    try:
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                _update(status, lock, message="Waiting for packets…")
                continue
            if len(data) < HEADER_SIZE:
                continue

            packets += 1
            h = parse_header(data)

            if h["format"] != 2025 and not warned_format:
                log(f"WARNING: packet format is {h['format']}, expected 2025. "
                    "Set 'UDP Format' to 2025 in the game's telemetry settings.")
                warned_format = True

            # Lock onto the first session we see and open exactly one file.
            if locked_uid is None:
                locked_uid = h["session_uid"]
                fname = time.strftime("f1_session_%Y%m%d_%H%M%S.csv")
                path = out_dir / fname
                csv_file = open(path, "w", newline="")
                writer = csv.writer(csv_file)
                writer.writerow(CSV_HEADER)
                files_written.append(path)
                log(f"New session -> logging to {path}")
                _update(status, lock, session_file=fname, message="Recording…")

            # Ignore packets from any other session/source (e.g. a second device
            # broadcasting to the same port), so one recording -> one clean CSV.
            elif h["session_uid"] != locked_uid:
                ignored += 1
                if not warned_other:
                    log("Ignoring telemetry from another session/source on this "
                        "port; locked to the first session seen.")
                    warned_other = True
                if ignored % 30 == 0:
                    _update(status, lock, ignored=ignored)
                continue

            # ---- Packet 2: Lap Data (lap number, distance, running lap time)
            if h["id"] == 2:
                stride = (len(data) - HEADER_SIZE - LAPDATA_TRAILING) / NUM_CARS
                if stride != int(stride):
                    if not warned_stride:
                        log(f"WARNING: unexpected LapData size ({len(data)} bytes). "
                            "Game update may have changed the format.")
                        warned_stride = True
                    continue
                base = HEADER_SIZE + int(stride) * h["player"]
                last_lap_ms  = struct.unpack_from("<I", data, base + 0)[0]
                cur_lap_ms   = struct.unpack_from("<I", data, base + 4)[0]
                lap_distance = struct.unpack_from("<f", data, base + 20)[0]
                cur_lap_num  = struct.unpack_from("<B", data, base + 33)[0]

                lap_state = {"lap": cur_lap_num, "lap_distance": lap_distance,
                             "lap_time_ms": cur_lap_ms}
                _update(status, lock, current_lap=cur_lap_num,
                        last_lap_ms=cur_lap_ms, packets=packets)

                if cur_lap_num != last_lap_num and last_lap_num != 0 and last_lap_ms > 0:
                    m, s = divmod(last_lap_ms / 1000.0, 60)
                    log(f"  Lap {last_lap_num} complete: {int(m)}:{s:06.3f}")
                    _update(status, lock, last_completed_lap=last_lap_num,
                            last_completed_ms=last_lap_ms)
                last_lap_num = cur_lap_num

            # ---- Packet 6: Car Telemetry (speed, pedals, gear, RPM)
            elif h["id"] == 6 and writer:
                stride = 60  # CarTelemetryData size, stable in the 2025 format
                base = HEADER_SIZE + stride * h["player"]
                (speed, throttle, steer, brake, clutch, gear, rpm, drs) = \
                    struct.unpack_from("<HfffBbHB", data, base)
                writer.writerow([f"{h['session_time']:.3f}", h["frame"],
                                 lap_state["lap"],
                                 f"{lap_state['lap_distance']:.1f}",
                                 lap_state["lap_time_ms"],
                                 speed, f"{throttle:.3f}", f"{brake:.3f}",
                                 gear, rpm, drs])
                rows += 1
                if rows % 30 == 0:
                    _update(status, lock, rows=rows, packets=packets)
    finally:
        _update(status, lock, rows=rows, packets=packets, ignored=ignored)
        if csv_file:
            csv_file.close()
            log("CSV saved.")
        sock.close()

    return files_written


# ---------------------------------------------------------- CLI entry
def main():
    stop = threading.Event()
    try:
        record_session(Path.cwd(), stop)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped. Run f1-analyze on the CSV.")


if __name__ == "__main__":
    main()
