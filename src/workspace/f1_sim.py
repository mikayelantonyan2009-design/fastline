"""
Synthetic F1 25 telemetry emitter
=================================
Sends fake but internally-consistent UDP packets (LapData + CarTelemetry) to the
logger, so you can try the recorder / web UI / analyzer WITHOUT a PS5 and the game.

Usage:
    python -m workspace.f1_sim                 # ~3 laps to 127.0.0.1:20777
    python -m workspace.f1_sim --laps 4 --port 20777 --host 127.0.0.1

It's also imported by the test suite and by the web UI's "demo" button.
"""

import argparse
import math
import socket
import struct
import time

from .f1_logger import PORT, HEADER_SIZE, NUM_CARS

# LapData: pick any per-car stride >= 34 that keeps the packet an integer size.
LAP_STRIDE = 57
LAP_PACKET_LEN = HEADER_SIZE + LAP_STRIDE * NUM_CARS + 2   # +2 trailing bytes
TEL_STRIDE = 60
TEL_PACKET_LEN = HEADER_SIZE + TEL_STRIDE * NUM_CARS

TRACK_LEN_M = 5000.0      # synthetic lap length
SESSION_UID = 0x1234ABCD


def _header(packet_id, session_time, frame, uid=SESSION_UID):
    buf = bytearray(HEADER_SIZE)
    struct.pack_into("<HBBBBBQfIIBB", buf, 0,
                     2025, 25, 1, 0, 1, packet_id, uid,
                     session_time, frame, frame, 0, 255)
    return buf


def _lap_packet(session_time, frame, lap_num, lap_ms, last_lap_ms, distance, uid):
    buf = bytearray(LAP_PACKET_LEN)
    buf[0:HEADER_SIZE] = _header(2, session_time, frame, uid)
    base = HEADER_SIZE  # player index 0
    struct.pack_into("<I", buf, base + 0, last_lap_ms)
    struct.pack_into("<I", buf, base + 4, lap_ms)
    struct.pack_into("<f", buf, base + 20, distance)
    struct.pack_into("<B", buf, base + 33, lap_num)
    return buf


def _tel_packet(session_time, frame, speed, throttle, brake, gear, rpm, uid):
    buf = bytearray(TEL_PACKET_LEN)
    buf[0:HEADER_SIZE] = _header(6, session_time, frame, uid)
    base = HEADER_SIZE
    struct.pack_into("<HfffBbHB", buf, base,
                     int(speed), float(throttle), 0.0, float(brake),
                     0, int(gear), int(rpm), 0)
    return buf


def simulate(host="127.0.0.1", port=PORT, laps=3, hz=120, warmup=0.2,
             session_uid=SESSION_UID):
    """Drive `laps` synthetic laps at ~`hz` samples/sec. Returns rows sent.
    `session_uid` lets a test emit a second, competing source."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    time.sleep(warmup)  # give the listener a moment to bind

    dt = 1.0 / hz
    steps_per_lap = int(hz * 8)          # ~8 s per synthetic lap
    session_time = 0.0
    frame = 0
    last_lap_ms = 0
    rows = 0

    for lap in range(1, laps + 1):
        # each lap is a hair different so lap times differ (for "two fastest")
        pace = 1.0 + 0.02 * math.sin(lap)
        for step in range(steps_per_lap + 1):
            frac = step / steps_per_lap
            distance = frac * TRACK_LEN_M
            lap_ms = int(frac * 8000 * pace)
            # a corner mid-lap: speed dips, brake spikes, gear drops
            corner = math.exp(-((frac - 0.5) ** 2) / 0.01)
            speed = 300 - 180 * corner
            throttle = max(0.0, 1.0 - 1.2 * corner)
            brake = min(1.0, 1.4 * corner)
            gear = max(2, min(8, int(speed / 40) + 1))
            rpm = int(6000 + speed * 20)

            sock.sendto(_lap_packet(session_time, frame, lap, lap_ms,
                                    last_lap_ms, distance, session_uid), (host, port))
            sock.sendto(_tel_packet(session_time, frame, speed, throttle,
                                    brake, gear, rpm, session_uid), (host, port))
            rows += 1
            frame += 1
            session_time += dt
            time.sleep(dt)
        last_lap_ms = int(8000 * pace)

    # one extra lap-packet bump so the last full lap registers as "completed"
    sock.sendto(_lap_packet(session_time, frame, laps + 1, 0, last_lap_ms, 0.0,
                            session_uid), (host, port))
    sock.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--hz", type=int, default=120)
    args = ap.parse_args()
    print(f"Emitting {args.laps} synthetic laps to {args.host}:{args.port} …")
    rows = simulate(args.host, args.port, laps=args.laps, hz=args.hz)
    print(f"Done. Sent {rows} telemetry frames.")


if __name__ == "__main__":
    main()
