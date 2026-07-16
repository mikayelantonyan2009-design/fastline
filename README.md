# workspace

F1 25 telemetry toolkit — record a session's UDP telemetry, analyze it, and
compare laps as an "engineer overlay" chart, all from a local web UI.

## Setup

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

## Web UI (recommended)

```bash
f1-web            # -> http://127.0.0.1:5000
```

From the page you can **Start / Stop recording**, watch live lap status, then
pick any two laps of a session and view the comparison chart inline.

No PS5? Start recording, then click **Demo feed** to generate synthetic laps.

Sessions are written to `./sessions/` (override with `WORKSPACE_SESSIONS_DIR`).

## Command line

```bash
f1-logger                         # record to a CSV until Ctrl+C
f1-analyze sessions/f1_session_*.csv          # chart the two fastest laps
f1-analyze sessions/f1_session_*.csv --laps 3 5
f1-sim --laps 3                   # send synthetic telemetry (no PS5 needed)
```

## In-game settings (F1 25 → Telemetry Settings)

    UDP Telemetry : On    UDP Format : 2025    Port : 20777
    UDP IP Address: this machine's IP    Send Rate: 60Hz

## Development

```bash
pytest       # run tests (drives sim -> recorder -> analyzer end to end)
ruff check   # lint
```
