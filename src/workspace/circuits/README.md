# Circuit layout files

Each `*.json` file here is one circuit's **layout** — a centreline traced as X/Y
points, plus labels for the start/finish line and the corners. The data lives
entirely in these files; the renderer (`workspace/circuit_map.py`) only knows how
to read and draw them.

**To add a circuit, add a file. You never edit code.** Copy `TEMPLATE.json` to
`<id>.json`, fill it in, and it shows up automatically:

```bash
f1-map --list          # your new circuit appears here
f1-map <id>            # draw the outline
```

Files are discovered by globbing this directory. `TEMPLATE.json` is skipped.
Override the directory with the `WORKSPACE_CIRCUITS_DIR` environment variable to
keep your own layouts outside the package.

## File format

```jsonc
{
  "id":        "mc-1929",        // unique id = <country-code>-<first-GP-year>; also the file stem
  "name":      "Monaco",
  "country":   "Monaco",
  "flag":      "🇲🇨",            // stored for the web UI; not drawn in the PNG (font has no flag glyphs)
  "length_m":  3337,             // real track length, used to map telemetry onto the trace
  "turns":     19,

  "centerline": [[x, y], ...],   // the traced path, in ANY units (see Coordinates below)

  "labels": {
    "start_finish": 0,           // index into centerline where the S/F line sits (usually 0)
    "corners": [                 // one entry per numbered corner
      { "n": 1, "index": 15 },   // corner 1 is at centerline[15]
      { "n": 2, "index": 37 }
    ],
    "sectors": {                 // optional: indices where each sector ends
      "sector1_end": 64,
      "sector2_end": 132
    }
  },

  "overlays": {                  // optional markers, drawn as dots if present
    "speed_trap": 107,           // centerline index of the speed trap
    "drs_detection": [153]       // centerline index/indices of DRS detection points
  }
}
```

Only `id`, `name`, `length_m`, `centerline`, and `labels` are required. `country`,
`flag`, `turns`, `labels.sectors`, and `overlays` are optional.

### Coordinates

Units and origin are arbitrary — the renderer **normalises and scales** the
centreline to fit the display while preserving aspect ratio, and flips the Y axis
so the map is upright regardless of the source. So a path traced in image pixels
(Y increasing downward) works directly; so does one in metres. Only the *shape*
and the *order* of the points matter.

- Points are an **ordered loop in racing direction**. Don't repeat the first
  point at the end — the renderer closes the loop for you.
- `start_finish` and every corner/overlay field is an **index into
  `centerline`** (0-based), not an X/Y coordinate. More points = a smoother
  curve; ~120–260 points is typical.

## Tracing a new circuit

The included layouts were traced from the official F1 circuit maps by extracting
the road centreline from the image (mask the track, average its inner/outer edge
contours, resample to an ordered loop), then snapping the turn-number labels to
the nearest centreline index. Tracing from the image beats drawing by hand.
Corner and start/finish indices come straight out of that process.

## Rendering

```bash
f1-map <id>                                       # outline only
f1-map <id> --csv session.csv --lap 5 --channel speed
f1-map <id> --csv session.csv --channel throttle  # uses the fastest complete lap
```

With a reference lap, the trace is colour-coded by `speed`, `throttle`, `brake`,
or `gear`. Without lap data, the outline is drawn alone. Ask for a circuit with
no file and `f1-map` tells you so and lists what's available — it never crashes.
