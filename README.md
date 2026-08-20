# laser-eye

Camera-guided auto-positioning for a GRBL laser engraver that has no sensors of its
own. An overhead webcam detects the real outline and position of whatever workpiece is
on the bed — including irregular, non-rectangular pieces — and automatically aligns an
uploaded design to it, instead of manually jogging and eyeballing placement every job.

## What it does

- Detects a workpiece's actual outline (not just a bounding box) via background
  subtraction against a saved reference photo of the empty bed.
- Aligns an uploaded **SVG** (vector design) to the detected outline: centered, rotated
  to match, and clipped to fit — exported ready to open in LightBurn.
- Aligns an uploaded **photo**: background removed automatically, rotated to match the
  workpiece's orientation, exported as a plain PNG (with the target position reported),
  since raster laser software burns images axis-aligned with no rotation of its own.
- Optional direct GRBL connection for jogging/calibration convenience (never for running
  burn jobs — that stays in LightBurn/LaserGRBL).
- Optional RF-DETR-based detection as a trained-model alternative to the classical
  background-subtraction method, once enough samples are collected through the app's own
  data-collection page.

This project deliberately does **not** generate G-code or run burn jobs itself — it
exports a ready-to-run file, and you open that in LightBurn or LaserGRBL to actually
fire the laser. See `context.md` for the full reasoning behind that split, along with
every other architecture decision and bug fix made along the way.

## Hardware

- A GRBL-based laser engraver (built/tested against a Comgrow machine).
- A fixed, overhead USB webcam pointed at the bed - must not move once calibrated.
- A plain, matte, contrasting mat under the workpiece area, ideally with consistent
  dedicated lighting rather than relying on ambient room light.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The RF-DETR detection method needs its own CUDA-matched PyTorch install; see
`CLAUDE.md` for the exact commands (skip this if you're only using classical detection,
the default).

## Running it

```
.venv\Scripts\python app.py
```

Serves at `http://localhost:5000` (and your LAN IP, for phone access). First-time setup:
Settings (camera/bed size) → Calibration (empty-bed reference photo, bed area, and 4+
reference points) → Design & Export.

## Documentation

- **`CLAUDE.md`** — architecture reference: how each module works and why.
- **`context.md`** — full decision log: every major design choice and real bug found
  during development, with root causes and reasoning, not just what the code does.

## Author

[saleh-sabti](https://github.com/saleh-sabti)
