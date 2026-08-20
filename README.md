# laser-eye

My Comgrow laser engraver has no idea what's on its bed. No camera, no sensors, nothing.
So every time I wanted to engrave an oddly-shaped piece of wood, I'd end up manually
jogging the head around and eyeballing the alignment. This fixes that: point a webcam at
the bed, and it figures out where your workpiece actually is - the real outline, not
just a rough box - and lines your design up automatically.

## How it works

A camera sits fixed above the bed. Once it's calibrated, it can turn any pixel in a
photo into a real machine coordinate. Finding a workpiece is just a matter of comparing
the current photo against a saved "empty bed" reference - wherever they differ is where
something got placed.

From there:
- Upload an **SVG** and it gets centered, rotated to match the piece's angle, and
  clipped to fit inside the outline, ready to open in LightBurn.
- Upload a **photo** and the background gets stripped automatically, then it's rotated
  to match the piece before export (as a plain PNG, since raster laser software doesn't
  rotate jobs on its own - the target position gets printed out so you know where to
  jog it).

There's also an optional direct GRBL connection for jogging around during calibration,
and an optional RF-DETR model you can train on your own collected samples if the
classical detection ever isn't robust enough for your setup.

One thing this does **not** do: run the actual burn. That stays in LightBurn or
LaserGRBL - this just gets the design lined up and hands off a ready-to-run file.
Rebuilding a whole G-code engine (power curves, raster DPI, safety limits, all of it)
wasn't worth it when LightBurn already does that well.

## Hardware you'll need

- A GRBL-based engraver (built against a Comgrow machine)
- A webcam mounted somewhere fixed above the bed - it can't move once calibrated
- A plain, matte mat under the work area helps a lot with detection accuracy, and
  consistent lighting matters more than you'd think

## Getting it running

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

Then open `http://localhost:5000`. First time through: set your camera and bed size in
Settings, then head to Calibration - capture an empty-bed reference photo, mark out the
actual bed area, and add a few reference points. After that it's just Design & Export
for every job.

RF-DETR needs its own CUDA-matched PyTorch install if you want to use it - see
`CLAUDE.md` for the exact commands.

## More detail

- `CLAUDE.md` - how the code is actually organized, module by module
- `context.md` - the long version: every decision made building this and why, plus the
  real bugs hit along the way

---

Made by [saleh-sabti](https://github.com/saleh-sabti)
