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

- Detects a workpiece's actual outline (not just a bounding box) via background
  subtraction against a saved reference photo of the empty bed.
- Aligns an uploaded **SVG** (vector design) to the detected outline: centered, rotated
  to match, and clipped to fit - exported ready to open in LightBurn.
- Aligns an uploaded **photo**: background removed automatically, rotated to match the
  workpiece's orientation, exported as a plain PNG (with the target position reported),
  since raster laser software burns images axis-aligned with no rotation of its own.
- Optional direct GRBL connection for jogging/calibration convenience (never for running
  burn jobs - that stays in LightBurn/LaserGRBL).
- Optional RF-DETR-based detection as a trained-model alternative to the classical
  background-subtraction method, once enough samples are collected through the app's own
  data-collection page.

This deliberately does **not** generate G-code or run burn jobs itself - it exports a
ready-to-run file, and you open that in LightBurn or LaserGRBL to actually fire the
laser. Rebuilding a whole G-code engine (power curves, raster DPI, safety limits, all of
it) wasn't worth it when LightBurn already does that well. See `context.md` for the full
reasoning behind that split, along with every other architecture decision and bug fix
made along the way.

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

## Safety

This controls a real laser and a real motorized machine. The direct GRBL connection
(used only for jogging during calibration) forces the laser off whenever it's active,
and never sends a fire command under any circumstance - but it does physically move the
machine, including toward its limit switches. Watch the machine while jogging, don't
leave it running unattended, and treat this like any other piece of code driving
hardware: read it before you run it, and don't trust it blindly. Use at your own risk.

## License

MIT - see `LICENSE`.

---

Made by [saleh-sabti](https://github.com/saleh-sabti)
