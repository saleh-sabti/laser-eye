# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `context.md` in this same directory for the full decision log (why things are built
the way they are, what's been verified against real hardware, and what's still open) -
read that before making architectural changes here.

## What this is

A local Flask app that gives a Comgrow laser engraver (GRBL firmware, no sensors) camera
vision: an overhead webcam detects the real outline/position of an irregular workpiece on
the bed, and the app auto-aligns an uploaded design (SVG or photo) to it. It does **not**
run burn jobs itself - it exports a ready-to-run file, and you open that in LightBurn or
LaserGRBL to actually fire the laser. A separate, narrowly-scoped direct GRBL connection
exists only for jogging during calibration - never for burning.

## Running it

The virtualenv lives at **`C:\venvs\laser`**, not inside this project folder - it was
deliberately moved out of `Desktop\laser` because it's ~4GB (PyTorch/CUDA included) and
sat inside OneDrive-synced Desktop, which caused real sync-lock problems (a folder
rename failed for several minutes with "device or resource busy" while OneDrive
uploaded it). Keep it there; don't recreate `.venv` inside the project.

```
C:\venvs\laser\Scripts\python app.py
```
Serves on `http://localhost:5000` (and the LAN IP, for phone access - bind is `0.0.0.0`).
It's a Flask dev server (`debug=True`) tied to the terminal process; it does not persist
across reboots or stay up indefinitely, and needs restarting after a crash or long idle
period. No test suite or build/lint step exists - verification during development has
been done via ad-hoc scripts against `laser_align/` functions directly (see the note in
`context.md` about isolating any such script from the real `calibration_data/` files).

Installing dependencies (into `C:\venvs\laser`, not a project-local `.venv`):
`requirements.txt` covers everything except the CUDA-specific PyTorch stack, which needs
its own index and matched versions:
```
C:\venvs\laser\Scripts\python -m pip install -r requirements.txt
C:\venvs\laser\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
C:\venvs\laser\Scripts\python -m pip install rfdetr supervision
```
The RF-DETR/torch stack is only needed if using the "rfdetr" detection method in
Settings; "classical" (the default) needs nothing beyond the base `requirements.txt`.

## Architecture

**`laser_align/`** is the pipeline; **`app.py`** is a thin Flask layer over it (routes
call straight into these modules, no service/business-logic layer in between);
**`templates/`** renders it.

Request flow for the core feature (Design & Export page): camera frame ->
`detection.detect_object()` (or `rfdetr_detect.detect_object()` if that method is
selected and trained) -> a `Detection` (outline + centroid + angle, in both pixel and
machine-mm coordinates) -> `align.align_and_clip()` for SVGs or `photo_align.align_photo()`
for images -> `export.write_svg()` / `photo_align.save_png()`.

- **`config.py`** - all persistent settings live in one `calibration_data/settings.json`,
  loaded/merged via `load_settings()`/`save_settings()`. `calibration_data/` (sibling of
  `laser_align/`) holds every piece of persistent state: settings, the saved homography +
  calibration points, the empty-bed reference photo, and (if used) the RF-DETR training
  dataset and model checkpoint. Treat this directory as real user data, not scratch space.
- **`camera.py`** - wraps `cv2.VideoCapture`. Auto-reconnects if the device wasn't
  plugged in yet at startup, and auto-detects+recovers from a "stale" connection (frames
  gone flat black with near-zero variance - happens after long continuous use). All
  public methods are guarded by a `threading.RLock` - Flask runs `threaded=True` and
  multiple routes hit the same shared `Camera` object concurrently (the live stream polls
  continuously); without the lock this caused silent native-level process crashes.
- **`calibration.py`** - fits a `cv2.findHomography` from user-supplied (pixel, machine-mm)
  point pairs; `pixels_to_mm()` is the actual pixel->real-world conversion everything else
  depends on. `find_duplicate_conflict()` guards against the easy mistake of clicking a
  new point but forgetting to update the mm fields (browser autocomplete tends to do this
  for you unless the calling form disables it).
- **`detection.py`** - background-subtraction against a saved reference frame, diffed
  across all three color channels (not grayscale - see `context.md` for why that matters),
  Otsu-thresholded within the optional bed ROI (`settings["bed_roi_px"]`, camera pixels -
  Otsu must run on the ROI crop itself, not a zero-padded full frame, or its threshold
  skews badly), then the traced contour is filled solid before use (a real workpiece has
  no holes, so gaps in the diff signal shouldn't produce any in the detected shape).
  Candidate contours are walked largest-first and must clear both a minimum mean-diff
  magnitude and a minimum solidity to be accepted - real wood-vs-mat contrast and shape
  coherence are both far stronger than residual sensor noise, which otherwise passes a
  pure area filter. See `context.md` for the measured numbers behind these thresholds.
- **`rfdetr_detect.py`** - same `Detection` output shape as `detection.py`, backed by a
  fine-tuned RF-DETR segmentation model instead. Raises `ModelNotTrainedError` until a
  checkpoint exists at `calibration_data/rfdetr_model/checkpoint_best_ema.pth`; callers
  (see `app.py`'s `_detect_current`) catch that and fall back to the classical detector.
- **`dataset.py`** - turns classical-detector output into COCO-format training samples
  (image + auto-generated segmentation polygon) for fine-tuning RF-DETR, so building a
  dataset needs no manual annotation. No training script exists yet.
- **`align.py`** / **`export.py`** - SVG path handling via `svgelements`, clipping via
  `shapely`. Two non-obvious conversions live here: `MM_PER_PX` (svgelements resolves all
  geometry into 96-CSS-px space regardless of source units) and the left-to-right operand
  order for `svgelements.Matrix` multiplication (opposite of the usual column-vector
  convention) - get either wrong and placement silently comes out wrong, not erroring.
- **`photo_align.py`** - background removal via `rembg` (local/offline), then rotation is
  baked directly into the output pixels (not left for the destination software to apply),
  because raster laser software burns images axis-aligned with no rotation of its own.
- **`grbl.py`** - direct serial connection to GRBL, used only by the Calibration page's
  jog panel. Two things matter if touching this file: (1) every public method is wrapped
  in a `threading.RLock` for the same concurrency reason as `camera.py` - GRBL status
  polling happens continuously while jog commands can land at the same time; (2) raw
  `pyserial` I/O is funneled through `_write()`/`_readline()`, which convert
  `serial.SerialException` (e.g. the connection physically dropping) into this project's
  own `GrblError` - anything that calls `self._ser` directly instead bypasses that and
  will crash the Flask request uncaught. `GrblAlarmError` (a limit switch trip) is
  handled distinctly from other errors throughout, since recovering from it needs a
  specific user action (`unlock()`/`home()`), not just retrying.
- **`app.py`** - module-level globals (`_camera`, `_grbl`, `_pending_points`,
  `_last_export_path`) hold state across requests, since this is a single-user local tool
  with no session/database layer. `_grbl_call()` is the shared error-handling wrapper for
  every GRBL route - it drops the global connection automatically if `Grbl.is_broken`
  gets set, so a dead serial link surfaces as "not connected" instead of repeating the
  same failing call on every future poll.
