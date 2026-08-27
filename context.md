# Project Context & Decision Log

This file exists so a future Claude session (or you) can pick this project back up
without re-deriving everything from scratch. It covers what was decided, why, and
what's actually been verified vs. what's still open. See `CLAUDE.md` for a more
code-oriented reference (file layout, how to run things).

## The problem this solves

A Comgrow diode laser engraver (GRBL firmware, CH340 USB-serial adapter, confirmed
115200 baud for LightBurn/LaserGRBL use, COM3 / 921600 baud for this app's direct GRBL
connection) has no sensors - it doesn't know what's on its bed or where. Positioning a
design on an irregularly-shaped piece of wood meant manually jogging and eyeballing
alignment every single job. The goal: an overhead camera + this app detects the actual
object outline and auto-positions the design, cutting manual alignment out of the loop.

## Big architecture decisions, and why

**Custom Python/OpenCV vision pipeline, not LightBurn's built-in camera feature.**
LightBurn has a paid camera-alignment module that does something similar, but it's
closed-source and built around aligning to a fixed reference/eyeballing placement, not
detecting an arbitrary irregular object's actual outline. That gap - real outline
detection for organic/irregular shapes, not just a bounding box or a rectangle - is
specifically what this project fills.

**Hybrid run-time: this app only does vision + alignment. LightBurn (or LaserGRBL) still
does the actual burning.** Originally considered having this app generate G-code and
stream it directly, but reimplementing raster/vector engraving (power curves, DPI/line
spacing, overscan, safety limits) from scratch is a huge, already-solved problem. So the
app's output is a ready-to-run file (positioned SVG for vector jobs, a plain PNG for
photo/raster jobs), and you open that in LightBurn/LaserGRBL and press Start yourself.

**Fixed overhead camera, not head-mounted.** Simpler calibration and coordinate math;
the camera must never move after calibration or the pixel-to-mm mapping breaks.

**Calibration reference: the machine's own coordinates, not a physical ruler.** You jog
the laser to a point, read its real X/Y (from LightBurn, or now from this app's own GRBL
jog panel), click the matching spot in a photo, and enter the X/Y. Four-plus such
(pixel, mm) pairs fit a homography (`cv2.findHomography`) that maps any pixel to a real
machine position. Calibration points must be on **fixed features of the bed/machine**
(a screw, a mat corner) - never on the workpiece itself, since the workpiece changes
every job.

**Camera choice: a plain USB webcam for v1, not the spare drone FPV camera.** The drone
camera is almost certainly analog (2-wire signal+ground), which would need a capture
dongle and correct power supply - real uncertainty for no real benefit. A cheap UVC
webcam removes all of that. (Never actually revisited/needed - the webcam has worked.)

## Detection pipeline (`laser_align/detection.py`)

Approach: capture one "empty bed" reference photo, then for every job compare the
current photo against it - wherever they differ is "something got placed there." This
works for arbitrary/irregular shapes with no training data and no assumption about
geometry, unlike edge/color-range detection.

**Real bug found and fixed**: the first version diffed on grayscale only. Wood grain
dark enough to match the dark mat's brightness produced *holes* in the detected mask -
only the bright outer rim of a piece was detected, not the whole shape (confirmed via
diagnostic images against the user's real camera photo). Fixed by diffing across all
three color channels (not collapsing to grayscale first - the wood's color, not just its
brightness, differs from the mat even where luminance is similar), switching to an
Otsu-auto-picked threshold instead of a fixed value, and - critically - filling whatever
outer contour is traced solid before use, since a real workpiece doesn't have holes; any
gaps in the diff signal shouldn't punch a hole in the detected silhouette.

A background-subtraction detector is inherently sensitive to lighting drift between the
reference capture and job time. This is a known, accepted limitation for now (see
RF-DETR section below for the planned mitigation).

**Second real bug found and fixed (2026-08-21)**: outlines were smoothed into a rounded
blob, missing all the workpiece's actual jagged edge detail - traced to the morphological
kernel sizes (5x5 open, 15x15 close) being enormous relative to how small an object can
appear in frame (a real workpiece measured only 43px wide in the 640x480 camera view -
the 15px closing kernel alone was erasing ~35% of the object's own width worth of detail
on each pass). Fixed by shrinking both kernels to 3x3 and the pre-diff blur from 9x9 to
3x3. This was safe to do *because* the fill-solid step (previous fix, above) already
handles interior holes unconditionally regardless of kernel size - the closing kernel's
only remaining job is bridging tiny gaps in the boundary ring itself, which a much
smaller kernel handles fine while preserving real small-scale detail. Verified against
the same real photo across several kernel-size combinations before picking these values.

**Separate contamination bug hit the same day**: the "empty bed" reference photo wasn't
actually empty - it still had a *different*, larger piece of wood sitting on it from
earlier testing. This produced a bizarre two-lobed outline (real object in one lobe,
the phantom old-wood-shaped region in the other, since "wood disappeared" reads as
"different from reference" exactly as much as "wood appeared" does). Not a code bug -
just a reminder to actually clear the bed before capturing the reference frame, and that
a wrong reference photo can produce very misleading-looking detection failures that look
like a precision/algorithm problem at first glance.

**Third real bug found and fixed, same day**: after the precision fix above (smaller
kernels), false detections started appearing along the machine frame/cables and the
belt/lead-screw track even with a genuinely correct empty-bed reference - long squiggly
green outlines tracing those areas instead of (or alongside) the real object. Root cause:
those areas have fine repeating texture and sharp edges that are very sensitive to small
exposure/vibration differences between the reference photo and the current one, and the
smaller kernels (needed for real edge precision) no longer smoothed that noise away the
way the old oversized kernels accidentally did. This is a fundamental tension - no single
global threshold/kernel setting is simultaneously "sensitive enough for a 43px-wide
object" and "insensitive enough to ignore frame/cable texture." Solved properly instead
of by further threshold tuning: added an optional **bed ROI** (`bed_roi_px` in settings,
`[x0,y0,x1,y1]` in camera pixels) that `detection.py` zeroes the diff out for entirely
before Otsu/contours ever run - the machine frame/cables/belt track can never
legitimately have a workpiece on them anyway, so excluding them outright is strictly
correct, not a tuning compromise. Set via a new two-click corner-selection UI on the
Calibration page. Verified against the exact frame that was previously producing false
detections - completely clean with the ROI set, no further tuning needed. **The user
still needs to actually set this via the UI** - it defaults to the whole frame
(`None`) until they do.

**Fourth and fifth bugs, discovered once the user actually set the ROI**: the bed-ROI
fix above was itself buggy in two ways, both found from the user's real screenshots
after use.

1. Zeroing the diff to 0 *outside* the ROI before running Otsu (as originally
   implemented) skews Otsu's histogram badly - the huge zero-padded region pulls the
   chosen threshold way down. Measured on a real empty-bed frame: 13-14 computed on the
   ROI alone vs. 7 after zero-padding first, low enough that ordinary sensor noise
   *inside* the ROI started reading as foreground - producing a squiggly outline that
   traced almost exactly along the ROI's own boundary/the mat-to-frame contrast edge.
   Fixed by running Otsu on the ROI crop directly, then placing that sub-mask into a
   full-frame-sized zero array, instead of thresholding the whole (partially zeroed)
   frame at once.
2. Even after that fix, a smaller residual noise blob (~2400px, ordinary sensor noise
   that happened to cluster together in the middle of the ROI) still passed the area
   filter and got reported as a detected object on a genuinely empty bed - because its
   pixel *area* overlapped with real small objects' area (a real 43px-wide piece was
   ~2880px; the noise blob was ~2379-2656px depending on kernel size, too close to
   separate by area alone). Fixed with two additional, better-separated signals, both
   measured directly against real footage before picking thresholds: **mean diff
   magnitude inside the candidate contour** (noise ~20, a real object ~200+ - wood-vs-mat
   contrast is far stronger than sensor noise, so this threshold sits at 60 with wide
   margin either side) and **solidity** (contour area / convex hull area: noise ~0.31,
   a real object ~0.77-0.79 even for a genuinely jagged irregular piece - threshold set
   at 0.5). A candidate contour must clear both to be accepted; `detection.py` now walks
   candidate contours largest-first and returns the first one that passes both checks,
   rather than assuming the single largest area is automatically the right answer.

Both fixes verified against the user's actual failing frame (empty bed, ROI set) -
correctly reports "no object found" now - and cross-checked against the real jagged wood
piece from the precision fix above to confirm it still detects correctly (solidity 0.789,
comfortably clears the 0.5 floor).

**Sixth bug, found via a new feature (2026-08-22)**: added real-world dimensions
(`Detection.length_mm`/`width_mm`, via `cv2.minAreaRect` on `contour_mm`, shown as a
label baked into `draw_overlay()`) so the app reports how big the detected piece actually
is, not just its outline/position. First test returned length=287,007mm - not a bug in
the new code; the *saved calibration itself* was bad. Its 4 points were pixel-clustered
into a 282x117px band (only 44% width / 24% height of the 640x480 frame, all in one
horizontal strip) - a homography fit from that reproduces those 4 points exactly but
extrapolates wildly for anything outside the strip, which is where the actual workpiece
was. This is the third distinct calibration-data-quality failure this project has hit
(duplicate points, physically-impossible values, now clustering) despite the UI
instructions already saying to spread points around the bed. Added an automatic guard in
`calibration.compute_homography()` this time instead of relying on the instructions
alone: rejects a save if either pixel axis spans less than 35% of the frame dimension
(`MIN_SPREAD_FRACTION`), with a message telling the user to spread points across the
whole bed. Verified it actually catches the exact bad calibration that caused this.
**The user needs to redo calibration again** - the bad one was cleared.

**Camera resolution (2026-08-22)**: the app had been running at 640x480 the entire
project - OpenCV/DSHOW default to that unless a higher resolution is explicitly
requested, and nothing ever did. Checked the actual hardware directly
(`cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)`) and confirmed the user's camera supports up to
1920x1080. That's the single biggest precision lever available - roughly 3x finer
resolution (≈0.63mm/px down to ≈0.21mm/px on a 400mm bed) for free, no algorithm changes
needed. Wired it through as a proper setting (`camera_width`/`camera_height` in
`config.py`, a resolution picker on the Settings page) rather than just hardcoding a
higher default, because raising it invalidates any saved calibration/bed-ROI (both are
pixel coordinates tied to whatever resolution they were captured at) - `settings_page()`
now detects a resolution change and deliberately clears both rather than leaving them
silently misaligned. Also had to make `detection.py`'s object-size floor
resolution-independent (`MIN_OBJECT_AREA_FRACTION`, a fraction of frame area, instead of
the old hardcoded `MIN_OBJECT_AREA_PX` tuned only for 640x480) - an absolute pixel count
does not carry over to a different resolution, since the same physical object covers
proportionally more pixels at a higher resolution. **Not yet re-verified**: the
blur/morphology kernel sizes in `detection.py` (3x3) and the noise-rejection thresholds
(`MIN_MEAN_DIFF_INSIDE`, `MIN_SOLIDITY`) were empirically tuned at 640x480 against real
footage - they're likely still fine at higher resolution (kernels get proportionally
smaller relative to any given object, if anything more precision-preserving) but haven't
been re-validated against real noise footage at 1080p the way the original values were.

**Camera angle/crop question, same day**: the user asked about digitally adjusting
camera angle and cropping, and whether AI was needed for that. Both were already mostly
covered - angle/perspective correction is what the calibration homography *is*, and
cropping is the bed-ROI feature - so the answer was mostly "you already have this."
Classical geometry (an exact homography fit) is the right tool here, not a learned
model - AI is used elsewhere in this project (RF-DETR) for recognizing wood under varying
conditions, a genuinely different, non-geometric problem. The one real gap was a
physically upside-down camera mount, which the math doesn't care about but a human
looking at the live preview would - added `camera_rotate_180` (`Camera.read()` applies
`cv2.rotate(..., ROTATE_180)` right after capture, before anything else ever sees the
frame, so nothing downstream needs to know rotation happened). Deliberately scoped to
180 only, not a general angle: 180 is a plain flip with no dimension change, so it's a
one-line addition; 90/270 would swap width<->height and require touching everywhere that
currently assumes `camera_width` x `camera_height` directly - not done. Like a resolution
change, toggling rotation now clears calibration/bed-ROI too, for the same reason (their
saved pixel coordinates are tied to the orientation they were captured under).

## Alignment & export (`laser_align/align.py`, `export.py`, `photo_align.py`)

Two input types, handled differently:

- **SVG (vector) designs**: centered/rotated onto the detected centroid/orientation,
  clipped to the detected outline (shrunk inward by a safety margin) using Shapely, then
  written out as an SVG with absolute mm coordinates for LightBurn to open directly.
  **Real bug found and fixed**: svgelements resolves all SVG geometry into a canonical
  96-CSS-px document space internally, so a `20mm` design was being read as ~75.6 units,
  wildly larger than the real object - clipping came back empty. Fixed with an explicit
  `MM_PER_PX = 25.4/96` conversion baked into the placement transform. Relatedly,
  **svgelements' `Matrix.__mul__` composes left-to-right** (leftmost operand applied
  first) - the opposite of the standard column-vector convention - which silently
  produced wildly wrong placements until the multiplication order was reversed to match.
- **Photo (PNG/JPEG) designs**: background removed locally via `rembg` (chose this over
  Adobe's cloud background-removal tool specifically to keep the app usable with no
  internet connection per job - a one-time ~1GB model download, then fully offline;
  correcting an earlier, wrong ~175MB estimate). Since raster laser software
  (LaserGRBL, and LightBurn's image mode) burns images axis-aligned with no concept of
  absolute canvas position, the rotation to match the object's orientation is *baked
  directly into the output pixels* instead of relying on the destination software to
  rotate anything - the result is a plain PNG, and the app reports the target center
  X/Y for you to set as the machine's job origin.

**One thing still unverified**: whether LightBurn's workspace origin/axis setup matches
the coordinate system these exports assume. A `flip_y` option exists in `export.py` for
this but hasn't been confirmed necessary or unnecessary against a real burn yet.

## RF-DETR: optional trained-model detection path

The user found roboflow/rf-detr on GitHub and asked whether/how it could help. Assessed
honestly: RF-DETR (now with segmentation checkpoints) could be more robust to lighting
drift than background subtraction, since a trained model recognizes wood directly
instead of diffing against one reference photo - but it ships knowing nothing about this
specific bed/wood/mat, so it's useless without a labeled dataset first.

**Decision**: build both, selectable via a Settings toggle (`detection_method`:
"classical" or "rfdetr"), with automatic fallback to classical if no model is trained
yet. Rather than requiring manual polygon annotation, a **Training Data** page lets you
capture a bed photo *plus the classical detector's own outline* as one auto-labeled COCO
sample with a single click - so building the dataset is just "place a piece, confirm the
outline looks right, click save," repeated 50-200+ times across different pieces/
positions/lighting. This caps the trained model's ceiling at roughly the classical
detector's own accuracy, but should still generalize better across lighting conditions
than diffing against a single fixed reference photo.

**Status**: PyTorch 2.6.0+cu124 and torchvision 0.21.0+cu124 (matched versions - the
first attempt silently pulled a mismatched torchvision from default PyPI, breaking
`torchvision::nms`) plus `rfdetr`/`supervision` are installed and confirmed working
against the user's RTX 2080 Ti (11GB VRAM, confirmed via `nvidia-smi` and `torch.cuda.
is_available()`). `laser_align/rfdetr_detect.py` is a scaffold (`RFDETRSegPreview`-based
inference wrapper) that raises `ModelNotTrainedError` until a real checkpoint exists at
`calibration_data/rfdetr_model/checkpoint_best_ema.pth` - **the actual training script
has not been written yet**; the real training config fields were partially explored
(`TrainConfig` needs `dataset_dir`, `output_dir`, `epochs`, `batch_size`, `dataset_file`
at minimum) but not finalized, and the exact expected COCO directory layout (train/valid
split convention) needs verification against RF-DETR's docs before training will work.
Also unverified: whether `dataset.py`'s COCO output (single `images/` + one
`annotations.json`, no train/valid split) matches what RF-DETR's training pipeline
actually expects as `dataset_dir`.

## Direct GRBL connection (`laser_align/grbl.py`) - jogging only, never burning

Added after the user asked whether the app could control GRBL directly. Scope
deliberately limited to **jogging/calibration convenience** - it does not, and should
not, ever send burn/spindle-on commands; that stays exclusively in LightBurn. Connects
at COM3 / 921600 baud (the user's actual values - not the 115200 used for LightBurn
elsewhere, this machine's GRBL is configured for a higher rate on this connection).

Key design points:
- Jog targets/readouts use **work coordinates** (matching what LightBurn's display
  shows), computed as MPos minus the WCO GRBL reports periodically - not raw machine
  coordinates.
- **Limit switches**: the machine has physical switches on X- and Y-. Hitting one during
  a jog puts GRBL into an ALARM state that blocks *all* further motion, including moving
  away from the switch, until explicitly cleared. Implemented `unlock()` (`$X` - clears
  in place, matches LightBurn's lightning-bolt icon per the user's own description) and
  `home()` (`$H` - full homing cycle, re-zeros properly, matches LightBurn's house icon).
  Also added `set_origin_here()` (`G10 L20 P1 X0 Y0`) and jog-to-arbitrary-XY, covering
  the LightBurn jog-console icons that made sense to replicate. Icons deliberately *not*
  replicated: the keyboard-jog-lock padlock (not applicable, no keyboard binding here),
  the job-outline "frame" function (LightBurn owns "the loaded job," this app doesn't),
  and the pointer/lightbulb (fires the laser at low power - this app **never** fires the
  laser under any circumstance, by design; that capability stays exclusively in
  LightBurn, which already has the machine's safe pointer power configured).
- Only one program can hold the serial port open at a time - LightBurn must be
  disconnected before using this app's jog panel, and vice versa.

**Two serious bugs found and fixed here, both worth remembering as a pattern**:
1. **Race condition** (silent process crashes, no Python traceback): the live camera
   view polls continuously while other requests (snapshots, jogging) can land at the
   same time, all hitting the *same* shared `Camera`/`Grbl` object from different Flask
   request threads (Flask runs `threaded=True`). Neither `camera.py` nor `grbl.py` had
   any synchronization - concurrent reads/writes on the same serial/camera handle from
   different threads caused native-level crashes that killed the whole process with zero
   Python-catchable error. Fixed by adding a `threading.RLock()` around every public
   method in both classes (reentrant because some methods call others internally while
   already holding it). Stress-tested by deliberately firing concurrent requests at both
   - confirmed fixed. **This was very likely the cause of most of the random "server
   just died" incidents throughout this session** - worth checking first if that
   happens again with any *new* hardware-touching code added later.
2. **Uncaught `serial.SerialException`**: when the physical serial connection actually
   drops (e.g. the CH340 adapter glitching from electrical noise right as a limit switch
   triggers - which is what happened to the user), pyserial raises its own
   `SerialException`, which is *not* a subclass of this project's `GrblError` - so it
   wasn't being caught anywhere, and crashed every subsequent request that touched the
   connection (e.g. the position-poll running every 1.5s), making the whole jog panel
   look permanently broken. Fixed by wrapping the two lowest-level serial I/O calls
   (`_write`/`_readline` in `grbl.py`) to catch `serial.SerialException` and convert it
   to `GrblError`, and by having the Flask routes drop the stale global `Grbl` connection
   automatically when this happens (via a `is_broken` flag) so the UI cleanly shows "not
   connected" instead of repeating the same failing call forever.

**Another lesson from this session, unrelated to the two bugs above**: earlier ad-hoc
test/smoke-test scripts for the vision pipeline called `detection.save_reference_frame()`
and `calibration.save_calibration()` directly against the *real* `calibration_data/`
files the live app reads from - which silently overwrote the user's actual completed
calibration and reference photo with synthetic test data partway through the session,
requiring a full redo. Any future throwaway test/verification script that touches
persistent app state (`laser_align/config.py`'s `DATA_DIR` and everything under it)
should either monkeypatch those paths to an isolated temp directory first, or operate on
in-memory objects only (as the later, corrected tests did) - never write through the
real config paths.

## Automatic calibration via ArUco markers (`laser_align/aruco_calibration.py`)

Added 2026-08-22 after the user asked for calibration to stop being manual/repetitive -
this was the plan sketched in the very first session (see the "Big architecture
decisions" section above) finally built. Four `cv2.aruco` markers (`DICT_4X4_50`) get
printed and fixed *permanently* somewhere the camera can always see but a workpiece never
covers (e.g. taped to the bed frame corners). Each marker's real machine position is
registered once, ever, through the same jog-and-enter-X/Y idea as manual points -
`aruco_markers.json` stores `{marker_id: [mm_x, mm_y]}`. After that, "Calibrate now"
detects whichever registered markers are currently visible and feeds them straight into
`calibration.save_calibration()` - the *exact same* function manual calibration uses, so
the automatic spread-check guard (points can't be clustered in one area) applies here for
free, and the result is an ordinary `calibration.json` the rest of the app can't tell was
auto-generated. Verified the full pipeline (generate markers -> detect them in a synthetic
photo -> match against registered positions -> fit homography -> confirm a known point
reprojects correctly) in isolation before wiring into the UI.

This also incidentally fixes the project's oldest standing fragility - "the camera must
never move or the whole calibration breaks" - since re-running takes one click instead of
redoing 4+ manual points, it's now cheap enough to treat as routine maintenance instead of
a disaster to avoid.

**Not yet addressed**: the earlier gantry-occlusion problem (the machine physically
blocking part of the workpiece from camera view) is unrelated to this - markers fixed
outside the working area are unaffected by gantry position, but a workpiece under the
gantry is still invisible regardless of how calibration was obtained. That's still a
"park the gantry off the bed before capturing" procedural fix, not something either
calibration method touches.

**Real bug found in the ArUco flow's first real use (2026-08-22)**: `Detection.length_mm`/
`width_mm` came back "0 x 0mm" on real hardware. Traced it to the homography itself being
degenerate - two of the four calibration points (produced via ArUco auto-calibrate) had
*identical* machine coordinates (370, 0mm) from different pixel locations (a marker's
registered position was mistyped/duplicated), which collapsed an entire row of the fitted
matrix to ~0 - every point mapped to Y=0 regardless of input. The outline itself still
looked correct on screen (`draw_overlay()` draws raw pixel coordinates, unaffected), which
made this look like a cosmetic dimension-label bug at first - it wasn't. With the mapping
broken like this, actual design alignment/export would have been silently wrong too, not
just the label. Root cause: `find_duplicate_conflict()` (the check that would have caught
this) was only ever wired into the manual point-entry route - the newer ArUco registration
route (`/calibration/aruco/register`) had no equivalent check. Fixed by moving the
duplicate-detection logic directly into `calibration.compute_homography()` itself instead
of each UI entry point re-implementing it - protects every current and future calibration
path (manual, ArUco, anything else) uniformly. Verified it catches the exact bad point
pair that caused this.

## UI

Redesigned from bare default-browser-styled HTML into a proper design system on request
("modern UI/UX") - CSS custom-property tokens with automatic dark-mode support via
`prefers-color-scheme`, card-based layout, pill navigation with active-page highlighting,
consistent buttons/badges/inputs. All self-contained (system font stack, no external CDN
calls) so it keeps working with zero internet dependency at the workbench - deliberately
consistent with the project's "runs standalone" philosophy throughout.

Pages: Dashboard, Settings (camera/bed/detection-method/GRBL-port config), Calibration
(reference points + optional direct GRBL jog panel), Live View, Design & Export,
Training Data (RF-DETR dataset collection).

**Calibration page decluttered, reference-frame capture moved to Live View
(2026-08-22).** Two user complaints once ArUco calibration (above) existed alongside the
older manual flow: (1) the ArUco explanation/marker-download grid/registry table/
registration form was always fully expanded, even after setup was long done and the only
thing anyone actually clicks day-to-day is "Calibrate now"; (2) the "empty-bed reference
frame" capture card lived on the Calibration page, disconnected from the live camera feed
you actually need to be looking at to confirm the bed is clear before capturing it.
Fixed both without adding new routes or state: the ArUco setup content is now wrapped in
a native `<details>` element, auto-expanded only while fewer than
`aruco_calibration.MIN_MARKERS_REQUIRED` markers are registered and collapsed by default
once setup is done (the "Calibrate now" button itself stays outside it, always visible);
the reference-frame card moved to `templates/live.html` as-is, with `live_page()` in
`app.py` now passing `has_reference=detection.has_reference_frame()` the same way
`calibration_page()`/`dashboard()` already did. No new logic - pure template
reorganization, verified via the Flask test client (`GET /calibration` and `GET /live`
both 200) before restarting the server.

## Current status (as of this session)

**Working and verified against real hardware:**
- Camera capture (with auto-recovery from both "device not yet plugged in at startup"
  and "stale/dead long-running connection" states)
- Detection (color-based, fixed for wood-grain-vs-mat contrast, verified against a real
  photo of the user's actual wood piece)
- Vector SVG alignment/clipping/export (verified against synthetic test data end-to-end)
- Photo alignment/background-removal/export (verified against synthetic test data
  end-to-end; rembg confirmed working with real background removal on a test image)
- Direct GRBL jog connection, position readout, alarm detection/recovery (verified
  against the user's real machine at COM3/921600, including a real successful 1mm jog
  with position confirmed before/after)
- Thread-safety fixes for both camera and GRBL (stress-tested under real concurrent
  load)

**Not yet done / open items:**
- Calibration needs to be (re)completed by the user with real values read off LightBurn
  or the app's own jog panel - it's been wiped/reset multiple times during this session
  (once by contaminated test data, once deliberately after finding physically-impossible
  saved values like Y=897mm on a 400mm bed)
- The LightBurn origin/coordinate-system match (`flip_y` question in `export.py`) has
  never been verified against a real burn
- No end-to-end real burn has been done yet with an app-exported file
- **RF-DETR training is now ready to run (2026-08-21), just waiting on data.**
  `train_rfdetr.py` at the project root does it: `dataset.export_for_training()` splits
  the flat collected samples into RF-DETR's actual required layout (confirmed against
  its source, `rfdetr/datasets/coco.py`'s `build_roboflow_from_coco` - it's Roboflow's
  convention: `train/`+`valid/` folders, each with its own `_annotations.coco.json`
  alongside the images directly, *not* a single flat `images/`+`annotations.json` like
  `dataset.py` uses for ongoing collection - `dataset_file="roboflow"` is the `TrainConfig`
  value that selects this builder), then fine-tunes `RFDETRSegNano` (smallest variant -
  right call for a necessarily small personal dataset, a bigger model would just overfit
  faster) and writes `checkpoint_best_ema.pth` straight into
  `calibration_data/rfdetr_model/`, exactly where `rfdetr_detect.py` looks for it. Also
  corrected `rfdetr_detect.py`'s model loading while here - the original guess
  (`RFDETRSegPreview(pretrain_weights=...)`) wasn't the real documented API;
  `RFDETR.from_checkpoint(path)` is. Verified the export step's output structure against
  the 5 real samples collected so far, but **actual training has not been run** - only 5
  of the recommended 50-200+ samples exist (guarded by
  `dataset.MIN_SAMPLES_TO_TRAIN = 20`, which `export_for_training()` refuses below).
  Collecting samples across *varied lighting conditions* specifically would directly
  address the lighting-drift fragility noted above, since a trained model doesn't depend
  on matching one fixed reference photo the way classical detection does.

## Working notes for whoever picks this up next

- Project folder: `C:\Users\saleh\OneDrive\Desktop\laser` (renamed from
  `laser-auto-align` on 2026-08-21). **The venv is not inside it** - it lives at
  `C:\venvs\laser`, moved out deliberately because it's ~4GB (PyTorch/CUDA) and sat
  inside OneDrive-synced Desktop, which locked the whole folder mid-sync and blocked a
  simple rename for several minutes ("device or resource busy"). Don't recreate a
  project-local `.venv` - use `C:\venvs\laser\Scripts\python`. See `CLAUDE.md` for exact
  run/install commands.
- The dev server is not persistent - it's tied to a terminal process and needs
  restarting after any reboot, long idle period, or crash. Check `http://localhost:5000/`
  before assuming it's up.
- The user communicates partly via voice-to-text, which sometimes garbles technical
  terms badly (e.g. "Google Adfluence motivation" for a completely unrelated question
  about ArUco/homography calibration points; "water axis" for "Y axis"). When a message
  doesn't parse, it's very often worth interpreting it in light of the immediately
  preceding topic rather than taking it literally.

## Planned next: placement editor + background-removal rework (2026-08-27)

Design proposal written to `docs/manual_placement_and_bg_removal_plan.md` - not built
yet. Summary of what was decided this session:

- **Placement editor** on a *rectified* (top-down, `cv2.warpPerspective(frame, S @ H,
  ...)`) view of the bed, so all in-browser placement math is affine, not perspective.
  Two modes: **auto-fit** (detect piece -> place design inside the outline, centered/
  rotated/scaled - essentially today's `align_and_clip`/`align_photo` shown on the bed
  view) and **manual** (drag/rotate/tilt/scale + numeric mm entry, starting from the
  auto-fit placement). Runs on a *frozen snapshot*, not the MJPEG stream. Manual mode
  needs only calibration, not a detection.
- **BLOCKER identified: the Y-axis / LightBurn coordinate convention is still
  unverified.** `export.write_svg` has a `flip_y` arg that `app.py` never passes
  (hardcoded False, no setting). Survivable for auto-align (design still lands on the
  piece); fatal for manual placement. The user already sees the symptom - the red
  centroid crosshair on the preview "doesn't show the actual place" any more. Likely
  either stale calibration (wiped several times, still needs redoing) or the Y flip.
  Plan: wire `flip_y` into settings + a Settings toggle, add a "jog head to a clicked
  point on the rectified bed" check, then one test SVG import into LightBurn. **This is
  step 1, before the editor is worth building.**
- **Raster/LaserGRBL placement**: the user's core complaint is the photo export is
  "just a cutout" - LaserGRBL asks for a size but there's no way to say *where*.
  LaserGRBL has no absolute-coordinate workspace (engraves from the head's current
  position). Fix = a **Position Card** with every raster export: image size, the corner
  machine X/Y to jog to, a "jog head there now" button (reuses `grbl.jog_to`), then
  LaserGRBL steps (set origin = current position, set size, engrave). LightBurn users
  get an SVG-wrapped base64 `<image>` at absolute mm instead (one unified path, verify
  once).
- **Background removal** - confirmed symptom is **ragged/fuzzy edges** (worst case for a
  laser). Fix, local/offline first: binarize the alpha at an adjustable threshold ->
  keep largest connected component -> smooth the binary contour -> optional alpha
  matting -> optional model swap (check installed rembg sessions first) -> manual
  erase/restore brush. Plus: run `remove()` in a background thread with a poll endpoint
  so the UI shows elapsed time + ETA instead of hanging. Plus optional **Gemini ("nano
  banana") method** (setting, like `detection_method`) - user's manual workflow is
  "remove background, white background, black crisp edges", which is ideal for laser
  raster; needs `GEMINI_API_KEY` + internet, clearly marked, local rembg stays default.
- Still no real end-to-end burn - the user is waiting on the placement/positioning
  story to be trustworthy first.
- README media (`docs/demo.gif`, `design-export.png`, `screen-recording.gif`) still
  shows the pre-redesign bare-HTML UI and needs refreshing against the current UI (a
  hands-on step - needs the app running with the camera).
