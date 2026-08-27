# Manual placement + auto-fit + background-removal improvements — design proposal

Status: proposal, not yet built. Written 2026-08-27, refined same day after user feedback.

Requests from the user:

1. **Placement editor** on a virtual view of the real bed, with **two modes**:
   - **Auto-fit** — after the piece is detected, the app places the design inside the
     piece edges automatically (centered, rotated to the piece, scaled to fit).
   - **Manual** — drag / rotate / tilt / scale / position by hand, starting from the
     auto-fit placement.
   Either way: export, and the burn lands in the same physical spot you see on screen.
2. **Better background removal** — confirmed symptom: **ragged / fuzzy edges**. Also:
   the current output is "just a cutout" — it tells LaserGRBL a size but not *where* to
   put it, which is the whole point that's missing.
3. Optionally use **Gemini ("nano banana")** for background removal — user's current
   manual workflow is: give it the design → "remove background, white background, black
   crisp edges."
4. Housekeeping: push recent changes to GitHub, update `context.md`, refresh the
   README screenshots to the redesigned UI.

Note: **no real end-to-end burn has been done yet** — the user is still waiting on the
placement/positioning story to be trustworthy before cutting anything.

---

## Part 1 — Manual placement mode

### The core idea: place on a *rectified* (top-down) view of the bed, not the raw camera

The calibration homography `H` maps camera pixels → machine mm. It is a **perspective**
transform: a rectangle in mm becomes a skewed quadrilateral in the camera image. If we
overlaid the design directly on the raw camera feed, its real-world size would appear to
change as you dragged it around the frame — unusable for precise placement.

Instead, warp the camera frame into a straight top-down view where 1 screen pixel = a
fixed fraction of a mm:

```
k = 3                      # display px per mm (2–4 is plenty)
S = [[k,0,0],[0,k,0],[0,0,1]]
rectified = cv2.warpPerspective(frame, S @ H, (bed_w_mm*k, bed_h_mm*k))
```

Now display-pixel ↔ mm is a single scalar in both directions, and **every bit of
placement math in the browser is plain affine** (translate / rotate / scale) — no
perspective, no homography in the drag path.

**Free bonus:** this is the first screen that makes a bad calibration *visible*. If the
rectified bed looks sheared, bowed, or smeared, that's the homography, not the placement
tool. (`MIN_SPREAD_FRACTION` is a floor — it does not prove the calibration is good
across the whole bed.) We draw `Detection.contour_mm` straight onto this view (it's
already in mm, no re-warping) so you can see the piece outline and snap to it.

### Workflow / UX

New page: **🖐️ Place & Export** (or fold into Design & Export as a mode toggle).

1. **Upload design** (SVG or photo). Photo goes through background removal first
   (Part 2), and you land on the touch-up step before placement.
2. **Capture bed** — press once. The editor runs on a *frozen snapshot*, not the live
   MJPEG stream. You cannot drag onto moving video, and the placement has to correspond
   to the exact frame the piece was sitting in. A visible **"Recapture"** button if the
   piece moves. (This also keeps the camera lock out of the drag path.)
3. **Rectified bed** fills the canvas, with the detected piece outline drawn on top.
4. **Pick a mode:**
   - **Auto-fit** (default when a piece is detected) — design is placed centered on the
     piece centroid, rotated to `angle_deg`, and scaled to the largest size that fits
     inside the outline minus the safety margin. This is essentially today's
     `align_and_clip` / `align_photo` behaviour, now shown *on the bed view* so you can
     see it before committing. Buttons: "fit inside edges", "match piece rotation",
     size as % of max fit.
   - **Manual** — starts from the auto-fit placement, then:
     - Drag to move. Corner handles to scale (Shift = keep aspect). Rotation handle for
       tilt. Arrow keys nudge 0.5 mm; Shift+arrows 5 mm.
     - **Numeric panel** beside the canvas: X, Y (center or corner — toggle), width,
       height, angle. "Exactly" often means *typing 12.5 mm*, not dragging.
     - **Snap buttons:** snap to piece centroid, snap angle to `angle_deg`, snap to bed
       center, "re-fit to edges".
   - Manual mode still works with **no detection** — it needs only *calibration*. No
     outline just means no snapping / no auto-fit.
5. **Clip toggle** (both modes): "keep design inside the piece outline (−1.5 mm)".
   Default on for photos, off for SVG. When on, the parts that *would* be clipped are
   drawn ghosted rather than just disappearing, so nothing vanishes silently.
6. Live readout: design size in mm, distance from piece edges, warning if any part is
   outside the bed.
7. **Verify placement without burning** — **"Jog head to design center"** button. Uses
   the existing `grbl.jog_to(x, y)` (already wrapped by `_grbl_call`, already jog-only
   scope). The head physically moves to where you dropped the design; you look at the
   bed and confirm. This is the honest end-to-end check that app-mm == real-bed-mm.
8. **Export** — SVG in absolute mm (see below), or the raster path (see below).

### Client does the dragging, server does the export

- On upload, the server flattens the SVG to normalized polylines (reuse `align.py`'s
  `_flatten_path`) and sends them to the browser as JSON. The browser transforms them
  client-side via SVG/CSS transforms at 60 fps.
- The browser POSTs back only the final transform: `(tx_mm, ty_mm, rotation_deg,
  scale)` plus the clip toggle.
- The parsed design is held in a module global (consistent with this project's
  deliberate no-session design), not re-parsed per request. Note the current code
  deletes the SVG temp file in a `finally` immediately after aligning — the manual flow
  needs to keep the parsed geometry around.

### Export math — avoid the known svgelements traps

`context.md` documents two silent-wrong-placement bugs in the `svgelements.Matrix`
path: the `MM_PER_PX` unit resolution, and `Matrix.__mul__` composing left-to-right.
For manual placement, **don't rebuild a Matrix chain.** Apply the user's transform with
Shapely (`affine_transform` / `rotate` / `scale`) directly on the already-flattened
polylines, then write them out with the existing `export.write_svg`.

Add one guard: after building the export geometry, assert its centroid equals the
requested `(tx, ty)` within a small tolerance. If the browser and server disagree on any
convention (Y direction, degrees vs radians, rotation origin), this fails loudly
instead of shipping a wrong burn.

### BLOCKER (do this first): the Y-axis convention is unverified

**This is already visible as a symptom.** The user reports the red target crosshair on
the preview "doesn't show the actual place" any more — "it became bad." That marker is
`draw_overlay`'s centroid cross; if where it *says* the design center is doesn't match
where the head actually goes, every export is wrong by that same error. Two likely
causes, check in order:

1. **Stale / bad calibration.** `context.md` records the calibration has been wiped
   several times and the user still needs to redo it with real values. The rectified
   bed view (below) makes this obvious — if the warped bed looks sheared or the outline
   doesn't sit on the real piece, redo calibration before anything else.
2. **Y-axis flip.** `export.write_svg` takes a `flip_y` argument, but `app.py` never
   passes it — hardcoded `False`, not configurable. The LightBurn origin/axis match has
   never been verified against a real burn.

For *auto-align* a wrong Y flip was survivable — the design still lands roughly on the
detected piece. For *manual placement* a Y flip puts the design somewhere completely
different. **We should not ship a precision placement tool on an unresolved axis
convention.**

Resolve it cheaply, no laser fired:

1. Add `flip_y` to `settings.json` (default `False`) + a Settings toggle, and actually
   pass it through every `write_svg` call.
2. Add the **"Jog head to a clicked point"** check: click anywhere on the rectified bed
   → the head physically jogs there. Confirms the homography maps app-mm to the real
   bed correctly (and directly diagnoses the "red cross is wrong" complaint).
3. Do **one** test SVG import into LightBurn to confirm LightBurn agrees with our mm
   (position + orientation). Flip the setting if it lands mirrored.

Jogging answers "is my homography right?"; the one import answers "does LightBurn agree
with my mm?" Both are one-time.

### Raster (photo) exact placement — the "where do I put it" problem

This is the user's loudest complaint: the current export is *just a cutout*. LaserGRBL
asks for a size but there's no way to tell it *where* on the bed to burn — so the tool
saves you nothing over doing it by hand.

**Why LaserGRBL is different from LightBurn:** LaserGRBL has no absolute-coordinate
workspace. It engraves an image starting from wherever the laser head currently is
(the origin), growing in one direction. So "exact placement" for LaserGRBL means:
*put the head at the right spot first, then engrave.*

**The fix — a Position Card shown with every raster export:**

```
┌─ Where to put this ───────────────────────────────┐
│  Image size:      84.0 × 61.5 mm                   │
│  Engraving origin: LaserGRBL "top-left"            │
│  Jog the head to:  X 128.4   Y 302.7  mm           │
│                   [ Jog head there now ]           │
│                                                   │
│  Then in LaserGRBL:                                │
│   1. Set current position as origin                │
│   2. Load aligned_design.png, set size 84.0×61.5   │
│   3. Engrave                                       │
└───────────────────────────────────────────────────┘
```

- The **"Jog head there now"** button uses the app's existing GRBL jog connection
  (`grbl.jog_to`) — the same one the calibration page uses. After it moves, the user
  disconnects the app, opens LaserGRBL, sets origin = current position, and the image
  burns exactly where they placed it in the editor.
- The reported X/Y is the corner matching LaserGRBL's configured engraving direction
  (usually top-left) — computed from the placement rectangle, not the centroid, because
  that's the anchor raster software actually uses.
- Rotation stays baked into the pixels as today (raster software can't rotate).

**LightBurn users** get the SVG path instead: wrap the image in an SVG as a base64
`<image>` at absolute mm coords, same as the vector export — one unified code path.
**Verify once:** import one such SVG into LightBurn, confirm position + scale survive.

---

## Part 2 — Background removal

Confirmed symptom: **ragged / fuzzy edges.** For a laser this is the worst failure mode
— semi-transparent edge pixels dither into a ring of speckle when burned.

### 2a. Fix the edges (local, offline — priority order)

1. **Binarize the alpha** at an adjustable threshold — every pixel becomes fully opaque
   or fully transparent. Hard edges only. This alone removes the "fuzzy" look.
2. **Keep only the largest connected component** (`cv2.connectedComponentsWithStats`),
   close small holes. Removes floating background specks / halo.
3. **Optional smoothing** of the binary mask contour (small
   `cv2.morphologyEx` open+close, or `approxPolyDP`) so the hard edge is also a *clean*
   edge, not stair-stepped.
4. **Alpha matting** as a toggle (helps hair/thin detail):
   `remove(img, alpha_matting=True, alpha_matting_foreground_threshold=240,
   alpha_matting_background_threshold=10, alpha_matting_erode_size=10)`.
5. **Model swap** — only after checking what's actually installed (`rembg.__version__`
   + available sessions). Alternatives like `isnet-general-use` / `birefnet-general` are
   each a separate large download; the project's premise is offline-after-first-download.
   Pick from what's there or pre-download one deliberately, don't guess.
6. **Manual brush touch-up** — erase / restore, brush size, undo. ~80 lines of JS,
   between upload and placement. Makes quality independent of any model.

### 2b. Processing time / progress indicator

`rembg` on CPU takes anywhere from ~2 s to 30 s+ depending on image size and model, and
right now the page just hangs with no feedback. Fix:

- Run background removal in a **background thread** keyed by a job id (consistent with
  the project's module-global state style).
- Frontend polls `/design/bg/status?job=…` → `{state, elapsed_s, eta_s}`.
- Show a spinner with a **live elapsed counter** and a rough estimate ("usually
  5–15 s at this size"), estimate refined from the last few runs.
- Downscale very large uploads before `remove()` (rembg's model input is small anyway)
  — biggest single speed lever.

### 2c. Optional: Gemini ("nano banana") background removal

The user's current manual workflow is: hand the design to Gemini, prompt "remove
background, white background, black crisp edges." That's actually a great fit for laser
raster — it returns exactly the 1-bit-ready image you want.

Add a **background-removal method** setting, same pattern as `detection_method`:

| method            | needs        | notes                                             |
|-------------------|--------------|---------------------------------------------------|
| `local_rembg`     | nothing      | default; fully offline                            |
| `gemini`          | API key + internet | `GEMINI_API_KEY` in settings; per-image API call; best edges |

- Gemini path: POST the image to the Gemini image API with the user's standard prompt
  (editable in Settings), get back the cleaned image, run it through the same
  binarize + largest-component cleanup as the local path so downstream code doesn't care
  which produced it.
- Clearly marked in the UI as "needs internet" so it's an informed choice — the rest of
  the app stays usable with no connection.

### UX

After upload: **before / after side by side**, progress spinner during processing,
then threshold slider + alpha-matting toggle + method picker + brush.
"Looks good → continue to placement."

---

## Build order

1. **`flip_y` into settings + Settings toggle + pass it through `write_svg`.**
   *(unblocks everything else — the placement editor is not trustworthy until the
   coordinate convention is settled)*
2. **`/bed/rectified.jpg` endpoint + "jog head to clicked point"** — reusable, and
   immediately diagnoses the "red cross is wrong" complaint on its own.
3. **User runs the one-time checks**: redo calibration if the rectified bed looks off;
   jog-to-point to confirm the mapping; one test SVG import into LightBurn for `flip_y`.
4. **Placement editor** — rectified bed canvas, auto-fit mode + manual mode, clip
   toggle, numeric panel, transform round-trip assertion on export.
5. **Raster Position Card** — size + jog-to-corner button + LaserGRBL steps; verify the
   LightBurn embedded-image SVG path separately.
6. **Background removal** — binarize + largest-component + smoothing, then progress
   indicator, then brush, then (optional) Gemini method.
7. **Housekeeping** (below).

---

## Housekeeping

- ✅ Plan + `printables/` + `screenshots/` committed; `context.md` updated; pushed.
- ✅ Neubrutalist white UI merged to `main` (`e204ca3` + merge `34c03ba`); the
  `nervous-lalande-bda051` worktree and branch removed.
- ✅ README media refreshed to the new UI (`ui-dashboard.jpg`, `ui-calibration.jpg`,
  `bed-view-rectified.jpg`); stale old-UI `demo.gif` / `design-export.png` /
  `screen-recording.gif` removed; design-log entries added for the redesign, the
  ArUco-only switch, and Bed View.
- ⬜ User has newer screen recordings to add to the README — waiting on the files.

---

## Confirmed with the user

- Background-removal symptom: **ragged / fuzzy edges** → binarize alpha + smoothing +
  optional alpha matting / model swap / Gemini.
- Start with the **`flip_y` verification** first.
- No real burn done yet — placement trust is the blocker to doing one.
