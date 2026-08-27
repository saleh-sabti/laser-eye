"""Automatic calibration via ArUco markers fixed permanently to the
machine, outside where a workpiece ever sits (e.g. taped/screwed to the
bed frame corners). Each marker's real machine position is registered
once, ever - after that, recalibrating needs no manual clicking at all:
detect the markers in a photo, match them against their known positions,
done. This also makes the whole system self-correcting against small
camera nudges, since it's cheap enough to just re-run any time instead of
trusting one calibration indefinitely.

Reuses calibration.py's compute_homography()/save_calibration() directly,
so the automatic spread-check guard (points can't be clustered into one
area of the frame) applies here too, and the result is the exact same
calibration.json the rest of the app already reads from - nothing else
needs to know whether a calibration came from manual clicking or markers.
"""
import io
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import DATA_DIR, save_settings
from .calibration import CalibrationPoint, CalibrationError, save_calibration

MARKERS_PATH = DATA_DIR / "aruco_markers.json"
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_SIZE_PX = 300
MARKER_MARGIN_PX = 60
MIN_MARKERS_REQUIRED = 4

# Each marker's reference point is ONE of its corners - the one facing away
# from the centre of the bed when the four markers are fixed at the four bed
# corners in id order (0 top-left, 1 top-right, 2 bottom-left, 3 bottom-
# right). Using the outward corner instead of the marker centre makes the
# four registered points span the largest possible rectangle - best
# conditioning for the homography (calibration.compute_homography's spread
# guard), and the points land on the machine's actual reachable extent
# instead of ~2cm inside it. A printed corner is also a far sharper thing to
# jog the laser dot onto by hand than the middle of the marker pattern.
#
# detectMarkers() returns each marker's four corners clockwise from the
# marker's own top-left: [TL, TR, BR, BL]. _CORNER_INDEX picks the right one
# per id; CORNER_NAME is the human label (shown in the UI, printed on the
# sheet).
CORNER_NAME = {0: "top-left", 1: "top-right", 2: "bottom-left", 3: "bottom-right"}
_CORNER_INDEX = {0: 0, 1: 1, 2: 3, 3: 2}


class NoMarkersConfiguredError(RuntimeError):
    pass


class NotEnoughMarkersDetectedError(RuntimeError):
    pass


def generate_marker_image(marker_id: int) -> np.ndarray:
    """The printable marker, padded with a white quiet zone (which also helps
    real-world detection), with a right-angle bracket drawn in that margin
    around ONE corner - the corner that is this marker's reference point (see
    CORNER_NAME: marker 0 -> its top-left, 1 -> top-right, 2 -> bottom-left,
    3 -> bottom-right).

    When registering this marker, jog the laser dot exactly onto that corner
    - where the two black edges of the marker's outer border meet, inside the
    bracket - and enter the machine X/Y it reads. detect_markers() reports
    the sub-pixel position of the same corner, so the point you register and
    the point the app detects are physically the same spot.

    The bracket sits entirely in the white margin, clear of the marker's own
    pixels, so it never interferes with detection. A filled triangle and a
    '#id' label in the top margin mark which way is up - fix all four sheets
    the same way round (triangle toward the back of the machine) so each id
    lands on the bed corner it is meant to."""
    marker = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, MARKER_SIZE_PX)
    size = MARKER_SIZE_PX + 2 * MARKER_MARGIN_PX
    canvas = np.full((size, size), 255, dtype=np.uint8)
    m0 = MARKER_MARGIN_PX
    m1 = MARKER_MARGIN_PX + MARKER_SIZE_PX
    canvas[m0:m1, m0:m1] = marker

    # This marker's reference corner, in canvas pixels (the outer corner of
    # the marker's border).
    cx, cy = {
        0: (m0, m0),   # top-left
        1: (m1, m0),   # top-right
        2: (m0, m1),   # bottom-left
        3: (m1, m1),   # bottom-right
    }[marker_id]

    # Right-angle bracket framing that corner, held well out into the margin.
    # The gap matters: detect_markers() runs cv2.cornerSubPix around this
    # corner, and any ink inside that search window would bias the refined
    # position outward. At the ~60-120px a marker realistically spans in the
    # camera view a 30px (of 420) gap keeps the bracket clear of even a
    # generous window. Nothing is drawn between the bracket and the corner -
    # the human aligns the laser to the marker's real border corner, the
    # bracket just says which one.
    gap, arm = 30, 40
    on_left, on_top = cx == m0, cy == m0
    vx = cx - gap if on_left else cx + gap
    vy = cy - gap if on_top else cy + gap
    arm_x = vx + arm if on_left else vx - arm
    arm_y = vy + arm if on_top else vy - arm
    cv2.line(canvas, (vx, vy), (arm_x, vy), 0, 3, cv2.LINE_AA)
    cv2.line(canvas, (vx, vy), (vx, arm_y), 0, 3, cv2.LINE_AA)

    # 'this edge points to the back of the machine' cue for whoever fixes the
    # sheet - keeps id -> bed-corner consistent regardless of marker rotation
    c = size // 2
    tri = np.array([[c, 8], [c - 10, 26], [c + 10, 26]], np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [tri], 0)
    cv2.putText(canvas, f"#{marker_id}", (c + 18, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2, cv2.LINE_AA)
    return canvas


# --- printable sheet -------------------------------------------------------

SHEET_DPI = 300
SHEET_MARKER_MM = 40.0   # printed size of the ArUco pattern (the black square
                         # incl. its 1-module border). The cut-out tile is ~1.4x
                         # this once the un-trimmable white quiet zone is added.
                         # 40 mm reads reliably on a 640x480 feed over a 400 mm
                         # bed with room to spare for corner sub-pixel accuracy,
                         # and four fit comfortably on one A4 / Letter page.
_PAPER_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
_TILE_MM = SHEET_MARKER_MM * (MARKER_SIZE_PX + 2 * MARKER_MARGIN_PX) / MARKER_SIZE_PX


def _sheet_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Pillow's built-in bitmap font, scaled - avoids depending on any font
    file being installed (this app is meant to run offline on a workbench PC)."""
    try:
        return ImageFont.load_default(size=px)
    except TypeError:            # very old Pillow: size arg unsupported
        return ImageFont.load_default()


def build_marker_sheet(paper: str = "a4") -> bytes:
    """All four markers on one printable page (PDF bytes) at SHEET_MARKER_MM,
    each labelled with its bed corner and carrying the same '#id / this-way-up'
    cue as the individual images. A 50 mm scale bar at the bottom lets you
    confirm the printer isn't rescaling - ArUco detection does not need an exact
    size, but every marker must be the SAME size and square, which 'fit to
    page' breaks."""
    mm = SHEET_DPI / 25.4
    page_w_mm, page_h_mm = _PAPER_MM.get(paper, _PAPER_MM["a4"])
    W, H = round(page_w_mm * mm), round(page_h_mm * mm)

    margin = round(15 * mm)
    col_gap = round(12 * mm)
    row_label_gap = round(16 * mm)
    header_h = round(28 * mm)

    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)

    tile_px = round((MARKER_SIZE_PX + 2 * MARKER_MARGIN_PX) * (SHEET_MARKER_MM / MARKER_SIZE_PX) * mm)

    title_f = _sheet_font(round(4.8 * mm))
    body_f = _sheet_font(round(3.1 * mm))
    label_f = _sheet_font(round(3.6 * mm))

    draw.text((margin, margin), "ArUco calibration markers", font=title_f, fill="black")
    intro = (
        "Print at 100% / \"Actual size\" - NOT \"fit to page\" or \"shrink to fit\".\n"
        f"Each pattern prints {SHEET_MARKER_MM:.0f} mm; cut out the {_TILE_MM:.0f} mm "
        "square (keep the whole white border).\n"
        "Fix one at each corner of the bed, all the same way up, triangle to the BACK "
        "of the machine.\n"
        "Each marker's reference point is the corner inside its bracket."
    )
    draw.multiline_text((margin, margin + round(6.5 * mm)), intro, font=body_f,
                        fill="black", spacing=round(1.4 * mm))

    grid_top = margin + header_h
    col_x = [margin, margin + tile_px + col_gap]
    row_pitch = tile_px + row_label_gap
    row_y = [grid_top, grid_top + row_pitch]

    for mid in range(4):
        x = col_x[mid % 2]
        y = row_y[mid // 2]
        tile = Image.fromarray(generate_marker_image(mid)).convert("RGB")
        tile = tile.resize((tile_px, tile_px), Image.LANCZOS)
        page.paste(tile, (x, y))
        # cut frame flush with the tile edge
        draw.rectangle([x, y, x + tile_px - 1, y + tile_px - 1], outline="black", width=1)
        draw.text((x, y + tile_px + round(3 * mm)),
                  f"#{mid}  ·  {CORNER_NAME[mid]}", font=label_f, fill="black")

    bar_mm = 50.0
    bar_px = round(bar_mm * mm)
    bar_y = row_y[1] + row_pitch + round(6 * mm)
    if bar_y + round(8 * mm) < H - margin:
        draw.line([margin, bar_y, margin + bar_px, bar_y], fill="black", width=2)
        for end in (margin, margin + bar_px):
            draw.line([end, bar_y - round(1.6 * mm), end, bar_y + round(1.6 * mm)],
                      fill="black", width=2)
        draw.text((margin + bar_px + round(3 * mm), bar_y - round(2 * mm)),
                  "50 mm - measure this to check your printer's scale",
                  font=body_f, fill="black")

    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=float(SHEET_DPI))
    return buf.getvalue()


def detect_markers(frame: np.ndarray) -> dict[int, tuple[float, float]]:
    """Every visible marker's reference-corner pixel position (sub-pixel
    refined), keyed by marker ID - regardless of whether that ID has a
    registered machine position yet. Which corner is the reference is fixed
    per id by _CORNER_INDEX and matches the bracket generate_marker_image
    prints on that marker's sheet."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    detector = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    result = {}
    if ids is None:
        return result
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        mid = int(marker_id)
        pts = marker_corners.reshape(4, 2).astype(np.float32)  # clockwise from the marker's own TL
        cv2.cornerSubPix(gray, pts, (3, 3), (-1, -1), criteria)
        ref = pts[_CORNER_INDEX.get(mid, 0)]
        result[mid] = (float(ref[0]), float(ref[1]))
    return result


def load_marker_positions() -> dict[int, tuple[float, float]]:
    if not MARKERS_PATH.exists():
        return {}
    data = json.loads(MARKERS_PATH.read_text())
    return {int(k): (v[0], v[1]) for k, v in data.items()}


def save_marker_position(marker_id: int, machine_x_mm: float, machine_y_mm: float) -> None:
    positions = load_marker_positions()
    positions[marker_id] = (machine_x_mm, machine_y_mm)
    MARKERS_PATH.write_text(json.dumps({str(k): list(v) for k, v in positions.items()}, indent=2))


def remove_marker_position(marker_id: int) -> None:
    positions = load_marker_positions()
    positions.pop(marker_id, None)
    MARKERS_PATH.write_text(json.dumps({str(k): list(v) for k, v in positions.items()}, indent=2))


def clear_marker_positions() -> None:
    MARKERS_PATH.unlink(missing_ok=True)


def auto_calibrate(frame: np.ndarray, frame_width: int = 640, frame_height: int = 480) -> np.ndarray:
    known_positions = load_marker_positions()
    if not known_positions:
        raise NoMarkersConfiguredError(
            "No marker positions registered yet - print markers and register each "
            "one's real machine position first."
        )

    detected = detect_markers(frame)
    matched = [
        CalibrationPoint(
            pixel_x=px, pixel_y=py,
            machine_x_mm=known_positions[mid][0], machine_y_mm=known_positions[mid][1],
        )
        for mid, (px, py) in detected.items() if mid in known_positions
    ]
    if len(matched) < MIN_MARKERS_REQUIRED:
        raise NotEnoughMarkersDetectedError(
            f"Only {len(matched)} registered marker(s) detected in the current photo "
            f"(need at least {MIN_MARKERS_REQUIRED}) - check they're all in frame, "
            f"clean, well lit, and not blocked by the gantry or a workpiece."
        )

    # CalibrationError (e.g. the spread-check) propagates as-is - same
    # safety guard as manual calibration applies here automatically.
    homography = save_calibration(matched, frame_width, frame_height)

    # The markers sit at the corners of the reachable bed, so the box their
    # outer (reference) corners bound is exactly the area detection should
    # look inside - anything past them is machine frame / cables / belt
    # track. Set it from the same frame, so calibrating the homography and
    # framing the bed are one action.
    corner_px = [detected[mid] for mid in known_positions if mid in detected]
    xs = [px for px, _ in corner_px]
    ys = [py for _, py in corner_px]
    save_settings({"bed_roi_px": [
        int(round(min(xs))), int(round(min(ys))),
        int(round(max(xs))), int(round(max(ys))),
    ]})
    return homography
