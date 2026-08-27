"""Pixel <-> machine-mm calibration.

The reference points come from the machine itself: the user jogs the laser
(pointer / low power, in LightBurn) to a point, reads the real machine
(X, Y) off LightBurn, clicks the matching spot in a captured photo of the
bed, and enters the coordinates here. Four or more such (pixel, mm) pairs
are enough to fit a homography that maps any pixel in later photos to a
real machine position.
"""
import json
import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .config import CALIBRATION_PATH

MIN_POINTS = 4

# Browsers commonly auto-refill number inputs with whatever was typed last,
# so it's easy to click a new spot in the photo and forget to update the
# machine X/Y fields before submitting - producing two different pixel
# locations mapped to the same real-world point (or vice versa), which
# silently corrupts the homography. These tolerances catch that mistake.
DUPLICATE_PIXEL_TOLERANCE_PX = 3.0
DUPLICATE_MM_TOLERANCE_MM = 0.5

# A homography fitted from points crammed into one small area of the frame
# reproduces those points exactly, but extrapolates wildly for anything
# outside that area - the classic failure mode is points clustered in one
# strip/corner instead of spread across the bed. Caught in practice: 4
# points spanning only 44% of frame width and 24% of height produced a
# detected object over 170,000mm off from reality. Each axis must span at
# least this fraction of the frame to catch that before it's saved.
MIN_SPREAD_FRACTION = 0.35


@dataclass
class CalibrationPoint:
    pixel_x: float
    pixel_y: float
    machine_x_mm: float
    machine_y_mm: float


class CalibrationError(ValueError):
    pass


def find_duplicate_conflict(points: list[CalibrationPoint], candidate: CalibrationPoint) -> str | None:
    """Return a warning if `candidate` looks like an accidental duplicate
    of an existing point - most commonly, a different pixel was clicked but
    the machine X/Y fields still held the previous point's values.
    """
    for p in points:
        pixel_dist = math.hypot(p.pixel_x - candidate.pixel_x, p.pixel_y - candidate.pixel_y)
        mm_dist = math.hypot(p.machine_x_mm - candidate.machine_x_mm, p.machine_y_mm - candidate.machine_y_mm)
        if mm_dist < DUPLICATE_MM_TOLERANCE_MM and pixel_dist > DUPLICATE_PIXEL_TOLERANCE_PX:
            return (
                f"Machine coordinates ({candidate.machine_x_mm}, {candidate.machine_y_mm}) mm "
                "match a point already added, but you clicked a different spot in the photo - "
                "looks like the Machine X/Y fields weren't updated for this new point."
            )
        if pixel_dist < DUPLICATE_PIXEL_TOLERANCE_PX and mm_dist > DUPLICATE_MM_TOLERANCE_MM:
            return (
                "That's essentially the same spot in the photo as a point you already added, "
                "but with different machine coordinates - click a different spot."
            )
    return None


def compute_homography(
    points: list[CalibrationPoint], frame_width: int = 640, frame_height: int = 480
) -> np.ndarray:
    if len(points) < MIN_POINTS:
        raise CalibrationError(
            f"Need at least {MIN_POINTS} calibration points, got {len(points)}."
        )

    xs = [p.pixel_x for p in points]
    ys = [p.pixel_y for p in points]
    x_spread, y_spread = max(xs) - min(xs), max(ys) - min(ys)
    min_x_spread, min_y_spread = frame_width * MIN_SPREAD_FRACTION, frame_height * MIN_SPREAD_FRACTION
    if x_spread < min_x_spread or y_spread < min_y_spread:
        raise CalibrationError(
            f"These points only span {x_spread:.0f}x{y_spread:.0f}px of the "
            f"{frame_width}x{frame_height}px frame - too clustered in one area. Reset and "
            f"redo them spread across the whole visible bed (e.g. one point per corner), "
            f"or the mapping will extrapolate badly for anything outside where you clicked."
        )

    # Checked here, not just in find_duplicate_conflict()'s live UI check,
    # so *every* entry path is protected the same way - this was missed for
    # ArUco marker registration (no per-marker duplicate check existed
    # there) and produced a real degenerate homography in practice: two
    # markers both registered at (370, 0)mm collapsed an entire axis of the
    # fitted matrix to ~0, so every detection mapped to Y=0 regardless of
    # input (dimensions came out "0 x 0mm", and actual alignment/export
    # would have been wrong too, not just the dimension label).
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            mm_dist = math.hypot(
                points[i].machine_x_mm - points[j].machine_x_mm,
                points[i].machine_y_mm - points[j].machine_y_mm,
            )
            pixel_dist = math.hypot(
                points[i].pixel_x - points[j].pixel_x, points[i].pixel_y - points[j].pixel_y
            )
            if mm_dist < DUPLICATE_MM_TOLERANCE_MM and pixel_dist > DUPLICATE_PIXEL_TOLERANCE_PX:
                raise CalibrationError(
                    f"Points {i + 1} and {j + 1} both have machine position "
                    f"({points[i].machine_x_mm}, {points[i].machine_y_mm}) mm but different "
                    f"pixel locations - one was entered wrong (this always breaks the "
                    f"mapping). Fix or remove one before saving."
                )

    pixel_pts = np.array([[p.pixel_x, p.pixel_y] for p in points], dtype=np.float32)
    mm_pts = np.array([[p.machine_x_mm, p.machine_y_mm] for p in points], dtype=np.float32)

    # With exactly 4 points (the ArUco path always gives 4, one per corner
    # marker), the pixel quad and the mm quad - walked in the *same* point
    # order - must both be simple convex quadrilaterals wound the same way.
    # If a marker's registered machine position doesn't match the corner it
    # physically sits at (e.g. two markers' positions swapped), the mm quad
    # comes out self-intersecting (a "bowtie") while the pixel quad is fine.
    # findHomography still returns a matrix for that - it reproduces the 4
    # points but its projective denominator passes through zero inside the
    # frame, so interior points map to garbage (six-figure mm). Hit for
    # real: markers registered as (0,0)/(10,370)/(370,370)/(370,0) in ID
    # order, with 2 and 3 effectively swapped - bed centre mapped to
    # ~130,000 mm. Caught here so it can't be saved.
    if len(points) == 4:
        _check_quads_consistent(pixel_pts, mm_pts)

    homography, _ = cv2.findHomography(pixel_pts, mm_pts, method=0)
    if homography is None:
        raise CalibrationError(
            "Could not fit a homography from these points - check they "
            "aren't collinear or duplicated."
        )

    # Final guard: the fitted homography must stay finite across the whole
    # frame. Its denominator w = h20*x + h21*y + h22 blowing through zero
    # inside the frame (the failure mode above, and anything else that makes
    # the fit near-degenerate) sends points near that line to infinity.
    h = homography
    corners = [(0, 0), (frame_width, 0), (0, frame_height), (frame_width, frame_height),
               (frame_width / 2, frame_height / 2)]
    ws = [h[2, 0] * x + h[2, 1] * y + h[2, 2] for x, y in corners]
    if min(ws) <= 0 < max(ws) or min(abs(w) for w in ws) < 1e-6:
        raise CalibrationError(
            "This calibration diverges inside the camera frame - the mapping "
            "would send parts of the bed to impossible coordinates. Almost "
            "always a marker's registered machine position doesn't match the "
            "corner it's physically at. Re-register by jogging the laser to "
            "each marker's bracketed corner and reading the real X/Y."
        )
    return homography


def _hull_order(quad: np.ndarray) -> np.ndarray:
    """Indices that put the 4 points in convex-polygon order (CCW by angle
    around their centroid). Point order in a calibration isn't guaranteed
    (ArUco returns marker-ID order), so both quads get sorted the same way
    before they're compared."""
    c = quad.mean(axis=0)
    return np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))


def _is_simple_quad(quad: np.ndarray) -> bool:
    """True if the 4 vertices, in the given order, form a non-self-intersecting
    (simple) quadrilateral - every consecutive turn goes the same way."""
    signs = []
    for i in range(4):
        a, b, cc = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (cc[1] - b[1]) - (b[1] - a[1]) * (cc[0] - b[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def _check_quads_consistent(pixel_pts: np.ndarray, mm_pts: np.ndarray) -> None:
    order = _hull_order(pixel_pts)
    # Walk the mm points in the order the pixel points sit around the frame.
    # If they were each registered to the corner they actually occupy, this
    # traces a simple quadrilateral. If two markers' positions are swapped,
    # it traces a bowtie.
    if not _is_simple_quad(mm_pts[order]):
        raise CalibrationError(
            "Taken in the order the markers sit around the frame, the four "
            "registered machine positions trace a twisted (self-crossing) shape "
            "rather than a rectangle - two markers' positions are almost "
            "certainly mixed up. Re-register by jogging the laser to each "
            "marker's bracketed corner and entering the real X/Y it reads."
        )


def save_calibration(
    points: list[CalibrationPoint], frame_width: int = 640, frame_height: int = 480
) -> np.ndarray:
    homography = compute_homography(points, frame_width, frame_height)
    CALIBRATION_PATH.write_text(json.dumps({
        "points": [asdict(p) for p in points],
        "homography": homography.tolist(),
    }, indent=2))
    return homography


def load_homography() -> np.ndarray | None:
    if not CALIBRATION_PATH.exists():
        return None
    data = json.loads(CALIBRATION_PATH.read_text())
    return np.array(data["homography"], dtype=np.float64)


def load_calibration_points() -> list[CalibrationPoint]:
    if not CALIBRATION_PATH.exists():
        return []
    data = json.loads(CALIBRATION_PATH.read_text())
    return [CalibrationPoint(**p) for p in data["points"]]


def pixels_to_mm(homography: np.ndarray, pixel_points: np.ndarray) -> np.ndarray:
    """pixel_points: Nx2 array of (x, y) pixel coords -> Nx2 array of mm coords."""
    pts = pixel_points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(pts, homography)
    return transformed.reshape(-1, 2)
