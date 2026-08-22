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

    pixel_pts = np.array([[p.pixel_x, p.pixel_y] for p in points], dtype=np.float32)
    mm_pts = np.array([[p.machine_x_mm, p.machine_y_mm] for p in points], dtype=np.float32)
    homography, _ = cv2.findHomography(pixel_pts, mm_pts, method=0)
    if homography is None:
        raise CalibrationError(
            "Could not fit a homography from these points - check they "
            "aren't collinear or duplicated."
        )
    return homography


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
