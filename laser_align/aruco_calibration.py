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
import json

import cv2
import numpy as np

from .config import DATA_DIR
from .calibration import CalibrationPoint, CalibrationError, save_calibration

MARKERS_PATH = DATA_DIR / "aruco_markers.json"
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_SIZE_PX = 300
MIN_MARKERS_REQUIRED = 4


class NoMarkersConfiguredError(RuntimeError):
    pass


class NotEnoughMarkersDetectedError(RuntimeError):
    pass


def generate_marker_image(marker_id: int) -> np.ndarray:
    return cv2.aruco.generateImageMarker(DICTIONARY, marker_id, MARKER_SIZE_PX)


def detect_markers(frame: np.ndarray) -> dict[int, tuple[float, float]]:
    """Every visible marker's pixel center, keyed by marker ID - regardless
    of whether that ID has a registered machine position yet."""
    detector = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)
    result = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            center = marker_corners[0].mean(axis=0)
            result[int(marker_id)] = (float(center[0]), float(center[1]))
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
    return save_calibration(matched, frame_width, frame_height)
