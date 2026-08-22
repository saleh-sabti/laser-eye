"""Find the workpiece's real outline in a bed photo.

Approach: background-subtract against a stored "empty bed" reference frame
(captured once whenever the mat/lighting is set up), which works for
arbitrary/irregular shapes because it makes no assumption about geometry -
unlike e.g. looking for straight edges or a specific color range.

The diff is computed across all three color channels, not on a grayscale
conversion - wood grain that's dark enough to match a dark mat in pure
brightness still reads as clearly different once color (not just
luminance) is taken into account. Whatever outer boundary that produces
gets filled solid before use: a real workpiece doesn't have holes, so any
gaps in the diff signal (e.g. a grain patch that still slips through)
shouldn't punch a hole in the detected silhouette.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import REFERENCE_FRAME_PATH
from .calibration import pixels_to_mm

MIN_OBJECT_AREA_PX = 400
# Real wood-vs-mat contrast is dramatically stronger than residual sensor
# noise, and noise blobs (formed by morphological closing bridging small
# scattered specks together) are much more fragmented than a real object's
# contour. Measured on real footage: noise ~20 mean-diff / ~0.3 solidity,
# a real piece ~200+ mean-diff / ~0.8 solidity - these thresholds sit with
# a wide margin on either side of both.
MIN_MEAN_DIFF_INSIDE = 60
MIN_SOLIDITY = 0.5


class NoReferenceFrameError(RuntimeError):
    pass


class NoObjectFoundError(RuntimeError):
    pass


@dataclass
class Detection:
    contour_px: np.ndarray       # Nx2 pixel coords of the outline
    contour_mm: np.ndarray       # Nx2 machine-mm coords of the outline
    centroid_px: tuple[float, float]
    centroid_mm: tuple[float, float]
    angle_deg: float             # principal axis angle, machine-mm frame
    length_mm: float             # real-world size: longer side of the oriented
    width_mm: float              # bounding box, shorter side - both in mm


def save_reference_frame(frame: np.ndarray) -> None:
    cv2.imwrite(str(REFERENCE_FRAME_PATH), frame)


def has_reference_frame() -> bool:
    return REFERENCE_FRAME_PATH.exists()


def _load_reference_frame() -> np.ndarray:
    if not REFERENCE_FRAME_PATH.exists():
        raise NoReferenceFrameError(
            "No empty-bed reference frame saved yet - capture one with "
            "nothing on the bed first."
        )
    return cv2.imread(str(REFERENCE_FRAME_PATH))


def _color_diff(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    f_blur = cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16)
    r_blur = cv2.GaussianBlur(reference, (3, 3), 0).astype(np.int16)
    diff = np.abs(f_blur - r_blur).sum(axis=2)
    return np.clip(diff, 0, 255).astype(np.uint8)


def _largest_filled_contour(
    frame: np.ndarray, reference: np.ndarray, roi: tuple[int, int, int, int] | None = None
) -> np.ndarray | None:
    diff = _color_diff(frame, reference)

    # Otsu picks the threshold from the image's own diff histogram instead
    # of a fixed cutoff, so it adapts to how much of a difference the
    # current lighting/exposure actually produces. Critically, this has to
    # run on just the region we actually care about: padding the rest of
    # the frame with zeros *before* computing it (an earlier version of
    # this did that) skews the histogram badly - the huge zero-padded area
    # pulls Otsu's chosen split way down, to the point where ordinary
    # sensor noise inside the real ROI starts reading as "different".
    # Measured on a real empty-bed frame: threshold 13-14 computed on the
    # ROI alone vs. 7 after zero-padding the rest first - low enough to
    # misclassify most of the ROI's own background noise as foreground.
    if roi is not None:
        x0, y0, x1, y1 = roi
        roi_diff = diff[y0:y1, x0:x1]
        _, roi_mask = cv2.threshold(roi_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.zeros(diff.shape, dtype=np.uint8)
        mask[y0:y1, x0:x1] = roi_mask
    else:
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kernels are small on purpose: a workpiece can occupy well under 50px
    # across in the frame (e.g. a narrow offcut), and the interior-hole
    # problem (dark wood grain reading as background) is handled separately
    # below by filling the traced contour solid - so these only need to
    # bridge tiny gaps in the boundary ring, not smooth/fill large regions.
    # Bigger kernels here erase genuine small-scale jagged edge detail.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_OBJECT_AREA_PX]
    contours.sort(key=cv2.contourArea, reverse=True)

    for candidate in contours:
        hull = cv2.convexHull(candidate)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(candidate) / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue

        candidate_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(candidate_mask, [candidate], -1, 255, thickness=cv2.FILLED)
        mean_diff_inside = diff[candidate_mask > 0].mean()
        if mean_diff_inside < MIN_MEAN_DIFF_INSIDE:
            continue

        filled = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(filled, [candidate], -1, 255, thickness=cv2.FILLED)
        return filled

    return None


def _principal_angle_deg(points_mm: np.ndarray) -> float:
    centered = points_mm - points_mm.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    principal_axis = vt[0]
    return float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0])))


def _oriented_dimensions_mm(contour_mm: np.ndarray) -> tuple[float, float]:
    """Real-world size via the minimum-area bounding box (the smallest
    rectangle, at any angle, that fully contains the outline) - the
    standard way to answer "how big is this piece actually" for an
    irregular shape. Returns (longer side, shorter side) in mm.
    """
    (_, _), (w, h), _ = cv2.minAreaRect(contour_mm.astype(np.float32))
    return (max(w, h), min(w, h))


def detect_object(
    frame: np.ndarray, homography: np.ndarray, roi: tuple[int, int, int, int] | None = None
) -> Detection:
    reference = _load_reference_frame()
    filled_mask = _largest_filled_contour(frame, reference, roi=roi)
    if filled_mask is None:
        raise NoObjectFoundError("No object found on the bed against the reference frame.")

    # re-extract from the filled mask so the returned contour is the clean
    # solid outer boundary, not the (possibly ragged/gapped) raw diff shape
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_px = contour.reshape(-1, 2).astype(np.float64)

    moments = cv2.moments(contour)
    centroid_px = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    contour_mm = pixels_to_mm(homography, contour_px)
    centroid_mm_arr = pixels_to_mm(homography, np.array([centroid_px]))
    centroid_mm = (float(centroid_mm_arr[0][0]), float(centroid_mm_arr[0][1]))

    angle_deg = _principal_angle_deg(contour_mm)
    length_mm, width_mm = _oriented_dimensions_mm(contour_mm)

    return Detection(
        contour_px=contour_px,
        contour_mm=contour_mm,
        centroid_px=centroid_px,
        centroid_mm=centroid_mm,
        angle_deg=angle_deg,
        length_mm=length_mm,
        width_mm=width_mm,
    )


def draw_overlay(frame: np.ndarray, detection: Detection) -> np.ndarray:
    overlay = frame.copy()
    pts = detection.contour_px.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cx, cy = int(detection.centroid_px[0]), int(detection.centroid_px[1])
    cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    label = f"{detection.length_mm:.0f} x {detection.width_mm:.0f} mm"
    text_pos = (cx + 12, cy - 12)
    cv2.putText(overlay, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return overlay
