"""Top-down ("rectified") view of the bed.

The calibration homography maps camera pixels -> machine mm. It's a
*perspective* transform, so a rectangle in mm is a skewed quadrilateral in
the camera image - you can't drag a design around the raw camera view and
keep its real size constant.

`rectify_frame()` warps the camera frame into a straight overhead image
where the mapping to machine mm is a single scalar (`px_per_mm`) in both
directions and the Y axis points up, the way the machine's does. Every bit
of placement math done on top of this view is then plain affine.

It's also the most direct way to *see* whether calibration is any good: if
the warped bed looks sheared, bowed, or smeared, that's the homography, not
whatever is drawn on top of it.
"""
import cv2
import numpy as np

# Display resolution of the rectified view, in pixels per mm of bed. 3 is
# plenty for placement work (a 400mm bed -> 1200px) and keeps the warp
# cheap; nothing downstream depends on the exact value since it's reported
# back with every image.
DEFAULT_PX_PER_MM = 3.0


def _mm_to_view_matrix(bed_height_mm: float, px_per_mm: float) -> np.ndarray:
    """machine mm -> rectified-image pixels. Scales by px_per_mm and flips Y
    so row 0 is the top of the bed at Y = bed_height (machine Y points up,
    image rows point down)."""
    return np.array([
        [px_per_mm, 0.0, 0.0],
        [0.0, -px_per_mm, px_per_mm * bed_height_mm],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rectify_frame(
    frame: np.ndarray,
    homography: np.ndarray,
    bed_width_mm: float,
    bed_height_mm: float,
    px_per_mm: float = DEFAULT_PX_PER_MM,
) -> np.ndarray:
    """Warp `frame` (camera pixels) into a top-down image of the bed,
    `bed_width_mm * px_per_mm` by `bed_height_mm * px_per_mm` pixels."""
    m = _mm_to_view_matrix(bed_height_mm, px_per_mm) @ homography
    size = (round(bed_width_mm * px_per_mm), round(bed_height_mm * px_per_mm))
    return cv2.warpPerspective(frame, m, size)


def view_px_to_mm(
    x_px: float, y_px: float, bed_height_mm: float, px_per_mm: float = DEFAULT_PX_PER_MM
) -> tuple[float, float]:
    """A clicked pixel in the rectified view -> machine mm."""
    return x_px / px_per_mm, bed_height_mm - y_px / px_per_mm


def mm_to_view_px(
    x_mm: float, y_mm: float, bed_height_mm: float, px_per_mm: float = DEFAULT_PX_PER_MM
) -> tuple[float, float]:
    """Machine mm -> pixel in the rectified view (inverse of view_px_to_mm)."""
    return x_mm * px_per_mm, (bed_height_mm - y_mm) * px_per_mm


def draw_mm_polyline(
    view: np.ndarray,
    points_mm: np.ndarray,
    bed_height_mm: float,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    color: tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
    closed: bool = True,
) -> np.ndarray:
    """Draw an mm-space polyline (e.g. a detected outline) onto the rectified
    view. No re-warping - the outline is already in mm."""
    out = view.copy()
    pts = np.array(
        [mm_to_view_px(x, y, bed_height_mm, px_per_mm) for x, y in points_mm],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def draw_grid(
    view: np.ndarray,
    bed_width_mm: float,
    bed_height_mm: float,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    step_mm: float = 50.0,
    color: tuple[int, int, int] = (120, 120, 120),
) -> np.ndarray:
    """Light reference grid every `step_mm`, labelled in mm."""
    out = view.copy()
    x = 0.0
    while x <= bed_width_mm + 1e-6:
        px = int(round(x * px_per_mm))
        cv2.line(out, (px, 0), (px, out.shape[0]), color, 1, cv2.LINE_AA)
        cv2.putText(out, f"{x:.0f}", (px + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        x += step_mm
    y = 0.0
    while y <= bed_height_mm + 1e-6:
        py = int(round((bed_height_mm - y) * px_per_mm))
        cv2.line(out, (0, py), (out.shape[1], py), color, 1, cv2.LINE_AA)
        cv2.putText(out, f"{y:.0f}", (2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        y += step_mm
    return out
