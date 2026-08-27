"""Hand-placement of a design on the bed.

The Design & Export editor lets you move / rotate / scale a design over a
photo of the bed and export it landing exactly where you put it. This
module is the geometry behind that: parse the design once into millimetre
polylines centred on the origin, then apply the editor's transform
(translate to a machine-mm centre, rotate, scale) and optionally clip it to
the detected workpiece outline.

Everything downstream (`export.write_svg`) already speaks
`align.AlignedDesign` (a list of machine-mm polylines), so `place()` returns
that.

Y convention: SVG space is Y-down; this negates Y on load so the design is
Y-up, matching the machine coordinates the homography maps to - so what the
editor draws (design -> machine mm -> pixel, via the homography) sits on the
real bed the same way it will burn. `export.write_svg`'s `flip_y` is the
last step, for LightBurn's own axis setup.
"""
import math

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseMultipartGeometry

from .align import AlignedDesign, _design_paths_and_bbox, _flatten_path, MM_PER_PX

DEFAULT_SAFETY_MARGIN_MM = 1.5
DEFAULT_TRACE_LONG_MM = 100.0   # a traced raster has no real-world size; start it this
                                # big on its longer side, then the editor / fit-to-piece
                                # rescales.

Polyline = list[tuple[float, float]]


def load_svg_mm(svg_path: str) -> tuple[list[Polyline], float, float]:
    """Parse an SVG into millimetre polylines centred on (0, 0). Returns
    (polylines, width_mm, height_mm)."""
    paths, (xmin, ymin, xmax, ymax) = _design_paths_and_bbox(svg_path)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w_mm = (xmax - xmin) * MM_PER_PX
    h_mm = (ymax - ymin) * MM_PER_PX

    polylines: list[Polyline] = []
    for p in paths:
        for sub in _flatten_path(p):
            if len(sub) < 2:
                continue
            polylines.append([
                ((x - cx) * MM_PER_PX, -(y - cy) * MM_PER_PX)  # negate Y: SVG down -> machine up
                for x, y in sub
            ])
    if not polylines:
        raise ValueError("SVG has no drawable paths.")
    return polylines, w_mm, h_mm


def trace_image_to_mm(
    cutout: Image.Image, long_side_mm: float = DEFAULT_TRACE_LONG_MM
) -> tuple[list[Polyline], float, float]:
    """Trace a background-removed RGBA image to millimetre polylines centred
    on the origin. Traces the dark 'ink' inside the subject (for line art /
    black-on-white designs); if there is no ink to speak of, falls back to
    the subject's silhouette. Returns (polylines, width_mm, height_mm)."""
    rgba = np.array(cutout.convert("RGBA"))
    alpha = rgba[:, :, 3]
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)

    subject = alpha > 128
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (ink > 0) & subject
    if ink.mean() < 0.005:                      # basically no dark lines -> use the outline
        ink = subject
    mask = (ink.astype(np.uint8)) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    min_len = max(8.0, 0.004 * (w + h))         # drop specks
    polys_px: list[Polyline] = []
    for c in contours:
        if cv2.arcLength(c, True) < min_len:
            continue
        approx = cv2.approxPolyDP(c, 1.2, True).reshape(-1, 2)
        if len(approx) < 2:
            continue
        line = [(float(x), float(y)) for x, y in approx]
        line.append(line[0])                    # close it
        polys_px.append(line)
    if not polys_px:
        raise ValueError("Nothing to trace - the image came out blank after background removal.")

    xs = [x for line in polys_px for x, _ in line]
    ys = [y for line in polys_px for _, y in line]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    scale = long_side_mm / max(bw, bh, 1.0)

    polylines = [
        [((x - cx) * scale, -(y - cy) * scale) for x, y in line]   # negate Y: image down -> machine up
        for line in polys_px
    ]
    return polylines, bw * scale, bh * scale


def place(
    polylines: list[Polyline],
    center_x_mm: float,
    center_y_mm: float,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    flip_x: bool = False,
) -> list[Polyline]:
    """Apply the editor transform to origin-centred polylines: optional
    mirror, scale, rotate, then translate the centre to (center_x_mm,
    center_y_mm) in machine millimetres."""
    a = math.radians(rotation_deg)
    ca, sa = math.cos(a), math.sin(a)
    sx = -scale if flip_x else scale
    out: list[Polyline] = []
    for line in polylines:
        placed: Polyline = []
        for x, y in line:
            x2, y2 = x * sx, y * scale
            rx = x2 * ca - y2 * sa
            ry = x2 * sa + y2 * ca
            placed.append((rx + center_x_mm, ry + center_y_mm))
        out.append(placed)
    return out


def clip_to_outline(
    polylines: list[Polyline], contour_mm, safety_margin_mm: float = DEFAULT_SAFETY_MARGIN_MM
) -> list[Polyline]:
    """Keep only the parts of each polyline inside the detected outline
    (shrunk inward by the safety margin)."""
    boundary = Polygon(contour_mm).buffer(-safety_margin_mm)
    if boundary.is_empty:
        return []
    out: list[Polyline] = []
    for line in polylines:
        if len(line) < 2:
            continue
        clipped = LineString(line).intersection(boundary)
        out.extend(_extract_linestrings(clipped))
    return out


def _extract_linestrings(geom) -> list[Polyline]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [list(geom.coords)]
    if isinstance(geom, BaseMultipartGeometry):
        result: list[Polyline] = []
        for part in geom.geoms:
            result.extend(_extract_linestrings(part))
        return result
    return []


def polylines_bbox_center(polylines: list[Polyline]) -> tuple[float, float]:
    """Midpoint of the axis-aligned bounding box - what "the design is
    centred at (tx, ty)" actually means (place() keeps a shape centred on
    its own bbox centred on the translation target)."""
    xs = [x for line in polylines for x, _ in line]
    ys = [y for line in polylines for _, y in line]
    if not xs:
        return (0.0, 0.0)
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


def to_aligned_design(polylines: list[Polyline]) -> AlignedDesign:
    return AlignedDesign(lines_mm=[list(line) for line in polylines])
