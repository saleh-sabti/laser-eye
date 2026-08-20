"""Fit an uploaded SVG design onto a detected object.

Design SVGs must declare real-world units on width/height (mm/in/cm/pt) -
the normal output of laser design software (LightBurn, Inkscape with mm
document units, etc.). svgelements resolves all geometry into a canonical
96-CSS-px document space regardless of the source unit, so MM_PER_PX below
converts that back to millimeters; a design with no physical unit on
width/height would be spec-treated as raw pixels and come out wrong here.

The design's bounding-box center is translated/rotated onto the detected
object's centroid/orientation, then every path is clipped against the
detected outline (shrunk inward by a safety margin) so nothing lands off
the edge of an irregular piece.
"""
import math
from dataclasses import dataclass

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseMultipartGeometry
from svgelements import SVG, Path, Matrix

from .detection import Detection

DEFAULT_SAFETY_MARGIN_MM = 1.5
CURVE_SAMPLES = 16
MM_PER_PX = 25.4 / 96


@dataclass
class AlignedDesign:
    lines_mm: list[list[tuple[float, float]]]  # clipped polylines, machine mm


def _flatten_path(path: Path, samples_per_curve: int = CURVE_SAMPLES) -> list[list[tuple[float, float]]]:
    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for seg in path.segments(transformed=True):
        name = seg.__class__.__name__
        if name == "Move":
            if current:
                subpaths.append(current)
            current = [(seg.end[0], seg.end[1])]
        elif name == "Close":
            if current:
                current.append(current[0])
        else:
            steps = 2 if name == "Line" else samples_per_curve
            for i in range(1, steps + 1):
                pt = seg.point(i / steps)
                current.append((pt[0], pt[1]))
    if current:
        subpaths.append(current)
    return subpaths


def _design_paths_and_bbox(svg_path: str) -> tuple[list[Path], tuple[float, float, float, float]]:
    svg = SVG.parse(svg_path)
    paths = [Path(el) for el in svg.elements() if hasattr(el, "segments")]
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    for p in paths:
        bbox = p.bbox()
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        xmin, ymin = min(xmin, x0), min(ymin, y0)
        xmax, ymax = max(xmax, x1), max(ymax, y1)
    if xmin is math.inf:
        raise ValueError("SVG has no drawable paths.")
    return paths, (xmin, ymin, xmax, ymax)


def align_and_clip(
    svg_path: str,
    detection: Detection,
    safety_margin_mm: float = DEFAULT_SAFETY_MARGIN_MM,
) -> AlignedDesign:
    paths, (xmin, ymin, xmax, ymax) = _design_paths_and_bbox(svg_path)
    design_cx, design_cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    target_cx, target_cy = detection.centroid_mm
    angle_rad = math.radians(detection.angle_deg)

    # svgelements' Matrix `*` composes left-to-right (leftmost applied
    # first), the opposite of the usual column-vector convention - so this
    # reads in the actual order of application: center the design, convert
    # its units to mm, rotate, then move onto the detected centroid.
    transform = (
        Matrix.translate(-design_cx, -design_cy)
        * Matrix.scale(MM_PER_PX)
        * Matrix.rotate(angle_rad)
        * Matrix.translate(target_cx, target_cy)
    )
    for p in paths:
        p *= transform

    boundary = Polygon(detection.contour_mm).buffer(-safety_margin_mm)

    clipped_lines: list[list[tuple[float, float]]] = []
    for p in paths:
        for subpath in _flatten_path(p):
            if len(subpath) < 2:
                continue
            clipped = LineString(subpath).intersection(boundary)
            clipped_lines.extend(_extract_linestrings(clipped))

    return AlignedDesign(lines_mm=clipped_lines)


def _extract_linestrings(geom) -> list[list[tuple[float, float]]]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [list(geom.coords)]
    if isinstance(geom, BaseMultipartGeometry):
        result = []
        for part in geom.geoms:
            result.extend(_extract_linestrings(part))
        return result
    return []
