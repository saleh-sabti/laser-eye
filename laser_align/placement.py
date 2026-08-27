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

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseMultipartGeometry

from .align import AlignedDesign, _design_paths_and_bbox, _flatten_path, MM_PER_PX

DEFAULT_SAFETY_MARGIN_MM = 1.5

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
