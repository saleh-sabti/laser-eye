"""Fit a photo onto a detected object.

Unlike vector designs, this is meant to work with raster laser software
(LaserGRBL, or LightBurn's image mode) that just burns a plain rectangular
image axis-aligned, with no concept of absolute canvas position or
rotation. So instead of relying on the destination software to place or
rotate anything, everything is baked directly into the output pixels:
background removed, rotated to match the object's orientation, and scaled
to the object's real-world footprint. The result is a plain PNG - you set
the machine's job origin to the reported target position yourself.

Background removal uses rembg (a local, offline model) rather than a cloud
API, so this keeps working without an internet connection after the
one-time model download.
"""
import io
from dataclasses import dataclass

from PIL import Image
from rembg import remove
from shapely.geometry import Polygon
from shapely.affinity import rotate as shapely_rotate

from .detection import Detection

DEFAULT_SAFETY_MARGIN_MM = 1.5
MM_PER_INCH = 25.4


@dataclass
class AlignedPhoto:
    image: Image.Image          # RGBA, already rotated/scaled/cropped
    dpi: tuple[float, float]    # embed so DPI-aware importers auto-size correctly
    width_mm: float
    height_mm: float
    target_x_mm: float          # where to center this on the bed (detected centroid)
    target_y_mm: float
    angle_deg: float            # for your own reference - already baked into the pixels


def remove_background(photo_bytes: bytes) -> Image.Image:
    source = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    return remove(source)


def _autocrop_to_subject(img: Image.Image) -> Image.Image:
    bbox = img.split()[-1].getbbox()
    if bbox is None:
        raise ValueError(
            "Background removal left nothing visible - is the subject "
            "clear against its background in the photo?"
        )
    return img.crop(bbox)


def _oriented_footprint_mm(detection: Detection, safety_margin_mm: float) -> tuple[float, float]:
    """The object's own width/height along its principal axis, i.e. what
    it measures when you un-rotate it back to axis-aligned - this is the
    footprint the (soon to be rotated) photo needs to fit inside.
    """
    polygon = Polygon(detection.contour_mm).buffer(-safety_margin_mm)
    if polygon.is_empty:
        raise ValueError(
            "Safety margin leaves no usable area on this object - it may "
            "be too small, or the margin too large."
        )
    centroid = polygon.centroid
    unrotated = shapely_rotate(polygon, -detection.angle_deg, origin=centroid, use_radians=False)
    minx, miny, maxx, maxy = unrotated.bounds
    return maxx - minx, maxy - miny


def align_photo(
    photo_bytes: bytes,
    detection: Detection,
    safety_margin_mm: float = DEFAULT_SAFETY_MARGIN_MM,
) -> AlignedPhoto:
    cutout = _autocrop_to_subject(remove_background(photo_bytes))

    footprint_w_mm, footprint_h_mm = _oriented_footprint_mm(detection, safety_margin_mm)
    scale_mm_per_px = min(footprint_w_mm / cutout.width, footprint_h_mm / cutout.height)
    width_mm = cutout.width * scale_mm_per_px
    height_mm = cutout.height * scale_mm_per_px

    rotated = cutout.rotate(-detection.angle_deg, expand=True, resample=Image.BICUBIC)
    # canvas grew from the rotation - keep the same physical scale, so its
    # printed size grows proportionally too
    canvas_scale = cutout.width / rotated.width if rotated.width else 1.0
    dpi_value = MM_PER_INCH / (scale_mm_per_px / canvas_scale) if scale_mm_per_px else 96.0

    return AlignedPhoto(
        image=rotated,
        dpi=(dpi_value, dpi_value),
        width_mm=rotated.width * scale_mm_per_px * canvas_scale,
        height_mm=rotated.height * scale_mm_per_px * canvas_scale,
        target_x_mm=detection.centroid_mm[0],
        target_y_mm=detection.centroid_mm[1],
        angle_deg=detection.angle_deg,
    )


def save_png(aligned: AlignedPhoto, path: str) -> None:
    aligned.image.save(path, dpi=aligned.dpi)
