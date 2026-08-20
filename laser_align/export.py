"""Write the aligned/clipped design as an SVG in absolute machine-mm
coordinates, ready to be opened directly in LightBurn.

IMPORTANT one-time check (see plan milestone 4): this assumes LightBurn's
workspace origin/axis setup ("Absolute Coords" + origin corner in device
settings) matches the machine coordinate system these mm values come from.
If a test import lands mirrored or offset, flip `flip_y` in settings
rather than touching this file's geometry logic.
"""
from .align import AlignedDesign


def to_svg(
    aligned: AlignedDesign,
    bed_width_mm: float,
    bed_height_mm: float,
    flip_y: bool = False,
) -> str:
    def y(v: float) -> float:
        return (bed_height_mm - v) if flip_y else v

    path_elements = []
    for line in aligned.lines_mm:
        if len(line) < 2:
            continue
        d = f"M {line[0][0]:.3f},{y(line[0][1]):.3f} " + " ".join(
            f"L {x:.3f},{y(v):.3f}" for x, v in line[1:]
        )
        path_elements.append(f'<path d="{d}" fill="none" stroke="black" stroke-width="0.1" />')

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{bed_width_mm}mm" height="{bed_height_mm}mm" '
        f'viewBox="0 0 {bed_width_mm} {bed_height_mm}">\n'
        + "\n".join(path_elements)
        + "\n</svg>\n"
    )


def write_svg(path: str, aligned: AlignedDesign, bed_width_mm: float, bed_height_mm: float, flip_y: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_svg(aligned, bed_width_mm, bed_height_mm, flip_y=flip_y))
