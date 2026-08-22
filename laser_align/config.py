"""Persistent app settings: camera device, bed size, and file locations.

Everything lives in a single JSON file so the whole setup can be inspected
or hand-edited without touching code.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "calibration_data"
DATA_DIR.mkdir(exist_ok=True)

SETTINGS_PATH = DATA_DIR / "settings.json"
CALIBRATION_PATH = DATA_DIR / "calibration.json"
REFERENCE_FRAME_PATH = DATA_DIR / "empty_bed_reference.png"

DEFAULT_SETTINGS = {
    "camera_index": 0,
    "camera_width": 640,   # OpenCV/DSHOW default to 640x480 unless told otherwise - most
    "camera_height": 480,  # webcams support far more; raising this is the single biggest
                            # precision lever available (more real pixels per mm of bed).
                            # Changing it invalidates any saved calibration/bed area, since
                            # those are pixel coordinates at whatever resolution they were
                            # set at - redo both after changing this.
    "bed_width_mm": 400.0,
    "bed_height_mm": 400.0,
    "detection_method": "classical",  # "classical" or "rfdetr"
    "grbl_port": "COM3",
    "grbl_baud": 921600,
    "bed_roi_px": None,  # [x0, y0, x1, y1] in camera pixels - the actual bed surface,
                         # excluding the machine frame/cables/belt track. None = whole frame.
}


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        settings = {**DEFAULT_SETTINGS, **json.loads(SETTINGS_PATH.read_text())}
    else:
        settings = dict(DEFAULT_SETTINGS)
    return settings


def save_settings(settings: dict) -> None:
    merged = {**load_settings(), **settings}
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
