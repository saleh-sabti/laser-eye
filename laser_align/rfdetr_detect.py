"""RF-DETR-based detection - the trained-model alternative to the classical
background-subtraction detector in detection.py.

Returns the same Detection shape so app.py can swap between methods
without the rest of the pipeline (alignment, export) caring which one
produced it. Needs a model actually fine-tuned on your own bed/wood photos
first (see dataset.py for how training samples get collected, and
train_rfdetr.py for the training script) - until MODEL_PATH exists, this
raises ModelNotTrainedError so the app can fall back to the classical
detector with a clear message instead of crashing.
"""
import numpy as np

from .calibration import pixels_to_mm
from .config import DATA_DIR
from .detection import Detection, _principal_angle_deg

MODEL_DIR = DATA_DIR / "rfdetr_model"
MODEL_PATH = MODEL_DIR / "checkpoint_best_ema.pth"
CONFIDENCE_THRESHOLD = 0.5

_cached_model = None


class ModelNotTrainedError(RuntimeError):
    pass


def is_trained() -> bool:
    return MODEL_PATH.exists()


def _get_model():
    global _cached_model
    if _cached_model is None:
        from rfdetr import RFDETR  # deferred: only needed in this mode
        _cached_model = RFDETR.from_checkpoint(str(MODEL_PATH))
    return _cached_model


def detect_object(frame: np.ndarray, homography: np.ndarray) -> Detection:
    if not is_trained():
        raise ModelNotTrainedError(
            "No fine-tuned RF-DETR model yet - collect training samples on "
            "the Training Data page and run the training script first."
        )
    model = _get_model()
    predictions = model.predict(frame, threshold=CONFIDENCE_THRESHOLD)

    if len(predictions) == 0:
        from .detection import NoObjectFoundError
        raise NoObjectFoundError("RF-DETR found no object on the bed.")

    best = max(range(len(predictions)), key=lambda i: predictions.confidence[i])
    mask = predictions.mask[best]  # boolean HxW mask, per supervision.Detections convention

    import cv2
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_px = contour.reshape(-1, 2).astype(np.float64)

    moments = cv2.moments(contour)
    centroid_px = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    contour_mm = pixels_to_mm(homography, contour_px)
    centroid_mm_arr = pixels_to_mm(homography, np.array([centroid_px]))
    centroid_mm = (float(centroid_mm_arr[0][0]), float(centroid_mm_arr[0][1]))
    angle_deg = _principal_angle_deg(contour_mm)

    return Detection(
        contour_px=contour_px,
        contour_mm=contour_mm,
        centroid_px=centroid_px,
        centroid_mm=centroid_mm,
        angle_deg=angle_deg,
    )
