"""Builds a labeled training set for fine-tuning RF-DETR, using the
classical background-subtraction detector as an auto-labeler.

Every sample is a real bed photo plus the outline our own detector found
for it, written out in COCO instance-segmentation format (images/ +
annotations.json) - the format RF-DETR's training script expects. This
means collecting data is just "put a piece down, click save" repeated many
times across different pieces/positions/lighting, with no manual polygon
drawing required. The trained model's quality is capped by how good the
classical detector's outlines already are, but it can still end up more
robust than the classical detector to lighting drift, since it learns
general wood-vs-mat visual cues across many real conditions instead of
diffing against one fixed reference photo.
"""
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR
from .detection import Detection

DATASET_DIR = DATA_DIR / "training_data"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_PATH = DATASET_DIR / "annotations.json"
EXPORT_DIR = DATASET_DIR / "rfdetr_export"

CATEGORY_ID = 1
CATEGORY_NAME = "wood"
MIN_SAMPLES_TO_TRAIN = 20  # a soft floor - 50-200+ is the real recommended target


def _load_annotations() -> dict:
    if ANNOTATIONS_PATH.exists():
        return json.loads(ANNOTATIONS_PATH.read_text())
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": "object"}],
    }


def _save_annotations(data: dict) -> None:
    ANNOTATIONS_PATH.write_text(json.dumps(data, indent=2))


def sample_count() -> int:
    return len(_load_annotations()["images"])


def add_sample(frame: np.ndarray, detection: Detection) -> int:
    """Save `frame` plus `detection`'s outline as one training sample.
    Returns the new total sample count.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_annotations()

    image_id = len(data["images"]) + 1
    filename = f"sample_{image_id:05d}.jpg"
    cv2.imwrite(str(IMAGES_DIR / filename), frame)

    h, w = frame.shape[:2]
    contour = detection.contour_px.astype(np.float64)
    x0, y0 = contour.min(axis=0)
    x1, y1 = contour.max(axis=0)
    segmentation = contour.flatten().tolist()

    data["images"].append({"id": image_id, "file_name": filename, "width": w, "height": h})
    data["annotations"].append({
        "id": image_id,
        "image_id": image_id,
        "category_id": CATEGORY_ID,
        "segmentation": [segmentation],
        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
        "area": float(cv2.contourArea(detection.contour_px.astype(np.float32))),
        "iscrowd": 0,
    })

    _save_annotations(data)
    return len(data["images"])


def dataset_dir() -> Path:
    return DATASET_DIR


def export_for_training(train_ratio: float = 0.85, seed: int = 0) -> Path:
    """Split the flat collected samples into RF-DETR's expected layout -
    Roboflow's COCO convention (`train/`, `valid/`, each with its own
    `_annotations.coco.json` and images alongside it, not in a subfolder) -
    and write that out fresh under EXPORT_DIR. Re-running this after
    collecting more samples just regenerates the split from scratch, so
    it's safe to call every time before training.
    """
    data = _load_annotations()
    images = data["images"]
    if len(images) < MIN_SAMPLES_TO_TRAIN:
        raise ValueError(
            f"Only {len(images)} samples collected - collect at least "
            f"{MIN_SAMPLES_TO_TRAIN} (50-200+ recommended) on the Training "
            f"Data page before training."
        )

    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * train_ratio))
    split_at = min(split_at, len(shuffled) - 1)  # keep at least 1 in valid
    splits = {"train": shuffled[:split_at], "valid": shuffled[split_at:]}

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    for split_name, split_images in splits.items():
        split_dir = EXPORT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_anns = []
        for img in split_images:
            shutil.copy2(IMAGES_DIR / img["file_name"], split_dir / img["file_name"])
            split_anns.extend(anns_by_image.get(img["id"], []))

        coco = {
            "images": split_images,
            "annotations": split_anns,
            "categories": data["categories"],
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))

    return EXPORT_DIR
