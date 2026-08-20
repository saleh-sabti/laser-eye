"""Fine-tunes RF-DETR on the samples collected via the Training Data page.

Run this from the project root once you've collected enough samples (the
Training Data page shows your current count - 50-200+ is the real target,
though it'll refuse below laser_align.dataset.MIN_SAMPLES_TO_TRAIN):

    C:\\venvs\\laser\\Scripts\\python train_rfdetr.py

On success, the best checkpoint is copied to
calibration_data/rfdetr_model/checkpoint_best_ema.pth, which is exactly
where laser_align/rfdetr_detect.py looks for it - after this finishes, just
switch Settings -> Detection method to "RF-DETR" to start using it (the app
auto-falls-back to classical if that file isn't present, so this is safe to
run repeatedly / re-run after collecting more data without breaking
anything in the meantime).

Uses the smallest segmentation variant (RFDETRSegNano) since the dataset
here is necessarily small (a personal one-bed-one-camera setup, not a
large-scale corpus) - a bigger model would just overfit faster on little
data, and this needs to run on a single consumer GPU in a reasonable time.
"""
import sys

from laser_align import dataset
from laser_align.config import DATA_DIR

OUTPUT_DIR = DATA_DIR / "rfdetr_model"


def main() -> None:
    print(f"Collected samples: {dataset.sample_count()}")
    try:
        export_dir = dataset.export_for_training()
    except ValueError as e:
        print(f"Not ready to train yet: {e}")
        sys.exit(1)
    print(f"Exported train/valid split to {export_dir}")

    from rfdetr import RFDETRSegNano

    model = RFDETRSegNano()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.train(
        dataset_dir=str(export_dir),
        dataset_file="roboflow",
        output_dir=str(OUTPUT_DIR),
        epochs=100,
        batch_size=4,
    )

    produced = OUTPUT_DIR / "checkpoint_best_ema.pth"
    if not produced.exists():
        print(f"Training finished but no checkpoint found at {produced} - check the output above.")
        sys.exit(1)
    print(f"Done. Model ready at {produced}")
    print('Switch Settings -> Detection method to "RF-DETR" to start using it.')


if __name__ == "__main__":
    main()
