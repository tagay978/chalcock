"""Train a bottle detector on the merged dataset built by scripts/build_dataset.py.

The dataset is small (~1000 train images over 50 classes), so this leans on COCO-pretrained
weights and keeps the default mosaic/HSV augmentation, turning mosaic off for the last few
epochs so the model finishes on undistorted images.

Run:  python scripts/train.py                      # yolo11s, 200 epochs
      python scripts/train.py --model yolo11m.pt --epochs 300
"""
from __future__ import annotations

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset", "data.yaml")
WEIGHTS = os.path.join(ROOT, "weights")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA,
                    help="dataset yaml; use dataset_abstain/data.yaml for the 51-class model")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    ap.add_argument("--save-period", type=int, default=20,
                    help="also write weights/epochN.pt every N epochs, so intermediate models "
                         "can be tested; -1 keeps only last and best")
    ap.add_argument("--cache", default="",
                    help="'ram' or 'disk'; off by default because caching camera originals in "
                         "RAM overflows the 32-bit offset buffer")
    args = ap.parse_args()

    from ultralytics import YOLO

    # Prefer the locally staged weights: the GitHub release CDN is unreliable from here.
    local = os.path.join(WEIGHTS, args.model)
    model = YOLO(local if os.path.exists(local) else args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        seed=0,
        cache=args.cache or False,
        save_period=args.save_period,
        patience=50,
        close_mosaic=15,
        project=os.path.join(ROOT, "runs"),
        name=args.name or os.path.splitext(args.model)[0],
        exist_ok=True,
        plots=True,
    )

    metrics = model.val(split="test")
    print("\ntest mAP50-95:", round(float(metrics.box.map), 4))
    print("test mAP50   :", round(float(metrics.box.map50), 4))


if __name__ == "__main__":
    main()
