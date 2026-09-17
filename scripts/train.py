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
    ap.add_argument("--patience", type=int, default=50,
                    help="stop after this many epochs without a better validation score; "
                         "set it to --epochs to always run the full schedule")
    ap.add_argument("--save-period", type=int, default=20,
                    help="also write weights/epochN.pt every N epochs, so intermediate models "
                         "can be tested; -1 keeps only last and best")
    ap.add_argument("--aug", default="default", choices=["default", "strong"],
                    help="'strong' widens the scale range and adds rotation, shear and mixup; "
                         "training photos are mostly one bottle filling the frame while real "
                         "ones are small and off-axis, and scale is the gap that matters")
    ap.add_argument("--stop-on", default="fitness", choices=["fitness", "val_loss"],
                    help="what --patience counts against. Ultralytics counts epochs without a "
                         "better mAP; val_loss counts epochs without a lower validation loss")
    ap.add_argument("--cache", default="",
                    help="'ram' or 'disk'; off by default because caching camera originals in "
                         "RAM overflows the 32-bit offset buffer")
    args = ap.parse_args()

    from ultralytics import YOLO

    # 1 +/- scale is the zoom range, so 0.9 spans a tenth of the original size to nearly twice
    # it. Bottles are never upside down, so flipud stays off. copy_paste needs segmentation
    # masks and does nothing on a detection dataset, so it is left alone.
    AUG = {
        "default": {},
        "strong": dict(degrees=10.0, translate=0.2, scale=0.9, shear=2.0, perspective=0.0005,
                       flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1, hsv_v=0.5, erasing=0.4),
    }

    # Prefer the locally staged weights: the GitHub release CDN is unreliable from here.
    local = os.path.join(WEIGHTS, args.model)
    model = YOLO(local if os.path.exists(local) else args.model)

    if args.stop_on == "val_loss":
        # Ultralytics' own patience watches fitness, a weighted mAP. Watching the validation
        # loss instead needs a callback: it stops the run by setting trainer.stop, which the
        # epoch loop checks. Ultralytics' counter is pushed out of the way so only this fires.
        watch = {"best": float("inf"), "since": 0, "at": 0}

        def stop_on_val_loss(trainer):
            losses = [v for k, v in (trainer.metrics or {}).items()
                      if k.startswith("val/") and k.endswith("_loss")]
            if not losses:
                return
            total = float(sum(losses))
            if total < watch["best"] - 1e-4:
                watch.update(best=total, since=0, at=trainer.epoch + 1)
            else:
                watch["since"] += 1
                if watch["since"] >= args.patience:
                    trainer.stop = True
                    print(f"\nval loss has not improved for {args.patience} epochs "
                          f"(best {watch['best']:.4f} at epoch {watch['at']}); stopping")

        model.add_callback("on_fit_epoch_end", stop_on_val_loss)

    model.train(
        data=args.data,
        **AUG[args.aug],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        seed=0,
        cache=args.cache or False,
        save_period=args.save_period,
        patience=args.epochs if args.stop_on == "val_loss" else args.patience,
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
