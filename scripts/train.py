"""Train a bottle detector on the merged dataset built by scripts/build_dataset.py.

The dataset is small (~1000 train images over 50 classes), so this leans on COCO-pretrained
weights and keeps the default mosaic/HSV augmentation, turning mosaic off for the last few
epochs so the model finishes on undistorted images.

There is no fixed schedule: --epochs is only a ceiling, and the run ends when the validation
score stops improving. On the 200-epoch run this replaced, mAP peaked at epoch 73 and the
remaining 127 epochs produced nothing.

Run:  python scripts/train.py --aug strong
      python scripts/train.py --model yolo11m.pt --patience 40
"""
from __future__ import annotations

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset", "data.yaml")
WEIGHTS = os.path.join(ROOT, "weights")


def make_stopper(stop_on, patience, min_epochs, mosaic_tail, ceiling, log=print):
    """The callback that decides when a run is finished.

    Ultralytics' own stopper has no floor, only watches fitness, and runs before the
    on_fit_epoch_end callbacks - which means a callback can overrule it. So it is pushed out of
    reach and the whole decision lives here.

    Stopping is not immediate: mosaic is turned off first and the run continues for a short
    tail, so the model finishes on undistorted images the way a fixed schedule with
    close_mosaic would have let it. The ceiling is left to ultralytics, so overruling its stop
    can never turn the epoch loop into an endless one.
    """
    watch = {"best": None, "since": 0, "at": 0, "tail": None}

    def score(trainer):
        if stop_on == "fitness":
            return None if trainer.fitness is None else float(trainer.fitness)
        losses = [v for k, v in (trainer.metrics or {}).items()
                  if k.startswith("val/") and k.endswith("_loss")]
        return -float(sum(losses)) if losses else None          # negated: higher is better

    def stopper(trainer):
        epoch = trainer.epoch + 1
        now = score(trainer)
        if now is None:
            return
        if watch["best"] is None or now > watch["best"] + 1e-4:
            watch.update(best=now, since=0, at=epoch)
        else:
            watch["since"] += 1

        if epoch >= ceiling:
            return                                              # the ceiling still ends the run

        if watch["tail"] is not None:
            trainer.stop = epoch >= watch["tail"]
            return

        trainer.stop = False                                    # overrule the built-in stopper
        if epoch < min_epochs or watch["since"] < patience:
            return

        shown = watch["best"] if stop_on == "fitness" else -watch["best"]
        log(f"\nno better {stop_on} for {patience} epochs "
            f"(best {shown:.4f} at epoch {watch['at']})")
        if mosaic_tail > 0:
            watch["tail"] = epoch + mosaic_tail
            log(f"closing mosaic and running {mosaic_tail} more epochs")
            trainer._close_dataloader_mosaic()
            trainer.train_loader.reset()
        else:
            trainer.stop = True

    stopper.watch = watch
    return stopper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA,
                    help="dataset yaml; use dataset_abstain/data.yaml for the 51-class model")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=400,
                    help="ceiling, not a schedule; early stopping decides when to finish")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=30,
                    help="finish after this many epochs without a better score. 30 because on "
                         "the 200-epoch run mAP peaked at epoch 73 and the longest gap between "
                         "new bests before that was well inside 30")
    ap.add_argument("--min-epochs", type=int, default=60,
                    help="never stop before this. mAP was already at 98%% of its final value by "
                         "epoch 34 but did not actually peak until 73, so an early plateau is "
                         "not a finished model")
    ap.add_argument("--mosaic-tail", type=int, default=15,
                    help="on deciding to stop, turn mosaic off and run this many more epochs so "
                         "the model finishes on undistorted images")
    ap.add_argument("--save-period", type=int, default=20,
                    help="also write weights/epochN.pt every N epochs, so intermediate models "
                         "can be tested; -1 keeps only last and best")
    ap.add_argument("--aug", default="default", choices=["default", "strong"],
                    help="'strong' widens the scale range and adds rotation, shear and mixup; "
                         "training photos are mostly one bottle filling the frame while real "
                         "ones are small and off-axis, and scale is the gap that matters")
    ap.add_argument("--stop-on", default="fitness", choices=["fitness", "val_loss", "none"],
                    help="what --patience counts: epochs without a better mAP (fitness, the "
                         "weighted mAP ultralytics ranks checkpoints by), or without a lower "
                         "validation loss. 'none' runs the full --epochs")
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

    if args.stop_on != "none":
        model.add_callback("on_fit_epoch_end",
                           make_stopper(args.stop_on, args.patience, args.min_epochs,
                                        args.mosaic_tail, args.epochs))


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
        patience=args.epochs,          # the callback above owns the decision
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
