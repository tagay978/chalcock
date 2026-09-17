"""Score every checkpoint of a run against the real-world test set.

`train.py --save-period 20` leaves weights/epoch20.pt, epoch40.pt and so on beside best.pt.
Validation mAP already says which epoch was best on data drawn from the training sessions; this
says which one is best on real photos, which is a different question and the one that matters.

Run:  python scripts/eval_checkpoints.py --run yolo11s_v10
      python scripts/eval_checkpoints.py --run yolo11s_v10 --conf 0.15,0.25,0.4
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_realworld as E  # noqa: E402


def checkpoints(run):
    wdir = os.path.join(ROOT, "runs", run, "weights")
    if not os.path.isdir(wdir):
        raise SystemExit(f"no such run: {wdir}")
    epochs = []
    for fn in os.listdir(wdir):
        m = re.fullmatch(r"epoch(\d+)\.pt", fn)
        if m:
            epochs.append((int(m.group(1)), os.path.join(wdir, fn)))
    epochs.sort()
    for name in ("best.pt", "last.pt"):
        p = os.path.join(wdir, name)
        if os.path.exists(p):
            epochs.append((name.replace(".pt", ""), p))
    return epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory under runs/")
    ap.add_argument("--conf", default="0.4")
    ap.add_argument("--level", default="ingredient", choices=["ingredient", "brand"])
    ap.add_argument("--path", default="two-stage", choices=["two-stage", "direct"])
    ap.add_argument("--coco-weights", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    coco = YOLO(args.coco_weights) if args.path == "two-stage" else None
    names = yaml.safe_load(open(os.path.join(E.TESTSET, "data.yaml"), encoding="utf-8"))["names"]
    images = os.path.join(E.TESTSET, "images")
    confs = [float(c) for c in args.conf.split(",")]

    points = checkpoints(args.run)
    print(f"{args.run}: {len(points)} checkpoints, {args.path}, {args.level} level\n")
    print(f"  {'epoch':>7} {'conf':>5} {'found':>9} {'correct':>10} {'acc':>7} {'falsealarm':>11}")

    for tag, path in points:
        brand = YOLO(path)
        model_names = [brand.names[i] for i in range(len(brand.names))]
        gt_key = E.make_key(names, args.level)
        pred_key = E.make_key(model_names, args.level)
        emittable = {pred_key(i) for i in range(len(model_names))} - {E.canon(E.UNKNOWN)}
        truth = E.load_truth(names, lambda cid: gt_key(cid) in emittable)
        n_known = sum(len(k) for k, _ in truth.values())
        for conf in confs:
            preds = {}
            for fn in truth:
                image = os.path.join(images, fn)
                preds[fn] = (E.predict_two_stage(coco, brand, image, conf) if args.path == "two-stage"
                             else E.predict_direct(brand, image, conf))
            s = E.evaluate(truth, preds, None, gt_key, pred_key)
            acc = s["correct"] / s["localised"] if s["localised"] else 0
            fa = s["unknown_hit"] / s["unknown"] if s["unknown"] else 0
            print(f"  {str(tag):>7} {conf:>5} {s['localised']:>5}/{n_known:<3} "
                  f"{s['correct']:>6}/{n_known:<3} {acc:>6.1%} {fa:>10.1%}")

    print("\nA later epoch winning on validation does not have to win here: validation photos come"
          "\nfrom the same sessions as training, real photos do not.")


if __name__ == "__main__":
    main()
