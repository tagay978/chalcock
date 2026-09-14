"""Pre-annotate harvested images with the trained detector, for human correction.

These are draft boxes, not ground truth. Harvested photos are mostly other people's bars, so
many bottles will not be any of the 50 classes and the model will still guess one - every file
this writes has to be reviewed before it goes near the training set.

Output is a Roboflow/YOLO-shaped folder that annotation tools import directly:

    harvest/prelabelled/
      images/*.jpg
      labels/*.txt          (empty file = model found nothing; keep it, review it)
      data.yaml
      review.csv            per-image detection count and mean confidence, worst first

Run:  python scripts/pseudo_label.py --weights runs/yolo11s/weights/best.pt --conf 0.5
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "harvest", "images")
OUT = os.path.join(ROOT, "harvest", "prelabelled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "yolo11s", "weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.5,
                    help="higher than usual: a missed bottle is cheaper to add than a wrong label is to spot")
    ap.add_argument("--source", default=SRC)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(f"weights not found: {args.weights}")
    files = sorted(f for f in os.listdir(args.source)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"no images in {args.source}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names

    for sub in ("images", "labels"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    review = []
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        paths = [os.path.join(args.source, f) for f in batch]
        for fn, result in zip(batch, model.predict(paths, conf=args.conf, verbose=False)):
            stem = os.path.splitext(fn)[0]
            shutil.copy2(os.path.join(args.source, fn), os.path.join(OUT, "images", fn))
            boxes = result.boxes
            lines, confs, labels = [], [], []
            for cls, conf, xywhn in zip(boxes.cls.tolist(), boxes.conf.tolist(),
                                        boxes.xywhn.tolist()):
                x, y, w, h = xywhn
                lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                confs.append(conf)
                labels.append(names[int(cls)])
            with open(os.path.join(OUT, "labels", stem + ".txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            review.append({
                "file": fn,
                "detections": len(lines),
                "mean_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
                "min_conf": round(min(confs), 3) if confs else 0.0,
                "classes": "|".join(sorted(set(labels))),
            })
        print(f"  labelled {min(i + 16, len(files))}/{len(files)}", end="\r")

    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": OUT.replace("\\", "/"), "train": "images", "val": "images",
                        "nc": len(names), "names": [names[i] for i in range(len(names))]},
                       fh, sort_keys=False, allow_unicode=True)

    # Worst first: zero detections and low confidence are where the model is guessing.
    review.sort(key=lambda r: (r["detections"] > 0, r["mean_conf"]))
    with open(os.path.join(OUT, "review.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "detections", "mean_conf", "min_conf", "classes"])
        writer.writeheader()
        writer.writerows(review)

    total = sum(r["detections"] for r in review)
    empty = sum(1 for r in review if r["detections"] == 0)
    print(f"\nwrote {OUT}")
    print(f"  {len(files)} images, {total} draft boxes, {empty} with nothing detected")
    print(f"  review worst-first in {os.path.join(OUT, 'review.csv')}")
    print("\nThese labels are guesses. Correct them before merging into dataset/.")


if __name__ == "__main__":
    main()
