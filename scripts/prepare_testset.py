"""Build an annotation-ready real-world test set from harvested photos.

The point of this set is to measure what the training split cannot: whether the detector finds
bottles at shelf scale, and whether it can keep quiet about bottles outside its 50 classes.

Boxes are proposed by a COCO-pretrained detector, which is good at finding bottles, and every
proposal is written as class `unknown_bottle`. Annotation is then relabelling rather than
drawing: change the class on the bottles you recognise, delete false boxes, add anything missed.
Leaving a box as `unknown_bottle` is a real annotation, not a skip - it is what makes the
out-of-vocabulary measurement possible.

Run:  python scripts/prepare_testset.py
      python scripts/prepare_testset.py --limit 60 --min-bottles 3
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "harvest", "images")
OUT = os.path.join(ROOT, "testset")
UNKNOWN = "unknown_bottle"
COCO_BOTTLE = 39


def class_names():
    """The 50 trained classes, in their trained order, plus the out-of-vocabulary class."""
    data_yaml = os.path.join(ROOT, "dataset", "data.yaml")
    if os.path.exists(data_yaml):
        names = yaml.safe_load(open(data_yaml, encoding="utf-8"))["names"]
    else:
        names = sorted(yaml.safe_load(
            open(os.path.join(ROOT, "data", "bottles.yaml"), encoding="utf-8"))["bottles"])
    return list(names) + [UNKNOWN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-bottles", type=int, default=2,
                    help="skip photos with fewer proposed bottles than this")
    ap.add_argument("--max-bottles", type=int, default=25,
                    help="skip liquor-store walls; they cost hours to annotate for little signal")
    ap.add_argument("--limit", type=int, default=60, help="how many photos to include")
    ap.add_argument("--source", default=SRC)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.source)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"no images in {args.source} - run scripts/harvest_images.py first")

    from ultralytics import YOLO

    model = YOLO(args.model)
    names = class_names()
    unknown_id = len(names) - 1

    for sub in ("images", "labels"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    rows, kept, skipped = [], 0, {"too_few": 0, "too_many": 0}
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        paths = [os.path.join(args.source, f) for f in batch]
        for fn, result in zip(batch, model.predict(paths, conf=args.conf, verbose=False)):
            if kept >= args.limit:
                break
            boxes = [xywhn for cls, xywhn in zip(result.boxes.cls.tolist(),
                                                 result.boxes.xywhn.tolist())
                     if int(cls) == COCO_BOTTLE]
            if len(boxes) < args.min_bottles:
                skipped["too_few"] += 1
                continue
            if len(boxes) > args.max_bottles:
                skipped["too_many"] += 1
                continue

            stem = os.path.splitext(fn)[0]
            shutil.copy2(os.path.join(args.source, fn), os.path.join(OUT, "images", fn))
            with open(os.path.join(OUT, "labels", stem + ".txt"), "w", encoding="utf-8") as fh:
                for x, y, w, h in boxes:
                    fh.write(f"{unknown_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            rows.append({"file": fn, "proposed_boxes": len(boxes), "annotated": "no"})
            kept += 1
        print(f"  scanned {min(i + 16, len(files))}/{len(files)}, kept {kept}", end="\r")
        if kept >= args.limit:
            break

    # The test set is committed with its images, so it has to carry its own licence record
    # rather than depend on harvest/, which is regenerated.
    harvest_manifest = os.path.join(ROOT, "harvest", "attribution.csv")
    if os.path.exists(harvest_manifest):
        included = {r["file"] for r in rows}
        with open(harvest_manifest, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            attribution = [r for r in reader if r["file"] in included]
            fields = reader.fieldnames or []
        with open(os.path.join(OUT, "attribution.csv"), "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(attribution)
        missing = len(included) - len(attribution)
        if missing:
            print(f"\n  WARNING: {missing} images have no attribution row", flush=True)

    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": OUT.replace("\\", "/"), "train": "images", "val": "images",
                        "nc": len(names), "names": names},
                       fh, sort_keys=False, allow_unicode=True)
    # labelImg reads and rewrites classes.txt in the save directory when saving YOLO format.
    # Writing it up front pins the class order to the trained one, so ids stay comparable.
    with open(os.path.join(OUT, "labels", "classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(names) + "\n")

    with open(os.path.join(OUT, "progress.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "proposed_boxes", "annotated"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["proposed_boxes"] for r in rows)
    print(f"\nwrote {OUT}")
    print(f"  {kept} photos, {total} proposed boxes (all class '{UNKNOWN}', id {unknown_id})")
    print(f"  skipped: {skipped['too_few']} with < {args.min_bottles} bottles, "
          f"{skipped['too_many']} with > {args.max_bottles}")
    print(f"\n  {len(names)} classes: the 50 trained ones, in training order, plus '{UNKNOWN}'")
    print("\nAnnotate by relabelling, not drawing:")
    print("  - change the class on bottles you recognise as one of the 50")
    print(f"  - leave everything else as '{UNKNOWN}' (this is the point, not a skip)")
    print("  - delete boxes that are not bottles, add bottles the proposer missed")
    print("\nThen measure with: python scripts/eval_realworld.py")


if __name__ == "__main__":
    main()
