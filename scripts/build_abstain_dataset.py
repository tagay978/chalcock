"""Add an `unknown_bottle` class so the detector can decline to name a bottle.

The trained set contains 50 brands and nothing else, so the model has no place to put a bottle
it does not recognise and must name one of the 50. On the real-world test set 90% of bottles are
outside the vocabulary, and 7.4% of them draw a brand label - which is what makes the
recommender propose drinks from bottles that are not on the shelf.

Negatives are added as **close-up crops**, not whole shelf photos. Every positive in the trained
set is a close-up (median box covers 20% of the frame) while harvested shelf photos are wide
(median 1.2%), so pasting scenes in wholesale would let the model separate the classes by
apparent size instead of by label - it would learn "small bottle means unknown" and carry that
straight into inference. Cropping each negative bottle to fill the frame removes the shortcut and
matches how --two-stage queries the model at inference time.

Test-set images are excluded by filename, otherwise the evaluation measures memorisation.

Run:  python scripts/build_abstain_dataset.py
      python scripts/build_abstain_dataset.py --max-negatives 800 --scenes
"""
from __future__ import annotations

import argparse
import os
import random
import shutil

import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DATA = os.path.join(ROOT, "dataset")
HARVEST = os.path.join(ROOT, "harvest", "images")
TESTSET = os.path.join(ROOT, "testset", "images")
OUT = os.path.join(ROOT, "dataset_abstain")
UNKNOWN = "unknown_bottle"
COCO_BOTTLE = 39
PAD = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-negatives", type=int, default=1200, help="cap on negative crops")
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--min-crop", type=int, default=48, help="reject crops smaller than this")
    ap.add_argument("--scenes", action="store_true",
                    help="also add whole shelf photos; off by default to avoid a size shortcut")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(SRC_DATA):
        raise SystemExit("run scripts/build_dataset.py first")

    meta = yaml.safe_load(open(os.path.join(SRC_DATA, "data.yaml"), encoding="utf-8"))
    names = list(meta["names"])
    if UNKNOWN in names:
        raise SystemExit(f"{UNKNOWN} already present in dataset/data.yaml")
    unknown_id = len(names)
    names.append(UNKNOWN)

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    counts = {}
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(OUT, split, sub), exist_ok=True)
        src = os.path.join(SRC_DATA, split)
        if not os.path.isdir(src):
            continue
        for sub in ("images", "labels"):
            for fn in os.listdir(os.path.join(src, sub)):
                shutil.copy2(os.path.join(src, sub, fn), os.path.join(OUT, split, sub, fn))
        counts[split] = len(os.listdir(os.path.join(OUT, split, "images")))
    print("copied positives:", counts)

    held_out = {os.path.splitext(f)[0] for f in os.listdir(TESTSET)} if os.path.isdir(TESTSET) else set()
    pool = [f for f in sorted(os.listdir(HARVEST))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.splitext(f)[0] not in held_out]
    print(f"negative source: {len(pool)} harvested photos "
          f"({len(held_out)} test-set images excluded)")
    if not pool:
        raise SystemExit("no harvested images available - run scripts/harvest_images.py")

    from ultralytics import YOLO

    model = YOLO(args.model)
    rng = random.Random(args.seed)
    made = {"train": 0, "valid": 0}
    scenes = {"train": 0, "valid": 0}

    for i in range(0, len(pool), 16):
        if sum(made.values()) >= args.max_negatives:
            break
        batch = pool[i:i + 16]
        paths = [os.path.join(HARVEST, f) for f in batch]
        for fn, result in zip(batch, model.predict(paths, conf=args.conf, verbose=False)):
            if sum(made.values()) >= args.max_negatives:
                break
            stem = os.path.splitext(fn)[0]
            image = Image.open(os.path.join(HARVEST, fn)).convert("RGB")
            W, H = image.size
            boxes = [xyxy for cls, xyxy in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist())
                     if int(cls) == COCO_BOTTLE]

            for j, (x1, y1, x2, y2) in enumerate(boxes):
                if sum(made.values()) >= args.max_negatives:
                    break
                px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
                box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
                       min(W, int(x2 + px)), min(H, int(y2 + py)))
                if box[2] - box[0] < args.min_crop or box[3] - box[1] < args.min_crop:
                    continue
                split = "valid" if rng.random() < args.valid_frac else "train"
                crop = image.crop(box)
                name = f"neg_{stem}_{j:02d}"
                crop.save(os.path.join(OUT, split, "images", name + ".jpg"), quality=92)
                # The bottle fills the crop by construction, minus the padding.
                cw, ch = crop.size
                bw = (x2 - x1) / cw
                bh = (y2 - y1) / ch
                cx = (x1 - box[0] + (x2 - x1) / 2) / cw
                cy = (y1 - box[1] + (y2 - y1) / 2) / ch
                with open(os.path.join(OUT, split, "labels", name + ".txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write(f"{unknown_id} {cx:.6f} {cy:.6f} "
                             f"{min(bw, 1.0):.6f} {min(bh, 1.0):.6f}\n")
                made[split] += 1

            if args.scenes and boxes:
                split = "valid" if rng.random() < args.valid_frac else "train"
                shutil.copy2(os.path.join(HARVEST, fn),
                             os.path.join(OUT, split, "images", "scene_" + fn))
                with open(os.path.join(OUT, split, "labels", "scene_" + stem + ".txt"), "w",
                          encoding="utf-8") as fh:
                    for x1, y1, x2, y2 in boxes:
                        fh.write(f"{unknown_id} {((x1 + x2) / 2) / W:.6f} {((y1 + y2) / 2) / H:.6f} "
                                 f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
                scenes[split] += 1

    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": OUT.replace("\\", "/"), "train": "train/images",
                        "val": "valid/images", "test": "test/images",
                        "nc": len(names), "names": names},
                       fh, sort_keys=False, allow_unicode=True)

    total_pos = sum(counts.values())
    total_neg = sum(made.values()) + sum(scenes.values())
    print(f"\nwrote {OUT}")
    print(f"  negative crops: train {made['train']}, valid {made['valid']}")
    if args.scenes:
        print(f"  negative scenes: train {scenes['train']}, valid {scenes['valid']}")
    print(f"  {len(names)} classes, '{UNKNOWN}' is id {unknown_id}")
    print(f"  positives {total_pos}, negatives {total_neg} "
          f"({100 * total_neg / (total_pos + total_neg):.0f}% of images)")
    print("\nNegative labels are assumed, not verified: a harvested bottle that happens to be")
    print("one of the 50 is being taught as 'unknown'. Measured base rate for that is ~10%.")


if __name__ == "__main__":
    main()
