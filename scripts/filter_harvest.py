"""Drop harvested images that do not actually show bottles.

Commons and Openverse match on text metadata, so a search for "liquor bottle shelf" also returns
magazine covers, landscapes and book bindings. Rather than eyeball them, run a COCO-pretrained
detector and keep only images containing at least --min-bottles of COCO class 'bottle'. This is
a relevance filter, not an annotation step - the brand labels still have to be drawn by hand.

Rejects are moved to harvest/rejected/ instead of deleted, so a bad threshold is recoverable.

Run:  python scripts/filter_harvest.py --min-bottles 2
      python scripts/filter_harvest.py --restore      # move everything back and start over
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARVEST = os.path.join(ROOT, "harvest")
IMG_DIR = os.path.join(HARVEST, "images")
REJECT_DIR = os.path.join(HARVEST, "rejected")
MANIFEST = os.path.join(HARVEST, "attribution.csv")
COCO_BOTTLE = 39


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    ap.add_argument("--min-bottles", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    os.makedirs(REJECT_DIR, exist_ok=True)

    if args.restore:
        moved = 0
        for fn in os.listdir(REJECT_DIR):
            shutil.move(os.path.join(REJECT_DIR, fn), os.path.join(IMG_DIR, fn))
            moved += 1
        print(f"restored {moved} images to {IMG_DIR}")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print("nothing to filter")
        return

    counts = {}
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        for fn, result in zip(batch, model.predict([os.path.join(IMG_DIR, f) for f in batch],
                                                   conf=args.conf, verbose=False)):
            counts[fn] = sum(1 for c in result.boxes.cls.tolist() if int(c) == COCO_BOTTLE)
        print(f"  scanned {min(i + 16, len(files))}/{len(files)}", end="\r")

    kept = {fn for fn, n in counts.items() if n >= args.min_bottles}
    for fn, n in sorted(counts.items()):
        if fn not in kept:
            shutil.move(os.path.join(IMG_DIR, fn), os.path.join(REJECT_DIR, fn))

    # Keep the manifest honest: attribution should describe what is actually in images/.
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
            fields = list(rows[0].keys()) if rows else []
        if rows:
            for row in rows:
                row["bottles_detected"] = counts.get(row["file"], "")
                row["kept"] = "yes" if row["file"] in kept else "no"
            with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields + ["bottles_detected", "kept"])
                writer.writeheader()
                writer.writerows(rows)

    print(f"\nkept {len(kept)}/{len(files)} images with >= {args.min_bottles} bottles")
    print(f"  moved {len(files) - len(kept)} to {REJECT_DIR}")
    hist = {}
    for n in counts.values():
        hist[n] = hist.get(n, 0) + 1
    print("  bottles per image:", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
