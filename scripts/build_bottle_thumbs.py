"""Crop one representative photo per detector class from the training set.

The web app needs a picture for each bottle it can name - on the shelf-detection
card and on the bottle's own info page. Rather than sourcing product shots (a
licensing chore for 112 brands), crop the bounding box out of whichever training
photo shows that bottle biggest: since the label is only the box quality lets us
tell a clean close-up from a tiny, angled shelf sighting.

Run:  python scripts/build_bottle_thumbs.py
      python scripts/build_bottle_thumbs.py --dataset dataset_v2 --out app/static/bottles
"""
from __future__ import annotations

import argparse
import os

import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAD = 0.08          # a little breathing room around the label
MAX_SIDE = 640       # no need for the full-res source in a UI thumbnail


def best_boxes(dataset: str):
    """class index -> (area, split, image_stem, box) for the biggest sighting of each class."""
    data = yaml.safe_load(open(os.path.join(dataset, "data.yaml"), encoding="utf-8"))
    names = data["names"]
    best = {}
    for split in ("train", "valid", "test"):
        labels_dir = os.path.join(dataset, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for fn in os.listdir(labels_dir):
            if not fn.endswith(".txt"):
                continue
            stem = fn[:-4]
            for line in open(os.path.join(labels_dir, fn), encoding="utf-8"):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                area = w * h
                cur = best.get(cls)
                if cur is None or area > cur[0]:
                    best[cls] = (area, split, stem, (cx, cy, w, h))
    return names, best


def crop(dataset: str, split: str, stem: str, box):
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(dataset, split, "images", stem + ext)
        if os.path.exists(path):
            break
    else:
        return None
    im = Image.open(path).convert("RGB")
    W, H = im.size
    cx, cy, w, h = box
    bw, bh = w * W, h * H
    x1, y1 = cx * W - bw / 2, cy * H - bh / 2
    x2, y2 = x1 + bw, y1 + bh
    px, py = bw * PAD, bh * PAD
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(W, x2 + px), min(H, y2 + py)
    crop = im.crop((int(x1), int(y1), int(x2), int(y2)))
    if max(crop.size) > MAX_SIDE:
        crop.thumbnail((MAX_SIDE, MAX_SIDE))
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset_v2"))
    ap.add_argument("--out", default=os.path.join(ROOT, "app", "static", "bottles"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names, best = best_boxes(args.dataset)
    written = 0
    for cls, name in enumerate(names):
        hit = best.get(cls)
        if hit is None:
            print(f"  no boxes at all for class {cls} ({name})")
            continue
        area, split, stem, box = hit
        im = crop(args.dataset, split, stem, box)
        if im is None:
            print(f"  image missing for {name} ({split}/{stem})")
            continue
        im.save(os.path.join(args.out, f"{name}.jpg"), quality=90)
        written += 1
    print(f"wrote {written}/{len(names)} bottle thumbnails to {args.out}")


if __name__ == "__main__":
    main()
