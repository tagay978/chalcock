"""Merge the 50 per-class Roboflow exports in finalloopy/ into one multi-class YOLO dataset.

Each source folder was labelled independently: the *same* photo can appear in several
folders, each time carrying only that folder's bottle as a box. Merging therefore has to
(1) recognise when two files are the same photo and (2) take the union of their boxes,
otherwise every shared photo teaches the model that the other bottles are background.

Run:  python scripts/build_dataset.py --report      # inspect only, writes nothing
      python scripts/build_dataset.py               # build dataset/
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

import yaml
from PIL import Image
import imagehash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "finalloopy")
OUT = os.path.join(ROOT, "dataset")
SPLITS = ("train", "valid", "test")
SPLIT_RANK = {"train": 0, "valid": 1, "test": 2}   # tie-break order
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PHASH_MAX_DIST = 2        # re-encoded copies differ by a bit or two, distinct bottles do not
IOU_SAME_BOX = 0.6        # boxes above this, with the same class, are the same annotation


def class_folders():
    return sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))


def id_to_class(folder, folders):
    """Map the label ids used inside one export to canonical class names.

    data.yaml names are unreliable (several are bare numbers like '56'), so a single-class
    export is keyed to its folder name. Multi-class exports must name real folders.
    """
    ypath = os.path.join(SRC, folder, "data.yaml")
    if not os.path.exists(ypath):
        return {0: folder}
    meta = yaml.safe_load(open(ypath, encoding="utf-8")) or {}
    names = meta.get("names") or []
    if len(names) < 2:
        return {0: folder}
    mapping = {}
    for i, raw in enumerate(names):
        cand = str(raw).replace("_", "").replace("-", "").lower()
        if cand not in folders:
            raise SystemExit(
                f"{folder}/data.yaml lists class '{raw}' (-> '{cand}') with no matching folder; "
                "add an explicit mapping before merging."
            )
        mapping[i] = cand
    return mapping


def read_boxes(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append((int(float(parts[0])), *(float(v) for v in parts[1:5])))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[3] * a[4] + b[3] * b[4] - inter)


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def scan():
    """Collect every source image with its boxes already remapped to canonical class names."""
    folders = class_folders()
    fset = set(folders)
    records, skipped = [], Counter()
    for folder in folders:
        mapping = id_to_class(folder, fset)
        for split in SPLITS:
            idir = os.path.join(SRC, folder, split, "images")
            ldir = os.path.join(SRC, folder, split, "labels")
            if not os.path.isdir(idir):
                continue
            for fn in sorted(os.listdir(idir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in IMG_EXT:
                    continue
                ipath = os.path.join(idir, fn)
                boxes = []
                for cid, *xywh in read_boxes(os.path.join(ldir, stem + ".txt")):
                    if cid not in mapping:
                        skipped[f"{folder}: unknown id {cid}"] += 1
                        continue
                    boxes.append((mapping[cid], *xywh))
                try:
                    with Image.open(ipath) as im:
                        im = im.convert("RGB")
                        size, ph = im.size, imagehash.phash(im)
                except Exception as exc:  # unreadable export
                    skipped[f"unreadable: {exc.__class__.__name__}"] += 1
                    continue
                records.append({
                    "path": ipath, "folder": folder, "split": split, "stem": stem, "ext": ext,
                    "boxes": boxes, "size": size, "phash": ph,
                    "md5": hashlib.md5(open(ipath, "rb").read()).hexdigest(),
                })
    return records, skipped


def group(records):
    """Union records that are the same photo: identical bytes, or a near-identical pHash."""
    u = Union()
    by_md5 = defaultdict(list)
    for i, r in enumerate(records):
        u.find(i)
        by_md5[r["md5"]].append(i)
    for idxs in by_md5.values():
        for j in idxs[1:]:
            u.join(idxs[0], j)

    exact_groups = sum(1 for v in by_md5.values() if len(v) > 1)

    # pHash buckets by image size keep the pairwise comparison small and avoid matching
    # a 640x640 export against a thumbnail of something else.
    near = 0
    by_size = defaultdict(list)
    for i, r in enumerate(records):
        by_size[r["size"]].append(i)
    for idxs in by_size.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if u.find(i) == u.find(j):
                    continue
                if records[i]["phash"] - records[j]["phash"] <= PHASH_MAX_DIST:
                    u.join(i, j)
                    near += 1

    groups = defaultdict(list)
    for i in range(len(records)):
        groups[u.find(i)].append(i)
    return list(groups.values()), exact_groups, near


def merge_boxes(members, records):
    merged = []
    for i in members:
        for box in records[i]["boxes"]:
            if any(box[0] == m[0] and iou(box, m) > IOU_SAME_BOX for m in merged):
                continue
            merged.append(box)
    return merged


def choose_split(members, records):
    votes = Counter(records[i]["split"] for i in members)
    top = max(votes.values())
    return sorted((s for s, n in votes.items() if n == top), key=lambda s: SPLIT_RANK[s])[0]


def sanitize(stem):
    stem = re.sub(r"\.rf\.[0-9a-f]{32}", "", stem, flags=re.I)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return stem[:60] or "img"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="analyse only, write nothing")
    args = ap.parse_args()

    if not os.path.isdir(SRC):
        sys.exit(f"source not found: {SRC}")

    print("scanning", SRC)
    records, skipped = scan()
    names = sorted({b[0] for r in records for b in r["boxes"]} | set(class_folders()))
    print(f"  {len(records)} source images, {sum(len(r['boxes']) for r in records)} boxes, "
          f"{len(names)} classes")
    for k, v in skipped.items():
        print(f"  skipped {v}x {k}")

    groups, exact, near = group(records)
    multi = [g for g in groups if len({records[i]["folder"] for i in g}) > 1]
    conflict = [g for g in groups if len({records[i]["split"] for i in g}) > 1]
    print(f"  {len(groups)} unique photos after dedup "
          f"({len(records) - len(groups)} duplicate files removed)")
    print(f"    byte-identical groups: {exact}, near-identical (pHash) merges: {near}")
    print(f"    photos shared across classes: {len(multi)}")
    print(f"    split conflicts resolved by majority: {len(conflict)}")
    for g in multi[:10]:
        folders = sorted({records[i]["folder"] for i in g})
        print(f"      {len(merge_boxes(g, records))} boxes <- {folders}")

    if args.report:
        return

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for split in SPLITS:
        os.makedirs(os.path.join(OUT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT, split, "labels"), exist_ok=True)

    cls_id = {n: i for i, n in enumerate(names)}
    per_split, per_class, empty = Counter(), Counter(), 0
    manifest = [("file", "split", "n_boxes", "classes", "sources")]

    for members in groups:
        rep = min(members, key=lambda i: (SPLIT_RANK[records[i]["split"]], records[i]["path"]))
        r = records[rep]
        split = choose_split(members, records)
        boxes = merge_boxes(members, records)
        name = f"{r['md5'][:12]}_{sanitize(r['stem'])}"
        shutil.copy2(r["path"], os.path.join(OUT, split, "images", name + r["ext"].lower()))
        with open(os.path.join(OUT, split, "labels", name + ".txt"), "w", encoding="utf-8") as fh:
            for c, x, y, w, h in boxes:
                fh.write(f"{cls_id[c]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        per_split[split] += 1
        for c, *_ in boxes:
            per_class[c] += 1
        if not boxes:
            empty += 1
        manifest.append((
            name + r["ext"].lower(), split, len(boxes),
            "|".join(sorted({c for c, *_ in boxes})),
            "|".join(sorted(records[i]["folder"] for i in members)),
        ))

    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"path": OUT.replace("\\", "/"), "train": "train/images", "val": "valid/images",
             "test": "test/images", "nc": len(names), "names": names},
            fh, sort_keys=False, allow_unicode=True,
        )
    with open(os.path.join(OUT, "manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(manifest)

    print(f"\nwrote {OUT}")
    print("  splits:", dict(per_split), f"(images with no box: {empty})")
    thin = [f"{c}:{per_class[c]}" for c in names if per_class[c] < 15]
    print(f"  classes: {len(names)}, boxes: {sum(per_class.values())}")
    if thin:
        print("  under 15 boxes:", ", ".join(thin))


if __name__ == "__main__":
    main()
