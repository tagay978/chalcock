"""Stage the photos whose labels are lost to a source contradiction, ready to relabel.

ouo_final copies a photo into every brand folder it contains and renames every box to that
folder's brand, so one rectangle arrives carrying two or three names. build_dataset_ouo.py
drops those rectangles - there is no way to tell which source was right - and a photo that
loses all of them drops out of the dataset entirely.

The rectangles themselves are not in doubt, only the names, so this writes them out with the
geometry intact and the class set to _unsure. Retype the names in label_viewer or
class_review; the saved overlay lands in label_fixes/ under the same key the builder uses, so
the next rebuild picks the photo back up.

Anything still left as _unsure is not in the real class list, so a rebuild skips it and says
how many it skipped. A name nobody gets round to fixing can never leak into training.

Run:  python scripts/build_conflict_todo.py
      python scripts/label_viewer.py --dataset dataset_todo
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from collections import Counter

import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_dataset_ouo as B  # noqa: E402

OUT = os.path.join(ROOT, "dataset_todo")
UNSURE = "_unsure"
MAX_SIDE = 1280


def clusters_of(members, records):
    """The same union-and-cluster pass the builder runs, but keeping the losers."""
    merged = []
    for i in members:
        for box in records[i]["boxes"]:
            if any(box[0] == m[0] and B.iou(box, m) > B.IOU_SAME for m in merged):
                continue
            merged.append(box)
    clusters = []
    for box in merged:
        for c in clusters:
            if B.iou(box, c[0]) >= B.SAME_RECT:
                c.append(box)
                break
        else:
            clusters.append([box])
    return clusters


def main():
    skipped = Counter()
    print("scanning ouo_final ...")
    records = B.scan_ouo(skipped, "brand")
    ouo_names = {b[0] for r in records for b in r["boxes"]}
    records += B.scan_old(skipped, ouo_names, "brand")
    groups, _ = B.group(records)
    fixes = B.load_fixes()
    print(f"  {len(groups)} unique photos, {len(fixes)} already hand-corrected")

    names = yaml.safe_load(open(os.path.join(ROOT, "dataset_v2", "data.yaml"),
                                encoding="utf-8"))["names"]
    cls_id = {n: i for i, n in enumerate(names)}
    cls_id[UNSURE] = len(names)

    rows, staged = [], []
    for members in groups:
        clusters = clusters_of(members, records)
        bad = [c for c in clusters if len({b[0] for b in c}) > 1]
        if not bad:
            continue
        rep = min(members, key=lambda i: (records[i]["source"] != "ouo", records[i]["path"]))
        r = records[rep]
        ext = ".jpg" if r["ext"] in (".jpg", ".jpeg") else r["ext"]
        key = f"{r['md5'][:12]}_{B.sanitize(r['stem'])}{ext}"
        if key in fixes:
            continue                       # the overlay already replaces these boxes
        boxes = [(UNSURE if c in bad else c[0][0], *c[0][1:]) for c in clusters]
        staged.append((key, r["path"], boxes))
        rows.append({
            "file": key,
            "unsure_boxes": len(bad),
            "known_boxes": len(clusters) - len(bad),
            "candidate_classes": " ".join(sorted({b[0] for c in bad for b in c})),
            "source": os.path.relpath(r["path"], ROOT),
        })

    rows.sort(key=lambda d: -d["unsure_boxes"])
    staged.sort(key=lambda s: -sum(1 for b in s[2] if b[0] == UNSURE))

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    for key, src, boxes in staged:
        dest = os.path.join(OUT, "images", key)
        with Image.open(src) as im:
            if max(im.size) > MAX_SIDE:
                im = im.convert("RGB")
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                im.save(dest, quality=92)
            else:
                shutil.copy2(src, dest)
        stem = os.path.splitext(key)[0]
        with open(os.path.join(OUT, "labels", stem + ".txt"), "w", encoding="utf-8") as fh:
            for c, x, y, w, h in boxes:
                fh.write(f"{cls_id[c]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": OUT.replace("\\", "/"), "train": "images", "val": "images",
                        "nc": len(names) + 1, "names": names + [UNSURE]},
                       fh, sort_keys=False, allow_unicode=True)

    out_csv = os.path.join(ROOT, "conflicts_todo.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    unsure = sum(r["unsure_boxes"] for r in rows)
    known = sum(r["known_boxes"] for r in rows)
    print(f"\nwrote {OUT}")
    print(f"  {len(rows)} photos, {unsure} boxes to name, {known} boxes already certain")
    print(f"  listing: {out_csv}")


if __name__ == "__main__":
    main()
