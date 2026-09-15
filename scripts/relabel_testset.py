"""Move the real-world test set onto the 119-class vocabulary.

The set was annotated against the old 50 classes, so 69 of the current model's outputs cannot be
credited even when they are right, and a correct naming counts as a false alarm because the
annotation says `unknown_bottle`. 229 of its 255 boxes carry that label and many are bottles the
model now knows.

Two things this deliberately does NOT do:

- **It does not pre-fill from the model under evaluation.** Seeding the labels with its own
  predictions and asking someone to confirm them turns the test set into a record of what the
  model already says, and the measured accuracy inflates to match. Every unnamed box stays
  `unknown_bottle`; naming it is the annotator's call.
- **It does not touch the boxes.** Their geometry was reviewed by hand once already.

What it does: rewrite `classes.txt` to the model's vocabulary plus `unknown_bottle`, renumber the
existing annotations onto it, and build a visual reference of all 119 classes - which is the
actual bottleneck, since nobody recognises 119 spirit brands by name alone.

Run:  python scripts/relabel_testset.py
      python scripts/relabel_testset.py --report
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import Counter

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET = os.path.join(ROOT, "testset")
DATA_V2 = os.path.join(ROOT, "dataset_v2")
UNKNOWN = "unknown_bottle"

# Same three the dataset builder needed, plus the two spellings the annotator used that
# ouo_final writes differently. Without these the boxes would be thrown back to unknown and
# the annotation work on them lost.
ALIAS = {
    "orangebitters": "orange",
    "gilbeygin": "gilbeysgin",
    "wildturkeykrye": "wildturkey",
    "grandmanier": "grandmarnier",
}


def canon(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_names(path, key="names"):
    return yaml.safe_load(open(path, encoding="utf-8"))[key]


def reference_sheet(names, dest, per_row=8, tile=190):
    """One cropped example per class, captioned, so brands can be matched by eye."""
    crops = {}
    for split in ("train", "valid"):
        idir = os.path.join(DATA_V2, split, "images")
        ldir = os.path.join(DATA_V2, split, "labels")
        if not os.path.isdir(idir):
            continue
        for fn in sorted(os.listdir(idir)):
            if len(crops) == len(names):
                break
            lpath = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
            if not os.path.exists(lpath):
                continue
            rows = [l.split() for l in open(lpath, encoding="utf-8") if l.strip()]
            if len(rows) != 1:      # a single-bottle photo gives an unambiguous example
                continue
            cid = int(rows[0][0])
            name = names[cid] if 0 <= cid < len(names) else None
            if not name or name in crops:
                continue
            x, y, w, h = (float(v) for v in rows[0][1:5])
            with Image.open(os.path.join(idir, fn)) as im:
                im = im.convert("RGB")
                W, H = im.size
                pad = 0.08
                box = (max(0, int((x - w / 2 - pad * w) * W)), max(0, int((y - h / 2 - pad * h) * H)),
                       min(W, int((x + w / 2 + pad * w) * W)), min(H, int((y + h / 2 + pad * h) * H)))
                if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                    continue
                crops[name] = im.crop(box)

    missing = [n for n in names if n not in crops]
    ordered = [n for n in names if n in crops]
    rows = (len(ordered) + per_row - 1) // per_row
    cap = 26
    sheet = Image.new("RGB", (per_row * tile, rows * (tile + cap)), "white")
    pen = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for i, name in enumerate(ordered):
        img = crops[name].copy()
        img.thumbnail((tile - 8, tile - 8))
        x0, y0 = (i % per_row) * tile, (i // per_row) * (tile + cap)
        sheet.paste(img, (x0 + (tile - img.width) // 2, y0 + (tile - img.height) // 2))
        label = name if len(name) <= 22 else name[:21] + "…"
        tb = pen.textbbox((0, 0), label, font=font)
        pen.text((x0 + (tile - (tb[2] - tb[0])) // 2, y0 + tile + 4), label,
                 fill=(20, 20, 20), font=font)
        pen.rectangle([x0, y0, x0 + tile - 1, y0 + tile + cap - 1], outline=(220, 220, 220))
    sheet.save(dest)
    return len(ordered), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "yolo11s_v2", "weights", "best.pt"))
    ap.add_argument("--report", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    v2_yaml = os.path.join(DATA_V2, "data.yaml")
    if not os.path.exists(v2_yaml):
        raise SystemExit("dataset_v2 not found - run scripts/build_dataset_ouo.py first")
    new_names = list(load_names(v2_yaml)) + [UNKNOWN]
    by_canon = {canon(n): n for n in new_names}

    old_names = load_names(os.path.join(TESTSET, "data.yaml"))
    ldir = os.path.join(TESTSET, "labels")
    idir = os.path.join(TESTSET, "images")

    renamed, dropped, kept_unknown = Counter(), Counter(), 0
    rewrites = {}
    for fn in sorted(os.listdir(idir)):
        stem = os.path.splitext(fn)[0]
        path = os.path.join(ldir, stem + ".txt")
        if not os.path.exists(path):
            continue
        out = []
        for line in open(path, encoding="utf-8"):
            p = line.split()
            if len(p) < 5:
                continue
            old = old_names[int(p[0])] if 0 <= int(p[0]) < len(old_names) else UNKNOWN
            hit = by_canon.get(canon(ALIAS.get(old, old)))
            if hit is None:
                # A name the annotator invented that the model still has no class for.
                dropped[old] += 1
                hit = UNKNOWN
            if hit == UNKNOWN:
                kept_unknown += 1
            elif hit != old:
                renamed[f"{old} -> {hit}"] += 1
            out.append(f"{new_names.index(hit)} {' '.join(p[1:5])}")
        rewrites[path] = out

    total = sum(len(v) for v in rewrites.values())
    named = total - kept_unknown
    print(f"test set: {len(rewrites)} photos, {total} boxes")
    print(f"  already named, carried over: {named}")
    print(f"  still '{UNKNOWN}', to be reviewed: {kept_unknown}")
    if renamed:
        print("  spelling changes:")
        for k, v in renamed.most_common():
            print(f"    {k}  x{v}")
    if dropped:
        print(f"  names with no class in the 119, reset to {UNKNOWN}:")
        for k, v in dropped.most_common():
            print(f"    {k}  x{v}")

    if args.report:
        return

    backup = os.path.join(TESTSET, "labels_50class_backup")
    if not os.path.exists(backup):
        shutil.copytree(ldir, backup)
        print(f"\n  previous annotations copied to {os.path.basename(backup)}")

    for path, lines in rewrites.items():
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
    with open(os.path.join(ldir, "classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(new_names) + "\n")
    meta = yaml.safe_load(open(os.path.join(TESTSET, "data.yaml"), encoding="utf-8"))
    meta["nc"], meta["names"] = len(new_names), new_names
    yaml.safe_dump(meta, open(os.path.join(TESTSET, "data.yaml"), "w", encoding="utf-8"),
                   sort_keys=False, allow_unicode=True)

    sheet_path = os.path.join(TESTSET, "class_reference.png")
    shown, missing = reference_sheet(load_names(v2_yaml), sheet_path)
    print(f"\n  rewrote {len(rewrites)} label files onto {len(new_names)} classes")
    print(f"  reference sheet: {shown} of {len(new_names) - 1} classes -> "
          f"{os.path.basename(sheet_path)}")
    if missing:
        print(f"    no single-bottle example for: {', '.join(missing[:10])}")
    print("\nOpen the set with: python scripts/annotate.py")
    print(f"Name the bottles you recognise; leave the rest as {UNKNOWN}.")
    print("Do not guess from the model's output - that is what this set exists to check.")


if __name__ == "__main__":
    main()
