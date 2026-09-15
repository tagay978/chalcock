"""Build the training set from ouo_final, folding in whatever the old dataset still adds.

ouo_final is the fuller collection: 122 brand folders under 21 spirit categories, 13,384 boxes
against the old merge's 1,593. It is laid out category/brand/<image>.jpg + <image>.txt with a
`classes.txt` beside them, and the label ids index that file - which is per-folder, so the same
id means different things in different folders and the mapping has to be read locally.

The old `dataset/` is mostly redundant: 93% of its images already appear in ouo_final at pHash
distance <= 2, because both trace back to the same 2023 collection and the old copies merely went
through Roboflow's re-encoding (which is why MD5 finds no overlap at all). What is left still
carries annotation work, so it is merged rather than dropped, and deduplication is perceptual for
the same reason it had to be in build_dataset.py.

Run:  python scripts/build_dataset_ouo.py --report
      python scripts/build_dataset_ouo.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import shutil
from collections import Counter, defaultdict

import sys

import imagehash
import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUO = os.path.join(ROOT, "ouo_final")
OLD = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "dataset_v2")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PHASH_MAX = 2
IOU_SAME = 0.6

# The old merge used three names ouo_final spells differently. Everything else lines up once
# punctuation is ignored: 47 of the old 50 matched directly.
ALIAS = {
    "grandmanier": "grandmarnier",      # the old export misspelled Grand Marnier
    "orangebitters": "orange",
    "wildturkeykrye": "wildturkey",     # ouo_final files it under ryewhiskey/wildturkey
}

# Classes that are the same bottle under different names. Splitting one product across several
# labels is not a harmless redundancy: the model has to divide its probability between classes it
# cannot tell apart, and recall collapses. Bacardi Carta Oro, Oro and Gold are one rum; Carta
# Blanca, Superior and White Rum are another; the `_nolabel` Wild Turkeys are the same bottles
# photographed without the neck label. Confirmed by eye against the class reference crops.
SYNONYMS = {
    "bacardicartaoro": "bacardi_gold",
    "bacardioro": "bacardi_gold",
    "bacardigold": "bacardi_gold",
    "bacardicartablanca": "bacardi_white",
    "bacardisuperior": "bacardi_white",
    "bacardiwhiterum": "bacardi_white",
    "wildturkey_101proof": "wildturkey_101",
    "wildturkey_101proof_nolabel": "wildturkey_101",
    "wildturkey_8y_nolabel": "wildturkey_8y",
}


# Brand families for --level family: variants of one brand collapsed to the brand.
#
# Most of these are free - every Ballantine's is scotch, every Camus is cognac - so the model
# stops splitting its probability over label stripes it cannot read at shelf scale.
#
# Two are not free, and are here because they were asked for. Wild Turkey spans bourbon and rye,
# and Absolut spans plain, citron and vanilla vodka; merging them means the recommender can no
# longer tell a Manhattan's rye from a bourbon, or a Cosmopolitan's citron vodka from plain.
# Both are marked below; drop them from this map to keep the distinction.
FAMILIES = {
    # mixes ingredients - see above
    "wildturkey": ["wildturkey", "wildturkey_101", "wildturkey_81proof", "wildturkey_8y",
                   "wildturkey_bourbon"],
    "absolutevodka": ["absolutevodka", "absolutecitron", "absolutevanilla"],
    # same ingredient throughout
    "ballantine": ["ballantine12", "ballantinefinest", "ballantinemasters"],
    "bushmills": ["bushmillsblackbush", "bushmillsoriginal"],
    "camus": ["camus_vsop", "camus_xo"],
    "hennessy": ["hennessy_vsop", "hennessy_xo"],
    "johnbarr": ["johnbarrfinest", "johnbarrreserve"],
    "johnniewalker": ["johnnieblack", "johnniered"],
    "josecuervo": ["josecuervoespecial", "josecuervoespecialsilver"],
    "tanqueray": ["tanqueray", "tanquerayten"],
}
# Deliberately NOT merged: bacardi (gold vs white rum), havanaclub and captainmorgan (each spans
# white, gold and dark), chartreuse (green and yellow are different liqueurs). Collapsing those
# would remove ingredients recipes actually ask for.
FAMILY_OF = {member: fam for fam, members in FAMILIES.items() for member in members}


# Each spirit category maps to the ingredient key data/ingredient_rules.yaml already speaks.
CATEGORY_INGREDIENT = {
    "bitters": "aromatic_bitters", "blendedwhiskey": "scotch", "bourbonwhiskey": "bourbon",
    "cachaca": "cachaca", "calvados": "calvados", "cognac": "cognac", "darkrum": "dark_rum",
    "gin": "gin", "goldrum": "gold_rum", "grappa": "grappa", "irishwhiskey": "irish_whiskey",
    "liqueur": "liqueur", "mezcal": "mezcal", "ryewhiskey": "rye", "tequila": "tequila",
    "vermouth": "sweet_vermouth", "vodka": "vodka", "vodkacitron": "citron_vodka",
    "vodkavanilla": "vanilla_vodka", "whiterum": "white_rum", "loopy": "rum",
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


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


def read_label(path):
    """Return YOLO rows, or None when the file is not a label at all.

    64 files in ouo_final are JPEGs that were saved with a .txt extension, and 34 are empty.
    """
    raw = open(path, "rb").read()
    if not raw or b"\x00" in raw[:200]:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    rows = []
    for line in text.split("\n"):
        p = line.split()
        if len(p) != 5:
            continue
        try:
            rows.append((int(p[0]), *(float(v) for v in p[1:5])))
        except ValueError:
            return None
    return rows


def fingerprint(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im.size, imagehash.phash(im)


def scan_ouo(skipped, level="brand"):
    records = []
    for cat in sorted(os.listdir(OUO)):
        cdir = os.path.join(OUO, cat)
        if not os.path.isdir(cdir):
            continue
        for brand in sorted(os.listdir(cdir)):
            bdir = os.path.join(cdir, brand)
            if not os.path.isdir(bdir):
                continue
            cls_path = os.path.join(bdir, "classes.txt")
            names = ([l.strip() for l in open(cls_path, encoding="utf-8") if l.strip()]
                     if os.path.exists(cls_path) else [])
            for fn in sorted(os.listdir(bdir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in IMG_EXT:
                    continue
                ipath = os.path.join(bdir, fn)
                lpath = os.path.join(bdir, stem + ".txt")
                if not os.path.exists(lpath):
                    skipped["image with no label"] += 1
                    continue
                rows = read_label(lpath)
                if rows is None:
                    skipped["label file is not a label"] += 1
                    continue
                if not rows:
                    skipped["empty label"] += 1
                    continue
                boxes = []
                for cid, *xywh in rows:
                    name = names[cid] if 0 <= cid < len(names) else None
                    # calvados/1 has a folder and a class both literally named "1".
                    if not name or name.isdigit():
                        name = cat
                    name = SYNONYMS.get(name, name)
                    if level == "family":
                        name = FAMILY_OF.get(name, name)
                    boxes.append((name, *xywh))
                try:
                    size, ph = fingerprint(ipath)
                except Exception:
                    skipped["unreadable image"] += 1
                    continue
                records.append({"path": ipath, "stem": stem, "ext": ext.lower(), "boxes": boxes,
                                "size": size, "phash": ph, "source": "ouo",
                                "category": cat, "brand": brand,
                                "md5": hashlib.md5(open(ipath, "rb").read()).hexdigest()})
    return records


def scan_old(skipped, ouo_names, level="brand"):
    """Read the old merge, renaming its classes onto ouo_final's spelling.

    The two sets write the same bottle differently - `camusvsop` against `camus_vsop` - so
    matching on the name as written would split one bottle into two classes. Ignoring
    punctuation lines up 47 of the old 50; ALIAS covers the three that need a human.
    """
    yaml_path = os.path.join(OLD, "data.yaml")
    if not os.path.exists(yaml_path):
        return []
    names = yaml.safe_load(open(yaml_path, encoding="utf-8"))["names"]
    by_norm = {norm(n): n for n in ouo_names}
    rename = {}
    for n in names:
        target = ALIAS.get(n) or by_norm.get(norm(n))
        rename[n] = target or n
    unmatched = [n for n, t in rename.items() if t == n and norm(n) not in by_norm]
    if unmatched:
        print(f"  old classes with no counterpart in ouo_final: {', '.join(unmatched)}")
    names = [SYNONYMS.get(rename[n], rename[n]) for n in names]
    if level == "family":
        names = [FAMILY_OF.get(n, n) for n in names]
    records = []
    for split in ("train", "valid", "test"):
        idir = os.path.join(OLD, split, "images")
        if not os.path.isdir(idir):
            continue
        for fn in sorted(os.listdir(idir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in IMG_EXT:
                continue
            ipath = os.path.join(idir, fn)
            lpath = os.path.join(OLD, split, "labels", stem + ".txt")
            rows = read_label(lpath) if os.path.exists(lpath) else None
            if not rows:
                skipped["old: no usable label"] += 1
                continue
            boxes = [(names[cid], *xywh) for cid, *xywh in rows if 0 <= cid < len(names)]
            try:
                size, ph = fingerprint(ipath)
            except Exception:
                skipped["unreadable image"] += 1
                continue
            records.append({"path": ipath, "stem": stem, "ext": ext.lower(), "boxes": boxes,
                            "size": size, "phash": ph, "source": "old",
                            "category": None, "brand": None,
                            "md5": hashlib.md5(open(ipath, "rb").read()).hexdigest()})
    return records


def group(records):
    u = Union()
    by_md5 = defaultdict(list)
    for i, r in enumerate(records):
        u.find(i)
        by_md5[r["md5"]].append(i)
    for idxs in by_md5.values():
        for j in idxs[1:]:
            u.join(idxs[0], j)

    # pHash buckets keep this from being quadratic over 5,000 images.
    near = 0
    buckets = defaultdict(list)
    for i, r in enumerate(records):
        buckets[str(r["phash"])[:8]].append(i)
    for idxs in buckets.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if u.find(i) == u.find(j):
                    continue
                if records[i]["phash"] - records[j]["phash"] <= PHASH_MAX:
                    u.join(i, j)
                    near += 1

    groups = defaultdict(list)
    for i in range(len(records)):
        groups[u.find(i)].append(i)
    return list(groups.values()), near


def merge_boxes(members, records):
    merged = []
    for i in members:
        for box in records[i]["boxes"]:
            if any(box[0] == m[0] and iou(box, m) > IOU_SAME for m in merged):
                continue
            merged.append(box)
    return merged


def sanitize(stem):
    stem = re.sub(r"\.rf\.[0-9a-f]{32}", "", stem, flags=re.I)
    return (re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "img")[:56]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="analyse only, write nothing")
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", choices=["brand", "family", "ingredient"], default="brand",
                    help="train on the 113 brands, or on what they pour. Ballantine's 12, Finest "
                         "and Masters differ by a label stripe and all three are scotch, so the "
                         "ingredient view trades distinctions the recommender never uses for "
                         "roughly five times the data per class.")
    ap.add_argument("--out", default=None, help="output directory; defaults by --level")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="cap the long edge on copy; ouo_final holds camera originals up to "
                         "3024x4032, and training resizes to 640 anyway")
    args = ap.parse_args()

    if not os.path.isdir(OUO):
        raise SystemExit(f"not found: {OUO}")

    global OUT
    suffix = {"brand": "dataset_v2", "family": "dataset_fam", "ingredient": "dataset_ing"}
    OUT = args.out or OUT.replace("dataset_v2", suffix[args.level])

    to_ingredient = None
    if args.level == "family":
        mixed = []
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import recommend as R
        rules = R.Rules()
        for fam, members in FAMILIES.items():
            ings = {rules.bottle(m).get("ingredient") for m in members} - {None}
            if len(ings) > 1:
                mixed.append(f"{fam} ({', '.join(sorted(ings))})")
        print(f"  folding {len(FAMILY_OF)} classes into {len(FAMILIES)} brand families")
        if mixed:
            print(f"  these families merge different ingredients: {'; '.join(mixed)}")
    if args.level == "ingredient":
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import recommend as R
        rules = R.Rules()
        to_ingredient = lambda n: (rules.bottle(n).get("ingredient")
                                   or CATEGORY_INGREDIENT.get(n, n))

    skipped = Counter()
    print("scanning ouo_final ...")
    records = scan_ouo(skipped, args.level)
    print(f"  {len(records)} labelled images, {sum(len(r['boxes']) for r in records)} boxes")
    print("scanning dataset ...")
    ouo_names = {b[0] for r in records for b in r["boxes"]}
    old = scan_old(skipped, ouo_names, args.level)
    print(f"  {len(old)} labelled images, {sum(len(r['boxes']) for r in old)} boxes")
    records += old
    for k, v in skipped.items():
        print(f"  skipped {v}x {k}")

    if to_ingredient is not None:
        unresolved = Counter()
        for r in records:
            mapped = []
            for name, *xywh in r["boxes"]:
                ing = to_ingredient(name)
                if not ing or ing in ("unknown", ""):
                    unresolved[name] += 1
                    ing = CATEGORY_INGREDIENT.get(r["category"] or "", name)
                mapped.append((ing, *xywh))
            r["boxes"] = mapped
        if unresolved:
            print(f"  {len(unresolved)} classes fell back to their category: "
                  f"{', '.join(list(unresolved)[:6])}")

    groups, near = group(records)
    from_old = sum(1 for g in groups if all(records[i]["source"] == "old" for i in g))
    mixed = sum(1 for g in groups if len({records[i]["source"] for i in g}) > 1)
    names = sorted({b[0] for r in records for b in r["boxes"]})
    print(f"\n{len(groups)} unique photos "
          f"({len(records) - len(groups)} duplicates collapsed, {near} by pHash)")
    print(f"  photos only the old dataset had: {from_old}")
    print(f"  photos present in both:          {mixed}")
    print(f"  classes: {len(names)}")

    per_class = Counter(b[0] for g in groups for b in merge_boxes(g, records))
    thin = [f"{c}:{per_class[c]}" for c in names if per_class[c] < 15]
    print(f"  boxes: {sum(per_class.values())}")
    if thin:
        print(f"  under 15 boxes: {', '.join(thin)}")

    if args.report:
        return

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(OUT, split, sub), exist_ok=True)

    # Assign splits per photo, ordered so each class fills valid and test before train takes the
    # rest; without this the rare classes land entirely in one split.
    rng = random.Random(args.seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    rarity = {}
    for gi in order:
        boxes = merge_boxes(groups[gi], records)
        rarity[gi] = min((per_class[b[0]] for b in boxes), default=10**9)
    order.sort(key=lambda gi: rarity[gi])

    quota = defaultdict(lambda: {"train": 0, "valid": 0, "test": 0})
    assign = {}
    for gi in order:
        boxes = merge_boxes(groups[gi], records)
        key = min(boxes, key=lambda b: per_class[b[0]])[0] if boxes else "_"
        # A class with 7 boxes can otherwise land entirely in valid, leaving nothing to train
        # on; train and valid each get one photo before the fractions apply.
        if quota[key]["train"] == 0:
            split = "train"
        elif quota[key]["valid"] == 0:
            split = "valid"
        elif quota[key]["valid"] < per_class[key] * args.valid_frac:
            split = "valid"
        elif quota[key]["test"] < per_class[key] * args.test_frac:
            split = "test"
        else:
            split = "train"
        quota[key][split] += len(boxes)
        assign[gi] = split

    cls_id = {n: i for i, n in enumerate(names)}
    counts, box_counts = Counter(), Counter()
    resized = [0]
    manifest = [("file", "split", "n_boxes", "classes", "sources")]
    for gi, members in enumerate(groups):
        split = assign[gi]
        rep = min(members, key=lambda i: (records[i]["source"] != "ouo", records[i]["path"]))
        r = records[rep]
        boxes = merge_boxes(members, records)
        name = f"{r['md5'][:12]}_{sanitize(r['stem'])}"
        ext = ".jpg" if r["ext"] in (".jpg", ".jpeg") else r["ext"]
        dest = os.path.join(OUT, split, "images", name + ext)
        with Image.open(r["path"]) as im:
            if max(im.size) > args.max_side:
                im = im.convert("RGB")
                im.thumbnail((args.max_side, args.max_side))
                im.save(dest, quality=92)
                resized[0] += 1
            else:
                shutil.copy2(r["path"], dest)
        with open(os.path.join(OUT, split, "labels", name + ".txt"), "w", encoding="utf-8") as fh:
            for c, x, y, w, h in boxes:
                fh.write(f"{cls_id[c]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        counts[split] += 1
        for c, *_ in boxes:
            box_counts[c] += 1
        manifest.append((name + ext, split, len(boxes),
                         "|".join(sorted({c for c, *_ in boxes})),
                         "|".join(sorted({records[i]["source"] for i in members}))))

    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": OUT.replace("\\", "/"), "train": "train/images",
                        "val": "valid/images", "test": "test/images",
                        "nc": len(names), "names": names},
                       fh, sort_keys=False, allow_unicode=True)
    with open(os.path.join(OUT, "manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(manifest)

    # The category folder already says what each brand pours, so the ingredient map comes free.
    ingredient = {}
    for r in records:
        if r["source"] != "ouo":
            continue
        for c, *_ in r["boxes"]:
            ingredient.setdefault(c, CATEGORY_INGREDIENT.get(r["category"], r["category"]))
    with open(os.path.join(OUT, "class_ingredients.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"bottles": {c: {"label": c, "ingredient": ingredient.get(c, "unknown")}
                                    for c in names}},
                       fh, sort_keys=True, allow_unicode=True)

    print(f"\nwrote {OUT}")
    print("  splits:", dict(counts))
    print(f"  {resized[0]} images shrunk to a {args.max_side}px long edge")
    missing = [c for c in names if c not in ingredient]
    if missing:
        print(f"  no category for {len(missing)} classes (from the old set only): "
              f"{', '.join(missing[:8])}")
    print(f"  class_ingredients.yaml written for {len(names)} classes")


if __name__ == "__main__":
    main()
