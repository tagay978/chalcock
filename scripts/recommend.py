"""Turn detected bottles into IBA cocktails you can actually make.

    python scripts/recommend.py --bottles gordons,extradry,orangebitters
    python scripts/recommend.py --image shelf.jpg --weights runs/yolo11s/weights/best.pt
    python scripts/recommend.py --bottles ... --loose --missing 1
    python scripts/recommend.py --check      # audit the ingredient rules against the recipes

A recipe is makeable when every `bar` ingredient it calls for is on the shelf. Pantry items
(juice, syrup, soda, egg, herbs) are assumed present; add anything else you own with --assume.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
COCO_BOTTLE = 39
# A model trained with scripts/build_abstain_dataset.py can answer "a bottle, but not one of
# mine". That is not an ingredient, so it never reaches the recipe matcher.
ABSTAIN = "unknown_bottle"

# Leading quantity: "30 ml", "1 dash", "2 Bar Spoons", "3/4 Bar Spoon", "1-3 slices", "6/8 pcs".
QTY = re.compile(
    r"^\s*(?:[0-9]+(?:\s*[/-]\s*[0-9]+)?(?:[.,][0-9]+)?\s*)?"
    r"(?:ml|cl|oz|dash(?:es)?|drops?|bar\s*spoons?|barspoons?|tsp|teaspoons?|tablespoons?|"
    r"cubes?|leaves|sprigs?|slices?|wedges?|chunks?|parts?|pcs?|pinch(?:\s+of)?)?\s*",
    re.I,
)
NOISE = re.compile(r"\b(fresh(?:ly)?|squeezed|chilled|strong|raw|whole|thin|small|up to taste|"
                   r"optional|to serve on the side|top up(?:\s+with)?|top with|fill up with|"
                   r"a splash of|splash of|few drops of|few drops|few dashes|dash of|a dash of|"
                   r"a pinch of|and|of)\b", re.I)


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def canon(name):
    """Ignore punctuation and case: the 50-class map says `camusvsop`, ouo_final `camus_vsop`."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class Rules:
    def __init__(self):
        cfg = load("ingredient_rules.yaml")
        self.patterns = [(re.compile(p["match"], re.I), p["key"]) for p in cfg["patterns"]]
        self.pantry = set(cfg["pantry"])
        self.equivalents = {k: set(v) for k, v in (cfg.get("equivalents") or {}).items()}
        self.substitutes = {k: set(v) for k, v in (cfg.get("substitutes") or {}).items()}

        # The hand-written 50-class map, plus the 119-class one generated from ouo_final's
        # category folders, so a detector trained on either vocabulary resolves to ingredients.
        self.bottles = dict(load("bottles.yaml")["bottles"])
        generated = os.path.join(ROOT, "dataset_v2", "class_ingredients.yaml")
        if os.path.exists(generated):
            extra = (yaml.safe_load(open(generated, encoding="utf-8")) or {}).get("bottles") or {}
            known = {canon(k) for k in self.bottles}
            for k, v in extra.items():
                if canon(k) not in known:
                    self.bottles[k] = v
        # ouo_final files 42 brands under one `liqueur` folder, which is too coarse to match a
        # recipe. The ingredient patterns below already know aperol, campari, absinthe and the
        # rest by name, so run the brand through them and keep the category only as a fallback.
        for name, entry in self.bottles.items():
            if entry.get("ingredient") not in (None, "", "liqueur", "unknown"):
                continue
            key, _ = self.normalize(name)
            if key and key not in self.pantry:
                entry["ingredient"] = key

        # Lookup that tolerates the two spellings of the same bottle.
        self._by_canon = {canon(k): k for k in self.bottles}

        # Every ingredient key the rules know about, so a detector trained at ingredient level
        # (whose classes ARE these keys) resolves without a bottle entry to look up.
        self._ingredients = ({key for _, key in self.patterns}
                             | {v.get("ingredient") for v in self.bottles.values()}) - {None}

    def bottle(self, cls):
        """Resolve a predicted class name to its bottle entry.

        Handles three vocabularies: the hand-written 50, the generated 113, and a model trained
        with --level ingredient, whose class names are the ingredient keys themselves.
        """
        if cls in self.bottles:
            return self.bottles[cls]
        hit = self._by_canon.get(canon(cls))
        if hit:
            return self.bottles[hit]
        if cls in self._ingredients:
            return {"label": cls.replace("_", " ").title(), "ingredient": cls}
        return {}

    def normalize(self, line: str):
        """Reduce one IBA ingredient line to (canonical_key | None, cleaned_text)."""
        # The site mixes precomposed and combining accents (Crème, Bénédictine), so fold them
        # away and match against plain ASCII.
        line = "".join(ch for ch in unicodedata.normalize("NFKD", line)
                       if not unicodedata.combining(ch))
        text = QTY.sub("", line, count=1)
        text = re.sub(r"\(.*?\)", " ", text)
        text = NOISE.sub(" ", text)
        text = re.sub(r"[*]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" .,-/")
        for rx, key in self.patterns:
            if rx.search(text):
                return key, text
        return None, text

    def satisfied_by(self, need: str, have: set[str], loose: bool):
        """Is `need` covered by the shelf? Returns the key that covers it, or None."""
        if need in have:
            return need
        for alt in self.equivalents.get(need, ()):
            if alt in have:
                return alt
        if loose:
            for alt in self.substitutes.get(need, ()):
                if alt in have:
                    return alt
        return None


def recipe_requirements(cocktail, rules: Rules):
    """Split a recipe into the bar bottles it needs and the pantry items it assumes."""
    bar, pantry, unknown = [], [], []
    for line in cocktail["ingredients"]:
        key, text = rules.normalize(line)
        if key is None:
            unknown.append(text)
        elif key in rules.pantry:
            pantry.append(key)
        else:
            bar.append((key, line))
    return bar, pantry, unknown


def detect(image: str, weights: str, conf: float):
    from ultralytics import YOLO

    if not os.path.exists(weights):
        sys.exit(f"weights not found: {weights}  (train first, or pass --weights)")
    results = YOLO(weights).predict(image, conf=conf, verbose=False)
    found, declined = Counter(), 0
    for r in results:
        for c in r.boxes.cls.tolist():
            name = r.names[int(c)]
            if name == ABSTAIN:
                declined += 1
                continue
            found[name] += 1
    if declined:
        print(f"  ({declined} bottles the model declined to name)")
    return found


def detect_two_stage(image: str, weights: str, conf: float, coco_weights: str, pad: float = 0.15):
    """Find bottles with a COCO detector, then identify each crop with the brand model.

    The brand model was trained on close-ups: a bottle covers ~20% of the frame in training but
    ~1% in a real shelf photo, so running it on the full image finds almost nothing. Cropping to
    each bottle first puts the subject back at the scale the model was trained on.
    """
    from PIL import Image
    from ultralytics import YOLO

    if not os.path.exists(coco_weights):
        sys.exit(f"COCO weights not found: {coco_weights}")
    coco, brand = YOLO(coco_weights), YOLO(weights)

    im = Image.open(image).convert("RGB")
    width, height = im.size
    crops = []
    for r in coco.predict(image, conf=0.25, verbose=False):
        for cls, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            if int(cls) != COCO_BOTTLE:
                continue
            x1, y1, x2, y2 = xyxy
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
                   min(width, int(x2 + px)), min(height, int(y2 + py)))
            if box[2] - box[0] >= 20 and box[3] - box[1] >= 20:
                crops.append(im.crop(box).resize((640, 640)))

    found, declined = Counter(), 0
    for i in range(0, len(crops), 32):
        for r in brand.predict(crops[i:i + 32], conf=conf, verbose=False):
            if not len(r.boxes):
                continue
            name = r.names[int(r.boxes.cls[r.boxes.conf.argmax()])]
            if name == ABSTAIN:
                declined += 1
                continue
            found[name] += 1
    print(f"  ({len(crops)} bottles found, {sum(found.values())} named"
          + (f", {declined} declined" if declined else "") + ")")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottles", help="comma-separated detector class names")
    ap.add_argument("--image", help="photo of the shelf; runs the detector")
    ap.add_argument("--weights", default=next(
        (p for p in (os.path.join(ROOT, "runs", r, "weights", "best.pt")
                     for r in ("yolo11s_ing", "yolo11s_v3", "yolo11s_v2", "yolo11s")) if os.path.exists(p)), ""))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--one-stage", dest="two_stage", action="store_false",
                    help="run the detector straight at the photo; finds far less on real shelves")
    ap.add_argument("--coco-weights", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    ap.add_argument("--assume", default="", help="extra ingredient keys you own, comma-separated")
    ap.add_argument("--loose", action="store_true", help="allow near substitutes (aged rum for white)")
    ap.add_argument("--missing", type=int, default=0, help="also show recipes short by N bottles")
    ap.add_argument("--check", action="store_true", help="audit rules against the recipe data")
    args = ap.parse_args()

    rules = Rules()
    recipes = json.load(open(os.path.join(DATA, "iba_cocktails.json"), encoding="utf-8"))["cocktails"]

    if args.check:
        unknown, bar_keys = Counter(), Counter()
        for c in recipes:
            bar, _, unk = recipe_requirements(c, rules)
            for u in unk:
                unknown[u] += 1
            for k, _ in bar:
                bar_keys[k] += 1
        print(f"{len(recipes)} recipes, {len(bar_keys)} distinct bar ingredients")
        covered = {b["ingredient"] for b in rules.bottles.values()}
        print(f"\nbar ingredients the 50 detector classes can supply: "
              f"{len([k for k in bar_keys if k in covered])}/{len(bar_keys)}")
        print("\nunmapped ingredient text:", len(unknown) or "none")
        for text, n in unknown.most_common():
            print(f"  {n:3}  {text}")
        return

    if args.image:
        if args.two_stage:
            found = detect_two_stage(args.image, args.weights, args.conf, args.coco_weights)
        else:
            found = detect(args.image, args.weights, args.conf)
        print(f"detected in {os.path.basename(args.image)}:")
        for cls, n in found.most_common():
            print(f"  {rules.bottle(cls).get('label', cls)}" + (f" x{n}" if n > 1 else ""))
        classes = list(found)
    elif args.bottles:
        classes = [c.strip() for c in args.bottles.split(",") if c.strip()]
    else:
        ap.error("pass --bottles or --image")

    unknown_cls = [c for c in classes if not rules.bottle(c)]
    if unknown_cls:
        sys.exit(f"unknown bottle class: {', '.join(unknown_cls)}")

    have = {rules.bottle(c)["ingredient"] for c in classes}
    have |= {a.strip() for a in args.assume.split(",") if a.strip()}
    print(f"\nshelf: {len(classes)} bottles -> {len(have)} ingredients "
          f"({', '.join(sorted(have))})\n")

    makeable, short = [], []
    for c in recipes:
        bar, _, _ = recipe_requirements(c, rules)
        missing, used = [], {}
        for key, line in bar:
            src = rules.satisfied_by(key, have, args.loose)
            if src is None:
                missing.append(key)
            else:
                used[key] = src
        if not missing:
            makeable.append((c, used))
        elif len(missing) <= args.missing and used:
            # `used` matters: a shelf of scotch alone is genuinely one bottle from a Bellini,
            # and saying so is useless. A near miss has to build on something you have.
            short.append((c, missing))

    print(f"=== makeable now: {len(makeable)} of {len(recipes)} IBA cocktails ===")
    for c, used in sorted(makeable, key=lambda x: (x[0]["category"], x[0]["name"])):
        subs = [f"{k}->{v}" for k, v in used.items() if k != v]
        note = f"   [{', '.join(subs)}]" if subs else ""
        print(f"\n{c['name']}  ({c['category']}){note}")
        for line in c["ingredients"]:
            print(f"    - {line}")
        if c["method"]:
            print(f"    method: {c['method']}")
        if c["garnish"] and c["garnish"].upper() != "N/A":
            print(f"    garnish: {c['garnish']}")

    if short:
        print(f"\n=== one or two bottles away: {len(short)} ===")
        for c, missing in sorted(short, key=lambda x: (len(x[1]), x[0]["name"])):
            print(f"  {c['name']:28} needs {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
