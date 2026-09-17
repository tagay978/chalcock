"""Web service: upload a photo of a shelf, get the cocktails you can actually make.

Serves the same pipeline the CLI uses - COCO finds the bottles, the brand model names them,
data/ maps names to ingredients and ingredients to IBA recipes - so the demo cannot drift from
the measured numbers. scripts/recommend.py is imported rather than reimplemented.

Run:  python app/main.py
      python app/main.py --weights runs/yolo11s_abstain/weights/best.pt --port 8000
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys

from collections import OrderedDict

import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import recommend as R  # noqa: E402  - path set above

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
COCO_BOTTLE = 39
# Crop the bottle exactly as the COCO detector boxed it. Padding used to add 15% on
# each side; it pulls in whatever stands next to the bottle, which on a packed shelf is
# another bottle.
PAD = 0.0
NAMED = (34, 197, 94)
# yolo11s_v10/epoch80 at conf 0.4, picked by eye against real shelf photos - see the weights
# default in main() below for which checkpoint this pairs with.
DEFAULT_CONF = 0.4

app = FastAPI(title="Cocktail bottle detector")
state: dict = {}
loaded: "OrderedDict[str, dict]" = OrderedDict()   # key -> model + its metadata
MAX_LOADED = 4        # each weights file costs GPU memory, so keep only a few resident


def discover_models():
    """Every checkpoint on disk, newest run first.

    train.py --save-period writes weights/epochN.pt beside best.pt and last.pt, and which epoch
    does best on real photos is not the one validation picks - so they are all worth offering.
    """
    runs = os.path.join(ROOT, "runs")
    found = []
    if not os.path.isdir(runs):
        return found
    for run in sorted(os.listdir(runs)):
        wdir = os.path.join(runs, run, "weights")
        if not os.path.isdir(wdir):
            continue
        tags = []
        for fn in os.listdir(wdir):
            if not fn.endswith(".pt"):
                continue
            tag = fn[:-3]
            order = (0 if tag == "best" else 1 if tag == "last" else 2,
                     int(tag[5:]) if tag.startswith("epoch") and tag[5:].isdigit() else 0)
            tags.append((order, tag, os.path.join(wdir, fn)))
        for _, tag, path in sorted(tags):
            found.append({"key": f"{run}/{tag}", "run": run, "tag": tag, "path": path})
    return found


def describe(yolo):
    """Class count, whether the classes are ingredients, and the confidence that suits it.

    Confidence does not transfer between models: 113 classes split the softmax far more finely
    than 52, so the same cut means something different. Measured best operating points on the
    real-world set were 0.02-0.25 for the 50-class brand model, 0.05 for the 113-class one and
    0.4 for the ingredient-level one.
    """
    classes = set(yolo.names.values())
    ingredient = len(classes & state["rules"]._ingredients) > len(classes) / 2
    return {
        "classes": len(classes),
        "ingredient_level": ingredient,
        "abstains": R.ABSTAIN in classes,
        "default_conf": 0.4 if ingredient else 0.25 if len(classes) <= 60 else 0.05,
    }


def get_model(key: str):
    """Load a checkpoint on demand and keep the last few resident."""
    if key in loaded:
        loaded.move_to_end(key)
        return loaded[key]
    entry = next((m for m in discover_models() if m["key"] == key), None)
    if entry is None:
        return None

    from ultralytics import YOLO

    yolo = YOLO(entry["path"])
    meta = {"yolo": yolo, **entry, **describe(yolo)}
    loaded[key] = meta
    while len(loaded) > MAX_LOADED:
        dropped, _ = loaded.popitem(last=False)
        print(f"unloaded {dropped}")
    print(f"loaded {key}: {meta['classes']} "
          f"{'ingredient' if meta['ingredient_level'] else 'brand'} classes, "
          f"default conf {meta['default_conf']}")
    return meta


def load_models(weights: str, coco_weights: str):
    from ultralytics import YOLO

    if not os.path.exists(weights):
        raise SystemExit(f"weights not found: {weights}")
    state["coco"] = YOLO(coco_weights)
    state["rules"] = R.Rules()
    state["recipes"] = R.json.load(
        open(os.path.join(ROOT, "data", "iba_cocktails.json"), encoding="utf-8"))["cocktails"]
    state["recipes_by_slug"] = {c["slug"]: c for c in state["recipes"]}
    state["photos"] = load_credits("cocktails")
    state["ingredient_photos"] = load_credits("ingredients")
    state["bottle_photos"] = load_bottle_photos()
    state["ingredient_info"] = yaml.safe_load(
        open(os.path.join(ROOT, "data", "ingredient_info.yaml"), encoding="utf-8"))["ingredients"]

    rel = os.path.relpath(weights, os.path.join(ROOT, "runs")).replace("\\", "/")
    parts = rel.split("/")
    state["default_key"] = f"{parts[0]}/{os.path.splitext(parts[-1])[0]}"
    print(f"cocktail photos: {len(state['photos'])}, ingredient photos: {len(state['ingredient_photos'])}")
    print(f"checkpoints available: {len(discover_models())}")
    if get_model(state["default_key"]) is None:
        raise SystemExit(f"could not load {state['default_key']}")


def load_credits(subdir: str):
    """slug -> photo path and the credit its licence requires.

    Shared by fetch_cocktail_images.py's output (app/static/cocktails) and
    fetch_ingredient_images.py's (app/static/ingredients) - same manifest shape, same policy.
    """
    path = os.path.join(STATIC, subdir, "attribution.csv")
    if not os.path.exists(path):
        return {}
    import csv

    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if os.path.exists(os.path.join(STATIC, subdir, row["file"])):
                out[row["slug"]] = {
                    "src": f"/static/{subdir}/{row['file']}",
                    "creator": row.get("creator") or "unknown",
                    "license": row.get("license") or "",
                    "license_url": row.get("license_url") or "",
                    "landing_url": row.get("landing_url") or "",
                }
    return out


def load_bottle_photos():
    """class name -> thumbnail URL, from scripts/build_bottle_thumbs.py's output."""
    d = os.path.join(STATIC, "bottles")
    if not os.path.isdir(d):
        return {}
    return {R.canon(os.path.splitext(fn)[0]): f"/static/bottles/{fn}"
            for fn in os.listdir(d) if fn.lower().endswith((".jpg", ".jpeg", ".png"))}


def bottle_photo(cls: str):
    return state["bottle_photos"].get(R.canon(cls))


def crop_data_uri(image: Image.Image, box, pad: float = 0.06) -> str:
    """The bottle exactly as it appeared in this shelf, not a generic stock shot - so the
    'identified bottles' list reads as a receipt of what was found, not a product catalogue."""
    W, H = image.size
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    crop = image.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                       min(W, int(x2 + px)), min(H, int(y2 + py))))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ingredient_info(key: str):
    """Category + blurb for a canonical ingredient key, with a generic fallback."""
    info = state["ingredient_info"].get(key)
    if info:
        return info
    return {"category": "재료", "blurb": f"{key.replace('_', ' ')} 계열의 재료입니다."}


def cocktails_using(ingredient: str):
    """Every IBA cocktail this ingredient can help make, direct match first."""
    rules, recipes = state["rules"], state["recipes"]
    direct, sub = [], []
    for c in recipes:
        bar, _, _ = R.recipe_requirements(c, rules)
        for key, _line in bar:
            if key == ingredient:
                direct.append(c)
                break
            if ingredient in rules.equivalents.get(key, ()) or ingredient in rules.substitutes.get(key, ()):
                sub.append(c)
                break
    entry = lambda c: {"name": c["name"], "slug": c["slug"], "category": c["category"],
                       "photo": state["photos"].get(c["slug"])}
    seen = {c["slug"] for c in direct}
    return ([entry(c) for c in sorted(direct, key=lambda c: c["name"])],
            [entry(c) for c in sorted(sub, key=lambda c: c["name"]) if c["slug"] not in seen])


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def analyse(image: Image.Image, conf: float, two_stage: bool = True, model=None):
    """Name the bottles in a photo and draw the result.

    One stage runs the detector straight at the photo, which is the honest end-to-end path.
    Two stage lets a COCO detector propose bottles first and runs the brand model on each crop,
    which recovers bottles that are too small at shelf scale for the detector to fire on.
    """
    brand, coco, rules = model["yolo"], state["coco"], state["rules"]
    W, H = image.size

    if not two_stage:
        detected, declined, boxes = [], 0, []
        for r in brand.predict(image, conf=conf, verbose=False):
            for cls, score, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                        r.boxes.xyxy.tolist()):
                box = tuple(int(v) for v in xyxy)
                boxes.append(box)
                name = r.names[int(cls)]
                if name == R.ABSTAIN:
                    declined += 1
                    continue
                detected.append({"cls": name, "conf": round(float(score), 3), "box": box})
        return draw(image, detected, rules), detected, declined, len(boxes)

    crops, boxes = [], []
    for r in coco.predict(image, conf=0.25, verbose=False):
        for cls, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            if int(cls) != COCO_BOTTLE:
                continue
            x1, y1, x2, y2 = xyxy
            px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
            box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
                   min(W, int(x2 + px)), min(H, int(y2 + py)))
            if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                continue
            crops.append(image.crop(box).resize((640, 640)))
            boxes.append(box)

    detected, declined = [], 0
    for i in range(0, len(crops), 32):
        chunk = crops[i:i + 32]
        for j, r in enumerate(brand.predict(chunk, conf=conf, verbose=False)):
            if not len(r.boxes):
                declined += 1
                continue
            best = r.boxes.conf.argmax()
            name = r.names[int(r.boxes.cls[best])]
            if name == R.ABSTAIN:
                declined += 1
                continue
            detected.append({"cls": name, "conf": round(float(r.boxes.conf[best]), 3),
                             "box": boxes[i + j]})
    return draw(image, detected, rules), detected, declined, len(boxes)


def draw(image, detected, rules):
    """Draw a box only for bottles the model actually named - an unlabeled box (the model saw
    a bottle but wouldn't commit to a brand) would just read as visual noise to someone who
    isn't debugging the detector."""
    W = image.width

    canvas = image.copy()
    pen = ImageDraw.Draw(canvas)
    label_font = font(max(14, W // 60))
    for hit in detected:
        box = hit["box"]
        label = rules.bottle(hit["cls"]).get("label", hit["cls"])
        text = f"{label} {hit['conf']:.2f}"
        pen.rectangle(box, outline=NAMED, width=max(3, W // 300))
        tb = pen.textbbox((0, 0), text, font=label_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, box[1] - th - 8)
        pen.rectangle([box[0], ty, box[0] + tw + 10, ty + th + 8], fill=NAMED)
        pen.text((box[0] + 5, ty + 3), text, fill=(255, 255, 255), font=label_font)
    return canvas


def match_cocktails(classes, loose: bool):
    rules, recipes = state["rules"], state["recipes"]
    have = {rules.bottle(c)["ingredient"] for c in classes if rules.bottle(c)}
    makeable, nearly = [], []
    for c in recipes:
        bar, _, _ = R.recipe_requirements(c, rules)
        missing, subs, satisfied = [], [], 0
        for key, _line in bar:
            src = rules.satisfied_by(key, have, loose)
            if src is None:
                missing.append(key)
                continue
            satisfied += 1
            if src != key:
                subs.append(f"{key} → {src}")
        entry = {"name": c["name"], "slug": c["slug"], "category": c["category"],
                 "ingredients": c["ingredients"], "method": c["method"],
                 "garnish": c["garnish"], "url": c["url"], "subs": subs,
                 "photo": state["photos"].get(c["slug"])}
        if not missing:
            makeable.append(entry)
        elif len(missing) <= 2 and satisfied:
            # Without `satisfied`, a shelf holding only scotch is told it is "one bottle away"
            # from a Bellini - true, and useless. A near miss has to build on what is there.
            nearly.append({**entry, "missing": sorted(missing)})
    makeable.sort(key=lambda x: (x["category"], x["name"]))
    nearly.sort(key=lambda x: (len(x["missing"]), x["name"]))
    return have, makeable, nearly[:12]


@app.get("/")
def index():
    # The page is edited while the server runs; a cached copy hides the change and looks like
    # a broken feature.
    return FileResponse(os.path.join(STATIC, "index.html"),
                        headers={"Cache-Control": "no-store"})


@app.get("/api/info")
def info():
    meta = loaded[state["default_key"]]
    return {"classes": meta["classes"], "abstains": meta["abstains"],
            "recipes": len(state["recipes"]), "bottles": len(state["rules"].bottles),
            "default_conf": meta["default_conf"], "model": state["default_key"],
            "level": "ingredient" if meta["ingredient_level"] else "brand"}


@app.get("/api/models")
def models():
    """Every checkpoint that can be selected, with what is already known about it."""
    out = []
    for m in discover_models():
        known = loaded.get(m["key"])
        out.append({"key": m["key"], "run": m["run"], "tag": m["tag"],
                    "loaded": known is not None,
                    "classes": known["classes"] if known else None,
                    "level": ("ingredient" if known["ingredient_level"] else "brand")
                             if known else None,
                    "default_conf": known["default_conf"] if known else None})
    return {"models": out, "current": state["default_key"]}


@app.get("/api/samples")
def samples():
    """Test-set photos, offered so the demo can be tried without finding a bar."""
    d = os.path.join(ROOT, "testset", "images")
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))[:12]


@app.get("/samples/{name}")
def sample(name: str):
    d = os.path.join(ROOT, "testset", "images")
    path = os.path.normpath(os.path.join(d, name))
    if not path.startswith(os.path.abspath(d)) or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.0, loose: bool = False,
                 two_stage: bool = False, model: str = ""):
    meta = get_model(model or state["default_key"])
    if meta is None:
        return JSONResponse({"error": f"no such checkpoint: {model}"}, status_code=400)

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "could not read that file as an image"}, status_code=400)

    if max(image.size) > 1600:            # keep inference and the response payload sane
        image.thumbnail((1600, 1600))

    canvas, detected, declined, total = analyse(image, conf or DEFAULT_CONF, two_stage, meta)
    classes = [d["cls"] for d in detected]
    have, makeable, nearly = match_cocktails(classes, loose)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=88)
    rules = state["rules"]

    seen, bottles = set(), []
    for d in sorted(detected, key=lambda x: -x["conf"]):
        if d["cls"] in seen:
            continue
        seen.add(d["cls"])
        entry = rules.bottle(d["cls"])
        bottles.append({"cls": d["cls"],
                        "label": entry.get("label", d["cls"]),
                        "ingredient": entry.get("ingredient", "?"),
                        "photo": bottle_photo(d["cls"]),
                        "crop": crop_data_uri(image, d["box"]),
                        "conf": d["conf"]})

    return {
        "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        "bottles_found": total,
        "named": len(detected),
        "declined": declined,
        "two_stage": two_stage,
        "model": meta["key"],
        "classes": meta["classes"],
        "level": "ingredient" if meta["ingredient_level"] else "brand",
        "default_conf": meta["default_conf"],
        "abstains": meta["abstains"],
        "bottles": bottles,
        "ingredients": sorted(have),
        "makeable": makeable,
        "nearly": nearly,
    }


@app.get("/api/bottle/{cls}")
def bottle_detail(cls: str):
    rules = state["rules"]
    entry = rules.bottle(cls)
    if not entry:
        return JSONResponse({"error": f"unknown bottle class: {cls}"}, status_code=404)
    ingredient = entry.get("ingredient", "")
    info = ingredient_info(ingredient)
    direct, sub = cocktails_using(ingredient)

    photo, credit = bottle_photo(cls), None
    if photo is None:
        # No brand-specific crop (a generic ingredient page, e.g. reached from a recipe's
        # "45 ml Gin" rather than a detected bottle) - fall back to a licensed stock photo,
        # which needs the credit a brand crop from our own training data does not.
        fallback = state["ingredient_photos"].get(ingredient)
        if fallback:
            photo, credit = fallback["src"], fallback

    return {"cls": cls, "label": entry.get("label", cls), "ingredient": ingredient,
            "category": info["category"], "blurb": info["blurb"],
            "photo": photo, "credit": credit, "cocktails": direct, "cocktails_sub": sub}


@app.get("/api/cocktail/{slug}")
def cocktail_detail(slug: str):
    c = state["recipes_by_slug"].get(slug)
    if c is None:
        return JSONResponse({"error": f"unknown cocktail: {slug}"}, status_code=404)
    rules = state["rules"]
    ingredients = []
    for line in c["ingredients"]:
        key, _ = rules.normalize(line)
        bottle_key = key if key and key not in rules.pantry else None
        ingredients.append({"text": line, "key": bottle_key})
    return {"name": c["name"], "slug": c["slug"], "category": c["category"],
            "ingredients": ingredients, "method": c["method"], "garnish": c["garnish"],
            "url": c["url"], "photo": state["photos"].get(c["slug"])}


def main():
    ap = argparse.ArgumentParser()
    # yolo11s_v10/epoch80 chosen by eye over best.pt: at conf 0.4 it reads shelves better than
    # the run's own best-mAP checkpoint does. Falls back down the list if that file is missing.
    default = next((p for p in (
                        os.path.join(ROOT, "runs", "yolo11s_v10", "weights", "epoch80.pt"),
                        *(os.path.join(ROOT, "runs", r, "weights", "best.pt")
                          for r in ("yolo11s_v10", "yolo11s_ing", "yolo11s_v3",
                                    "yolo11s_v2", "yolo11s")))
                    if os.path.exists(p)), "")
    ap.add_argument("--weights", default=default)
    ap.add_argument("--coco-weights", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    load_models(args.weights, args.coco_weights)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    import uvicorn

    print(f"\n  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
