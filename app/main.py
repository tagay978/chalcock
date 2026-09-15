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

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import recommend as R  # noqa: E402  - path set above

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
COCO_BOTTLE = 39
PAD = 0.15
NAMED = (34, 197, 94)
DECLINED = (100, 116, 139)

app = FastAPI(title="Cocktail bottle detector")
state: dict = {}


def load_models(weights: str, coco_weights: str):
    from ultralytics import YOLO

    if not os.path.exists(weights):
        raise SystemExit(f"weights not found: {weights}")
    state["brand"] = YOLO(weights)
    state["coco"] = YOLO(coco_weights)
    state["rules"] = R.Rules()
    state["recipes"] = R.json.load(
        open(os.path.join(ROOT, "data", "iba_cocktails.json"), encoding="utf-8"))["cocktails"]
    state["abstains"] = R.ABSTAIN in state["brand"].names.values()
    state["photos"] = load_photo_credits()
    print(f"cocktail photos: {len(state['photos'])}")
    print(f"brand model: {os.path.relpath(weights, ROOT)} "
          f"({len(state['brand'].names)} classes, "
          f"{'can abstain' if state['abstains'] else 'no abstain class'})")


def load_photo_credits():
    """slug -> photo path and the credit its licence requires, from fetch_cocktail_images.py."""
    path = os.path.join(STATIC, "cocktails", "attribution.csv")
    if not os.path.exists(path):
        return {}
    import csv

    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if os.path.exists(os.path.join(STATIC, "cocktails", row["file"])):
                out[row["slug"]] = {
                    "src": f"/static/cocktails/{row['file']}",
                    "creator": row.get("creator") or "unknown",
                    "license": row.get("license") or "",
                    "license_url": row.get("license_url") or "",
                    "landing_url": row.get("landing_url") or "",
                }
    return out


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def analyse(image: Image.Image, conf: float, two_stage: bool = False):
    """Name the bottles in a photo and draw the result.

    One stage runs the detector straight at the photo, which is the honest end-to-end path.
    Two stage lets a COCO detector propose bottles first and runs the brand model on each crop,
    which recovers bottles that are too small at shelf scale for the detector to fire on.
    """
    brand, coco, rules = state["brand"], state["coco"], state["rules"]
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
        return draw(image, boxes, detected, rules), detected, declined, len(boxes)

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
    return draw(image, boxes, detected, rules), detected, declined, len(boxes)


def draw(image, boxes, detected, rules):
    W = image.width

    canvas = image.copy()
    pen = ImageDraw.Draw(canvas)
    label_font = font(max(14, W // 60))
    for box in boxes:
        hit = next((d for d in detected if d["box"] == box), None)
        if hit is None:
            pen.rectangle(box, outline=DECLINED, width=2)
            continue
        label = rules.bottles.get(hit["cls"], {}).get("label", hit["cls"])
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
    have = {rules.bottles[c]["ingredient"] for c in classes if c in rules.bottles}
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
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/info")
def info():
    rules = state["rules"]
    return {"classes": len(state["brand"].names), "abstains": state["abstains"],
            "recipes": len(state["recipes"]), "bottles": len(rules.bottles)}


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
async def detect(file: UploadFile = File(...), conf: float = 0.25, loose: bool = False,
                 two_stage: bool = False):
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "could not read that file as an image"}, status_code=400)

    if max(image.size) > 1600:            # keep inference and the response payload sane
        image.thumbnail((1600, 1600))

    canvas, detected, declined, total = analyse(image, conf, two_stage)
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
        bottles.append({"cls": d["cls"],
                        "label": rules.bottles.get(d["cls"], {}).get("label", d["cls"]),
                        "ingredient": rules.bottles.get(d["cls"], {}).get("ingredient", "?"),
                        "conf": d["conf"]})

    return {
        "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        "bottles_found": total,
        "named": len(detected),
        "declined": declined,
        "two_stage": two_stage,
        "abstains": state["abstains"],
        "bottles": bottles,
        "ingredients": sorted(have),
        "makeable": makeable,
        "nearly": nearly,
    }


def main():
    ap = argparse.ArgumentParser()
    default_abstain = os.path.join(ROOT, "runs", "yolo11s_abstain", "weights", "best.pt")
    default_plain = os.path.join(ROOT, "runs", "yolo11s", "weights", "best.pt")
    ap.add_argument("--weights", default=default_abstain if os.path.exists(default_abstain)
                    else default_plain)
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
