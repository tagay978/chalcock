"""Render what the detector does on real test photos, against the annotations.

Colour tells you the outcome at a glance, which a metrics table cannot:

    green   named correctly
    red     named something else - the failure that invents ingredients
    blue    abstained (said "a bottle, not one of mine")
    grey    annotated bottle the model said nothing about

With --compare, two models are drawn side by side on the same photo, so the effect of adding
the abstain class is visible rather than inferred.

Run:  python scripts/show_predictions.py --out samples/
      python scripts/show_predictions.py --compare runs/yolo11s_abstain/weights/best.pt
"""
from __future__ import annotations

import argparse
import os

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET = os.path.join(ROOT, "testset")
COCO_BOTTLE = 39
ABSTAIN = "unknown_bottle"
# Crop the bottle exactly as the COCO detector boxed it. Padding used to add 15% on
# each side; it pulls in whatever stands next to the bottle, which on a packed shelf is
# another bottle.
PAD = 0.0
IOU_MATCH = 0.5

CORRECT = (34, 197, 94)
WRONG = (239, 68, 68)
DECLINED = (59, 130, 246)
MISSED = (148, 163, 184)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def truth_boxes(stem, names, W, H):
    path = os.path.join(TESTSET, "labels", stem + ".txt")
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:5])
        out.append((names[cid] if cid < len(names) else str(cid),
                    ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)))
    return out


def predict(image_path, brand, coco, conf, two_stage):
    """Return [(class_name, (x1,y1,x2,y2), confidence)] in pixel coordinates."""
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    if not two_stage:
        out = []
        for r in brand.predict(image_path, conf=conf, verbose=False):
            for cid, c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                    r.boxes.xyxy.tolist()):
                out.append((r.names[int(cid)], tuple(xyxy), c))
        return out

    regions, origins = [], []
    for r in coco.predict(image_path, conf=0.25, verbose=False):
        for cid, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            if int(cid) != COCO_BOTTLE:
                continue
            x1, y1, x2, y2 = xyxy
            px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
            box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
                   min(W, int(x2 + px)), min(H, int(y2 + py)))
            if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                continue
            regions.append(im.crop(box).resize((640, 640)))
            origins.append(box)

    out = []
    for i in range(0, len(regions), 32):
        for j, r in enumerate(brand.predict(regions[i:i + 32], conf=conf, verbose=False)):
            if not len(r.boxes):
                continue
            best = r.boxes.conf.argmax()
            out.append((r.names[int(r.boxes.cls[best])], origins[i + j], float(r.boxes.conf[best])))
    return out


def render(image_path, stem, names, preds, title, width=900):
    im = Image.open(image_path).convert("RGB")
    scale = width / im.width
    im = im.resize((width, int(im.height * scale)))
    gt = truth_boxes(stem, names, im.width / scale, im.height / scale)

    draw = ImageDraw.Draw(im)
    small, big = font(15), font(17)
    matched = set()

    for name, box, conf in preds:
        box = tuple(v * scale for v in box)
        best, best_iou = None, IOU_MATCH
        for i, (gname, gbox) in enumerate(gt):
            score = iou(box, tuple(v * scale for v in gbox))
            if i not in matched and score >= best_iou:
                best, best_iou = i, score
        if best is not None:
            matched.add(best)
        if name == ABSTAIN:
            colour, text = DECLINED, f"declined {conf:.2f}"
        elif best is not None and gt[best][0] == name:
            colour, text = CORRECT, f"{name} {conf:.2f}"
        else:
            colour, text = WRONG, f"{name} {conf:.2f}"
        draw.rectangle(box, outline=colour, width=4)
        tb = draw.textbbox((0, 0), text, font=small)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, box[1] - th - 6)
        draw.rectangle([box[0], ty, box[0] + tw + 8, ty + th + 6], fill=colour)
        draw.text((box[0] + 4, ty + 2), text, fill=(255, 255, 255), font=small)

    for i, (gname, gbox) in enumerate(gt):
        if i in matched:
            continue
        draw.rectangle(tuple(v * scale for v in gbox), outline=MISSED, width=2)

    banner = Image.new("RGB", (im.width, 30), (24, 24, 27))
    ImageDraw.Draw(banner).text((8, 6), title, fill=(255, 255, 255), font=big)
    out = Image.new("RGB", (im.width, im.height + 30), "white")
    out.paste(banner, (0, 0))
    out.paste(im, (0, 30))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "runs", "yolo11s_v10", "weights", "epoch80.pt"))
    ap.add_argument("--compare", help="second model, drawn beside the first")
    ap.add_argument("--coco-weights", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--direct", action="store_true", help="skip the COCO crop stage")
    ap.add_argument("--limit", type=int, default=6, help="how many photos to render")
    ap.add_argument("--out", default=os.path.join(ROOT, "samples"))
    args = ap.parse_args()

    names = yaml.safe_load(open(os.path.join(TESTSET, "data.yaml"), encoding="utf-8"))["names"]

    from ultralytics import YOLO

    models = [("before: 50 classes", YOLO(args.weights))]
    if args.compare:
        models.append(("after: + abstain", YOLO(args.compare)))
    coco = None if args.direct else YOLO(args.coco_weights)

    # Show the photos that actually contain bottles the model was trained on; a photo of pure
    # out-of-vocabulary bottles shows nothing about naming.
    scored = []
    for fn in sorted(os.listdir(os.path.join(TESTSET, "images"))):
        stem = os.path.splitext(fn)[0]
        gt = truth_boxes(stem, names, 1, 1)
        known = sum(1 for n, _ in gt if n != ABSTAIN)
        scored.append((known, len(gt), fn))
    scored.sort(reverse=True)

    os.makedirs(args.out, exist_ok=True)
    written = []
    for known, total, fn in scored[:args.limit]:
        stem = os.path.splitext(fn)[0]
        path = os.path.join(TESTSET, "images", fn)
        panels = []
        for label, model in models:
            preds = predict(path, model, coco, args.conf, not args.direct)
            panels.append(render(path, stem, names, preds,
                                 f"{label}   |   {known} in-vocabulary of {total} bottles"))
        if len(panels) == 1:
            sheet = panels[0]
        else:
            gap = 12
            sheet = Image.new("RGB", (sum(p.width for p in panels) + gap,
                                      max(p.height for p in panels)), "white")
            x = 0
            for p in panels:
                sheet.paste(p, (x, 0))
                x += p.width + gap
        dest = os.path.join(args.out, f"{stem}.png")
        sheet.save(dest)
        written.append(dest)
        print(f"  {dest}  ({known} in-vocabulary of {total})")

    print(f"\nwrote {len(written)} samples to {args.out}")
    print("  green = named right, red = named wrong, blue = abstained, grey = nothing said")


if __name__ == "__main__":
    main()
