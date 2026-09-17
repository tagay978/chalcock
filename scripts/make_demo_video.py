"""Render a demo video of the pipeline running on real test photos.

Made for a portfolio: a title card, then one segment per photo showing the bottles being found,
named or declined, and the cocktails that fall out of the result, then a closing card with what
this run actually found - not a hand-labelled accuracy number, since the current testset has no
ground truth to measure against yet.

Everything shown is computed by the same code the CLI and the web app use, so the video cannot
show a result the system does not actually produce.

Run:  python scripts/make_demo_video.py
      python scripts/make_demo_video.py --weights runs/yolo11s_abstain/weights/best.pt --n 6
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import recommend as R  # noqa: E402

TESTSET = os.path.join(ROOT, "testset", "images")
W, H = 1280, 720
FPS = 30
COCO_BOTTLE = 39
# Crop the bottle exactly as the COCO detector boxed it. Padding used to add 15% on
# each side; it pulls in whatever stands next to the bottle, which on a packed shelf is
# another bottle.
PAD = 0.0

BG = (13, 15, 20)
PANEL = (22, 26, 34)
LINE = (39, 45, 58)
TEXT = (232, 234, 240)
MUTED = (141, 149, 168)
ACCENT = (240, 160, 75)
GREEN = (34, 197, 94)
GREY = (100, 116, 139)


def font(size, bold=False):
    names = (["malgunbd.ttf", "arialbd.ttf"] if bold else ["malgun.ttf", "arial.ttf"])
    for name in names + ["DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_H1, F_BODY, F_SMALL = font(46, True), font(26, True), font(19), font(15)


def blank():
    return Image.new("RGB", (W, H), BG)


def write(frames, image, seconds):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for _ in range(int(seconds * FPS)):
        frames.write(frame)


def fade(frames, image, seconds=0.4, out=False):
    arr = np.array(image).astype(np.float32)
    steps = max(1, int(seconds * FPS))
    for i in range(steps):
        t = (i + 1) / steps
        if out:
            t = 1 - t
        blended = (arr * t + np.array(BG, dtype=np.float32) * (1 - t)).astype(np.uint8)
        frames.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def analyse(path, brand, coco, conf, two_stage=True):
    """Same pass the app runs. One stage is the detector end to end; two stage lets COCO
    propose bottles first and names each crop, which finds more at shelf scale."""
    image = Image.open(path).convert("RGB")
    iw, ih = image.size

    if not two_stage:
        results = []
        for r in brand.predict(path, conf=conf, verbose=False):
            for cls, score, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                        r.boxes.xyxy.tolist()):
                name = r.names[int(cls)]
                box = tuple(int(v) for v in xyxy)
                results.append((box, None, 0.0) if name == R.ABSTAIN
                               else (box, name, float(score)))
        return image, results

    crops, boxes = [], []
    for r in coco.predict(path, conf=0.25, verbose=False):
        for cls, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            if int(cls) != COCO_BOTTLE:
                continue
            x1, y1, x2, y2 = xyxy
            px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
            box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
                   min(iw, int(x2 + px)), min(ih, int(y2 + py)))
            if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                continue
            crops.append(image.crop(box).resize((640, 640)))
            boxes.append(box)

    results = []
    for i in range(0, len(crops), 32):
        for j, r in enumerate(brand.predict(crops[i:i + 32], conf=conf, verbose=False)):
            name, score = None, 0.0
            if len(r.boxes):
                best = r.boxes.conf.argmax()
                cand = r.names[int(r.boxes.cls[best])]
                if cand != R.ABSTAIN:
                    name, score = cand, float(r.boxes.conf[best])
            results.append((boxes[i + j], name, score))
    return image, results


def photo_panel(image, results, revealed, rules):
    """Left half: the photo with the first `revealed` boxes drawn."""
    box_w, box_h = 690, 560
    scale = min(box_w / image.width, box_h / image.height)
    shown = image.resize((int(image.width * scale), int(image.height * scale)))
    canvas = Image.new("RGB", (box_w, box_h), BG)
    ox, oy = (box_w - shown.width) // 2, (box_h - shown.height) // 2
    canvas.paste(shown, (ox, oy))

    draw = ImageDraw.Draw(canvas)
    for (box, name, score) in results[:revealed]:
        b = (box[0] * scale + ox, box[1] * scale + oy, box[2] * scale + ox, box[3] * scale + oy)
        if name is None:
            draw.rectangle(b, outline=GREY, width=2)
            continue
        draw.rectangle(b, outline=GREEN, width=3)
        label = rules.bottle(name).get("label", name)
        text = f"{label} {score:.2f}"
        tb = draw.textbbox((0, 0), text, font=F_SMALL)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, b[1] - th - 7)
        draw.rectangle([b[0], ty, b[0] + tw + 9, ty + th + 7], fill=GREEN)
        draw.text((b[0] + 4, ty + 2), text, fill=(255, 255, 255), font=F_SMALL)
    return canvas


def segment(frames, path, brand, coco, conf, rules, recipes, index, total, two_stage):
    image, results = analyse(path, brand, coco, conf, two_stage)
    named = [(b, n, s) for b, n, s in results if n]
    classes = [n for _, n, _ in named]
    have = {rules.bottle(c)["ingredient"] for c in classes if rules.bottle(c)}

    makeable = []
    for c in recipes:
        bar, _, _ = R.recipe_requirements(c, rules)
        satisfied, missing = 0, 0
        for key, _ in bar:
            if rules.satisfied_by(key, have, False) is None:
                missing += 1
            else:
                satisfied += 1
        if not missing and satisfied:
            makeable.append(c["name"])

    def build(revealed, show_results):
        frame = blank()
        draw = ImageDraw.Draw(frame)
        draw.text((48, 34), f"실사 테스트 사진 {index}/{total}", font=F_H1, fill=TEXT)
        draw.line([(48, 76), (W - 48, 76)], fill=LINE, width=1)
        frame.paste(photo_panel(image, results, revealed, rules), (48, 100))

        x = 790
        draw.rectangle([x, 100, W - 48, 660], fill=PANEL, outline=LINE)
        y = 122
        draw.text((x + 22, y), "검출", font=F_SMALL, fill=MUTED)
        y += 26
        draw.text((x + 22, y), f"{revealed} / {len(results)} 병", font=F_H1, fill=ACCENT)
        y += 48

        if show_results:
            draw.text((x + 22, y), "식별된 병", font=F_SMALL, fill=MUTED)
            y += 26
            if named:
                for _, name, score in named[:5]:
                    label = rules.bottle(name).get("label", name)
                    draw.text((x + 22, y), f"· {label}", font=F_BODY, fill=TEXT)
                    draw.text((x + 300, y + 2), f"{score:.2f}", font=F_SMALL, fill=GREEN)
                    y += 28
            else:
                draw.text((x + 22, y), "없음 — 전부 기권", font=F_BODY, fill=GREY)
                y += 28
            y += 14

            draw.text((x + 22, y), "재료", font=F_SMALL, fill=MUTED)
            y += 26
            draw.text((x + 22, y), ", ".join(sorted(have)) or "—", font=F_BODY, fill=TEXT)
            y += 42

            draw.text((x + 22, y), "만들 수 있는 IBA 칵테일", font=F_SMALL, fill=MUTED)
            y += 26
            if makeable:
                for name in makeable[:5]:
                    draw.text((x + 22, y), f"· {name}", font=F_BODY, fill=ACCENT)
                    y += 28
            else:
                draw.text((x + 22, y), "없음", font=F_BODY, fill=GREY)
        return frame

    fade(frames, build(0, False), 0.3)
    for k in range(1, len(results) + 1):
        write(frames, build(k, False), 0.09)
    write(frames, build(len(results), True), 3.2)
    fade(frames, build(len(results), True), 0.3, out=True)
    return len(named)


def pick_photos(n, brand, coco, conf, two_stage):
    """Prefer photos the model actually has something to say about.

    testset/labels carries no ground truth for the current photos (they were just swapped in
    fresh, see git log) - so instead of ranking by annotation, this runs the real detector once
    per photo and ranks by how many bottles it named. Alphabetical order would just as easily
    open on a wine-shop shelf with nothing in vocabulary, which makes for a dull demo even
    though it is an honest result.
    """
    files = sorted(f for f in os.listdir(TESTSET) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    scored = []
    for fn in files:
        _, results = analyse(os.path.join(TESTSET, fn), brand, coco, conf, two_stage)
        scored.append((sum(1 for _, name, _ in results if name), fn))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [fn for _, fn in scored[:n]]


def card(frames, lines, seconds=3.0):
    frame = blank()
    draw = ImageDraw.Draw(frame)
    y = H // 2 - 22 * len(lines)
    for text, f, colour in lines:
        tb = draw.textbbox((0, 0), text, font=f)
        draw.text(((W - (tb[2] - tb[0])) // 2, y), text, font=f, fill=colour)
        y += (tb[3] - tb[1]) + 26
    fade(frames, frame, 0.5)
    write(frames, frame, seconds)
    fade(frames, frame, 0.5, out=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "runs", "yolo11s_v10", "weights", "epoch80.pt"))
    ap.add_argument("--coco-weights", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--n", type=int, default=5, help="how many photos to show")
    ap.add_argument("--two-stage", dest="two_stage", action="store_true", default=False,
                    help="crop each COCO-proposed bottle before naming it, instead of running "
                         "the brand model on the full photo")
    ap.add_argument("--out", default=os.path.join(ROOT, "samples", "demo.mp4"))
    args = ap.parse_args()

    from ultralytics import YOLO

    brand, coco = YOLO(args.weights), YOLO(args.coco_weights)
    rules = R.Rules()
    recipes = R.json.load(open(os.path.join(ROOT, "data", "iba_cocktails.json"),
                               encoding="utf-8"))["cocktails"]

    picked = pick_photos(args.n, brand, coco, args.conf, args.two_stage)
    if not picked:
        raise SystemExit(f"no images in {TESTSET}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not frames.isOpened():
        raise SystemExit(f"could not open {args.out} for writing")

    card(frames, [
        ("찰칵 Chalcock", F_TITLE, ACCENT),
        ("술장 사진으로 만들 수 있는 칵테일 찾기", F_BODY, TEXT),
        ("YOLO11 bottle detection  ·  IBA 102 cocktails", F_SMALL, MUTED),
    ], 2.2)

    total_named = 0
    for i, fn in enumerate(picked, 1):
        print(f"  rendering {i}/{len(picked)}: {fn}")
        total_named += segment(frames, os.path.join(TESTSET, fn), brand, coco, args.conf, rules,
                               recipes, i, len(picked), args.two_stage)

    # No hand-labelled ground truth for the current testset yet (it was just replaced with fresh
    # shelf photos - see git log), so this closes on what the run actually did, not a number that
    # would otherwise go stale the moment the model or test photos change.
    run = os.path.basename(os.path.dirname(os.path.dirname(args.weights)))
    tag = os.path.splitext(os.path.basename(args.weights))[0]
    card(frames, [
        (f"{run}/{tag}  ·  conf {args.conf}  ·  {'2' if args.two_stage else '1'}단계 검출", F_H1, TEXT),
        (f"이 {len(picked)}장에서 브랜드 {total_named}개 식별", F_BODY, ACCENT),
        ("", F_SMALL, MUTED),
        ("github.com/tagay978/cocktail-bottle-detector", F_SMALL, MUTED),
    ], 4.0)

    frames.release()
    size = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}  ({size:.1f} MB, {len(picked)} photos)")


if __name__ == "__main__":
    main()
