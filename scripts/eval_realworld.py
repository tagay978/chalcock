"""Measure the detector against the hand-annotated real-world test set.

Standard mAP answers "how well does it box the 50 classes" and stays silent on the failure this
project actually has: a bar is full of bottles that are not any of the 50, and the model has no
way to say so. So this reports two things side by side.

  known bottles      - of the annotated bottles that ARE one of the 50, how many were found and
                       named correctly (localisation and classification split apart, because
                       they fail for different reasons and need different fixes)
  unknown bottles    - of the annotated bottles that are NOT, how many drew a confident
                       known-class prediction anyway. This is the open-set false alarm rate and
                       it is what makes the recommender hallucinate ingredients.

Both inference paths are evaluated, since --two-stage changes the scale the model sees.

Run:  python scripts/eval_realworld.py
      python scripts/eval_realworld.py --conf 0.25,0.4,0.5 --paths direct,two-stage
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET = os.path.join(ROOT, "testset")
UNKNOWN = "unknown_bottle"
COCO_BOTTLE = 39
IOU_MATCH = 0.5


def to_xyxy(box):
    _, x, y, w, h = box
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def load_truth(names, trained):
    """Read the annotations, split per image into in-vocabulary and out-of-vocabulary boxes.

    Out-of-vocabulary is wider than the `unknown_bottle` proposal class: an annotator who
    recognises a bottle the model was never trained on (Campari, Jack Daniel's) may well name it,
    and counting those as missed detections would punish the model for lacking an output it
    never had. Anything the model cannot emit belongs on the open-set side.
    """
    truth = {}
    ldir = os.path.join(TESTSET, "labels")
    for fn in sorted(os.listdir(os.path.join(TESTSET, "images"))):
        stem = os.path.splitext(fn)[0]
        path = os.path.join(ldir, stem + ".txt")
        known, unknown = [], []
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                box = (cid, *(float(v) for v in parts[1:5]))
                in_vocab = cid < len(names) and names[cid] in trained
                (known if in_vocab else unknown).append(box)
        truth[fn] = (known, unknown)
    return truth


def predict_direct(model, path, conf):
    out = []
    for r in model.predict(path, conf=conf, verbose=False):
        for cid, c, xywhn in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                 r.boxes.xywhn.tolist()):
            out.append((int(cid), *xywhn, c))
    return out


def predict_two_stage(coco, brand, path, conf, pad=0.15):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    W, H = im.size
    regions, origins = [], []
    for r in coco.predict(path, conf=0.25, verbose=False):
        for cid, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            if int(cid) != COCO_BOTTLE:
                continue
            x1, y1, x2, y2 = xyxy
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
                   min(W, int(x2 + px)), min(H, int(y2 + py)))
            if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                continue
            regions.append(im.crop(box).resize((640, 640)))
            # Report the crop in normalised image coordinates so it matches the annotation.
            origins.append((((box[0] + box[2]) / 2) / W, ((box[1] + box[3]) / 2) / H,
                            (box[2] - box[0]) / W, (box[3] - box[1]) / H))

    out = []
    for i in range(0, len(regions), 32):
        chunk = regions[i:i + 32]
        for j, r in enumerate(brand.predict(chunk, conf=conf, verbose=False)):
            if not len(r.boxes):
                continue
            best = r.boxes.conf.argmax()
            out.append((int(r.boxes.cls[best]), *origins[i + j], float(r.boxes.conf[best])))
    return out


def evaluate(truth, predictions, names=None):
    """Match predictions to annotations and count the four outcomes that matter.

    Prediction ids are the model's, ground-truth ids are the test set's; they agree because
    testset/labels/classes.txt was written from the trained class list in trained order, with
    the extra classes appended after it.
    """
    stats = dict(known=0, localised=0, correct=0, unknown=0, unknown_hit=0,
                 preds=0, matched_pred=0)

    for fn, (known, unknown) in truth.items():
        preds = predictions.get(fn, [])
        stats["known"] += len(known)
        stats["unknown"] += len(unknown)
        stats["preds"] += len(preds)

        used = set()
        for gt in known:
            gbox = to_xyxy(gt)
            best, best_iou = None, IOU_MATCH
            for i, p in enumerate(preds):
                if i in used:
                    continue
                score = iou(gbox, to_xyxy(p[:5]))
                if score >= best_iou:
                    best, best_iou = i, score
            if best is not None:
                used.add(best)
                stats["localised"] += 1
                stats["matched_pred"] += 1
                if preds[best][0] == gt[0]:
                    stats["correct"] += 1

        for gt in unknown:
            gbox = to_xyxy(gt)
            for i, p in enumerate(preds):
                if i in used:
                    continue
                if iou(gbox, to_xyxy(p[:5])) >= IOU_MATCH:
                    used.add(i)
                    stats["matched_pred"] += 1
                    stats["unknown_hit"] += 1
                    break
    return stats


def report(label, stats):
    k, u = stats["known"], stats["unknown"]
    pct = lambda n, d: f"{100 * n / d:5.1f}%" if d else "    - "
    print(f"\n  {label}")
    print(f"    known bottles annotated      {k}")
    print(f"      found (IoU>={IOU_MATCH})          {stats['localised']:4}  {pct(stats['localised'], k)}")
    print(f"      found AND named right      {stats['correct']:4}  {pct(stats['correct'], k)}")
    print(f"      naming accuracy when found {'':4}  {pct(stats['correct'], stats['localised'])}")
    print(f"    unknown bottles annotated    {u}")
    print(f"      given a known-class label  {stats['unknown_hit']:4}  {pct(stats['unknown_hit'], u)}   <- false alarms")
    print(f"    predictions made             {stats['preds']}")
    print(f"      precision                  {'':4}  {pct(stats['correct'], stats['preds'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "yolo11s", "weights", "best.pt"))
    ap.add_argument("--coco-weights", default=os.path.join(ROOT, "weights", "yolo11m.pt"))
    ap.add_argument("--conf", default="0.25,0.5")
    ap.add_argument("--paths", default="direct,two-stage")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(TESTSET, "images")):
        raise SystemExit(f"{TESTSET} not found - run scripts/prepare_testset.py first")

    names = yaml.safe_load(open(os.path.join(TESTSET, "data.yaml"), encoding="utf-8"))["names"]

    from ultralytics import YOLO

    brand = YOLO(args.weights)
    trained = set(brand.names.values())

    truth = load_truth(names, trained)
    total_known = sum(len(k) for k, _ in truth.values())
    total_unknown = sum(len(u) for _, u in truth.values())
    print(f"test set: {len(truth)} photos, {total_known} in-vocabulary bottles, "
          f"{total_unknown} out-of-vocabulary")

    # Bottles the annotator named that the model has no output for: the concrete shortlist of
    # classes worth adding.
    named_oov = Counter()
    for _, unknown in truth.values():
        for gt in unknown:
            label = names[gt[0]] if gt[0] < len(names) else str(gt[0])
            if label != UNKNOWN:
                named_oov[label] += 1
    if named_oov:
        print("  named but not a trained class: "
              + ", ".join(f"{k} x{v}" for k, v in named_oov.most_common()))

    if total_known == 0:
        print("\nNo in-vocabulary bottles annotated - nothing to measure.")
        return
    coco = YOLO(args.coco_weights) if "two-stage" in args.paths else None

    for conf in [float(c) for c in args.conf.split(",")]:
        print(f"\n=== conf {conf} ===")
        for path_name in [p.strip() for p in args.paths.split(",")]:
            preds = {}
            for fn in truth:
                image = os.path.join(TESTSET, "images", fn)
                preds[fn] = (predict_direct(brand, image, conf) if path_name == "direct"
                             else predict_two_stage(coco, brand, image, conf))
            report(path_name, evaluate(truth, preds, names))


if __name__ == "__main__":
    main()
