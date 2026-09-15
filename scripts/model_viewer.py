"""Desktop viewer: pick a checkpoint, pick a photo, see what it detects.

The web demo can do this too, but it lives behind a browser cache that hides edits and makes a
working feature look broken. This is a plain Tk window with no such problem, and it is the
faster way to answer "which epoch actually reads my shelf".

Every runs/*/weights/*.pt is offered. Checkpoints load on demand and the last few stay resident,
since each costs GPU memory. Confidence follows the checkpoint: 52 ingredient classes work
around 0.4 where 113 brand classes want 0.05, and carrying one model's threshold to another
returns nothing at all.

Run:  python scripts/model_viewer.py
      python scripts/model_viewer.py --image path/to/shelf.jpg
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import recommend as R  # noqa: E402

COCO_BOTTLE = 39
PAD = 0.0            # crop the COCO box as drawn; see app/main.py
MAX_LOADED = 3
NAMED = (34, 197, 94)
DECLINED = (100, 116, 139)

BG = "#0d0f14"
PANEL = "#161a22"
TEXT = "#e8eaf0"
MUTED = "#8d95a8"
ACCENT = "#f0a04b"


def discover():
    runs = os.path.join(ROOT, "runs")
    out = []
    if not os.path.isdir(runs):
        return out
    for run in sorted(os.listdir(runs)):
        wdir = os.path.join(runs, run, "weights")
        if not os.path.isdir(wdir):
            continue
        tags = []
        for fn in os.listdir(wdir):
            if fn.endswith(".pt"):
                tag = fn[:-3]
                order = (0 if tag == "best" else 1 if tag == "last" else 2,
                         int(tag[5:]) if tag.startswith("epoch") and tag[5:].isdigit() else 0)
                tags.append((order, tag, os.path.join(wdir, fn)))
        for _, tag, path in sorted(tags):
            out.append((f"{run} / {tag}", path))
    return out


class Viewer:
    def __init__(self, root, start_image=None):
        self.root = root
        self.rules = R.Rules()
        self.recipes = R.json.load(
            open(os.path.join(ROOT, "data", "iba_cocktails.json"), encoding="utf-8"))["cocktails"]
        self.models = OrderedDict()
        self.coco = None
        self.image_path = start_image
        self.checkpoints = discover()
        self.build()
        if start_image:
            self.run()

    # ---------- layout ----------
    def build(self):
        self.root.title("Shelf to Cocktail - model viewer")
        self.root.configure(bg=BG)
        self.root.geometry("1180x780")

        bar = tk.Frame(self.root, bg=PANEL, padx=12, pady=10)
        bar.pack(fill="x")

        tk.Label(bar, text="모델", bg=PANEL, fg=MUTED).pack(side="left")
        self.model_var = tk.StringVar(value=self.default_choice())
        self.model_box = ttk.Combobox(bar, textvariable=self.model_var, width=34,
                                      state="readonly",
                                      values=[name for name, _ in self.checkpoints])
        self.model_box.pack(side="left", padx=(6, 16))
        self.model_box.bind("<<ComboboxSelected>>", lambda _e: self.run())

        tk.Label(bar, text="신뢰도", bg=PANEL, fg=MUTED).pack(side="left")
        self.conf = tk.DoubleVar(value=0.4)
        self.conf_label = tk.Label(bar, text="0.40", bg=PANEL, fg=ACCENT, width=5)
        tk.Scale(bar, from_=0.01, to=0.8, resolution=0.01, orient="horizontal",
                 variable=self.conf, showvalue=False, length=150, bg=PANEL, fg=TEXT,
                 troughcolor=BG, highlightthickness=0,
                 command=lambda v: self.conf_label.config(text=f"{float(v):.2f}")
                 ).pack(side="left", padx=6)
        self.conf_label.pack(side="left", padx=(0, 6))
        tk.Button(bar, text="다시 실행", command=self.run, bg=PANEL, fg=TEXT).pack(side="left")

        self.two_stage = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="2단계 검출", variable=self.two_stage, bg=PANEL, fg=MUTED,
                       selectcolor=BG, activebackground=PANEL, command=self.run
                       ).pack(side="left", padx=12)

        tk.Button(bar, text="사진 열기…", command=self.pick_image,
                  bg=ACCENT, fg="#1a1205").pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Label(body, bg=BG)
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        side = tk.Frame(body, bg=PANEL, width=380)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self.info = tk.Label(side, text="사진을 열어 주세요", bg=PANEL, fg=ACCENT,
                             font=("Malgun Gothic", 11, "bold"), justify="left", anchor="w",
                             wraplength=350)
        self.info.pack(fill="x", padx=14, pady=(14, 6))
        self.detail = tk.Text(side, bg=PANEL, fg=TEXT, relief="flat", wrap="word",
                              font=("Malgun Gothic", 10), padx=12)
        self.detail.pack(fill="both", expand=True, pady=(0, 12))

    def default_choice(self):
        for want in ("yolo11s_ing / best", "yolo11s_fam / best"):
            if any(name == want for name, _ in self.checkpoints):
                return want
        return self.checkpoints[0][0] if self.checkpoints else ""

    # ---------- models ----------
    def model(self, name):
        if name in self.models:
            self.models.move_to_end(name)
            return self.models[name]
        path = dict(self.checkpoints)[name]
        from ultralytics import YOLO

        yolo = YOLO(path)
        classes = set(yolo.names.values())
        ingredient = len(classes & self.rules._ingredients) > len(classes) / 2
        meta = {"yolo": yolo, "classes": len(classes), "ingredient": ingredient,
                "conf": 0.4 if ingredient else 0.25 if len(classes) <= 60 else 0.05}
        self.models[name] = meta
        while len(self.models) > MAX_LOADED:
            self.models.popitem(last=False)
        return meta

    # ---------- actions ----------
    def pick_image(self):
        path = filedialog.askopenfilename(
            title="술장 사진 선택",
            initialdir=os.path.join(ROOT, "testset", "images"),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")])
        if path:
            self.image_path = path
            self.run()

    def run(self):
        if not self.image_path:
            return
        self.info.config(text="분석 중…")
        self.root.update_idletasks()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            meta = self.model(self.model_var.get())
            # A checkpoint the user has not tried yet gets its own threshold rather than the
            # previous model's, which is usually wrong by an order of magnitude.
            if self.model_var.get() not in getattr(self, "_seen", set()):
                self._seen = getattr(self, "_seen", set()) | {self.model_var.get()}
                self.conf.set(meta["conf"])
                self.conf_label.config(text=f"{meta['conf']:.2f}")
            result = self.analyse(meta, float(self.conf.get()))
        except Exception as exc:                      # keep the window alive on a bad checkpoint
            self.root.after(0, lambda: self.info.config(text=f"실패: {exc}"))
            return
        self.root.after(0, lambda: self.show(*result, meta))

    def analyse(self, meta, conf):
        brand = meta["yolo"]
        image = Image.open(self.image_path).convert("RGB")
        if max(image.size) > 1600:
            image.thumbnail((1600, 1600))
        W, H = image.size

        if not self.two_stage.get():
            boxes, detected = [], []
            for r in brand.predict(image, conf=conf, verbose=False):
                for cls, score, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                            r.boxes.xyxy.tolist()):
                    box = tuple(int(v) for v in xyxy)
                    boxes.append(box)
                    detected.append((r.names[int(cls)], float(score), box))
            return image, boxes, detected

        if self.coco is None:
            from ultralytics import YOLO
            self.coco = YOLO(os.path.join(ROOT, "weights", "yolo11m.pt"))

        crops, boxes = [], []
        for r in self.coco.predict(image, conf=0.25, verbose=False):
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

        detected = []
        for i in range(0, len(crops), 32):
            for j, r in enumerate(brand.predict(crops[i:i + 32], conf=conf, verbose=False)):
                if not len(r.boxes):
                    continue
                best = r.boxes.conf.argmax()
                name = r.names[int(r.boxes.cls[best])]
                if name == R.ABSTAIN:
                    continue
                detected.append((name, float(r.boxes.conf[best]), boxes[i + j]))
        return image, boxes, detected

    # ---------- output ----------
    def show(self, image, boxes, detected, meta):
        canvas = image.copy()
        pen = ImageDraw.Draw(canvas)
        try:
            fnt = ImageFont.truetype("malgun.ttf", max(14, canvas.width // 55))
        except OSError:
            fnt = ImageFont.load_default()
        named_boxes = {b: (n, c) for n, c, b in detected}
        for box in boxes:
            hit = named_boxes.get(box)
            if hit is None:
                pen.rectangle(box, outline=DECLINED, width=2)
                continue
            label = self.rules.bottle(hit[0]).get("label", hit[0])
            text = f"{label} {hit[1]:.2f}"
            pen.rectangle(box, outline=NAMED, width=3)
            tb = pen.textbbox((0, 0), text, font=fnt)
            ty = max(0, box[1] - (tb[3] - tb[1]) - 8)
            pen.rectangle([box[0], ty, box[0] + (tb[2] - tb[0]) + 10, ty + (tb[3] - tb[1]) + 8],
                          fill=NAMED)
            pen.text((box[0] + 5, ty + 3), text, fill=(255, 255, 255), font=fnt)

        fit = canvas.copy()
        fit.thumbnail((max(400, self.canvas.winfo_width() or 760),
                       max(400, self.canvas.winfo_height() or 700)))
        self.photo = ImageTk.PhotoImage(fit)
        self.canvas.config(image=self.photo)

        have, lines = set(), []
        for name, score, _ in sorted(detected, key=lambda d: -d[1]):
            entry = self.rules.bottle(name)
            if not entry:
                continue
            have.add(entry["ingredient"])
            lines.append(f"  {entry.get('label', name)}  ({entry['ingredient']})  {score:.2f}")

        makeable = []
        for c in self.recipes:
            bar, _, _ = R.recipe_requirements(c, self.rules)
            missing = [k for k, _ in bar if self.rules.satisfied_by(k, have, False) is None]
            if not missing and bar:
                makeable.append(c)

        self.info.config(
            text=f"{meta['classes']} {'재료' if meta['ingredient'] else '브랜드'} 클래스 · "
                 f"병 {len(boxes)}개 검출 · {len(detected)}개 식별 · 재료 {len(have)}종")
        self.detail.config(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", "식별된 병\n", "h")
        self.detail.insert("end", ("\n".join(lines) or "  없음") + "\n\n")
        self.detail.insert("end", f"만들 수 있는 IBA 칵테일 ({len(makeable)})\n", "h")
        for c in makeable[:25]:
            self.detail.insert("end", f"  · {c['name']}\n")
            self.detail.insert("end", f"      {' / '.join(c['ingredients'][:3])}\n")
        self.detail.tag_config("h", foreground=ACCENT, font=("Malgun Gothic", 10, "bold"))
        self.detail.config(state="disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="open this photo straight away")
    args = ap.parse_args()

    if not discover():
        raise SystemExit("no checkpoints under runs/*/weights/")
    root = tk.Tk()
    Viewer(root, args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
