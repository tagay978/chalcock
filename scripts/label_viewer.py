"""Browse a dataset with its labels drawn on the photos.

Answers the questions a metrics table cannot: did the merge put the right boxes on this photo,
is this class actually one bottle or three under different names, why does that image score badly.
Shows ground truth only - scripts/model_viewer.py is the one that runs a model.

Handles both layouts in the repo: the built datasets (train/valid/test each with images/ and
labels/) and testset/ (a single images/ and labels/ pair).

Run:  python scripts/label_viewer.py
      python scripts/label_viewer.py --dataset dataset_ing --split train --class gin
"""
from __future__ import annotations

import argparse
import colorsys
import os
import tkinter as tk
from tkinter import ttk

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BG = "#0d0f14"
PANEL = "#161a22"
TEXT = "#e8eaf0"
MUTED = "#8d95a8"
ACCENT = "#f0a04b"


def datasets():
    """Every directory here that looks like a YOLO dataset."""
    out = []
    for name in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, name)
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "data.yaml")):
            continue
        splits = [s for s in ("train", "valid", "test")
                  if os.path.isdir(os.path.join(path, s, "images"))]
        if not splits and os.path.isdir(os.path.join(path, "images")):
            splits = ["."]                      # testset keeps images/ and labels/ at the top
        if splits:
            out.append((name, path, splits))
    return out


def class_colour(index, total):
    """Spread hues so neighbouring class ids stay visually distinct."""
    h = (index * 0.6180339887) % 1.0            # golden ratio keeps consecutive ids far apart
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


class Browser:
    def __init__(self, root, dataset=None, split=None, only=None):
        self.root = root
        self.sets = datasets()
        if not self.sets:
            raise SystemExit("no dataset with a data.yaml found under the repository")
        self.only = only
        self.index = 0
        self.files = []
        self.build()
        if dataset:
            self.ds_var.set(dataset)
        self.reload_splits()
        if split:
            self.split_var.set(split)
        self.reload_files()

    # ---------- layout ----------
    def build(self):
        self.root.title("Label viewer")
        self.root.configure(bg=BG)
        self.root.geometry("1200x800")

        bar = tk.Frame(self.root, bg=PANEL, padx=12, pady=10)
        bar.pack(fill="x")

        tk.Label(bar, text="데이터셋", bg=PANEL, fg=MUTED).pack(side="left")
        self.ds_var = tk.StringVar(value=self.sets[0][0])
        self.ds_box = ttk.Combobox(bar, textvariable=self.ds_var, width=16, state="readonly",
                                   values=[n for n, _, _ in self.sets])
        self.ds_box.pack(side="left", padx=(6, 14))
        self.ds_box.bind("<<ComboboxSelected>>", lambda _e: (self.reload_splits(),
                                                            self.reload_files()))

        tk.Label(bar, text="split", bg=PANEL, fg=MUTED).pack(side="left")
        self.split_var = tk.StringVar()
        self.split_box = ttk.Combobox(bar, textvariable=self.split_var, width=8, state="readonly")
        self.split_box.pack(side="left", padx=(6, 14))
        self.split_box.bind("<<ComboboxSelected>>", lambda _e: self.reload_files())

        tk.Label(bar, text="클래스", bg=PANEL, fg=MUTED).pack(side="left")
        self.class_var = tk.StringVar(value="(전체)")
        self.class_box = ttk.Combobox(bar, textvariable=self.class_var, width=24, state="readonly")
        self.class_box.pack(side="left", padx=(6, 14))
        self.class_box.bind("<<ComboboxSelected>>", lambda _e: self.reload_files())

        tk.Button(bar, text="◀ 이전", command=lambda: self.step(-1),
                  bg=PANEL, fg=TEXT).pack(side="left")
        tk.Button(bar, text="다음 ▶", command=lambda: self.step(1),
                  bg=PANEL, fg=TEXT).pack(side="left", padx=(6, 14))

        self.counter = tk.Label(bar, text="", bg=PANEL, fg=ACCENT)
        self.counter.pack(side="left")

        self.canvas = tk.Label(self.root, bg=BG)
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        side = tk.Frame(self.root, bg=PANEL, width=330)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self.title = tk.Label(side, text="", bg=PANEL, fg=ACCENT, anchor="w", justify="left",
                              wraplength=300, font=("Malgun Gothic", 10, "bold"))
        self.title.pack(fill="x", padx=12, pady=(14, 6))
        self.detail = tk.Text(side, bg=PANEL, fg=TEXT, relief="flat", wrap="word",
                              font=("Consolas", 10), padx=10)
        self.detail.pack(fill="both", expand=True, pady=(0, 12))

        for key, delta in (("<Left>", -1), ("<Right>", 1), ("<Prior>", -10), ("<Next>", 10)):
            self.root.bind(key, lambda _e, d=delta: self.step(d))
        self.root.bind("<Configure>", lambda _e: self.render())

    # ---------- data ----------
    def current_set(self):
        return next(s for s in self.sets if s[0] == self.ds_var.get())

    def reload_splits(self):
        _, _, splits = self.current_set()
        self.split_box["values"] = splits
        if self.split_var.get() not in splits:
            self.split_var.set(splits[0])

    def dirs(self):
        _, path, _ = self.current_set()
        split = self.split_var.get()
        base = path if split == "." else os.path.join(path, split)
        return os.path.join(base, "images"), os.path.join(base, "labels")

    def names(self):
        _, path, _ = self.current_set()
        return yaml.safe_load(open(os.path.join(path, "data.yaml"), encoding="utf-8"))["names"]

    def boxes(self, stem):
        _, ldir = self.dirs()
        path = os.path.join(ldir, stem + ".txt")
        out = []
        if not os.path.exists(path):
            return out
        for line in open(path, encoding="utf-8"):
            p = line.split()
            if len(p) != 5:
                continue
            try:
                out.append((int(p[0]), *(float(v) for v in p[1:5])))
            except ValueError:
                continue
        return out

    def reload_files(self):
        idir, _ = self.dirs()
        names = self.names()
        everything = sorted(f for f in os.listdir(idir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))

        present = sorted({names[c] for f in everything
                          for c, *_ in self.boxes(os.path.splitext(f)[0])
                          if 0 <= c < len(names)})
        self.class_box["values"] = ["(전체)"] + present
        wanted = self.only or self.class_var.get()
        self.only = None
        if wanted and wanted != "(전체)" and wanted in present:
            self.class_var.set(wanted)
            self.files = [f for f in everything
                          if any(0 <= c < len(names) and names[c] == wanted
                                 for c, *_ in self.boxes(os.path.splitext(f)[0]))]
        else:
            self.class_var.set("(전체)")
            self.files = everything
        self.index = 0
        self.render()

    def step(self, delta):
        if not self.files:
            return
        self.index = (self.index + delta) % len(self.files)
        self.render()

    # ---------- drawing ----------
    def render(self):
        if not self.files:
            self.counter.config(text="0 / 0")
            self.canvas.config(image="")
            self.title.config(text="이 조건에 맞는 사진이 없습니다")
            return
        fn = self.files[self.index]
        stem = os.path.splitext(fn)[0]
        idir, _ = self.dirs()
        names = self.names()
        boxes = self.boxes(stem)

        image = Image.open(os.path.join(idir, fn)).convert("RGB")
        W, H = image.size
        pen = ImageDraw.Draw(image)
        try:
            fnt = ImageFont.truetype("malgun.ttf", max(15, W // 48))
        except OSError:
            fnt = ImageFont.load_default()

        listing = []
        for cid, x, y, w, h in boxes:
            name = names[cid] if 0 <= cid < len(names) else f"?{cid}"
            colour = class_colour(cid, len(names))
            box = ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
            pen.rectangle(box, outline=colour, width=max(3, W // 300))
            tb = pen.textbbox((0, 0), name, font=fnt)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, box[1] - th - 8)
            pen.rectangle([box[0], ty, box[0] + tw + 10, ty + th + 8], fill=colour)
            pen.text((box[0] + 5, ty + 3), name, fill=(20, 20, 20), font=fnt)
            listing.append(f"{name:<26} {w * 100:5.1f}% x {h * 100:5.1f}%")

        avail_w = max(400, self.canvas.winfo_width() or 820)
        avail_h = max(400, self.canvas.winfo_height() or 720)
        shown = image.copy()
        shown.thumbnail((avail_w, avail_h))
        self.photo = ImageTk.PhotoImage(shown)
        self.canvas.config(image=self.photo)

        self.counter.config(text=f"{self.index + 1} / {len(self.files)}")
        self.title.config(text=f"{fn}\n{W}x{H} · 박스 {len(boxes)}개")
        self.detail.config(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", "클래스            박스 크기 (가로 x 세로)\n", "h")
        self.detail.insert("end", "\n".join(listing) or "  라벨 없음")
        self.detail.insert("end", "\n\n← → 로 이동, PgUp/PgDn 은 10장씩\n", "h")
        self.detail.tag_config("h", foreground=MUTED)
        self.detail.config(state="disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="directory name, e.g. dataset_ing")
    ap.add_argument("--split", help="train, valid, test")
    ap.add_argument("--class", dest="only", help="show only photos containing this class")
    args = ap.parse_args()

    root = tk.Tk()
    Browser(root, args.dataset, args.split, args.only)
    root.mainloop()


if __name__ == "__main__":
    main()
