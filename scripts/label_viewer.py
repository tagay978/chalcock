"""Browse a dataset with its labels drawn on the photos, and fix them in place.

Answers what a metrics table cannot: did the merge put the right boxes on this photo, is this
class really one bottle or three under different names. Shows ground truth only -
scripts/model_viewer.py is the one that runs a model.

Editing:
    click a box          select it
    drag on empty space  draw a new box, with the class in the "새 박스" dropdown
    Delete / BackSpace   remove the selected box
    class dropdown       retype the selected box
    Ctrl+S               write the label file
    F                    flag this photo as needing work (review.csv)

Nothing is written until you save, and leaving a photo with unsaved edits asks first.

Handles both layouts in the repo: the built datasets (train/valid/test each with images/ and
labels/) and testset/ (a single images/ and labels/ pair).

Run:  python scripts/label_viewer.py
      python scripts/label_viewer.py --dataset dataset_ing --split train --class gin
      python scripts/label_viewer.py --flagged        # only photos already flagged
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import os
import tkinter as tk
from tkinter import messagebox, ttk

import yaml
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_NEW_BOX = 8          # pixels on screen; below this a drag is a click, not a box

BG = "#0d0f14"
PANEL = "#161a22"
TEXT = "#e8eaf0"
MUTED = "#8d95a8"
ACCENT = "#f0a04b"
SELECT = "#ffffff"


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


def class_colour(index):
    """Spread hues so neighbouring class ids stay visually distinct."""
    h = (index * 0.6180339887) % 1.0            # golden ratio keeps consecutive ids far apart
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


class Browser:
    def __init__(self, root, dataset=None, split=None, only=None, flagged_only=False):
        self.root = root
        self.sets = datasets()
        if not self.sets:
            raise SystemExit("no dataset with a data.yaml found under the repository")
        self.only = only
        self.flagged_only = flagged_only
        self.index = 0
        self.files = []
        self.boxes = []          # [cid, x, y, w, h] in normalised coords, edited in place
        self.selected = None
        self.dirty = False
        self.drag = None
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
        self.root.geometry("1280x840")

        bar = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        bar.pack(fill="x")

        tk.Label(bar, text="데이터셋", bg=PANEL, fg=MUTED).pack(side="left")
        self.ds_var = tk.StringVar(value=self.sets[0][0])
        box = ttk.Combobox(bar, textvariable=self.ds_var, width=15, state="readonly",
                           values=[n for n, _, _ in self.sets])
        box.pack(side="left", padx=(6, 12))
        box.bind("<<ComboboxSelected>>", lambda _e: (self.reload_splits(), self.reload_files()))

        tk.Label(bar, text="split", bg=PANEL, fg=MUTED).pack(side="left")
        self.split_var = tk.StringVar()
        self.split_box = ttk.Combobox(bar, textvariable=self.split_var, width=7, state="readonly")
        self.split_box.pack(side="left", padx=(6, 12))
        self.split_box.bind("<<ComboboxSelected>>", lambda _e: self.reload_files())

        tk.Label(bar, text="필터", bg=PANEL, fg=MUTED).pack(side="left")
        self.class_var = tk.StringVar(value="(전체)")
        self.class_box = ttk.Combobox(bar, textvariable=self.class_var, width=20, state="readonly")
        self.class_box.pack(side="left", padx=(6, 12))
        self.class_box.bind("<<ComboboxSelected>>", lambda _e: self.reload_files())

        tk.Button(bar, text="◀", command=lambda: self.step(-1), bg=PANEL, fg=TEXT,
                  width=3).pack(side="left")
        tk.Button(bar, text="▶", command=lambda: self.step(1), bg=PANEL, fg=TEXT,
                  width=3).pack(side="left", padx=(4, 12))
        self.counter = tk.Label(bar, text="", bg=PANEL, fg=ACCENT, width=12)
        self.counter.pack(side="left")

        self.save_btn = tk.Button(bar, text="저장 (Ctrl+S)", command=self.save,
                                  bg=PANEL, fg=TEXT)
        self.save_btn.pack(side="right")
        self.flag_btn = tk.Button(bar, text="표시 (F)", command=self.toggle_flag,
                                  bg=PANEL, fg=TEXT)
        self.flag_btn.pack(side="right", padx=8)

        edit = tk.Frame(self.root, bg=PANEL, padx=12, pady=6)
        edit.pack(fill="x")
        tk.Label(edit, text="새 박스 / 선택한 박스 클래스", bg=PANEL, fg=MUTED).pack(side="left")
        self.new_class = tk.StringVar()
        self.new_class_box = ttk.Combobox(edit, textvariable=self.new_class, width=26,
                                          state="readonly")
        self.new_class_box.pack(side="left", padx=(8, 12))
        self.new_class_box.bind("<<ComboboxSelected>>", lambda _e: self.retype_selected())
        tk.Button(edit, text="선택 삭제 (Del)", command=self.delete_selected,
                  bg=PANEL, fg=TEXT).pack(side="left")
        self.hint = tk.Label(edit, text="빈 곳을 드래그하면 새 박스", bg=PANEL, fg=MUTED)
        self.hint.pack(side="left", padx=14)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg=BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        side = tk.Frame(body, bg=PANEL, width=330)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self.title = tk.Label(side, text="", bg=PANEL, fg=ACCENT, anchor="w", justify="left",
                              wraplength=300, font=("Malgun Gothic", 10, "bold"))
        self.title.pack(fill="x", padx=12, pady=(14, 6))
        self.detail = tk.Text(side, bg=PANEL, fg=TEXT, relief="flat", wrap="word",
                              font=("Consolas", 10), padx=10)
        self.detail.pack(fill="both", expand=True, pady=(0, 12))

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        for key, delta in (("<Left>", -1), ("<Right>", 1), ("<Prior>", -10), ("<Next>", 10)):
            self.root.bind(key, lambda _e, d=delta: self.step(d))
        self.root.bind("<Delete>", lambda _e: self.delete_selected())
        self.root.bind("<BackSpace>", lambda _e: self.delete_selected())
        self.root.bind("<Control-s>", lambda _e: self.save())
        self.root.bind("<f>", lambda _e: self.toggle_flag())
        self.root.bind("<F>", lambda _e: self.toggle_flag())
        self.root.bind("<Configure>", lambda _e: self.render())

    # ---------- paths ----------
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

    def review_path(self):
        _, path, _ = self.current_set()
        return os.path.join(path, "review.csv")

    # ---------- label io ----------
    def read_boxes(self, stem):
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
                out.append([int(p[0]), *(float(v) for v in p[1:5])])
            except ValueError:
                continue
        return out

    def save(self):
        if not self.files:
            return
        _, ldir = self.dirs()
        stem = os.path.splitext(self.files[self.index])[0]
        with open(os.path.join(ldir, stem + ".txt"), "w", encoding="utf-8") as fh:
            for cid, x, y, w, h in self.boxes:
                fh.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        self.dirty = False
        self.render()

    # ---------- flags ----------
    def load_flags(self):
        path = self.review_path()
        flags = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    flags[row["file"]] = row.get("note", "")
        return flags

    def write_flags(self, flags):
        with open(self.review_path(), "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "split", "note"])
            writer.writeheader()
            for fn, note in sorted(flags.items()):
                writer.writerow({"file": fn, "split": self.split_var.get(), "note": note})

    def toggle_flag(self):
        if not self.files:
            return
        fn = self.files[self.index]
        flags = self.load_flags()
        if fn in flags:
            del flags[fn]
        else:
            flags[fn] = "needs labelling"
        self.write_flags(flags)
        self.render()

    # ---------- navigation ----------
    def may_leave(self):
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("저장하지 않은 수정",
                                           "이 사진의 수정을 저장할까요?")
        if answer is None:
            return False
        if answer:
            self.save()
        self.dirty = False
        return True

    def step(self, delta):
        if not self.files or not self.may_leave():
            return
        self.index = (self.index + delta) % len(self.files)
        self.load_current()

    def reload_files(self):
        if not self.may_leave():
            return
        idir, _ = self.dirs()
        names = self.names()
        everything = sorted(f for f in os.listdir(idir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))

        present = sorted({names[c] for f in everything
                          for c, *_ in self.read_boxes(os.path.splitext(f)[0])
                          if 0 <= c < len(names)})
        self.class_box["values"] = ["(전체)"] + present
        self.new_class_box["values"] = names
        if self.new_class.get() not in names:
            self.new_class.set(names[0])

        wanted = self.only or self.class_var.get()
        self.only = None
        if self.flagged_only:
            flagged = set(self.load_flags())
            self.flagged_only = False
            self.files = [f for f in everything if f in flagged]
        elif wanted and wanted != "(전체)" and wanted in present:
            self.class_var.set(wanted)
            self.files = [f for f in everything
                          if any(0 <= c < len(names) and names[c] == wanted
                                 for c, *_ in self.read_boxes(os.path.splitext(f)[0]))]
        else:
            self.class_var.set("(전체)")
            self.files = everything
        self.index = 0
        self.load_current()

    def load_current(self):
        self.selected = None
        self.dirty = False
        if self.files:
            self.boxes = self.read_boxes(os.path.splitext(self.files[self.index])[0])
        else:
            self.boxes = []
        self.render()

    # ---------- geometry ----------
    def layout(self):
        """Scale and offset that map image coords onto the canvas."""
        cw = max(200, self.canvas.winfo_width())
        ch = max(200, self.canvas.winfo_height())
        scale = min(cw / self.img_w, ch / self.img_h)
        w, h = self.img_w * scale, self.img_h * scale
        return scale, (cw - w) / 2, (ch - h) / 2

    def to_canvas(self, box):
        scale, ox, oy = self.layout()
        _, x, y, w, h = box
        return ((x - w / 2) * self.img_w * scale + ox, (y - h / 2) * self.img_h * scale + oy,
                (x + w / 2) * self.img_w * scale + ox, (y + h / 2) * self.img_h * scale + oy)

    def to_image(self, x0, y0, x1, y1):
        scale, ox, oy = self.layout()
        x0, x1 = sorted((max(0, (x0 - ox) / scale), max(0, (x1 - ox) / scale)))
        y0, y1 = sorted((max(0, (y0 - oy) / scale), max(0, (y1 - oy) / scale)))
        x1, y1 = min(x1, self.img_w), min(y1, self.img_h)
        return ((x0 + x1) / 2 / self.img_w, (y0 + y1) / 2 / self.img_h,
                (x1 - x0) / self.img_w, (y1 - y0) / self.img_h)

    # ---------- mouse ----------
    def on_press(self, event):
        hit = None
        for i, box in enumerate(self.boxes):
            x0, y0, x1, y1 = self.to_canvas(box)
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                area = (x1 - x0) * (y1 - y0)
                if hit is None or area < hit[1]:     # the smallest box under the cursor wins
                    hit = (i, area)
        if hit is not None:
            self.selected = hit[0]
            names = self.names()
            cid = self.boxes[self.selected][0]
            if 0 <= cid < len(names):
                self.new_class.set(names[cid])
            self.drag = None
        else:
            self.selected = None
            self.drag = (event.x, event.y, event.x, event.y)
        self.render()

    def on_drag(self, event):
        if self.drag:
            self.drag = (self.drag[0], self.drag[1], event.x, event.y)
            self.render()

    def on_release(self, event):
        if not self.drag:
            return
        x0, y0, x1, y1 = self.drag
        self.drag = None
        if abs(x1 - x0) < MIN_NEW_BOX or abs(y1 - y0) < MIN_NEW_BOX:
            self.render()
            return
        names = self.names()
        cid = names.index(self.new_class.get()) if self.new_class.get() in names else 0
        self.boxes.append([cid, *self.to_image(x0, y0, x1, y1)])
        self.selected = len(self.boxes) - 1
        self.dirty = True
        self.render()

    # ---------- editing ----------
    def delete_selected(self):
        if self.selected is None or self.selected >= len(self.boxes):
            return
        del self.boxes[self.selected]
        self.selected = None
        self.dirty = True
        self.render()

    def retype_selected(self):
        if self.selected is None:
            return
        names = self.names()
        if self.new_class.get() in names:
            self.boxes[self.selected][0] = names.index(self.new_class.get())
            self.dirty = True
            self.render()

    # ---------- drawing ----------
    def render(self):
        self.canvas.delete("all")
        if not self.files:
            self.counter.config(text="0 / 0")
            self.title.config(text="이 조건에 맞는 사진이 없습니다")
            return

        fn = self.files[self.index]
        idir, _ = self.dirs()
        names = self.names()
        image = Image.open(os.path.join(idir, fn)).convert("RGB")
        self.img_w, self.img_h = image.size
        scale, ox, oy = self.layout()
        shown = image.resize((max(1, int(self.img_w * scale)), max(1, int(self.img_h * scale))))
        self.photo = ImageTk.PhotoImage(shown)
        self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")

        listing = []
        for i, box in enumerate(self.boxes):
            cid = box[0]
            name = names[cid] if 0 <= cid < len(names) else f"?{cid}"
            colour = SELECT if i == self.selected else class_colour(cid)
            width = 4 if i == self.selected else 2
            x0, y0, x1, y1 = self.to_canvas(box)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=colour, width=width)
            self.canvas.create_rectangle(x0, max(0, y0 - 18), x0 + 9 * len(name) + 8, y0,
                                         fill=colour, outline=colour)
            self.canvas.create_text(x0 + 4, max(0, y0 - 17), text=name, anchor="nw",
                                    fill="#14171d", font=("Consolas", 9, "bold"))
            mark = "►" if i == self.selected else " "
            listing.append(f"{mark} {name:<24} {box[3] * 100:5.1f}% x {box[4] * 100:5.1f}%")

        if self.drag:
            self.canvas.create_rectangle(*self.drag, outline=ACCENT, width=2, dash=(4, 3))

        flagged = fn in self.load_flags()
        self.counter.config(text=f"{self.index + 1} / {len(self.files)}")
        self.flag_btn.config(text="표시 해제 (F)" if flagged else "표시 (F)",
                             fg=ACCENT if flagged else TEXT)
        self.save_btn.config(fg=ACCENT if self.dirty else TEXT)
        self.title.config(
            text=f"{fn}\n{self.img_w}x{self.img_h} · 박스 {len(self.boxes)}개"
                 + ("  · 수정됨" if self.dirty else "")
                 + ("  · 표시됨" if flagged else ""))
        self.detail.config(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", "클래스                   박스 크기\n", "h")
        self.detail.insert("end", "\n".join(listing) or "  라벨 없음")
        self.detail.insert("end",
                           "\n\n← →  이동 (PgUp/PgDn 10장)\n"
                           "클릭  박스 선택\n"
                           "드래그 새 박스\n"
                           "Del   선택 삭제\n"
                           "Ctrl+S 저장\n"
                           "F     검수 표시\n", "h")
        self.detail.tag_config("h", foreground=MUTED)
        self.detail.config(state="disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="directory name, e.g. dataset_ing")
    ap.add_argument("--split", help="train, valid, test")
    ap.add_argument("--class", dest="only", help="show only photos containing this class")
    ap.add_argument("--flagged", action="store_true", help="show only photos flagged for review")
    args = ap.parse_args()

    root = tk.Tk()
    Browser(root, args.dataset, args.split, args.only, args.flagged)
    root.mainloop()


if __name__ == "__main__":
    main()
