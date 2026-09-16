"""Review one class at a time: every box that carries it, cropped, on a grid.

The photo-by-photo viewer is the wrong shape for finding a systematic mistake. If a whole shelf
was labelled `chartreusegreen`, paging through photos shows one image at a time; laying every
`chartreusegreen` crop side by side makes the wrong ones obvious in a glance.

    click a crop        select it (click again to deselect)
    drag / shift-click  select a run
    A / Escape          select all on the page / clear
    pick a class, 적용   retype every selected crop
    Delete              remove every selected box
    Ctrl+S              write the label files and the label_fixes/ overlay

Edits are held in memory until saved, and saving writes both the live label file and the
durable overlay, exactly as the photo viewer does.

Run:  python scripts/class_review.py
      python scripts/class_review.py --dataset dataset_v2 --class chartreusegreen
"""
from __future__ import annotations

import argparse
import os
import tkinter as tk
from tkinter import messagebox, ttk

import yaml
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXES = os.path.join(ROOT, "label_fixes")
COLS, ROWS = 8, 4
TILE = 150
PAD = 0.06          # a little air around the crop so the bottle shape reads

BG = "#0d0f14"
PANEL = "#161a22"
TEXT = "#e8eaf0"
MUTED = "#8d95a8"
ACCENT = "#f0a04b"
PICKED = "#ffffff"

PREFERRED = ["dataset_v2", "dataset_fam", "dataset_ing", "testset"]


def datasets():
    out = []
    for name in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, name)
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "data.yaml")):
            continue
        splits = [s for s in ("train", "valid", "test")
                  if os.path.isdir(os.path.join(path, s, "images"))]
        if not splits and os.path.isdir(os.path.join(path, "images")):
            splits = ["."]
        if splits:
            out.append((name, path, splits))
    out.sort(key=lambda d: (PREFERRED.index(d[0]) if d[0] in PREFERRED else len(PREFERRED), d[0]))
    return out


class Review:
    def __init__(self, root, dataset=None, split=None, klass=None):
        self.root = root
        self.sets = datasets()
        if not self.sets:
            raise SystemExit("no dataset with a data.yaml found under the repository")
        self.want_class = klass
        self.page = 0
        self.items = []          # (image filename, box index, [cid, x, y, w, h])
        self.labels = {}         # image filename -> list of boxes, edited in place
        self.touched = set()
        self.picked = set()
        self.tiles = []
        self.build()
        self.reload_splits()
        if dataset:
            self.ds_var.set(dataset)
            self.reload_splits()
        if split:
            self.split_var.set(split)
        self.reload_classes()

    # ---------- layout ----------
    def build(self):
        self.root.title("Class review")
        self.root.configure(bg=BG)
        self.root.geometry("1320x860")

        bar = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        bar.pack(fill="x")

        tk.Label(bar, text="데이터셋", bg=PANEL, fg=MUTED).pack(side="left")
        self.ds_var = tk.StringVar(value=self.sets[0][0])
        box = ttk.Combobox(bar, textvariable=self.ds_var, width=14, state="readonly",
                           values=[n for n, _, _ in self.sets])
        box.pack(side="left", padx=(6, 12))
        box.bind("<<ComboboxSelected>>", lambda _e: (self.reload_splits(), self.reload_classes()))

        tk.Label(bar, text="split", bg=PANEL, fg=MUTED).pack(side="left")
        self.split_var = tk.StringVar()
        self.split_box = ttk.Combobox(bar, textvariable=self.split_var, width=7, state="readonly")
        self.split_box.pack(side="left", padx=(6, 12))
        self.split_box.bind("<<ComboboxSelected>>", lambda _e: self.reload_classes())

        tk.Label(bar, text="클래스", bg=PANEL, fg=MUTED).pack(side="left")
        self.class_var = tk.StringVar()
        self.class_box = ttk.Combobox(bar, textvariable=self.class_var, width=24, state="readonly")
        self.class_box.pack(side="left", padx=(6, 12))
        self.class_box.bind("<<ComboboxSelected>>", lambda _e: self.load_items())

        tk.Button(bar, text="◀", command=lambda: self.turn(-1), bg=PANEL, fg=TEXT,
                  width=3).pack(side="left")
        tk.Button(bar, text="▶", command=lambda: self.turn(1), bg=PANEL, fg=TEXT,
                  width=3).pack(side="left", padx=(4, 10))
        self.counter = tk.Label(bar, text="", bg=PANEL, fg=ACCENT, width=26, anchor="w")
        self.counter.pack(side="left")

        self.save_btn = tk.Button(bar, text="저장 (Ctrl+S)", command=self.save, bg=PANEL, fg=TEXT)
        self.save_btn.pack(side="right")

        act = tk.Frame(self.root, bg=PANEL, padx=12, pady=6)
        act.pack(fill="x")
        self.picked_label = tk.Label(act, text="선택 0개", bg=PANEL, fg=ACCENT, width=12,
                                     anchor="w")
        self.picked_label.pack(side="left")
        tk.Label(act, text="→", bg=PANEL, fg=MUTED).pack(side="left", padx=6)
        self.to_class = tk.StringVar()
        self.to_box = ttk.Combobox(act, textvariable=self.to_class, width=24)
        self.to_box.pack(side="left", padx=(0, 8))
        self.to_box.bind("<KeyRelease>", self.filter_classes)
        tk.Button(act, text="적용", command=self.retype, bg=ACCENT, fg="#1a1205").pack(side="left")
        tk.Button(act, text="선택 삭제 (Del)", command=self.delete, bg=PANEL,
                  fg=TEXT).pack(side="left", padx=10)
        tk.Label(act, text="클릭 선택 · 드래그/Shift 연속 선택 · A 전체 · Esc 해제",
                 bg=PANEL, fg=MUTED).pack(side="left", padx=12)

        self.grid = tk.Frame(self.root, bg=BG)
        self.grid.pack(fill="both", expand=True, padx=8, pady=8)

        self.root.bind("<Control-s>", lambda _e: self.save())
        self.root.bind("<Delete>", lambda _e: self.guarded(self.delete))
        self.root.bind("<BackSpace>", lambda _e: self.guarded(self.delete))
        self.root.bind("<a>", lambda _e: self.guarded(self.select_all))
        self.root.bind("<A>", lambda _e: self.guarded(self.select_all))
        self.root.bind("<Escape>", lambda _e: self.guarded(self.clear_pick))
        self.root.bind("<Left>", lambda _e: self.guarded(lambda: self.turn(-1)))
        self.root.bind("<Right>", lambda _e: self.guarded(lambda: self.turn(1)))

    def guarded(self, fn):
        """Shortcuts must not fire while the class box is being typed into."""
        if self.root.focus_get() is self.to_box:
            return
        fn()

    def filter_classes(self, event):
        """Type to narrow the target class list; 113 names are not worth scrolling."""
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab"):
            return
        typed = self.to_class.get().lower()
        allnames = self.names()
        matches = [n for n in allnames if typed in n.lower()] if typed else allnames
        self.to_box["values"] = matches or allnames

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

    def reload_classes(self):
        if not self.confirm_discard():
            return
        idir, ldir = self.dirs()
        names = self.names()
        self.labels, self.touched = {}, set()
        counts = {}
        for fn in sorted(os.listdir(idir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            path = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
            rows = []
            if os.path.exists(path):
                for line in open(path, encoding="utf-8"):
                    p = line.split()
                    if len(p) != 5:
                        continue
                    try:
                        rows.append([int(p[0]), *(float(v) for v in p[1:5])])
                    except ValueError:
                        continue
            self.labels[fn] = rows
            for r in rows:
                if 0 <= r[0] < len(names):
                    counts[names[r[0]]] = counts.get(names[r[0]], 0) + 1

        ordered = sorted(counts, key=lambda n: -counts[n])
        self.class_box["values"] = [f"{n}  ({counts[n]})" for n in ordered]
        self.to_box["values"] = names
        self.class_names = ordered
        wanted = self.want_class
        self.want_class = None
        pick = wanted if wanted in ordered else (ordered[0] if ordered else "")
        if pick:
            self.class_var.set(f"{pick}  ({counts[pick]})")
        self.load_items()

    def selected_class(self):
        return self.class_var.get().split("  (")[0]

    def load_items(self):
        names = self.names()
        want = self.selected_class()
        self.items = []
        for fn, rows in self.labels.items():
            for i, r in enumerate(rows):
                if 0 <= r[0] < len(names) and names[r[0]] == want:
                    self.items.append((fn, i))
        self.page = 0
        self.picked.clear()
        self.render()

    def turn(self, delta):
        pages = max(1, (len(self.items) + COLS * ROWS - 1) // (COLS * ROWS))
        self.page = (self.page + delta) % pages
        self.picked.clear()
        self.render()

    # ---------- selection ----------
    def on_click(self, index, event):
        if event.state & 0x0001 and self.picked:        # shift: extend from the last pick
            last = max(self.picked)
            lo, hi = sorted((last, index))
            self.picked.update(range(lo, hi + 1))
        elif index in self.picked:
            self.picked.discard(index)
        else:
            self.picked.add(index)
        self.paint_selection()

    def select_all(self):
        start = self.page * COLS * ROWS
        self.picked = set(range(start, min(start + COLS * ROWS, len(self.items))))
        self.paint_selection()

    def clear_pick(self):
        self.picked.clear()
        self.paint_selection()

    # ---------- editing ----------
    def retype(self):
        names = self.names()
        target = self.to_class.get()
        if target not in names:
            messagebox.showinfo("클래스", f"'{target}' 는 이 데이터셋에 없는 클래스입니다")
            return
        if not self.picked:
            return
        cid = names.index(target)
        for i in sorted(self.picked):
            fn, bi = self.items[i]
            self.labels[fn][bi][0] = cid
            self.touched.add(fn)
        self.after_edit()

    def delete(self):
        if not self.picked:
            return
        # Delete from the back so the remaining indices stay valid.
        for i in sorted(self.picked, reverse=True):
            fn, bi = self.items[i]
            del self.labels[fn][bi]
            self.touched.add(fn)
            for j, (ofn, obi) in enumerate(self.items):
                if ofn == fn and obi > bi:
                    self.items[j] = (ofn, obi - 1)
        self.after_edit()

    def after_edit(self):
        self.picked.clear()
        self.load_items()
        self.save_btn.config(fg=ACCENT if self.touched else TEXT)

    def confirm_discard(self):
        if not self.touched:
            return True
        answer = messagebox.askyesnocancel("저장하지 않은 수정",
                                           f"{len(self.touched)}장의 수정을 저장할까요?")
        if answer is None:
            return False
        if answer:
            self.save()
        self.touched.clear()
        return True

    def save(self):
        if not self.touched:
            return
        _, ldir = self.dirs()
        names = self.names()
        for fn in sorted(self.touched):
            rows = self.labels[fn]
            stem = os.path.splitext(fn)[0]
            with open(os.path.join(ldir, stem + ".txt"), "w", encoding="utf-8") as fh:
                for cid, x, y, w, h in rows:
                    fh.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            os.makedirs(FIXES, exist_ok=True)
            with open(os.path.join(FIXES, fn + ".txt"), "w", encoding="utf-8") as fh:
                for cid, x, y, w, h in rows:
                    label = names[cid] if 0 <= cid < len(names) else str(cid)
                    fh.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        saved = len(self.touched)
        self.touched.clear()
        self.save_btn.config(fg=TEXT)
        self.counter.config(text=f"{saved}장 저장됨")
        self.root.after(1500, self.update_counter)

    # ---------- drawing ----------
    def update_counter(self):
        pages = max(1, (len(self.items) + COLS * ROWS - 1) // (COLS * ROWS))
        self.counter.config(text=f"{self.selected_class()} · {len(self.items)}개 · "
                                 f"{self.page + 1}/{pages} 쪽")

    def crop(self, fn, box):
        idir, _ = self.dirs()
        with Image.open(os.path.join(idir, fn)) as im:
            im = im.convert("RGB")
            W, H = im.size
            _, x, y, w, h = box
            px, py = w * PAD, h * PAD
            rect = (max(0, int((x - w / 2 - px) * W)), max(0, int((y - h / 2 - py) * H)),
                    min(W, int((x + w / 2 + px) * W)), min(H, int((y + h / 2 + py) * H)))
            if rect[2] - rect[0] < 4 or rect[3] - rect[1] < 4:
                rect = (0, 0, W, H)
            out = im.crop(rect)
        out.thumbnail((TILE - 12, TILE - 12))
        return out

    def paint_selection(self):
        for index, frame in self.tiles:
            frame.config(highlightbackground=PICKED if index in self.picked else PANEL,
                         highlightthickness=3 if index in self.picked else 1)
        self.picked_label.config(text=f"선택 {len(self.picked)}개")

    def render(self):
        for child in self.grid.winfo_children():
            child.destroy()
        self.tiles = []
        self.photos = []

        start = self.page * COLS * ROWS
        page_items = self.items[start:start + COLS * ROWS]
        for n, (fn, bi) in enumerate(page_items):
            index = start + n
            box = self.labels[fn][bi]
            frame = tk.Frame(self.grid, bg=PANEL, highlightbackground=PANEL,
                             highlightthickness=1, padx=3, pady=3)
            frame.grid(row=n // COLS, column=n % COLS, padx=4, pady=4)
            try:
                photo = ImageTk.PhotoImage(self.crop(fn, box))
            except Exception:
                continue
            self.photos.append(photo)
            lbl = tk.Label(frame, image=photo, bg=PANEL)
            lbl.pack()
            cap = tk.Label(frame, text=fn[:18], bg=PANEL, fg=MUTED, font=("Consolas", 7))
            cap.pack()
            for widget in (frame, lbl, cap):
                widget.bind("<Button-1>", lambda e, i=index: self.on_click(i, e))
            self.tiles.append((index, frame))

        self.paint_selection()
        self.update_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--split")
    ap.add_argument("--class", dest="klass", help="open on this class")
    args = ap.parse_args()

    root = tk.Tk()
    Review(root, args.dataset, args.split, args.klass)
    root.mainloop()


if __name__ == "__main__":
    main()
