# Bottle detection for cocktail recommendation

Detects liquor bottles in a photo of a home bar shelf, so the set of detected bottles can be
matched against a cocktail database to recommend drinks that are actually makeable.

This repository holds the detector: the dataset merge and the YOLO training.

## The dataset problem this fixes

`finalloopy/` contains **50 separate Roboflow exports, one per bottle**. Each was annotated on
its own, which means a single shelf photo showing three bottles was uploaded to three different
projects, and in each copy only *that* project's bottle carries a box:

| copy | file lives in | boxes it has |
| --- | --- | --- |
| A | `gordons/train/` | `gordons` only |
| B | `wyborowa/train/` | `wyborowa` only |
| C | `plantationoriginaldark/train/` | `plantationoriginaldark` only |

Training on that as-is is actively harmful — every copy teaches the model that the other two
bottles are background. `scripts/build_dataset.py` merges the exports into one 50-class dataset:

1. **Class ids are remapped from the folder name**, not from `data.yaml`. Several exports name
   their class with a bare number (`'56'`, `'61'`), so the yaml names are unusable. The one
   genuine multi-class export (`camusxo`, `nc: 2`) is resolved through its yaml names.
2. **The same photo is recognised across folders** by MD5 for byte-identical copies, plus a
   perceptual hash (distance ≤ 2, same resolution) for copies that Roboflow re-encoded — those
   have different bytes *and* different filenames, so neither alone is enough.
3. **Boxes are unioned** per photo, dropping duplicates of the same class above IoU 0.6.
4. **One split per photo.** When copies disagreed (one export put it in `train`, another in
   `valid`) the split is decided by majority, tie broken toward `train`. Without this the merged
   set would leak training images into validation.

Result: **1547 source files → 1494 unique photos, 1593 boxes, 50 classes**, of which 41 photos
carry boxes from more than one original folder.

```
train 1038 · valid 314 · test 142
```

## Recipes

`data/iba_cocktails.json` holds all **102 official IBA cocktails** (34 Unforgettables, 34
Contemporary Classics, 34 New Era Drinks) with exact measurements, method and garnish, scraped
from iba-world.com by `scripts/fetch_iba.py` rather than written from memory.

Getting from a detected bottle to a recipe takes two mappings:

- `data/bottles.yaml` — each of the 50 detector classes to what it pours (`gordons` → `gin`).
- `data/ingredient_rules.yaml` — the 225 different ways the IBA writes an ingredient
  ("Fresh Lime Juice", "Freshly Squeezed Lime juice", "Fresh lime") down to one canonical key,
  split into `bar` items that must be on the shelf and `pantry` items assumed present.
  `equivalents` are freely interchangeable (Cointreau for triple sec); `substitutes` are only
  accepted under `--loose` (aged rum where the IBA asks for white).

## Usage

```bash
# 1. merge the 50 exports into dataset/
python scripts/build_dataset.py --report   # inspect first, writes nothing
python scripts/build_dataset.py

# 2. train
python scripts/train.py                              # yolo11s, 200 epochs
python scripts/train.py --model yolo11m.pt --epochs 300

# 3. recommend
python scripts/recommend.py --image shelf.jpg --weights runs/yolo11s/weights/best.pt
python scripts/recommend.py --bottles gordons,extradry,maraschino,orangebitters
python scripts/recommend.py --bottles ... --loose --missing 1
python scripts/recommend.py --check        # audit the rules against the recipe data
```

`dataset/manifest.csv` records, for every merged photo, which source folders it came from and
which classes it ended up with.

### What the 50 bottles actually reach

All 50 classes together supply 28 distinct ingredients, which covers **23 of the 77** bar
ingredients the IBA list calls for — enough for **29 cocktails** exactly as specified, or 49
with `--loose`. The single biggest gap is that the shelf has aged, dark, spiced and coconut rum
but **no white rum**, which alone blocks the Daiquiri, Mojito, Cuba Libre and Piña Colada.
Campari (Negroni, Americano, Boulevardier, Cardinale) and Angostura bitters (Manhattan, Old
Fashioned) are the next two.

## Layout

```
finalloopy/   50 per-class Roboflow exports (source of truth, tracked)
data/         IBA recipes + bottle and ingredient mappings
scripts/      build_dataset.py, train.py, fetch_iba.py, recommend.py
dataset/      merged YOLO dataset — generated, gitignored
runs/         training output — gitignored
weights/      pretrained checkpoints — gitignored
```

## Environment

Python 3.12, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124, Ultralytics 8.4.x, CUDA GPU.

> `torchvision` must be the CUDA build matching torch. A `+cpu` torchvision alongside a `+cu124`
> torch fails at NMS during validation.
