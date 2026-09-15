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

## Results, and why the headline number is misleading

YOLO11s, 640px, early-stopped at epoch 149 (best at 99):

| split | mAP50 | mAP50-95 |
| --- | --- | --- |
| valid | 0.991 | 0.986 |
| test | 0.952 | 0.940 |

Both splits come from the same 2023 photo sessions as the training images, so this measures
memorisation of those sessions, not detection of a bottle in the wild. Weakest test classes are
`extradry` (0.533), `gordons` (0.720) and the near-identical pairs the label set contains —
`johnbarrfinest`/`johnbarrreserve` (0.724/0.746), `camusvsop` (0.745).

Run against 34 real bar photos harvested from Commons and Openverse, the same weights produce
**2 detections at conf 0.5, and 4 at conf 0.25** — across photos containing several hundred
bottles. The cause is scale:

| | training set | real shelf photos |
| --- | --- | --- |
| median box area (% of image) | 20.3% | 1.24% |
| 75th percentile | 26.8% | 2.69% |

Three quarters of real-world bottles are smaller than 95% of the training bottles. The model was
trained on close-ups and never sees a bottle at shelf scale.

`recommend.py --two-stage` works around this by detecting bottles with a COCO-pretrained model
and running the brand model on each crop, which puts the subject back at the trained scale. That
raises attempts roughly fourfold, **but the identifications are mostly wrong**: on a photo of six
Johnnie Walker bottles it returned two "Camus XO" (a cognac), one "Johnnie Walker Red" for a Gold
Label, and one correct Black Label.

The deeper problem is that there is no way to answer "none of these". Every crop is forced into
one of 50 classes, and a real bar is full of bottles outside that vocabulary — the photo above
contains Gold Label, Green Label and Explorers' Club, none of which are classes. Before the
accuracy numbers mean anything the model needs either a background/unknown class or a rejection
threshold calibrated on out-of-vocabulary bottles, and a hand-annotated real-world test set to
measure against.

## Adding more images

The training set is ~20 photos per class, all from the original 2023 collection, and its
validation split comes from the same photo sessions as its training split — which is why
validation mAP is near-saturated and should not be read as real-world accuracy. New images need
to come from a different distribution to mean anything.

`scripts/harvest_images.py` collects candidates from sources that license reuse — Openverse
(CC-licensed aggregator) and Wikimedia Commons — and records the attribution each licence
requires in `harvest/attribution.csv`, which is tracked in git even though the images are not.
Images under NoDerivatives are rejected outright (annotating is a derivative); NonCommercial is
rejected unless `--allow-nc`.

```bash
python scripts/harvest_images.py --queries shelf --limit 50   # multi-bottle shelf scenes
python scripts/filter_harvest.py --min-bottles 2              # drop text-match false positives
python scripts/pseudo_label.py --weights runs/yolo11s/weights/best.pt
```

Text search matches metadata, not pixels, so roughly a third of results are magazine covers and
landscapes; `filter_harvest.py` runs a COCO-pretrained detector and keeps only images that
actually contain bottles. `pseudo_label.py` then writes draft boxes so annotation is correction
rather than drawing from scratch.

Two things this does **not** solve. Harvested photos are other people's bars, so most bottles in
them are not among the 50 classes — the model will still put a label on them, and those guesses
must be reviewed and deleted. And for brand coverage specifically, photographing the actual
bottles is both cleaner and higher quality than anything a search returns.

## The real-world test set

Nothing above can be improved while the only yardstick is a split that shares photo sessions with
training. `scripts/prepare_testset.py` builds an annotation-ready set from the harvested photos:

```bash
python scripts/prepare_testset.py --limit 60
```

Boxes are proposed by the COCO detector, which is good at finding bottles even when the brand
model is not, and every proposal is written as class **`unknown_bottle`** (id 50, appended after
the 50 trained classes in their training order). Annotation is therefore relabelling rather than
drawing — the slow part is already done.

Three rules make the set measure what it needs to:

1. Change the class on bottles you recognise as one of the 50.
2. **Leave everything else as `unknown_bottle`.** This is an annotation, not a skip: it is the
   only way to measure how often the model puts a brand name on a bottle it has never seen.
3. Delete boxes that are not bottles; add bottles the proposer missed.

Annotate with:

```bash
pip install labelImg
python scripts/annotate.py
```

`annotate.py` opens labelImg with both directories already selected, and works around two
failures that are silent in a conda install:

- Qt cannot load `qjpeg.dll` unless `<prefix>/Library/bin` is on PATH, because that is where its
  libjpeg lives. Nothing reports an error — JPEG just vanishes from the supported formats, so
  `Open Dir` scans the folder, matches none of the 56 `.jpg` files, and shows an empty list.
- labelImg 1.8.6 passes float coordinates to `QPainter.drawRect`/`drawLine`, which modern PyQt5
  rejects, so the canvas raises as soon as you drag a box. Cast them to `int` in
  `libs/canvas.py`; `annotate.py` checks and tells you if the copy is unpatched.

**Set the format button in the left toolbar to YOLO before saving.** labelImg loads `.txt`
regardless of format but saves PascalVOC XML by default, which would leave the work out of the
`.txt` files entirely. Roboflow imports the folder as-is if you would rather annotate in a
browser.

Then:

```bash
python scripts/eval_realworld.py
```

which reports, for both the direct and `--two-stage` paths, how many known bottles were found,
how many were named correctly, and — the number that matters — what fraction of `unknown_bottle`
annotations drew a confident known-class prediction anyway.

### Data provenance

`finalloopy/` carries a CC BY 4.0 marking from its Roboflow export, but the underlying images
were collected in 2023 from Instagram and Bing image search. A licence applied at export does
not grant rights the collector did not have, so that marking should not be relied on for
redistribution or publication. Anything harvested by `harvest_images.py` is licence-checked at
download and attributed in `harvest/attribution.csv`; the original set is not. Worth resolving
before the dataset ships with a paper.

## Layout

```
finalloopy/   50 per-class Roboflow exports (source of truth, tracked)
data/         IBA recipes + bottle and ingredient mappings
scripts/      build_dataset.py, train.py, fetch_iba.py, recommend.py,
              harvest_images.py, filter_harvest.py, pseudo_label.py
dataset/      merged YOLO dataset — generated, gitignored
harvest/      harvested candidates — images gitignored, attribution.csv tracked
runs/         training output — gitignored
weights/      pretrained checkpoints — gitignored
```

## Environment

Python 3.12, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124, Ultralytics 8.4.x, CUDA GPU.

> `torchvision` must be the CUDA build matching torch. A `+cpu` torchvision alongside a `+cu124`
> torch fails at NMS during validation.
