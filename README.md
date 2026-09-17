# Chalcock — bottle detection for cocktail recommendation

![Chalcock demo: detecting bottles on two real shelf photos, then clicking through to a bottle's info page and a cocktail's recipe page](docs/demo.gif)

Upload a photo of a home bar shelf, get back which bottles it recognises and which of the 102
official IBA cocktails they can actually make. A YOLO11 detector finds and names the bottles;
`data/bottles.yaml` maps each brand to what it pours; `data/ingredient_rules.yaml` matches that
against `data/iba_cocktails.json`.

## Run it

```bash
pip install fastapi "uvicorn[standard]" python-multipart ultralytics
python app/main.py                 # http://127.0.0.1:8000
```

Test-set photos in `testset/images/` are offered as one-click samples. Clicking a bottle opens
what it is and what it can make; clicking a cocktail opens its recipe.

Regenerate the gif above with `python scripts/capture_demo_pages.py` (needs the app running and
`pip install playwright && playwright install chromium`, for real screenshots of the results and
detail pages) followed by `python scripts/make_demo_video.py`.

## The model

`runs/yolo11s_v10/weights/epoch80.pt` — YOLO11s, 112 brand classes, trained on `dataset_v2`
(3,400+ photos, ~13,000 boxes, merged and deduplicated from a 2023 crawl plus hand photography;
see `scripts/build_dataset_ouo.py`). The web app runs it one-stage (straight on the full photo)
at conf 0.4, chosen by comparing it against a two-stage crop-and-classify path on real shelf
photos — two-stage's forced square resize distorts a bottle's label enough to cost it more
accuracy than the extra scale-invariance it buys back.

**Honest limitation:** 112 brands is a small slice of any real bar. A bottle outside the
vocabulary gets forced into the nearest-looking trained class rather than being declined — an
`unknown_bottle` abstain class was tried and made real-world results worse, not better, because
it learned to separate training-photo source rather than bottle identity. The real-world test
set was recently replaced with fresh, unannotated shelf photos, so a rigorous accuracy number
against it doesn't exist yet; treat the detections as "best guess, shown with its confidence,"
not a verified accuracy claim.

## Layout

```
app/          FastAPI web demo (main.py) + static frontend
scripts/      training, dataset build, harvesting, eval, demo video
data/         IBA recipes + bottle/ingredient mappings
dataset_v2/   training set — generated, gitignored (see scripts/build_dataset_ouo.py)
runs/         training output — gitignored except the checkpoint above stays local
weights/      pretrained checkpoints — gitignored
testset/      hand-picked real shelf photos used for demo samples
label_fixes/  hand-corrected labels for dataset_v2 — tracked, not generated
```

Raw crawled/candidate images (2023, unresolved licensing) are never tracked — see `.gitignore`.
Anything fetched by `scripts/harvest_images.py`, `fetch_cocktail_images.py`, or
`fetch_ingredient_images.py` is licence-checked at download time instead.

## Environment

Python 3.12, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124, Ultralytics 8.4.x, CUDA GPU.

> `torchvision` must be the CUDA build matching torch — a `+cpu` torchvision alongside a
> `+cu124` torch fails at NMS during validation.
