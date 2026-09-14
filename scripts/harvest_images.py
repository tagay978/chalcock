"""Harvest additional bottle/shelf photos from sources that license reuse.

Only Openverse (CC-licensed aggregator) and Wikimedia Commons are queried, and every image is
written with the attribution its licence requires into harvest/attribution.csv. Nothing is kept
unless it carries a licence from --licenses, so the result can be redistributed with the paper.

Images arrive UNLABELLED. Run scripts/pseudo_label.py afterwards to pre-annotate them with the
trained detector, then correct the boxes before merging into the training set.

Run:  python scripts/harvest_images.py --queries shelf --limit 200
      python scripts/harvest_images.py --queries bottles --limit 5
      python scripts/harvest_images.py --list-queries
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import yaml
from PIL import Image
import imagehash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "harvest")
IMG_DIR = os.path.join(OUT, "images")
MANIFEST = os.path.join(OUT, "attribution.csv")
UA = {"User-Agent": "cocktail-bottle-detector/1.0 (research dataset; contact via repository)"}

# Scene queries bring multi-bottle shelves, which is what the app actually sees at inference.
SHELF_QUERIES = [
    "liquor bottle shelf", "home bar shelf", "liquor cabinet", "bar bottles",
    "whisky bottle collection", "spirits shelf", "back bar bottles", "liquor store shelf",
    "gin bottles shelf", "rum bottles shelf", "bottle display bar",
]


def get(url: str, tries: int = 3, binary: bool = False, quiet: bool = False):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            return raw if binary else raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404, 410):   # not transient; a retry will not help
                if not quiet:
                    print(f"    failed: HTTP {exc.code}", file=sys.stderr)
                return None
            if attempt == tries - 1:
                if not quiet:
                    print(f"    failed: HTTP {exc.code}", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))
        except Exception as exc:
            if attempt == tries - 1:
                if not quiet:
                    print(f"    failed: {exc.__class__.__name__}", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))
    return None


def openverse(query: str, limit: int, licenses: str):
    params = urllib.parse.urlencode({
        "q": query, "page_size": min(limit, 100), "license": licenses,
        "mature": "false", "size": "medium,large",
    })
    body = get(f"https://api.openverse.org/v1/images/?{params}")
    if not body:
        return []
    try:
        results = json.loads(body).get("results", [])
    except json.JSONDecodeError:
        return []
    out = []
    for r in results:
        if not r.get("url"):
            continue
        out.append({
            "source": "openverse/" + (r.get("provider") or "?"),
            "title": (r.get("title") or "").strip(),
            "creator": (r.get("creator") or "").strip(),
            "creator_url": r.get("creator_url") or "",
            "license": f"CC {(r.get('license') or '').upper()} {r.get('license_version') or ''}".strip(),
            "license_url": r.get("license_url") or "",
            "landing_url": r.get("foreign_landing_url") or "",
            "url": r["url"],
            # Flickr answers 403 to direct hotlinks, so fall back to Openverse's own
            # thumbnail endpoint, which exists for exactly this purpose.
            "fallback_url": r.get("thumbnail") or "",
        })
    return out


def commons(query: str, limit: int):
    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": query, "gsrnamespace": "6", "gsrlimit": min(limit, 50),
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "1280",
    })
    body = get(f"https://commons.wikimedia.org/w/api.php?{params}")
    if not body:
        return []
    try:
        pages = (json.loads(body).get("query") or {}).get("pages", {})
    except json.JSONDecodeError:
        return []
    out = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}

        def field(name):
            return re.sub(r"<[^>]+>", "", (meta.get(name) or {}).get("value", "")).strip()

        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        out.append({
            "source": "wikimedia-commons",
            "title": page.get("title", ""),
            "creator": field("Artist"),
            "creator_url": "",
            "license": field("LicenseShortName"),
            "license_url": field("LicenseUrl"),
            "landing_url": info.get("descriptionurl", ""),
            "url": url,
        })
    return out


ALLOWED = re.compile(r"cc[ -]?(by|by[ -]sa|0)\b|public domain|pdm|cc0", re.I)


def acceptable(licence: str, allow_nc: bool) -> bool:
    if not licence:
        return False
    if not allow_nc and re.search(r"\bnc\b|noncommercial", licence, re.I):
        return False
    if re.search(r"\bnd\b|noderiv", licence, re.I):   # no derivatives: cannot crop or annotate
        return False
    return bool(ALLOWED.search(licence))


def existing_hashes():
    """Fingerprint what we already have so harvesting cannot re-add the current dataset."""
    md5s, phs = set(), []
    for base in (os.path.join(ROOT, "dataset"), IMG_DIR):
        for dirpath, _, files in os.walk(base):
            for fn in files:
                if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    md5s.add(hashlib.md5(open(path, "rb").read()).hexdigest())
                    with Image.open(path) as im:
                        phs.append(imagehash.phash(im.convert("RGB")))
                except Exception:
                    continue
    return md5s, phs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="shelf",
                    help="'shelf', 'bottles' (one query per detector class), or a comma-separated list")
    ap.add_argument("--limit", type=int, default=50, help="results requested per query per source")
    ap.add_argument("--sources", default="openverse,commons")
    ap.add_argument("--licenses", default="by,by-sa,cc0,pdm", help="Openverse licence filter")
    ap.add_argument("--allow-nc", action="store_true", help="also keep NonCommercial images")
    ap.add_argument("--min-side", type=int, default=400, help="reject images smaller than this")
    ap.add_argument("--list-queries", action="store_true")
    args = ap.parse_args()

    bottles = yaml.safe_load(open(os.path.join(ROOT, "data", "bottles.yaml"), encoding="utf-8"))["bottles"]
    if args.queries == "shelf":
        queries = SHELF_QUERIES
    elif args.queries == "bottles":
        queries = [f"{b['label']} bottle" for b in bottles.values()]
    else:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    if args.list_queries:
        print("\n".join(queries))
        return

    os.makedirs(IMG_DIR, exist_ok=True)
    print("fingerprinting what we already have")
    md5s, phs = existing_hashes()
    print(f"  {len(md5s)} images already on disk\n")

    rows, kept, seen_urls = [], 0, set()
    skipped = {"licence": 0, "duplicate": 0, "small": 0, "error": 0}

    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
                seen_urls.add(row["url"])

    sources = [s.strip() for s in args.sources.split(",")]
    for qi, query in enumerate(queries, 1):
        candidates = []
        if "openverse" in sources:
            candidates += openverse(query, args.limit, args.licenses)
        if "commons" in sources:
            candidates += commons(query, args.limit)
        print(f"[{qi}/{len(queries)}] {query!r}: {len(candidates)} candidates")

        for c in candidates:
            if c["url"] in seen_urls:
                continue
            seen_urls.add(c["url"])
            if not acceptable(c["license"], args.allow_nc):
                skipped["licence"] += 1
                continue
            raw = get(c["url"], binary=True, quiet=True)
            if not raw and c.get("fallback_url"):
                raw = get(c["fallback_url"], binary=True)
            if not raw:
                skipped["error"] += 1
                continue
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert("RGB")
                    if min(im.size) < args.min_side:
                        skipped["small"] += 1
                        continue
                    ph = imagehash.phash(im)
                    width, height = im.size
            except Exception:
                skipped["error"] += 1
                continue

            md5 = hashlib.md5(raw).hexdigest()
            if md5 in md5s or any(ph - p <= 2 for p in phs):
                skipped["duplicate"] += 1
                continue
            md5s.add(md5)
            phs.append(ph)

            name = f"{md5[:12]}.jpg"
            with open(os.path.join(IMG_DIR, name), "wb") as fh:
                fh.write(raw)
            c.update(file=name, width=width, height=height, query=query)
            rows.append(c)
            kept += 1
            time.sleep(0.2)   # be polite to the source
        time.sleep(0.5)

    fields = ["file", "source", "title", "creator", "creator_url", "license", "license_url",
              "landing_url", "url", "query", "width", "height"]
    with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nkept {kept} new images in {IMG_DIR}")
    print(f"  skipped: {skipped}")
    print(f"  attribution for all {len(rows)} images in {MANIFEST}")
    print("\nThese are unlabelled. Pre-annotate with:")
    print("  python scripts/pseudo_label.py --weights runs/yolo11s/weights/best.pt")


if __name__ == "__main__":
    main()
