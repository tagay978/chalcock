"""Fetch one licensed photo per IBA cocktail for the web demo.

Wikimedia Commons only, and only licences that permit reuse, with the attribution each one
requires written next to the files. The IBA's own photos are not used: the recipes are facts and
fine to cite, their photography is not ours to redistribute.

Matching is the whole problem here. A search for "Paradise cocktail" will happily return a
sunset, so a candidate has to look like a cocktail photo - the file title has to mention the
drink, and the image has to be roughly portrait or square rather than a landscape scene.

Run:  python scripts/fetch_cocktail_images.py
      python scripts/fetch_cocktail_images.py --only negroni,daiquiri --force
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from harvest_images import acceptable  # noqa: E402  - same licence policy as image harvesting

OUT = os.path.join(ROOT, "app", "static", "cocktails")
MANIFEST = os.path.join(OUT, "attribution.csv")
UA = {"User-Agent": "cocktail-bottle-detector/1.0 (portfolio demo; contact via repository)"}
MIN_SIDE = 300


def get(url, binary=False, tries=3):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                raw = r.read()
            return raw if binary else raw.decode("utf-8", "replace")
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def strip(html_text):
    return re.sub(r"<[^>]+>", "", html_text or "").strip()


def search(term, limit=8):
    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": term, "gsrnamespace": "6", "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": "900",
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
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        out.append({
            "title": page.get("title", ""),
            "creator": strip((meta.get("Artist") or {}).get("value", "")),
            "license": strip((meta.get("LicenseShortName") or {}).get("value", "")),
            "license_url": strip((meta.get("LicenseUrl") or {}).get("value", "")),
            "landing_url": info.get("descriptionurl", ""),
            "url": url,
            "index": page.get("index", 99),
        })
    return sorted(out, key=lambda r: r["index"])


def looks_like_drink(title, name):
    """The file title must mention the drink, so a 'Paradise' search cannot return a landscape."""
    words = [w for w in re.split(r"[^A-Za-z]+", name.lower()) if len(w) > 2]
    low = title.lower()
    return bool(words) and any(w in low for w in words)


def pick(cocktail, force_terms=None):
    name = cocktail["name"]
    terms = force_terms or [f"{name} cocktail", f"{name} drink", name]
    for term in terms:
        for cand in search(term):
            if not acceptable(cand["license"], allow_nc=False):
                continue
            if not looks_like_drink(cand["title"], name):
                continue
            raw = get(cand["url"], binary=True)
            if not raw:
                continue
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if min(w, h) < MIN_SIDE:
                        continue
                    # A cocktail photo is portrait or square; a wide frame is usually a scene.
                    if w / h > 1.7:
                        continue
                    im.thumbnail((900, 900))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=86)
            except Exception:
                continue
            return cand, buf.getvalue()
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated slugs, for re-fetching a few")
    ap.add_argument("--force", action="store_true", help="refetch even if a file exists")
    args = ap.parse_args()

    recipes = json.load(open(os.path.join(ROOT, "data", "iba_cocktails.json"),
                             encoding="utf-8"))["cocktails"]
    wanted = {s.strip() for s in args.only.split(",")} if args.only else None
    os.makedirs(OUT, exist_ok=True)

    rows = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows[row["slug"]] = row

    got, skipped, failed = 0, 0, []
    for i, c in enumerate(recipes, 1):
        slug = c["slug"]
        if wanted and slug not in wanted:
            continue
        dest = os.path.join(OUT, slug + ".jpg")
        if os.path.exists(dest) and not args.force:
            skipped += 1
            continue
        cand, data = pick(c)
        if cand is None:
            failed.append(slug)
            print(f"  [{i:3}/{len(recipes)}] {c['name']:28} no usable image")
            continue
        with open(dest, "wb") as fh:
            fh.write(data)
        rows[slug] = {"slug": slug, "name": c["name"], "file": slug + ".jpg",
                      "title": cand["title"], "creator": cand["creator"],
                      "license": cand["license"], "license_url": cand["license_url"],
                      "landing_url": cand["landing_url"]}
        got += 1
        print(f"  [{i:3}/{len(recipes)}] {c['name']:28} {cand['license']}")
        time.sleep(0.25)

    fields = ["slug", "name", "file", "title", "creator", "license", "license_url", "landing_url"]
    with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for slug in sorted(rows):
            writer.writerow(rows[slug])

    print(f"\n{got} fetched, {skipped} already present, {len(failed)} without a usable image")
    if failed:
        print("  no image:", ", ".join(failed))
    print(f"  attribution for {len(rows)} images in {MANIFEST}")


if __name__ == "__main__":
    main()
