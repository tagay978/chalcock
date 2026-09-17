"""Fetch one licensed bottle photo per generic ingredient (gin, campari, sweet vermouth...).

Same policy as fetch_cocktail_images.py - Wikimedia Commons only, licence checked with
harvest_images.acceptable, attribution written next to the files - but for the ingredient info
page a recipe links to when a cocktail calls for "Gin" rather than a specific brand the detector
knows. Search terms are hand-picked per key (data/ingredient_rules.yaml's keys are not always
searchable English on their own, e.g. "creme_de_cacao_white").

Run:  python scripts/fetch_ingredient_images.py
      python scripts/fetch_ingredient_images.py --only gin,campari --force
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

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fetch_cocktail_images import get, search  # noqa: E402 - reuse the same Commons client
from harvest_images import acceptable  # noqa: E402

OUT = os.path.join(ROOT, "app", "static", "ingredients")
MANIFEST = os.path.join(OUT, "attribution.csv")
MIN_SIDE = 300

# key -> (search phrase, extra fallback phrase)
TERMS = {
    "absinthe": "Absinthe bottle",
    "aged_rum": "Aged rum bottle",
    "aguardiente": "Aguardiente bottle",
    "allspice_dram": "Pimento dram bottle",
    "amaretto": "Amaretto bottle",
    "amaro": "Amaro bottle",
    "angostura_bitters": "Angostura bitters bottle",
    "aperol": "Aperol bottle",
    "apricot_brandy": "Apricot brandy bottle",
    "benedictine": "Benedictine liqueur bottle",
    "blackstrap_rum": "Blackstrap rum bottle",
    "bourbon": "Bourbon whiskey bottle",
    "bourbon_or_rye": "Bourbon whiskey bottle",
    "brandy": "Brandy bottle",
    "cachaca": "Cachaca bottle",
    "calvados": "Calvados apple brandy bottle",
    "campari": "Campari bottle",
    "champagne": "Champagne bottle",
    "chartreuse_green": "Green Chartreuse bottle",
    "chartreuse_yellow": "Yellow Chartreuse bottle",
    "cherry_brandy": "Cherry brandy bottle",
    "citron_vodka": "Citron vodka bottle",
    "coconut_rum": "Coconut rum bottle",
    "coffee_liqueur": "Coffee liqueur bottle",
    "cognac": "Cognac bottle",
    "cointreau": "Cointreau bottle",
    "creme_de_cacao_white": "Creme de cacao bottle",
    "creme_de_cassis": "Creme de cassis bottle",
    "creme_de_menthe_white": "Creme de menthe bottle",
    "creme_de_mure": "Creme de mure liqueur",
    "creme_de_violette": "Creme de violette bottle",
    "curacao": "Curacao liqueur bottle",
    "cynar": "Cynar bottle",
    "demerara_rum": "Demerara rum bottle",
    "drambuie": "Drambuie bottle",
    "dry_vermouth": "Dry vermouth bottle",
    "falernum": "Falernum bottle",
    "fernet": "Fernet Branca bottle",
    "frangelico": "Frangelico bottle",
    "gin": "Gin bottle",
    "gold_rum": "Gold rum bottle",
    "goslings_rum": "Goslings rum bottle",
    "grand_marnier": "Grand Marnier bottle",
    "grappa": "Grappa bottle",
    "irish_whiskey": "Irish whiskey bottle",
    "islay_scotch": "Islay whisky bottle",
    "jamaican_rum": "Jamaican rum bottle",
    "lillet_blanc": "Lillet Blanc bottle",
    "maraschino": "Maraschino liqueur bottle",
    "mezcal": "Mezcal bottle",
    "old_tom_gin": "Old Tom gin bottle",
    "orange_bitters": "Orange bitters bottle",
    "overproof_rum": "Overproof rum bottle",
    "passion_fruit_liqueur": "Passion fruit liqueur bottle",
    "peach_brandy": "Peach brandy bottle",
    "peach_schnapps": "Peach schnapps bottle",
    "pernod": "Pernod bottle",
    "peychauds_bitters": "Peychauds bitters bottle",
    "pisco": "Pisco bottle",
    "port": "Port wine bottle",
    "prosecco": "Prosecco bottle",
    "raspberry_liqueur": "Raspberry liqueur bottle",
    "red_wine": "Red wine bottle",
    "rhum_agricole": "Rhum agricole bottle",
    "rum": "Rum bottle",
    "rye": "Rye whiskey bottle",
    "scotch": "Scotch whisky bottle",
    "sherry_amontillado": "Amontillado sherry bottle",
    "sherry_palo_cortado": "Palo Cortado sherry bottle",
    "sparkling_wine": "Sparkling wine bottle",
    "sweet_vermouth": "Sweet vermouth bottle",
    "tequila": "Tequila bottle",
    "triple_sec": "Triple sec bottle",
    "vanilla_vodka": "Vanilla vodka bottle",
    "vodka": "Vodka bottle",
    "white_rum": "White rum bottle",
    "white_wine": "White wine bottle glass",
}


# Commons' search is full-text, not "is this a photo of X" - a Public Domain hit for "Old Tom
# gin" turned out to be a scanned 1749 novel called Tom Jones. Both checks below exist because
# either alone let a scanned book through: the extension check catches library scans (still
# rendered as a JPG thumbnail by the API, so the URL alone does not reveal the source), the
# keyword check catches an image whose title just doesn't mention the ingredient at all.
NON_PHOTO_EXT = re.compile(r"\.(pdf|djvu|tif|tiff|svg|gif|xcf)$", re.I)
# A scanned document (patent filing, cave survey, library plate) is still a plain .jpg as far
# as the extension check is concerned - these title fragments are how such scans get flagged.
DOCUMENT_HINTS = re.compile(
    r"\b(patent|specification|dpla|survey|grotte|grotta|cave|plan|map|manuscript|"
    r"gazette|advert(isement)?|poster)\b", re.I)


def core_words(term):
    """The content words worth requiring in a candidate's title - not "bottle" itself, which
    would accept anything, and not words too short to mean anything on their own."""
    words = [w for w in re.split(r"[^A-Za-z]+", term.lower()) if len(w) > 2 and w != "bottle"]
    return words or [term.lower()]


def looks_like_bottle(title, words, im):
    """A generic ingredient search returns book scans and landscapes too; keep it to something
    the title actually claims is the ingredient, and photographic (portrait/near-square) rather
    than a wide scene.

    A pixel-level "is this a photo" check was tried and dropped: a clear-glass bottle on a
    plain background quantises down to almost as few colours as a scanned page does, so it
    rejected as many good photos as bad ones. The title checks below catch the real offenders
    (a 1749 novel for "Old Tom gin", a cave survey for "Calvados") without that collateral
    damage.
    """
    if NON_PHOTO_EXT.search(title) or DOCUMENT_HINTS.search(title):
        return False
    low = title.lower()
    if not any(re.search(rf"\b{re.escape(w)}", low) for w in words):
        return False
    w, h = im.size
    return w / h <= 1.15


def pick(key, term, force_terms=None):
    terms = force_terms or [term, term.replace(" bottle", "")]
    words = core_words(term)
    for t in terms:
        for cand in search(t):
            if not acceptable(cand["license"], allow_nc=False):
                continue
            if NON_PHOTO_EXT.search(cand["title"]) or DOCUMENT_HINTS.search(cand["title"]):
                continue
            raw = get(cand["url"], binary=True)
            if not raw:
                continue
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if min(w, h) < MIN_SIDE or not looks_like_bottle(cand["title"], words, im):
                        continue
                    im.thumbnail((700, 700))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=86)
            except Exception:
                continue
            return cand, buf.getvalue()
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated ingredient keys, for re-fetching a few")
    ap.add_argument("--force", action="store_true", help="refetch even if a file exists")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.only.split(",")} if args.only else None
    os.makedirs(OUT, exist_ok=True)

    rows = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows[row["slug"]] = row

    got, skipped, failed = 0, 0, []
    items = sorted(TERMS.items())
    for i, (key, term) in enumerate(items, 1):
        if wanted and key not in wanted:
            continue
        dest = os.path.join(OUT, key + ".jpg")
        if os.path.exists(dest) and not args.force:
            skipped += 1
            continue
        cand, data = pick(key, term)
        if cand is None:
            failed.append(key)
            print(f"  [{i:3}/{len(items)}] {key:24} no usable image")
            continue
        with open(dest, "wb") as fh:
            fh.write(data)
        rows[key] = {"slug": key, "name": term, "file": key + ".jpg",
                    "title": cand["title"], "creator": cand["creator"],
                    "license": cand["license"], "license_url": cand["license_url"],
                    "landing_url": cand["landing_url"]}
        got += 1
        print(f"  [{i:3}/{len(items)}] {key:24} {cand['license']}")
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
