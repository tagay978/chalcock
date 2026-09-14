"""Scrape the official IBA cocktail list into data/iba_cocktails.json.

The IBA site is the authority for these recipes, so the specs are pulled from it rather than
written by hand. Each cocktail page exposes three Elementor sections - Ingredients, Method,
Garnish - and the category comes from which of the three listing pages the cocktail appears on.

Run:  python scripts/fetch_iba.py
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "iba_cocktails.json")
BASE = "https://iba-world.com"
CATEGORIES = {
    "the-unforgettables": "The Unforgettables",
    "the-contemporary": "Contemporary Classics",
    "the-new-era": "New Era Drinks",
}
UA = {"User-Agent": "Mozilla/5.0 (compatible; cocktail-dataset-builder/1.0)"}


def get(url: str, tries: int = 4) -> str:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as exc:
            if attempt == tries - 1:
                raise
            print(f"    retry {attempt + 1} ({exc.__class__.__name__})", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def strip_tags(fragment: str) -> str:
    fragment = re.sub(r"<li[^>]*>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<(p|br|/p|/div)[^>]*>", "\n", fragment, flags=re.I)
    text = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(text).replace("\xa0", " ")


def lines(fragment: str) -> list[str]:
    return [re.sub(r"\s+", " ", ln).strip() for ln in strip_tags(fragment).split("\n") if ln.strip()]


def section(page: str, heading: str) -> str:
    """Return the Elementor shortcode block that follows the given <h4> heading."""
    m = re.search(rf">{re.escape(heading)}</h4>", page, re.I)
    if not m:
        return ""
    after = page[m.end():]
    block = re.search(r'<div class="elementor-shortcode">(.*?)</div>', after, re.S)
    return block.group(1) if block else ""


def slugs_by_category() -> dict[str, str]:
    """Walk each category's paginated listing so every cocktail carries its official category."""
    found: dict[str, str] = {}
    for slug, label in CATEGORIES.items():
        page_no, seen_here = 1, 0
        while True:
            url = f"{BASE}/cocktails/{slug}/" if page_no == 1 else f"{BASE}/cocktails/{slug}/page/{page_no}/"
            try:
                page = get(url)
            except Exception:
                break
            hits = set(re.findall(r"/iba-cocktail/([a-z0-9-]+)/", page))
            new = [h for h in hits if h not in found]
            if not new:
                break
            for h in new:
                found[h] = label
            seen_here += len(new)
            page_no += 1
        print(f"  {label}: {seen_here}")
    return found


def main():
    print("collecting cocktail list")
    cats = slugs_by_category()
    print(f"  {len(cats)} cocktails total\n")

    cocktails = []
    for i, (slug, category) in enumerate(sorted(cats.items()), 1):
        page = get(f"{BASE}/iba-cocktail/{slug}/")
        title = re.search(r"<title>(.*?)</title>", page, re.S)
        name = html.unescape(title.group(1)).split("–")[0].split("|")[0].strip() if title else slug
        ingredients = lines(section(page, "Ingredients"))
        method = " ".join(lines(section(page, "Method")))
        garnish = " ".join(lines(section(page, "Garnish")))
        if not ingredients:
            print(f"  !! no ingredients parsed for {slug}", file=sys.stderr)
        cocktails.append({
            "name": name,
            "slug": slug,
            "category": category,
            "ingredients": ingredients,
            "method": method,
            "garnish": garnish,
            "url": f"{BASE}/iba-cocktail/{slug}/",
        })
        print(f"  [{i:3}/{len(cats)}] {name:32} {len(ingredients)} ingredients")
        time.sleep(0.3)   # be polite to the source

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"source": f"{BASE}/cocktails/all-cocktails/",
                   "retrieved": time.strftime("%Y-%m-%d"),
                   "cocktails": cocktails}, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {OUT} ({len(cocktails)} cocktails)")
    empty = [c["slug"] for c in cocktails if not c["ingredients"]]
    if empty:
        print("  MISSING ingredients:", ", ".join(empty))


if __name__ == "__main__":
    main()
