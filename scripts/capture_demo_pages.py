"""Screenshot the real app pages (results, bottle detail, cocktail detail) for the demo video.

make_demo_video.py's segments are PIL mockups of the detector panel, not the actual web UI - this
captures the real thing (via a headless Playwright browser) so the video can show what page 2
(results) and page 3 (bottle/cocktail detail) actually look like. Needs `pip install playwright`
+ `playwright install chromium`, and the app already running.

Run:  python app/main.py --port 8712   # in another terminal
      python scripts/capture_demo_pages.py
"""
import os

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "samples", "_pages")
BASE = "http://127.0.0.1:8712"

os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    # --- photo 1: full flow (results -> bottle detail -> cocktail detail) ---
    page.goto(BASE, wait_until="networkidle")
    page.click('#samples img[src$="/1.jpeg"]')
    page.wait_for_url("**/#/results")
    page.wait_for_selector("#shot")
    page.wait_for_timeout(400)
    page.screenshot(path=f"{OUT}/demo_results_1.png", full_page=True)

    page.click('.grid > div:first-child a.row-card >> nth=1')   # Grand Marnier
    page.wait_for_selector(".detail-sheet")
    page.wait_for_timeout(300)
    page.screenshot(path=f"{OUT}/demo_bottle.png", full_page=True)

    page.click('.detail-body a.row-card >> nth=0')
    page.wait_for_selector(".detail-sheet")
    page.wait_for_timeout(300)
    page.screenshot(path=f"{OUT}/demo_cocktail.png", full_page=True)

    # --- photo 2: just the results page ---
    page.goto(BASE, wait_until="networkidle")
    page.click('#samples img[src$="/6.jpg"]')
    page.wait_for_url("**/#/results")
    page.wait_for_selector("#shot")
    page.wait_for_timeout(400)
    page.screenshot(path=f"{OUT}/demo_results_2.png", full_page=True)

    browser.close()

print("done")
