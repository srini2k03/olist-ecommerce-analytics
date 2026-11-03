"""Smoke test for the in-browser (stlite) dashboard built into site/app.

Serves site/ over HTTP, opens /app/ in Chromium and waits until the Streamlit
app has rendered its title, with no Python exception shown on the page.
Usage: python scripts/smoke_browser_app.py [--screenshot PATH]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
TITLE = "Olist marketplace analytics"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=240)
    args = parser.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/app/"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(url)
            page.get_by_text(TITLE).first.wait_for(timeout=args.timeout_s * 1000)
            page.wait_for_timeout(5000)  # let the remaining elements render
            exceptions = page.locator(".stException").count()
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshot), full_page=True)
            browser.close()
    finally:
        server.shutdown()
    if exceptions:
        print(f"FAIL: {exceptions} Streamlit exception block(s) rendered")
        return 1
    if errors:
        print("page errors (non-fatal):", *errors[:5], sep="\n  ")
    print(f"OK: in-browser dashboard rendered '{TITLE}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
