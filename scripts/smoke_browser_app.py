"""Smoke test for the in-browser (stlite) dashboard built into site/app.

Serves site/ over HTTP, opens /app/ in Chromium and waits until the Streamlit
app has rendered its title, with no Python exception shown on the page. It
prints the time to first render, and on failure it prints the browser console,
the page's visible text and any Streamlit exception so the CI log shows the cause.
Usage: python scripts/smoke_browser_app.py [--screenshot PATH] [--pages "A,B"]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
import time
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
TITLE = "Olist marketplace analytics"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 (stdlib signature)
        pass


def dump_page(page, console: list[str], errors: list[str]) -> None:
    """Print what the browser saw, for the CI log."""
    print("---- browser console (last 60 lines) ----")
    for line in console[-60:]:
        print("  ", line[:400])
    print("---- page errors ----")
    for e in errors[:20]:
        print("  ", e[:600])
    try:
        for i, el in enumerate(page.locator(".stException").all()[:3]):
            print(f"---- Streamlit exception {i + 1} ----")
            print(el.inner_text()[:3000])
        print("---- visible page text (first 3000 chars) ----")
        print(page.locator("body").inner_text()[:3000])
    except PlaywrightError as exc:  # page may be gone
        print("could not read the page:", exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=240)
    parser.add_argument(
        "--pages", default="", help="comma-separated sidebar page labels to open after load"
    )
    parser.add_argument("--report", type=Path, default=None, help="write timings as JSON")
    args = parser.parse_args()

    handler = functools.partial(QuietHandler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/app/"
    console: list[str] = []
    errors: list[str] = []
    timings: dict[str, float] = {}
    ok = True
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
            page.on("pageerror", lambda e: errors.append(str(e)))
            t0 = time.perf_counter()
            page.goto(url)
            try:
                page.get_by_text(TITLE).first.wait_for(timeout=args.timeout_s * 1000)
                timings["first_render_s"] = round(time.perf_counter() - t0, 1)
                print(f"title rendered after {timings['first_render_s']} s")
                page.wait_for_timeout(5000)  # let the remaining elements render
                for label in [x.strip() for x in args.pages.split(",") if x.strip()]:
                    t1 = time.perf_counter()
                    page.get_by_role("link", name=label).first.click()
                    heading = TITLE if label == "Executive overview" else label
                    page.get_by_role("heading", name=heading).first.wait_for(timeout=120_000)
                    page.wait_for_timeout(500)
                    status = page.locator("[data-testid='stStatusWidget']")
                    status.wait_for(state="hidden", timeout=180_000)
                    page.wait_for_timeout(500)
                    timings[f"page:{label}"] = round(time.perf_counter() - t1, 2)
                    if page.locator(".stException").count():
                        print(f"FAIL: exception on page {label!r}")
                        ok = False
                        break
            except PlaywrightError as exc:
                print(f"FAIL: {type(exc).__name__}: {str(exc).splitlines()[0]}")
                ok = False
            exceptions = page.locator(".stException").count()
            if exceptions:
                print(f"FAIL: {exceptions} Streamlit exception block(s) rendered")
                ok = False
            if not ok:
                dump_page(page, console, errors)
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshot), full_page=True)
            browser.close()
    finally:
        server.shutdown()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(timings, indent=2))
    print("timings:", json.dumps(timings))
    if not ok:
        return 1
    if errors:
        print("page errors (non-fatal):", *errors[:5], sep="\n  ")
    print(f"OK: in-browser dashboard rendered '{TITLE}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
