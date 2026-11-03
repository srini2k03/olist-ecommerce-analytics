"""End-to-end smoke test: start `streamlit run` and drive the page with Playwright.

Usage: python scripts/smoke_app.py [--screenshot results/figures/app_screenshot.png]
Set PLAYWRIGHT_CHROMIUM_EXECUTABLE to use a system Chrome instead of the
Playwright-managed browser.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 (localhost)
                if r.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"{url} did not come up")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path, default=None)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    port = free_port()
    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        cwd=ROOT,
    )
    try:
        wait_for(f"http://127.0.0.1:{port}/_stcore/health")
        exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=exe)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.get_by_text("Olist marketplace analytics").first.wait_for(timeout=60_000)
            page.get_by_text("GMV").first.wait_for()
            if args.screenshot:
                page.wait_for_timeout(3000)
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshot), full_page=False)
            for tab in ["Cohort retention", "Segments", "Delivery impact", "Churn-risk lookup"]:
                page.get_by_role("tab", name=tab).click()
                page.wait_for_timeout(800)
            page.get_by_text("P(repeat in 180 days)").first.wait_for(timeout=30_000)
            if args.screenshot:
                page.wait_for_timeout(1500)
                churn_png = args.screenshot.with_name(args.screenshot.stem + "_churn.png")
                page.screenshot(path=str(churn_png), full_page=False)
            if page.locator("[data-testid='stException']").count():
                raise RuntimeError("Streamlit rendered an exception")
            if errors:
                raise RuntimeError(f"browser errors: {errors}")
            browser.close()
        print("smoke test passed")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=20)


if __name__ == "__main__":
    raise SystemExit(main())
