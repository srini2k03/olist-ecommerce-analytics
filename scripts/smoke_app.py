"""End-to-end smoke test: start `streamlit run` and click through all 10 dashboard pages.

Usage:
    python scripts/smoke_app.py [--screenshot results/figures/app_overview.png]
                                [--shots-dir DIR] [--width 1280] [--report timings.json]

Fails on any Streamlit exception or browser error. With --screenshot it saves the overview
there and the churn-risk lookup next to it as <stem>_churn.png (the README figures). With
--shots-dir it saves every page. Set PLAYWRIGHT_CHROMIUM_EXECUTABLE to use a system Chrome.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "src")]  # works without `pip install -e .`
TITLE = "Olist marketplace analytics"


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


def page_titles() -> list[tuple[str, str]]:
    from dashboard.app import PAGES

    return [(p[0], p[1]) for p in PAGES]


def settle(page, ms: int = 600) -> None:
    """Wait until Streamlit has finished running the script (no 'Running...' status)."""
    page.wait_for_timeout(ms)
    page.locator("[data-testid='stStatusWidget']").wait_for(state="hidden", timeout=120_000)
    page.wait_for_timeout(ms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--shots-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    port = free_port()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    if args.theme == "light":
        cmd += [
            "--theme.base",
            "light",
            "--theme.backgroundColor",
            "#f6f6f2",
            "--theme.secondaryBackgroundColor",
            "#ffffff",
            "--theme.textColor",
            "#0c0f14",
            "--theme.primaryColor",
            "#0a7d52",
            "--theme.borderColor",
            "#e3e4e0",
            "--theme.codeBackgroundColor",
            "#ffffff",
            "--theme.dataframeHeaderBackgroundColor",
            "#ffffff",
            "--theme.linkColor",
            "#0a7d52",
        ]
    proc = subprocess.Popen(cmd, cwd=ROOT)  # noqa: S603
    timings: dict[str, float] = {}
    try:
        wait_for(f"http://127.0.0.1:{port}/_stcore/health")
        exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=exe)
            height = 1000 if args.width >= 800 else 844
            if args.shots_dir and args.width >= 800:
                # Streamlit scrolls inside its own container, so a tall viewport is the
                # simplest way to capture whole pages.
                height = 3400
            # The OS colour scheme must not change the app's look: emulate the opposite one.
            scheme = "light" if args.theme == "dark" else "dark"
            page = browser.new_page(
                viewport={"width": args.width, "height": height}, color_scheme=scheme
            )
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            t0 = time.perf_counter()
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.get_by_role("heading", name=TITLE).first.wait_for(timeout=90_000)
            settle(page)
            timings["first_render_s"] = round(time.perf_counter() - t0, 2)
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshot), full_page=False)
            for title, url_path in page_titles():
                t1 = time.perf_counter()
                if args.width < 800:  # the sidebar is collapsed on phones: open by URL
                    path = "" if title == "Executive overview" else url_path  # default page is /
                    page.goto(f"http://127.0.0.1:{port}/{path}", wait_until="networkidle")
                else:
                    page.get_by_role("link", name=title).first.click()
                heading = TITLE if title == "Executive overview" else title
                page.get_by_role("heading", name=heading).first.wait_for(timeout=120_000)
                settle(page)
                timings[f"page:{title}"] = round(time.perf_counter() - t1, 2)
                if page.locator("[data-testid='stException']").count():
                    text = page.locator("[data-testid='stException']").first.inner_text()
                    raise RuntimeError(f"Streamlit exception on {title!r}: {text[:800]}")
                if args.shots_dir:
                    args.shots_dir.mkdir(parents=True, exist_ok=True)
                    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
                    page.screenshot(
                        path=str(args.shots_dir / f"{args.width}-{args.theme}-{slug}.png"),
                        full_page=True,
                    )
                if title == "Customers" and args.screenshot:
                    page.get_by_text("P(repeat in 180 days)").first.wait_for(timeout=30_000)
                    page.get_by_text("Churn-risk lookup").first.scroll_into_view_if_needed()
                    page.wait_for_timeout(800)
                    churn = args.screenshot.with_name(args.screenshot.stem + "_churn.png")
                    page.screenshot(path=str(churn), full_page=False)
            if errors:
                raise RuntimeError(f"browser errors: {errors}")
            browser.close()
        print("timings:", json.dumps(timings))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(timings, indent=2))
        print("smoke test passed")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=20)


if __name__ == "__main__":
    raise SystemExit(main())
