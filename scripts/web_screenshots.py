"""Refresh the screenshots embedded in ``src/sweets/web/README.md``.

Boots a temporary uvicorn server against the built frontend bundle, drives
Playwright through every tab + interaction, and writes the resulting PNGs into
``docs/web/screenshots/``.

Prerequisites
-------------
- ``pixi install -e web`` (gets fastapi, uvicorn, playwright, opera_utils,
  etc.) and the bundled chromium: ``playwright install chromium``.
- ``pixi run -e web web-install && pixi run -e web web-build`` so a
  ``src/sweets/web/frontend/dist/`` exists for the backend to serve.
- Network access to ASF / CMR — the search step hits real granule services
  to produce a representative results overlay. Without network the screenshot
  will just show an empty search panel, which is also fine.

Usage
-----
::

    pixi run -e web web-screenshots
    # or directly:
    python scripts/web_screenshots.py

Refresh cadence
---------------
Re-run after any UI change that touches the layout, the map overlays, or the
config form. The script is idempotent; it overwrites the PNGs in place. CI
doesn't run it (it needs a GUI browser binary and live ASF / CMR queries).
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "web" / "screenshots"
FRONTEND_DIST = REPO_ROOT / "src" / "sweets" / "web" / "frontend" / "dist"

# A tight Los Angeles AOI + 2-month window that consistently returns ~30+
# OPERA CSLC granules spanning two tracks — gives us a real missing-data
# overlay + a useful track picker to demo.
DEMO_BBOX = (-118.50, 33.95, -118.00, 34.35)
DEMO_DATES = ("2024-01-01", "2024-03-01")


def free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: float = 30.0) -> None:
    """Poll the backend until /api/health responds."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.3)
    msg = f"backend at {url} did not come up within {timeout}s"
    raise RuntimeError(msg)


def take_shots(port: int) -> None:
    """Drive Playwright through every demo flow."""
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}/"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        page.goto(base, wait_until="networkidle")

        # 1. Search (empty)
        page.screenshot(path=str(OUT_DIR / "01-search-empty.png"))
        print("01-search-empty.png")

        # 2. Search results over Los Angeles
        w, s, e, n = DEMO_BBOX
        page.get_by_label("West", exact=True).fill(str(w))
        page.get_by_label("South", exact=True).fill(str(s))
        page.get_by_label("East", exact=True).fill(str(e))
        page.get_by_label("North", exact=True).fill(str(n))
        page.get_by_label("Start", exact=True).fill(DEMO_DATES[0])
        page.get_by_label("End", exact=True).fill(DEMO_DATES[1])
        page.get_by_role("button", name="Search", exact=True).click()
        # Wait for the track list (only appears after results land) so the
        # screenshot includes the missing-data summary + track chips.
        try:
            page.wait_for_selector(".track-list", timeout=60_000)
        except Exception as ex:
            print(f"  (no track-list within 60s: {ex})", file=sys.stderr)
        time.sleep(1.5)  # leaflet animation settle
        page.screenshot(path=str(OUT_DIR / "02-search-results.png"))
        print("02-search-results.png")

        # 2b. Pick the highest-count track to narrow the stack
        page.locator(".track-item").first.click()
        try:
            page.wait_for_selector(".track-item.selected", timeout=30_000)
        except Exception:
            pass
        time.sleep(1.5)
        page.screenshot(path=str(OUT_DIR / "02b-search-track-narrowed.png"))
        print("02b-search-track-narrowed.png")

        # 2c. Expand the granule list
        page.locator("h2:has-text('Granules') button").click()
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "02c-granule-list.png"))
        print("02c-granule-list.png")

        # 3. Config (basic, full-page)
        page.get_by_role("navigation").get_by_text("Config", exact=True).click()
        page.wait_for_selector("text=Configure a new job", timeout=15_000)
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "03-config-form.png"))
        print("03-config-form.png")

        # 3b. Config + show advanced (dolphin / tropo). Take two scrolled
        # captures because .content.full clips at the viewport.
        page.get_by_label("Show advanced", exact=False).check()
        time.sleep(0.6)
        page.evaluate("document.querySelector('.content.full').scrollTop = 1050")
        time.sleep(0.2)
        page.screenshot(path=str(OUT_DIR / "03b-config-advanced-dolphin.png"))
        print("03b-config-advanced-dolphin.png")

        page.evaluate("document.querySelector('.content.full').scrollTop = 2400")
        time.sleep(0.2)
        page.screenshot(path=str(OUT_DIR / "03c-config-advanced-tropo.png"))
        print("03c-config-advanced-tropo.png")

        # 4. Jobs (empty)
        page.get_by_role("navigation").get_by_text("Jobs", exact=True).click()
        page.wait_for_selector("text=Jobs (")
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "04-jobs-empty.png"))
        print("04-jobs-empty.png")

        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip rebuilding the frontend (use the existing dist/).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind backend to this port (default: OS-assigned).",
    )
    args = parser.parse_args()

    if not args.no_build:
        print("building frontend...")
        subprocess.check_call(
            ["npm", "run", "build"],
            cwd=str(REPO_ROOT / "src" / "sweets" / "web" / "frontend"),
        )

    if not FRONTEND_DIST.exists():
        msg = (
            f"frontend bundle missing at {FRONTEND_DIST}; run "
            "`pixi run -e web web-install && pixi run -e web web-build`"
        )
        raise SystemExit(msg)

    port = args.port or free_port()
    print(f"starting uvicorn on :{port}")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "sweets.web.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ},
    )
    try:
        wait_for_server(f"http://127.0.0.1:{port}/api/health")
        take_shots(port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\nWrote {len(list(OUT_DIR.glob('*.png')))} PNGs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
