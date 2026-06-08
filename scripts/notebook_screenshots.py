"""Take screenshots of the IFG workflow notebook in JupyterLab.

Boots a temporary JupyterLab server, drives Playwright through the notebook
interface, executes key cells, and saves PNGs to ``docs/screenshots/``.

Prerequisites
-------------
- ``pixi run -e web playwright install chromium``
- Notebook must exist at ``docs/ifg_workflow.ipynb``

Usage
-----
::

    pixi run -e web python scripts/notebook_screenshots.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "docs" / "ifg_workflow.ipynb"
OUT_DIR = REPO_ROOT / "docs" / "screenshots"
TOKEN = "sweets-nb-token"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: float = 40.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status in (200, 302, 403):
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    msg = f"JupyterLab at {url} did not start within {timeout}s"
    raise RuntimeError(msg)


def take_shots(port: int) -> None:
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}"
    # Open the notebook directly via the /lab/tree URL with the token
    nb_url = f"{base}/lab/tree/docs/ifg_workflow.ipynb?token={TOKEN}"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        # ── 1. Load notebook ──────────────────────────────────────────────────
        page.goto(nb_url, wait_until="networkidle")
        # Wait for JupyterLab to fully render the notebook editor
        page.wait_for_selector(".jp-NotebookPanel", timeout=30_000)
        time.sleep(2)

        # Dismiss "Select Kernel" dialog if it appears — pick the first option
        try:
            dialog = page.locator(".jp-Dialog")
            if dialog.is_visible(timeout=3_000):
                # Click the first kernel option in the select/list
                try:
                    page.get_by_role("button", name="Select").click(timeout=2_000)
                except Exception:
                    pass
                try:
                    page.keyboard.press("Enter")
                except Exception:
                    pass
                time.sleep(1)
        except Exception:
            pass

        page.screenshot(path=str(OUT_DIR / "nb-01-loaded.png"))
        print("nb-01-loaded.png")

        # ── 2. Run all cells ──────────────────────────────────────────────────
        # Use the Run menu > Run All Cells
        # First dismiss any lingering dialog with Escape
        page.keyboard.press("Escape")
        time.sleep(0.3)

        page.get_by_role("menubar").get_by_text("Run").click(timeout=10_000)
        time.sleep(0.5)
        page.get_by_role("menuitem", name="Run All Cells", exact=True).click(
            timeout=5_000
        )
        # Dismiss any "restart kernel?" dialog
        try:
            page.get_by_role("button", name="Restart").click(timeout=3_000)
        except Exception:
            pass
        # Wait for cells to execute (widgets render after kernel execution)
        time.sleep(8)
        page.screenshot(path=str(OUT_DIR / "nb-02-after-run.png"))
        print("nb-02-after-run.png")

        # Helper: scroll the JupyterLab notebook panel to a specific pixel offset.
        # JupyterLab windowed notebook uses jp-WindowedPanel-outer as the scroller.
        def nb_scroll(px: int) -> None:
            page.evaluate(
                """(px) => {
                    // Try the windowed panel outer container first (JLab 4.x)
                    const candidates = [
                        document.querySelector('.jp-WindowedPanel-outer'),
                        document.querySelector('.jp-NotebookPanel-notebook'),
                        document.querySelector('.jp-Notebook'),
                        document.querySelector('.lm-ScrollPanel'),
                        document.scrollingElement,
                    ];
                    const el = candidates.find(e => e && e.scrollHeight > e.clientHeight);
                    if (el) { el.scrollTop = px; }
                }""",
                px,
            )
            time.sleep(0.8)

        # ── 3. Scroll to top — title + config section ─────────────────────────
        nb_scroll(0)
        page.screenshot(path=str(OUT_DIR / "nb-03-top-config.png"))
        print("nb-03-top-config.png")

        # ── 4. Scroll slightly to show the widget tabs fully ──────────────────
        nb_scroll(400)
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "nb-04-widget-tabs.png"))
        print("nb-04-widget-tabs.png")

        # ── 5. Click "Search & Download" tab (ipywidgets renders as role=tab) ─
        try:
            # ipywidgets tab bar titles are <li> with class jp-TabBar-tab
            # Try clicking by text content
            page.locator(
                ".widget-tab-bar .p-TabBar-tab", has_text="Search & Download"
            ).first.click(timeout=4_000)
            time.sleep(1)
            nb_scroll(400)
            page.screenshot(path=str(OUT_DIR / "nb-05-search-tab.png"))
            print("nb-05-search-tab.png")
        except Exception as exc:
            print(f"  (search tab: {exc})", file=sys.stderr)

        # ── 6. Click "Processing" tab ─────────────────────────────────────────
        try:
            page.locator(
                ".widget-tab-bar .p-TabBar-tab", has_text="Processing"
            ).first.click(timeout=4_000)
            time.sleep(1)
            nb_scroll(400)
            page.screenshot(path=str(OUT_DIR / "nb-06-processing-tab.png"))
            print("nb-06-processing-tab.png")
        except Exception as exc:
            print(f"  (processing tab: {exc})", file=sys.stderr)

        # ── 7. Config builder section ─────────────────────────────────────────
        nb_scroll(1600)
        page.screenshot(path=str(OUT_DIR / "nb-07-config-builder.png"))
        print("nb-07-config-builder.png")

        # ── 8. Run section ────────────────────────────────────────────────────
        nb_scroll(2400)
        page.screenshot(path=str(OUT_DIR / "nb-08-run-section.png"))
        print("nb-08-run-section.png")

        # ── 9. Results section ────────────────────────────────────────────────
        nb_scroll(3200)
        page.screenshot(path=str(OUT_DIR / "nb-09-results.png"))
        print("nb-09-results.png")

        browser.close()


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"Notebook not found: {NOTEBOOK}")
        print("Run:  python scripts/make_ifg_notebook.py")
        return 1

    port = free_port()
    print(f"Starting JupyterLab on port {port}...")

    env = {
        **os.environ,
        "JUPYTER_TOKEN": TOKEN,
    }
    import shutil

    jupyter_bin = shutil.which("jupyter") or sys.executable + " -m jupyter"
    proc = subprocess.Popen(
        [
            jupyter_bin,
            "lab",
            "--no-browser",
            f"--port={port}",
            f"--ServerApp.token={TOKEN}",
            "--ServerApp.disable_check_xsrf=True",
            f"--notebook-dir={REPO_ROOT}",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(f"http://127.0.0.1:{port}/lab?token={TOKEN}")
        print(f"JupyterLab ready at http://127.0.0.1:{port}")
        take_shots(port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    pngs = list(OUT_DIR.glob("nb-*.png"))
    print(f"\nWrote {len(pngs)} PNG(s) to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
