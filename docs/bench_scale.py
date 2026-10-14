"""Measure EasyAnnotate at 20,000 images: what a person waits for.

    uv run python docs/bench_scale.py            # prints a table
    uv run python docs/bench_scale.py 5000       # another size

Builds a real folder of small JPEGs, opens it as an attached project the way
`easyannotate annotate` does, then times the requests the editor makes and
the editor's own boot in a real browser. The gates are the ones docs/PLAN.md
set before any of this was built.
"""

from __future__ import annotations

import os
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
work = Path(tempfile.mkdtemp(prefix="easyannotate-bench-"))
os.environ["DATAFACTORY_WORKSPACE"] = str(work / "ws")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from PIL import Image  # noqa: E402

from data_factory.annotate import attach  # noqa: E402
from data_factory.service.app import app  # noqa: E402
from data_factory.service.deps import get_store  # noqa: E402

rows: list[tuple[str, float, str]] = []


def timed(label: str, gate_ms: float | None, fn, repeat: int = 5):
    times = []
    result = None
    for _ in range(repeat):
        t = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t) * 1000)
    ms = statistics.median(times)
    verdict = "" if gate_ms is None else ("pass" if ms <= gate_ms else f"OVER {gate_ms:g}ms")
    rows.append((label, ms, verdict))
    print(f"  {label:<44} {ms:9.1f} ms  {verdict}", flush=True)
    return result


def main() -> None:
    folder = work / "images"
    folder.mkdir()
    seed = folder / "img000000.jpg"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (48, 64, 3), np.uint8)).save(seed, quality=80)
    t = time.perf_counter()
    for i in range(1, N):
        shutil.copyfile(seed, folder / f"img{i:06d}.jpg")
    print(f"built {N:,} images in {time.perf_counter() - t:.1f}s", flush=True)

    store = get_store()
    t = time.perf_counter()
    project = attach.open_folder(store, folder)["project"]
    open_s = time.perf_counter() - t
    rows.append((f"open a {N:,}-image folder (first time)", open_s * 1000, ""))
    print(f"  open folder: {open_s:.1f}s", flush=True)

    # Uncertainty scores, as a model run would leave them, for the queue.
    rng = random.Random(7)
    with store.write() as conn:
        conn.executemany(
            "UPDATE image SET uncertainty = ?, state = 'predicted' WHERE project_id = ? AND seq = ?",
            [(rng.random() * 5, project.id, s) for s in range(0, N, 2)],
        )

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8796, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:8796/api/projects/{project.id}"

    with httpx.Client(timeout=60) as client:
        images = f"{base}/images"
        timed("image list, first page (200)", 200, lambda: client.get(images, params={"limit": 200}).raise_for_status())
        timed("image list, last page (200)", 200, lambda: client.get(images, params={"limit": 200, "offset": N - 200}).raise_for_status())
        timed("least-sure-first queue, first page", 200, lambda: client.get(images, params={"state": "predicted", "order": "uncertain", "limit": 200}).raise_for_status())
        last = client.get(images, params={"limit": 1, "offset": N - 1}).json()["images"][0]
        timed("locate an image near the end", 200, lambda: client.get(images, params={"limit": 1, "locate": last["id"]}).raise_for_status())
        timed("locate it in the uncertainty queue", 200, lambda: client.get(images, params={"state": "predicted", "order": "uncertain", "limit": 1, "locate": last["id"]}).raise_for_status())
        detail = timed("open an image (shapes)", 300, lambda: client.get(f"{images}/{last['id']}").json())

        version = [detail["version"]]

        def save():
            response = client.put(
                f"{images}/{last['id']}/shapes",
                json={"version": version[0], "shapes": [{"label": "door", "bbox": [1, 1, 30, 30]}]},
            )
            response.raise_for_status()
            version[0] = response.json()["version"]

        timed("save a shape (round trip)", 100, save)
        timed("thumbnail", None, lambda: client.get(f"{images}/{last['id']}/thumb").raise_for_status())

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sync_playwright = None
    if sync_playwright is not None:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            t = time.perf_counter()
            page.goto(f"http://127.0.0.1:8796/annotate?project={project.id}")
            page.wait_for_function("window.__editor && window.__editor.image && document.querySelector('.strip-item')")
            boot = (time.perf_counter() - t) * 1000
            rows.append(("editor boot to first image, in a browser", boot, "pass" if boot <= 2000 else "OVER 2000ms"))
            print(f"  editor boot: {boot:.0f} ms", flush=True)
            nodes = page.locator(".strip-item").count()
            rows.append((f"strip rows in the DOM for {N:,} images", nodes, ""))
            t = time.perf_counter()
            page.evaluate("const l = document.querySelector('[data-role=strip-list]'); l.scrollTop = l.scrollHeight")
            page.wait_for_selector(f".strip-item >> text=img{N - 1:06d}.jpg")
            jump = (time.perf_counter() - t) * 1000
            rows.append(("scroll the strip to the last image", jump, "pass" if jump <= 1000 else "OVER 1000ms"))
            browser.close()
    server.should_exit = True

    print(f"\n| What ({N:,} images) | Median | Gate |\n|---|---:|---|")
    for label, value, verdict in rows:
        unit = "" if "rows in the DOM" in label else " ms"
        print(f"| {label} | {value:,.0f}{unit} | {verdict} |")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
