"""Record docs/demo.gif from the real editor on the real demo project.

Not a mock-up: it boots the service, builds `easyannotate demo`, drives the
page with Playwright, and writes the frames it saw. Re-run it whenever the
editor changes so the GIF on the README cannot go stale.

    uv run python docs/make_demo_gif.py            # writes docs/demo.gif
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "demo.gif"

work = Path(tempfile.mkdtemp(prefix="easyannotate-gif-"))
os.environ["DATAFACTORY_WORKSPACE"] = str(work / "ws")

import uvicorn  # noqa: E402
from PIL import Image  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from data_factory.annotate import attach, demo, review  # noqa: E402
from data_factory.service.app import app  # noqa: E402
from data_factory.service.deps import get_store  # noqa: E402

WIDTH, HEIGHT = 1280, 760
frames: list[tuple[Image.Image, int]] = []  # (frame, milliseconds)


def shoot(page, hold_ms: int) -> None:
    png = page.screenshot()
    frames.append((Image.open(io.BytesIO(png)).convert("RGB"), hold_ms))


def main() -> None:
    folder = work / "demo"
    built = demo.build(folder, images=12)
    store = get_store()
    project = attach.open_folder(store, folder, name="EasyAnnotate demo")["project"]
    with tempfile.TemporaryDirectory() as holding:
        review.import_predictions(
            store, project, demo.write_predictions(built["predictions"], Path(holding) / "run")
        )

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8798, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
        page.goto(f"http://127.0.0.1:8798/annotate?project={project.id}", wait_until="networkidle")
        page.wait_for_function("window.__editor && window.__editor.image")
        page.wait_for_timeout(600)
        rect = page.locator("#canvas").bounding_box()

        def at(x, y):
            return rect["x"] + x, rect["y"] + y

        def drag(x1, y1, x2, y2, steps=12):
            page.mouse.move(*at(x1, y1))
            page.mouse.down()
            for i in range(1, steps + 1):
                page.mouse.move(*at(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps))
                if i in (steps // 3, 2 * steps // 3):
                    shoot(page, 90)
            shoot(page, 120)
            page.mouse.up()

        # 1. A predicted image: dashed boxes, the review panel.
        page.wait_for_timeout(300)
        shoot(page, 1600)

        # 2. Drop weak guesses with the slider, then accept with A.
        slider = page.locator("#conf")
        slider.fill("45")
        page.wait_for_timeout(200)
        shoot(page, 1200)
        before = page.evaluate("window.__editor.image.id")
        page.keyboard.press("a")
        # Accept moves on to the next image.
        page.wait_for_function(f"window.__editor.image.id !== '{before}'")
        page.wait_for_timeout(400)
        shoot(page, 1300)

        # 3. On an untouched image: make a brand-new class by typing it, then
        #    draw. No schema step -- "/" , a name, Enter, drag.
        page.select_option("[data-role=strip-filter]", "todo")
        page.wait_for_timeout(300)
        page.locator(".strip-item").first.click()
        page.wait_for_timeout(600)
        page.keyboard.press("Escape")
        shoot(page, 900)
        page.keyboard.press("/")
        for ch in "pallet":
            page.keyboard.type(ch)
            shoot(page, 70)
        page.keyboard.press("Enter")
        page.wait_for_timeout(700)
        shoot(page, 900)
        drag(120, 90, 380, 300)
        page.wait_for_timeout(700)
        shoot(page, 1400)

        # 4. A second box in the same class, then a nudge.
        drag(520, 380, 760, 560)
        page.wait_for_timeout(500)
        shoot(page, 500)
        for _ in range(4):
            page.keyboard.press("Shift+ArrowRight")
            shoot(page, 80)
        page.wait_for_timeout(600)
        shoot(page, 1300)

        # 5. The shortcuts overlay, then the saved status.
        page.keyboard.press("?")
        page.wait_for_timeout(200)
        shoot(page, 1800)
        page.keyboard.press("?")
        page.wait_for_timeout(200)
        shoot(page, 1200)
        browser.close()
    server.should_exit = True

    # Downscaled and quantised to one shared palette: a README GIF has to load
    # on a phone, and a per-frame palette makes colours flicker between frames.
    target_width = 960
    scaled = [
        f.resize((target_width, round(f.height * target_width / f.width)), Image.LANCZOS)
        for f, _ in frames
    ]
    palette = scaled[0].quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in scaled]
    first, *rest = quantised
    first.save(
        OUT,
        save_all=True,
        append_images=rest,
        duration=[d for _, d in frames],
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
