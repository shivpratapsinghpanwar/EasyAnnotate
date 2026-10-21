"""Measure the canvas spike honestly.

A first attempt drove this with Playwright's synthetic pointer events and got
~20ms per frame regardless of whether there were 300 shapes or 3000 -- which is
not a rendering cost, it is the ~20ms CDP round trip of each synthetic event.
The measurement was of the harness.

So the numbers below are taken inside the page: the cost of draw() alone, the
cost of hit-testing alone, and the frame interval of a loop that invalidates
every frame. Those are what a real user's hand would actually be waiting on.

Gate from the plan: pan, zoom and drag at 60fps with 300 shapes on a 4000x3000
image. 60fps is 16.7ms per frame.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "spike.html").as_uri()
SHOTS = HERE / "shots"
BUDGET_MS = 16.7


def load(page, shapes: int):
    page.goto(f"{PAGE}?shapes={shapes}")
    page.wait_for_function("window.__ready === true", timeout=90_000)
    page.wait_for_timeout(500)


def show(label: str, stats: dict, budget: float = BUDGET_MS) -> float:
    verdict = "PASS" if stats["p95"] <= budget else "OVER"
    print(f"  {verdict}  {label:<38} p50 {stats['p50']:6.2f}ms  p95 {stats['p95']:6.2f}ms  "
          f"worst {stats['worst']:6.2f}ms", flush=True)
    return stats["p95"]


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})

    load(page, 300)
    print(f"\n{page.evaluate('window.__spike.shapeCount()')} shapes, 4000x3000 image, "
          f"1600x1000 viewport. Budget {BUDGET_MS}ms/frame for 60fps.\n", flush=True)

    print("cost of one draw(), by zoom level:", flush=True)
    for zoom, name in ((0.35, "fit (whole image)"), (1.0, "1:1"), (3.0, "3x zoomed in")):
        page.evaluate(f"window.__spike.zoom({zoom})")
        page.wait_for_timeout(100)
        show(f"draw at {name}", page.evaluate("window.__spike.measureDraw(150)"))

    print("\ncost of one hit-test (paid on every pointer move):", flush=True)
    page.evaluate("window.__spike.zoom(0.35)")
    show("hit-test, 300 shapes", page.evaluate("window.__spike.measureHitTest(3000)"), budget=2.0)

    print("\nsustained interaction, driven by requestAnimationFrame:", flush=True)
    print("  (the loop is vsync-locked, so a p50 of 16.7ms IS 60fps -- judge the"
          " interval\n   by p50, and judge our own cost by draw())", flush=True)
    worst_draw = 0.0
    for mode in ("pan", "zoom", "drag"):
        result = page.evaluate(f"window.__spike.stress(2000, '{mode}')")
        frame, drawn = result["frame"], result["draw"]
        fps = 1000.0 / frame["p50"] if frame["p50"] else 0
        verdict = "PASS" if frame["p50"] <= 17.0 else "OVER"
        print(f"  {verdict}  {mode + ': sustained':<38} {fps:4.0f} fps "
              f"(interval p50 {frame['p50']:5.2f}ms)", flush=True)
        worst_draw = max(worst_draw, show(f"{mode}: our draw() within the frame", drawn))

    print("\nheadroom -- how many shapes before it breaks:", flush=True)
    print("  (p50 is the honest steady-state cost; a large gap to p95 is a GC "
          "pause, not\n   per-frame work)", flush=True)
    for count in (300, 1000, 3000, 10000):
        load(page, count)
        page.evaluate("window.__spike.zoom(0.35)")
        page.wait_for_timeout(400)
        page.evaluate("window.__spike.measureDraw(40)")      # warm up, discard
        draw = page.evaluate("window.__spike.measureDraw(200)")
        hit = page.evaluate("window.__spike.measureHitTest(2000)")
        flag = "PASS" if draw["p50"] <= BUDGET_MS else "OVER"
        print(f"  {flag}  {count:>6} shapes   draw p50 {draw['p50']:6.2f}ms  "
              f"p95 {draw['p95']:7.2f}ms   hit-test p50 {hit['p50']:5.3f}ms "
              f"p95 {hit['p95']:5.3f}ms", flush=True)

    load(page, 300)
    page.evaluate("window.__spike.zoom(0.35)")
    page.wait_for_timeout(300)
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / "canvas-spike.png"))

    print(f"\nworst cost of our own draw() during sustained interaction: "
          f"{worst_draw:.2f}ms of the {BUDGET_MS}ms frame "
          f"({worst_draw / BUDGET_MS:.0%} of the budget)", flush=True)
    browser.close()
