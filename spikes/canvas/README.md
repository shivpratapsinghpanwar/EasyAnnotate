# Canvas spike — can a hand-written editor hold 60fps?

The plan commits the annotation editor to hand-written ES modules with no build
step, so that the wheel stays self-contained. That decision is only defensible
if a plain canvas can actually carry a real editor. This spike answers that
before any of the editor exists.

```bash
pip install playwright && playwright install chromium
python spikes/canvas/bench.py
```

## The answer: yes, comfortably

Sustained interaction, 300 shapes on a 4000×3000 image in a 1600×1000 viewport:

| Interaction | Sustained | Our `draw()` within the frame |
|---|---|---|
| pan | **60 fps** | 0.6 ms p50 |
| zoom | **60 fps** | 0.9 ms p50 |
| drag with hit-testing | **60 fps** | 0.7 ms p50 |

Worst case our own rendering used **27% of the 16.7 ms frame budget**. The rest
of the frame is idle, which is the headroom the real editor will spend on a
filmstrip, a label rail and an undo stack.

**Hit-testing is effectively free** — 0.0 ms median, 0.2 ms p95, and still
0.4 ms at 10,000 shapes. The plan's decision to use a linear scan over shapes
rather than an R-tree is correct; a spatial index would be pure complexity.

## Headroom

| Shapes | `draw()` p50 |
|---|---|
| 300 | 0.5 ms |
| 1,000 | 1.5 ms |
| 3,000 | 4.6 ms |
| 10,000 | 21.2 ms — over budget |

The ceiling is around 5,000 shapes on one image. A realistic image has tens.

## The one technique that matters

The first run showed **29 ms per frame at fit-zoom** and 4.8 ms at 1:1 — the
opposite of the intuition that zoomed-out is cheaper. Drawing a 12-megapixel
bitmap scaled down to 1600px wide makes the browser resample all 12 megapixels
every frame; at 1:1 it only touches the visible region.

The fix is **mip levels**: pre-build the image halved repeatedly, then draw from
the level nearest the display size. That took fit-zoom from 29 ms to 0.6 ms —
a 48× improvement, and the difference between "unusable" and "idle".

One wrinkle worth keeping: at full resolution use the plain three-argument
`drawImage(bitmap, 0, 0)`. The explicit source-rectangle form costs 15 ms
against 4.8 ms for identical output, because it drops a fast blit path.

## Two measurement traps, recorded so nobody repeats them

**Driving it with Playwright measures Playwright.** The first harness used
synthetic pointer events and reported ~20 ms per frame whether there were 300
shapes or 3,000. That flatness was the tell: it was the ~20 ms CDP round trip
per synthetic event, not rendering. The real numbers are taken inside the page.

**A vsync-locked loop cannot beat 16.7 ms.** Judging the frame *interval* by
p95 marks a perfect 60 fps as a failure. Judge the interval by p50, and judge
our own cost by timing `draw()` directly.

## What the real editor takes from this

- Mip levels for the image, chosen per zoom.
- One `draw(ctx, scene, viewport)` as the only function touching the canvas.
- Linear hit-testing, tolerance constant in *screen* pixels (divided by scale),
  so a handle feels the same size at every zoom.
- Line widths divided by scale, so strokes stay 2 screen pixels — the detail
  that makes a canvas feel native rather than like a zoomed picture of an editor.
- Model shapes dashed and human shapes solid, opacity scaled by confidence:
  legible at any zoom and safe for colour-blind users, unlike a hue change.
