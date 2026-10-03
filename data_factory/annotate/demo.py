"""A demo project that shows the whole loop in the first minute.

``easyannotate demo`` builds a folder of generated scenes -- nothing from a
real dataset ships in the package -- and opens it with three kinds of image
already in it, because each is a different first impression:

* a few **labelled by hand**, so resume, export and the class list have
  something to show;
* a run of **model predictions** waiting for review, slightly wrong on
  purpose, so the dashed boxes, the confidence slider and ``A`` to accept make
  sense on sight;
* the rest **untouched**, to draw on.

Deterministic: the same seed draws the same scenes, so a screenshot in the
README stays true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core import imaging
from ..core.annotations import voc
from ..core.annotations.model import Annotation, Shape

CLASSES = ("crate", "ball", "cone")
SIZE = (960, 640)  # width, height
MARKER = ".easyannotate-demo"


def _scene(rng: np.random.Generator) -> tuple[np.ndarray, list[Shape]]:
    """One picture and the truth about it: crates, balls and cones on a floor."""
    import cv2

    width, height = SIZE
    # A floor with a gentle gradient and grain, so a box edge is a judgement
    # rather than a perfectly crisp line.
    top, bottom = rng.integers(70, 120, 3), rng.integers(30, 70, 3)
    ramp = np.linspace(0, 1, height)[:, None, None]
    image = (top * (1 - ramp) + bottom * ramp).astype(np.float32)
    image = np.repeat(image, width, axis=1)
    image += rng.normal(0, 6, image.shape)

    shapes: list[Shape] = []
    for _ in range(int(rng.integers(2, 6))):
        kind = CLASSES[int(rng.integers(0, len(CLASSES)))]
        w = int(rng.integers(60, 200))
        h = int(w * rng.uniform(0.6, 1.3))
        x = int(rng.integers(10, width - w - 10))
        y = int(rng.integers(10, height - h - 10))
        colour = tuple(int(c) for c in rng.integers(90, 240, 3))
        if kind == "crate":
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (25, 20, 15), 4)
            cv2.line(image, (x, y), (x + w, y + h), (25, 20, 15), 3)
            shapes.append(Shape(label=kind, bbox=(x, y, x + w, y + h)))
        elif kind == "ball":
            r = min(w, h) // 2
            cx, cy = x + r, y + r
            cv2.circle(image, (cx, cy), r, colour, -1)
            cv2.circle(image, (cx - r // 3, cy - r // 3), max(2, r // 5), (245, 245, 245), -1)
            shapes.append(Shape(label=kind, bbox=(cx - r, cy - r, cx + r, cy + r)))
        else:
            points = np.array([[x + w // 2, y], [x + w, y + h], [x, y + h]], np.int32)
            cv2.fillPoly(image, [points], colour)
            shapes.append(
                Shape(
                    label=kind,
                    bbox=(x, y, x + w, y + h),
                    polygon=[(int(px), int(py)) for px, py in points],
                )
            )
    return np.clip(image, 0, 255).astype(np.uint8), shapes


def _jitter(rng: np.random.Generator, shape: Shape) -> Shape:
    """What a half-trained model returns: close, not exact, with a score."""
    x1, y1, x2, y2 = shape.bbox
    w, h = x2 - x1, y2 - y1
    dx, dy = rng.normal(0, w * 0.06), rng.normal(0, h * 0.06)
    grow = rng.normal(1, 0.06)
    nx1, ny1 = x1 + dx, y1 + dy
    return Shape(
        label=shape.label,
        bbox=(
            int(nx1),
            int(ny1),
            int(nx1 + w * grow),
            int(ny1 + h * grow),
        ),
        confidence=round(float(rng.uniform(0.35, 0.98)), 3),
    )


def build(folder: Path, *, images: int = 24, seed: int = 7) -> dict[str, Any]:
    """Write the demo into ``folder``; return what the caller must import.

    Returns ``{"fresh": bool, "predictions": {name: [Shape]}}``. ``fresh`` is
    False when the folder was already built, so a second ``demo`` reopens the
    same project -- with whatever was done to it -- instead of rebuilding it.
    """
    folder = Path(folder)
    if (folder / MARKER).exists():
        return {"fresh": False, "predictions": {}}
    folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    labelled = max(1, images // 6)
    predicted_until = labelled + images // 2
    predictions: dict[str, list[Shape]] = {}

    for index in range(images):
        name = f"scene_{index:03d}.jpg"
        pixels, truth = _scene(rng)
        imaging.save(pixels, folder / name, quality=90)
        if index < labelled:
            voc.write(
                Annotation(filename=name, width=SIZE[0], height=SIZE[1], shapes=truth),
                folder / f"scene_{index:03d}.xml",
            )
        elif index < predicted_until:
            found = [_jitter(rng, shape) for shape in truth if rng.random() > 0.12]
            # Now and then a false positive, which is most of what reviewing is.
            if rng.random() < 0.3:
                x, y = int(rng.integers(0, SIZE[0] - 90)), int(rng.integers(0, SIZE[1] - 90))
                found.append(
                    Shape(
                        label=CLASSES[int(rng.integers(0, len(CLASSES)))],
                        bbox=(x, y, x + 80, y + 80),
                        confidence=round(float(rng.uniform(0.2, 0.5)), 3),
                    )
                )
            predictions[name] = found

    (folder / MARKER).write_text("generated by easyannotate demo\n", encoding="utf-8")
    return {"fresh": True, "predictions": predictions}


def write_predictions(predictions: dict[str, list[Shape]], target: Path) -> Path:
    """The predictions as Ultralytics writes them with ``--save-conf``: one
    ``.txt`` per image, ``class cx cy w h confidence`` normalised, beside a
    ``data.yaml`` naming the classes. So the demo imports through exactly the
    path a real model run takes."""
    target.mkdir(parents=True, exist_ok=True)
    width, height = SIZE
    names = "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    (target / "data.yaml").write_text(f"names:\n{names}", encoding="utf-8")
    for name, shapes in predictions.items():
        lines = []
        for shape in shapes:
            x1, y1, x2, y2 = shape.bbox
            lines.append(
                f"{CLASSES.index(shape.label)} {(x1 + x2) / 2 / width:.6f} "
                f"{(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} "
                f"{(y2 - y1) / height:.6f} {shape.confidence:.3f}"
            )
        (target / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
