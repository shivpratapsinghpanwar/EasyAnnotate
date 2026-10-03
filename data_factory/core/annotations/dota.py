"""DOTA: rotated boxes as four corners, one ``.txt`` per image.

The aerial-imagery format, and what every rotated-box detector reads::

    x1 y1 x2 y2 x3 y3 x4 y4 category difficult

Absolute pixels, corners in order around the box, class by *name* -- so unlike
YOLO there is no class list to lose. Files may start with ``imagesource:`` and
``gsd:`` header lines, which are kept out of the way rather than rejected.

A rotated box is stored in the neutral model as a four-point polygon with
``kind="obb"``; its axis-aligned ``bbox`` is derived, which is what lets a
rotated dataset still come out as plain YOLO or VOC boxes when asked.
"""

from __future__ import annotations

from pathlib import Path

from ..fs import write_text_atomic
from .model import Annotation, AnnotationError, Point, Shape

HEADER_PREFIXES = ("imagesource:", "gsd:")


def read(path: Path, width: int, height: int, *, filename: str | None = None) -> Annotation:
    shapes: list[Shape] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnnotationError(f"{path.name}: cannot read ({exc})") from exc

    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith(HEADER_PREFIXES):
            continue
        parts = line.split()
        if len(parts) < 9:
            raise AnnotationError(
                f"{path.name}:{number}: expected 8 coordinates and a class, got {len(parts)} fields"
            )
        try:
            values = [float(v) for v in parts[:8]]
        except ValueError as exc:
            raise AnnotationError(f"{path.name}:{number}: malformed coordinates") from exc
        label = parts[8]
        difficult = 0
        if len(parts) > 9:
            try:
                difficult = 1 if int(float(parts[9])) else 0
            except ValueError:
                difficult = 0
        shapes.append(corners_to_shape(label, values, difficult=difficult))

    return Annotation(
        filename=filename or path.with_suffix(".jpg").name,
        width=width,
        height=height,
        shapes=shapes,
        source_path=path,
    )


def corners_to_shape(label: str, values: list[float], *, difficult: int = 0) -> Shape:
    corners: list[Point] = [
        (int(round(values[i])), int(round(values[i + 1]))) for i in range(0, 8, 2)
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return Shape(
        label=label,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        polygon=corners,
        difficult=difficult,
        kind="obb",
    )


def corners_of(shape: Shape) -> list[Point]:
    """Four corners for any shape: its own if it is a rotated box, else the
    corners of its axis-aligned box. A polygon that is not a rectangle is
    written as its bounding box -- the honest rotated-box approximation is a
    minimum-area rectangle, and that is a transformation the user should ask
    for, not one a writer performs silently."""
    if shape.shape_kind == "obb" and shape.polygon and len(shape.polygon) == 4:
        return [(int(x), int(y)) for x, y in shape.polygon]
    x1, y1, x2, y2 = shape.bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def write(annotation: Annotation, path: Path) -> Path:
    lines = []
    for shape in annotation.shapes:
        if shape.shape_kind in ("point", "line"):
            continue  # nothing rotated-box shaped to say about these
        coords = " ".join(f"{x} {y}" for x, y in corners_of(shape))
        lines.append(f"{coords} {shape.label.replace(' ', '-')} {int(bool(shape.difficult))}")
    return write_text_atomic(path, "\n".join(lines) + ("\n" if lines else ""))
