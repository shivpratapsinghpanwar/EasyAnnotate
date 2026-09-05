"""LabelMe JSON reading and writing.

LabelMe sidecars are the lingua franca of hand-drawn polygons -- the SAM and
polygon annotators in this repository both descend from that lineage, and most
academic segmentation datasets ship in it. One ``.json`` per image, absolute
float coordinates, and a ``shape_type`` naming the geometry.

We read ``polygon``, ``rectangle``, ``linestrip`` and ``point``; anything else
(circles, masks) is reported rather than silently dropped, because a segmenter
losing half its objects to an unhandled shape type is the kind of failure that
only shows up as poor accuracy weeks later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..fs import write_text_atomic
from .model import Annotation, AnnotationError, Point, Shape

#: Shape types we can faithfully represent as a bbox and optional polygon.
SUPPORTED_SHAPES = ("polygon", "rectangle", "linestrip", "line", "point")


def read(
    path: Path, *, strict: bool = False, skipped_out: list[str] | None = None
) -> Annotation:
    """Parse a LabelMe ``.json``.

    Unhandled shape types are appended to ``skipped_out`` when one is given, so
    a converter can warn about them; :class:`Annotation` is a slots dataclass
    and deliberately has nowhere to stash out-of-band notes.
    """
    try:
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"{path.name}: cannot parse ({exc})") from exc

    width = int(document.get("imageWidth") or 0)
    height = int(document.get("imageHeight") or 0)
    filename = str(document.get("imagePath") or path.with_suffix(".jpg").name)
    # imagePath is often a relative path like "../images/x.jpg"; we only ever
    # want the basename, since pairing is by stem.
    filename = Path(filename.replace("\\", "/")).name

    shapes: list[Shape] = []
    skipped: list[str] = []
    for raw in document.get("shapes", []):
        shape = _read_shape(raw, path, skipped)
        if shape is not None:
            shapes.append(shape)

    if skipped and strict:
        raise AnnotationError(
            f"{path.name}: unsupported shape type(s): {', '.join(sorted(set(skipped)))}"
        )

    if (width <= 0 or height <= 0) and shapes:
        width = max(s.bbox[2] for s in shapes) + 1
        height = max(s.bbox[3] for s in shapes) + 1

    if skipped and skipped_out is not None:
        skipped_out.extend(sorted(set(skipped)))

    return Annotation(
        filename=filename,
        width=width,
        height=height,
        shapes=shapes,
        source_path=path,
    )


def _read_shape(raw: dict[str, Any], path: Path, skipped: list[str]) -> Shape | None:
    label = str(raw.get("label") or "").strip()
    kind = str(raw.get("shape_type") or "polygon").lower()
    points_raw = raw.get("points") or []
    if not label or not points_raw:
        return None
    if kind not in SUPPORTED_SHAPES:
        skipped.append(kind)
        return None

    try:
        points: list[Point] = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in points_raw]
    except (TypeError, ValueError, IndexError) as exc:
        raise AnnotationError(f"{path.name}: bad point list for {label!r}") from exc

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    # A rectangle is stored as its two opposite corners, so it has no outline;
    # a point has no area at all and only survives as a degenerate box.
    polygon = points if kind in ("polygon", "linestrip", "line") and len(points) >= 3 else None
    return Shape(label=label, bbox=bbox, polygon=polygon)


def write(annotation: Annotation, path: Path, *, image_path: str | None = None) -> Path:
    """Serialise to LabelMe JSON."""
    shapes: list[dict[str, Any]] = []
    for shape in annotation.shapes:
        if shape.has_polygon:
            assert shape.polygon is not None
            points = [[int(x), int(y)] for x, y in shape.polygon]
            kind = "polygon"
        else:
            xmin, ymin, xmax, ymax = shape.bbox
            points = [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]
            kind = "rectangle"
        shapes.append(
            {
                "label": shape.label,
                "points": points,
                "group_id": None,
                "shape_type": kind,
                "flags": {},
            }
        )

    document = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path or annotation.filename,
        # imageData is where LabelMe embeds a base64 copy of the whole image.
        # Omitting it keeps label files small and diffable; LabelMe reloads the
        # pixels from imagePath instead.
        "imageData": None,
        "imageHeight": annotation.height,
        "imageWidth": annotation.width,
    }
    write_text_atomic(path, json.dumps(document, indent=2))
    return path
