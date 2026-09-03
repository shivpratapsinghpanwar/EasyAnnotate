"""Draw annotations onto copies of the images, for eyeballing a dataset.

Merges three near-identical scripts. Colour is now derived from the class name
rather than fixed green, so a mislabelled object is visible at a glance instead
of requiring you to read every caption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..annotations import voc
from ..annotations.model import Annotation, AnnotationError
from ..pairs import PairIndex
from ..registry import Param, operation
from ..types import Action, Context, OpResult

#: Distinct, reasonably colour-blind-friendly BGR values (OpenCV order).
PALETTE = (
    (255, 176, 0), (0, 200, 255), (80, 220, 80), (200, 80, 220),
    (60, 60, 240), (240, 220, 0), (140, 140, 255), (0, 140, 255),
)

PARAMS = (
    Param("folder", "dir", "Folder of image/label pairs.", required=True, must_exist=True),
    Param("output", "dir", "Where annotated copies are written.", required=True),
    Param("draw", "choice", "Which shapes to render.",
          default="auto", choices=("auto", "bbox", "polygon", "both")),
    Param("split_by_type", "bool",
          "Route polygon-annotated and bbox-only images into separate sub-folders.",
          default=False),
    Param("thickness", "int", "Line thickness.", default=2, minimum=1, maximum=10),
    Param("font_scale", "float", "Caption size.", default=0.55, minimum=0.1, maximum=3.0),
    Param("show_labels", "bool", "Draw the class name next to each shape.", default=True),
    Param("limit", "int", "Stop after this many images (0 = all). Handy for a quick look.",
          default=0, minimum=0),
    Param("recursive", "bool", "Include sub-folders.", default=False),
    Param("label_ext", "str", "Sidecar label extension.", default=".xml"),
)


@operation(
    "verify.visualize",
    title="Render annotations onto images",
    category="verify",
    summary="Write a copy of each image with its boxes and polygons drawn on, colour-coded by class.",
    params=PARAMS,
    notes=(
        "Non-destructive: the source folder is never touched.\n\n"
        "draw=auto renders polygons where an object has one and falls back to its\n"
        "box otherwise, which is the honest view of a mixed dataset. Use\n"
        "split_by_type to see at a glance how much of the set has masks."
    ),
)
def visualize(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="verify.visualize", dry_run=ctx.dry_run)
    cv2, np = _load_cv2()

    folder: Path = params["folder"]
    output: Path = params["output"]
    mode: str = params["draw"]
    limit = int(params["limit"])

    index = PairIndex.scan(folder, recursive=params["recursive"], label_ext=params["label_ext"])
    pairs = index.complete()
    if limit:
        pairs = pairs[:limit]
    result.counts["scanned"] = len(index)

    for position, pair in enumerate(pairs, start=1):
        ctx.check_cancelled()
        ctx.emit(position, len(pairs), pair.stem)
        assert pair.image is not None and pair.label is not None

        try:
            annotation = voc.read(pair.label)
        except AnnotationError as exc:
            result.warnings.append(f"{pair.label.name}: {exc}")
            result.bump("unreadable_labels")
            continue

        image = cv2.imread(str(pair.image))
        if image is None:
            result.warnings.append(f"{pair.image.name}: OpenCV could not decode this image")
            result.bump("unreadable_images")
            continue

        drawn = _render(cv2, np, image, annotation, params)
        bucket = ""
        if params["split_by_type"]:
            bucket = "polygons" if annotation.has_polygons else "bbox_only"
            result.bump(bucket)

        destination = output / bucket / pair.image.name if bucket else output / pair.image.name
        if not ctx.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), drawn)
        result.record(Action.WRITE, pair.image, destination, f"{len(annotation.shapes)} shape(s)")
        result.bump("rendered")
        if annotation.is_background:
            result.bump("background_images")

    result.data["output"] = output
    result.data["mode"] = mode
    return result


def _render(cv2: Any, np: Any, image: Any, annotation: Annotation, params: dict[str, Any]) -> Any:
    canvas = image.copy()
    thickness = int(params["thickness"])
    scale = float(params["font_scale"])
    mode = params["draw"]

    for shape in annotation.shapes:
        colour = colour_for(shape.label)
        want_polygon = mode in ("polygon", "both") or (mode == "auto" and shape.has_polygon)
        want_bbox = mode in ("bbox", "both") or (mode == "auto" and not shape.has_polygon)

        if want_polygon and shape.has_polygon:
            assert shape.polygon is not None
            points = np.array(shape.polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [points], isClosed=True, color=colour, thickness=thickness)
        if want_bbox:
            xmin, ymin, xmax, ymax = shape.bbox
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), colour, thickness)

        if params["show_labels"]:
            _caption(cv2, canvas, shape.label, shape.bbox, colour, scale)

    if annotation.is_background:
        cv2.putText(
            canvas, "BACKGROUND", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 240), 2, cv2.LINE_AA
        )
    return canvas


def _caption(
    cv2: Any, canvas: Any, text: str, bbox: tuple[int, int, int, int], colour: Any, scale: float
) -> None:
    xmin, ymin = bbox[0], bbox[1]
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    # Flip the caption inside the box when the object sits against the top edge,
    # so it never gets clipped off the canvas.
    top = ymin - height - 6 if ymin - height - 6 > 0 else ymin + 2
    cv2.rectangle(canvas, (xmin, top), (xmin + width + 8, top + height + 6), colour, -1)
    cv2.putText(
        canvas, text, (xmin + 4, top + height + 1),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA,
    )


def colour_for(label: str) -> tuple[int, int, int]:
    """Stable colour per class name, so a class looks the same in every image."""
    return PALETTE[sum(ord(c) for c in label) % len(PALETTE)]


def _load_cv2() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"this operation needs opencv-python and numpy ({exc})") from exc
    return cv2, np
