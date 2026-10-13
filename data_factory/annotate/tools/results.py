"""A tool's reply, read into shapes -- whichever form it chose.

Shared by the HTTP routes and the command line (``easyannotate predict``),
which is why it lives here and not in the service layer: ``annotate/`` must
not import FastAPI, and the CLI must not need it.
"""

from __future__ import annotations

from typing import Any

from . import postprocess


def shapes_from_result(result: Any, label: str, width: int, height: int) -> list[Any]:
    """Read a tool's reply into shapes, whichever form it used.

    A tool may answer with a mask, with polygons, or with boxes. All three are
    reasonable things for a model to produce, and none of them should require
    a different endpoint.
    """
    if not isinstance(result, dict):
        return []
    made = []
    name = label or "object"

    from ...core.annotations.model import AnnotationError, Shape

    # Ready-made shapes: the most natural form for a tool to answer in, and
    # what the reference sidecar uses. Parsed through Shape.from_dict because
    # a sidecar is a separate process -- possibly a third party's -- and its
    # output is no more trusted than a browser's.
    for raw in result.get("shapes") or []:
        if not isinstance(raw, dict):
            continue
        payload = dict(raw)
        if label:
            payload["label"] = label
        payload.setdefault("label", name)
        try:
            shape = Shape.from_dict(payload)
        except AnnotationError:
            continue
        if shape.polygon and len(shape.polygon) > 3:
            shape.polygon = postprocess.simplify_polygon(shape.polygon)
        made.append(shape.clamped(width, height) if width and height else shape)

    mask = result.get("mask")
    if mask is not None:
        shape = postprocess.mask_to_shape(
            mask, name, width, height, confidence=result.get("score")
        )
        if shape is not None:
            made.append(shape)

    for polygon in result.get("polygons") or []:
        points = [(int(x), int(y)) for x, y in polygon]
        if len(points) < 3:
            continue
        simplified = postprocess.simplify_polygon(points)
        xs = [p[0] for p in simplified]
        ys = [p[1] for p in simplified]
        made.append(
            Shape(
                label=name,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                polygon=simplified,
                confidence=result.get("score"),
            ).clamped(width, height)
        )

    for box in result.get("boxes") or []:
        try:
            x1, y1, x2, y2 = (int(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        made.append(
            Shape(label=name, bbox=(x1, y1, x2, y2), confidence=result.get("score")).clamped(
                width, height
            )
        )

    return made
