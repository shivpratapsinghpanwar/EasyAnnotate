"""Turning a model's mask into a shape a person can edit.

This lives in the **base package**, not in the tools distribution, and that is
deliberate. A segmentation model returns a raster mask; what an annotation tool
needs is an outline with a manageable number of draggable vertices. Those are
different problems, and only the second one is ours.

The legacy SAM script did no simplification at all: it wrote every boundary
pixel as a polygon vertex, which is why its XML ran to hundreds of kilobytes
per image and why dragging a point in it was hopeless. A 512x512 mask has a
boundary thousands of pixels long. Nobody edits a two-thousand-vertex polygon.

So: contours, then Douglas-Peucker simplification, then a hard vertex cap, then
clamping to the image. Pure OpenCV and numpy -- no torch, nothing that only
exists when a sidecar is installed, and therefore testable without one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.annotations.model import Point, Shape

#: Simplification tolerance as a fraction of the contour's own perimeter.
#: Scale-free on purpose: a fixed pixel epsilon over-simplifies small objects
#: and under-simplifies large ones, and datasets contain both.
DEFAULT_EPSILON = 0.004

#: Above this, a polygon stops being editable and starts being a blob. The cap
#: is enforced by relaxing epsilon until it fits, so the shape degrades
#: smoothly instead of being truncated into nonsense.
MAX_VERTICES = 80

#: Below this many pixels a "mask" is noise, not an object.
MIN_AREA = 24


class MaskError(ValueError):
    """Raised when a mask cannot be turned into a usable outline."""


def _as_binary(mask: Any) -> np.ndarray:
    """Accept whatever a model handed back and make it a 0/1 uint8 array."""
    array = np.asarray(mask)
    if array.ndim == 3:
        # A single-channel mask with a trailing axis, or a batch of one.
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[0] == 1:
            array = array[0]
        else:
            raise MaskError(f"expected a 2D mask, got shape {array.shape}")
    if array.ndim != 2:
        raise MaskError(f"expected a 2D mask, got {array.ndim} dimensions")

    if array.dtype == bool:
        return array.astype(np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        # Models emit logits or probabilities; 0.5 is the conventional cut.
        return (array > 0.5).astype(np.uint8)
    return (array > 0).astype(np.uint8)


def mask_to_polygon(
    mask: Any,
    *,
    epsilon: float = DEFAULT_EPSILON,
    max_vertices: int = MAX_VERTICES,
    min_area: int = MIN_AREA,
) -> list[Point] | None:
    """The largest outline in ``mask``, simplified to something editable.

    Returns ``None`` when the mask holds nothing worth keeping, which is a
    normal answer -- a click on background produces an empty mask, and that
    should leave the canvas alone rather than raise.

    Only the largest contour is kept. A mask with several disconnected blobs
    almost always means the model caught something extra, and one shape per
    click is what the interaction promises; the alternative -- silently
    creating four shapes from one click -- is worse.
    """
    import cv2

    binary = _as_binary(mask)
    if not binary.any():
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    simplified = _simplify(largest, epsilon, max_vertices)
    points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
    # Fewer than three points cannot enclose an area, and every format
    # downstream would reject it.
    return points if len(points) >= 3 else None


def _simplify(contour: Any, epsilon: float, max_vertices: int) -> Any:
    """Douglas-Peucker, relaxed until the vertex cap is met.

    Doubling epsilon rather than truncating the point list matters: dropping
    the tail of a contour leaves a shape with a straight line where the object
    used to be, which looks like a bug. Relaxing keeps the outline closed and
    recognisable, just coarser.
    """
    import cv2

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return contour

    tolerance = epsilon
    simplified = cv2.approxPolyDP(contour, tolerance * perimeter, True)
    attempts = 0
    while len(simplified) > max_vertices and attempts < 12:
        tolerance *= 1.7
        simplified = cv2.approxPolyDP(contour, tolerance * perimeter, True)
        attempts += 1
    return simplified


def mask_to_shape(
    mask: Any,
    label: str,
    width: int,
    height: int,
    *,
    confidence: float | None = None,
    **options: Any,
) -> Shape | None:
    """A mask, as a clamped, labelled :class:`Shape` ready to store."""
    polygon = mask_to_polygon(mask, **options)
    if polygon is None:
        return None

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    shape = Shape(
        label=label,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        polygon=polygon,
        confidence=confidence,
    )
    # The same clamp the editor's own saves go through, so a mask that runs off
    # the edge of the image cannot produce coordinates no format can express.
    return shape.clamped(width, height) if width and height else shape


def simplify_polygon(
    points: list[Point],
    *,
    epsilon: float = DEFAULT_EPSILON,
    max_vertices: int = MAX_VERTICES,
) -> list[Point]:
    """Simplify an outline that is already a polygon.

    For predictions imported from a model that emitted polygons directly --
    those are just as capable of arriving with two thousand vertices as a mask
    is, and they land in exactly the same editor.
    """
    if len(points) <= 3:
        return list(points)


    contour = np.array([[list(p)] for p in points], dtype=np.int32)
    simplified = _simplify(contour, epsilon, max_vertices)
    out = [(int(p[0][0]), int(p[0][1])) for p in simplified]
    return out if len(out) >= 3 else list(points)
