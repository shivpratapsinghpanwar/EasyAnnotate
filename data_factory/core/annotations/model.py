"""A format-neutral annotation model.

Readers parse into these types and writers serialise out of them. Keeping a
neutral middle means adding YOLO or CreateML later costs one reader and one
writer, not a new converter for every existing pair of formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Point = tuple[int, int]

#: What a shape's geometry means. ``box`` and ``polygon`` are the originals;
#: ``line`` is an open polyline (vertices in ``polygon``, no area); ``point``
#: is one location (a degenerate ``bbox``); ``obb`` is a rotated box, stored as
#: its four corners in ``polygon`` -- which is exactly what YOLO-OBB and DOTA
#: write, so the export is a copy rather than a conversion.
KINDS = ("box", "polygon", "line", "point", "obb")

#: Version of the JSON shape/annotation form. Bump when a change would stop an
#: older build reading a newer payload; ``Annotation.from_dict`` refuses
#: anything newer than it understands rather than dropping fields silently.
WIRE_VERSION = 1


class AnnotationError(Exception):
    """Raised when a label file cannot be parsed or is structurally invalid."""


@dataclass(slots=True)
class Shape:
    """One annotated object.

    Every shape has a bounding box; a polygon is optional. A polygon-only
    source still yields a bbox because we derive it from the outline, which is
    what lets detection and segmentation datasets come out of the same file.
    """

    label: str
    bbox: tuple[int, int, int, int]
    polygon: list[Point] | None = None
    pose: str = "Unspecified"
    truncated: int = 0
    difficult: int = 0
    occluded: int = 0

    #: Detector score, when this shape came from a model rather than a person.
    #: ``None`` for anything hand-drawn -- which is how the review UI tells a
    #: prediction from a human label, and what a confidence filter reads.
    confidence: float | None = None

    #: One of :data:`KINDS`, or empty for the historical rule: a polygon if
    #: there is an outline of three or more points, else a box. Readers that
    #: predate kinds leave it empty and lose nothing.
    kind: str = ""

    @property
    def shape_kind(self) -> str:
        if self.kind:
            return self.kind
        return "polygon" if self.polygon is not None and len(self.polygon) >= 3 else "box"

    @property
    def has_polygon(self) -> bool:
        """Whether the outline encloses an area a mask could be made from.

        Fewer than three points cannot, and COCO rejects a segmentation with
        under six numbers, so those are bbox-only. A polyline has vertices but
        no area, so it is *not* a polygon here -- that keeps every
        segmentation writer from turning a lane marking into a sliver mask.
        """
        return (
            self.polygon is not None
            and len(self.polygon) >= 3
            and self.shape_kind in ("polygon", "obb")
        )

    @property
    def is_point(self) -> bool:
        return self.shape_kind == "point"

    @property
    def is_line(self) -> bool:
        return self.shape_kind == "line"

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def bbox_area(self) -> int:
        return self.width * self.height

    def polygon_area(self) -> float:
        """Shoelace area of the outline, falling back to the box."""
        if self.shape_kind in ("point", "line"):
            return 0.0
        if not self.has_polygon:
            return float(self.bbox_area)
        assert self.polygon is not None
        total = 0.0
        count = len(self.polygon)
        for i in range(count):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % count]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    def flat_polygon(self) -> list[int]:
        """``[x1, y1, x2, y2, ...]`` as COCO segmentation wants it."""
        if not self.has_polygon:
            return []
        assert self.polygon is not None
        flat: list[int] = []
        for x, y in self.polygon:
            flat.extend((int(x), int(y)))
        return flat

    def clamped(self, width: int, height: int) -> Shape:
        """Return a copy with all coordinates inside the image bounds."""

        def clamp_x(value: int) -> int:
            return max(0, min(int(value), max(0, width - 1)))

        def clamp_y(value: int) -> int:
            return max(0, min(int(value), max(0, height - 1)))

        xmin, ymin, xmax, ymax = self.bbox
        polygon = (
            [(clamp_x(x), clamp_y(y)) for x, y in self.polygon] if self.polygon else None
        )
        return Shape(
            label=self.label,
            bbox=(clamp_x(xmin), clamp_y(ymin), clamp_x(xmax), clamp_y(ymax)),
            polygon=polygon,
            pose=self.pose,
            truncated=self.truncated,
            difficult=self.difficult,
            occluded=self.occluded,
            confidence=self.confidence,
            kind=self.kind,
        )

    # -- wire format -------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """Lossless, JSON-safe form. ``from_dict`` reverses it exactly.

        This is what crosses the wire to the annotation editor and what is
        stored per shape. Every field round-trips, including the flags a
        detection format would drop -- losing ``difficult`` on a save because
        the browser never knew about it is the kind of quiet damage that only
        shows up in a training run weeks later.
        """
        return {
            "label": self.label,
            "bbox": list(self.bbox),
            "polygon": [[int(x), int(y)] for x, y in self.polygon] if self.polygon else None,
            "pose": self.pose,
            "truncated": self.truncated,
            "difficult": self.difficult,
            "occluded": self.occluded,
            "confidence": self.confidence,
            "kind": self.shape_kind,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Shape:
        """Rebuild a shape from its wire form, validating as we go.

        Raises :class:`AnnotationError` rather than letting a malformed
        payload become a half-built shape -- this parses untrusted input from
        a browser, so it fails loudly and specifically.
        """
        if not isinstance(raw, dict):
            raise AnnotationError(f"a shape must be an object, got {type(raw).__name__}")

        label = str(raw.get("label") or "").strip()
        if not label:
            raise AnnotationError("a shape needs a non-empty label")

        bbox = raw.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise AnnotationError(f"{label}: bbox must be four numbers, got {bbox!r}")
        try:
            xmin, ymin, xmax, ymax = (int(round(float(v))) for v in bbox)
        except (TypeError, ValueError) as exc:
            raise AnnotationError(f"{label}: bbox holds a non-number") from exc
        # Normalise rather than reject: a box dragged up-and-left is a normal
        # thing to do with a mouse, and the corners simply arrive swapped.
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin

        polygon = _points_from(raw.get("polygon"), label)
        kind = str(raw.get("kind") or "").strip().lower()
        if kind and kind not in KINDS:
            raise AnnotationError(f"{label}: unknown shape kind {kind!r}; known: {', '.join(KINDS)}")
        if kind == "line" and (polygon is None or len(polygon) < 2):
            raise AnnotationError(f"{label}: a line needs at least two points")
        if kind == "obb" and (polygon is None or len(polygon) != 4):
            raise AnnotationError(f"{label}: a rotated box needs exactly four corners")
        if kind == "point":
            polygon = None
        # The two historical kinds are inferred from the geometry, so they are
        # stored as empty: a shape must round-trip to *equal* itself.
        if kind in ("box", "polygon"):
            kind = ""
        confidence = raw.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise AnnotationError(f"{label}: confidence must be a number") from exc

        return cls(
            label=label,
            bbox=(xmin, ymin, xmax, ymax),
            polygon=polygon,
            pose=str(raw.get("pose") or "Unspecified"),
            truncated=_flag(raw.get("truncated")),
            difficult=_flag(raw.get("difficult")),
            occluded=_flag(raw.get("occluded")),
            confidence=confidence,
            kind=kind,
        )


@dataclass(slots=True)
class Annotation:
    """Everything one label file says about one image."""

    filename: str
    width: int
    height: int
    depth: int = 3
    shapes: list[Shape] = field(default_factory=list)
    folder: str = "images"
    source_path: Path | None = None

    @property
    def is_background(self) -> bool:
        """A deliberate hard negative: valid file, zero objects."""
        return not self.shapes

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.shapes]

    @property
    def label_set(self) -> set[str]:
        return {s.label for s in self.shapes}

    @property
    def has_polygons(self) -> bool:
        return any(s.has_polygon for s in self.shapes)

    def as_dict(self) -> dict[str, Any]:
        """Lossless, JSON-safe form.

        The previous version of this method reported ``polygon_points`` as a
        *count*, which made it useless for anything but a glance -- you could
        not reconstruct the annotation from it. It had no callers. This one
        round-trips.
        """
        return {
            "wire": WIRE_VERSION,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "folder": self.folder,
            "shapes": [s.as_dict() for s in self.shapes],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Annotation:
        if not isinstance(raw, dict):
            raise AnnotationError(f"an annotation must be an object, got {type(raw).__name__}")
        version = int(raw.get("wire", WIRE_VERSION) or WIRE_VERSION)
        if version > WIRE_VERSION:
            raise AnnotationError(
                f"this annotation is wire version {version}; this build understands "
                f"up to {WIRE_VERSION}. Upgrade Data Factory to read it."
            )
        shapes = raw.get("shapes") or []
        if not isinstance(shapes, list):
            raise AnnotationError("shapes must be a list")
        return cls(
            filename=str(raw.get("filename") or ""),
            width=int(raw.get("width") or 0),
            height=int(raw.get("height") or 0),
            depth=int(raw.get("depth") or 3),
            shapes=[Shape.from_dict(s) for s in shapes],
            folder=str(raw.get("folder") or "images"),
        )


def _flag(value: Any) -> int:
    """VOC's 0/1 attributes, tolerant of the bools a browser will send."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return 1 if int(value) else 0
    except (TypeError, ValueError):
        return 0


def _points_from(raw: Any, label: str) -> list[Point] | None:
    """Parse ``[[x, y], ...]``, rejecting anything that is not a point list."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise AnnotationError(f"{label}: polygon must be a list of points")
    points: list[Point] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise AnnotationError(f"{label}: each polygon point must be [x, y], got {pair!r}")
        try:
            points.append((int(round(float(pair[0]))), int(round(float(pair[1])))))
        except (TypeError, ValueError) as exc:
            raise AnnotationError(f"{label}: polygon holds a non-number") from exc
    # An empty list and a one-point list both mean "no outline"; normalising to
    # None here keeps `has_polygon` the single place that defines the rule.
    return points or None
