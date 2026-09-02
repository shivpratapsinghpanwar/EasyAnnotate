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

    @property
    def has_polygon(self) -> bool:
        # Fewer than three points cannot enclose an area, and COCO rejects a
        # segmentation with under six numbers, so treat those as bbox-only.
        return self.polygon is not None and len(self.polygon) >= 3

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

    def clamped(self, width: int, height: int) -> "Shape":
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
        return {
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "shapes": [
                {
                    "label": s.label,
                    "bbox": list(s.bbox),
                    "polygon_points": len(s.polygon) if s.polygon else 0,
                }
                for s in self.shapes
            ],
        }
