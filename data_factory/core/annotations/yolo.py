"""YOLO (Darknet / Ultralytics) reading and writing.

A YOLO label is one ``.txt`` per image, coordinates normalised to 0-1, and a
class *index* rather than a name -- so the class list is external state. That is
the whole difficulty of this format: read a label file without the right
``data.yaml`` and every class comes out wrong but nothing errors. We therefore
always carry the class list explicitly, and refuse to guess it.

Two line shapes are supported, which is how Ultralytics distinguishes tasks:

    detection      ``cls cx cy w h``
    segmentation   ``cls x1 y1 x2 y2 x3 y3 ...``  (>= 3 points)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..fs import write_text_atomic
from .model import Annotation, AnnotationError, Point, Shape

#: Where Ultralytics expects things. ``images/x.jpg`` <-> ``labels/x.txt``.
IMAGE_DIR = "images"
LABEL_DIR = "labels"
DATA_YAML = "data.yaml"


@dataclass
class ClassList:
    """The index-to-name table a YOLO dataset cannot be read without."""

    names: list[str] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {name: i for i, name in enumerate(self.names)}

    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> ClassList:
        """Build a stable list by sorting -- reproducible across runs."""
        return cls(sorted({label.strip() for label in labels if label.strip()}))

    def id_for(self, label: str) -> int:
        if label not in self._index:
            self._index[label] = len(self.names)
            self.names.append(label)
        return self._index[label]

    def name_for(self, index: int) -> str:
        if 0 <= index < len(self.names):
            return self.names[index]
        # Better a visibly wrong name than a silent misattribution to class 0.
        return f"class_{index}"

    def __len__(self) -> int:
        return len(self.names)


def read(
    path: Path,
    classes: ClassList,
    width: int,
    height: int,
    *,
    filename: str | None = None,
) -> Annotation:
    """Parse one ``.txt`` into an :class:`Annotation` in absolute pixels."""
    shapes: list[Shape] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnnotationError(f"{path.name}: cannot read ({exc})") from exc

    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except (ValueError, IndexError) as exc:
            raise AnnotationError(f"{path.name}:{number}: malformed line {line!r}") from exc

        label = classes.name_for(class_id)
        if len(values) == 4:
            shapes.append(_box_shape(label, values, width, height))
        elif len(values) == 5:
            # `cls cx cy w h conf` -- what `--save-conf` writes, and therefore
            # what comes back from a real inference run. Rejecting it meant the
            # one file people most want to load was the one that would not
            # parse. Five values are unambiguous: a polygon needs at least six
            # and an even number of them.
            shapes.append(_box_shape(label, values[:4], width, height, confidence=values[4]))
        elif len(values) >= 6 and len(values) % 2 == 0:
            shapes.append(_polygon_shape(label, values, width, height))
        elif len(values) >= 7 and len(values) % 2 == 1:
            # A segmentation line with a trailing confidence.
            shapes.append(
                _polygon_shape(label, values[:-1], width, height, confidence=values[-1])
            )
        else:
            raise AnnotationError(
                f"{path.name}:{number}: expected 4 box values (optionally plus a "
                f"confidence) or an even number of >= 6 polygon values, got {len(values)}"
            )

    return Annotation(
        filename=filename or path.with_suffix(".jpg").name,
        width=width,
        height=height,
        shapes=shapes,
        source_path=path,
    )


def _box_shape(
    label: str,
    values: Sequence[float],
    width: int,
    height: int,
    *,
    confidence: float | None = None,
) -> Shape:
    cx, cy, bw, bh = values
    half_w, half_h = bw * width / 2.0, bh * height / 2.0
    centre_x, centre_y = cx * width, cy * height
    return Shape(
        label=label,
        bbox=(
            int(round(centre_x - half_w)),
            int(round(centre_y - half_h)),
            int(round(centre_x + half_w)),
            int(round(centre_y + half_h)),
        ),
        confidence=confidence,
    )


def _polygon_shape(
    label: str,
    values: Sequence[float],
    width: int,
    height: int,
    *,
    confidence: float | None = None,
) -> Shape:
    polygon: list[Point] = [
        (int(round(values[i] * width)), int(round(values[i + 1] * height)))
        for i in range(0, len(values), 2)
    ]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return Shape(
        label=label,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        polygon=polygon,
        confidence=confidence,
        # Eight values is YOLO-OBB's line, a rotated box by its corners. A
        # genuine four-vertex segmentation reads the same way, and treating
        # it as a rotated box costs nothing: it is one.
        kind="obb" if len(polygon) == 4 else "",
    )


def write(
    annotation: Annotation,
    path: Path,
    classes: ClassList,
    *,
    segmentation: bool = True,
) -> Path:
    """Write one ``.txt``. Emits polygon lines when the shape has an outline.

    A background image legitimately produces an empty file -- that is how YOLO
    expresses a hard negative, and deleting it would silently drop the sample.
    """
    width = max(1, annotation.width)
    height = max(1, annotation.height)
    lines: list[str] = []

    for shape in annotation.shapes:
        class_id = classes.id_for(shape.label)
        if segmentation and shape.has_polygon:
            assert shape.polygon is not None
            coords = [
                f"{_unit(x / width):.6f} {_unit(y / height):.6f}" for x, y in shape.polygon
            ]
            lines.append(f"{class_id} " + " ".join(coords))
        else:
            xmin, ymin, xmax, ymax = shape.bbox
            cx = _unit(((xmin + xmax) / 2.0) / width)
            cy = _unit(((ymin + ymax) / 2.0) / height)
            bw = _unit((xmax - xmin) / width)
            bh = _unit((ymax - ymin) / height)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return write_text_atomic(path, "\n".join(lines) + ("\n" if lines else ""))


def _unit(value: float) -> float:
    """YOLO coordinates outside 0-1 are rejected by most trainers."""
    return max(0.0, min(1.0, value))


def write_data_yaml(
    path: Path,
    classes: ClassList,
    *,
    splits: dict[str, str] | None = None,
    root: str = ".",
) -> Path:
    """Write the ``data.yaml`` that makes a folder a *dataset* to Ultralytics.

    Hand-rolled rather than via PyYAML so the file keeps a readable, diffable
    layout -- these get committed to repositories and reviewed.
    """
    splits = splits or {"train": "images"}
    lines = [f"path: {root}"]
    for key in ("train", "val", "test"):
        if key in splits:
            lines.append(f"{key}: {splits[key]}")
    lines.append("")
    lines.append(f"nc: {len(classes)}")
    lines.append("names:")
    for index, name in enumerate(classes.names):
        lines.append(f"  {index}: {name}")
    return write_text_atomic(path, "\n".join(lines) + "\n")


def read_data_yaml(path: Path) -> tuple[ClassList, dict[str, str]]:
    """Read a ``data.yaml``, accepting both the list and dict ``names`` styles."""
    import yaml

    try:
        document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise AnnotationError(f"{path.name}: cannot parse ({exc})") from exc

    raw = document.get("names")
    if isinstance(raw, dict):
        # ``{0: cat, 1: dog}`` -- order by index, not by insertion.
        names = [str(raw[k]) for k in sorted(raw, key=lambda k: int(k))]
    elif isinstance(raw, list):
        names = [str(n) for n in raw]
    else:
        raise AnnotationError(f"{path.name}: no usable 'names' entry")

    splits = {k: str(document[k]) for k in ("train", "val", "test") if document.get(k)}
    return ClassList(names), splits


def find_data_yaml(folder: Path) -> Path | None:
    """Locate the class list for a folder, checking the usual parent layouts."""
    for candidate in (folder, folder.parent, folder.parent.parent):
        for name in (DATA_YAML, "dataset.yaml", "data.yml"):
            path = candidate / name
            if path.is_file():
                return path
    return None


def label_path_for(image: Path, root: Path) -> Path:
    """The ``.txt`` matching an image, honouring the images/labels convention.

    Falls back to a sidecar beside the image when the split layout is absent,
    because plenty of real YOLO exports in the wild are simply flat folders.
    """
    try:
        relative = image.relative_to(root)
    except ValueError:
        return image.with_suffix(".txt")
    parts = list(relative.parts)
    for index, part in enumerate(parts):
        if part == IMAGE_DIR:
            parts[index] = LABEL_DIR
            return root.joinpath(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")
