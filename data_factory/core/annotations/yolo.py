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


def read_label_text(path: Path) -> str:
    """The text of a label file, without the byte order mark some editors add.

    Notepad and PowerShell both put one at the start of a UTF-8 file. Left in,
    it glues itself to the first number on the first line, and the whole file
    is thrown away as malformed over three bytes nobody can see.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise AnnotationError(f"{path.name}: cannot read ({exc})") from exc
    except UnicodeDecodeError as exc:
        raise AnnotationError(f"{path.name}: not UTF-8 text ({exc})") from exc


def read(
    path: Path,
    classes: ClassList,
    width: int,
    height: int,
    *,
    filename: str | None = None,
    unnamed_out: list[int] | None = None,
) -> Annotation:
    """Parse one ``.txt`` into an :class:`Annotation` in absolute pixels.

    ``unnamed_out`` receives the class number of every object whose number
    the class list has no name for, so a caller can keep those numbers in
    place and say how many objects were affected.
    """
    shapes: list[Shape] = []
    text = read_label_text(path)

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
        if unnamed_out is not None and not 0 <= class_id < len(classes):
            unnamed_out.append(class_id)
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
            box = visible_box(shape.bbox, width, height)
            if box is None:
                continue
            xmin, ymin, xmax, ymax = box
            cx = ((xmin + xmax) / 2.0) / width
            cy = ((ymin + ymax) / 2.0) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # newline="\n": the same bytes on every platform, not CRLF on Windows.
    return write_text_atomic(path, "\n".join(lines) + ("\n" if lines else ""), newline="\n")


def _unit(value: float) -> float:
    """YOLO coordinates outside 0-1 are rejected by most trainers."""
    return max(0.0, min(1.0, value))


def visible_box(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """The part of a box that lies inside the image, or ``None`` if no part
    with any area does.

    The box is cut to the image first and converted after. Clamping the
    centre and the size separately, which this replaces, turned a box from
    -50 to 50 on a 100-wide image into one centred on the left edge and as
    wide as the whole image.
    """
    xmin, ymin, xmax, ymax = bbox
    left, right = max(0, min(xmin, xmax)), min(max(1, width), max(xmin, xmax))
    top, bottom = max(0, min(ymin, ymax)), min(max(1, height), max(ymin, ymax))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def fate(shape: Shape, width: int, height: int, *, segmentation: bool = True) -> str:
    """What :func:`write` does to a shape that does not fit: ``""`` when it is
    written as it stands, else ``"clipped"``, ``"dropped"`` or ``"outline"``
    (an outline with points moved onto the image edge).

    Uses the same geometry as the writer, so what a conversion reports and
    what it writes cannot drift apart -- and a dry run can report it too.
    """
    if segmentation and shape.has_polygon:
        assert shape.polygon is not None
        inside = all(0 <= x <= max(1, width) and 0 <= y <= max(1, height) for x, y in shape.polygon)
        return "" if inside else "outline"
    box = visible_box(shape.bbox, width, height)
    if box is None:
        return "dropped"
    return "" if box == tuple(shape.bbox) else "clipped"


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
        lines.append(f"  {index}: {_yaml_name(name)}")
    return write_text_atomic(path, "\n".join(lines) + "\n", newline="\n")


#: Words a YAML loader may read as a boolean or as nothing at all. PyYAML is
#: asked directly below; "y" and "n" are here for the loaders that follow the
#: YAML 1.1 list in full.
_YAML_WORDS = frozenset({"y", "n", "yes", "no", "on", "off", "true", "false", "null", "~"})


def _yaml_name(name: str) -> str:
    """A class name as YAML: bare when that is safe, quoted when it is not.

    Trainers load data.yaml with an ordinary YAML loader, which decides for
    itself what a bare word is: ``no`` is False, ``010`` is 8, ``1.50`` is
    1.5, ``~`` is nothing, and a name holding ``: `` or starting with ``#``
    is not a name at all. So a name is written bare only when it reads back
    as exactly the same text; otherwise it goes in double quotes, which every
    loader reads as text. Ordinary names stay bare and the file stays plain.
    """
    import yaml

    safe = name.strip() == name and name.lower() not in _YAML_WORDS and not _is_number(name)
    if safe:
        try:
            safe = yaml.safe_load(f"k: {name}\n") == {"k": name}
        except yaml.YAMLError:
            safe = False
    if safe:
        return name
    quoted = yaml.safe_dump(name, default_style='"', allow_unicode=True, width=float("inf"))
    return quoted.rstrip("\n")


def _is_number(text: str) -> bool:
    """Whether some YAML dialect could read this as a number (12, 1e3, 0o17)."""
    for parse in (float, lambda value: int(value, 0)):
        try:
            parse(text)
            return True
        except ValueError:
            continue
    return False


def read_data_yaml(path: Path) -> tuple[ClassList, dict[str, str]]:
    """Read a ``data.yaml``, accepting both the list and dict ``names`` styles.

    Every value is read as text. A typed load renames classes -- ``no``
    becomes False, ``010`` becomes 8, ``1.50`` becomes 1.5 -- and a class that
    comes back under another name is a different class to everything after.
    """
    import yaml

    try:
        # BaseLoader builds strings, lists and mappings, and nothing else.
        document: Any = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=yaml.BaseLoader)
    except Exception as exc:
        raise AnnotationError(f"{path.name}: cannot parse ({exc})") from exc
    if not isinstance(document, dict):
        raise AnnotationError(f"{path.name}: no usable 'names' entry")

    raw = document.get("names")
    if isinstance(raw, dict):
        # ``{0: cat, 1: dog}`` -- order by index, not by insertion.
        try:
            names = [str(raw[k]) for k in sorted(raw, key=lambda k: int(k))]
        except (TypeError, ValueError) as exc:
            raise AnnotationError(
                f"{path.name}: 'names' must be keyed by class number ({exc})"
            ) from exc
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


#: The other common place for YOLO class names: one name per line, the line
#: number being the class number. LabelImg and Label Studio both write it.
CLASSES_TXT = "classes.txt"


def find_classes_txt(folder: Path) -> Path | None:
    """Locate a ``classes.txt``: in the folder, in its ``labels/``, or failing
    that the first one anywhere beneath it (ordered by path, so the choice
    does not depend on the machine)."""
    for candidate in (folder / CLASSES_TXT, folder / LABEL_DIR / CLASSES_TXT):
        if candidate.is_file():
            return candidate
    found = [p for p in folder.rglob(CLASSES_TXT) if p.is_file()]
    return min(found, key=lambda p: p.relative_to(folder).parts) if found else None


def read_classes_txt(path: Path) -> ClassList:
    """Read a ``classes.txt``. A blank line in the middle still holds its
    number, under a placeholder name, so the names after it do not shift."""
    lines = [line.strip() for line in read_label_text(path).splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return ClassList([name or f"class_{index}" for index, name in enumerate(lines)])


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
