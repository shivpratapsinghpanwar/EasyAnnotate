"""Pascal VOC reading and writing.

The repository grew two incompatible polygon dialects: the polygon annotator
writes ``<polygon><x1>..<y1>..</polygon>`` while the SAM tool writes
``<polygon><pt><x>..<y>..</pt></polygon>``. Only the first was ever readable by
the COCO converter, so SAM-produced masks silently converted to empty
segmentations. This reader accepts both -- plus plain ``<bndbox>`` -- and the
writer emits the indexed dialect as the canonical form.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..fs import write_text_atomic
from .model import Annotation, AnnotationError, Point, Shape


def read(path: Path, *, dropped_out: list[str] | None = None) -> Annotation:
    """Parse a Pascal VOC XML file into an :class:`Annotation`.

    An object with no name cannot be kept -- there is nothing to call it --
    and is noted in ``dropped_out``, one entry each, so a caller can count
    them instead of losing them without a word.
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise AnnotationError(f"{path.name}: malformed XML ({exc})") from exc
    except OSError as exc:
        raise AnnotationError(f"{path.name}: cannot read ({exc})") from exc

    root = tree.getroot()
    width, height, depth = _read_size(root)
    filename = _text(root, "filename") or path.with_suffix(".jpg").name

    shapes: list[Shape] = []
    for obj in root.findall("object"):
        shape = _read_object(obj, path)
        if shape is not None:
            shapes.append(shape)
        elif dropped_out is not None:
            dropped_out.append("no name")

    # Some tools omit <size>; recover it from the shapes so downstream COCO
    # export is not left with a zero-dimension image.
    if (width <= 0 or height <= 0) and shapes:
        width = max(s.bbox[2] for s in shapes) + 1
        height = max(s.bbox[3] for s in shapes) + 1

    return Annotation(
        filename=filename,
        width=width,
        height=height,
        depth=depth,
        shapes=shapes,
        folder=_text(root, "folder") or "images",
        source_path=path,
    )


def _read_size(root: ET.Element) -> tuple[int, int, int]:
    size = root.find("size")
    if size is None:
        return 0, 0, 3
    return (
        _int(_text(size, "width")),
        _int(_text(size, "height")),
        _int(_text(size, "depth")) or 3,
    )


def _read_object(obj: ET.Element, path: Path) -> Shape | None:
    label = _text(obj, "name")
    if not label:
        return None

    polygon = _read_polygon(obj)
    bbox = _read_bbox(obj)

    if bbox is None and polygon:
        # Polygon-only object (the SAM tool's output shape): derive the box so
        # every consumer gets a bbox regardless of how it was annotated.
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = (min(xs), min(ys), max(xs), max(ys))
    if bbox is None:
        raise AnnotationError(f"{path.name}: object {label!r} has neither bndbox nor polygon")

    xmin, ymin, xmax, ymax = bbox
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    return Shape(
        label=label.strip(),
        bbox=(xmin, ymin, xmax, ymax),
        polygon=polygon,
        pose=_text(obj, "pose") or "Unspecified",
        truncated=_int(_text(obj, "truncated")),
        difficult=_int(_text(obj, "difficult")),
        occluded=_int(_text(obj, "occluded")),
        confidence=_score(obj),
    )


def _read_bbox(obj: ET.Element) -> tuple[int, int, int, int] | None:
    node = obj.find("bndbox")
    if node is None:
        return None
    try:
        return (
            _int(_text(node, "xmin")),
            _int(_text(node, "ymin")),
            _int(_text(node, "xmax")),
            _int(_text(node, "ymax")),
        )
    except (TypeError, ValueError):
        return None


def _read_polygon(obj: ET.Element) -> list[Point] | None:
    """Read either polygon dialect, whichever this file happens to use."""
    node = obj.find("polygon")
    if node is None:
        return None

    # Dialect A -- indexed children: <x1>,<y1>,<x2>,<y2>,...
    points: list[Point] = []
    index = 1
    while True:
        x_node = node.find(f"x{index}")
        y_node = node.find(f"y{index}")
        if x_node is None or y_node is None:
            break
        points.append((_int(x_node.text), _int(y_node.text)))
        index += 1
    if points:
        return points

    # Dialect B -- repeated <pt><x/><y/></pt> elements (the SAM tool).
    for pt in node.findall("pt"):
        x_node, y_node = pt.find("x"), pt.find("y")
        if x_node is None or y_node is None:
            continue
        points.append((_int(x_node.text), _int(y_node.text)))
    return points or None


def write(annotation: Annotation, path: Path) -> Path:
    """Serialise to Pascal VOC XML, emitting the indexed polygon dialect.

    The XML is written here, line by line, rather than by a library. It used
    to come from minidom, whose output is not the same on every Python: a
    double quote in a class name is ``&quot;`` on 3.12 and a plain quote on
    3.13. The rule is now fixed: two-space indent, ``&``, ``<`` and ``>``
    escaped in text and nothing else, LF line endings on every platform.
    For everything but that quote, it is the layout minidom produced.
    """
    lines = ['<?xml version="1.0" ?>', "<annotation>"]
    lines.append(_leaf("folder", annotation.folder, 1))
    lines.append(_leaf("filename", annotation.filename, 1))
    lines.append(_leaf("path", annotation.filename, 1))
    lines += ["  <source>", _leaf("database", "Unknown", 2), "  </source>"]
    lines += [
        "  <size>",
        _leaf("width", annotation.width, 2),
        _leaf("height", annotation.height, 2),
        _leaf("depth", annotation.depth, 2),
        "  </size>",
    ]
    lines.append(_leaf("segmented", "1" if annotation.has_polygons else "0", 1))

    for shape in annotation.shapes:
        lines.append("  <object>")
        lines.append(_leaf("name", shape.label, 2))
        lines.append(_leaf("pose", shape.pose, 2))
        lines.append(_leaf("truncated", shape.truncated, 2))
        lines.append(_leaf("difficult", shape.difficult, 2))
        lines.append(_leaf("occluded", shape.occluded, 2))
        # Written only when there is one, so a hand-drawn label produces
        # byte-identical XML to before -- a <score>1.0</score> appearing on
        # every ground-truth object would be a lie about where it came from.
        if shape.confidence is not None:
            lines.append(_leaf("score", f"{float(shape.confidence):.6f}", 2))

        lines.append("    <bndbox>")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), shape.bbox):
            lines.append(_leaf(tag, value, 3))
        lines.append("    </bndbox>")

        if shape.has_polygon:
            assert shape.polygon is not None
            lines.append("    <polygon>")
            for index, (x, y) in enumerate(shape.polygon, start=1):
                lines.append(_leaf(f"x{index}", int(x), 3))
                lines.append(_leaf(f"y{index}", int(y), 3))
            lines.append("    </polygon>")
        lines.append("  </object>")

    lines.append("</annotation>")
    # Atomic: a half-written XML is an unparseable label, and the original
    # is already gone by the time the write fails.
    return write_text_atomic(path, "\n".join(lines) + "\n", newline="\n")


def _leaf(tag: str, value: object, depth: int) -> str:
    """One ``<tag>text</tag>`` line. An element with no text is ``<tag/>``."""
    text = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    pad = "  " * depth
    return f"{pad}<{tag}>{text}</{tag}>" if text else f"{pad}<{tag}/>"


def write_background(filename: str, width: int, height: int, path: Path) -> Path:
    """Write a valid, object-free annotation -- an explicit hard negative."""
    return write(Annotation(filename=filename, width=width, height=height), path)


def _text(parent: ET.Element, tag: str) -> str | None:
    node = parent.find(tag)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _int(value: str | None) -> int:
    """Parse a coordinate, tolerating the decimals some tools emit."""
    if value is None:
        return 0
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return 0


def _score(obj: ET.Element) -> float | None:
    """A detector's confidence, if this object carries one.

    Not part of the Pascal VOC spec, which predates anyone wanting to round
    trip model output. Reading it is harmless -- files without it are
    unaffected -- and it is what lets predictions survive a trip through the
    format the rest of this tool already speaks.
    """
    raw = _text(obj, "score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
