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
from xml.dom import minidom

from ..fs import write_text_atomic
from .model import Annotation, AnnotationError, Point, Shape


def read(path: Path) -> Annotation:
    """Parse a Pascal VOC XML file into an :class:`Annotation`."""
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
    """Serialise to Pascal VOC XML, emitting the indexed polygon dialect."""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = annotation.folder
    ET.SubElement(root, "filename").text = annotation.filename
    ET.SubElement(root, "path").text = annotation.filename

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(annotation.width)
    ET.SubElement(size, "height").text = str(annotation.height)
    ET.SubElement(size, "depth").text = str(annotation.depth)

    ET.SubElement(root, "segmented").text = "1" if annotation.has_polygons else "0"

    for shape in annotation.shapes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = shape.label
        ET.SubElement(obj, "pose").text = shape.pose
        ET.SubElement(obj, "truncated").text = str(shape.truncated)
        ET.SubElement(obj, "difficult").text = str(shape.difficult)
        ET.SubElement(obj, "occluded").text = str(shape.occluded)
        # Written only when there is one, so a hand-drawn label produces
        # byte-identical XML to before -- a <score>1.0</score> appearing on
        # every ground-truth object would be a lie about where it came from.
        if shape.confidence is not None:
            ET.SubElement(obj, "score").text = f"{float(shape.confidence):.6f}"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(shape.bbox[0])
        ET.SubElement(bndbox, "ymin").text = str(shape.bbox[1])
        ET.SubElement(bndbox, "xmax").text = str(shape.bbox[2])
        ET.SubElement(bndbox, "ymax").text = str(shape.bbox[3])

        if shape.has_polygon:
            assert shape.polygon is not None
            polygon = ET.SubElement(obj, "polygon")
            for index, (x, y) in enumerate(shape.polygon, start=1):
                ET.SubElement(polygon, f"x{index}").text = str(int(x))
                ET.SubElement(polygon, f"y{index}").text = str(int(y))

    pretty = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    # Atomic: a half-written XML is an unparseable label, and the original
    # is already gone by the time the write fails.
    return write_text_atomic(path, pretty)


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
