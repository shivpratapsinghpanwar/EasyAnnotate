"""Flat CSV annotations -- one row per object.

Not a training format, a *working* format. A CSV opens in Excel, diffs in git,
and lets someone who does not write Python fix forty mislabelled boxes in two
minutes. Roboflow, TensorFlow's object-detection API and Keras-RetinaNet all
grew a CSV export for the same reason.

    filename,width,height,class,xmin,ymin,xmax,ymax,polygon

``polygon`` holds space-separated ``x y`` pairs and is blank for plain boxes,
so a detection-only dataset produces an unremarkable eight-column file.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..fs import write_text_atomic
from .model import Annotation, AnnotationError, KINDS, Point, Shape

HEADER = ("filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax", "polygon", "kind")

#: Header spellings other tools use, mapped onto ours.
ALIASES = {
    "image": "filename",
    "image_name": "filename",
    "image_id": "filename",
    "name": "filename",
    "label": "class",
    "class_name": "class",
    "category": "class",
    "img_width": "width",
    "img_height": "height",
    "x_min": "xmin",
    "y_min": "ymin",
    "x_max": "xmax",
    "y_max": "ymax",
    "x1": "xmin",
    "y1": "ymin",
    "x2": "xmax",
    "y2": "ymax",
    "segmentation": "polygon",
}


def write(annotations: Iterable[Annotation], path: Path) -> Path:
    """Write every object of every annotation as one row.

    Background images get a row with an empty class, so a round trip does not
    lose them -- an image absent from the CSV is indistinguishable from one
    that was never annotated.
    """
    # Built in memory and written in one atomic swap, so a crash part-way
    # through cannot leave a truncated CSV where a complete one used to be.
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(HEADER)
    for annotation in annotations:
        if annotation.is_background:
            writer.writerow(
                [annotation.filename, annotation.width, annotation.height, "", "", "", "", "", "", ""]
            )
            continue
        for shape in annotation.shapes:
            polygon = ""
            if shape.has_polygon or (shape.is_line and shape.polygon):
                assert shape.polygon is not None
                polygon = " ".join(f"{int(x)} {int(y)}" for x, y in shape.polygon)
            writer.writerow(
                [
                    annotation.filename,
                    annotation.width,
                    annotation.height,
                    shape.label,
                    *shape.bbox,
                    polygon,
                    shape.shape_kind,
                ]
            )
    # newline="" because csv already emitted its own \r\n terminators.
    return write_text_atomic(path, buffer.getvalue(), newline="")


def read(path: Path) -> list[Annotation]:
    """Read a CSV back into one :class:`Annotation` per distinct filename."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise AnnotationError(f"{path.name}: cannot read ({exc})") from exc

    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        raise AnnotationError(f"{path.name}: empty file")

    columns = {_canonical(name): name for name in reader.fieldnames}
    missing = {"filename", "class", "xmin", "ymin", "xmax", "ymax"} - set(columns)
    if missing - {"class"}:
        raise AnnotationError(
            f"{path.name}: missing column(s) {', '.join(sorted(missing))}. "
            f"Found: {', '.join(reader.fieldnames)}"
        )

    grouped: dict[str, Annotation] = {}
    for number, row in enumerate(reader, start=2):
        filename = (row.get(columns["filename"]) or "").strip()
        if not filename:
            continue
        annotation = grouped.get(filename)
        if annotation is None:
            annotation = Annotation(
                filename=filename,
                width=_int(row.get(columns.get("width", ""), "")),
                height=_int(row.get(columns.get("height", ""), "")),
                source_path=path,
            )
            grouped[filename] = annotation

        label = (row.get(columns.get("class", ""), "") or "").strip()
        if not label:
            continue  # a background row: the image entry above is the point
        try:
            bbox = (
                _int(row[columns["xmin"]]),
                _int(row[columns["ymin"]]),
                _int(row[columns["xmax"]]),
                _int(row[columns["ymax"]]),
            )
        except (KeyError, ValueError) as exc:
            raise AnnotationError(f"{path.name}:{number}: bad coordinates") from exc

        kind = (row.get(columns.get("kind", ""), "") or "").strip().lower()
        annotation.shapes.append(
            Shape(
                label=label,
                bbox=bbox,
                polygon=_parse_polygon(row.get(columns.get("polygon", ""), "")),
                kind=kind if kind in KINDS else "",
            )
        )

    for annotation in grouped.values():
        if annotation.width <= 0 or annotation.height <= 0:
            if annotation.shapes:
                annotation.width = max(s.bbox[2] for s in annotation.shapes) + 1
                annotation.height = max(s.bbox[3] for s in annotation.shapes) + 1

    return [grouped[k] for k in sorted(grouped)]


def _canonical(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return ALIASES.get(key, key)


def _int(value: str | None) -> int:
    if value is None or not str(value).strip():
        return 0
    return int(round(float(str(value).strip())))


def _parse_polygon(raw: str | None) -> list[Point] | None:
    if not raw or not raw.strip():
        return None
    numbers: Sequence[str] = raw.replace(",", " ").split()
    if len(numbers) < 6 or len(numbers) % 2:
        return None
    try:
        values = [float(n) for n in numbers]
    except ValueError:
        return None
    return [(int(round(values[i])), int(round(values[i + 1]))) for i in range(0, len(values), 2)]
