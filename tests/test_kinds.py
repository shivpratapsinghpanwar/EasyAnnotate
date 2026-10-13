"""Points, lines and rotated boxes: through the model, the store, every format.

A kind that survives the editor but not the export is a kind that lies.
"""

from __future__ import annotations

import json

import pytest

from conftest import make_image
from data_factory.annotate import images, projects, shapes
from data_factory.annotate import store as store_mod
from data_factory.annotate.store import Store
from data_factory.core.annotations import coco, csvfmt, dota, formats, labelme, yolo
from data_factory.core.annotations.model import Annotation, AnnotationError, Shape
from data_factory.core.types import Context

POINT = Shape(label="eye", bbox=(40, 30, 40, 30), kind="point")
LINE = Shape(label="seam", bbox=(10, 10, 90, 60), polygon=[(10, 10), (50, 60), (90, 20)], kind="line")
OBB = Shape(
    label="van",
    bbox=(20, 20, 80, 70),
    polygon=[(30, 20), (80, 35), (70, 70), (20, 55)],
    kind="obb",
)
BOX = Shape(label="door", bbox=(5, 5, 50, 90))


def _annotation(*shapes: Shape) -> Annotation:
    return Annotation(filename="a.jpg", width=120, height=90, shapes=list(shapes))


# ------------------------------------------------------------------- model


@pytest.mark.parametrize("shape", [POINT, LINE, OBB, BOX], ids=lambda s: s.shape_kind)
def test_every_kind_survives_the_wire(shape):
    again = Shape.from_dict(json.loads(json.dumps(shape.as_dict())))
    assert again == shape
    assert again.shape_kind == shape.shape_kind


def test_a_line_has_no_area_and_is_not_a_polygon():
    assert not LINE.has_polygon
    assert LINE.polygon_area() == 0
    assert OBB.has_polygon


def test_malformed_kinds_are_refused():
    with pytest.raises(AnnotationError):
        Shape.from_dict({"label": "x", "bbox": [0, 0, 1, 1], "kind": "blob"})
    with pytest.raises(AnnotationError):
        Shape.from_dict({"label": "x", "bbox": [0, 0, 1, 1], "kind": "obb", "polygon": [[0, 0], [1, 1], [2, 2]]})
    with pytest.raises(AnnotationError):
        Shape.from_dict({"label": "x", "bbox": [0, 0, 1, 1], "kind": "line", "polygon": [[0, 0]]})


# ------------------------------------------------------------------- formats


def test_labelme_keeps_points_and_lines(tmp_path):
    path = labelme.write(_annotation(POINT, LINE, OBB, BOX), tmp_path / "a.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert [s["shape_type"] for s in raw["shapes"]] == ["point", "linestrip", "rotation", "rectangle"]
    back = labelme.read(path)
    assert [s.shape_kind for s in back.shapes] == ["point", "line", "obb", "box"]
    assert back.shapes[2].polygon == OBB.polygon
    assert back.shapes[0].bbox == POINT.bbox
    assert back.shapes[1].polygon == LINE.polygon


def test_yolo_writes_a_rotated_box_as_obb_and_reads_it_back(tmp_path):
    classes = yolo.ClassList(["van", "door"])
    path = yolo.write(_annotation(OBB, BOX), tmp_path / "a.txt", classes)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines[0].split()) == 9  # class + four corners: YOLO-OBB's line
    assert len(lines[1].split()) == 5
    back = yolo.read(path, classes, 120, 90)
    assert back.shapes[0].shape_kind == "obb"
    assert back.shapes[0].polygon == OBB.polygon


def test_coco_writes_a_point_as_a_visible_keypoint():
    doc = coco.CocoBuilder()
    doc.add(_annotation(POINT))
    entry = doc.build()["annotations"][0]
    assert entry["keypoints"] == [40, 30, 2]
    assert entry["num_keypoints"] == 1
    assert entry["bbox"] == [40, 30, 1, 1]


def test_csv_carries_the_kind(tmp_path):
    path = csvfmt.write([_annotation(POINT, LINE, OBB, BOX)], tmp_path / "a.csv")
    back = csvfmt.read(path)[0]
    assert [s.shape_kind for s in back.shapes] == ["point", "line", "obb", "box"]
    assert back.shapes[1].polygon == LINE.polygon


def test_dota_round_trips_through_the_dataset_layer(tmp_path):
    source = tmp_path / "src"
    make_image(source / "a.jpg", 1, size=(120, 90))
    labelme.write(_annotation(OBB, BOX, POINT), source / "a.json")
    dataset = formats.read_dataset(source, "labelme")

    out = tmp_path / "dota"
    report = formats.write_dataset(dataset, out, "dota", ctx=Context())
    assert report["written"] == 1
    text = (out / "labels" / "train" / "a.txt").read_text(encoding="utf-8").splitlines()
    assert text[0] == "30 20 80 35 70 70 20 55 van 0"
    assert text[1] == "5 5 50 5 50 90 5 90 door 0"
    assert len(text) == 2  # a point has no rotated-box form

    assert formats.detect(out) == "dota"
    back = formats.read_dataset(out, "auto")
    assert back.source_format == "dota"
    kinds = {s.label: s.shape_kind for s in back.items[0].annotation.shapes}
    assert kinds == {"van": "obb", "door": "obb"}
    van = next(s for s in back.items[0].annotation.shapes if s.label == "van")
    assert van.polygon == OBB.polygon


# --------------------------------------------------------------------- store


def test_the_store_keeps_kinds(tmp_path):
    store = Store(tmp_path / "annotations.db")
    try:
        assert store.one("PRAGMA user_version")[0] == store_mod.SCHEMA_VERSION
        project = projects.create(store, "Kinds")
        image = images.add(store, project, "a.jpg", width=120, height=90)
        shapes.save(store, project, image, [(None, POINT), (None, LINE), (None, OBB), (None, BOX)])
        stored = shapes.listing(store, project, images.get(store, project, image.id))
        assert [s.shape.shape_kind for s in stored] == ["point", "line", "obb", "box"]
        assert stored[1].shape.polygon == LINE.polygon
        wire = [s.as_dict()["kind"] for s in stored]
        assert wire == ["point", "line", "obb", "box"]
    finally:
        store.close()
