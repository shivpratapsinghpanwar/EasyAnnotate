"""Converter defects found by comparing output byte for byte with a second
implementation, each pinned by the test that reproduces it.

The shape of most tests here is the converter's own failure mode: nothing
crashes, the output parses, and it means something other than the input did.
So they convert, read back, and compare -- and where something cannot be kept,
they assert the run *said so*, with a count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import IMAGE_SIZE, make_image

import data_factory as df
from data_factory.annotate import review
from data_factory.core.annotations import coco as coco_fmt
from data_factory.core.annotations import dota, formats, labelme, voc, yolo
from data_factory.core.annotations.model import AnnotationError
from data_factory.core.types import Context

W, H = IMAGE_SIZE
BOM = b"\xef\xbb\xbf"


def run(name: str, params: dict, ctx: Context | None = None):
    return df.get_operation(name).run(params, ctx or Context())


def coco_document(images: int = 1) -> dict:
    return {
        "images": [
            {"id": i, "file_name": f"img{i:03d}.jpg", "width": W, "height": H}
            for i in range(1, images + 1)
        ],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 50, 60]}
            for i in range(1, images + 1)
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }


# ------------------------------------------------- byte order marks, bad JSON


class TestByteOrderMark:
    """Notepad and PowerShell both put a byte order mark at the start of a
    UTF-8 file. It is not part of the label, and every reader must agree."""

    def test_yolo_label_with_a_bom_is_read(self, tmp_path: Path):
        path = tmp_path / "a.txt"
        path.write_bytes(BOM + b"0 0.5 0.5 0.2 0.2\n")
        annotation = yolo.read(path, yolo.ClassList(["cat"]), W, H)
        assert [s.label for s in annotation.shapes] == ["cat"]

    def test_dota_label_with_a_bom_is_read(self, tmp_path: Path):
        path = tmp_path / "a.txt"
        path.write_bytes(BOM + b"10 10 50 10 50 40 10 40 plane 0\n")
        annotation = dota.read(path, W, H)
        assert [s.label for s in annotation.shapes] == ["plane"]

    def test_dota_folder_with_a_bom_before_its_header_is_still_dota(self, tmp_path: Path):
        make_image(tmp_path / "a.jpg", 1)
        (tmp_path / "a.txt").write_bytes(
            BOM + b"imagesource:GoogleEarth\ngsd:0.1\n10 10 50 10 50 40 10 40 plane 0\n"
        )
        assert formats.detect(tmp_path) == "dota"

    def test_labelme_file_with_a_bom_is_read(self, tmp_path: Path):
        path = tmp_path / "a.json"
        document = {
            "shapes": [{"label": "cat", "points": [[1, 2], [30, 40]], "shape_type": "rectangle"}],
            "imagePath": "a.jpg", "imageWidth": W, "imageHeight": H,
        }
        path.write_bytes(BOM + json.dumps(document).encode("utf-8"))
        assert labelme.read(path).shapes[0].bbox == (1, 2, 30, 40)

    def test_coco_file_with_a_bom_is_read(self, tmp_path: Path):
        path = tmp_path / "instances_train.json"
        path.write_bytes(BOM + json.dumps(coco_document()).encode("utf-8"))
        assert coco_fmt.load(path)["categories"][0]["name"] == "cat"

        dataset = formats.read_dataset(tmp_path, "auto")
        assert dataset.source_format == "coco"
        assert dataset.stats()["objects"] == 1

    def test_data_yaml_with_a_bom_is_read(self, tmp_path: Path):
        path = tmp_path / "data.yaml"
        path.write_bytes(BOM + b"names:\n  0: cat\n  1: dog\n")
        classes, _ = yolo.read_data_yaml(path)
        assert classes.names == ["cat", "dog"]

    def test_voc_file_with_a_bom_is_read(self, tmp_path: Path):
        path = tmp_path / "a.xml"
        path.write_bytes(
            BOM + b"<annotation><filename>a.jpg</filename><object><name>cat</name>"
            b"<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>"
            b"</object></annotation>"
        )
        assert voc.read(path).shapes[0].bbox == (1, 2, 30, 40)

    def test_predictions_json_with_a_bom_is_read(self, tmp_path: Path):
        path = tmp_path / "results.json"
        results = [{"image_id": 1, "category_id": 0, "bbox": [1, 2, 3, 4], "score": 0.9}]
        path.write_bytes(BOM + json.dumps(results).encode("utf-8"))
        annotations, _ = review.read_predictions(path)
        assert annotations[0].shapes[0].confidence == 0.9

    def test_a_label_file_that_is_not_utf8_is_a_named_error(self, tmp_path: Path):
        """PowerShell's `>` writes UTF-16. That is a skipped file with a
        reason, not a UnicodeDecodeError out of the middle of a conversion."""
        path = tmp_path / "a.txt"
        path.write_bytes("0 0.5 0.5 0.2 0.2\n".encode("utf-16"))
        with pytest.raises(AnnotationError, match="a.txt"):
            yolo.read(path, yolo.ClassList(["cat"]), W, H)


class TestMalformedJson:
    """A broken file is a normal thing to be handed. It gets a named, one-line
    error -- never a traceback, and never a silent empty dataset."""

    def test_malformed_coco_raises_a_named_error(self, tmp_path: Path):
        path = tmp_path / "instances_train.json"
        path.write_text('{"images": [', encoding="utf-8")
        with pytest.raises(AnnotationError, match="instances_train.json"):
            coco_fmt.load(path)

    def test_converting_a_malformed_coco_file_fails_cleanly(self, tmp_path: Path):
        src = tmp_path / "src"
        (src / "annotations").mkdir(parents=True)
        (src / "annotations" / "instances_train.json").write_text(
            '{"annotations": [], "images": [{"id": 1,', encoding="utf-8"
        )
        result = run(
            "convert.dataset",
            {"folder": str(src), "output": str(tmp_path / "out"), "to": "voc",
             "source_format": "coco"},
        )
        assert not result.ok
        assert "instances_train.json" in result.errors[0]
        assert "cannot parse" in result.errors[0]

    @pytest.mark.parametrize(
        "document",
        [
            {"images": [{"file_name": "a.jpg"}], "annotations": [], "categories": []},
            {"images": 5, "annotations": [], "categories": []},
            {"images": [], "annotations": [7], "categories": [{"name": "cat"}]},
            {"images": [{"id": 1, "file_name": "a.jpg", "width": "wide"}],
             "annotations": [], "categories": []},
        ],
        ids=["image-without-id", "images-not-a-list", "junk-entries", "width-not-a-number"],
    )
    def test_valid_json_of_the_wrong_shape_never_raises_anything_else(
        self, tmp_path: Path, document: dict
    ):
        src = tmp_path / "src"
        (src / "annotations").mkdir(parents=True)
        (src / "annotations" / "instances_train.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        # Either it reads (skipping what it cannot use) or it fails by name;
        # what it must not do is escape as a KeyError or TypeError.
        result = run(
            "convert.dataset",
            {"folder": str(src), "output": str(tmp_path / "out"), "to": "voc",
             "source_format": "coco"},
        )
        assert result.ok or "instances_train.json" in result.errors[0] or (
            "no annotated images" in result.errors[0]
        )

    def test_a_json_list_handed_to_the_labelme_reader_is_a_named_error(self, tmp_path: Path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps([{"image_id": 1, "bbox": [1, 2, 3, 4]}]), encoding="utf-8")
        with pytest.raises(AnnotationError, match="results.json"):
            labelme.read(path)

    def test_a_json_file_that_is_not_labelme_is_not_an_empty_image(self, tmp_path: Path):
        """A stray settings file used to become a 0x0 background image."""
        make_image(tmp_path / "a.jpg", 1)
        (tmp_path / "a.json").write_text(
            json.dumps({"shapes": [], "imagePath": "a.jpg", "imageWidth": W, "imageHeight": H}),
            encoding="utf-8",
        )
        (tmp_path / "settings.json").write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
        dataset = formats.read_dataset(tmp_path, "labelme")
        assert [item.annotation.filename for item in dataset.items] == ["a.jpg"]
        assert any("settings.json" in w for w in dataset.warnings)
