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
from conftest import IMAGE_SIZE, make_image, make_pair

import data_factory as df
from data_factory.annotate import review
from data_factory.core.annotations import coco as coco_fmt
from data_factory.core.annotations import dota, formats, labelme, voc, yolo
from data_factory.core.annotations.model import Annotation, AnnotationError, Shape
from data_factory.core.types import Context
from data_factory.core.writer import DatasetWriter

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


# ------------------------------------------------------- COCO or LabelMe?


class TestCocoDetection:
    """What a .json file is gets decided from its own top-level keys.

    It used to be decided from the first 4096 characters, so a COCO file whose
    ``images`` array came first and held more than about forty images -- which
    includes this tool's own export -- was taken for LabelMe and read back as
    one empty image.
    """

    def test_the_tools_own_coco_export_reads_back(self, tmp_path: Path):
        src = tmp_path / "voc"
        for index in range(71):
            make_pair(src, f"img{index:03d}", "Open" if index % 2 else "Close", index)
        source = formats.read_dataset(src, "voc")

        out = tmp_path / "coco"
        formats.write_dataset(source, out, "coco", ctx=Context())
        exported = (out / "annotations" / "instances_train.json").read_text(encoding="utf-8")
        assert '"annotations"' not in exported[:4096]  # the case the sniff missed

        assert formats.detect(out) == "coco"
        back = formats.read_dataset(out, "auto")
        assert back.source_format == "coco"
        assert back.stats()["images"] == 71
        assert back.stats()["objects"] == 71
        assert back.stats()["missing_images"] == 0

        def content(dataset: formats.Dataset) -> list:
            return sorted(
                (item.annotation.filename, [(s.label, s.bbox) for s in item.annotation.shapes])
                for item in dataset.items
            )

        assert content(back) == content(source)

    def test_naming_the_format_reads_the_same_export(self, tmp_path: Path):
        src = tmp_path / "voc"
        for index in range(71):
            make_pair(src, f"img{index:03d}", "Open", index)
        out, again = tmp_path / "coco", tmp_path / "again"
        run("convert.dataset", {"folder": str(src), "output": str(out), "to": "coco"})

        result = run(
            "convert.dataset",
            {"folder": str(out), "output": str(again), "to": "voc", "source_format": "coco"},
        )
        assert result.ok, result.errors
        assert result.counts["images"] == 71 and result.counts["objects"] == 71
        assert len(list(again.rglob("*.xml"))) == 71

    def test_a_compact_coco_file_with_many_images_first_is_coco(self, tmp_path: Path):
        (tmp_path / "_annotations.coco.json").write_text(
            json.dumps(coco_document(images=500)), encoding="utf-8"
        )
        assert formats.detect(tmp_path) == "coco"
        assert formats.read_dataset(tmp_path, "auto").stats()["objects"] == 500

    def test_a_labelme_file_is_not_coco_because_of_a_word_in_it(self, tmp_path: Path):
        """A class called "categories" used to be enough to call the file COCO."""
        make_image(tmp_path / "a.jpg", 1)
        document = {
            "shapes": [{"label": "categories", "points": [[1, 2], [30, 40]],
                        "shape_type": "rectangle"}],
            "imagePath": "a.jpg", "imageWidth": W, "imageHeight": H,
        }
        (tmp_path / "a.json").write_text(json.dumps(document), encoding="utf-8")
        assert formats.detect(tmp_path) == "labelme"
        assert formats.read_dataset(tmp_path, "auto").labels == ["categories"]

    def test_a_broken_coco_file_is_named_even_when_its_start_gives_nothing_away(
        self, tmp_path: Path
    ):
        src = tmp_path / "src"
        (src / "annotations").mkdir(parents=True)
        (src / "annotations" / "instances_train.json").write_text(
            '{"images": [{"id": 1,', encoding="utf-8"
        )
        result = run(
            "convert.dataset",
            {"folder": str(src), "output": str(tmp_path / "out"), "to": "voc",
             "source_format": "coco"},
        )
        assert not result.ok
        assert "instances_train.json" in result.errors[0]
        assert "cannot parse" in result.errors[0]

    def test_a_results_list_is_not_a_dataset_and_the_error_says_so(self, tmp_path: Path):
        (tmp_path / "results.json").write_text(
            json.dumps([{"image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.5}]),
            encoding="utf-8",
        )
        with pytest.raises(AnnotationError, match="detection results"):
            formats.read_dataset(tmp_path, "coco")

    def test_a_good_file_is_read_and_a_broken_one_beside_it_is_reported(self, tmp_path: Path):
        annotations = tmp_path / "annotations"
        annotations.mkdir()
        (annotations / "instances_train.json").write_text(
            json.dumps(coco_document(images=3)), encoding="utf-8"
        )
        (annotations / "instances_val.json").write_text('{"images": [', encoding="utf-8")
        dataset = formats.read_dataset(tmp_path, "coco")
        assert dataset.stats()["images"] == 3
        assert any("instances_val.json" in w for w in dataset.warnings)


# ------------------------------------------------------------- COCO writer


def _coco_names(path: Path) -> dict[int, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {c["id"]: c["name"] for c in document["categories"]}


class TestCocoWriter:
    def test_a_category_id_means_the_same_class_in_every_split_file(self, tmp_path: Path):
        """Each split got its own numbering, by first appearance: id 1 was
        Open in instances_train.json and Close in instances_val.json."""
        src = tmp_path / "src"
        make_pair(src / "train", "a", "Open", 1)
        make_pair(src / "val", "b", "Close", 2)
        make_pair(src / "val", "c", "Open", 3)
        out = tmp_path / "coco"
        formats.write_dataset(formats.read_dataset(src, "voc"), out, "coco", ctx=Context())

        train = _coco_names(out / "annotations" / "instances_train.json")
        val = _coco_names(out / "annotations" / "instances_val.json")
        assert train == val == {1: "Close", 2: "Open"}

        document = json.loads((out / "annotations" / "instances_train.json").read_text())
        assert [train[a["category_id"]] for a in document["annotations"]] == ["Open"]

    def test_category_ids_follow_a_declared_class_order(self, tmp_path: Path):
        src = tmp_path / "src"
        make_pair(src, "a", "Close", 1)
        make_pair(src, "b", "Open", 2)
        dataset = formats.read_dataset(src, "voc")
        dataset.classes = ["Open", "Handle", "Close"]
        out = tmp_path / "coco"
        formats.write_dataset(dataset, out, "coco", ctx=Context())
        names = _coco_names(out / "annotations" / "instances_train.json")
        assert names == {1: "Open", 2: "Handle", 3: "Close"}

    def test_the_streaming_writer_numbers_every_split_alike(self, tmp_path: Path):
        writer = DatasetWriter(tmp_path / "out", "coco", Context())
        box = (10, 10, 60, 70)
        writer.add(Annotation("a.jpg", W, H, shapes=[Shape("Open", box)]), stem="a", split="train")
        writer.add(Annotation("b.jpg", W, H, shapes=[Shape("Close", box)]), stem="b", split="val")
        writer.close()
        train = _coco_names(tmp_path / "out" / "annotations" / "instances_train.json")
        val = _coco_names(tmp_path / "out" / "annotations" / "instances_val.json")
        assert train == val == {1: "Close", 2: "Open"}

    def test_an_image_that_is_not_on_disk_keeps_its_own_extension(self, tmp_path: Path):
        """a.png became a.jpg in the COCO file whenever the image was missing."""
        annotation = Annotation("a.png", W, H, shapes=[Shape("Open", (10, 10, 60, 70))])
        dataset = formats.Dataset(items=[formats.Item(annotation=annotation)])
        out = tmp_path / "coco"
        formats.write_dataset(dataset, out, "coco", ctx=Context())
        document = json.loads((out / "annotations" / "instances_train.json").read_text())
        assert document["images"][0]["file_name"] == "a.png"
