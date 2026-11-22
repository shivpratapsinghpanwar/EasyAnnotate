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


# ------------------------------------------------------------------ splits


class TestSplits:
    def test_a_file_name_is_never_a_split(self, tmp_path: Path):
        """photo_test.xml went to test/ and train.xml to train/."""
        src = tmp_path / "src"
        for stem in ("photo_test", "train", "val", "shot_valid"):
            make_pair(src, stem, "Open", 1)
        dataset = formats.read_dataset(src, "voc")
        assert {item.stem: item.split for item in dataset.items} == {
            "photo_test": "", "train": "", "val": "", "shot_valid": "",
        }
        out = tmp_path / "out"
        formats.write_dataset(dataset, out, "voc", ctx=Context())
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == []

    def test_a_folder_named_for_a_split_still_is_one(self, tmp_path: Path):
        src = tmp_path / "src"
        make_pair(src / "train", "photo_test", "Open", 1)
        make_pair(src / "Valid", "a", "Open", 2)
        dataset = formats.read_dataset(src, "voc")
        assert {item.stem: item.split for item in dataset.items} == {
            "photo_test": "train", "a": "val",
        }

    @pytest.mark.parametrize("fmt", ["voc", "labelme", "yolo", "dota"])
    def test_an_unlabelled_image_keeps_its_split(self, tmp_path: Path, fmt: str):
        src = tmp_path / "src"
        annotation = Annotation("a.jpg", W, H, shapes=[Shape("Open", (10, 10, 60, 70))])
        if fmt in ("yolo", "dota"):
            make_image(src / "images" / "val" / "a.jpg", 1)
            make_image(src / "images" / "val" / "bare.jpg", 2)
            label = src / "labels" / "val" / "a.txt"
            label.parent.mkdir(parents=True)
            if fmt == "yolo":
                yolo.write(annotation, label, yolo.ClassList(["Open"]))
                yolo.write_data_yaml(src / "data.yaml", yolo.ClassList(["Open"]))
            else:
                dota.write(annotation, label)
        else:
            make_image(src / "val" / "a.jpg", 1)
            make_image(src / "val" / "bare.jpg", 2)
            if fmt == "voc":
                voc.write(annotation, src / "val" / "a.xml")
            else:
                labelme.write(annotation, src / "val" / "a.json")
        dataset = formats.read_dataset(src, fmt)
        assert {item.stem: item.split for item in dataset.items} == {"a": "val", "bare": "val"}


# ------------------------------------------------------- YOLO class numbers


def yolo_folder(root: Path, numbers: list[int], *, names: list[str] | None = None) -> Path:
    """Flat YOLO labels, one box per class number, with or without data.yaml."""
    for position, number in enumerate(numbers):
        make_image(root / f"img{position:02d}.jpg", position)
        (root / f"img{position:02d}.txt").write_text(
            f"{number} 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8"
        )
    if names is not None:
        yolo.write_data_yaml(root / "data.yaml", yolo.ClassList(names))
    return root


def class_column(root: Path) -> list[int]:
    labels = sorted(p for p in root.rglob("*.txt") if p.name != "classes.txt")
    return [int(line.split()[0]) for p in labels for line in p.read_text().splitlines()]


class TestYoloClassNumbers:
    """A YOLO label is a class *number*. Whatever else a conversion does, a
    YOLO folder converted to YOLO must come out with the numbers it went in
    with -- anything else relabels the dataset without an error."""

    @pytest.mark.parametrize(
        "numbers", [list(range(12)), [0, 3, 12], [2, 10]], ids=["twelve", "gaps", "no-zero"]
    )
    def test_numbers_survive_when_there_is_no_names_file(self, tmp_path: Path, numbers):
        src = yolo_folder(tmp_path / "src", numbers)
        out = tmp_path / "out"
        result = run("convert.dataset", {"folder": str(src), "output": str(out), "to": "yolo"})
        assert result.ok
        assert class_column(out / "labels") == numbers
        names, _ = yolo.read_data_yaml(out / "data.yaml")
        assert names.names == [f"class_{i}" for i in range(max(numbers) + 1)]

    def test_numbers_beyond_the_names_file_survive_and_are_reported(self, tmp_path: Path):
        src = yolo_folder(tmp_path / "src", [0, 1, 5], names=["cat", "dog"])
        out = tmp_path / "out"
        result = run("convert.dataset", {"folder": str(src), "output": str(out), "to": "yolo"})
        assert class_column(out / "labels") == [0, 1, 5]
        names, _ = yolo.read_data_yaml(out / "data.yaml")
        assert names.names == ["cat", "dog", "class_2", "class_3", "class_4", "class_5"]
        assert any("data.yaml does not name" in w and "5" in w for w in result.warnings)

    def test_classes_names_the_numbers_when_there_is_no_names_file(self, tmp_path: Path):
        """`--classes dog cat` gave dog, cat, class_0, class_1 -- and class 0
        was written out as class 2."""
        src = yolo_folder(tmp_path / "src", [0, 1, 0])
        out = tmp_path / "out"
        result = run(
            "convert.dataset",
            {"folder": str(src), "output": str(out), "to": "yolo", "classes": ["dog", "cat"]},
        )
        assert result.data["classes"] == ["dog", "cat"]
        assert class_column(out / "labels") == [0, 1, 0]
        assert any("0 = dog, 1 = cat" in w for w in result.warnings)
        assert not any("Point --classes" in w for w in result.warnings)

        back = formats.read_dataset(out, "yolo")
        assert [i.annotation.shapes[0].label for i in back.items] == ["dog", "cat", "dog"]

    def test_classes_shorter_than_the_numbers_in_use_moves_nothing(self, tmp_path: Path):
        src = yolo_folder(tmp_path / "src", [0, 3])
        out = tmp_path / "out"
        result = run(
            "convert.dataset",
            {"folder": str(src), "output": str(out), "to": "yolo", "classes": ["dog", "cat"]},
        )
        assert result.data["classes"] == ["dog", "cat", "class_2", "class_3"]
        assert class_column(out / "labels") == [0, 3]
        assert any("'classes' does not name" in w and "3" in w for w in result.warnings)

    def test_without_names_the_warning_says_how_to_give_them(self, tmp_path: Path):
        src = yolo_folder(tmp_path / "src", [0])
        warning = next(w for w in formats.read_dataset(src, "yolo").warnings if "no data.yaml" in w)
        assert "number order" in warning and "classes.txt" in warning

    @pytest.mark.parametrize("where", [".", "labels"])
    def test_classes_txt_is_read(self, tmp_path: Path, where: str):
        src = tmp_path / "src"
        make_image(src / "images" / "a.jpg", 1)
        (src / "labels").mkdir()
        (src / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (src / where / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
        dataset = formats.read_dataset(src, "auto")
        assert dataset.source_format == "yolo"
        assert dataset.classes == ["cat", "dog"]
        assert dataset.items[0].annotation.shapes[0].label == "dog"
        assert not any("no data.yaml" in w for w in dataset.warnings)

    def test_classes_reordering_known_names_says_so(self, tmp_path: Path):
        src = yolo_folder(tmp_path / "src", [0, 1], names=["cat", "dog"])
        out = tmp_path / "out"
        result = run(
            "convert.dataset",
            {"folder": str(src), "output": str(out), "to": "yolo", "classes": ["dog", "cat"]},
        )
        assert result.data["classes"] == ["dog", "cat"]
        assert class_column(out / "labels") == [1, 0]  # the same objects, renumbered on request
        assert any("cat 0 -> 1" in w and "dog 1 -> 0" in w for w in result.warnings)

    def test_classes_in_the_order_the_source_already_uses_is_silent(self, tmp_path: Path):
        src = yolo_folder(tmp_path / "src", [0, 1], names=["cat", "dog"])
        result = run(
            "convert.dataset",
            {"folder": str(src), "output": str(tmp_path / "out"), "to": "yolo",
             "classes": ["cat", "dog"]},
        )
        assert result.warnings == []


# ------------------------------------------------------- data.yaml class names

AWKWARD_NAMES = [
    "no", "yes", "010", "1.50", "~", "0x1F", "on", "null", "True", "1e3", "y", "12",
    "a: b", "#hash", '"quoted', "'single'", " padded ", "a #b", "- dash", "[x]", "{x}",
    "*star", "&amp", "!bang", "%pct", "@at", "`tick", "|", ">", "?", "tail:", "",
    "two\nlines", "tab\there", "caf\u00e9", "\u732b", "plain", "traffic light", "10mm_bolt",
]


class TestDataYamlNames:
    def test_names_are_read_as_text_whatever_yaml_would_make_of_them(self, tmp_path: Path):
        """no became False, 010 became 8, 1.50 became 1.5, ~ became None."""
        path = tmp_path / "data.yaml"
        path.write_text(
            "names:\n  0: no\n  1: yes\n  2: 010\n  3: 1.50\n  4: ~\n  5: 0x1F\n", encoding="utf-8"
        )
        classes, _ = yolo.read_data_yaml(path)
        assert classes.names == ["no", "yes", "010", "1.50", "~", "0x1F"]

        path.write_text("names: [no, 010, 1.50]\n", encoding="utf-8")
        assert yolo.read_data_yaml(path)[0].names == ["no", "010", "1.50"]

    def test_every_name_survives_being_written_and_read(self, tmp_path: Path):
        path = yolo.write_data_yaml(tmp_path / "data.yaml", yolo.ClassList(AWKWARD_NAMES))
        assert yolo.read_data_yaml(path)[0].names == AWKWARD_NAMES

    def test_a_trainer_reads_the_written_names_as_the_same_text(self, tmp_path: Path):
        """Trainers load data.yaml with an ordinary YAML loader, which types
        what it reads. So this checks with one, not with our own reader."""
        import yaml

        path = yolo.write_data_yaml(tmp_path / "data.yaml", yolo.ClassList(AWKWARD_NAMES))
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
        assert [loaded[i] for i in range(len(AWKWARD_NAMES))] == AWKWARD_NAMES
        assert all(isinstance(name, str) for name in loaded.values())

    def test_ordinary_names_are_still_written_bare(self, tmp_path: Path):
        path = yolo.write_data_yaml(
            tmp_path / "data.yaml", yolo.ClassList(["cat", "traffic light", "Open"])
        )
        text = path.read_text(encoding="utf-8")
        assert "  0: cat\n  1: traffic light\n  2: Open\n" in text

    def test_names_keyed_by_something_other_than_a_number_fail_by_name(self, tmp_path: Path):
        path = tmp_path / "data.yaml"
        path.write_text("names:\n  first: cat\n", encoding="utf-8")
        with pytest.raises(AnnotationError, match="data.yaml"):
            yolo.read_data_yaml(path)


# ------------------------------------------------- YOLO boxes past the edge


def dataset_of(*shapes: Shape, width: int = 100, height: int = 80) -> formats.Dataset:
    annotation = Annotation("a.jpg", width, height, shapes=list(shapes))
    return formats.Dataset(items=[formats.Item(annotation=annotation)])


class TestYoloClipping:
    def test_a_box_past_the_edge_becomes_its_visible_part(self, tmp_path: Path):
        """Centre and size were clamped one by one: a box from -50 to 50 on a
        100-wide image came out centred on the edge and as wide as the image."""
        path = tmp_path / "a.txt"
        annotation = Annotation("a.jpg", 100, 80, shapes=[Shape("cat", (-50, 20, 50, 60))])
        yolo.write(annotation, path, yolo.ClassList(["cat"]))
        assert path.read_text() == "0 0.250000 0.500000 0.500000 0.500000\n"

        back = yolo.read(path, yolo.ClassList(["cat"]), 100, 80)
        assert back.shapes[0].bbox == (0, 20, 50, 60)

    def test_a_box_inside_the_image_is_written_as_before(self, tmp_path: Path):
        path = tmp_path / "a.txt"
        annotation = Annotation("a.jpg", 100, 80, shapes=[Shape("cat", (10, 20, 50, 60))])
        yolo.write(annotation, path, yolo.ClassList(["cat"]))
        assert path.read_text() == "0 0.300000 0.500000 0.400000 0.500000\n"

    def test_clipped_and_dropped_boxes_are_counted(self, tmp_path: Path):
        dataset = dataset_of(
            Shape("cat", (-50, 20, 50, 60)),    # clipped
            Shape("cat", (90, 70, 130, 95)),    # clipped
            Shape("cat", (10, 10, 10, 40)),     # no area
            Shape("cat", (120, 10, 150, 40)),   # wholly outside
            Shape("cat", (10, 20, 50, 60)),     # fine
        )
        report = formats.write_dataset(dataset, tmp_path / "out", "yolo", ctx=Context())
        warnings = [str(w) for w in report["warnings"]]
        assert any(w.startswith("2 box(es) reach outside their image") for w in warnings)
        assert any(w.startswith("2 box(es) have no area inside their image") for w in warnings)
        lines = (tmp_path / "out" / "labels" / "train" / "a.txt").read_text().splitlines()
        assert len(lines) == 3

        preview = formats.write_dataset(dataset, tmp_path / "dry", "yolo", ctx=Context(dry_run=True))
        assert preview["warnings"] == report["warnings"]

    def test_a_clean_dataset_gets_no_such_warning(self, tmp_path: Path):
        dataset = dataset_of(Shape("cat", (10, 20, 50, 60)), Shape("cat", (0, 0, 100, 80)))
        report = formats.write_dataset(dataset, tmp_path / "out", "yolo", ctx=Context())
        assert report["warnings"] == []

    def test_an_outline_with_points_outside_is_counted_too(self, tmp_path: Path):
        outline = [(-10, 10), (50, 10), (50, 70), (-10, 70), (20, 40)]
        dataset = dataset_of(Shape("cat", (-10, 10, 50, 70), polygon=outline))
        report = formats.write_dataset(dataset, tmp_path / "out", "yolo", ctx=Context())
        assert any("1 outline(s) have points outside" in str(w) for w in report["warnings"])
