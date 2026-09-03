"""Format readers, writers, and round trips.

A converter's failure mode is not a crash, it is arriving with fewer objects
than it left with. So most of these tests convert something and then read it
back, asserting on what survived -- and on what the tool *said* it lost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import IMAGE_SIZE, make_image, make_pair, write_voc

from data_factory.core.annotations import coco as coco_fmt
from data_factory.core.annotations import csvfmt, formats, labelme, yolo
from data_factory.core.annotations.model import Annotation, AnnotationError, Shape
from data_factory.core.types import Context

W, H = IMAGE_SIZE


def sample(filename="x.jpg", polygon=None) -> Annotation:
    return Annotation(
        filename=filename,
        width=W,
        height=H,
        shapes=[
            Shape(label="Open", bbox=(10, 10, 60, 70), polygon=polygon),
            Shape(label="Close", bbox=(70, 20, 110, 80)),
        ],
    )


class TestYolo:
    def test_box_roundtrips_within_a_pixel(self, tmp_path: Path):
        classes = yolo.ClassList(["Close", "Open"])
        path = tmp_path / "x.txt"
        yolo.write(sample(), path, classes)
        back = yolo.read(path, classes, W, H)
        assert [s.label for s in back.shapes] == ["Open", "Close"]
        for before, after in zip(sample().shapes, back.shapes):
            for a, b in zip(before.bbox, after.bbox):
                assert abs(a - b) <= 1

    def test_polygon_roundtrips(self, tmp_path: Path):
        polygon = [(10, 10), (60, 10), (60, 70), (10, 70)]
        classes = yolo.ClassList(["Open", "Close"])
        path = tmp_path / "x.txt"
        yolo.write(sample(polygon=polygon), path, classes)
        back = yolo.read(path, classes, W, H)
        assert back.shapes[0].has_polygon
        assert len(back.shapes[0].polygon) == 4

    def test_segmentation_can_be_flattened_to_boxes(self, tmp_path: Path):
        polygon = [(10, 10), (60, 10), (60, 70)]
        classes = yolo.ClassList(["Open"])
        path = tmp_path / "x.txt"
        yolo.write(sample(polygon=polygon), path, classes, segmentation=False)
        assert all(len(line.split()) == 5 for line in path.read_text().splitlines())

    def test_a_background_writes_an_empty_file_not_no_file(self, tmp_path: Path):
        path = tmp_path / "bg.txt"
        yolo.write(Annotation(filename="bg.jpg", width=W, height=H), path, yolo.ClassList([]))
        assert path.exists() and path.read_text() == ""

    def test_coordinates_are_clamped_into_the_unit_range(self, tmp_path: Path):
        annotation = Annotation(
            filename="x.jpg", width=W, height=H,
            shapes=[Shape(label="Open", bbox=(-20, -20, W + 40, H + 40))],
        )
        path = tmp_path / "x.txt"
        yolo.write(annotation, path, yolo.ClassList(["Open"]))
        values = [float(v) for v in path.read_text().split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_data_yaml_roundtrips(self, tmp_path: Path):
        classes = yolo.ClassList(["cat", "dog"])
        path = yolo.write_data_yaml(tmp_path / "data.yaml", classes, splits={"train": "images/train"})
        back, splits = yolo.read_data_yaml(path)
        assert back.names == ["cat", "dog"]
        assert splits["train"] == "images/train"

    def test_data_yaml_accepts_the_list_style_names(self, tmp_path: Path):
        path = tmp_path / "data.yaml"
        path.write_text("nc: 2\nnames: [cat, dog]\ntrain: images\n", encoding="utf-8")
        classes, _ = yolo.read_data_yaml(path)
        assert classes.names == ["cat", "dog"]

    def test_unknown_class_index_is_named_visibly_not_silently_class_zero(self):
        classes = yolo.ClassList(["cat"])
        assert classes.name_for(7) == "class_7"

    def test_malformed_line_raises(self, tmp_path: Path):
        path = tmp_path / "x.txt"
        path.write_text("0 0.5 0.5 0.5\n", encoding="utf-8")  # 3 values: neither box nor polygon
        with pytest.raises(AnnotationError, match="expected 4 box values"):
            yolo.read(path, yolo.ClassList(["cat"]), W, H)

    def test_label_path_follows_the_images_labels_convention(self, tmp_path: Path):
        image = tmp_path / "images" / "train" / "a.jpg"
        assert yolo.label_path_for(image, tmp_path) == tmp_path / "labels" / "train" / "a.txt"

    def test_label_path_falls_back_to_a_sidecar_for_flat_folders(self, tmp_path: Path):
        image = tmp_path / "a.jpg"
        assert yolo.label_path_for(image, tmp_path) == tmp_path / "a.txt"


class TestLabelMe:
    def test_polygon_roundtrips(self, tmp_path: Path):
        polygon = [(10, 10), (60, 10), (60, 70)]
        path = tmp_path / "x.json"
        labelme.write(sample(polygon=polygon), path)
        back = labelme.read(path)
        assert back.shapes[0].polygon == polygon
        assert back.width == W

    def test_a_box_is_written_as_a_rectangle_and_read_back_as_a_box(self, tmp_path: Path):
        path = tmp_path / "x.json"
        labelme.write(sample(), path)
        document = json.loads(path.read_text())
        assert document["shapes"][0]["shape_type"] == "rectangle"
        assert labelme.read(path).shapes[0].bbox == (10, 10, 60, 70)

    def test_unsupported_shape_types_are_reported_not_silently_dropped(self, tmp_path: Path):
        path = tmp_path / "x.json"
        path.write_text(
            json.dumps(
                {
                    "shapes": [
                        {"label": "a", "points": [[0, 0], [10, 10]], "shape_type": "circle"},
                        {"label": "b", "points": [[0, 0], [10, 10]], "shape_type": "rectangle"},
                    ],
                    "imageWidth": W, "imageHeight": H, "imagePath": "x.jpg",
                }
            ),
            encoding="utf-8",
        )
        skipped: list[str] = []
        annotation = labelme.read(path, skipped_out=skipped)
        assert len(annotation.shapes) == 1
        assert skipped == ["circle"]

    def test_image_path_is_reduced_to_a_basename(self, tmp_path: Path):
        path = tmp_path / "x.json"
        path.write_text(
            json.dumps({"shapes": [], "imagePath": "../images/deep/x.jpg",
                        "imageWidth": W, "imageHeight": H}),
            encoding="utf-8",
        )
        assert labelme.read(path).filename == "x.jpg"

    def test_image_data_is_not_embedded(self, tmp_path: Path):
        path = tmp_path / "x.json"
        labelme.write(sample(), path)
        assert json.loads(path.read_text())["imageData"] is None


class TestCsv:
    def test_roundtrips_boxes_and_polygons(self, tmp_path: Path):
        polygon = [(10, 10), (60, 10), (60, 70)]
        path = tmp_path / "a.csv"
        csvfmt.write([sample(polygon=polygon)], path)
        back = csvfmt.read(path)
        assert len(back) == 1
        assert [s.label for s in back[0].shapes] == ["Open", "Close"]
        assert back[0].shapes[0].polygon == polygon

    def test_a_background_survives_the_roundtrip(self, tmp_path: Path):
        path = tmp_path / "a.csv"
        csvfmt.write([Annotation(filename="bg.jpg", width=W, height=H)], path)
        back = csvfmt.read(path)
        assert len(back) == 1 and back[0].is_background

    def test_alternative_column_names_are_accepted(self, tmp_path: Path):
        path = tmp_path / "a.csv"
        path.write_text(
            "image,label,x_min,y_min,x_max,y_max\nx.jpg,cat,1,2,3,4\n", encoding="utf-8"
        )
        back = csvfmt.read(path)
        assert back[0].shapes[0].label == "cat"
        assert back[0].shapes[0].bbox == (1, 2, 3, 4)

    def test_missing_columns_raise_with_a_useful_message(self, tmp_path: Path):
        path = tmp_path / "a.csv"
        path.write_text("filename,class\nx.jpg,cat\n", encoding="utf-8")
        with pytest.raises(AnnotationError, match="missing column"):
            csvfmt.read(path)


class TestCocoReader:
    def test_reads_back_what_the_builder_wrote(self, tmp_path: Path):
        builder = coco_fmt.CocoBuilder()
        builder.add(sample(polygon=[(10, 10), (60, 10), (60, 70)]))
        document = builder.build()
        back = coco_fmt.to_annotations(document)
        assert len(back) == 1
        assert {s.label for s in back[0].shapes} == {"Open", "Close"}
        assert back[0].shapes[0].has_polygon

    def test_background_images_survive(self):
        builder = coco_fmt.CocoBuilder()
        builder.add(Annotation(filename="bg.jpg", width=W, height=H))
        back = coco_fmt.to_annotations(builder.build())
        assert len(back) == 1 and back[0].is_background

    def test_rle_segmentation_is_reported_and_kept_as_a_box(self):
        document = {
            "images": [{"id": 1, "file_name": "x.jpg", "width": W, "height": H}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 10],
                 "segmentation": {"counts": "abc", "size": [H, W]}, "iscrowd": 1}
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        notes: list[str] = []
        back = coco_fmt.to_annotations(document, unsupported_out=notes)
        assert back[0].shapes[0].bbox == (1, 2, 11, 12)
        assert not back[0].shapes[0].has_polygon
        assert notes and "RLE" in notes[0]


class TestDetection:
    def test_voc_folder_is_detected(self, clean_dataset: Path):
        assert formats.detect(clean_dataset) == "voc"

    def test_yolo_folder_is_detected_by_its_data_yaml(self, tmp_path: Path):
        yolo.write_data_yaml(tmp_path / "data.yaml", yolo.ClassList(["cat"]))
        make_image(tmp_path / "images" / "train" / "a.jpg", 1)
        assert formats.detect(tmp_path) == "yolo"

    def test_coco_folder_is_detected(self, tmp_path: Path):
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances_train.json").write_text(
            json.dumps({"images": [], "annotations": [], "categories": []}), encoding="utf-8"
        )
        assert formats.detect(tmp_path) == "coco"

    def test_labelme_folder_is_detected(self, tmp_path: Path):
        make_image(tmp_path / "a.jpg", 1)
        labelme.write(sample("a.jpg"), tmp_path / "a.json")
        assert formats.detect(tmp_path) == "labelme"

    def test_an_empty_folder_raises_rather_than_guessing(self, tmp_path: Path):
        with pytest.raises(AnnotationError, match="cannot tell"):
            formats.detect(tmp_path)


class TestReadDataset:
    def test_voc_dataset_loads_with_its_backgrounds(self, clean_dataset: Path):
        dataset = formats.read_dataset(clean_dataset, "voc")
        assert dataset.stats()["images"] == 7
        assert dataset.stats()["backgrounds"] == 1
        assert dataset.labels == ["Close", "Open"]

    def test_unlabelled_images_are_carried_as_backgrounds(self, tmp_path: Path):
        make_image(tmp_path / "lonely.jpg", 1)
        make_pair(tmp_path, "a", "Open", 2)
        dataset = formats.read_dataset(tmp_path, "voc")
        assert dataset.stats()["images"] == 2

    def test_a_corrupt_label_warns_instead_of_aborting_the_load(self, dataset: Path):
        loaded = formats.read_dataset(dataset, "voc")
        assert any("malformed" in w for w in loaded.warnings)
        assert loaded.stats()["images"] > 0

    def test_splits_are_recovered_from_folder_names(self, tmp_path: Path):
        make_pair(tmp_path / "train", "a", "Open", 1)
        make_pair(tmp_path / "val", "b", "Open", 2)
        dataset = formats.read_dataset(tmp_path, "voc", recursive=True)
        assert sorted({i.split for i in dataset.items}) == ["train", "val"]

    def test_yolo_without_a_data_yaml_warns_about_the_class_names(self, tmp_path: Path):
        make_image(tmp_path / "a.jpg", 1)
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        loaded = formats.read_dataset(tmp_path, "yolo")
        assert any("no data.yaml" in w for w in loaded.warnings)
        assert loaded.items[0].annotation.shapes[0].label == "class_0"


class TestRoundTrips:
    """Every format out, and back in again, with the object count intact."""

    @pytest.mark.parametrize("fmt", ["voc", "yolo", "coco", "labelme", "csv"])
    def test_objects_survive_a_roundtrip(self, clean_dataset: Path, tmp_path: Path, fmt: str):
        source = formats.read_dataset(clean_dataset, "voc")
        out = tmp_path / fmt
        formats.write_dataset(source, out, fmt, ctx=Context())

        back = formats.read_dataset(out, fmt)
        assert back.stats()["objects"] == source.stats()["objects"], fmt
        assert set(back.labels) == set(source.labels), fmt

    @pytest.mark.parametrize("fmt", ["voc", "yolo", "coco", "labelme"])
    def test_polygons_survive_a_roundtrip(self, tmp_path: Path, fmt: str):
        src = tmp_path / "src"
        make_image(src / "a.jpg", 1)
        write_voc(src / "a.xml", "Open", polygon=[(10, 10), (60, 10), (60, 70), (10, 70)])

        source = formats.read_dataset(src, "voc")
        assert source.items[0].annotation.has_polygons

        out = tmp_path / f"out_{fmt}"
        formats.write_dataset(source, out, fmt, ctx=Context())
        back = formats.read_dataset(out, fmt)
        assert back.items[0].annotation.has_polygons, fmt

    def test_converting_to_csv_warns_that_outlines_are_lost(self, tmp_path: Path):
        src = tmp_path / "src"
        make_image(src / "a.jpg", 1)
        write_voc(src / "a.xml", "Open", polygon=[(10, 10), (60, 10), (60, 70)])
        source = formats.read_dataset(src, "voc")

        report = formats.write_dataset(source, tmp_path / "out", "csv", ctx=Context())
        assert any("lose their polygons" in str(w) for w in report["warnings"])

    def test_dry_run_writes_nothing(self, clean_dataset: Path, tmp_path: Path):
        source = formats.read_dataset(clean_dataset, "voc")
        out = tmp_path / "nothing"
        formats.write_dataset(source, out, "yolo", ctx=Context(dry_run=True))
        assert not out.exists() or not any(out.rglob("*.txt"))

    def test_yolo_export_writes_a_usable_data_yaml(self, clean_dataset: Path, tmp_path: Path):
        source = formats.read_dataset(clean_dataset, "voc")
        out = tmp_path / "yolo"
        formats.write_dataset(source, out, "yolo", ctx=Context())
        classes, splits = yolo.read_data_yaml(out / "data.yaml")
        assert set(classes.names) == {"Open", "Close"}
        assert "train" in splits
