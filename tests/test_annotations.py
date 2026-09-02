"""Pascal VOC reading/writing and COCO construction.

The dialect tests are the important ones: SAM-produced polygons used to convert
to empty COCO segmentations because only one dialect was ever readable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_voc

from data_factory.core.annotations import coco, voc
from data_factory.core.annotations.model import Annotation, AnnotationError, Shape


class TestVocReading:
    def test_indexed_polygon_dialect(self, tmp_path: Path):
        path = write_voc(
            tmp_path / "a.xml", "Open", bbox=(10, 10, 60, 70),
            polygon=[(10, 10), (60, 10), (60, 70)], dialect="indexed",
        )
        annotation = voc.read(path)
        assert annotation.shapes[0].polygon == [(10, 10), (60, 10), (60, 70)]
        assert annotation.shapes[0].bbox == (10, 10, 60, 70)

    def test_pt_polygon_dialect_with_no_bndbox(self, tmp_path: Path):
        """The SAM tool writes <pt> points and omits the box entirely."""
        path = write_voc(
            tmp_path / "b.xml", "Open", bbox=None,
            polygon=[(5, 5), (50, 5), (50, 55)], dialect="pt",
        )
        annotation = voc.read(path)
        assert annotation.shapes[0].polygon == [(5, 5), (50, 5), (50, 55)]
        # Derived from the outline, so downstream consumers still get a box.
        assert annotation.shapes[0].bbox == (5, 5, 50, 55)

    def test_decimal_coordinates_are_tolerated(self, tmp_path: Path):
        path = tmp_path / "c.xml"
        path.write_text(
            "<annotation><size><width>100</width><height>80</height></size>"
            "<object><name>Open</name><bndbox><xmin>10.7</xmin><ymin>5.2</ymin>"
            "<xmax>50.9</xmax><ymax>40.1</ymax></bndbox></object></annotation>",
            encoding="utf-8",
        )
        assert voc.read(path).shapes[0].bbox == (10, 5, 50, 40)

    def test_inverted_box_is_normalised(self, tmp_path: Path):
        path = write_voc(tmp_path / "d.xml", "Open", bbox=(60, 70, 10, 10))
        assert voc.read(path).shapes[0].bbox == (10, 10, 60, 70)

    def test_missing_size_is_recovered_from_the_shapes(self, tmp_path: Path):
        path = tmp_path / "e.xml"
        path.write_text(
            "<annotation><object><name>Open</name>"
            "<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>"
            "</object></annotation>",
            encoding="utf-8",
        )
        annotation = voc.read(path)
        assert (annotation.width, annotation.height) == (31, 41)

    def test_corrupt_file_raises(self, tmp_path: Path):
        path = tmp_path / "bad.xml"
        path.write_text("<annotation><object", encoding="utf-8")
        with pytest.raises(AnnotationError, match="malformed"):
            voc.read(path)

    def test_background_label_is_valid_and_empty(self, tmp_path: Path):
        path = tmp_path / "bg.xml"
        path.write_text(
            "<annotation><size><width>10</width><height>10</height></size></annotation>",
            encoding="utf-8",
        )
        annotation = voc.read(path)
        assert annotation.is_background and annotation.shapes == []


class TestVocWriting:
    def test_round_trip_preserves_geometry(self, tmp_path: Path):
        original = Annotation(
            filename="x.jpg", width=100, height=80,
            shapes=[Shape("Open", (5, 5, 50, 55), [(5, 5), (50, 5), (50, 55)])],
        )
        voc.write(original, tmp_path / "x.xml")
        restored = voc.read(tmp_path / "x.xml")
        assert restored.shapes[0].bbox == original.shapes[0].bbox
        assert restored.shapes[0].polygon == original.shapes[0].polygon
        assert restored.width == 100

    def test_pt_dialect_is_normalised_to_indexed_on_write(self, tmp_path: Path):
        source = write_voc(
            tmp_path / "s.xml", "Open", bbox=None,
            polygon=[(1, 2), (3, 4), (5, 6)], dialect="pt",
        )
        voc.write(voc.read(source), tmp_path / "out.xml")
        text = (tmp_path / "out.xml").read_text(encoding="utf-8")
        assert "<x1>1</x1>" in text and "<pt>" not in text

    def test_background_writes_a_valid_object_free_file(self, tmp_path: Path):
        voc.write_background("img.jpg", 100, 80, tmp_path / "bg.xml")
        assert voc.read(tmp_path / "bg.xml").is_background


class TestShapeGeometry:
    def test_polygon_area_uses_the_shoelace_formula(self):
        # A right triangle with legs 40 and 50 has area 1000, not the box's 2000.
        shape = Shape("x", (0, 0, 40, 50), [(0, 0), (40, 0), (40, 50)])
        assert shape.polygon_area() == pytest.approx(1000.0)
        assert shape.bbox_area == 2000

    def test_area_falls_back_to_the_box_without_a_polygon(self):
        assert Shape("x", (0, 0, 10, 10)).polygon_area() == pytest.approx(100.0)

    def test_two_point_polygon_is_not_treated_as_a_polygon(self):
        # COCO needs at least three points; two cannot enclose an area.
        assert Shape("x", (0, 0, 5, 5), [(0, 0), (5, 5)]).has_polygon is False

    def test_clamping_keeps_coordinates_inside_the_image(self):
        clamped = Shape("x", (-5, -5, 500, 400)).clamped(100, 80)
        assert clamped.bbox == (0, 0, 99, 79)


class TestCocoBuilder:
    def _two_spellings(self) -> list[Annotation]:
        return [
            Annotation("a.jpg", 100, 80, shapes=[Shape("Open", (0, 0, 10, 10))]),
            Annotation("b.jpg", 100, 80, shapes=[Shape("open", (0, 0, 10, 10))]),
        ]

    def test_casefold_merges_case_variants_and_reports_it(self):
        builder = coco.CocoBuilder(case_policy="casefold")
        for annotation in self._two_spellings():
            builder.add(annotation)
        document = builder.build()
        # This is the Project_Door V3 defect: four classes where there are two.
        assert len(document["categories"]) == 1
        assert builder.categories.merges() == {"Open": ["Open", "open"]}

    def test_exact_policy_keeps_them_separate(self):
        builder = coco.CocoBuilder(case_policy="exact")
        for annotation in self._two_spellings():
            builder.add(annotation)
        assert len(builder.build()["categories"]) == 2

    def test_polygons_become_segmentation_and_area(self):
        builder = coco.CocoBuilder()
        builder.add(
            Annotation("a.jpg", 100, 80,
                       shapes=[Shape("Open", (0, 0, 40, 50), [(0, 0), (40, 0), (40, 50)])])
        )
        annotation = builder.build()["annotations"][0]
        assert annotation["segmentation"] == [[0, 0, 40, 0, 40, 50]]
        assert annotation["area"] == pytest.approx(1000.0)
        assert annotation["bbox"] == [0, 0, 40, 50]

    def test_background_image_appears_with_no_annotations(self):
        builder = coco.CocoBuilder()
        builder.add(Annotation("bg.jpg", 100, 80))
        document = builder.build()
        assert len(document["images"]) == 1
        assert document["annotations"] == []
        assert builder.stats()["background_images"] == 1

    def test_label_map_is_applied_during_export(self):
        builder = coco.CocoBuilder(label_map={"Open": "door_open"})
        builder.add(Annotation("a.jpg", 10, 10, shapes=[Shape("Open", (0, 0, 5, 5))]))
        assert builder.build()["categories"][0]["name"] == "door_open"

    def test_summarise_flags_case_collisions(self):
        document = {
            "images": [{"id": 1}, {"id": 2}],
            "annotations": [{"image_id": 1, "category_id": 1, "segmentation": [[1, 2, 3, 4, 5, 6]]}],
            "categories": [{"id": 1, "name": "Open"}, {"id": 2, "name": "open"}],
        }
        summary = coco.summarise(document)
        assert summary["case_collisions"] == {"open": ["Open", "open"]}
        assert summary["with_segmentation"] == 1
        assert summary["background_images"] == 1
