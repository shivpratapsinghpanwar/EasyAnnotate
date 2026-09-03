"""Interoperability with the tools people actually train in.

Asserting that an export is "COCO format" is cheap and frequently wrong. These
tests check it against pycocotools -- the reference implementation Detectron2,
MMDetection, torchvision and every published COCO benchmark sit on top of. If
our file loads and evaluates there, it loads everywhere that matters.

pycocotools is an optional developer dependency; these skip without it rather
than forcing a compiler onto anyone running the main suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import make_image, write_voc

import data_factory as df
from data_factory.core.types import Context

pytest.importorskip("pycocotools", reason="optional: pip install pycocotools")


@pytest.fixture
def segmented(tmp_path: Path) -> Path:
    """A small dataset with real polygons, since that is the fragile part."""
    folder = tmp_path / "src"
    for index in range(4):
        make_image(folder / f"poly{index}.jpg", index)
        write_voc(
            folder / f"poly{index}.xml",
            "Open",
            bbox=(10, 10, 60, 70),
            polygon=[(10, 10), (60, 12), (58, 70), (12, 66)],
        )
    for index in range(2):
        make_image(folder / f"box{index}.jpg", 50 + index)
        write_voc(folder / f"box{index}.xml", "Close", bbox=(20, 20, 80, 60))
    return folder


class TestCocoInteroperability:
    def _export(self, source: Path, destination: Path) -> Path:
        df.run(
            "convert.dataset",
            {"folder": str(source), "output": str(destination), "to": "coco"},
            Context(),
        )
        return next((destination / "annotations").glob("instances_*.json"))

    def test_the_export_loads_in_pycocotools(self, segmented: Path, tmp_path: Path):
        from pycocotools.coco import COCO

        coco = COCO(str(self._export(segmented, tmp_path / "coco")))
        assert len(coco.imgs) == 6
        assert len(coco.anns) == 6
        assert len(coco.cats) == 2

    def test_polygons_decode_to_non_empty_masks(self, segmented: Path, tmp_path: Path):
        """The real test of a segmentation export: does the reference decoder
        turn our polygons into pixels, or into an empty mask nobody notices?"""
        from pycocotools.coco import COCO

        coco = COCO(str(self._export(segmented, tmp_path / "coco")))
        segmented_anns = [a for a in coco.anns.values() if a["segmentation"]]
        assert len(segmented_anns) == 4
        for ann in segmented_anns:
            assert coco.annToMask(ann).sum() > 0

    def test_cocoeval_scores_ground_truth_against_itself_perfectly(
        self, segmented: Path, tmp_path: Path
    ):
        """COCOeval is what every benchmark number is computed with. Feeding
        the ground truth back as predictions must score 1.0 -- anything less
        means our boxes, areas or category ids are malformed in a way a
        trainer would silently inherit."""
        import contextlib
        import io

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco = COCO(str(self._export(segmented, tmp_path / "coco")))
        predictions = [
            {"image_id": a["image_id"], "category_id": a["category_id"],
             "bbox": a["bbox"], "score": 1.0}
            for a in coco.anns.values()
        ]
        path = tmp_path / "preds.json"
        path.write_text(json.dumps(predictions), encoding="utf-8")

        with contextlib.redirect_stdout(io.StringIO()):
            evaluator = COCOeval(coco, coco.loadRes(str(path)), "bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        assert evaluator.stats[0] > 0.99

    def test_area_is_the_polygon_area_not_the_box_area(self, segmented: Path, tmp_path: Path):
        """COCO stratifies AP by object area. Reporting box area for a polygon
        puts objects in the wrong size bucket, so AP_small/medium/large are
        computed against the wrong split -- one of the bugs this tool fixed."""
        from pycocotools.coco import COCO

        coco = COCO(str(self._export(segmented, tmp_path / "coco")))
        for ann in coco.anns.values():
            if not ann["segmentation"]:
                continue
            box_area = ann["bbox"][2] * ann["bbox"][3]
            assert ann["area"] < box_area, "polygon area should be under its box"
            assert abs(ann["area"] - coco.annToMask(ann).sum()) / ann["area"] < 0.1


class TestYoloInteroperability:
    """Ultralytics finds labels by string-swapping /images/ for /labels/ and
    reads classes from data.yaml. Both conventions have to hold exactly."""

    def _export(self, source: Path, destination: Path) -> Path:
        df.run(
            "convert.dataset",
            {"folder": str(source), "output": str(destination), "to": "yolo"},
            Context(),
        )
        return destination

    def test_data_yaml_carries_what_a_trainer_needs(self, segmented: Path, tmp_path: Path):
        out = self._export(segmented, tmp_path / "yolo")
        document = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        assert {"nc", "names", "train"} <= set(document)
        assert document["nc"] == len(document["names"]) == 2

    def test_every_image_has_a_label_where_ultralytics_looks(
        self, segmented: Path, tmp_path: Path
    ):
        out = self._export(segmented, tmp_path / "yolo")
        images = sorted((out / "images").rglob("*.jpg"))
        assert images
        for image in images:
            relative = image.relative_to(out / "images")
            assert (out / "labels" / relative).with_suffix(".txt").is_file(), image

    def test_coordinates_are_normalised_and_class_ids_in_range(
        self, segmented: Path, tmp_path: Path
    ):
        out = self._export(segmented, tmp_path / "yolo")
        classes = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))["nc"]
        lines = 0
        for label in (out / "labels").rglob("*.txt"):
            for line in label.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                lines += 1
                assert 0 <= int(parts[0]) < classes
                assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])
        assert lines == 6


class TestLabelMeInteroperability:
    def test_documents_match_the_labelme_schema(self, segmented: Path, tmp_path: Path):
        out = tmp_path / "labelme"
        df.run(
            "convert.dataset",
            {"folder": str(segmented), "output": str(out), "to": "labelme"},
            Context(),
        )
        for path in out.glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            assert {"version", "flags", "shapes", "imagePath", "imageHeight",
                    "imageWidth"} <= set(document)
            for shape in document["shapes"]:
                assert {"label", "points", "shape_type"} <= set(shape)
                assert shape["shape_type"] in ("polygon", "rectangle")
