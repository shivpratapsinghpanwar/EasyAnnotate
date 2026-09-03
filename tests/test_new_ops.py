"""The operations added to bring the tool up to what people expect of one.

These run through the registry rather than calling the functions directly, so
each test also checks the operation is declared correctly -- a wrong param kind
or a missing default fails here rather than in a user's terminal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from conftest import make_image, make_pair, write_voc

import data_factory as df
from data_factory.core.annotations import formats, yolo
from data_factory.core.types import Context


def run(name: str, params: dict, ctx: Context | None = None):
    return df.get_operation(name).run(params, ctx or Context())


@pytest.fixture
def split_dataset(tmp_path: Path) -> Path:
    """A dataset already divided into train/ and val/."""
    root = tmp_path / "split"
    for index in range(4):
        make_pair(root / "train", f"open{index}", "Open", 100 + index)
        make_pair(root / "train", f"close{index}", "Close", 200 + index)
    for index in range(2):
        make_pair(root / "val", f"vopen{index}", "Open", 300 + index)
        make_pair(root / "val", f"vclose{index}", "Close", 400 + index)
    return root


class TestConvert:
    def test_voc_to_yolo(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "yolo"
        result = run("convert.dataset", {"folder": str(clean_dataset), "output": str(out), "to": "yolo"})
        assert result.ok
        assert result.counts["images"] == 7
        assert (out / "data.yaml").exists()
        classes, _ = yolo.read_data_yaml(out / "data.yaml")
        assert set(classes.names) == {"Open", "Close"}

    def test_yolo_back_to_voc_keeps_the_objects(self, clean_dataset: Path, tmp_path: Path):
        yolo_dir, voc_dir = tmp_path / "y", tmp_path / "v"
        run("convert.dataset", {"folder": str(clean_dataset), "output": str(yolo_dir), "to": "yolo"})
        result = run("convert.dataset", {"folder": str(yolo_dir), "output": str(voc_dir), "to": "voc"})
        assert result.ok
        before = formats.read_dataset(clean_dataset, "voc").stats()
        after = formats.read_dataset(voc_dir, "voc").stats()
        assert after["objects"] == before["objects"]

    def test_source_format_is_detected_when_not_given(self, clean_dataset: Path, tmp_path: Path):
        result = run("convert.dataset", {"folder": str(clean_dataset), "output": str(tmp_path / "o"), "to": "coco"})
        assert result.data["source_format"] == "voc"

    def test_converting_in_place_is_refused(self, clean_dataset: Path):
        result = run("convert.dataset", {"folder": str(clean_dataset), "output": str(clean_dataset), "to": "yolo"})
        assert not result.ok
        assert "must differ" in result.errors[0]

    def test_dry_run_writes_nothing(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "o"
        result = run(
            "convert.dataset",
            {"folder": str(clean_dataset), "output": str(out), "to": "yolo"},
            Context(dry_run=True),
        )
        assert result.ok
        assert not out.exists() or not any(out.rglob("*.txt"))

    def test_polygon_loss_to_csv_is_warned_about(self, tmp_path: Path):
        src = tmp_path / "src"
        make_image(src / "a.jpg", 1)
        write_voc(src / "a.xml", "Open", polygon=[(10, 10), (60, 10), (60, 70)])
        result = run("convert.dataset", {"folder": str(src), "output": str(tmp_path / "o"), "to": "csv"})
        assert any("lose their polygons" in w for w in result.warnings)

    def test_detect_reports_the_format_and_classes(self, clean_dataset: Path):
        result = run("convert.detect", {"folder": str(clean_dataset)})
        assert result.data["format"] == "voc"
        assert set(result.data["classes"]) == {"Open", "Close"}
        assert "yolo" in result.data["convertible_to"]


class TestResize:
    def test_images_and_boxes_are_both_rescaled(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "resized"
        result = run(
            "image.resize",
            {"folder": str(clean_dataset), "output": str(out), "width": 60, "height": 45,
             "mode": "stretch"},
        )
        assert result.ok and result.counts["resized"] == 7

        from data_factory.core import imaging

        image = next(out.glob("*.jpg"))
        assert imaging.image_size(image) == (60, 45)

        loaded = formats.read_dataset(out, "voc")
        for item in loaded.items:
            for shape in item.annotation.shapes:
                assert shape.bbox[2] <= 60 and shape.bbox[3] <= 45

    def test_letterbox_keeps_the_aspect_ratio(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "lb"
        run("image.resize", {"folder": str(clean_dataset), "output": str(out),
                             "width": 200, "height": 200, "mode": "letterbox"})
        from data_factory.core import imaging

        assert imaging.image_size(next(out.glob("*.jpg"))) == (200, 200)

    def test_it_can_convert_format_while_resizing(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "y"
        result = run("image.resize", {"folder": str(clean_dataset), "output": str(out),
                                      "width": 64, "height": 64, "format": "yolo"})
        assert result.ok and (out / "data.yaml").exists()

    def test_resizing_in_place_is_refused(self, clean_dataset: Path):
        result = run("image.resize", {"folder": str(clean_dataset), "output": str(clean_dataset),
                                      "width": 10, "height": 10})
        assert not result.ok


class TestTile:
    def test_a_large_image_produces_several_tiles(self, tmp_path: Path):
        src = tmp_path / "src"
        make_image(src / "big.jpg", 1, size=(400, 300))
        write_voc(src / "big.xml", "Open", bbox=(50, 50, 120, 120), size=(400, 300))

        result = run(
            "image.tile",
            {"folder": str(src), "output": str(tmp_path / "tiles"),
             "tile_width": 128, "tile_height": 128, "overlap": 0.2},
        )
        assert result.ok
        assert result.counts["tiles_written"] >= 1
        # Only tiles that actually contain the object are kept by default.
        assert result.counts["tiles_skipped_empty"] > 0

    def test_keep_empty_retains_the_background_tiles(self, tmp_path: Path):
        src = tmp_path / "src"
        make_image(src / "big.jpg", 1, size=(400, 300))
        write_voc(src / "big.xml", "Open", bbox=(50, 50, 120, 120), size=(400, 300))

        result = run(
            "image.tile",
            {"folder": str(src), "output": str(tmp_path / "t2"), "tile_width": 128,
             "tile_height": 128, "overlap": 0.0, "keep_empty": True},
        )
        assert result.counts["tiles_skipped_empty"] == 0
        assert result.counts["tiles_written"] > 4

    def test_tiles_come_out_the_requested_size(self, tmp_path: Path):
        src = tmp_path / "src"
        make_image(src / "big.jpg", 1, size=(400, 300))
        write_voc(src / "big.xml", "Open", bbox=(10, 10, 380, 280), size=(400, 300))
        out = tmp_path / "t3"
        run("image.tile", {"folder": str(src), "output": str(out),
                           "tile_width": 128, "tile_height": 128, "overlap": 0.0})
        from data_factory.core import imaging

        for image in out.glob("*.jpg"):
            assert imaging.image_size(image) == (128, 128)


class TestAutoOrient:
    def test_upright_images_pass_through_unchanged(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "o"
        result = run("image.autoorient", {"folder": str(clean_dataset), "output": str(out)})
        assert result.ok
        assert result.counts["rotated"] == 0
        assert result.counts["unchanged"] == 7
        assert any("non-upright" in w for w in result.warnings)


class TestAugment:
    def test_it_multiplies_the_dataset(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "aug"
        result = run(
            "augment.dataset",
            {"folder": str(clean_dataset), "output": str(out), "multiplier": 2,
             "flip_horizontal": True, "brightness": 0.3, "seed": 1},
        )
        assert result.ok
        assert result.counts["generated"] == 14  # 7 images x 2 variants
        assert result.counts["total_images"] == 21  # plus the originals

        loaded = formats.read_dataset(out, "voc")
        assert loaded.stats()["images"] == 21

    def test_the_same_seed_reproduces_the_same_files(self, clean_dataset: Path, tmp_path: Path):
        params = {"folder": str(clean_dataset), "multiplier": 2, "flip_horizontal": True,
                  "rotate": 15.0, "brightness": 0.3, "seed": 7, "include_original": False}
        a, b = tmp_path / "a", tmp_path / "b"
        run("augment.dataset", {**params, "output": str(a)})
        run("augment.dataset", {**params, "output": str(b)})

        for left in a.glob("*.jpg"):
            right = b / left.name
            assert right.exists(), left.name
            assert left.read_bytes() == right.read_bytes(), left.name

    def test_seeding_is_stable_across_processes(self):
        """The in-process test above cannot catch a per-process hash seed.

        Python randomises str.__hash__ per interpreter, so a seed derived from
        hash() reproduces perfectly within one run and silently differs between
        runs -- which is the case that actually matters.
        """
        import subprocess
        import sys

        code = (
            "from data_factory.core.ops.augment import _seed_for;"
            "print(_seed_for(7, 'open0', 1))"
        )
        first = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        second = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": "12345"},
        ).stdout.strip()
        assert first == second and first

    def test_a_different_seed_gives_different_output(self, clean_dataset: Path, tmp_path: Path):
        params = {"folder": str(clean_dataset), "multiplier": 2, "flip_horizontal": True,
                  "rotate": 20.0, "include_original": False}
        a, b = tmp_path / "a", tmp_path / "b"
        run("augment.dataset", {**params, "output": str(a), "seed": 1})
        run("augment.dataset", {**params, "output": str(b), "seed": 2})
        differing = sum(
            1 for left in a.glob("*.jpg")
            if (b / left.name).exists() and left.read_bytes() != (b / left.name).read_bytes()
        )
        assert differing > 0

    def test_annotations_follow_a_flip(self, tmp_path: Path):
        src = tmp_path / "src"
        make_image(src / "a.jpg", 1)
        write_voc(src / "a.xml", "Open", bbox=(10, 20, 40, 60))
        out = tmp_path / "aug"
        run(
            "augment.dataset",
            {"folder": str(src), "output": str(out), "multiplier": 1, "flip_horizontal": True,
             "brightness": 0.0, "include_original": False, "seed": 3},
        )
        loaded = formats.read_dataset(out, "voc")
        box = loaded.items[0].annotation.shapes[0].bbox
        # Either flipped or not (the roll is random), but never left corrupt.
        assert box[2] > box[0] and box[3] > box[1]
        assert 0 <= box[0] and box[2] < 120

    def test_no_augmentations_enabled_fails_loudly(self, clean_dataset: Path, tmp_path: Path):
        result = run(
            "augment.dataset",
            {"folder": str(clean_dataset), "output": str(tmp_path / "o"), "flip_horizontal": False,
             "brightness": 0.0, "contrast": 0.0},
        )
        assert not result.ok
        assert "no augmentations are enabled" in result.errors[0]

    def test_splits_filter_restricts_what_is_augmented(self, split_dataset: Path, tmp_path: Path):
        out = tmp_path / "aug"
        result = run(
            "augment.dataset",
            {"folder": str(split_dataset), "output": str(out), "multiplier": 1,
             "flip_horizontal": True, "splits": ["train"], "seed": 5},
        )
        assert result.ok
        # 8 train images augmented once; the 4 val images copied through only.
        assert result.counts["generated"] == 8
        assert not any((out / "val").glob("*_aug*.jpg"))
        assert any((out / "train").glob("*_aug*.jpg"))


class TestMerge:
    def test_two_datasets_combine(self, tmp_path: Path):
        a, b = tmp_path / "a", tmp_path / "b"
        make_pair(a, "x", "Open", 1)
        make_pair(a, "y", "Open", 2)
        make_pair(b, "z", "Close", 3)

        result = run("dataset.merge", {"folders": [str(a), str(b)], "output": str(tmp_path / "m")})
        assert result.ok
        assert result.counts["images"] == 3
        assert set(result.data["classes"]) == {"Open", "Close"}

    def test_colliding_filenames_do_not_overwrite_each_other(self, tmp_path: Path):
        a, b = tmp_path / "a", tmp_path / "b"
        make_pair(a, "image_001", "Open", 1)
        make_pair(b, "image_001", "Close", 2)

        out = tmp_path / "m"
        result = run("dataset.merge", {"folders": [str(a), str(b)], "output": str(out)})
        assert result.ok
        assert result.counts["images"] == 2
        assert len(list(out.rglob("*.jpg"))) == 2
        assert formats.read_dataset(out, "voc").stats()["objects"] == 2

    def test_label_map_reconciles_class_names(self, tmp_path: Path):
        a, b = tmp_path / "a", tmp_path / "b"
        make_pair(a, "x", "car", 1)
        make_pair(b, "y", "truck", 2)

        result = run(
            "dataset.merge",
            {"folders": [str(a), str(b)], "output": str(tmp_path / "m"),
             "label_map": {"car": "vehicle", "truck": "vehicle"}},
        )
        assert result.data["classes"] == ["vehicle"]
        assert result.counts["classes"] == 1

    def test_merging_one_folder_is_refused(self, tmp_path: Path):
        a = tmp_path / "a"
        make_pair(a, "x", "Open", 1)
        result = run("dataset.merge", {"folders": [str(a)], "output": str(tmp_path / "m")})
        assert not result.ok

    def test_splits_are_preserved(self, split_dataset: Path, tmp_path: Path):
        other = tmp_path / "other"
        make_pair(other / "train", "extra", "Open", 900)
        out = tmp_path / "m"
        result = run(
            "dataset.merge",
            {"folders": [str(split_dataset), str(other)], "output": str(out)},
        )
        assert result.ok
        assert (out / "train").is_dir() and (out / "val").is_dir()


class TestFilter:
    def test_keeping_one_class_drops_the_others_objects_and_images(
        self, clean_dataset: Path, tmp_path: Path
    ):
        out = tmp_path / "f"
        result = run(
            "dataset.filter",
            {"folder": str(clean_dataset), "output": str(out), "keep_classes": ["Open"]},
        )
        assert result.ok
        assert result.counts["images_kept"] == 3
        loaded = formats.read_dataset(out, "voc")
        assert loaded.labels == ["Open"]

    def test_backgrounds_can_be_kept_as_hard_negatives(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "f"
        result = run(
            "dataset.filter",
            {"folder": str(clean_dataset), "output": str(out), "keep_classes": ["Open"],
             "keep_backgrounds": True},
        )
        # 3 Open images, plus 3 Close images emptied by the filter, plus the bg.
        assert result.counts["images_kept"] == 7

    def test_dropping_a_class(self, clean_dataset: Path, tmp_path: Path):
        result = run(
            "dataset.filter",
            {"folder": str(clean_dataset), "output": str(tmp_path / "f"), "drop_classes": ["Close"]},
        )
        assert result.counts["images_kept"] == 3

    def test_min_area_removes_small_objects(self, tmp_path: Path):
        src = tmp_path / "src"
        make_pair(src, "big", "Open", 1, bbox=(0, 0, 100, 80))
        make_pair(src, "small", "Open", 2, bbox=(0, 0, 5, 5))
        result = run(
            "dataset.filter",
            {"folder": str(src), "output": str(tmp_path / "f"), "min_area": 1000},
        )
        assert result.counts["images_kept"] == 1
        assert result.counts["objects_removed"] == 1

    def test_an_unknown_class_name_warns(self, clean_dataset: Path, tmp_path: Path):
        result = run(
            "dataset.filter",
            {"folder": str(clean_dataset), "output": str(tmp_path / "f"),
             "keep_classes": ["Nonexistent"]},
        )
        assert any("not present" in w for w in result.warnings)

    def test_require_polygons_keeps_only_outlined_objects(self, tmp_path: Path):
        src = tmp_path / "src"
        make_image(src / "poly.jpg", 1)
        write_voc(src / "poly.xml", "Open", polygon=[(10, 10), (60, 10), (60, 70)])
        make_pair(src, "boxonly", "Open", 2)
        result = run(
            "dataset.filter",
            {"folder": str(src), "output": str(tmp_path / "f"), "require_polygons": True},
        )
        assert result.counts["images_kept"] == 1


class TestSample:
    def test_a_fraction_is_taken(self, clean_dataset: Path, tmp_path: Path):
        result = run(
            "dataset.sample",
            {"folder": str(clean_dataset), "output": str(tmp_path / "s"), "fraction": 0.5},
        )
        assert result.ok
        assert result.counts["sampled"] == 4  # round(7 * 0.5)

    def test_an_explicit_count_wins(self, clean_dataset: Path, tmp_path: Path):
        result = run(
            "dataset.sample",
            {"folder": str(clean_dataset), "output": str(tmp_path / "s"), "count": 3},
        )
        assert result.counts["sampled"] == 3

    def test_the_same_seed_selects_the_same_images(self, clean_dataset: Path, tmp_path: Path):
        a, b = tmp_path / "a", tmp_path / "b"
        for out in (a, b):
            run("dataset.sample", {"folder": str(clean_dataset), "output": str(out),
                                   "count": 3, "seed": 11})
        assert sorted(p.name for p in a.glob("*.jpg")) == sorted(p.name for p in b.glob("*.jpg"))

    def test_per_class_caps_the_dominant_class(self, tmp_path: Path):
        src = tmp_path / "src"
        for index in range(10):
            make_pair(src, f"open{index}", "Open", index)
        for index in range(2):
            make_pair(src, f"close{index}", "Close", 100 + index)

        result = run(
            "dataset.sample",
            {"folder": str(src), "output": str(tmp_path / "s"), "per_class": 3},
        )
        counts = result.data["images_per_class"]
        assert counts["Open"] <= 3
        assert counts["Close"] == 2  # under the cap, so untouched

    def test_asking_for_more_than_exists_returns_everything(self, clean_dataset: Path, tmp_path: Path):
        result = run(
            "dataset.sample",
            {"folder": str(clean_dataset), "output": str(tmp_path / "s"), "count": 999},
        )
        assert result.counts["sampled"] == 7


class TestHealth:
    def test_it_finds_the_case_variant_classes(self, dataset: Path):
        result = run("inspect.health", {"folder": str(dataset)})
        assert result.ok
        issues = [f["issue"] for f in result.data["findings"]]
        assert any("capitalisation" in i for i in issues)

    def test_a_balanced_dataset_reports_no_errors(self, clean_dataset: Path):
        result = run("inspect.health", {"folder": str(clean_dataset)})
        assert result.counts["errors"] == 0

    def test_severe_imbalance_is_an_error(self, tmp_path: Path):
        src = tmp_path / "src"
        for index in range(60):
            make_pair(src, f"common{index}", "Common", index)
        make_pair(src, "rare0", "Rare", 999)

        result = run("inspect.health", {"folder": str(src)})
        errors = [f for f in result.data["findings"] if f["severity"] == "error"]
        assert any("imbalance" in f["issue"].lower() for f in errors)

    def test_every_finding_carries_a_fix(self, dataset: Path):
        result = run("inspect.health", {"folder": str(dataset)})
        assert result.data["findings"]
        for finding in result.data["findings"]:
            assert finding["fix"].strip()

    def test_zero_area_boxes_are_flagged(self, tmp_path: Path):
        src = tmp_path / "src"
        make_pair(src, "bad", "Open", 1, bbox=(10, 10, 10, 10))
        result = run("inspect.health", {"folder": str(src)})
        assert any("Zero-area" in f["issue"] for f in result.data["findings"])

    def test_it_works_on_a_yolo_dataset_too(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "y"
        run("convert.dataset", {"folder": str(clean_dataset), "output": str(out), "to": "yolo"})
        result = run("inspect.health", {"folder": str(out)})
        assert result.ok and result.data["format"] == "yolo"


class TestLeakage:
    def test_a_clean_split_reports_nothing(self, split_dataset: Path):
        result = run("inspect.leakage", {"folder": str(split_dataset), "method": "exact"})
        assert result.ok
        assert result.counts["leaked_pairs"] == 0

    def test_a_duplicated_image_across_splits_is_caught(self, split_dataset: Path):
        leaked = split_dataset / "val" / "leaked.jpg"
        leaked.write_bytes((split_dataset / "train" / "open0.jpg").read_bytes())

        result = run("inspect.leakage", {"folder": str(split_dataset), "method": "exact"})
        assert result.counts["leaked_pairs"] == 1
        assert result.counts["affected_images"] == 1
        assert any("optimistic" in w for w in result.warnings)

    def test_perceptual_matching_catches_a_recompressed_copy(self, split_dataset: Path):
        from conftest import make_similar_image

        make_similar_image(split_dataset / "val" / "nearly.jpg", 100)
        result = run(
            "inspect.leakage",
            {"folder": str(split_dataset), "method": "phash", "threshold": 8},
        )
        assert result.counts["leaked_pairs"] >= 1

    def test_one_split_cannot_be_checked(self, tmp_path: Path):
        make_pair(tmp_path / "train", "a", "Open", 1)
        result = run("inspect.leakage", {"folder": str(tmp_path)})
        assert not result.ok
        assert "at least two splits" in result.errors[0]

    def test_the_yolo_images_layout_is_understood(self, split_dataset: Path, tmp_path: Path):
        out = tmp_path / "y"
        run("convert.dataset", {"folder": str(split_dataset), "output": str(out), "to": "yolo"})
        result = run("inspect.leakage", {"folder": str(out), "method": "exact"})
        assert result.ok
        assert set(result.data["per_split"]) == {"train", "val"}


class TestReport:
    def test_it_writes_a_self_contained_html_file(self, clean_dataset: Path, tmp_path: Path):
        target = tmp_path / "report.html"
        result = run("report.html", {"folder": str(clean_dataset), "output": str(target)})
        assert result.ok and target.exists()

        document = target.read_text(encoding="utf-8")
        assert document.startswith("<!doctype html>")
        # Self-contained: nothing fetched from a network at open time.
        for forbidden in ("http://", "https://", "<script"):
            assert forbidden not in document

    def test_the_report_names_the_classes_and_counts(self, clean_dataset: Path, tmp_path: Path):
        target = tmp_path / "r.html"
        run("report.html", {"folder": str(clean_dataset), "output": str(target)})
        document = target.read_text(encoding="utf-8")
        assert "Open" in document and "Close" in document

    def test_json_sidecar_is_optional_and_valid(self, clean_dataset: Path, tmp_path: Path):
        target = tmp_path / "r.html"
        run("report.html", {"folder": str(clean_dataset), "output": str(target), "json_too": True})
        payload = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        assert payload["totals"]["images"] == 7

    def test_dry_run_writes_no_file(self, clean_dataset: Path, tmp_path: Path):
        target = tmp_path / "r.html"
        result = run(
            "report.html", {"folder": str(clean_dataset), "output": str(target)}, Context(dry_run=True)
        )
        assert result.ok and not target.exists()

    def test_quotes_in_a_class_name_are_escaped(self, tmp_path: Path):
        """A label only has to survive XML to reach the renderer -- and then
        it is attacker-ish input as far as the HTML is concerned."""
        src = tmp_path / "src"
        make_pair(src, "a", 'say "hi"', 1)
        target = tmp_path / "r.html"
        run("report.html", {"folder": str(src), "output": str(target)})
        document = target.read_text(encoding="utf-8")
        assert "&quot;hi&quot;" in document

    def test_the_renderer_escapes_markup_in_any_field(self):
        """Directly, because a class name containing < cannot survive VOC XML
        but can easily arrive from COCO or CSV, which have no such restriction."""
        from data_factory.core.ops.report import _render

        document = _render(
            "<img src=x onerror=alert(1)>",
            {
                "totals": {"images": 1, "objects": 1, "classes": 1, "backgrounds": 0,
                           "polygon_objects": 0, "images_with_polygons": 0},
                "per_class": [{"class": "<script>alert(1)</script>", "objects": 1,
                               "images": 1, "polygons": 0, "share": 1.0, "median_area_px": 10}],
                "object_sizes": {"tiny": 0, "small": 0, "medium": 0, "large": 1},
                "relative_area": {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0},
                "aspect_ratio": {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0},
                "objects_per_image": {"mean": 1.0, "max": 1, "histogram": {}},
                "image_sizes": {"distinct": 1, "most_common": [{"size": "1x1", "images": 1}]},
                "splits": {"(none)": 1},
                "geometry": {"out_of_bounds": 0, "degenerate": 0},
                "findings": [],
                "balance": {"ratio": 1.0, "gini": 0.0},
                "source_format": "coco",
            },
        )
        assert "<script>alert" not in document
        assert "&lt;script&gt;" in document
        assert "<img src=x" not in document


class TestRegistryIntegration:
    def test_every_new_operation_is_discoverable(self):
        names = {op.name for op in df.list_operations()}
        for expected in (
            "convert.dataset", "convert.detect", "image.resize", "image.tile",
            "image.autoorient", "augment.dataset", "dataset.merge", "dataset.filter",
            "dataset.sample", "inspect.health", "inspect.leakage", "report.html",
        ):
            assert expected in names, expected

    def test_the_schema_covers_them_with_usable_params(self):
        schema = df.schema()
        by_name = {op["name"]: op for op in schema["operations"]}
        convert = by_name["convert.dataset"]
        target = next(p for p in convert["params"] if p["name"] == "to")
        assert target["kind"] == "choice"
        assert set(target["choices"]) == set(formats.FORMATS)

    def test_read_only_operations_are_not_marked_destructive(self):
        for name in ("inspect.health", "inspect.leakage", "convert.detect"):
            assert not df.get_operation(name).destructive, name
