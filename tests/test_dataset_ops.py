"""Splitting, sorting, COCO export, auditing, and visualisation."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import make_image, make_pair, write_voc

import data_factory as df
from data_factory import Context


class TestSplit:
    def test_every_pair_lands_in_exactly_one_split_with_both_files(self, clean_dataset: Path):
        output = clean_dataset.parent / "splits"
        result = df.run("dataset.split", {"folder": clean_dataset, "output": output, "seed": 1})

        assert result.counts["train"] + result.counts["val"] + result.counts["test"] == 7
        stems = []
        for name in ("train", "val", "test"):
            images = sorted(p.stem for p in (output / name).glob("*.jpg"))
            labels = sorted(p.stem for p in (output / name).glob("*.xml"))
            # A label must never be separated from its image.
            assert images == labels
            stems += images
        assert len(stems) == len(set(stems)) == 7

    def test_the_same_seed_gives_the_same_split(self, clean_dataset: Path):
        first = clean_dataset.parent / "a"
        second = clean_dataset.parent / "b"
        df.run("dataset.split", {"folder": clean_dataset, "output": first, "seed": 7})
        df.run("dataset.split", {"folder": clean_dataset, "output": second, "seed": 7})

        def listing(root: Path) -> dict[str, list[str]]:
            return {n: sorted(p.name for p in (root / n).glob("*")) for n in ("train", "val", "test")}

        assert listing(first) == listing(second)

    def test_a_different_seed_gives_a_different_split(self, tmp_path: Path):
        folder = tmp_path / "many"
        for index in range(20):
            make_pair(folder, f"img{index:02d}", "Open", index)

        one = df.run("dataset.split", {"folder": folder, "output": tmp_path / "s1", "seed": 1})
        two = df.run("dataset.split", {"folder": folder, "output": tmp_path / "s2", "seed": 2})
        names = lambda root: sorted(p.name for p in (root / "train").glob("*.jpg"))
        assert names(tmp_path / "s1") != names(tmp_path / "s2")
        assert one.counts["train"] == two.counts["train"]

    def test_stratification_spreads_each_class_across_the_splits(self, tmp_path: Path):
        folder = tmp_path / "balanced"
        for index in range(10):
            make_pair(folder, f"open{index:02d}", "Open", index)
        for index in range(10):
            make_pair(folder, f"close{index:02d}", "Close", 100 + index)

        result = df.run(
            "dataset.split",
            {"folder": folder, "output": tmp_path / "out", "ratios": "0.5,0.5",
             "names": "train,val", "seed": 3},
        )
        per_split = result.data["per_split_labels"]
        # Each class is halved, rather than one class landing entirely in train.
        assert per_split["train"] == {"Close": 5, "Open": 5}
        assert per_split["val"] == {"Close": 5, "Open": 5}

    def test_unstratified_splitting_is_still_available(self, tmp_path: Path):
        folder = tmp_path / "u"
        for index in range(10):
            make_pair(folder, f"img{index}", "Open", index)
        result = df.run(
            "dataset.split",
            {"folder": folder, "output": tmp_path / "o", "stratify": False, "seed": 1},
        )
        # Stratification off means no per-stratum breakdown is reported.
        assert "strata" not in result.data
        assert result.counts["train"] == 8

    def test_empty_split_still_gets_a_folder_and_a_warning(self, tmp_path: Path):
        """Downstream steps address splits by name; a missing folder breaks them."""
        folder = tmp_path / "tiny"
        make_pair(folder, "only", "Open", 1)
        output = tmp_path / "out"
        result = df.run("dataset.split", {"folder": folder, "output": output, "seed": 1})

        assert (output / "test").is_dir()
        assert result.counts["test"] == 0
        assert any("is empty" in w for w in result.warnings)

    def test_ratios_and_names_must_agree(self, clean_dataset: Path):
        result = df.run(
            "dataset.split",
            {"folder": clean_dataset, "output": clean_dataset.parent / "x",
             "ratios": "0.5,0.5", "names": "train,val,test"},
        )
        assert not result.ok and "must match" in result.errors[0]

    def test_images_labels_layout(self, clean_dataset: Path):
        output = clean_dataset.parent / "split2"
        df.run(
            "dataset.split",
            {"folder": clean_dataset, "output": output, "layout": "images_labels", "seed": 1},
        )
        assert (output / "train" / "images").is_dir()
        assert (output / "train" / "labels").is_dir()
        assert list((output / "train" / "images").glob("*.jpg"))

    def test_move_mode_empties_the_source(self, clean_dataset: Path):
        df.run(
            "dataset.split",
            {"folder": clean_dataset, "output": clean_dataset.parent / "moved", "mode": "move"},
        )
        assert not list(clean_dataset.glob("*.jpg"))


class TestSort:
    def test_pairs_group_by_class_with_a_background_bucket(self, clean_dataset: Path):
        output = clean_dataset.parent / "byclass"
        result = df.run("dataset.sort", {"folder": clean_dataset, "output": output})
        assert result.counts["sorted"] == 7
        assert len(list((output / "Open").glob("*.jpg"))) == 3
        assert len(list((output / "Close").glob("*.jpg"))) == 3
        assert len(list((output / "background").glob("*.jpg"))) == 1

    def test_multi_class_images_go_to_the_mixed_bucket(self, tmp_path: Path):
        folder = tmp_path / "mix"
        make_image(folder / "m.jpg", 1)
        (folder / "m.xml").write_text(
            "<annotation><size><width>120</width><height>90</height></size>"
            "<object><name>Open</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
            "<xmax>10</xmax><ymax>10</ymax></bndbox></object>"
            "<object><name>Close</name><bndbox><xmin>20</xmin><ymin>20</ymin>"
            "<xmax>30</xmax><ymax>30</ymax></bndbox></object></annotation>",
            encoding="utf-8",
        )
        df.run("dataset.sort", {"folder": folder, "output": tmp_path / "out"})
        assert (tmp_path / "out" / "mixed" / "m.jpg").exists()


class TestCocoExport:
    def test_case_variants_collapse_to_one_class(self, dataset: Path):
        """The door dataset's V3 defect: four classes exported where there are two."""
        output = dataset.parent / "coco"
        result = df.run("export.coco", {"folder": dataset, "output": output})

        document = json.loads((output / "annotations" / "instances.json").read_text())
        assert len(document["categories"]) == 2
        assert any("merged case-variant" in w for w in result.warnings)

    def test_exact_policy_preserves_the_split(self, dataset: Path):
        output = dataset.parent / "coco_exact"
        df.run("export.coco", {"folder": dataset, "output": output, "case_policy": "exact"})
        document = json.loads((output / "annotations" / "instances.json").read_text())
        assert len(document["categories"]) == 4

    def test_both_polygon_dialects_export_as_segmentation(self, dataset: Path):
        """SAM's <pt> polygons used to export as empty masks."""
        output = dataset.parent / "coco_seg"
        df.run("export.coco", {"folder": dataset, "output": output})
        document = json.loads((output / "annotations" / "instances.json").read_text())
        segmented = [a for a in document["annotations"] if a["segmentation"]]
        assert len(segmented) == 2

    def test_backgrounds_are_images_with_no_annotations(self, clean_dataset: Path):
        output = clean_dataset.parent / "coco_bg"
        result = df.run("export.coco", {"folder": clean_dataset, "output": output})
        assert result.counts["background_images"] == 1
        document = json.loads((output / "annotations" / "instances.json").read_text())
        annotated = {a["image_id"] for a in document["annotations"]}
        assert len(document["images"]) - len(annotated) == 1

    def test_backgrounds_can_be_excluded(self, clean_dataset: Path):
        result = df.run(
            "export.coco",
            {"folder": clean_dataset, "output": clean_dataset.parent / "c2",
             "include_backgrounds": False},
        )
        assert result.counts["backgrounds_skipped"] == 1

    def test_images_are_copied_alongside(self, clean_dataset: Path):
        output = clean_dataset.parent / "coco_imgs"
        df.run("export.coco", {"folder": clean_dataset, "output": output})
        assert len(list((output / "images").glob("*.jpg"))) == 7

    def test_split_export_writes_one_file_per_split(self, clean_dataset: Path):
        splits = clean_dataset.parent / "splits"
        df.run("dataset.split", {"folder": clean_dataset, "output": splits, "seed": 1})
        output = clean_dataset.parent / "coco_split"
        result = df.run(
            "export.coco",
            {"folder": splits, "output": output, "splits": "train,val,test"},
        )
        assert result.ok
        for name in ("train", "val", "test"):
            assert (output / "annotations" / f"instances_{name}.json").exists()

    def test_missing_split_folder_names_what_is_there(self, clean_dataset: Path):
        result = df.run(
            "export.coco",
            {"folder": clean_dataset, "output": clean_dataset.parent / "x", "splits": "nope"},
        )
        assert not result.ok and "not found" in result.errors[0]


class TestAudit:
    def test_finds_every_planted_defect(self, dataset: Path):
        result = df.run("inspect.dataset", {"folder": dataset, "check_duplicates": "exact"})
        assert result.counts["corrupt_labels"] == 1
        assert result.counts["duplicate_images"] == 1
        assert result.counts["background_images"] == 1
        assert result.data["case_variants"] == {
            "open": ["Open", "open"], "close": ["Close", "close"]
        }
        severities = {i["severity"] for i in result.data["issues"]}
        assert "high" in severities

    def test_out_of_bounds_boxes_are_reported(self, tmp_path: Path):
        folder = tmp_path / "geom"
        make_image(folder / "big.jpg", 1)
        write_voc(folder / "big.xml", "Open", bbox=(0, 0, 5000, 4000))
        result = df.run("inspect.dataset", {"folder": folder})
        assert result.counts["geometry_problems"] == 1
        assert "extends past" in result.data["geometry_problems"][0]["problem"]

    def test_zero_area_box_is_reported(self, tmp_path: Path):
        folder = tmp_path / "zero"
        make_image(folder / "z.jpg", 1)
        write_voc(folder / "z.xml", "Open", bbox=(10, 10, 10, 10))
        result = df.run("inspect.dataset", {"folder": folder})
        assert "zero or negative area" in result.data["geometry_problems"][0]["problem"]

    def test_size_mismatch_between_label_and_image(self, tmp_path: Path):
        folder = tmp_path / "size"
        make_image(folder / "s.jpg", 1)  # really 120x90
        write_voc(folder / "s.xml", "Open", bbox=(1, 1, 20, 20), size=(640, 480))
        result = df.run("inspect.dataset", {"folder": folder, "check_images": True})
        assert result.counts["size_mismatches"] == 1

    def test_a_clean_dataset_reports_no_problems(self, clean_dataset: Path):
        result = df.run("inspect.dataset", {"folder": clean_dataset})
        assert result.data["issues"] == [{"severity": "none", "issue": "no problems found", "fix": ""}]

    def test_auditing_never_writes(self, dataset: Path):
        before = {str(p.relative_to(dataset)) for p in dataset.rglob("*") if p.is_file()}
        df.run("inspect.dataset", {"folder": dataset, "check_duplicates": "exact"})
        assert {str(p.relative_to(dataset)) for p in dataset.rglob("*") if p.is_file()} == before

    def test_inspect_coco_flags_collisions(self, dataset: Path):
        output = dataset.parent / "coco"
        df.run("export.coco", {"folder": dataset, "output": output, "case_policy": "exact"})
        result = df.run(
            "inspect.coco", {"file": output / "annotations" / "instances.json"}
        )
        assert result.data["case_collisions"]
        assert any("differing only by case" in w for w in result.warnings)


class TestVisualize:
    def test_renders_a_copy_per_pair_without_touching_the_source(self, clean_dataset: Path):
        before = sorted(p.name for p in clean_dataset.iterdir())
        output = clean_dataset.parent / "viz"
        result = df.run("verify.visualize", {"folder": clean_dataset, "output": output})
        assert result.counts["rendered"] == 7
        assert len(list(output.glob("*.jpg"))) == 7
        assert sorted(p.name for p in clean_dataset.iterdir()) == before

    def test_split_by_type_separates_polygons_from_boxes(self, dataset: Path):
        output = dataset.parent / "viz2"
        result = df.run(
            "verify.visualize", {"folder": dataset, "output": output, "split_by_type": True}
        )
        assert result.counts["polygons"] == 2
        assert (output / "polygons").is_dir() and (output / "bbox_only").is_dir()

    def test_limit_caps_the_work(self, clean_dataset: Path):
        result = df.run(
            "verify.visualize",
            {"folder": clean_dataset, "output": clean_dataset.parent / "v3", "limit": 2},
        )
        assert result.counts["rendered"] == 2

    def test_dry_run_writes_nothing(self, clean_dataset: Path):
        output = clean_dataset.parent / "v4"
        result = df.run(
            "verify.visualize", {"folder": clean_dataset, "output": output}, Context(dry_run=True)
        )
        assert result.counts["rendered"] == 7
        assert not output.exists()
