"""Cleaning operations: dedup, orphans, renaming, and label edits."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from conftest import make_image, make_pair, make_similar_image, write_voc

import data_factory as df
from data_factory import Context


def snapshot(folder: Path) -> set[str]:
    return {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}


class TestDedup:
    def test_exact_duplicates_move_with_their_labels(self, dataset: Path):
        result = df.run("clean.dedup", {"folders": [dataset], "method": "exact"})
        assert result.counts["duplicates"] == 1
        # dup.jpg matched a.jpg, and dup.xml went with it.
        quarantined = {p.name for p in (dataset / "_duplicates").rglob("*") if p.is_file()}
        assert quarantined == {"dup.jpg", "dup.xml"}
        assert not (dataset / "dup.jpg").exists()
        assert (dataset / "a.jpg").exists()

    def test_dry_run_changes_nothing(self, dataset: Path):
        before = snapshot(dataset)
        result = df.run("clean.dedup", {"folders": [dataset]}, Context(dry_run=True))
        assert result.counts["duplicates"] == 1
        assert result.dry_run is True
        assert snapshot(dataset) == before

    def test_report_action_only_records(self, dataset: Path):
        before = snapshot(dataset)
        result = df.run("clean.dedup", {"folders": [dataset], "action": "report"})
        assert result.counts["duplicates"] == 1
        assert snapshot(dataset) == before
        assert all(a.action.value == "skip" for a in result.actions)

    def test_annotated_copy_is_the_one_kept(self, tmp_path: Path):
        folder = tmp_path / "d"
        folder.mkdir()
        # The bare copy sorts first, so a naive "keep the first" would discard
        # the annotated one and throw away real work.
        make_image(folder / "aaa_bare.jpg", 1)
        (folder / "zzz_labelled.jpg").write_bytes((folder / "aaa_bare.jpg").read_bytes())
        write_voc(folder / "zzz_labelled.xml", "Open")

        df.run("clean.dedup", {"folders": [folder], "keep": "annotated"})
        assert (folder / "zzz_labelled.jpg").exists()
        assert not (folder / "aaa_bare.jpg").exists()

    def test_keep_first_ignores_annotations(self, tmp_path: Path):
        folder = tmp_path / "d"
        folder.mkdir()
        make_image(folder / "aaa_bare.jpg", 1)
        (folder / "zzz_labelled.jpg").write_bytes((folder / "aaa_bare.jpg").read_bytes())
        write_voc(folder / "zzz_labelled.xml", "Open")

        df.run("clean.dedup", {"folders": [folder], "keep": "first"})
        assert (folder / "aaa_bare.jpg").exists()

    def test_cross_folder_deduplication(self, tmp_path: Path):
        first, second = tmp_path / "one", tmp_path / "two"
        make_image(first / "x.jpg", 1)
        (second / "y.jpg").parent.mkdir(parents=True, exist_ok=True)
        (second / "y.jpg").write_bytes((first / "x.jpg").read_bytes())

        result = df.run(
            "clean.dedup",
            {"folders": [first, second], "quarantine": tmp_path / "dupes"},
        )
        assert result.counts["duplicates"] == 1
        assert (first / "x.jpg").exists()
        assert not (second / "y.jpg").exists()

    def test_perceptual_matching_catches_a_near_duplicate(self, tmp_path: Path):
        folder = tmp_path / "p"
        folder.mkdir()
        make_image(folder / "a.jpg", 11)
        make_similar_image(folder / "a_bright.jpg", 11)
        make_image(folder / "different.jpg", 999)

        exact = df.run("clean.dedup", {"folders": [folder], "method": "exact", "action": "report"})
        assert exact.counts["duplicates"] == 0  # bytes differ

        perceptual = df.run(
            "clean.dedup",
            {"folders": [folder], "method": "phash", "threshold": 8, "action": "report"},
        )
        assert perceptual.counts["duplicates"] == 1

    def test_unreadable_image_is_warned_not_fatal(self, tmp_path: Path):
        folder = tmp_path / "u"
        folder.mkdir()
        make_image(folder / "good.jpg", 1)
        (folder / "bad.jpg").write_bytes(b"not an image")
        result = df.run(
            "clean.dedup", {"folders": [folder], "method": "phash", "action": "report"}
        )
        assert result.ok and result.counts["unreadable"] == 1


class TestOrphans:
    def test_report_lists_both_kinds_without_touching_anything(self, dataset: Path):
        before = snapshot(dataset)
        result = df.run("clean.orphans", {"folder": dataset, "action": "report"})
        assert result.counts["images_without_labels_handled"] == 1
        assert result.counts["labels_without_images_handled"] == 1
        assert snapshot(dataset) == before

    def test_move_sorts_orphans_into_labelled_buckets(self, dataset: Path):
        df.run("clean.orphans", {"folder": dataset, "action": "move"})
        assert (dataset / "_orphans" / "images_without_labels" / "lonely_image.jpg").exists()
        assert (dataset / "_orphans" / "labels_without_images" / "lonely_label.xml").exists()

    def test_scope_limits_which_kind_is_touched(self, dataset: Path):
        df.run("clean.orphans", {"folder": dataset, "scope": "labels", "action": "delete"})
        assert not (dataset / "lonely_label.xml").exists()
        assert (dataset / "lonely_image.jpg").exists()

    def test_unannotated_images_move_to_a_queue(self, dataset: Path):
        target = dataset.parent / "to_annotate"
        result = df.run("clean.unannotated", {"folder": dataset, "target": target})
        assert result.counts["unannotated"] == 1
        assert (target / "lonely_image.jpg").exists()

    def test_copy_mode_leaves_the_source_intact(self, dataset: Path):
        target = dataset.parent / "queue"
        df.run("clean.unannotated", {"folder": dataset, "target": target, "mode": "copy"})
        assert (target / "lonely_image.jpg").exists()
        assert (dataset / "lonely_image.jpg").exists()


class TestRename:
    def test_pairs_are_renamed_together(self, clean_dataset: Path):
        result = df.run("clean.rename", {"folder": clean_dataset, "prefix": "Door", "pad": 4})
        assert result.counts["pairs_renamed"] == 7
        assert (clean_dataset / "Door0001.jpg").exists()
        assert (clean_dataset / "Door0001.xml").exists()

    def test_overlapping_target_names_do_not_lose_files(self, tmp_path: Path):
        """Renaming b2,b3,b1 -> b1,b2,b3 must not clobber anything.

        The original script renamed each pair through its own temporary name,
        which still collides when a target belongs to a file later in the queue.
        """
        folder = tmp_path / "overlap"
        for index, stem in enumerate(["b2", "b3", "b1"], start=1):
            make_pair(folder, stem, "Open", index)

        result = df.run("clean.rename", {"folder": folder, "prefix": "b", "pad": 0, "start": 1})
        assert result.counts["pairs_renamed"] == 3
        assert sorted(p.name for p in folder.glob("*.jpg")) == ["b1.jpg", "b2.jpg", "b3.jpg"]
        assert len(list(folder.glob("*.xml"))) == 3
        assert not list(folder.glob(".dfx-rename-*"))

    def test_label_filename_is_retargeted(self, clean_dataset: Path):
        df.run("clean.rename", {"folder": clean_dataset, "prefix": "Door", "pad": 4})
        root = ET.parse(clean_dataset / "Door0001.xml").getroot()
        # The original left this pointing at the old image name.
        assert root.findtext("filename") == "Door0001.jpg"
        assert root.findtext("path") == "Door0001.jpg"

    def test_retargeting_can_be_switched_off(self, clean_dataset: Path):
        df.run(
            "clean.rename",
            {"folder": clean_dataset, "prefix": "Door", "pad": 4, "update_label_filename": False},
        )
        assert ET.parse(clean_dataset / "Door0001.xml").getroot().findtext("filename") != "Door0001.jpg"

    def test_conflict_with_an_uninvolved_file_is_skipped_and_warned(self, tmp_path: Path):
        folder = tmp_path / "conflict"
        make_pair(folder, "src", "Open", 1)
        make_image(folder / "Door0001.jpg", 2)  # already there, and not part of the rename

        result = df.run("clean.rename", {"folder": folder, "prefix": "Door", "pad": 4})
        assert result.counts.get("conflicts") == 1
        assert any("already exists" in w for w in result.warnings)
        assert (folder / "src.jpg").exists()

    def test_include_all_renames_unpaired_images_too(self, dataset: Path):
        result = df.run(
            "clean.rename", {"folder": dataset, "prefix": "X", "include": "all", "pad": 3}
        )
        assert result.counts["selected"] == 8  # 7 pairs + the lone image


class TestLabels:
    def test_list_counts_classes_and_flags_case_variants(self, dataset: Path):
        result = df.run("labels.list", {"folder": dataset})
        assert result.counts["corrupt"] == 1
        assert result.data["case_variants"] == {
            "open": ["Open", "open"],
            "close": ["Close", "close"],
        }
        assert result.counts["polygon_objects"] == 2

    def test_fold_case_unifies_on_the_most_common_spelling(self, tmp_path: Path):
        folder = tmp_path / "case"
        make_pair(folder, "a", "Open", 1)
        make_pair(folder, "b", "Open", 2)
        make_pair(folder, "c", "open", 3)  # the minority spelling

        result = df.run("labels.rename", {"folder": folder, "fold_case": True})
        assert result.counts["objects_renamed"] == 1
        assert df.run("labels.list", {"folder": folder}).data["case_variants"] == {}
        assert df.run("labels.list", {"folder": folder}).data["per_label"] == {"Open": 3}

    def test_explicit_map_wins_over_case_folding(self, tmp_path: Path):
        folder = tmp_path / "m"
        make_pair(folder, "a", "planter_View", 1)
        df.run("labels.rename", {"folder": folder, "map": "planter_View=Plantar_View"})
        assert df.run("labels.list", {"folder": folder}).data["per_label"] == {"Plantar_View": 1}

    def test_dropping_a_class_leaves_a_valid_background(self, tmp_path: Path):
        folder = tmp_path / "drop"
        make_pair(folder, "a", "Junk", 1)
        result = df.run("labels.rename", {"folder": folder, "drop": ["Junk"]})
        assert result.counts["objects_dropped"] == 1
        summary = df.run("labels.list", {"folder": folder})
        assert summary.counts["objects"] == 0
        assert summary.counts["background_images"] == 1

    def test_no_instructions_is_a_warning_not_a_crash(self, clean_dataset: Path):
        result = df.run("labels.rename", {"folder": clean_dataset})
        assert result.ok and any("nothing to do" in w for w in result.warnings)

    def test_dry_run_leaves_labels_untouched(self, tmp_path: Path):
        folder = tmp_path / "dry"
        make_pair(folder, "a", "open", 1)
        make_pair(folder, "b", "Open", 2)
        make_pair(folder, "c", "Open", 3)
        df.run("labels.rename", {"folder": folder, "fold_case": True}, Context(dry_run=True))
        assert df.run("labels.list", {"folder": folder}).data["per_label"] == {"Open": 2, "open": 1}
