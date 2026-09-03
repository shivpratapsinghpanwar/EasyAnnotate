"""Parallelism and provenance.

Both make a claim that has to be checked rather than assumed: that going
parallel did not make results depend on thread timing, and that lineage is
recorded without any operation having to remember to do it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_image

import data_factory as df
from data_factory.core import parallel, provenance
from data_factory.core.types import Cancelled, Context


class TestParallel:
    def test_results_come_back_in_input_order(self):
        """Not completion order -- dedup keeps the first image it sees."""
        items = list(range(50))

        def work(n: int) -> int:
            # Deliberately make later items finish first if order were lost.
            return n * 2

        out = list(parallel.map_ordered(items, work, ctx=Context(), workers=8))
        assert [item for item, _ in out] == items
        assert [value for _, value in out] == [n * 2 for n in items]

    def test_serial_and_parallel_agree(self):
        items = [f"item{i}" for i in range(30)]
        work = len
        serial = list(parallel.map_ordered(items, work, ctx=Context(), workers=1))
        threaded = list(parallel.map_ordered(items, work, ctx=Context(), workers=6))
        assert serial == threaded

    def test_one_failure_does_not_abort_the_rest(self):
        def work(n: int) -> int:
            if n == 3:
                raise ValueError("bad one")
            return n

        seen: list[tuple[int, Exception]] = []
        out = list(
            parallel.map_ordered(
                list(range(6)), work, ctx=Context(), workers=4,
                on_error=lambda item, exc: seen.append((item, exc)),
            )
        )
        assert [value for _, value in out] == [0, 1, 2, None, 4, 5]
        assert len(seen) == 1 and seen[0][0] == 3

    def test_progress_is_reported_once_per_item(self):
        calls: list[int] = []
        ctx = Context(progress=lambda done, total, message="": calls.append(done))
        list(parallel.map_ordered(list(range(10)), lambda n: n, ctx=ctx, workers=4))
        assert calls == list(range(1, 11))

    def test_cancellation_stops_the_run(self):
        state = {"count": 0}

        def cancel_check() -> bool:
            state["count"] += 1
            return state["count"] > 3

        ctx = Context(cancel_check=cancel_check)
        with pytest.raises(Cancelled):
            list(parallel.map_ordered(list(range(100)), lambda n: n, ctx=ctx, workers=4))

    def test_worker_count_is_resolved_sensibly(self):
        assert parallel.resolve_workers(4) == 4
        assert 1 <= parallel.resolve_workers(0) <= parallel.MAX_WORKERS
        assert parallel.resolve_workers(999) == 32  # capped, not obeyed

    def test_an_empty_input_yields_nothing(self):
        assert list(parallel.map_ordered([], lambda n: n, ctx=Context())) == []

    def test_collect_drops_the_failures(self):
        def work(n: int) -> int | None:
            return None if n == 1 else n

        out = parallel.collect([0, 1, 2], work, ctx=Context(), workers=2)
        assert [item for item, _ in out] == [0, 2]


class TestDedupStaysDeterministic:
    """Parallel hashing must not change which duplicate is kept."""

    def test_the_same_duplicates_are_found_at_any_worker_count(self, tmp_path: Path):
        folder = tmp_path / "d"
        for index in range(8):
            make_image(folder / f"img{index}.jpg", index)
        # Two exact copies of the first image.
        (folder / "copy_a.jpg").write_bytes((folder / "img0.jpg").read_bytes())
        (folder / "copy_b.jpg").write_bytes((folder / "img0.jpg").read_bytes())

        results = []
        for workers in (1, 2, 8):
            result = df.get_operation("clean.dedup").run(
                {"folders": [str(folder)], "method": "exact", "action": "report",
                 "workers": workers},
                Context(dry_run=True),
            )
            results.append((result.counts["duplicates"], sorted(result.data["groups"])))

        assert results[0][0] == 2
        assert results[0] == results[1] == results[2]


class TestProvenance:
    def test_an_operation_records_what_it_did(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "yolo"
        df.run("convert.dataset", {"folder": str(clean_dataset), "output": str(out), "to": "yolo"})

        entries = provenance.read(out)
        assert len(entries) == 1
        assert entries[0]["op"] == "convert.dataset"
        assert entries[0]["params"]["to"] == "yolo"
        assert entries[0]["counts"]["images"] == 7
        assert entries[0]["at"]

    def test_steps_accumulate_in_order(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "built"
        df.run("convert.dataset", {"folder": str(clean_dataset), "output": str(out), "to": "voc"})
        df.run("image.resize", {"folder": str(out), "output": str(tmp_path / "r"),
                                "width": 64, "height": 64})
        df.run("augment.dataset", {"folder": str(tmp_path / "r"), "output": str(tmp_path / "a"),
                                   "multiplier": 1, "flip_horizontal": True, "seed": 9})

        final = provenance.read(tmp_path / "a")
        assert [e["op"] for e in final] == ["augment.dataset"]
        assert final[0]["params"]["seed"] == 9

    def test_the_seed_is_recorded_even_when_it_was_the_default(
        self, clean_dataset: Path, tmp_path: Path
    ):
        """'What seed was this built at' must be answerable without also
        knowing what the default was in the version that built it."""
        out = tmp_path / "s"
        df.run("dataset.sample", {"folder": str(clean_dataset), "output": str(out), "count": 2})
        assert provenance.read(out)[0]["params"]["seed"] == 42

    def test_a_dry_run_records_nothing(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "o"
        df.run(
            "convert.dataset",
            {"folder": str(clean_dataset), "output": str(out), "to": "yolo"},
            Context(dry_run=True),
        )
        assert provenance.read(out) == []

    def test_a_failed_run_records_nothing(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "o"
        out.mkdir()
        result = df.run(
            "augment.dataset",
            {"folder": str(clean_dataset), "output": str(out), "flip_horizontal": False,
             "brightness": 0.0, "contrast": 0.0},
        )
        assert not result.ok
        assert provenance.read(out) == []

    def test_read_only_operations_record_nothing(self, clean_dataset: Path):
        df.run("inspect.health", {"folder": str(clean_dataset)})
        assert provenance.read(clean_dataset) == []

    def test_the_file_is_valid_json_lines(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "o"
        df.run("dataset.sample", {"folder": str(clean_dataset), "output": str(out), "count": 2})
        df.run("dataset.sample", {"folder": str(clean_dataset), "output": str(out), "count": 3})

        text = provenance.history_path(out).read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) == 2
        for line in lines:
            assert json.loads(line)["op"] == "dataset.sample"

    def test_a_truncated_line_is_skipped_not_fatal(self, tmp_path: Path):
        out = tmp_path / "o"
        out.mkdir()
        path = provenance.history_path(out)
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"op": "a"}) + "\n" + '{"op": "b", "trunc' + "\n",
            encoding="utf-8",
        )
        assert [e["op"] for e in provenance.read(out)] == ["a"]

    def test_history_is_readable_through_the_operation(self, clean_dataset: Path, tmp_path: Path):
        out = tmp_path / "o"
        df.run("convert.dataset", {"folder": str(clean_dataset), "output": str(out), "to": "coco"})
        result = df.run("inspect.history", {"folder": str(out)})
        assert result.counts["steps"] == 1
        assert result.data["operations"] == ["convert.dataset"]

    def test_a_folder_with_no_history_warns_rather_than_failing(self, clean_dataset: Path):
        result = df.run("inspect.history", {"folder": str(clean_dataset)})
        assert result.ok
        assert result.counts["steps"] == 0
        assert any("no history" in w for w in result.warnings)

    def test_the_history_folder_does_not_confuse_format_detection(
        self, clean_dataset: Path, tmp_path: Path
    ):
        """.datafactory/history.jsonl sits inside the dataset it describes."""
        from data_factory.core.annotations import formats

        out = tmp_path / "o"
        df.run("convert.dataset", {"folder": str(clean_dataset), "output": str(out), "to": "voc"})
        assert provenance.history_path(out).exists()
        assert formats.detect(out) == "voc"
        assert formats.read_dataset(out, "auto").stats()["images"] == 7
