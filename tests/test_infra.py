"""Parallelism and provenance.

Both make a claim that has to be checked rather than assumed: that going
parallel did not make results depend on thread timing, and that lineage is
recorded without any operation having to remember to do it.
"""

from __future__ import annotations

import json
import time
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


class TestAtomicWrites:
    """A label file must never be observed half-written.

    The failure this prevents is silent and destructive: `write_text` truncates
    first and fills second, so a crash or a full disk between the two leaves an
    unparseable file *and* the original contents are already gone. Two writers
    at once interleave the same way. The annotation editor writes through to
    sidecar files on every save, so this is now on a hot path.
    """

    def test_the_file_is_written(self, tmp_path: Path):
        from data_factory.core.fs import write_text_atomic

        target = tmp_path / "nested" / "a.txt"
        write_text_atomic(target, "hello")
        assert target.read_text(encoding="utf-8") == "hello"

    def test_no_temporary_file_is_left_behind(self, tmp_path: Path):
        from data_factory.core.fs import write_text_atomic

        write_text_atomic(tmp_path / "a.txt", "hello")
        assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]

    def test_a_failed_write_leaves_the_original_intact(self, tmp_path: Path, monkeypatch):
        """The whole point. A partial write must not destroy what was there."""
        from data_factory.core import fs

        target = tmp_path / "a.txt"
        target.write_text("the original", encoding="utf-8")

        def explode(*_args, **_kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(fs.os, "replace", explode)

        with pytest.raises(OSError):
            fs.write_text_atomic(target, "replacement")

        assert target.read_text(encoding="utf-8") == "the original"
        assert list(tmp_path.iterdir()) == [target], "the temp file must be cleaned up"

    def test_a_rename_windows_transiently_refuses_is_retried(self, tmp_path: Path, monkeypatch):
        """Windows fails the rename if anything has the destination open --
        an antivirus scanner opening the file we just wrote is enough. Found
        by the concurrency test below, which failed within a few hundred
        iterations before this retry existed."""
        from data_factory.core import fs

        target = tmp_path / "a.txt"
        real_replace = fs.os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(5, "Access is denied")
            return real_replace(src, dst)

        monkeypatch.setattr(fs.os, "replace", flaky)
        monkeypatch.setattr(fs, "_REPLACE_PAUSE", 0.001)

        fs.write_text_atomic(target, "written anyway")
        assert target.read_text(encoding="utf-8") == "written anyway"
        assert calls["n"] == 3

    def test_a_rename_that_never_succeeds_still_raises(self, tmp_path: Path, monkeypatch):
        """Retrying must not turn a real failure into a silent one."""
        from data_factory.core import fs

        def always_denied(*_args, **_kwargs):
            raise PermissionError(5, "Access is denied")

        monkeypatch.setattr(fs.os, "replace", always_denied)
        monkeypatch.setattr(fs, "_REPLACE_PAUSE", 0.001)

        with pytest.raises(PermissionError):
            fs.write_text_atomic(tmp_path / "a.txt", "never lands")
        assert list(tmp_path.iterdir()) == []

    def test_concurrent_writers_never_produce_a_mixed_file(self, tmp_path: Path):
        """Either writer may win. Neither may produce a blend of the two."""
        import threading

        from data_factory.core.fs import write_text_atomic

        target = tmp_path / "busy.txt"
        first = "A" * 40_000
        second = "B" * 40_000
        seen: list[str] = []

        def read_eventually() -> str | None:
            # Windows refuses the *read* too, while a rename is swapping the
            # file in. Unlike a torn write this is loud and transient, so it is
            # tolerated here rather than retried inside every format reader.
            # Worth knowing: running `convert.dataset` over a folder while the
            # editor is writing through to it can fail this way, visibly.
            for _ in range(50):
                try:
                    return target.read_text(encoding="utf-8")
                except (PermissionError, OSError):
                    time.sleep(0.002)
            return None

        def writer(payload: str) -> None:
            for _ in range(20):
                write_text_atomic(target, payload)
                content = read_eventually()
                if content is not None:
                    seen.append(content)

        threads = [threading.Thread(target=writer, args=(p,)) for p in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert seen, "the test did not actually read anything"
        for content in seen:
            assert content in (first, second), "a reader saw a half-written file"

    def test_voc_and_labelme_writers_are_atomic(self, tmp_path: Path):
        """Guard against the writers being changed back to write_text."""
        import inspect

        from data_factory.core.annotations import coco, csvfmt, labelme, voc, yolo

        for module in (voc, labelme, yolo, coco, csvfmt):
            source = inspect.getsource(module)
            assert "write_text_atomic" in source, f"{module.__name__} lost its atomic write"
            assert "path.write_text(" not in source, f"{module.__name__} writes non-atomically"

    def test_csv_does_not_gain_blank_lines_between_rows(self, tmp_path: Path):
        """csv writes its own \r\n; translating those again doubles them."""
        from data_factory.core.annotations import csvfmt
        from data_factory.core.annotations.model import Annotation, Shape

        path = tmp_path / "out.csv"
        csvfmt.write(
            [
                Annotation(filename="a.jpg", width=10, height=10,
                           shapes=[Shape(label="x", bbox=(1, 1, 5, 5))]),
                Annotation(filename="b.jpg", width=10, height=10,
                           shapes=[Shape(label="y", bbox=(2, 2, 6, 6))]),
            ],
            path,
        )
        raw = path.read_bytes()
        assert b"\r\r\n" not in raw
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) == 3, rows
