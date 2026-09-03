"""The recipe runner and the command line -- the two front-ends over the core."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml
from conftest import make_pair

from data_factory.cli.main import main
from data_factory.core.types import Context
from data_factory.pipeline import Recipe, RecipeError, run


def write_recipe(path: Path, document: dict) -> Path:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


class TestRecipeLoading:
    def test_variables_are_substituted(self, tmp_path: Path, clean_dataset: Path):
        path = write_recipe(
            tmp_path / "r.yaml",
            {
                "name": "demo",
                "vars": {"root": str(clean_dataset)},
                "steps": [{"op": "labels.list", "with": {"folder": "${root}"}}],
            },
        )
        recipe = Recipe.load(path)
        assert recipe.steps[0].params["folder"] == str(clean_dataset)

    def test_overrides_beat_the_file(self, tmp_path: Path, clean_dataset: Path):
        path = write_recipe(
            tmp_path / "r.yaml",
            {
                "vars": {"root": "/not/used"},
                "steps": [{"op": "labels.list", "with": {"folder": "${root}"}}],
            },
        )
        recipe = Recipe.load(path, overrides={"root": str(clean_dataset)})
        assert recipe.steps[0].params["folder"] == str(clean_dataset)

    def test_unknown_operation_is_rejected_at_load_time(self, tmp_path: Path):
        path = write_recipe(tmp_path / "r.yaml", {"steps": [{"op": "clean.nonsense"}]})
        with pytest.raises(RecipeError, match="unknown operation"):
            Recipe.load(path)

    def test_bad_parameter_is_rejected_at_load_time(self, tmp_path: Path):
        path = write_recipe(
            tmp_path / "r.yaml",
            {"steps": [{"op": "clean.dedup", "with": {"folders": ["."], "method": "wrong"}}]},
        )
        with pytest.raises(RecipeError, match="not one of"):
            Recipe.load(path)

    def test_paths_produced_by_earlier_steps_are_not_required_to_exist_yet(
        self, tmp_path: Path, clean_dataset: Path
    ):
        """Step 2 reads what step 1 creates, so load-time must not check existence."""
        path = write_recipe(
            tmp_path / "r.yaml",
            {
                "vars": {"src": str(clean_dataset), "out": str(tmp_path / "built")},
                "steps": [
                    {"op": "dataset.split", "with": {"folder": "${src}", "output": "${out}/splits"}},
                    {"op": "export.coco",
                     "with": {"folder": "${out}/splits", "output": "${out}/coco",
                              "splits": "train,val,test"}},
                ],
            },
        )
        recipe = Recipe.load(path)  # must not raise
        assert len(recipe.steps) == 2

    def test_a_recipe_needs_steps(self, tmp_path: Path):
        path = write_recipe(tmp_path / "r.yaml", {"name": "empty"})
        with pytest.raises(RecipeError, match="non-empty"):
            Recipe.load(path)

    def test_a_string_path_is_accepted(self, tmp_path: Path, clean_dataset: Path):
        """Recipe.load("build.yaml") is what a library caller writes."""
        path = write_recipe(
            tmp_path / "r.yaml",
            {"steps": [{"op": "labels.list", "with": {"folder": str(clean_dataset)}}]},
        )
        assert len(Recipe.load(str(path)).steps) == 1

    def test_the_shipped_recipes_are_valid(self):
        """They are documentation people copy, so they must actually bind."""
        root = Path(__file__).resolve().parent.parent / "recipes"
        found = sorted(root.glob("*.yaml"))
        assert found, "no recipes shipped"
        for recipe in found:
            loaded = Recipe.load(recipe, overrides={"root": ".", "out": "./build"})
            assert loaded.steps, recipe.name

    def test_json_recipes_work_without_yaml(self, tmp_path: Path, clean_dataset: Path):
        path = tmp_path / "r.json"
        path.write_text(
            json.dumps({"steps": [{"op": "labels.list", "with": {"folder": str(clean_dataset)}}]}),
            encoding="utf-8",
        )
        assert len(Recipe.load(path).steps) == 1


class TestRecipeRunning:
    def test_steps_run_in_order_and_produce_the_dataset(
        self, tmp_path: Path, clean_dataset: Path
    ):
        out = tmp_path / "built"
        path = write_recipe(
            tmp_path / "r.yaml",
            {
                "name": "build",
                "vars": {"src": str(clean_dataset), "out": str(out)},
                "steps": [
                    {"name": "audit", "op": "inspect.dataset", "with": {"folder": "${src}"}},
                    {"op": "labels.rename", "with": {"folder": "${src}", "fold_case": True}},
                    {"op": "dataset.split",
                     "with": {"folder": "${src}", "output": "${out}/splits", "seed": 5}},
                    {"op": "export.coco",
                     "with": {"folder": "${out}/splits", "output": "${out}/coco",
                              "splits": "train,val,test"}},
                ],
            },
        )
        report = run(Recipe.load(path))
        assert report.ok
        assert len(report.results) == 4
        assert (out / "coco" / "annotations" / "instances_train.json").exists()

    def test_a_failing_step_stops_the_run(self, tmp_path: Path, clean_dataset: Path):
        path = write_recipe(
            tmp_path / "r.yaml",
            {
                "steps": [
                    {"name": "bad split", "op": "dataset.split",
                     "with": {"folder": str(clean_dataset), "output": str(tmp_path / "o"),
                              "ratios": "1,1", "names": "a,b,c"}},
                    {"op": "labels.list", "with": {"folder": str(clean_dataset)}},
                ]
            },
        )
        report = run(Recipe.load(path))
        assert not report.ok
        assert report.stopped_at == "bad split"
        assert len(report.results) == 1  # the second step never ran

    def test_continue_on_error_runs_the_rest(self, tmp_path: Path, clean_dataset: Path):
        path = write_recipe(
            tmp_path / "r.yaml",
            {
                "steps": [
                    {"op": "dataset.split",
                     "with": {"folder": str(clean_dataset), "output": str(tmp_path / "o"),
                              "ratios": "1,1", "names": "a,b,c"}},
                    {"op": "labels.list", "with": {"folder": str(clean_dataset)}},
                ]
            },
        )
        report = run(Recipe.load(path), continue_on_error=True)
        assert len(report.results) == 2

    def test_skip_leaves_a_step_out(self, tmp_path: Path, clean_dataset: Path):
        path = write_recipe(
            tmp_path / "r.yaml",
            {
                "steps": [
                    {"op": "labels.list", "with": {"folder": str(clean_dataset)}, "skip": True},
                    {"op": "labels.list", "with": {"folder": str(clean_dataset)}},
                ]
            },
        )
        assert len(run(Recipe.load(path)).results) == 1

    def test_dry_run_propagates_to_every_step(self, tmp_path: Path, clean_dataset: Path):
        before = sorted(p.name for p in clean_dataset.iterdir())
        path = write_recipe(
            tmp_path / "r.yaml",
            {
                "steps": [
                    {"op": "clean.rename", "with": {"folder": str(clean_dataset), "prefix": "Z"}},
                ]
            },
        )
        report = run(Recipe.load(path), Context(dry_run=True))
        assert report.ok
        assert sorted(p.name for p in clean_dataset.iterdir()) == before


class TestCommandLine:
    def _capture(self, argv: list[str], monkeypatch) -> tuple[int, str]:
        out = io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        monkeypatch.setattr("sys.stderr", out)
        code = main(argv)
        return code, out.getvalue()

    def test_list_shows_every_operation(self, monkeypatch):
        code, text = self._capture(["list"], monkeypatch)
        assert code == 0
        assert "clean.dedup" in text and "export.coco" in text

    def test_explain_prints_parameters_and_notes(self, monkeypatch):
        code, text = self._capture(["explain", "clean.dedup"], monkeypatch)
        assert code == 0
        assert "--threshold" in text and "keep=annotated" in text

    def test_explain_unknown_operation_fails_cleanly(self, monkeypatch):
        code, text = self._capture(["explain", "nope"], monkeypatch)
        assert code == 2 and "unknown operation" in text

    def test_schema_is_valid_json_covering_all_operations(self, monkeypatch, tmp_path: Path):
        target = tmp_path / "schema.json"
        code, _ = self._capture(["schema", "--output", str(target)], monkeypatch)
        assert code == 0
        document = json.loads(target.read_text(encoding="utf-8"))
        assert document["version"] == 1
        names = {op["name"] for op in document["operations"]}
        assert {"clean.dedup", "dataset.split", "export.coco"} <= names
        # This is what a web front-end builds its forms from.
        dedup = next(op for op in document["operations"] if op["name"] == "clean.dedup")
        folders = next(p for p in dedup["params"] if p["name"] == "folders")
        assert folders["required"] is True and folders["kind"] == "dirs"

    def test_running_an_operation_reports_counts(self, monkeypatch, clean_dataset: Path):
        code, text = self._capture(
            ["labels.list", "--folder", str(clean_dataset)], monkeypatch
        )
        assert code == 0
        assert "classes" in text and "Open" in text

    def test_json_output_is_machine_readable(self, monkeypatch, clean_dataset: Path):
        code, text = self._capture(
            ["labels.list", "--folder", str(clean_dataset), "--json"], monkeypatch
        )
        assert code == 0
        document = json.loads(text)
        assert document["op"] == "labels.list"
        assert document["counts"]["classes"] == 2

    def test_missing_required_parameter_exits_two(self, monkeypatch):
        code, text = self._capture(["clean.dedup"], monkeypatch)
        assert code == 2 and "missing required parameter" in text

    def test_dry_run_needs_no_confirmation_and_changes_nothing(
        self, monkeypatch, clean_dataset: Path
    ):
        before = sorted(p.name for p in clean_dataset.iterdir())
        code, text = self._capture(
            ["clean.rename", "--folder", str(clean_dataset), "--prefix", "Q", "--dry-run"],
            monkeypatch,
        )
        assert code == 0
        assert "dry run" in text
        assert sorted(p.name for p in clean_dataset.iterdir()) == before

    def test_report_action_needs_no_confirmation(self, monkeypatch, clean_dataset: Path):
        """--action report writes nothing, so it must not prompt."""
        code, _ = self._capture(
            ["clean.orphans", "--folder", str(clean_dataset), "--action", "report"], monkeypatch
        )
        assert code == 0

    def test_negated_boolean_flag(self, monkeypatch, tmp_path: Path):
        folder = tmp_path / "n"
        make_pair(folder / "sub", "deep", "Open", 1)
        code, text = self._capture(
            ["labels.list", "--folder", str(folder), "--no-recursive", "--json"], monkeypatch
        )
        assert code == 0
        # recursive defaults to True for labels.list; --no-recursive turns it off.
        assert json.loads(text)["counts"]["label_files"] == 0
