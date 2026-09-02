"""Registry, parameter coercion, BK-tree, and pair indexing."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from data_factory.core import registry
from data_factory.core.bktree import BKTree
from data_factory.core.pairs import PairIndex
from data_factory.core.registry import Param, ParamError
from data_factory.core.types import Action, Context, FileAction, OpResult


class TestParamCoercion:
    def test_int_range_is_enforced(self):
        param = Param("threshold", "int", "h", minimum=0, maximum=10)
        assert param.coerce("7") == 7
        with pytest.raises(ParamError):
            param.coerce("11")

    def test_bool_accepts_the_usual_spellings(self):
        param = Param("flag", "bool", "h")
        assert all(param.coerce(v) is True for v in ("1", "true", "YES", "on", True))
        assert all(param.coerce(v) is False for v in ("0", "false", "No", "off", False))
        with pytest.raises(ParamError):
            param.coerce("maybe")

    def test_choice_rejects_unknown_values(self):
        param = Param("mode", "choice", "h", choices=("a", "b"))
        assert param.coerce("a") == "a"
        with pytest.raises(ParamError, match="not one of"):
            param.coerce("c")

    def test_mapping_from_string_and_dict(self):
        param = Param("map", "mapping", "h")
        assert param.coerce("a=b,c=d") == {"a": "b", "c": "d"}
        assert param.coerce({"a": "b"}) == {"a": "b"}
        with pytest.raises(ParamError, match="old=new"):
            param.coerce("nope")

    def test_scalar_list_splits_but_a_real_list_is_left_alone(self):
        param = Param("names", "strs", "h")
        assert param.coerce("train,val") == ["train", "val"]
        # A list arrives from argparse nargs or YAML already split; splitting it
        # again would corrupt any element that legitimately contains a comma.
        assert param.coerce(["a,b", "c"]) == ["a,b", "c"]

    def test_must_exist_can_be_deferred(self, tmp_path: Path):
        param = Param("folder", "dir", "h", must_exist=True)
        missing = tmp_path / "later"
        with pytest.raises(ParamError, match="does not exist"):
            param.coerce(missing)
        # A recipe validates before the producing step has run.
        assert param.coerce(missing, check_exists=False) == missing


class TestOperationBinding:
    def test_defaults_are_applied_and_unknown_params_rejected(self):
        op = registry.get("clean.dedup")
        bound = op.bind({"folders": ["."]})
        assert bound["method"] == "exact"
        assert bound["threshold"] == 8
        with pytest.raises(ParamError, match="unknown parameter"):
            op.bind({"folders": ["."], "nonsense": 1})

    def test_missing_required_is_reported_by_name(self):
        op = registry.get("clean.dedup")
        with pytest.raises(ParamError, match="folders"):
            op.bind({})

    def test_destructiveness_depends_on_the_parameters(self):
        op = registry.get("clean.dedup")
        assert op.destructive is True
        assert op.destructive_for(op.bind({"folders": ["."], "action": "move"})) is True
        # Reporting writes nothing, so it must not demand confirmation.
        assert op.destructive_for(op.bind({"folders": ["."], "action": "report"})) is False

    def test_schema_covers_every_operation(self):
        schema = registry.schema()
        assert len(schema["operations"]) == len(registry.all_operations())
        for entry in schema["operations"]:
            assert entry["name"] and entry["summary"]
            for param in entry["params"]:
                assert param["kind"] in registry.PARAM_KINDS


class TestBKTree:
    def test_matches_brute_force(self):
        """The tree must return exactly what an exhaustive scan would."""

        def hamming(a: int, b: int) -> int:
            return bin(a ^ b).count("1")

        rng = random.Random(0)
        values = [rng.getrandbits(64) for _ in range(400)]
        tree = BKTree(hamming)
        for value in values:
            tree.add(value)

        for radius in (0, 3, 8, 16):
            query = rng.getrandbits(64)
            expected = sorted(v for v in values if hamming(query, v) <= radius)
            actual = sorted(item for _, item in tree.find(query, radius))
            assert actual == expected, f"radius {radius}"

    def test_empty_tree_finds_nothing(self):
        assert BKTree(lambda a, b: 0).find("x", 5) == []


class TestPairIndex:
    def test_pairs_orphans_and_backgrounds(self, dataset: Path):
        index = PairIndex.scan(dataset)
        summary = index.summary()
        assert summary["complete"] == 7
        assert summary["images_without_labels"] == 1
        assert summary["labels_without_images"] == 1

    def test_recursive_scan_keeps_same_named_files_in_different_folders_apart(
        self, tmp_path: Path
    ):
        from conftest import make_pair

        make_pair(tmp_path / "train", "img", "Open", 1)
        make_pair(tmp_path / "val", "img", "Open", 2)
        index = PairIndex.scan(tmp_path, recursive=True)
        # Keyed on sub-folder + stem, so train/img and val/img stay distinct.
        assert len(index.complete()) == 2
        assert {p.key for p in index} == {"train/img", "val/img"}

    def test_extension_choice_is_deterministic(self, tmp_path: Path):
        from conftest import make_image

        make_image(tmp_path / "x.png", 1)
        make_image(tmp_path / "x.jpg", 2)
        # .jpg outranks .png, so the winner does not depend on listing order.
        assert PairIndex.scan(tmp_path).pairs[0].image.suffix == ".jpg"


class TestOpResult:
    def test_serialises_paths_for_json(self, tmp_path: Path):
        result = OpResult(op="demo")
        result.record(Action.MOVE, tmp_path / "a", tmp_path / "b", "why")
        result.bump("moved")
        result.data["output"] = tmp_path
        document = result.as_dict()
        assert isinstance(document["data"]["output"], str)
        assert document["actions"][0]["action"] == "move"
        assert document["counts"]["moved"] == 1

    def test_cancellation_is_cooperative(self):
        ctx = Context(cancel_check=lambda: True)
        with pytest.raises(Exception, match="cancelled"):
            ctx.check_cancelled()
