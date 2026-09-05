"""Class-index stability, proved end to end through a real YOLO export.

``test_annotate_store.py`` proves the label table hands out stable indices.
This file proves the property survives all the way to the ``.txt`` files on
disk, which is where it actually matters -- the failure mode is not an
exception, it is a file that still parses and now means something else.

The shape of every test here is the same: export, change the project, export
again, and read the *first* export's files back with the *second* export's
class list. If the indices moved, the labels come back wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_factory.annotate import images, labels, projects, shapes
from data_factory.annotate.store import Store
from data_factory.core import registry
from data_factory.core.annotations import yolo
from data_factory.core.annotations.model import Shape
from data_factory.core.types import Context

import data_factory.core.ops  # noqa: F401  -- registers the operations


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "annotations.db")
    yield s
    s.close()


@pytest.fixture()
def project(store):
    return projects.create(store, "Doors")


def add_image(store, project, name, boxes, *, width=640, height=480, state="reviewed"):
    """Register an image and give it shapes, then mark it reviewed."""
    image = images.add(store, project, name, width=width, height=height)
    shapes.replace_all(
        store,
        project,
        image,
        [Shape(label=label, bbox=bbox) for label, bbox in boxes],
    )
    images.set_state(store, project, image, state)
    return image


def run_export(db: Path, project_id: str, output: Path, **overrides):
    params = {
        "database": str(db),
        "project": project_id,
        "output": str(output),
        "to": "yolo",
        "copy_images": False,
    }
    params.update(overrides)
    return registry.get("annotate.export").func(
        registry.get("annotate.export").bind(params), Context()
    )


def read_class_ids(output: Path) -> dict[str, list[int]]:
    """Map each label file's stem to the class ids it holds."""
    out: dict[str, list[int]] = {}
    for path in sorted((output / "labels").rglob("*.txt")):
        ids = [
            int(float(line.split()[0]))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        out[path.stem] = ids
    return out


def read_names(output: Path) -> list[str]:
    classes, _ = yolo.read_data_yaml(output / "data.yaml")
    return list(classes.names)


# ------------------------------------------------------- the regression test

def test_adding_a_class_between_exports_does_not_move_existing_indices(
    store, project, tmp_path
):
    """Export, add a class, export again. Monday's files must still parse
    to Monday's meanings under Friday's class list."""
    add_image(store, project, "a.jpg", [("Closed", (10, 10, 100, 100))])
    add_image(store, project, "b.jpg", [("Open", (20, 20, 120, 120))])

    monday = tmp_path / "monday"
    run_export(store.path, project.id, monday)
    monday_ids = read_class_ids(monday)
    monday_names = read_names(monday)

    # A class arrives mid-project -- and, to be adversarial about it, one that
    # sorts *before* both existing names. Under sorted() it would take index 0.
    add_image(store, project, "c.jpg", [("Ajar", (30, 30, 130, 130))])

    friday = tmp_path / "friday"
    run_export(store.path, project.id, friday)
    friday_names = read_names(friday)

    assert monday_names == ["Closed", "Open"]
    assert friday_names == ["Closed", "Open", "Ajar"], "a new class must be appended"

    # The decisive assertion: read Monday's files under Friday's names.
    for stem, ids in monday_ids.items():
        for class_id in ids:
            monday_meant = monday_names[class_id]
            friday_means = friday_names[class_id]
            assert monday_meant == friday_means, (
                f"{stem}.txt: index {class_id} meant {monday_meant!r} on Monday "
                f"and {friday_means!r} on Friday"
            )


def test_the_old_sorted_scheme_would_have_failed_that(store, project, tmp_path):
    """A guard on the test above: prove the scenario is actually dangerous.

    If sorting happened to give the same answer, the test above would pass
    without proving anything. This asserts the orders genuinely differ.
    """
    add_image(store, project, "a.jpg", [("Closed", (10, 10, 100, 100))])
    add_image(store, project, "b.jpg", [("Open", (20, 20, 120, 120))])
    add_image(store, project, "c.jpg", [("Ajar", (30, 30, 130, 130))])

    ordered = labels.class_list(store, project)
    assert ordered == ["Closed", "Open", "Ajar"]
    assert ordered != sorted(ordered), "the sorted order must differ, or this proves nothing"


def test_renaming_a_class_does_not_renumber_the_files(store, project, tmp_path):
    add_image(store, project, "a.jpg", [("Cosed", (10, 10, 100, 100))])
    add_image(store, project, "b.jpg", [("Open", (20, 20, 120, 120))])

    before_dir = tmp_path / "before"
    run_export(store.path, project.id, before_dir)
    before_ids = read_class_ids(before_dir)

    labels.rename(store, project, 0, "Closed")

    after_dir = tmp_path / "after"
    run_export(store.path, project.id, after_dir)

    assert read_names(after_dir) == ["Closed", "Open"]
    assert read_class_ids(after_dir) == before_ids, "a rename must not move any index"


def test_retiring_a_class_keeps_the_slot(store, project, tmp_path):
    add_image(store, project, "a.jpg", [("Closed", (10, 10, 100, 100))])
    add_image(store, project, "b.jpg", [("Handle", (20, 20, 60, 60))])
    add_image(store, project, "c.jpg", [("Open", (30, 30, 130, 130))])

    labels.retire(store, project, 1)  # Handle

    out = tmp_path / "out"
    run_export(store.path, project.id, out)
    names = read_names(out)

    assert names == ["Closed", "Handle", "Open"]
    assert names.index("Open") == 2, "retiring must not pull later classes down"


# ---------------------------------------------------------- what gets exported

def test_only_reviewed_images_are_exported_by_default(store, project, tmp_path):
    add_image(store, project, "done.jpg", [("Open", (10, 10, 100, 100))], state="reviewed")
    add_image(store, project, "raw.jpg", [("Open", (10, 10, 100, 100))], state="predicted")

    out = tmp_path / "out"
    result = run_export(store.path, project.id, out)

    assert result.ok
    assert result.counts["images"] == 1
    assert set(read_class_ids(out)) == {"done"}


def test_exporting_unreviewed_images_is_possible_but_warned_about(store, project, tmp_path):
    add_image(store, project, "done.jpg", [("Open", (10, 10, 100, 100))], state="reviewed")
    add_image(store, project, "raw.jpg", [("Open", (10, 10, 100, 100))], state="predicted")

    out = tmp_path / "out"
    result = run_export(store.path, project.id, out, states="reviewed,predicted")

    assert result.ok
    assert result.counts["images"] == 2
    assert any("not been checked by a person" in w for w in result.warnings)


def test_skipped_images_are_never_exported_by_default(store, project, tmp_path):
    add_image(store, project, "a.jpg", [("Open", (10, 10, 100, 100))], state="reviewed")
    add_image(store, project, "unclear.jpg", [("Open", (1, 1, 9, 9))], state="skipped")

    out = tmp_path / "out"
    run_export(store.path, project.id, out)
    assert set(read_class_ids(out)) == {"a"}


def test_a_background_image_exports_as_an_empty_file(store, project, tmp_path):
    add_image(store, project, "a.jpg", [("Open", (10, 10, 100, 100))])
    add_image(store, project, "empty.jpg", [])

    out = tmp_path / "out"
    result = run_export(store.path, project.id, out)

    assert result.counts["backgrounds"] == 1
    assert read_class_ids(out)["empty"] == [], "a hard negative is an empty file, not a missing one"


def test_an_unknown_state_is_refused(store, project, tmp_path):
    add_image(store, project, "a.jpg", [("Open", (10, 10, 100, 100))])
    result = run_export(store.path, project.id, tmp_path / "out", states="finished")
    assert not result.ok
    assert "unknown image state" in result.errors[0]


def test_a_project_with_no_labels_fails_clearly(store, project, tmp_path):
    images.add(store, project, "a.jpg", width=640, height=480)
    result = run_export(store.path, project.id, tmp_path / "out")
    assert not result.ok
    assert "no labels yet" in result.errors[0]


def test_another_owners_project_is_not_exportable(store, tmp_path):
    mine = projects.create(store, "mine", owner_id="alice")
    add_image(store, mine, "a.jpg", [("Open", (10, 10, 100, 100))])
    result = run_export(store.path, mine.id, tmp_path / "out", owner="bob")
    assert not result.ok
    assert "no such project" in result.errors[0]


# ------------------------------------------------------------ other formats

@pytest.mark.parametrize("fmt", ["voc", "coco", "labelme", "csv"])
def test_every_format_exports_from_a_project(store, project, tmp_path, fmt):
    add_image(store, project, "a.jpg", [("Open", (10, 10, 100, 100))])
    result = run_export(store.path, project.id, tmp_path / fmt, to=fmt)
    assert result.ok, result.errors
    assert result.counts["objects"] == 1


def test_polygons_survive_the_round_trip(store, project, tmp_path):
    image = images.add(store, project, "poly.jpg", width=640, height=480)
    shapes.replace_all(
        store,
        project,
        image,
        [Shape(label="Open", bbox=(10, 10, 100, 100), polygon=[(10, 10), (100, 10), (55, 100)])],
    )
    images.set_state(store, project, image, "reviewed")

    out = tmp_path / "out"
    run_export(store.path, project.id, out)
    text = (out / "labels" / "train" / "poly.txt").read_text(encoding="utf-8")
    # A polygon line carries three points, so six coordinates after the class.
    assert len(text.split()) == 7


# ---------------------------------------------------------------- the web gate

def test_annotate_export_is_hidden_from_the_web_form_list():
    from data_factory.service import binding

    names = [op["name"] for op in binding.public_schema()["operations"]]
    assert "annotate.export" not in names
    assert "convert.dataset" in names, "only the annotation ops are hidden"


def test_annotate_export_is_still_a_normal_operation_elsewhere():
    assert registry.get("annotate.export").hidden_from_web is True
    assert "annotate.export" in [op.name for op in registry.all_operations()]
    assert "annotate.export" in [op["name"] for op in registry.schema()["operations"]]
