"""Nothing is lost, nothing is wrong: the M0 fixes in docs/ROADMAP.md.

Each test here reproduces a way the backend used to lose or corrupt
annotation work without saying so, and failed before its fix.
"""

from __future__ import annotations

import sqlite3

import pytest

from conftest import make_image, write_voc
from data_factory.annotate import attach, images, labels, projects, review, shapes
from data_factory.annotate import store as store_mod
from data_factory.annotate.store import Conflict, Store
from data_factory.core.annotations.model import Shape


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "annotations.db")
    yield s
    s.close()


@pytest.fixture()
def project(store):
    return projects.create(store, "Doors")


# ------------------------------------------------------------- save race


def test_a_save_from_a_stale_read_is_refused(store, project):
    """Two requests read the image, both at version 0; the first saves.

    The second used to pass its version check against the copy it had read,
    then diff against the shapes as they were -- deleting the first save's
    box without a 409. The check now happens inside the write transaction.
    """
    first_view = images.add(store, project, "a.jpg", width=100, height=100)
    second_view = images.get(store, project, first_view.id)

    shapes.save(
        store, project, first_view,
        [(None, Shape(label="Open", bbox=(1, 1, 20, 20)))],
        base_version=0,
    )
    with pytest.raises(Conflict):
        shapes.save(
            store, project, second_view,
            [(None, Shape(label="Open", bbox=(30, 30, 60, 60)))],
            base_version=0,
        )

    fresh = images.get(store, project, first_view.id)
    assert [s.shape.bbox for s in shapes.listing(store, project, fresh)] == [(1, 1, 20, 20)]


# ------------------------------------------------------------- migrations


def test_a_crash_mid_migration_leaves_a_real_version(tmp_path, monkeypatch):
    """Each rung commits with its own version bump, so a failure part-way up
    the ladder leaves a database the next start can resume from."""
    path = tmp_path / "annotations.db"

    def explode(conn, target):
        raise RuntimeError("power cut")

    ladder = list(store_mod.MIGRATIONS)
    monkeypatch.setattr(store_mod, "MIGRATIONS", [ladder[0], explode, *ladder[2:]])
    with pytest.raises(RuntimeError):
        Store(path)

    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"project", "image", "shape"} <= tables

    monkeypatch.setattr(store_mod, "MIGRATIONS", ladder)
    healed = Store(path)
    try:
        assert healed.one("PRAGMA user_version")[0] == store_mod.SCHEMA_VERSION
    finally:
        healed.close()


# ------------------------------------------------------- filename collisions


def _boxes(store, project, rel_path):
    image = images.by_path(store, project, rel_path)
    return [s.shape.bbox for s in shapes.listing(store, project, image)]


def test_same_named_images_in_different_folders_keep_their_own_labels(store, tmp_path):
    """train/001 and val/001 share a stem. Matching on the stem alone gave
    both the first one's boxes -- and in attached mode the next save wrote
    them into val/001.xml on disk."""
    root = tmp_path / "set"
    for split, bbox in (("train", (5, 5, 30, 30)), ("val", (50, 40, 90, 80))):
        make_image(root / split / "001.jpg", 1 if split == "train" else 2)
        write_voc(root / split / "001.xml", "Open", bbox=bbox)

    summary = attach.open_folder(store, root)
    project = summary["project"]

    assert _boxes(store, project, "train/001.jpg") == [(5, 5, 30, 30)]
    assert _boxes(store, project, "val/001.jpg") == [(50, 40, 90, 80)]


def _two_folders_project(store):
    project = projects.create(store, "Split")
    labels.adopt(store, project, ["Close", "Open"])
    images.add(store, project, "images/train/001.jpg", width=200, height=200)
    images.add(store, project, "images/val/001.jpg", width=200, height=200)
    return project


def _yolo_predictions(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.yaml").write_text("names:\n  0: Close\n  1: Open\n", encoding="utf-8")
    for rel, line in rows.items():
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(line + "\n", encoding="utf-8")
    return folder


def test_predictions_that_keep_their_folders_find_their_images(store, tmp_path):
    project = _two_folders_project(store)
    run = _yolo_predictions(tmp_path / "run", {
        "labels/train/001.txt": "0 0.25 0.25 0.1 0.1 0.9",
        "labels/val/001.txt": "1 0.75 0.75 0.1 0.1 0.8",
    })
    report = review.import_predictions(store, project, run)

    assert report["matched"] == 2
    assert report["ambiguous_files"] == 0
    train = shapes.listing(store, project, images.by_path(store, project, "images/train/001.jpg"))
    val = shapes.listing(store, project, images.by_path(store, project, "images/val/001.jpg"))
    assert [s.shape.label for s in train] == ["Close"]
    assert [s.shape.label for s in val] == ["Open"]


def test_a_prediction_that_could_be_either_image_is_reported_not_guessed(store, tmp_path):
    project = _two_folders_project(store)
    run = _yolo_predictions(tmp_path / "flat", {"001.txt": "0 0.5 0.5 0.1 0.1 0.9"})
    report = review.import_predictions(store, project, run)

    assert report["matched"] == 0
    assert report["ambiguous_files"] == 1
    assert any("more than one image" in note for note in report["notes"])
    for rel in ("images/train/001.jpg", "images/val/001.jpg"):
        assert _boxes(store, project, rel) == []
