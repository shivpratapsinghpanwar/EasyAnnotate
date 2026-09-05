"""Attached projects: annotating a folder in place.

The promise being tested is narrow and concrete: **the label file on disk is
updated when you draw.** Not a copy of it, not on export -- the file that was
already there. Everything else here follows from that.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from conftest import make_image, make_pair, write_voc

from data_factory.annotate import attach, images, labels, projects, shapes
from data_factory.annotate.store import Store, StoreError
from data_factory.core.annotations.model import Shape


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "annotations.db")
    yield s
    s.close()


@pytest.fixture()
def voc_folder(tmp_path):
    folder = tmp_path / "voc_set"
    folder.mkdir()
    make_pair(folder, "a", "Open", 1)
    make_pair(folder, "b", "Close", 2)
    make_image(folder / "c.jpg", 3)  # no label yet
    return folder


@pytest.fixture()
def bare_folder(tmp_path):
    folder = tmp_path / "bare"
    folder.mkdir()
    for index in range(3):
        make_image(folder / f"p{index}.jpg", 10 + index)
    return folder


def only_image(store, project, name):
    found = images.by_path(store, project, name)
    assert found is not None, name
    return found


# ------------------------------------------------------------------ opening

def test_opening_a_folder_registers_its_images(store, voc_folder):
    summary = attach.open_folder(store, voc_folder)
    assert summary["images"] == 3
    assert summary["project"].is_attached
    assert Path(summary["project"].root) == voc_folder.resolve()


def test_nothing_is_copied(store, voc_folder, tmp_path):
    """The point of attached mode: no second copy of the dataset."""
    before = {p.name for p in voc_folder.iterdir()}
    attach.open_folder(store, voc_folder)
    assert {p.name for p in voc_folder.iterdir()} == before


def test_existing_labels_are_imported(store, voc_folder):
    summary = attach.open_folder(store, voc_folder)
    assert summary["shapes_imported"] == 2
    project = summary["project"]
    assert sorted(l.name for l in labels.listing(store, project)) == ["Close", "Open"]


def test_the_detected_format_is_what_gets_written_back(store, voc_folder):
    summary = attach.open_folder(store, voc_folder)
    assert summary["sidecar_format"] == "voc"
    assert summary["writes_back"] is True


def test_a_folder_of_bare_images_defaults_to_voc(store, bare_folder):
    summary = attach.open_folder(store, bare_folder)
    assert summary["sidecar_format"] == "voc"
    assert any("no labels found" in w for w in summary["warnings"])


def test_reopening_the_same_folder_returns_the_same_project(store, voc_folder):
    first = attach.open_folder(store, voc_folder)
    second = attach.open_folder(store, voc_folder)
    assert second["reused"] is True
    assert second["project"].id == first["project"].id


def test_reopening_does_not_re_import_over_your_work(store, voc_folder):
    """A second open must not overwrite edits with what is on disk."""
    first = attach.open_folder(store, voc_folder)
    project = first["project"]
    image = only_image(store, project, "a.jpg")
    shapes.save(store, project, image, [], base_version=image.version)
    assert shapes.listing(store, project, image) == []

    attach.open_folder(store, voc_folder)
    again = only_image(store, project, "a.jpg")
    assert shapes.listing(store, project, again) == [], "the re-open clobbered an edit"


def test_a_missing_folder_is_refused(store, tmp_path):
    with pytest.raises(StoreError):
        attach.open_folder(store, tmp_path / "not_there")


def test_a_folder_with_no_images_is_refused(store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("nothing", encoding="utf-8")
    with pytest.raises(StoreError):
        attach.open_folder(store, empty)


def test_attached_projects_can_be_disabled(store, voc_folder, monkeypatch):
    """A host must never hold a project pointing at a server-side path."""
    monkeypatch.setenv("DATAFACTORY_ALLOW_ATTACHED", "0")
    with pytest.raises(StoreError):
        attach.open_folder(store, voc_folder)


# ------------------------------------------------------------ write-through

def test_drawing_updates_the_xml_on_disk(store, voc_folder):
    """The whole promise, in one test."""
    project = attach.open_folder(store, voc_folder)["project"]
    image = only_image(store, project, "c.jpg")  # had no label file at all
    target = voc_folder / "c.xml"
    assert not target.exists()

    shapes.save(
        store, project, image,
        [(None, Shape(label="Handle", bbox=(5, 5, 40, 40)))],
        base_version=image.version,
    )
    attach.write_through(store, project, image)

    assert target.is_file(), "no sidecar was written"
    names = [n.text for n in ET.parse(target).getroot().iter("name")]
    assert names == ["Handle"]


def test_editing_an_existing_label_rewrites_it(store, voc_folder):
    project = attach.open_folder(store, voc_folder)["project"]
    image = only_image(store, project, "a.jpg")
    stored = shapes.listing(store, project, image)
    assert len(stored) == 1

    moved = Shape(label="Open", bbox=(11, 12, 55, 66))
    shapes.save(store, project, image, [(stored[0].id, moved)], base_version=image.version)
    attach.write_through(store, project, image)

    root = ET.parse(voc_folder / "a.xml").getroot()
    box = root.find(".//bndbox")
    assert [int(box.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")] == [11, 12, 55, 66]


def test_deleting_every_shape_leaves_a_valid_empty_annotation(store, voc_folder):
    """Not a deleted file: an image with no objects is a hard negative, and
    dropping the file would make it indistinguishable from an unlabelled one."""
    project = attach.open_folder(store, voc_folder)["project"]
    image = only_image(store, project, "a.jpg")
    shapes.save(store, project, image, [], base_version=image.version)
    attach.write_through(store, project, image)

    target = voc_folder / "a.xml"
    assert target.is_file()
    assert list(ET.parse(target).getroot().iter("object")) == []


def test_a_labelme_folder_writes_labelme(store, tmp_path):
    folder = tmp_path / "lm"
    folder.mkdir()
    make_image(folder / "x.jpg", 5)
    (folder / "x.json").write_text(
        json.dumps({
            "imagePath": "x.jpg", "imageWidth": 120, "imageHeight": 90,
            "shapes": [{"label": "Open", "shape_type": "rectangle",
                        "points": [[1, 1], [40, 40]]}],
        }),
        encoding="utf-8",
    )
    project = attach.open_folder(store, folder)["project"]
    image = only_image(store, project, "x.jpg")
    shapes.save(
        store, project, image,
        [(None, Shape(label="Open", bbox=(2, 3, 50, 60)))],
        base_version=image.version,
    )
    written = attach.write_through(store, project, image)

    assert written.name == "x.json"
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["shapes"][0]["label"] == "Open"


def test_a_yolo_folder_writes_txt_and_keeps_data_yaml_in_step(store, tmp_path):
    from data_factory.core.annotations import yolo

    folder = tmp_path / "y"
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir(parents=True)
    make_image(folder / "images" / "i0.jpg", 7)
    (folder / "labels" / "i0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (folder / "data.yaml").write_text("names:\n  0: Open\n  1: Close\n", encoding="utf-8")

    summary = attach.open_folder(store, folder)
    project = summary["project"]
    assert summary["sidecar_format"] == "yolo"
    # The existing class order must be inherited, not re-derived.
    assert labels.class_list(store, project) == ["Open", "Close"]

    image = only_image(store, project, "images/i0.jpg")
    shapes.save(
        store, project, image,
        [(None, Shape(label="Close", bbox=(10, 10, 50, 50)))],
        base_version=image.version,
    )
    written = attach.write_through(store, project, image)

    assert written == folder / "labels" / "i0.txt"
    assert written.read_text(encoding="utf-8").startswith("1 "), "Close must stay index 1"
    classes, _ = yolo.read_data_yaml(folder / "data.yaml")
    assert classes.names == ["Open", "Close"]


def test_a_new_class_is_appended_to_data_yaml_not_inserted(store, tmp_path):
    from data_factory.core.annotations import yolo

    folder = tmp_path / "y2"
    (folder / "images").mkdir(parents=True)
    make_image(folder / "images" / "i0.jpg", 7)
    (folder / "labels").mkdir(parents=True)
    (folder / "labels" / "i0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (folder / "data.yaml").write_text("names:\n  0: Open\n  1: Close\n", encoding="utf-8")

    project = attach.open_folder(store, folder)["project"]
    image = only_image(store, project, "images/i0.jpg")
    shapes.save(
        store, project, image,
        [(None, Shape(label="Ajar", bbox=(1, 1, 20, 20)))],
        base_version=image.version,
    )
    attach.write_through(store, project, image)

    classes, _ = yolo.read_data_yaml(folder / "data.yaml")
    assert classes.names == ["Open", "Close", "Ajar"], "an existing index moved"


def test_a_collected_format_does_not_write_through(store, tmp_path):
    """COCO keeps one file for the whole dataset. Rewriting it on every
    autosave would be absurd, so it is off -- and said so, not assumed."""
    folder = tmp_path / "coco"
    folder.mkdir()
    make_image(folder / "i0.jpg", 4)
    (folder / "instances.json").write_text(
        json.dumps({
            "images": [{"id": 1, "file_name": "i0.jpg", "width": 120, "height": 90}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                             "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "Open"}],
        }),
        encoding="utf-8",
    )

    summary = attach.open_folder(store, folder)
    assert summary["sidecar_format"] == ""
    assert summary["writes_back"] is False
    assert any("NOT written back" in w for w in summary["warnings"])

    project = summary["project"]
    image = only_image(store, project, "i0.jpg")
    assert attach.write_through(store, project, image) is None
    # The edit is still safe in the database.
    shapes.save(
        store, project, image,
        [(None, Shape(label="Open", bbox=(2, 2, 30, 30)))],
        base_version=image.version,
    )
    assert len(shapes.listing(store, project, image)) == 1


def test_a_read_only_sidecar_does_not_lose_the_edit(store, voc_folder, monkeypatch):
    """The folder belongs to the user. An unwritable file is a normal thing to
    meet, and the edit must survive it."""
    from data_factory.core.annotations import voc as voc_module

    project = attach.open_folder(store, voc_folder)["project"]
    image = only_image(store, project, "a.jpg")

    def refuse(*_args, **_kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(voc_module, "write", refuse)
    shapes.save(
        store, project, image,
        [(None, Shape(label="Open", bbox=(9, 9, 40, 40)))],
        base_version=image.version,
    )
    with pytest.raises(StoreError):
        attach.write_through(store, project, image)

    assert len(shapes.listing(store, project, image)) == 1, "the edit was lost"


# ------------------------------------------------------------------ schema

def test_a_fresh_database_climbs_the_whole_ladder(tmp_path):
    """Asserted against the constant and the column, not a literal version --
    a hard-coded number here just goes stale on the next migration."""
    from data_factory.annotate.store import SCHEMA_VERSION

    store = Store(tmp_path / "fresh.db")
    assert store.one("PRAGMA user_version")[0] == SCHEMA_VERSION
    columns = [r["name"] for r in store.query("PRAGMA table_info(project)")]
    assert "sidecar_format" in columns
    store.close()


def test_a_version_one_database_upgrades_in_place(tmp_path):
    """The ladder has to work on an existing database, not just a new one."""
    import sqlite3

    from data_factory.annotate.store import SCHEMA_FILE

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    conn.execute("PRAGMA user_version=1")
    conn.execute(
        "INSERT INTO project (id, owner_id, name, storage, retention, root,"
        " next_label_idx, created_at, updated_at)"
        " VALUES ('a' , 'local', 'old one', 'managed', 'keep', '', 0, 0, 0)"
    )
    conn.commit()
    conn.close()

    from data_factory.annotate.store import SCHEMA_VERSION

    store = Store(path)
    assert store.one("PRAGMA user_version")[0] == SCHEMA_VERSION
    row = store.one("SELECT name, sidecar_format FROM project WHERE id = 'a'")
    assert row["name"] == "old one", "existing rows must survive the migration"
    assert row["sidecar_format"] == ""
    store.close()
