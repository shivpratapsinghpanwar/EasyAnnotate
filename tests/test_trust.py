"""Nothing is lost, nothing is wrong: the M0 fixes in docs/ROADMAP.md.

Each test here reproduces a way the backend used to lose or corrupt
annotation work without saying so, and failed before its fix.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

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


# ------------------------------------------------- re-predicting reviewed work


def test_re_predicting_a_reviewed_image_sends_it_back_for_review(store, tmp_path):
    """overwrite_reviewed put raw model boxes on an image still marked
    reviewed, so the reviewed-only export shipped them as ground truth."""
    project = _two_folders_project(store)
    image = images.by_path(store, project, "images/train/001.jpg")
    images.set_state(store, project, image, "reviewed")

    run = _yolo_predictions(tmp_path / "again", {
        "labels/train/001.txt": "0 0.25 0.25 0.1 0.1 0.9",
    })
    review.import_predictions(store, project, run, overwrite_reviewed=True)

    assert images.get(store, project, image.id).state == "predicted"


# --------------------------------------------- attached folders stay in step


def _xml_names(path):
    import xml.etree.ElementTree as ET

    return sorted(o.findtext("name") for o in ET.parse(path).getroot().iter("object"))


@pytest.fixture
def attached_client(tmp_path, monkeypatch):
    """The real app, plus a VOC folder opened in place: a0 has an 'open' box,
    a1 a 'Open' box -- the case-variant defect the door set really has."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)
    import importlib

    from data_factory.service import app as app_module
    from data_factory.service import deps

    deps.reset_stores()
    importlib.reload(app_module)
    folder = tmp_path / "doors"
    make_image(folder / "a0.jpg", 1)
    make_image(folder / "a1.jpg", 2)
    write_voc(folder / "a0.xml", "open")
    write_voc(folder / "a1.xml", "Open")
    with TestClient(app_module.app) as client:
        project = attach.open_folder(deps.get_store(), folder)["project"]
        yield client, project, folder
    deps.reset_stores()


def _idx(client, project, name):
    listed = client.get(f"/api/projects/{project.id}/labels").json()["labels"]
    return next(label["idx"] for label in listed if label["name"] == name)


def test_merging_classes_rewrites_the_files_on_disk(attached_client):
    """Merge used to change only the database. The folder kept 'open', and
    the next unrelated save quietly rewrote one file -- leaving the others."""
    client, project, folder = attached_client
    source, target = _idx(client, project, "open"), _idx(client, project, "Open")

    response = client.post(
        f"/api/projects/{project.id}/labels/{source}/merge", json={"into": target}
    )
    assert response.status_code == 200, response.text
    assert _xml_names(folder / "a0.xml") == ["Open"]
    assert _xml_names(folder / "a1.xml") == ["Open"]


def test_renaming_a_class_rewrites_the_files_on_disk(attached_client):
    client, project, folder = attached_client
    idx = _idx(client, project, "Open")

    response = client.patch(f"/api/projects/{project.id}/labels/{idx}", json={"name": "Ajar"})
    assert response.status_code == 200, response.text
    assert _xml_names(folder / "a1.xml") == ["Ajar"]
    assert _xml_names(folder / "a0.xml") == ["open"]


def test_imported_predictions_reach_an_attached_folder_as_scored_boxes(store, tmp_path):
    folder = tmp_path / "doors"
    make_image(folder / "a0.jpg", 1)
    write_voc(folder / "a0.xml", "Open")
    project = attach.open_folder(store, folder)["project"]

    run = _yolo_predictions(tmp_path / "run", {"a0.txt": "0 0.5 0.5 0.2 0.2 0.75"})
    review.import_predictions(store, project, run, overwrite_reviewed=True)

    import xml.etree.ElementTree as ET

    scores = [o.findtext("score") for o in ET.parse(folder / "a0.xml").getroot().iter("object")]
    assert sorted(s is not None for s in scores) == [False, True]


# ------------------------------------------- retired names never merge classes


def test_a_retired_class_sharing_a_live_name_exports_as_its_own_class(store, tmp_path):
    """Retire 'Open', start a new 'Open': two indices, one name. Writers map
    by name, so the retired class's shapes were exported as the new one."""
    from data_factory.core.ops import annotate_export  # noqa: F401  (registers the op)

    project = projects.create(store, "Doors")
    image = images.add(store, project, "a.jpg", width=100, height=100)
    shapes.replace_all(store, project, image, [Shape(label="Open", bbox=(1, 1, 20, 20))])
    labels.retire(store, project, 0)
    shapes.save(
        store, project, images.get(store, project, image.id),
        [(s.id, s.shape) for s in shapes.listing(store, project, image)]
        + [(None, Shape(label="Open", bbox=(40, 40, 60, 60)))],
    )

    order = labels.class_list(store, project)
    assert len(set(order)) == len(order) == 2
    exported = shapes.annotation_for(store, project, images.get(store, project, image.id))
    by_box = {s.bbox: order.index(s.label) for s in exported.shapes}
    assert by_box == {(1, 1, 20, 20): 0, (40, 40, 60, 60): 1}


# ---------------------------------------------------------------- deployment


def test_the_docker_image_keeps_the_database_inside_its_volume():
    """The workspace was /data, so data_root() -- its parent -- was /. The
    database and every project went outside the volume and were lost on
    restart. Checked from the Dockerfile, since CI may have no Docker."""
    from pathlib import PurePosixPath

    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text(encoding="utf-8")
    workspace = re.search(r"DATAFACTORY_WORKSPACE=(\S+)", dockerfile).group(1)
    volumes = re.findall(r'VOLUME \["([^"]+)"\]', dockerfile)

    data_root = PurePosixPath(workspace).parent
    assert any(data_root == PurePosixPath(v) or PurePosixPath(v) in data_root.parents for v in volumes)


# ------------------------------------------------------------------ uploads


def test_an_oversized_upload_is_refused_before_it_touches_the_disk(tmp_path, monkeypatch):
    """The handler counted bytes as it read the UploadFile -- after Starlette
    had already spooled the whole body to a temp file. The cap now acts on
    the declared length, so nothing is spooled at all."""
    pytest.importorskip("fastapi")
    import importlib

    from fastapi.testclient import TestClient
    from starlette import formparsers

    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("DATAFACTORY_MAX_UPLOAD_MB", "1")
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)
    from data_factory.service import app as app_module
    from data_factory.service import deps

    deps.reset_stores()
    importlib.reload(app_module)

    spooled = []
    real = formparsers.SpooledTemporaryFile

    def spy(*args, **kwargs):
        spooled.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", spy)
    try:
        with TestClient(app_module.app) as client:
            big = b"x" * (3 * 1024 * 1024)
            response = client.post(
                "/api/datasets", files={"file": ("big.zip", big, "application/zip")}
            )
        assert response.status_code == 413
        assert spooled == []
    finally:
        monkeypatch.delenv("DATAFACTORY_MAX_UPLOAD_MB")
        importlib.reload(app_module)
        deps.reset_stores()


def test_a_decompression_bomb_is_refused_before_it_is_decoded(tmp_path, monkeypatch):
    """A 15000x15000 PNG is a few KB on disk and 675 MB once decoded. The
    thumbnail route decoded whatever it was given; it now reads the size from
    the header and refuses before decoding."""
    pytest.importorskip("fastapi")
    import importlib
    import io
    import zipfile

    from fastapi.testclient import TestClient
    from PIL import Image as PILImage

    from data_factory.core import imaging

    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)
    from data_factory.service import app as app_module
    from data_factory.service import deps

    deps.reset_stores()
    importlib.reload(app_module)

    bomb = io.BytesIO()
    PILImage.new("1", (15000, 15000)).save(bomb, "PNG")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("bomb.png", bomb.getvalue())

    decoded = []
    monkeypatch.setattr(imaging, "load", lambda *a, **k: decoded.append(1))
    monkeypatch.setattr(imaging, "load_reduced", lambda *a, **k: decoded.append(1), raising=False)
    try:
        with TestClient(app_module.app) as client:
            dataset = client.post(
                "/api/datasets", files={"file": ("d.zip", archive.getvalue(), "application/zip")}
            ).json()["id"]
            project = client.post(
                f"/api/datasets/{dataset}/promote", json={"import_annotations": False}
            ).json()["id"]
            image = client.get(f"/api/projects/{project}/images").json()["images"][0]["id"]
            response = client.get(f"/api/projects/{project}/images/{image}/thumb")
        assert response.status_code == 413
        assert decoded == []
    finally:
        deps.reset_stores()
