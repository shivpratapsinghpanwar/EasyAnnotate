"""Defects an independent review found in the loop, queue and SAM work.

Each test reproduces one, as the reviewer did, and failed before its fix.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading

import pytest

from conftest import make_image
from data_factory.annotate import images, labels, projects, review, shapes
from data_factory.annotate import store as store_mod
from data_factory.annotate.store import Store
from data_factory.annotate.tools.local import LocalTool
from data_factory.core.annotations.model import Shape

FAKE = "data_factory.annotate.tools.fake"


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "annotations.db")
    yield s
    s.close()


def box(label, x1, y1, x2, y2, conf):
    return Shape(label=label, bbox=(x1, y1, x2, y2), confidence=conf)


# ------------------------------------------------- 1. the tool-name override


def test_a_payload_cannot_rename_the_tool(tmp_path):
    """params.tool used to replace the route's tool name, so an allowed tool
    could be turned into the model-file loader."""
    image = make_image(tmp_path / "a.jpg", 1)
    evil = tmp_path / "evil.pt"
    evil.write_bytes(b"pickle")
    tool = LocalTool([sys.executable, "-m", FAKE])
    try:
        reply = tool.invoke("fake.text", {
            "tool": "fake.model", "model": str(evil), "text": "door",
            "image_id": "x", "image_path": str(image),
        })
    finally:
        tool.close()
    assert [s["label"] for s in reply["shapes"]] == ["door"]  # fake.text ran, not fake.model


def test_http_refuses_reserved_params(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from test_annotate_api import upload
    from test_tools_api import build_client

    make_image(tmp_path / "set" / "a.jpg", 1)
    with build_client(tmp_path, monkeypatch, tools=FAKE) as client:
        dataset_id = upload(client, tmp_path / "set")
        project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        image = client.get(f"/api/projects/{project['id']}/images").json()["images"][0]
        base = f"/api/projects/{project['id']}/tools/fake.text"
        sneaky = {"tool": "fake.model", "model": str(tmp_path / "evil.pt"), "text": "x"}
        assert client.post(base, json={"image_id": image["id"], "params": sneaky}).status_code == 400
        assert client.post(base + "/batch", json={"params": sneaky}).status_code == 400
        assert client.post(f"/api/projects/{project['id']}/tools/nope/prepare",
                           json={"image_id": image["id"]}).status_code == 404


# ------------------------------------------- 2. a health check killed the tool


def test_a_health_check_during_slow_work_does_not_kill_the_sidecar(tmp_path):
    image = make_image(tmp_path / "a.jpg", 1)
    tool = LocalTool([sys.executable, "-m", FAKE], env={**os.environ, "DFX_FAKE_WORK_MS": "6000"})
    try:
        tool.prepare("fake.box", "img", None)
        replies: list = []
        worker = threading.Thread(target=lambda: replies.append(
            tool.invoke("fake.box", {"image_id": "img", "image_path": str(image), "box": [1, 1, 9, 9]})
        ))
        worker.start()
        health = tool.health("fake.box")  # arrives while the invoke holds the sidecar
        worker.join(60)
        assert health.ready, health
        assert replies and replies[0]["shapes"], "the invoke was killed by the health check"
        assert tool.client._restarts == 0
    finally:
        tool.close()


# ------------------------------------------- 3. undoing a delete double-counted


def test_undoing_a_deleted_prediction_does_not_count_it_twice(store):
    project = projects.create(store, "Undo")
    labels.adopt(store, project, ["door"])
    image = images.add(store, project, "a.jpg", width=200, height=200)
    review.set_predictions(store, project, image, [box("door", 10, 10, 90, 150, 0.6)])
    stored = shapes.listing(store, project, images.get(store, project, image.id))
    shape_id = stored[0].id

    shapes.save(store, project, images.get(store, project, image.id), [])               # delete, autosaved
    shapes.save(store, project, images.get(store, project, image.id),
                [(shape_id, box("door", 10, 10, 90, 150, 0.6))])                         # Ctrl+Z
    images.set_state(store, project, images.get(store, project, image.id), "reviewed")

    agreement = shapes.agreement(store, project)
    assert agreement["model_shapes"] == 1
    assert agreement["agreement"] == 1.0


# ---------------------------------------------- 4. merging left a stale score


def test_merging_classes_rescores_the_images(store):
    project = projects.create(store, "Merge")
    labels.adopt(store, project, ["Close", "close"])
    image = images.add(store, project, "a.jpg", width=200, height=200)
    review.set_predictions(store, project, image, [
        box("Close", 10, 10, 90, 150, 0.98), box("close", 12, 12, 90, 148, 0.98),
    ])
    assert images.get(store, project, image.id).uncertainty > 1.0  # a class conflict
    labels.merge(store, project, 1, 0)
    assert images.get(store, project, image.id).uncertainty == pytest.approx(0.08)


# ------------------------------------------- 5. the migration never backfilled


def test_upgrading_scores_existing_predictions(tmp_path):
    path = tmp_path / "annotations.db"
    s = Store(path)
    project = projects.create(s, "Old")
    labels.adopt(s, project, ["door"])
    image = images.add(s, project, "a.jpg", width=200, height=200)
    review.set_predictions(s, project, image, [box("door", 10, 10, 90, 150, 0.5)])
    s.close()
    # As a v5 database left it: the column there, every score 0.
    with sqlite3.connect(path) as raw:
        raw.execute("UPDATE image SET uncertainty = 0")
        raw.execute("PRAGMA user_version = 5")

    upgraded = Store(path)
    try:
        assert upgraded.one("PRAGMA user_version")[0] == store_mod.SCHEMA_VERSION
        assert images.get(upgraded, project, image.id).uncertainty == pytest.approx(1.0)
    finally:
        upgraded.close()


# ------------------------------------- smaller: folder imports count predictions


def test_predictions_read_from_a_folder_are_counted(tmp_path, store):
    from data_factory.annotate import attach

    root = tmp_path / "doors"
    make_image(root / "a.jpg", 1, size=(120, 90))
    (root / "a.xml").write_text(
        "<annotation><filename>a.jpg</filename><size><width>120</width><height>90</height>"
        "<depth>3</depth></size><object><name>door</name><score>0.7</score><bndbox>"
        "<xmin>5</xmin><ymin>5</ymin><xmax>60</xmax><ymax>80</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    project = attach.open_folder(store, root)["project"]
    image = images.by_path(store, project, "a.jpg")
    row = store.one("SELECT predicted_count FROM image WHERE id = ?", (image.id,))
    assert row["predicted_count"] == 1
