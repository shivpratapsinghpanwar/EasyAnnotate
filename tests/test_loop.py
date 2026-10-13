"""The loop: reviewed images -> a model -> predictions on the rest.

Driven through the fake sidecar and a fake trainer, so it runs anywhere; the
real Ultralytics path is smoked by hand (see CLAUDE.md).
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from conftest import make_image, write_voc
from data_factory.annotate import attach, images, labels, loop, shapes
from data_factory.annotate.store import Store
from data_factory.annotate.tools.registry import REGISTRY

FAKE = "data_factory.annotate.tools.fake"


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "annotations.db")
    yield s
    s.close()


@pytest.fixture()
def fake_tools(monkeypatch):
    monkeypatch.setenv("DATAFACTORY_TOOLS_MODULE", FAKE)
    REGISTRY.close()
    yield
    REGISTRY.close()


@pytest.fixture()
def folder(tmp_path):
    """Three door photos: one labelled (so reviewed on import), two to do."""
    root = tmp_path / "doors"
    for index in range(3):
        make_image(root / f"d{index}.jpg", 30 + index, size=(120, 90))
    write_voc(root / "d0.xml", "door", bbox=(10, 10, 60, 70), size=(120, 90))
    return root


def test_predict_fills_the_unreviewed_images_and_leaves_reviewed_alone(store, folder, tmp_path, fake_tools):
    project = attach.open_folder(store, folder)["project"]
    model = tmp_path / "door.pt"
    model.write_bytes(b"pretend weights")
    before = shapes.listing(store, project, images.by_path(store, project, "d0.jpg"))

    report = loop.predict(store, project, folder, model, tool="fake.model")

    assert report["images"] == 2 and report["predicted"] == 2 and report["failed"] == 0
    for name in ("d1.jpg", "d2.jpg"):
        image = images.by_path(store, project, name)
        assert image.state == "predicted"
        made = shapes.listing(store, project, image)
        assert [(s.shape.label, s.origin, s.shape.confidence) for s in made] == [("door", "model", 0.42)]
        assert image.uncertainty == pytest.approx(0.84)  # 0.42 is nearly a coin toss: review it early
        # Attached: the prediction is on disk too, scored, so it stays one.
        xml = ET.parse(folder / name.replace(".jpg", ".xml")).getroot()
        assert xml.find("object/score") is not None
    reviewed = images.by_path(store, project, "d0.jpg")
    assert reviewed.state == "reviewed"
    assert [s.shape.bbox for s in shapes.listing(store, project, reviewed)] == [s.shape.bbox for s in before]


def test_predict_says_what_is_missing(store, folder, tmp_path, monkeypatch):
    project = attach.open_folder(store, folder)["project"]
    with pytest.raises(loop.LoopError, match="no model file"):
        loop.predict(store, project, folder, tmp_path / "missing.pt")
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    monkeypatch.delenv("DATAFACTORY_TOOLS_MODULE", raising=False)
    REGISTRY.close()
    with pytest.raises(loop.LoopError, match="tools install"):
        loop.predict(store, project, folder, model)


def test_train_exports_reviewed_work_in_the_projects_class_order(store, folder, tmp_path, monkeypatch):
    project = attach.open_folder(store, folder)["project"]
    labels.ensure(store, project, "handle")  # a second class, index 1
    monkeypatch.setenv("EASYANNOTATE_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("EASYANNOTATE_TRAIN_MODULE", "fake_trainer")
    monkeypatch.setenv("DATAFACTORY_TOOLS_PYTHON", sys.executable)
    tests_dir = str(Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))

    lines: list[str] = []
    trained = loop.train(store, project, epochs=3, on_line=lines.append)

    assert trained["model"].is_file()
    assert trained["model"].parent == tmp_path / "models" / "projects" / project.id
    assert trained["metrics"]["images"] == 1  # only the reviewed image
    assert trained["metrics"]["names"] == {"0": "door", "1": "handle"}  # indices as exported (JSON keys)
    assert [line for line in lines if "/3" in line] == ["1/3  mAP50 0.5", "2/3  mAP50 0.5", "3/3  mAP50 0.5"]
    assert loop.latest_model(project) == trained["model"]


def test_train_refuses_with_nothing_reviewed(store, tmp_path):
    root = tmp_path / "bare"
    make_image(root / "a.jpg", 1)
    project = attach.open_folder(store, root)["project"]
    with pytest.raises(loop.LoopError, match="review some images first"):
        loop.train(store, project)


def test_model_tools_are_refused_over_http(tmp_path, monkeypatch):
    """A model file is code. No HTTP route may load one."""
    pytest.importorskip("fastapi")
    from test_tools_api import build_client

    make_image(tmp_path / "set" / "a.jpg", 1)
    with build_client(tmp_path, monkeypatch, tools=FAKE) as client:
        from test_annotate_api import upload

        dataset_id = upload(client, tmp_path / "set")
        project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        image = client.get(f"/api/projects/{project['id']}/images").json()["images"][0]
        params = {"model": str(tmp_path / "evil.pt")}
        base = f"/api/projects/{project['id']}/tools/fake.model"
        assert client.post(base, json={"image_id": image["id"], "params": params}).status_code == 403
        assert client.post(base + "/batch", json={"params": params}).status_code == 403
        assert client.post(base + "/prepare", json={"image_id": image["id"]}).status_code == 403
        # And the editor is never offered it.
        assert "fake.model" in [t["name"] for t in client.get("/api/tools").json()["tools"]]
