"""Review hardest first: images ordered by how likely their predictions are wrong."""

from __future__ import annotations

import pytest

from data_factory.annotate import images, labels, projects, review, shapes
from data_factory.annotate.store import Store, StoreError
from data_factory.core.annotations.model import Shape


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "annotations.db")
    yield s
    s.close()


@pytest.fixture()
def project(store):
    made = projects.create(store, "Queue")
    labels.adopt(store, made, ["door", "window"])
    for name in ("sure.jpg", "unsure.jpg", "confused.jpg", "empty.jpg"):
        images.add(store, made, name, width=200, height=200)
    return made


def _predict(store, project, name, *shapes_):
    image = images.by_path(store, project, name)
    review.set_predictions(store, project, image, list(shapes_))
    return images.get(store, project, image.id)


def box(label, x1, y1, x2, y2, conf):
    return Shape(label=label, bbox=(x1, y1, x2, y2), confidence=conf)


def test_ambiguous_and_conflicting_predictions_score_highest(store, project):
    sure = _predict(store, project, "sure.jpg", box("door", 10, 10, 90, 190, 0.97), box("window", 120, 20, 180, 80, 0.95))
    unsure = _predict(store, project, "unsure.jpg", box("door", 10, 10, 90, 190, 0.5), box("door", 120, 10, 190, 190, 0.55))
    # Same object, two classes: the model is arguing with itself.
    confused = _predict(store, project, "confused.jpg", box("door", 10, 10, 90, 190, 0.9), box("window", 12, 12, 90, 188, 0.88))

    assert sure.uncertainty < 0.2
    assert unsure.uncertainty == pytest.approx(1.9, abs=0.01)  # 1.0 + 0.9
    assert confused.uncertainty > 1.0  # two fairly sure boxes, plus a conflict
    assert unsure.uncertainty > confused.uncertainty > sure.uncertainty


def test_the_uncertain_order_lists_hardest_first_and_position_agrees(store, project):
    _predict(store, project, "sure.jpg", box("door", 10, 10, 90, 190, 0.97))
    _predict(store, project, "unsure.jpg", box("door", 10, 10, 90, 190, 0.5))
    _predict(store, project, "confused.jpg", box("door", 10, 10, 90, 190, 0.9), box("window", 12, 12, 90, 188, 0.88))

    ordered = images.listing(store, project, state="predicted", order="uncertain")
    assert [i.name for i in ordered] == ["confused.jpg", "unsure.jpg", "sure.jpg"]
    for index, image in enumerate(ordered):
        assert images.position(store, project, image.id, state="predicted", order="uncertain") == index
    # The file order is untouched.
    assert [i.name for i in images.listing(store, project)][:3] == ["sure.jpg", "unsure.jpg", "confused.jpg"]


def test_correcting_a_prediction_settles_it(store, project):
    image = _predict(store, project, "unsure.jpg", box("door", 10, 10, 90, 190, 0.5))
    assert image.uncertainty == pytest.approx(1.0)
    stored = shapes.listing(store, project, image)
    moved = Shape(label="door", bbox=(12, 10, 92, 190), confidence=0.5)
    shapes.save(store, project, image, [(stored[0].id, moved)])
    assert images.get(store, project, image.id).uncertainty == 0.0


def test_an_unknown_order_is_refused(store, project):
    with pytest.raises(StoreError):
        images.listing(store, project, order="random")


def test_the_api_orders_and_locates_by_uncertainty(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    import importlib

    from data_factory.service import app as app_module
    from data_factory.service import deps

    deps.reset_stores()
    importlib.reload(app_module)
    store_ = deps.get_store()
    made = projects.create(store_, "Queue")
    labels.adopt(store_, made, ["door"])
    for name in ("a.jpg", "b.jpg"):
        images.add(store_, made, name, width=200, height=200)
    _predict(store_, made, "a.jpg", box("door", 1, 1, 50, 50, 0.99))
    _predict(store_, made, "b.jpg", box("door", 1, 1, 50, 50, 0.5))
    with TestClient(app_module.app) as client:
        base = f"/api/projects/{made.id}/images"
        listed = client.get(base, params={"state": "predicted", "order": "uncertain"}).json()["images"]
        assert [i["name"] for i in listed] == ["b.jpg", "a.jpg"]
        assert listed[0]["uncertainty"] == 1.0
        located = client.get(base, params={"state": "predicted", "order": "uncertain", "locate": listed[1]["id"], "limit": 1})
        assert located.json()["position"] == 1
        assert client.get(base, params={"order": "random"}).status_code == 400
    deps.reset_stores()


def test_a_scored_shape_saved_from_the_editor_stays_a_prediction(store, project):
    """'Find every person' on one image adds shapes with a confidence through
    the ordinary save. They used to be stored as hand-drawn work -- solid,
    not reviewable, invisible to agreement."""
    image = images.by_path(store, project, "empty.jpg")
    shapes.save(
        store, project, image,
        [(None, box("door", 10, 10, 90, 190, 0.61)), (None, Shape(label="door", bbox=(100, 10, 150, 60)))],
    )
    stored = shapes.listing(store, project, images.get(store, project, image.id))
    assert [(s.origin, s.shape.confidence) for s in stored] == [("model", 0.61), ("human", None)]
    assert shapes.agreement(store, project)["unreviewed"] == 1
