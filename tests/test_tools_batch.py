"""A tool run over many images is a model run: its answers become predictions.

Driven through the real fake sidecar as a separate process, like the rest of
the tools tests, so the boundary is exercised and no PyTorch is needed.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from conftest import make_image
from test_tools_api import FAKE_MODULE, build_client


def _wait(client, job_id: str, seconds: float = 30.0) -> dict:
    deadline = time.time() + seconds
    status = client.get(f"/api/jobs/{job_id}").json()
    while status["state"] not in ("finished", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.2)
        status = client.get(f"/api/jobs/{job_id}").json()
    return status


def test_a_batch_run_turns_every_unreviewed_image_into_predictions(tmp_path, monkeypatch):
    folder = tmp_path / "set"
    for index in range(3):
        make_image(folder / f"p{index}.jpg", 10 + index)

    with build_client(tmp_path, monkeypatch, tools=FAKE_MODULE) as client:
        from test_annotate_api import upload

        dataset_id = upload(client, folder)
        project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        listed = client.get(f"/api/projects/{project['id']}/images").json()["images"]
        # One image is already reviewed by a person; a batch must leave it alone.
        done = listed[0]
        assert client.patch(
            f"/api/projects/{project['id']}/images/{done['id']}", json={"state": "reviewed"}
        ).status_code == 200
        # The fake tool answers only for prepared images; prepare them.
        for image in listed[1:]:
            client.post(
                f"/api/projects/{project['id']}/tools/fake.box",
                json={"image_id": image["id"], "params": {"box": [1, 1, 5, 5]}, "label": "Door"},
            )

        started = client.post(
            f"/api/projects/{project['id']}/tools/fake.box/batch",
            json={"params": {"box": [10, 10, 60, 50]}, "label": "Door", "states": ["todo", "in_progress"]},
        )
        assert started.status_code == 200, started.text
        status = _wait(client, started.json()["id"])
        assert status["state"] == "finished", status

        counts = client.get(f"/api/projects/{project['id']}/images").json()["counts"]
        assert counts["reviewed"] == 1
        assert counts["predicted"] == 2
        for image in listed[1:]:
            detail = client.get(f"/api/projects/{project['id']}/images/{image['id']}").json()
            assert detail["state"] == "predicted"
            assert detail["shapes"] and all(s["origin"] == "model" for s in detail["shapes"])
            assert all(s["label"] == "Door" for s in detail["shapes"])


def test_a_batch_refuses_reviewed_images_and_unknown_tools(tmp_path, monkeypatch):
    folder = tmp_path / "set"
    make_image(folder / "a.jpg", 1)
    with build_client(tmp_path, monkeypatch, tools=FAKE_MODULE) as client:
        from test_annotate_api import upload

        dataset_id = upload(client, folder)
        project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        assert client.post(
            f"/api/projects/{project['id']}/tools/fake.box/batch", json={"states": ["reviewed"]}
        ).status_code == 400
        assert client.post(
            f"/api/projects/{project['id']}/tools/nope/batch", json={}
        ).status_code == 404
