"""The local link: the one route that takes a filesystem path, and its fences.

Written as refusals first, in the style of test_service.py: every way of not
being allowed in, then the one way in.
"""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from conftest import make_image, write_voc

TOKEN = "a-test-token-that-is-long-enough"


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)
    monkeypatch.delenv("EASYANNOTATE_LOCAL_TOKEN", raising=False)
    import importlib

    from data_factory.service import app as module
    from data_factory.service import deps

    deps.reset_stores()
    importlib.reload(module)
    yield module
    deps.reset_stores()


@pytest.fixture
def local(app_module, monkeypatch):
    """Local mode on, a client addressed to this machine."""
    monkeypatch.setenv("EASYANNOTATE_LOCAL_TOKEN", TOKEN)
    with TestClient(app_module.app, base_url="http://localhost:8765") as client:
        yield client


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "doors"
    make_image(root / "a.jpg", 1)
    write_voc(root / "a.xml", "Open")
    return root


def ok_headers(**extra):
    return {"X-Local-Token": TOKEN, **extra}


# ------------------------------------------------------------------ refusals


def test_local_mode_is_off_unless_a_launcher_turned_it_on(app_module, folder):
    with TestClient(app_module.app, base_url="http://localhost:8765") as client:
        assert client.get("/api/local/status", headers=ok_headers()).status_code == 404
        assert client.post("/api/local/open", json={"path": str(folder)}, headers=ok_headers()).status_code == 404


def test_no_token_or_a_wrong_one_is_refused(local, folder):
    assert local.get("/api/local/status").status_code == 403
    assert local.get("/api/local/status", headers={"X-Local-Token": "guess"}).status_code == 403
    assert local.post("/api/local/open", json={"path": str(folder)}).status_code == 403


@pytest.mark.parametrize("host", ["evil.example", "localhost.evil.example", "192.168.1.20:8765", "testserver"])
def test_a_request_addressed_to_another_host_is_refused(app_module, monkeypatch, folder, host):
    """DNS rebinding: an attacker's page reaches this server under the
    attacker's hostname. With the right token it is still refused."""
    monkeypatch.setenv("EASYANNOTATE_LOCAL_TOKEN", TOKEN)
    with TestClient(app_module.app, base_url="http://localhost:8765") as client:
        response = client.post("/api/local/open", json={"path": str(folder)}, headers=ok_headers(host=host))
        assert response.status_code == 403


@pytest.mark.parametrize("origin", ["http://evil.example", "https://localhost.evil.example", "null"])
def test_a_request_from_another_site_is_refused(local, folder, origin):
    response = local.post("/api/local/open", json={"path": str(folder)}, headers=ok_headers(origin=origin))
    assert response.status_code == 403


def test_hosted_mode_turns_it_off_even_with_the_token(local, folder, monkeypatch):
    monkeypatch.setenv("DATAFACTORY_ALLOW_ATTACHED", "0")
    assert local.post("/api/local/open", json={"path": str(folder)}, headers=ok_headers()).status_code == 404


def test_a_relative_or_missing_path_is_refused(local, tmp_path):
    assert local.post("/api/local/open", json={"path": "doors"}, headers=ok_headers()).status_code == 400
    missing = tmp_path / "nope"
    assert local.post("/api/local/open", json={"path": str(missing)}, headers=ok_headers()).status_code == 400


def test_the_launcher_never_mints_a_token_for_a_reachable_server(monkeypatch):
    from data_factory.cli.main import _local_fragment

    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)
    assert _local_fragment(argparse.Namespace(host="0.0.0.0", api_key=None)) == ""
    assert _local_fragment(argparse.Namespace(host="127.0.0.1", api_key="secret")) == ""
    monkeypatch.setenv("DATAFACTORY_API_KEY", "set-by-the-host")
    assert _local_fragment(argparse.Namespace(host="127.0.0.1", api_key=None)) == ""


# ------------------------------------------------------------------ the way in


@pytest.mark.parametrize("host", ["localhost:8765", "127.0.0.1:8765", "[::1]:8765"])
def test_the_right_token_on_this_machine_opens_a_folder(local, folder, host):
    response = local.post(
        "/api/local/open",
        json={"path": str(folder)},
        headers=ok_headers(host=host, origin=f"http://{host}"),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["images"] == 1 and body["imported"] == 1 and body["writes_back"]
    assert str(folder) not in response.text  # the page gets an id, never the path back
    # And the project is an ordinary one from here on.
    assert local.get(f"/api/projects/{body['project']}").status_code == 200


def test_the_launcher_mints_a_token_for_this_machine(monkeypatch):
    from data_factory.cli.main import _local_fragment

    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)
    # set, not delenv: pytest only restores what it recorded, and a token
    # minted here must not leak into the tests that run after this one.
    monkeypatch.setenv("EASYANNOTATE_LOCAL_TOKEN", "placeholder")
    fragment = _local_fragment(argparse.Namespace(host="127.0.0.1", api_key=None))
    assert fragment.startswith("#token=") and len(fragment) > 30
    import os

    assert os.environ["EASYANNOTATE_LOCAL_TOKEN"] == fragment.split("=", 1)[1]


# ------------------------------------------------------- AI tools and training


def _wait(client, job_id, seconds=60):
    import time

    deadline = time.time() + seconds
    job = client.get(f"/api/jobs/{job_id}").json()
    while job["state"] not in ("finished", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.1)
        job = client.get(f"/api/jobs/{job_id}").json()
    return job


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/api/local/tools", None),
    ("POST", "/api/local/tools/install", {}),
    ("POST", "/api/local/train", {"project": "0" * 32}),
    ("POST", "/api/local/predict", {"project": "0" * 32}),
])
def test_installing_and_training_are_local_only(local, method, path, body):
    """They download software and start processes: never for a web page."""
    assert local.request(method, path, json=body).status_code == 403
    assert local.request(method, path, json=body, headers={"X-Local-Token": "nope"}).status_code == 403


def test_the_tools_card_reports_this_machine(local):
    state = local.get("/api/local/tools", headers=ok_headers()).json()
    assert state["plan"]["device"] in ("cuda", "mps", "cpu")
    assert state["plan"]["summary"]
    assert "ram_gb" in state["hardware"]


def test_install_runs_as_one_job_and_a_second_click_joins_it(local, monkeypatch):
    import threading

    from data_factory.annotate.tools import install as installer

    release = threading.Event()
    calls = []

    def fake_install(*, source=None, device=None, on_line=print):
        calls.append(device)
        on_line("plan: pretend")
        release.wait(10)
        on_line("done: pretend")
        return {"device": "cpu", "summary": "pretend", "verified_device": "cpu"}

    monkeypatch.setattr(installer, "install", fake_install)
    first = local.post("/api/local/tools/install", json={"device": "cpu"}, headers=ok_headers()).json()
    second = local.post("/api/local/tools/install", json={}, headers=ok_headers()).json()
    assert second["id"] == first["id"]  # joined, not started twice
    release.set()
    job = _wait(local, first["id"])
    assert job["state"] == "finished" and job["result"]["data"]["profile"]["summary"] == "pretend"
    assert calls == ["cpu"]
    assert local.post("/api/local/tools/install", json={"device": "tpu"}, headers=ok_headers()).status_code == 400


def test_training_from_the_page_trains_on_reviewed_work(local, folder, tmp_path, monkeypatch):
    import os
    import sys
    from pathlib import Path

    monkeypatch.setenv("EASYANNOTATE_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("EASYANNOTATE_TRAIN_MODULE", "fake_trainer")
    monkeypatch.setenv("DATAFACTORY_TOOLS_PYTHON", sys.executable)
    tests_dir = str(Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))

    project = local.post("/api/local/open", json={"path": str(folder)}, headers=ok_headers()).json()["project"]
    assert local.post("/api/local/predict", json={"project": project}, headers=ok_headers()).status_code == 409

    job = local.post("/api/local/train", json={"project": project, "epochs": 2, "predict": False},
                     headers=ok_headers()).json()
    job = _wait(local, job["id"])
    assert job["state"] == "finished", job
    assert job["result"]["ok"], job["result"]
    assert job["result"]["data"]["model"].endswith(".pt")
    assert local.post("/api/local/train", json={"project": project, "epochs": 0},
                      headers=ok_headers()).status_code == 400
