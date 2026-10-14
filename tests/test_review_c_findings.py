"""Defects an independent review found in the AI-tools installer and the
app's Train/Predict routes. Each reproduction is the reviewer's, and failed
before its fix."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from conftest import make_image, write_voc
from data_factory import hardware
from data_factory.annotate import attach, images, loop, review, shapes
from data_factory.annotate.store import Store
from data_factory.annotate.tools.registry import REGISTRY

FAKE = "data_factory.annotate.tools.fake"
TOKEN = "a-test-token-that-is-long-enough"
H = {"X-Local-Token": TOKEN}
LOOPBACK = ("127.0.0.1", 50000)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("EASYANNOTATE_MODELS", str(tmp_path / "models"))
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)
    monkeypatch.setenv("EASYANNOTATE_LOCAL_TOKEN", TOKEN)
    from data_factory.service import app as module
    from data_factory.service import deps

    deps.reset_stores()
    importlib.reload(module)
    yield module
    deps.reset_stores()


@pytest.fixture
def local(app_module):
    with TestClient(app_module.app, base_url="http://localhost:8765", client=LOOPBACK) as client:
        yield client


def _folder(tmp_path, name):
    root = tmp_path / name
    make_image(root / "a.jpg", 1)
    write_voc(root / "a.xml", "Open")
    make_image(root / "b.jpg", 2)
    return root


def _wait(client, job_id, seconds=30):
    deadline = time.time() + seconds
    job = client.get(f"/api/jobs/{job_id}").json()
    while job["state"] not in ("finished", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.05)
        job = client.get(f"/api/jobs/{job_id}").json()
    return job


# ------------------------------------- 1. predict overwrites reviewed work

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


def test_predict_never_touches_an_image_reviewed_while_it_runs(store, tmp_path, fake_tools, monkeypatch):
    root = tmp_path / "doors"
    for i in range(3):
        make_image(root / f"d{i}.jpg", 30 + i, size=(120, 90))
    project = attach.open_folder(store, root)["project"]
    first = tmp_path / "door.pt"
    first.write_bytes(b"x")
    loop.predict(store, project, root, first, tool="fake.model")   # every image: one 'door' prediction

    # Second turn of the loop with a newer model. While it runs, the person
    # reviews d2 in the editor: Space accepts its 'door' prediction.
    second = tmp_path / "v2.pt"
    second.write_bytes(b"x")
    real = REGISTRY.invoke_waiting
    calls = {"n": 0}

    def person_reviews_d2_meanwhile(name, payload, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            d2 = images.by_path(store, project, "d2.jpg")
            review.accept_all(store, project, d2)
        return real(name, payload, **kw)

    monkeypatch.setattr(REGISTRY, "invoke_waiting", person_reviews_d2_meanwhile)
    loop.predict(store, project, root, second, tool="fake.model")

    d2 = images.by_path(store, project, "d2.jpg")
    labels_now = [s.shape.label for s in shapes.listing(store, project, d2)]
    assert (d2.state, labels_now) == ("reviewed", ["door"]), (
        f"reviewed image now state={d2.state!r}, accepted shapes replaced by: {labels_now}"
    )


# ---------------- 2. the train guard is global: project B joins project A's job

def test_training_project_b_does_not_return_project_a_job(local, tmp_path, monkeypatch):
    a = local.post("/api/local/open", json={"path": str(_folder(tmp_path, "A"))}, headers=H).json()["project"]
    b = local.post("/api/local/open", json={"path": str(_folder(tmp_path, "B"))}, headers=H).json()["project"]
    release = threading.Event()

    def slow_train(store, project, **kw):
        release.wait(10)
        raise loop.LoopError("stop here")

    monkeypatch.setattr(loop, "train", slow_train)
    try:
        job_a = local.post("/api/local/train", json={"project": a, "predict": False}, headers=H).json()
        job_b = local.post("/api/local/train", json={"project": b, "predict": False}, headers=H).json()
    finally:
        release.set()
    assert job_b["dataset"] == b, f"asked to train B, got A's job back: {job_b['id']} == {job_a['id']}"


# ----------------------------- 3. /predict has no guard: two clicks, two jobs

def test_predict_twice_runs_once(local, tmp_path, monkeypatch):
    project = local.post("/api/local/open", json={"path": str(_folder(tmp_path, "A"))}, headers=H).json()["project"]
    models = tmp_path / "models" / "projects" / project
    models.mkdir(parents=True)
    (models / "20260101-000000.pt").write_bytes(b"x")
    release = threading.Event()
    running = []

    def slow_predict(*a, **kw):
        running.append(1)
        release.wait(10)
        return {"images": 0, "predicted": 0, "shapes": 0, "failed": 0, "first_error": "", "seconds": 0}

    monkeypatch.setattr(loop, "predict", slow_predict)
    try:
        one = local.post("/api/local/predict", json={"project": project}, headers=H).json()
        two = local.post("/api/local/predict", json={"project": project}, headers=H).json()
        time.sleep(0.5)
        concurrent = len(running)
    finally:
        release.set()
    assert one["id"] == two["id"] and concurrent == 1, f"two predict jobs ran at once ({concurrent})"


# --------- 4. web predict that failed on every image reports success to the UI

def test_web_predict_that_failed_everywhere_is_not_a_success(local, tmp_path, monkeypatch):
    project = local.post("/api/local/open", json={"path": str(_folder(tmp_path, "A"))}, headers=H).json()["project"]
    models = tmp_path / "models" / "projects" / project
    models.mkdir(parents=True)
    (models / "20260101-000000.pt").write_bytes(b"x")

    class Spec:
        name, kind = "detect.model", "model"

    monkeypatch.setattr(REGISTRY, "spec", lambda name: Spec())

    def broken(*a, **k):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    monkeypatch.setattr(REGISTRY, "invoke_waiting", broken)
    job = local.post("/api/local/predict", json={"project": project}, headers=H).json()
    job = _wait(local, job["id"])
    data = job["result"]["data"]["predicted"]
    assert data["predicted"] == 0 and data["failed"] >= 1
    assert job["state"] != "finished" or not job["result"]["ok"], (
        "every image failed, yet the job is finished/ok -- the UI says "
        "'0 prediction(s) on 0 image(s)' and 'review them hardest first'"
    )


# ------------------------------------------------- 5. cancel is acknowledged and ignored

SLOW_TRAINER = '''
import argparse, json, time
p = argparse.ArgumentParser()
for f in ("--data","--model","--epochs","--imgsz","--out","--name","--device"): p.add_argument(f)
a = p.parse_args()
for e in range(20):
    print(f"EASYANNOTATE_EPOCH {e+1}/20", flush=True); time.sleep(0.15)
best = Path(a.out) / a.name / "best.pt"; best.parent.mkdir(parents=True, exist_ok=True); best.write_bytes(b"x")
print("EASYANNOTATE_RESULT " + json.dumps({"best": str(best), "metrics": {}}), flush=True)
'''


def test_cancelling_a_training_job_stops_it(local, tmp_path, monkeypatch):
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "slow_trainer.py").write_text(SLOW_TRAINER, encoding="utf-8")
    monkeypatch.setenv("EASYANNOTATE_TRAIN_MODULE", "slow_trainer")
    monkeypatch.setenv("DATAFACTORY_TOOLS_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(mods))
    project = local.post("/api/local/open", json={"path": str(_folder(tmp_path, "A"))}, headers=H).json()["project"]
    job = local.post("/api/local/train", json={"project": project, "predict": False}, headers=H).json()
    time.sleep(1.0)
    assert local.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
    job = _wait(local, job["id"])
    assert job["state"] == "cancelled", f"cancel accepted, but the run went on to {job['state']!r}"


# ------------------------------------------------- 6. hardware: Linux R525 drivers

@pytest.mark.parametrize("driver", ["525.105.17", "525.116.03", "525.125.06", "525.147.05"])
def test_linux_r525_drivers_get_a_cuda_build(driver):
    # Once compared as floats (525.105 < 525.60) and sent to the CPU. Now
    # compared as versions -- and given the build measured to run on R52x.
    assert hardware._cuda_build_for(driver, "Linux") == "cu118"


@pytest.mark.parametrize("driver, os_name, build", [
    ("529.04", "Windows", "cu118"),     # this machine: cu126 failed, cu118 ran
    ("560.76", "Windows", "cu126"),
    ("566.36", "Windows", "cu126"),
    ("572.16", "Windows", "cu128"),
    ("560.35.03", "Linux", "cu126"),
    ("470.82.01", "Linux", None),
])
def test_each_driver_gets_the_build_made_for_it(driver, os_name, build):
    assert hardware._cuda_build_for(driver, os_name) == build


def test_an_old_driver_gets_pinned_packages_and_says_why():
    hw = hardware.Hardware(os="Windows", cpu_count=8, ram_gb=8,
                           gpus=[hardware.Gpu("NVIDIA GeForce RTX 2050", 4.0, "529.04")])
    chosen = hardware.plan(hw)
    assert chosen.torch_packages == ("torch==2.7.1", "torchvision==0.22.1")
    assert chosen.torch_index.endswith("/cu118")
    assert any("driver update" in warning for warning in chosen.warnings)


def test_plan_for_a_12gb_card_on_driver_525_105():
    hw = hardware.Hardware(os="Linux", cpu_count=8, ram_gb=32,
                           gpus=[hardware.Gpu("NVIDIA RTX A2000 12GB", 12.0, "525.105.17")])
    chosen = hardware.plan(hw)
    assert chosen.device == "cuda", chosen.warnings


# ------------------------------------------------- 7. installer: probe failure

def _fake_env(tmp_path, monkeypatch, *, probe_rc, probe_out, popen_rc=0):
    from data_factory.annotate.tools import install as installer

    home = tmp_path / "toolshome"
    monkeypatch.setenv("DATAFACTORY_TOOLS_HOME", str(home))
    python = installer.tools_python(home)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")                          # the venv "exists"
    monkeypatch.setattr(hardware, "detect", lambda: hardware.Hardware(os="Windows", cpu_count=8, ram_gb=16))
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "bundled_wheels", lambda: [])
    commands = []

    class Proc:
        def __init__(self, cmd, **kw):
            commands.append(cmd)
            self.stdout = iter(["ok\n"])

        def wait(self):
            return popen_rc

        def poll(self):
            return popen_rc

    monkeypatch.setattr(installer.subprocess, "Popen", Proc)
    monkeypatch.setattr(installer.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], probe_rc, probe_out, "ImportError: DLL load failed"))
    return installer, home, commands


def test_a_torch_that_does_not_import_is_not_installed(tmp_path, monkeypatch):
    installer, home, _ = _fake_env(tmp_path, monkeypatch, probe_rc=1, probe_out="")
    try:
        installer.install(on_line=lambda line: None)
    except installer.InstallError:
        pass
    state = installer.status()
    assert not state["installed"], f"torch cannot import, yet installed=True: {state['profile']}"


def test_a_failed_reinstall_does_not_keep_reporting_installed(tmp_path, monkeypatch):
    installer, home, _ = _fake_env(tmp_path, monkeypatch, probe_rc=0, probe_out="cuda", popen_rc=1)
    (home / "profile.json").write_text(json.dumps({"device": "cuda", "summary": "old", "verified_device": "cuda"}))
    with pytest.raises(installer.InstallError):
        installer.install(on_line=lambda line: None)
    assert not installer.status()["installed"], "the reinstall failed half-way; profile still says installed"


# ---------------- 8. install job: the running sidecar is shut only AFTER the install

def test_install_stops_the_sidecar_before_touching_its_environment(local, monkeypatch):
    from data_factory.annotate.tools import install as installer

    order = []

    class Backend:
        def close(self):
            order.append("sidecar closed")

    REGISTRY._backend, REGISTRY._resolved = Backend(), True

    def fake_install(**kw):
        order.append("pip/uv replaces torch")
        return {"device": "cpu", "summary": "x", "verified_device": "cpu"}

    monkeypatch.setattr(installer, "install", fake_install)
    try:
        job = local.post("/api/local/tools/install", json={}, headers=H).json()
        _wait(local, job["id"])
    finally:
        REGISTRY._backend, REGISTRY._resolved = None, False
    assert order[0] == "sidecar closed", order


# ------------------------------------------ found measuring on this machine


def test_a_cuda_build_that_cannot_run_falls_back_to_the_cpu(tmp_path, monkeypatch):
    """torch.cuda.is_available() said True on a 529 driver with the cu126
    build, and every allocation failed. Whatever the plan, the probe does
    real work on the device, and a build that cannot is not reported as one."""
    probe = 'PROBE {"device": "cpu", "torch": "2.14.1+cu126", "error": "AcceleratorError: device busy or unavailable"}'
    installer, home, _ = _fake_env(tmp_path, monkeypatch, probe_rc=0, probe_out=probe)
    monkeypatch.setattr(hardware, "detect", lambda: hardware.Hardware(
        os="Windows", cpu_count=8, ram_gb=8, gpus=[hardware.Gpu("RTX 2050", 4.0, "566.36")]))
    lines = []
    profile = installer.install(on_line=lines.append)
    assert profile["device"] == "cpu" and profile["verified_device"] == "cpu"
    assert profile["sam"] == "sam2.1_t.pt"
    assert any("busy or unavailable" in line for line in lines)


def test_the_models_are_sized_for_the_card_they_run_on():
    hw = hardware.Hardware(os="Linux", cpu_count=8, ram_gb=32, gpus=[
        hardware.Gpu("GT 1030", 2.0, "570.10", index=0),
        hardware.Gpu("RTX 3060", 12.0, "570.10", index=1),
    ])
    chosen = hardware.plan(hw)
    assert (chosen.device, chosen.gpu_index, chosen.sam) == ("cuda", 1, "sam2.1_b.pt")


def test_old_folder_imports_get_their_predictions_counted(tmp_path):
    import sqlite3

    from data_factory.annotate import store as store_mod

    path = tmp_path / "annotations.db"
    root = tmp_path / "doors"
    make_image(root / "a.jpg", 1, size=(120, 90))
    (root / "a.xml").write_text(
        "<annotation><filename>a.jpg</filename><size><width>120</width><height>90</height>"
        "<depth>3</depth></size><object><name>door</name><score>0.7</score><bndbox>"
        "<xmin>5</xmin><ymin>5</ymin><xmax>60</xmax><ymax>80</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    s = Store(path)
    project = attach.open_folder(s, root)["project"]
    s.close()
    with sqlite3.connect(path) as raw:            # as a 0.4.0-dev import left it
        raw.execute("UPDATE image SET predicted_count = 0")
        raw.execute("PRAGMA user_version = 6")
    upgraded = Store(path)
    try:
        image = images.by_path(upgraded, project, "a.jpg")
        assert upgraded.one("SELECT predicted_count FROM image WHERE id = ?", (image.id,))[0] == 1
        assert upgraded.one("PRAGMA user_version")[0] == store_mod.SCHEMA_VERSION
    finally:
        upgraded.close()


def test_the_cli_package_exposes_main_without_recursing():
    # A fresh interpreter, where the lazy __getattr__ really runs. (Once the
    # submodule cli.main is imported, Python binds the package attribute
    # "main" to the module itself -- a name clash, not this bug.)
    done = subprocess.run(
        [sys.executable, "-c", "import data_factory.cli as cli; print(callable(cli.build_parser))"],
        capture_output=True, text=True, timeout=120,
    )
    assert done.stdout.strip() == "True", done.stderr[-500:]


def test_a_library_printing_to_stdout_does_not_lose_the_reply(tmp_path):
    """Ultralytics' download bar went to stdout with no newline, the reply was
    glued to it, and the editor reported "the tool exited" on a first prompt."""
    import os

    from data_factory.annotate.tools.local import LocalTool

    image = make_image(tmp_path / "a.jpg", 1)
    tool = LocalTool([sys.executable, "-m", FAKE], env={**os.environ, "DFX_FAKE_NOISE": "1"})
    try:
        tool.prepare("fake.box", "x", None)
        for _ in range(3):
            reply = tool.invoke("fake.box", {"image_id": "x", "image_path": str(image), "box": [1, 1, 9, 9]})
            assert reply["shapes"]
    finally:
        tool.close()


def test_the_client_salvages_a_reply_glued_to_junk():
    from data_factory.annotate.tools.local import _salvage

    line = b'\r\x1b[KDownloading 0% \xe2\x94\x80\r\x1b[K 100%{"v": 1, "id": 7, "ok": true, "result": {}}\n'
    assert _salvage(line)["id"] == 7
    assert _salvage(b"\r\x1b[Kjust a progress bar\n") is None
