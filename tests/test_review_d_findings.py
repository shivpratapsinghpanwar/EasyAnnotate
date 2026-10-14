"""Regressions an independent review found in the hardening and GPU fixes.

Each reproduction is the reviewer's, and failed before its fix.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import types

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from data_factory import hardware
from test_local_link import LOOPBACK, TOKEN, _wait, app_module, local, ok_headers  # noqa: F401


def test_a_small_card_on_an_old_driver_gets_the_cpu():
    for driver in ("530.30", "566.36"):
        hw = hardware.Hardware(os="Windows", cpu_count=8, ram_gb=16,
                               gpus=[hardware.Gpu(name="MX150", vram_gb=2.0, driver=driver)])
        chosen = hardware.plan(hw)
        assert (chosen.device, chosen.sam) == ("cpu", "sam2.1_t.pt"), driver


def test_without_a_profile_the_machine_decides_the_device(monkeypatch, tmp_path):
    """A tools Python named by DATAFACTORY_TOOLS_PYTHON has no profile; it
    used to be forced onto the CPU for inference and training alike."""
    from data_factory.annotate import loop

    monkeypatch.setenv("DATAFACTORY_TOOLS_HOME", str(tmp_path))
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tools"))
    from easyannotate_tools import sidecar

    assert sidecar._device() == "cuda"
    assert loop._train_device() == ""
    (tmp_path / "profile.json").write_text(json.dumps({"device": "cuda", "gpu_index": 1}))
    assert sidecar._device() == "cuda:1"
    assert loop._train_device() == "1"
    (tmp_path / "profile.json").write_text(json.dumps({"device": "cpu"}))
    assert sidecar._device() == "cpu"
    assert loop._train_device() == "cpu"


def test_no_sidecar_starts_while_the_tools_are_installed(local, monkeypatch):  # noqa: F811 -- a fixture
    from data_factory.annotate.tools import install as installer
    from data_factory.annotate.tools.registry import REGISTRY

    monkeypatch.setenv("DATAFACTORY_TOOLS_PYTHON", sys.executable)
    monkeypatch.setenv("DATAFACTORY_TOOLS_MODULE", "data_factory.annotate.tools.fake")
    REGISTRY.close()
    started, release = threading.Event(), threading.Event()

    def fake_install(*, source=None, device=None, on_line=print):
        started.set()
        release.wait(20)
        return {"device": "cpu", "summary": "x", "verified_device": "cpu"}

    monkeypatch.setattr(installer, "install", fake_install)
    try:
        job = local.post("/api/local/tools/install", json={"device": "cpu"}, headers=ok_headers()).json()
        assert started.wait(10)
        listing = local.get("/api/tools").json()      # the Assist panel, mid-install
        alive = REGISTRY._backend is not None and getattr(REGISTRY._backend, "client", None) is not None
    finally:
        release.set()
        _wait(local, job["id"])
    try:
        assert not alive, "a sidecar was started from the environment being replaced"
        assert listing["tools"] == []
        assert [t["name"] for t in local.get("/api/tools").json()["tools"]], "tools did not come back"
    finally:
        REGISTRY.close()


def test_the_fence_stands_aside_for_an_api_key(app_module, monkeypatch):  # noqa: F811 -- a fixture
    """A same-machine reverse proxy forwards its public Host; the key, not
    the Host, is what stops a rebinding page there."""
    monkeypatch.setenv("EASYANNOTATE_LOOPBACK", "1")
    monkeypatch.setenv("DATAFACTORY_API_KEY", "k")
    import importlib

    importlib.reload(app_module)
    with TestClient(app_module.app, base_url="http://localhost:8765", client=LOOPBACK) as client:
        proxied = client.get("/api/health", headers={"host": "annotate.example.com", "x-api-key": "k"})
    assert proxied.status_code == 200


def test_the_probe_is_found_after_glued_output(tmp_path, monkeypatch):
    from data_factory.annotate.tools import install as installer

    home = tmp_path / "toolshome"
    monkeypatch.setenv("DATAFACTORY_TOOLS_HOME", str(home))
    python = installer.tools_python(home)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    monkeypatch.setattr(hardware, "detect", lambda: hardware.Hardware(os="Windows", cpu_count=8, ram_gb=16))
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "bundled_wheels", lambda: [])

    class Proc:
        def __init__(self, cmd, **kw):
            self.stdout = iter(["ok\n"])

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(installer.subprocess, "Popen", Proc)
    glued = '\r\x1b[Kloading 100%PROBE {"device": "cpu", "torch": "2.7.1", "error": ""}\n'
    monkeypatch.setattr(installer.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, glued, ""))
    profile = installer.install(on_line=lambda line: None)
    assert profile["verified_device"] == "cpu" and profile["torch"] == "2.7.1"
