"""The installer brings YOLO-World's CLIP itself, with no git.

Found by installing from PyPI on a machine without git on PATH: Ultralytics'
own first-use install of CLIP needs git, failed, and every text prompt said
"No module named 'clip'". SAM worked, so nothing else showed it.
"""

import subprocess
from pathlib import Path

import pytest

from data_factory import hardware
from data_factory.annotate.tools import install as installer


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "toolshome"
    monkeypatch.setenv("DATAFACTORY_TOOLS_HOME", str(home))
    python = installer.tools_python(home)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    monkeypatch.setattr(hardware, "detect", lambda: hardware.Hardware(os="Windows", cpu_count=8, ram_gb=16))
    monkeypatch.setattr(installer, "bundled_wheels", lambda: [])
    state = {"commands": [], "fail": None}

    class Proc:
        def __init__(self, cmd, **kw):
            self.cmd = [str(part) for part in cmd]
            state["commands"].append(self.cmd)
            self.stdout = iter(["ok\n"])

        def wait(self):
            return 1 if state["fail"] and state["fail"](self.cmd) else 0

        def poll(self):
            return self.wait()

    monkeypatch.setattr(installer.subprocess, "Popen", Proc)
    probe = 'PROBE {"device": "cpu", "torch": "2.7.1", "error": ""}\n'
    monkeypatch.setattr(installer.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, probe, ""))
    return state


def _clip_after_the_tools(commands, tools_part):
    """CLIP is installed, into the same environment, after the tools."""
    tools = next(i for i, c in enumerate(commands) if any(tools_part in part for part in c))
    clip = next(i for i, c in enumerate(commands) if installer.CLIP in c)
    assert clip > tools, commands
    assert commands[clip][:-1] == commands[tools][:commands[clip].index(installer.CLIP)], "same pip, same environment"


@pytest.mark.parametrize("uv", ["uv", None])
def test_clip_is_installed_with_the_tools_from_pypi(env, monkeypatch, uv):
    monkeypatch.setattr(installer, "find_uv", lambda: uv)
    installer.install(on_line=lambda line: None)
    _clip_after_the_tools(env["commands"], "easyannotate-tools==")


def test_clip_is_installed_from_a_checkout_and_from_bundled_wheels(env, monkeypatch, tmp_path):
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    installer.install(source=tmp_path / "checkout", on_line=lambda line: None)
    _clip_after_the_tools(env["commands"], "checkout")

    env["commands"].clear()
    wheel = tmp_path / "easyannotate_tools-0.4.0-py3-none-any.whl"
    monkeypatch.setattr(installer, "bundled_wheels", lambda: [wheel])
    installer.install(on_line=lambda line: None)
    _clip_after_the_tools(env["commands"], "easyannotate_tools-0.4.0")


def test_no_github_costs_text_prompts_not_the_install(env, monkeypatch):
    """A network that reaches PyPI but blocks github.com: SAM must still install."""
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    env["fail"] = lambda cmd: installer.CLIP in cmd
    lines = []
    profile = installer.install(on_line=lines.append)
    assert profile["verified_device"] == "cpu"
    assert installer.status()["installed"]
    assert any("CLIP did not install" in line for line in lines), lines


def test_a_failed_tools_install_still_fails(env, monkeypatch):
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    env["fail"] = lambda cmd: any("easyannotate-tools==" in part for part in cmd)
    with pytest.raises(installer.InstallError):
        installer.install(on_line=lambda line: None)


def test_clip_needs_no_git_and_is_pinned():
    name, _, url = installer.CLIP.partition(" @ ")
    assert name == "clip"
    assert url.startswith("https://github.com/ultralytics/CLIP/archive/") and url.endswith(".zip")
    assert "git+" not in installer.CLIP
    commit = Path(url).stem
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "pin a commit, not a branch"
