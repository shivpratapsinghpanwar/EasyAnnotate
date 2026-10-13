"""The desktop window's bridge: a folder picked in the OS dialog becomes a
project, and the page only ever learns the project's id."""

from __future__ import annotations

import sys
import types

import pytest

from conftest import make_image, write_voc
from data_factory import desktop


class FakeWindow:
    def __init__(self, answer):
        self.answer = answer

    def create_file_dialog(self, kind):
        assert kind == "FOLDER"
        return self.answer


@pytest.fixture
def fake_webview(monkeypatch):
    module = types.ModuleType("webview")
    module.FOLDER_DIALOG = "FOLDER"
    monkeypatch.setitem(sys.modules, "webview", module)
    return module


def test_picking_a_folder_opens_it_as_an_attached_project(tmp_path, monkeypatch, fake_webview):
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    from data_factory.service import deps

    deps.reset_stores()
    folder = tmp_path / "doors"
    make_image(folder / "a.jpg", 1)
    write_voc(folder / "a.xml", "Open")

    bridge = desktop.Bridge()
    bridge.attach(FakeWindow([str(folder)]))
    try:
        result = bridge.pick_folder()
        assert "error" not in result
        assert result["images"] == 1 and result["imported"] == 1 and result["writes_back"]
        assert set(result) >= {"project", "name"}
        assert str(folder) not in str(result)  # the page learns an id, never a path
    finally:
        deps.reset_stores()


def test_cancelling_the_dialog_is_not_an_error(fake_webview):
    bridge = desktop.Bridge()
    bridge.attach(FakeWindow(None))
    assert bridge.pick_folder() == {"cancelled": True}


def test_without_a_window_the_browser_path_is_taken(monkeypatch):
    monkeypatch.setenv("EASYANNOTATE_NO_WINDOW", "1")
    assert desktop.open_window("http://127.0.0.1:1") is False


def test_a_free_port_is_found_when_the_preferred_one_is_taken():
    import socket

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        port = taken.getsockname()[1]
        found = desktop.free_port(port)
        assert found != port and found > 0
