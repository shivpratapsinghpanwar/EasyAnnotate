"""The heavy-tool boundary.

These are the tests that let us put SAM behind a process boundary with
confidence, and they run in about a second because the tool under test is a
fake that sleeps instead of a model that needs a GPU.

Most of them assert a *failure* behaves well. The legacy SAM script hung
permanently because its worker caught an exception, printed it, and replied
with nothing while the UI blocked on a read. Several tests below exist purely
so that cannot happen here.
"""

from __future__ import annotations

import sys
import time

import pytest

from data_factory.annotate.tools import ToolError, ToolNotReady, ToolUnavailable
from data_factory.annotate.tools.local import LocalTool, SidecarClient
from data_factory.annotate.tools.protocol import (
    ToolHealth,
    ToolSpec,
    decode_message,
    encode_message,
    fail,
    ok,
    raise_for,
)
from data_factory.annotate.tools.registry import NullTool, ToolRegistry

FAKE = [sys.executable, "-m", "data_factory.annotate.tools.fake"]


@pytest.fixture
def tool():
    client = LocalTool(FAKE)
    try:
        yield client
    finally:
        client.close()


class TestFraming:
    def test_a_message_round_trips(self):
        payload = {"id": 1, "method": "invoke", "params": {"box": [1, 2, 3, 4]}}
        assert decode_message(encode_message(payload)) == payload

    def test_frames_never_contain_a_newline(self):
        """One object per line is the entire framing rule."""
        encoded = encode_message({"id": 1, "text": "a\nb\r\nc"})
        assert encoded.count(b"\n") == 1 and encoded.endswith(b"\n")

    def test_unicode_survives(self):
        payload = {"id": 1, "label": "pörta — 門"}
        assert decode_message(encode_message(payload))["label"] == "pörta — 門"

    def test_junk_is_rejected_with_a_useful_message(self):
        with pytest.raises(ToolError, match="unparseable"):
            decode_message("not json at all")

    def test_a_bare_array_is_not_a_frame(self):
        with pytest.raises(ToolError, match="must be an object"):
            decode_message("[1, 2, 3]")

    def test_error_replies_map_back_to_the_right_exception(self):
        with pytest.raises(ToolNotReady) as caught:
            raise_for(fail(1, ToolNotReady("cold", retry_after_ms=250)))
        assert caught.value.retry_after_ms == 250

        with pytest.raises(ToolUnavailable):
            raise_for(fail(1, ToolUnavailable("gone")))

    def test_a_success_reply_yields_its_result(self):
        assert raise_for(ok(1, {"shapes": []})) == {"shapes": []}


class TestSpecs:
    def test_a_spec_round_trips(self):
        spec = ToolSpec(name="sam2.box", title="SAM", summary="s", requires=("gpu",))
        assert ToolSpec.from_dict(spec.as_dict()) == spec

    def test_a_spec_without_a_name_is_refused(self):
        with pytest.raises(ToolError, match="needs a name"):
            ToolSpec.from_dict({"title": "nameless"})

    def test_health_readiness(self):
        assert ToolHealth("ready").ready
        assert not ToolHealth("loading").ready
        assert not ToolHealth("unavailable", "no gpu").ready


class TestTheHappyPath:
    def test_describe_returns_the_tools(self, tool: LocalTool):
        specs = tool.describe()
        assert [s.name for s in specs] == ["fake.box"]
        assert specs[0].kind == "interactive"

    def test_health_is_ready(self, tool: LocalTool):
        assert tool.health("fake.box").ready

    def test_prepare_then_invoke_returns_a_shape(self, tool: LocalTool):
        tool.prepare("fake.box", "img-1")
        result = tool.invoke("fake.box", {"image_id": "img-1", "box": [10, 20, 60, 80]})
        assert result["shapes"][0]["bbox"] == [10, 20, 60, 80]
        assert result["shapes"][0]["confidence"] == 0.99

    def test_the_result_parses_as_a_Shape(self, tool: LocalTool):
        """The tool's output must drop straight into the annotation model."""
        from data_factory.core.annotations.model import Shape

        tool.prepare("fake.box", "img-1")
        result = tool.invoke("fake.box", {"image_id": "img-1", "box": [5, 5, 25, 45]})
        shape = Shape.from_dict(result["shapes"][0])
        assert shape.bbox == (5, 5, 25, 45)
        assert shape.has_polygon and shape.confidence == 0.99

    def test_many_calls_stay_matched_to_their_own_replies(self, tool: LocalTool):
        """Correlation by id. The legacy tool had one untagged queue, so a
        stale reply could be consumed by the wrong caller."""
        tool.prepare("fake.box", "img-1")
        for index in range(25):
            box = [index, index, index + 10, index + 10]
            result = tool.invoke("fake.box", {"image_id": "img-1", "box": box})
            assert result["shapes"][0]["bbox"] == box

    def test_concurrent_callers_each_get_their_own_reply(self, tool: LocalTool):
        from concurrent.futures import ThreadPoolExecutor

        tool.prepare("fake.box", "img-1")

        def call(index: int):
            box = [index, index, index + 5, index + 5]
            return index, tool.invoke("fake.box", {"image_id": "img-1", "box": box})

        with ThreadPoolExecutor(max_workers=8) as pool:
            for index, result in pool.map(call, range(24)):
                assert result["shapes"][0]["bbox"] == [index, index, index + 5, index + 5]


class TestFailuresBehaveWell:
    def test_a_cold_image_is_not_ready_rather_than_an_error(self, tool: LocalTool):
        """The UI retries this; it does not show the user a failure."""
        with pytest.raises(ToolNotReady) as caught:
            tool.invoke("fake.box", {"image_id": "never-prepared", "box": [0, 0, 1, 1]})
        assert caught.value.retry_after_ms == 250

    def test_an_exception_in_the_handler_still_produces_a_reply(self, tool: LocalTool):
        """THE test. This is the exact shape of the legacy deadlock."""
        with pytest.raises(ToolError, match="blew up"):
            tool.invoke("fake.box", {"image_id": "img", "behaviour": "raise"})
        # And the tool is still usable afterwards.
        tool.prepare("fake.box", "img-2")
        assert tool.invoke("fake.box", {"image_id": "img-2", "box": [1, 1, 2, 2]})["shapes"]

    def test_a_base_exception_also_produces_a_reply(self, tool: LocalTool):
        """SystemExit is not an Exception. Catching only Exception would hang."""
        with pytest.raises(ToolError):
            tool.invoke("fake.box", {"image_id": "img", "behaviour": "silent"})

    def test_an_unknown_method_is_an_error_not_a_silence(self, tool: LocalTool):
        with pytest.raises(ToolError, match="unknown method"):
            tool.client.call("nonsense")

    def test_a_hanging_handler_times_out_rather_than_blocking_forever(self, tool: LocalTool):
        started = time.monotonic()
        with pytest.raises(ToolError, match="longer than"):
            tool.client.call("invoke", {"tool": "fake.box", "behaviour": "hang"}, timeout=0.75)
        assert time.monotonic() - started < 5, "the timeout did not fire"

    def test_a_crash_mid_request_fails_that_request_immediately(self, tool: LocalTool):
        started = time.monotonic()
        with pytest.raises(ToolError):
            tool.invoke("fake.box", {"image_id": "img", "behaviour": "exit"})
        # Failed because the child died, not because a 15s timeout elapsed.
        assert time.monotonic() - started < 5

    def test_the_tool_recovers_after_a_crash(self, tool: LocalTool):
        with pytest.raises(ToolError):
            tool.invoke("fake.box", {"image_id": "img", "behaviour": "exit"})
        tool.prepare("fake.box", "img-3")
        result = tool.invoke("fake.box", {"image_id": "img-3", "box": [2, 2, 8, 8]})
        assert result["shapes"][0]["bbox"] == [2, 2, 8, 8]

    def test_repeated_crashes_eventually_stop_being_retried(self):
        """A tool that dies every time is broken; retrying forever would hide
        that while burning CPU."""
        client = LocalTool(FAKE)
        try:
            for _ in range(6):
                try:
                    client.invoke("fake.box", {"image_id": "i", "behaviour": "exit"})
                except ToolError:
                    pass
            with pytest.raises(ToolUnavailable, match="will not be restarted"):
                client.invoke("fake.box", {"image_id": "i", "box": [0, 0, 1, 1]})
        finally:
            client.close()

    def test_a_command_that_does_not_exist_is_unavailable_not_a_crash(self):
        client = SidecarClient(["definitely-not-a-real-program-xyz"])
        with pytest.raises(ToolUnavailable):
            client.call("describe")

    def test_health_never_raises(self):
        """The UI calls health to decide whether to draw a button at all."""
        client = LocalTool(["definitely-not-a-real-program-xyz"])
        assert client.health().state == "unavailable"

    def test_a_tool_that_prints_junk_to_stdout_does_not_break_the_session(self, tool: LocalTool):
        """Tools print. A stray print must not be mistaken for a reply."""
        tool.prepare("fake.box", "img-1")
        assert tool.invoke("fake.box", {"image_id": "img-1", "box": [0, 0, 4, 4]})["shapes"]


class TestAbsenceIsNormal:
    """The base wheel ships with no heavy tools. That is a supported state."""

    def test_the_null_backend_offers_nothing(self):
        assert NullTool().describe() == []
        assert not NullTool().health().ready

    def test_invoking_a_missing_tool_says_so_clearly(self):
        with pytest.raises(ToolUnavailable, match="no heavy tools are installed"):
            NullTool().invoke("sam2.box", {})

    def test_an_unconfigured_registry_is_empty_and_that_is_fine(self, monkeypatch):
        monkeypatch.delenv("DATAFACTORY_TOOLS_MODULE", raising=False)
        payload = ToolRegistry().as_dict()
        assert payload["tools"] == []
        assert "no tools module" in payload["detail"]

    def test_tools_can_be_switched_off_even_when_installed(self, monkeypatch):
        monkeypatch.setenv("DATAFACTORY_TOOLS_MODULE", "data_factory.annotate.tools.fake")
        monkeypatch.setenv("DATAFACTORY_TOOLS", "0")
        assert ToolRegistry().as_dict()["tools"] == []

    def test_a_broken_tools_module_degrades_to_empty_rather_than_raising(self, monkeypatch):
        monkeypatch.setenv("DATAFACTORY_TOOLS_MODULE", "data_factory.nope.not_a_module")
        monkeypatch.setenv("DATAFACTORY_TOOLS", "1")
        payload = ToolRegistry().as_dict()
        assert payload["tools"] == []
        assert payload["detail"]

    def test_prepare_never_raises_even_with_no_backend(self):
        """It is called speculatively on every navigation."""
        ToolRegistry(NullTool()).prepare("sam2.box", "img-1")

    def test_a_configured_tool_is_discovered_and_listed(self, monkeypatch):
        monkeypatch.setenv("DATAFACTORY_TOOLS_MODULE", "data_factory.annotate.tools.fake")
        monkeypatch.setenv("DATAFACTORY_TOOLS", "1")
        registry = ToolRegistry()
        try:
            payload = registry.as_dict()
            assert [t["name"] for t in payload["tools"]] == ["fake.box"]
            assert payload["tools"][0]["health"]["state"] == "ready"
        finally:
            registry.close()


class TestNoHeavyImports:
    def test_the_tools_package_does_not_pull_in_torch(self):
        """The entire reason for the process boundary."""
        import data_factory.annotate.tools as pkg  # noqa: F401

        assert "torch" not in sys.modules

    def test_a_fresh_interpreter_running_the_sidecar_never_imports_torch(self):
        """Checked in a subprocess, because this process may have imported it.

        numpy and cv2 *are* pulled in, by the parent package's __init__ — and
        that is fine: they are the base wheel's own dependencies, present in
        any environment that has torch anyway. torch is the one that costs
        seconds and half a gigabyte, and it must never appear.
        """
        import subprocess

        probe = subprocess.run(
            [sys.executable, "-c",
             "import data_factory.annotate.tools.fake, sys; print('torch' in sys.modules)"],
            capture_output=True, text=True, check=True,
        )
        assert probe.stdout.strip() == "False"
