"""The real sidecar's logic, with Ultralytics stood in for.

Torch is never installed in the editor's environment, so the models here are
fakes with the same shape of answer; the real models are exercised by
``easyannotate tools status`` and by hand. What this pins down is everything
around them: lazy loading, not-ready replies, prompt parsing, and that a
mask comes back as an editable outline.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from conftest import make_image  # noqa: E402
from data_factory.annotate.tools.protocol import ToolError, ToolNotReady  # noqa: E402
from easyannotate_tools import sidecar  # noqa: E402


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def tolist(self):
        return self.array.tolist()


class _Masks:
    def __init__(self, masks):
        self.data = [_Tensor(m) for m in masks]


class _Boxes:
    def __init__(self, rows):
        self.xyxy = _Tensor([r[:4] for r in rows])
        self.conf = _Tensor([r[4] for r in rows])
        self.cls = _Tensor([r[5] for r in rows])


class _Result:
    def __init__(self, masks=None, boxes=None):
        self.masks = _Masks(masks) if masks is not None else None
        self.boxes = _Boxes(boxes) if boxes is not None else None


class FakeSAM:
    def __init__(self, weights):
        self.weights = weights

    def predict(self, source, **kwargs):
        mask = np.zeros((90, 120), dtype=np.uint8)
        if "bboxes" in kwargs:
            x1, y1, x2, y2 = (int(v) for v in kwargs["bboxes"][0])
            mask[y1:y2, x1:x2] = 1
        else:
            px, py = (int(v) for v in kwargs["points"][0])
            mask[max(0, py - 10) : py + 10, max(0, px - 10) : px + 10] = 1
        return [_Result(masks=[mask])]


class FakeWorld:
    def __init__(self, weights):
        self.classes: list[str] = []

    def set_classes(self, classes):
        self.classes = list(classes)

    def predict(self, source, **kwargs):
        rows = [[10, 10, 50, 40, 0.9, 0], [60, 20, 100, 80, 0.3, 0]]
        if len(self.classes) > 1:
            rows.append([5, 60, 30, 85, 0.7, 1])
        return [_Result(boxes=rows)]


@pytest.fixture
def fake_ultralytics(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.SAM = FakeSAM
    module.YOLOWorld = FakeWorld
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return module


@pytest.fixture
def image(tmp_path):
    return make_image(tmp_path / "a.jpg", 1, size=(120, 90))


def _ready(tool, lazy):
    for _ in range(50):
        if lazy.state == "ready":
            return
        time.sleep(0.02)
    raise AssertionError(f"{lazy.name} never loaded: {lazy.error}")


def test_the_handshake_needs_no_torch():
    tool = sidecar.SidecarTool()
    specs = tool.describe()
    assert [s["name"] for s in specs] == ["sam.box", "sam.point", "detect.text"]
    assert tool.health({"tool": "sam.box"})["state"] == "ready"


def test_a_cold_model_answers_not_ready_then_works(fake_ultralytics, image):
    tool = sidecar.SidecarTool()
    params = {"tool": "sam.box", "image_path": str(image), "width": 120, "height": 90, "box": [20, 20, 70, 60]}
    with pytest.raises(ToolNotReady):
        tool.invoke(params)
    _ready(tool, tool.sam)
    reply = tool.invoke(params)
    shape = reply["shapes"][0]
    assert shape["polygon"] and len(shape["polygon"]) <= 12  # simplified, not a raster edge
    assert shape["bbox"] == [20, 20, 69, 59]


def test_a_click_outlines_what_is_under_it(fake_ultralytics, image):
    tool = sidecar.SidecarTool()
    params = {"tool": "sam.point", "image_path": str(image), "width": 120, "height": 90, "points": [[60, 45]]}
    with pytest.raises(ToolNotReady):
        tool.invoke(params)
    _ready(tool, tool.sam)
    shape = tool.invoke(params)["shapes"][0]
    assert shape["bbox"] == [50, 35, 69, 54]


def test_text_prompts_label_what_they_find(fake_ultralytics, image):
    tool = sidecar.SidecarTool()
    params = {"tool": "detect.text", "image_path": str(image), "width": 120, "height": 90, "text": "door, handle"}
    with pytest.raises(ToolNotReady):
        tool.invoke(params)
    _ready(tool, tool.world)
    shapes = tool.invoke(params)["shapes"]
    assert [s["label"] for s in shapes] == ["door", "door", "handle"]
    assert shapes[0]["confidence"] == 0.9 and shapes[0]["polygon"] is None


def test_bad_requests_are_refused_not_crashed(fake_ultralytics, image):
    tool = sidecar.SidecarTool()
    with pytest.raises(ToolError):
        tool.invoke({"tool": "sam.box", "image_path": str(image), "box": [1, 2]})
    with pytest.raises(ToolError):
        tool.invoke({"tool": "detect.text", "image_path": str(image), "text": ""})
    with pytest.raises(ToolError):
        tool.invoke({"tool": "nope", "image_path": str(image)})
    with pytest.raises(ToolError):
        tool.invoke({"tool": "sam.box", "image_path": str(image.with_name("gone.jpg")), "box": [1, 1, 5, 5]})


def test_a_model_that_fails_to_load_reports_it(monkeypatch, image):
    module = types.ModuleType("ultralytics")

    def explode(weights):
        raise RuntimeError("no CUDA")

    module.SAM = explode
    module.YOLOWorld = FakeWorld
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    tool = sidecar.SidecarTool()
    params = {"tool": "sam.box", "image_path": str(image), "width": 120, "height": 90, "box": [1, 1, 5, 5]}
    with pytest.raises(ToolNotReady):
        tool.invoke(params)
    for _ in range(50):
        if tool.sam.state == "unavailable":
            break
        time.sleep(0.02)
    with pytest.raises(ToolError) as caught:
        tool.invoke(params)
    assert "no CUDA" in str(caught.value)
    assert tool.health({"tool": "sam.box"})["state"] == "unavailable"
