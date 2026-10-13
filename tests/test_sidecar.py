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


class FakePredictor:
    """Ultralytics' SAM predictor: encode once with set_image, then prompt."""

    def __init__(self):
        self.image = None
        self.encodes = 0

    def set_image(self, image):
        self.image = image
        self.encodes += 1

    def __call__(self, **kwargs):
        assert self.image is not None, "prompted before any image was encoded"
        mask = np.zeros((90, 120), dtype=np.uint8)
        if "bboxes" in kwargs:
            x1, y1, x2, y2 = (int(v) for v in kwargs["bboxes"][0])
            mask[y1:y2, x1:x2] = 1
        else:
            px, py = (int(v) for v in kwargs["points"][0])
            mask[max(0, py - 10) : py + 10, max(0, px - 10) : px + 10] = 1
        return [_Result(masks=[mask])]


class FakeSAM:
    def __init__(self, weights):
        self.weights = weights
        self.predictor = None
        self.full_predicts = 0

    def predict(self, source, **kwargs):
        # The real thing builds a fresh predictor state here -- which is why
        # the sidecar may call it only once, to warm up.
        self.full_predicts += 1
        self.predictor = FakePredictor()
        return [_Result(masks=[np.zeros((64, 64), dtype=np.uint8)])]


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
    assert [s["name"] for s in specs] == ["sam.box", "sam.point", "detect.text", "detect.model"]
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


# ------------------------------------------------------------- encode-ahead


def test_one_encode_serves_every_prompt_on_an_image(fake_ultralytics, image, tmp_path):
    """SAM costs ~2.5s per encode and ~0.15s per prompt on a CPU. Encoding the
    open image once, ahead of the first click, is the whole speed-up."""
    tool = sidecar.SidecarTool()
    tool.prepare({"tool": "sam.box", "source": str(image)})
    _ready(tool, tool.sam)
    for _ in range(50):
        if tool._sam_key is not None:
            break
        time.sleep(0.02)
    predictor = tool.sam.model.predictor
    assert predictor.encodes == 1

    base = {"image_path": str(image), "width": 120, "height": 90}
    tool.invoke({**base, "tool": "sam.box", "box": [20, 20, 70, 60]})
    tool.invoke({**base, "tool": "sam.point", "points": [[60, 45]]})
    tool.invoke({**base, "tool": "sam.box", "box": [5, 5, 40, 30]})
    assert predictor.encodes == 1, "a prompt re-encoded an image that was already encoded"
    assert tool.sam.model.full_predicts == 1, "a full predict would wipe the cached features"

    other = make_image(tmp_path / "b.jpg", 2, size=(120, 90))
    tool.invoke({"image_path": str(other), "width": 120, "height": 90, "tool": "sam.box", "box": [1, 1, 30, 30]})
    assert predictor.encodes == 2 and predictor.image == str(other)


def test_a_superseded_prepare_does_no_work(fake_ultralytics, image, tmp_path):
    """Holding N fires a prepare per image; only the newest is worth encoding."""
    tool = sidecar.SidecarTool()
    with pytest.raises(ToolNotReady):
        tool.sam.get()
    _ready(tool, tool.sam)
    stale = sidecar._image_key(image)
    tool._want_key = "a-newer-image"
    tool._encode_when_ready(image, stale)
    assert tool.sam.model.predictor.encodes == 0
    assert tool._sam_key is None


def test_the_sidecar_keeps_its_weights_in_one_place(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYANNOTATE_MODELS", str(tmp_path / "models"))
    assert sidecar.models_dir() == tmp_path / "models"
    assert (tmp_path / "models").is_dir()
