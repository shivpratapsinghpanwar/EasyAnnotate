"""The sidecar: SAM 2 and text-prompted detection, over the editor's protocol.

Run as ``python -m easyannotate_tools.sidecar`` from the tools environment.
Everything heavy is imported lazily, inside the functions that need it, so
``describe`` and ``health`` answer in milliseconds and a missing CUDA build
fails one tool call rather than the handshake.

Models load on a background thread the first time they are asked for. Until
then a call answers ``not_ready`` with a retry hint -- the editor shows
"warming up" and the person keeps drawing. Blocking the handshake on a model
load would make the editor look hung for the length of a weights download.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from data_factory.annotate.tools import postprocess
from data_factory.annotate.tools.protocol import ToolError, ToolNotReady, ToolSpec
from data_factory.annotate.tools.serve import serve

SAM_WEIGHTS = "sam2.1_t.pt"
WORLD_WEIGHTS = "yolov8s-worldv2.pt"
#: Below this a YOLO-World hit is noise. The editor's slider filters further.
DEFAULT_MIN_CONFIDENCE = 0.15
MAX_DETECTIONS = 300
#: How long one request waits for its answer before replying not_ready and
#: letting the work continue. Under the wire's 15s invoke timeout, with room.
INLINE_WAIT_S = 8.0
#: Finished answers are kept this long for the retry that collects them.
RESULT_TTL_S = 600.0

SPECS = [
    ToolSpec(
        name="sam.box",
        title="Outline (SAM)",
        summary="Drag a box around an object; SAM 2 traces its outline.",
        kind="interactive",
        inputs={
            "type": "object",
            "properties": {"box": {"type": "array", "items": {"type": "number"}}},
            "required": ["box"],
        },
        version="sam2.1-tiny",
        requires=(),
    ),
    ToolSpec(
        name="sam.point",
        title="Outline from a click (SAM)",
        summary="Click an object; SAM 2 traces its outline.",
        kind="interactive",
        inputs={
            "type": "object",
            "properties": {"points": {"type": "array", "items": {"type": "array"}}},
            "required": ["points"],
        },
        version="sam2.1-tiny",
        requires=(),
    ),
    ToolSpec(
        name="detect.text",
        title="Find every...",
        summary="Type what to find; YOLO-World finds every instance as predictions to review.",
        kind="prompt",
        inputs={
            "type": "object",
            "properties": {"text": {"type": "string"}, "min_confidence": {"type": "number"}},
            "required": ["text"],
        },
        version="yolov8s-worldv2",
        requires=(),
    ),
]


class _Lazy:
    """A model that loads itself on a thread, once, and says when it is not ready."""

    def __init__(self, name: str, loader: Any) -> None:
        self.name = name
        self._loader = loader
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.model: Any = None
        self.error: str = ""

    def get(self) -> Any:
        with self._lock:
            if self.model is not None:
                return self.model
            if self.error:
                raise ToolError(f"{self.name} failed to load: {self.error}", code="unavailable")
            if self._thread is None:
                self._thread = threading.Thread(target=self._load, daemon=True, name=f"load-{self.name}")
                self._thread.start()
        raise ToolNotReady(f"{self.name} is loading", retry_after_ms=1500)

    def _load(self) -> None:
        try:
            model = self._loader()
        except Exception as exc:  # torch, download, CUDA -- all land here
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            self.model = model

    @property
    def state(self) -> str:
        with self._lock:
            if self.model is not None:
                return "ready"
            if self.error:
                return "unavailable"
            return "loading" if self._thread else "cold"


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_sam() -> Any:
    from ultralytics import SAM

    return SAM(SAM_WEIGHTS)


def _load_world() -> Any:
    from ultralytics import YOLOWorld

    model = YOLOWorld(WORLD_WEIGHTS)
    # The first set_classes installs and loads CLIP: a 338 MB download and a
    # torch.jit.load that holds the GIL for seconds. Doing it here, on the
    # loader thread, means it happens once while the tool reports "loading",
    # not in the middle of somebody's first prompt.
    model.set_classes(["object"])
    return model


class _Pending:
    """One computation in flight, or finished and waiting to be collected."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None
        self.finished_at = 0.0


class SidecarTool:
    def __init__(self) -> None:
        self.sam = _Lazy("SAM 2.1", _load_sam)
        self.world = _Lazy("YOLO-World", _load_world)
        self._world_classes: list[str] = []
        # Inference on a CPU can outlast the wire's timeout. A request waits a
        # while, then answers not_ready and leaves the work running; the
        # retry for the same request collects the finished answer. The wire
        # rule -- one reply per request, promptly -- holds on any hardware.
        self._pending: dict[str, _Pending] = {}
        self._pending_lock = threading.Lock()
        # Models are not thread-safe; one inference at a time.
        self._model_lock = threading.Lock()

    # -- protocol ----------------------------------------------------------

    def describe(self) -> list[dict[str, Any]]:
        return [spec.as_dict() for spec in SPECS]

    def health(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("tool") or "")
        lazy = self.world if name.startswith("detect") else self.sam
        return {
            "state": "ready" if lazy.state in ("ready", "cold", "loading") else "unavailable",
            "detail": f"{lazy.name}: {lazy.state}, device {_device()}" + (f" ({lazy.error})" if lazy.error else ""),
        }

    def prepare(self, params: dict[str, Any]) -> None:
        # Start loading the model the moment an image is opened, so the first
        # click pays for the image, not for the weights.
        name = str(params.get("tool") or "")
        try:
            (self.world if name.startswith("detect") else self.sam).get()
        except ToolNotReady:
            pass

    def invoke(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("tool") or "")
        path = _image_path(params)
        key = _request_key(name, path, params)

        with self._pending_lock:
            self._sweep_locked()
            pending = self._pending.get(key)
            if pending is None:
                pending = _Pending()
                self._pending[key] = pending
                threading.Thread(
                    target=self._compute, args=(key, pending, name, path, dict(params)),
                    daemon=True, name=f"invoke-{name}",
                ).start()

        if not pending.done.wait(INLINE_WAIT_S):
            raise ToolNotReady(f"{name} is still working on this image", retry_after_ms=1500)
        with self._pending_lock:
            self._pending.pop(key, None)
        if pending.error is not None:
            raise pending.error
        return pending.result

    def _compute(self, key: str, pending: _Pending, name: str, path: Path, params: dict[str, Any]) -> None:
        try:
            with self._model_lock:
                pending.result = self._run(name, path, params)
        except BaseException as exc:  # noqa: BLE001 -- every failure must reach the reply
            pending.error = exc if isinstance(exc, Exception) else ToolError(str(exc) or type(exc).__name__)
        finally:
            pending.finished_at = time.time()
            pending.done.set()

    def _sweep_locked(self) -> None:
        now = time.time()
        for key in [k for k, p in self._pending.items() if p.done.is_set() and now - p.finished_at > RESULT_TTL_S]:
            del self._pending[key]

    def _run(self, name: str, path: Path, params: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        if name == "sam.box":
            box = params.get("box")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                raise ToolError("sam.box needs box=[x1, y1, x2, y2]")
            shapes = self._sam(path, params, bboxes=[[float(v) for v in box]])
        elif name == "sam.point":
            points = params.get("points")
            if not (isinstance(points, (list, tuple)) and points):
                raise ToolError("sam.point needs points=[[x, y], ...]")
            shapes = self._sam(path, params, points=[[float(x), float(y)] for x, y in points], labels=[1] * len(points))
        elif name == "detect.text":
            text = str(params.get("text") or "").strip()
            if not text:
                raise ToolError("detect.text needs text")
            shapes = self._world(path, text, float(params.get("min_confidence") or DEFAULT_MIN_CONFIDENCE))
        else:
            raise ToolError(f"no such tool: {name}", code="unknown_tool")
        return {"shapes": shapes, "ms": int((time.time() - started) * 1000)}

    # -- models ------------------------------------------------------------

    def _sam(self, path: Path, params: dict[str, Any], **prompt: Any) -> list[dict[str, Any]]:
        model = self.sam.get()
        results = model.predict(str(path), device=_device(), verbose=False, **prompt)
        width, height = int(params.get("width") or 0), int(params.get("height") or 0)
        shapes: list[dict[str, Any]] = []
        for result in results:
            if result.masks is None:
                continue
            for mask in result.masks.data:
                array = mask.detach().cpu().numpy()
                # A click on empty floor makes SAM segment the floor -- the
                # whole picture. That is never the object somebody meant, so
                # it reads as "nothing there", which the editor already says.
                if array.size and float((array > 0.5).mean() if array.dtype.kind == "f" else (array > 0).mean()) > 0.9:
                    continue
                # Ours, not the model's: the editor needs an outline with a
                # manageable number of draggable vertices, not a raster.
                shape = postprocess.mask_to_shape(array, "object", width, height)
                if shape is not None:
                    shapes.append(shape.as_dict())
        return shapes

    def _world(self, path: Path, text: str, min_confidence: float) -> list[dict[str, Any]]:
        model = self.world.get()
        # One prompt, or several separated by commas: "door, handle".
        classes = [part.strip() for part in text.split(",") if part.strip()]
        if classes != self._world_classes:
            model.set_classes(classes)
            self._world_classes = classes
        results = model.predict(
            str(path), device=_device(), conf=min_confidence, max_det=MAX_DETECTIONS, verbose=False
        )
        shapes: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                shapes.append(
                    {
                        "label": classes[int(cls)] if int(cls) < len(classes) else text,
                        "bbox": [x1, y1, x2, y2],
                        "polygon": None,
                        "confidence": round(float(conf), 4),
                    }
                )
        return shapes


def _request_key(name: str, path: Path, params: dict[str, Any]) -> str:
    """The same request, asked again, is the same work: the retry that follows
    a not_ready must find what the first call started."""
    prompt = {k: v for k, v in params.items() if k in ("box", "points", "text", "min_confidence")}
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    raw = json.dumps([name, str(path), stamp, prompt], sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _image_path(params: dict[str, Any]) -> Path:
    raw = params.get("image_path")
    if not raw:
        raise ToolError("no image_path in the request")
    path = Path(str(raw))
    if not path.is_file():
        raise ToolError(f"image not found: {path.name}")
    return path


def main() -> int:
    return serve(SidecarTool())


if __name__ == "__main__":
    raise SystemExit(main())
