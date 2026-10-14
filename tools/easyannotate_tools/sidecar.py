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
import os
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
INLINE_WAIT_S = 2.5
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
        requires=("prepare",),
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
        requires=("prepare",),
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
    # Command-line only: the editor never shows it and the HTTP routes refuse
    # it, because loading a model file runs it. `easyannotate predict`.
    ToolSpec(
        name="detect.model",
        title="Predict with a trained model",
        summary="Runs an Ultralytics model (detect, segment or OBB) over an image.",
        kind="model",
        inputs={
            "type": "object",
            "properties": {"model": {"type": "string"}, "conf": {"type": "number"}},
            "required": ["model"],
        },
        version="ultralytics",
        requires=(),
    ),
]
#: Custom models kept loaded, most recent last. Two: the one in use, and the
#: one it is being compared against.
MAX_CUSTOM_MODELS = 2


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


def _profile() -> dict[str, Any]:
    """What `easyannotate tools install` chose for this machine: device and
    model sizes. Absent, the defaults are the small CPU-friendly models."""
    home = Path(os.environ.get("DATAFACTORY_TOOLS_HOME") or Path.home() / ".datafactory" / "tools")
    try:
        return json.loads((home / "profile.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _device() -> str:
    try:
        import torch

        wanted = _profile().get("device", "")
        if wanted == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() and wanted != "cpu" else "cpu"
    except Exception:
        return "cpu"


def _load_sam() -> Any:
    import numpy as np
    from ultralytics import SAM

    model = SAM(_profile().get("sam") or SAM_WEIGHTS)
    # Build the predictor once, on a tiny dummy image. After this the sidecar
    # only ever uses predictor.set_image + prompt calls: a full model.predict
    # on anything replaces the cached image features (measured: the next
    # prompt then returns an empty mask), so it must never run again.
    model.predict(np.zeros((64, 64, 3), np.uint8), bboxes=[[2, 2, 30, 30]], verbose=False)
    return model


def _load_world() -> Any:
    from ultralytics import YOLOWorld

    model = YOLOWorld(_profile().get("world") or WORLD_WEIGHTS)
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
        # Encode-ahead. SAM spends ~2.5s on CPU encoding an image and ~0.15s
        # answering each prompt against it, so the image the person has open
        # is encoded once, ahead of the first click. One image only: the
        # features live inside Ultralytics' predictor, and snapshotting them
        # across versions would be brittle.
        self._sam_key: str | None = None
        self._want_key: str | None = None
        self._custom_models: dict[str, Any] = {}

    # -- protocol ----------------------------------------------------------

    def describe(self) -> list[dict[str, Any]]:
        # Report the model sizes actually in use, from the install profile.
        profile = _profile()
        specs = []
        for spec in SPECS:
            entry = spec.as_dict()
            if spec.name.startswith("sam."):
                entry["version"] = Path(profile.get("sam") or SAM_WEIGHTS).stem
            elif spec.name == "detect.text":
                entry["version"] = Path(profile.get("world") or WORLD_WEIGHTS).stem
            specs.append(entry)
        return specs

    def health(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("tool") or "")
        lazy = self.world if name.startswith("detect") else self.sam
        return {
            "state": "ready" if lazy.state in ("ready", "cold", "loading") else "unavailable",
            "detail": f"{lazy.name}: {lazy.state}, device {_device()}" + (f" ({lazy.error})" if lazy.error else ""),
        }

    def prepare(self, params: dict[str, Any]) -> None:
        """Start loading the model and, for SAM, encoding this image. Returns
        at once: the work happens on a thread, and only the newest request
        is worth finishing -- holding N must not queue up a backlog of
        encodes for images already left behind."""
        name = str(params.get("tool") or "")
        lazy = self.world if name.startswith("detect") else self.sam
        try:
            lazy.get()
        except ToolNotReady:
            pass
        except ToolError:
            return
        if not name.startswith("sam"):
            return
        source = params.get("source") or params.get("image_path")
        if not source or not Path(str(source)).is_file():
            return
        path = Path(str(source))
        key = _image_key(path)
        self._want_key = key
        threading.Thread(target=self._encode_when_ready, args=(path, key), daemon=True,
                         name="sam-encode").start()

    def _encode_when_ready(self, path: Path, key: str) -> None:
        deadline = time.time() + 600  # first use downloads the weights
        while time.time() < deadline:
            if self._want_key != key:
                return  # the person moved on; a newer prepare owns the work
            try:
                self.sam.get()
                break
            except ToolNotReady:
                time.sleep(0.5)
            except ToolError:
                return
        with self._model_lock:
            if self._want_key != key or self._sam_key == key:
                return
            try:
                self.sam.model.predictor.set_image(str(path))
                self._sam_key = key
            except Exception:
                self._sam_key = None

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
        elif name == "detect.model":
            shapes = self._custom(path, params)
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
        predictor = model.predictor
        key = _image_key(path)
        if self._sam_key != key:
            # Not encoded ahead (no prepare, or a different image): encode now.
            self._sam_key = None
            predictor.set_image(str(path))
            self._sam_key = key
        results = predictor(**prompt)
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

    def _custom(self, path: Path, params: dict[str, Any]) -> list[dict[str, Any]]:
        weights = Path(str(params.get("model") or ""))
        if not weights.is_file():
            raise ToolError(f"no model file at {weights}")
        key = f"{weights}|{weights.stat().st_mtime_ns}"
        model = self._custom_models.pop(key, None)
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(weights))
        self._custom_models[key] = model
        while len(self._custom_models) > MAX_CUSTOM_MODELS:
            self._custom_models.pop(next(iter(self._custom_models)))

        conf = float(params.get("conf") or 0.25)
        results = model.predict(str(path), device=_device(), conf=conf, max_det=MAX_DETECTIONS, verbose=False)
        names = getattr(model, "names", {}) or {}
        shapes: list[dict[str, Any]] = []
        for result in results:
            obb = getattr(result, "obb", None)
            if obb is not None and len(obb):
                for corners, score, cls in zip(obb.xyxyxyxy.tolist(), obb.conf.tolist(), obb.cls.tolist()):
                    points = [[int(round(x)), int(round(y))] for x, y in corners]
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    shapes.append({
                        "label": str(names.get(int(cls), int(cls))),
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "polygon": points,
                        "kind": "obb",
                        "confidence": round(float(score), 4),
                    })
                continue
            boxes = result.boxes
            if boxes is None:
                continue
            outlines = result.masks.xy if getattr(result, "masks", None) is not None else None
            for index, (xyxy, score, cls) in enumerate(zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())):
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                polygon = None
                if outlines is not None and index < len(outlines) and len(outlines[index]) >= 3:
                    # Already in image coordinates -- unlike masks.data, which
                    # is at the model's input size.
                    polygon = [[int(round(x)), int(round(y))] for x, y in outlines[index].tolist()]
                shapes.append({
                    "label": str(names.get(int(cls), int(cls))),
                    "bbox": [x1, y1, x2, y2],
                    "polygon": polygon,
                    "confidence": round(float(score), 4),
                })
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


def _image_key(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return f"{path}|{stamp}"


def models_dir() -> Path:
    """Where weights live. Ultralytics downloads into the working directory,
    so the sidecar runs from here -- otherwise every place it happened to be
    started from collected another 100 MB of .pt files."""
    path = Path(os.environ.get("EASYANNOTATE_MODELS") or Path.home() / ".datafactory" / "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _request_key(name: str, path: Path, params: dict[str, Any]) -> str:
    """The same request, asked again, is the same work: the retry that follows
    a not_ready must find what the first call started."""
    prompt = {k: v for k, v in params.items()
              if k in ("box", "points", "text", "min_confidence", "model", "conf")}
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
    os.chdir(models_dir())
    return serve(SidecarTool())


if __name__ == "__main__":
    raise SystemExit(main())
