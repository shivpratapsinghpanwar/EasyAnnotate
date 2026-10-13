"""The loop: reviewed images make a model, the model predicts the rest, and a
person reviews its mistakes -- hardest first.

    easyannotate train ./doors          # reviewed -> model -> predictions
    easyannotate predict ./doors --model best.pt

**Command-line only, on purpose.** A ``.pt`` file is a pickle, and loading a
model runs it. Over HTTP that would let a client name a file for the server
to execute; here the path comes from a person typing it into their own
terminal -- the same rule that keeps attached folders off the web.

Training and inference run in the tools environment (``easyannotate tools
install``), never in this one: inference through the sidecar protocol, and
training as a subprocess of the tools interpreter, so torch is never imported
here and a crashed training run takes nothing else down with it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from . import attach as attach_mod
from . import images as images_mod
from . import labels as labels_mod
from . import review
from . import shapes as shapes_mod
from .projects import Project
from .store import Store, StoreError
from .tools.local import python_for_tools
from .tools.registry import REGISTRY
from .tools.results import shapes_from_result

MODEL_TOOL = "detect.model"
#: The module the tools interpreter runs to train. Overridable so the test
#: suite can stand in a trainer that needs no torch.
TRAIN_MODULE_ENV = "EASYANNOTATE_TRAIN_MODULE"
DEFAULT_TRAIN_MODULE = "easyannotate_tools.train"
#: Lines the trainer prints for us, among Ultralytics' own.
EPOCH_PREFIX = "EASYANNOTATE_EPOCH "
RESULT_PREFIX = "EASYANNOTATE_RESULT "
#: A newer model replaces an older one's untouched guesses on "predicted"
#: images too -- that is what each turn of the loop is for.
UNREVIEWED = ("todo", "predicted", "in_progress")


class LoopError(StoreError):
    """Something a person can act on: say what, and what to do."""


def models_home() -> Path:
    """Where trained models and training runs are kept. Shared with the
    sidecar, which downloads base weights here too."""
    path = Path(os.environ.get("EASYANNOTATE_MODELS") or Path.home() / ".datafactory" / "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- predict


def predict(
    store: Store,
    project: Project,
    image_root: Path,
    model: Path,
    *,
    conf: float = 0.25,
    states: Iterable[str] = UNREVIEWED,
    tool: str = MODEL_TOOL,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run a trained model over the project's unreviewed images.

    Its answers go in through ``set_predictions`` -- the same path as an
    imported run -- so they are dashed, scored, ordered by uncertainty, and
    never exported until somebody has looked at them. Reviewed images are
    never touched: finished work is not something a model gets to redo.
    """
    model = Path(model).expanduser().resolve()
    if not model.is_file():
        raise LoopError(f"no model file at {model}")
    if REGISTRY.spec(tool) is None:
        raise LoopError(
            "no inference tool is installed -- run: easyannotate tools install"
        )
    wanted = [s for s in states if s in images_mod.STATES and s != "reviewed"]
    if not wanted:
        raise LoopError("predict runs on unreviewed images only")

    targets: list[images_mod.Image] = []
    for state in wanted:
        offset = 0
        while True:
            page = images_mod.listing(store, project, state=state, limit=5000, offset=offset)
            targets.extend(page)
            if len(page) < 5000:
                break
            offset += 5000

    predicted = shapes_total = failed = 0
    touched: list[str] = []
    started = time.time()
    for index, image in enumerate(targets):
        if on_progress:
            on_progress(index, len(targets), image.name)
        path = (Path(image_root) / image.rel_path)
        if not path.is_file():
            failed += 1
            continue
        try:
            raw = REGISTRY.invoke_waiting(
                tool,
                {
                    "image_id": image.id,
                    "image_path": str(path),
                    "width": image.width,
                    "height": image.height,
                    "model": str(model),
                    "conf": conf,
                },
                budget_s=600.0,
            )
        except Exception:
            failed += 1
            continue
        made = shapes_from_result(raw, "", image.width, image.height)
        review.set_predictions(store, project, image, made)
        touched.append(image.id)
        predicted += 1
        shapes_total += len(made)
    if on_progress:
        on_progress(len(targets), len(targets), "done")
    attach_mod.sync(store, project, touched)
    return {
        "images": len(targets),
        "predicted": predicted,
        "shapes": shapes_total,
        "failed": failed,
        "seconds": round(time.time() - started, 1),
        "agreement": shapes_mod.agreement(store, project),
    }


# ------------------------------------------------------------------ train


def train(
    store: Store,
    project: Project,
    *,
    epochs: int = 30,
    imgsz: int = 640,
    base: str = "yolov8n.pt",
    on_line: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train a detector on the reviewed images, in the tools environment.

    Exports exactly what export would -- reviewed images, the project's own
    class order, so index 3 means in the model what it means in the files --
    and hands the folder to Ultralytics. Returns the saved model's path and
    its final metrics.
    """
    reviewed = images_mod.counts(store, project).get("reviewed", 0)
    if reviewed == 0:
        raise LoopError("nothing to learn from yet -- review some images first")
    classes = labels_mod.class_list(store, project)
    if not classes:
        raise LoopError("the project has no classes yet")

    python = python_for_tools()
    module = os.environ.get(TRAIN_MODULE_ENV) or DEFAULT_TRAIN_MODULE
    if module == DEFAULT_TRAIN_MODULE and not _can_import(python, "ultralytics"):
        raise LoopError("training needs the tools -- run: easyannotate tools install")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = models_home() / "runs" / project.id
    runs.mkdir(parents=True, exist_ok=True)
    log_path = runs / f"{stamp}.log"

    with tempfile.TemporaryDirectory(prefix="easyannotate-train-") as holding:
        data = Path(holding) / "data"
        _export_reviewed(store, project, data)
        yaml_path = _write_training_yaml(data, classes)
        on_line(f"training on {reviewed} reviewed image(s), {len(classes)} class(es), {epochs} epoch(s)")

        command = [
            python, "-m", module,
            "--data", str(yaml_path),
            "--model", base,
            "--epochs", str(int(epochs)),
            "--imgsz", str(int(imgsz)),
            "--out", str(runs),
            "--name", stamp,
        ]
        result: dict[str, Any] | None = None
        tail: list[str] = []
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=models_home(),
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                line = line.rstrip()
                tail = (tail + [line])[-15:]
                if line.startswith(EPOCH_PREFIX):
                    on_line(line[len(EPOCH_PREFIX):])
                elif line.startswith(RESULT_PREFIX):
                    result = json.loads(line[len(RESULT_PREFIX):])
            code = process.wait()
        if code != 0 or result is None:
            raise LoopError(
                f"training failed (exit {code}); full log: {log_path}\n" + "\n".join(tail[-8:])
            )

    best = Path(result["best"])
    if not best.is_file():
        raise LoopError(f"training reported {best}, which does not exist; log: {log_path}")
    saved = models_home() / "projects" / project.id / f"{stamp}.pt"
    saved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, saved)
    return {"model": saved, "metrics": result.get("metrics") or {}, "log": log_path}


def latest_model(project: Project) -> Path | None:
    folder = models_home() / "projects" / project.id
    found = sorted(folder.glob("*.pt")) if folder.is_dir() else []
    return found[-1] if found else None


def _export_reviewed(store: Store, project: Project, output: Path) -> None:
    from ..core import registry
    from ..core.types import Context

    operation = registry.get("annotate.export")
    params = {param.name: param.default for param in operation.params}
    params.update(
        {
            "database": store.path,
            "project": project.id,
            "output": output,
            "to": "yolo",
            "states": ["reviewed"],
            "owner": project.owner_id,
            "copy_images": True,
            # Boxes: a detector learns from boxes, and a polygon line in a
            # detection label file is a guess about what the trainer does.
            "segmentation": False,
            "include_empty": True,
        }
    )
    result = operation.func(params, Context())
    if not result.ok:
        raise LoopError("could not export the reviewed images: " + "; ".join(result.errors))


def _write_training_yaml(data: Path, classes: list[str]) -> Path:
    import yaml

    images = data / "images"
    splits = sorted(p.name for p in images.iterdir() if p.is_dir()) if images.is_dir() else []
    train_split = "train" if "train" in splits else (splits[0] if splits else "train")
    # A small project has no validation split. Validating on the training
    # images says nothing about generalisation, but it lets a first model
    # exist -- and the review queue, not mAP, is what judges it here.
    val_split = "val" if "val" in splits else train_split
    path = data / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(data),
                "train": f"images/{train_split}",
                "val": f"images/{val_split}",
                "names": dict(enumerate(classes)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _can_import(python: str, module: str) -> bool:
    if python == sys.executable:
        try:
            __import__(module)
            return True
        except ImportError:
            return False
    try:
        return subprocess.run(
            [python, "-c", f"import {module}"], capture_output=True, timeout=120
        ).returncode == 0
    except Exception:
        return False
