"""A project, as a script sees it."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from data_factory.annotate import attach, images, labels, loop, projects, review, shapes
from data_factory.annotate.store import LOCAL_OWNER, Store
from data_factory.core import registry
from data_factory.core.types import Context, OpResult
from data_factory.service.workspace import resolve_workspace_root

_stores: dict[Path, Store] = {}
# Opening a store migrates it; two threads doing that at once raced to
# "database is locked". One lock, as the server's cache has.
_stores_lock = threading.Lock()


def database_path() -> Path:
    """The annotation database the editor, CLI and library share.

    Beside the workspace -- ``DATAFACTORY_WORKSPACE``'s parent, by default
    ``~/.datafactory`` -- exactly where ``easyannotate serve`` keeps it.
    """
    return resolve_workspace_root().parent / "annotations.db"


def _store() -> Store:
    path = database_path().resolve()
    with _stores_lock:
        if path not in _stores:
            _stores[path] = Store(path)
        return _stores[path]


def close() -> None:
    """Release the database files. Needed before deleting the folder they are
    in -- Windows will not remove an open file."""
    with _stores_lock:
        for store in _stores.values():
            store.close()
        _stores.clear()


def open_folder(folder: str | Path, *, name: str = "", import_annotations: bool = True) -> Project:
    """Open a folder of images, editing its label files in place.

    Reopening the same folder returns the same project, with its review state.
    Label files already there (VOC, YOLO, COCO, LabelMe, DOTA) are loaded the
    first time.
    """
    store = _store()
    summary = attach.open_folder(
        store, Path(folder).expanduser(), name=name, import_annotations=import_annotations
    )
    return Project(store, summary["project"], notes=summary.get("warnings") or [])


def open_project(project_id: str) -> Project:
    """A project by id -- one made by uploading in the web interface, say."""
    store = _store()
    return Project(store, projects.get(store, project_id, LOCAL_OWNER))


def list_projects() -> list[Project]:
    store = _store()
    return [Project(store, p) for p in projects.listing(store, LOCAL_OWNER)]


class Project:
    """One annotation project. Cheap to hold; every call reads the database."""

    def __init__(self, store: Store, record: projects.Project, *, notes: list[str] | None = None) -> None:
        self._store = store
        self._record = record
        self.notes = list(notes or [])

    # -- identity ----------------------------------------------------------

    @property
    def id(self) -> str:
        return self._record.id

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def root(self) -> Path:
        """Where the images are: the folder you opened, or the managed copy."""
        if self._record.is_attached and self._record.root:
            return Path(self._record.root)
        # Beside *this project's* database -- not wherever the environment
        # points now, which may be another workspace by the time this is read.
        return self._store.path.parent / "projects" / self._record.id

    def __repr__(self) -> str:
        return f"<Project {self.name!r} {self.id[:8]} {self.counts()}>"

    # -- reading -----------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Images per state: todo, predicted, in_progress, reviewed, skipped, total."""
        return images.counts(self._store, self._record)

    def classes(self) -> list[str]:
        """The class list in index order -- what YOLO index *n* means."""
        return labels.class_list(self._store, self._record)

    def images(self, *, state: str = "", order: str = "", limit: int | None = None,
               offset: int = 0) -> list[dict[str, Any]]:
        """Images as dicts -- all of them unless ``limit`` says otherwise.
        ``order="uncertain"`` puts likely mistakes first."""
        found: list[dict[str, Any]] = []
        page = 5000  # the store's largest page
        while limit is None or len(found) < limit:
            want = page if limit is None else min(page, limit - len(found))
            batch = images.listing(self._store, self._record, state=state, order=order,
                                   limit=want, offset=offset + len(found))
            found.extend(image.as_dict() for image in batch)
            if len(batch) < want:
                break
        return found

    def shapes(self, image_id: str) -> list[dict[str, Any]]:
        image = images.get(self._store, self._record, image_id)
        return [s.as_dict() for s in shapes.listing(self._store, self._record, image)]

    def agreement(self) -> dict[str, Any]:
        """How often the model was right on reviewed images."""
        return shapes.agreement(self._store, self._record)

    # -- the loop ----------------------------------------------------------

    def import_predictions(self, source: str | Path, *, min_confidence: float = 0.0,
                           overwrite_reviewed: bool = False) -> dict[str, Any]:
        """Load a model run (YOLO .txt folder, COCO results .json, VOC, LabelMe)
        as predictions to review. Returns the matching report -- read it."""
        return review.import_predictions(
            self._store, self._record, Path(source).expanduser(),
            min_confidence=min_confidence, overwrite_reviewed=overwrite_reviewed,
        )

    def predict(self, model: str | Path, *, conf: float = 0.25,
                on_progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
        """Run an Ultralytics model over the unreviewed images. Needs the tools."""
        return loop.predict(self._store, self._record, self.root, Path(model).expanduser(), conf=conf,
                            on_progress=on_progress)

    def train(self, *, epochs: int = 30, imgsz: int = 640, base: str = "yolov8n.pt",
              on_line: Callable[[str], None] = print) -> dict[str, Any]:
        """Train on the reviewed images. Returns ``{"model": Path, "metrics": ...}``."""
        return loop.train(self._store, self._record, epochs=epochs, imgsz=imgsz, base=base,
                          on_line=on_line)

    def export(self, output: str | Path, *, to: str = "yolo",
               states: Iterable[str] = ("reviewed",), copy_images: bool = True,
               segmentation: bool = True, dry_run: bool = False) -> OpResult:
        """Write the project as a dataset. Reviewed images only, by default --
        exporting unreviewed model output into training is how a model poisons
        itself. Class indices are the project's own and never move."""
        if isinstance(states, str):
            states = [states]  # "reviewed", not r, e, v, i, e, w, e, d
        # Through Operation.run, like every other caller: the same validation
        # (a bad format is a ParamError), `~` expanded, and provenance written.
        return registry.get("annotate.export").run(
            {
                "database": str(self._store.path),
                "project": self._record.id,
                "output": str(output),
                "to": to,
                "states": list(states),
                "owner": self._record.owner_id,
                "copy_images": copy_images,
                "segmentation": segmentation,
            },
            Context(dry_run=dry_run),
        )
