"""Attached projects: annotate a folder where it lies.

The managed path copies a dataset into the workspace and treats the database as
the truth. That is right for an upload from a browser, and wrong for the case
this module exists for: a folder already on your disk, with sidecar files you
want *updated in place*. Copying a 2.5 GB dataset to annotate it, and then
having to export to get your edits back, is not what anyone means by "open this
folder".

So attached projects are:

* **Read where they are.** No copy. The image bytes are served from the folder.
* **Written through.** Every save writes the image's sidecar file as well as
  the database row, so the ``.xml`` on disk is current the moment you stop
  moving the mouse.
* **Local only.** A project pointing at a server-side path would be exactly the
  arbitrary-path primitive the service layer exists to refuse, so there is no
  HTTP route that creates one -- the CLI does, from a path the user typed
  themselves. Hosted deployments set ``DATAFACTORY_ALLOW_ATTACHED=0`` as well.

**Write-through only works for per-image formats.** VOC, LabelMe and YOLO keep
one file per image, so a save touches one small file. COCO and CSV keep one
file for the whole dataset; rewriting a 200 MB ``instances.json`` on every
autosave would be absurd, so those folders open read-through with write-through
off, and ``annotate.export`` produces the output instead. That is recorded on
the project as ``sidecar_format`` and reported, never silently assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core import fs, imaging
from ..core.annotations import formats, labelme, voc, yolo
from ..core.annotations.model import Annotation, AnnotationError
from ..core.types import Context
from . import images as images_mod
from . import labels as labels_mod
from . import projects as projects_mod
from . import shapes as shapes_mod
from .store import LOCAL_OWNER, Store, StoreError

#: Formats that keep one label file per image, and can therefore be written
#: through on every save.
SIDECAR_FORMATS = ("voc", "labelme", "yolo")

#: What a folder of bare images becomes. VOC because it is the format the
#: existing browser annotator exports, so the two interoperate on day one.
DEFAULT_SIDECAR = "voc"


class AttachError(StoreError):
    """Raised when a folder cannot be opened for annotation."""


def detect_sidecar_format(folder: Path) -> tuple[str, str]:
    """Work out what to write back, and why. Returns ``(format, note)``."""
    try:
        detected = formats.detect(folder)
    except AnnotationError:
        return DEFAULT_SIDECAR, (
            f"no labels found, so new annotations will be written as "
            f"{DEFAULT_SIDECAR.upper()} sidecar files"
        )
    if detected in SIDECAR_FORMATS:
        return detected, f"detected {detected}; edits are written back to those files"
    return "", (
        f"detected {detected}, which keeps one file for the whole dataset. "
        f"Edits are saved in the project but NOT written back to disk -- use "
        f"'datafactory annotate.export' to produce an updated dataset."
    )


def sidecar_path(project: projects_mod.Project, image: images_mod.Image) -> Path | None:
    """Where this image's label file lives, or None if we do not write one."""
    fmt = project.sidecar_format
    if not fmt or not project.root:
        return None
    root = Path(project.root)
    image_path = root / image.rel_path
    if fmt == "voc":
        return image_path.with_suffix(".xml")
    if fmt == "labelme":
        return image_path.with_suffix(".json")
    if fmt == "yolo":
        # Honours the images/ <-> labels/ convention when the folder has one,
        # and falls back to a sidecar beside the image when it does not.
        return yolo.label_path_for(image_path, root)
    return None


def write_through(
    store: Store, project: projects_mod.Project, image: images_mod.Image
) -> Path | None:
    """Write one image's annotations back to its sidecar file.

    Returns the path written, or None when this project does not write through.
    Every writer underneath is atomic, so a save landing while something else
    reads the folder cannot produce a truncated label file.
    """
    target = sidecar_path(project, image)
    if target is None:
        return None

    annotation = shapes_mod.annotation_for(store, project, image)
    # The folder is the user's, not ours: an unwritable file is a normal thing
    # to meet, and it must not lose the edit -- the database already has it.
    try:
        if project.sidecar_format == "voc":
            voc.write(annotation, target)
        elif project.sidecar_format == "labelme":
            labelme.write(annotation, target)
        elif project.sidecar_format == "yolo":
            classes = yolo.ClassList(labels_mod.class_list(store, project))
            yolo.write(annotation, target, classes)
            _refresh_data_yaml(store, project, classes)
        else:
            return None
    except OSError as exc:
        raise AttachError(f"could not write {target.name}: {exc}") from exc
    return target


def _refresh_data_yaml(store: Store, project: projects_mod.Project, classes) -> None:
    """Keep data.yaml in step with the label table.

    A YOLO folder whose .txt files use index 3 while its data.yaml lists three
    classes is a dataset that trains on nonsense, so the class list is rewritten
    alongside. Cheap: it is a handful of lines, written atomically.
    """
    root = Path(project.root)
    existing = yolo.find_data_yaml(root)
    target = existing or (root / yolo.DATA_YAML)
    try:
        yolo.write_data_yaml(target, classes)
    except OSError:
        # A read-only folder should not fail the save; the database is still
        # authoritative and export will produce a correct data.yaml.
        pass


def open_folder(
    store: Store,
    folder: Path | str,
    *,
    owner_id: str = LOCAL_OWNER,
    name: str = "",
    import_annotations: bool = True,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Open a folder for annotation, reusing the project if it is already open.

    Returns a summary dict rather than just the project, because the things
    worth telling the user -- how many images matched a label file, what will
    be written back, what could not be read -- are exactly the things a silent
    return value would drop.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise AttachError(f"{root} is not a folder")
    if not projects_mod.attached_allowed():
        raise AttachError("attached projects are disabled on this deployment")

    existing = _find_open(store, root, owner_id)
    reused = existing is not None

    warnings: list[str] = []
    sidecar_format = existing.sidecar_format if existing else ""
    if not reused:
        sidecar_format, note = detect_sidecar_format(root)
        warnings.append(note)

    project = existing or projects_mod.create(
        store,
        name or root.name,
        owner_id=owner_id,
        storage="attached",
        root=str(root),
        sidecar_format=sidecar_format,
    )

    files = sorted(fs.iter_images(root, recursive=True))
    if not files:
        raise AttachError(f"{root} has no images in it")

    entries, unreadable = _scan_sizes(root, files, ctx)
    if unreadable:
        warnings.append(
            f"{unreadable} image(s) could not be measured and were registered "
            f"with an unknown size; they still open in the editor"
        )

    registered = images_mod.add_many(store, project, entries)
    added = sum(1 for image in registered if image.version == 0 and image.state == "todo")

    imported = 0
    if import_annotations and not reused:
        imported, notes = _import_existing(store, project, root, registered)
        warnings.extend(notes)

    # Case-variant classes are the defect the real door set actually has:
    # Close/close/Open/open, from two people annotating in two tools. They are
    # imported faithfully -- folding them on the way in would silently change
    # the data -- but training a four-class model by accident is worse than
    # being told, so say it.
    duplicated = labels_mod.variants(store, project)
    for name, found in duplicated.items():
        warnings.append(
            f"{len(found)} classes differ only by case or spacing "
            f"({', '.join(repr(label.name) for label in found)}). "
            f"Merge them in the editor, or run labels.rename --fold-case."
        )

    return {
        "project": project,
        "reused": reused,
        "images": len(registered),
        "new_images": added,
        "shapes_imported": imported,
        "sidecar_format": project.sidecar_format,
        "writes_back": bool(project.sidecar_format),
        "warnings": warnings,
    }


def _scan_sizes(
    root: Path, files: list[Path], ctx: Context | None = None
) -> tuple[list[tuple[str, int, int]], int]:
    """Measure every image, in parallel.

    Serially this dominated opening a folder: 15ms per image on the real door
    set, which is 56 seconds for 3,686 images and about four minutes for the
    full 15,514 -- all of it spent waiting on the filesystem rather than the
    CPU, which is exactly the shape thread pools are for.

    ``image_size`` reads the header only and honours EXIF, so no image is
    decoded. An unreadable file is registered at size 0 rather than dropped:
    a corrupt image is a normal thing to find in a dataset, and it should be
    visible in the editor rather than quietly missing from it.
    """
    from ..core import parallel

    entries: list[tuple[str, int, int]] = []
    unreadable = 0

    def measure(path: Path) -> tuple[int, int]:
        try:
            return imaging.image_size(path)
        except Exception:
            return (0, 0)

    for path, size in parallel.map_ordered(
        files, measure, ctx=ctx or Context(), workers=0
    ):
        width, height = size or (0, 0)
        if not width:
            unreadable += 1
        entries.append((path.relative_to(root).as_posix(), width, height))
    return entries, unreadable


def _find_open(store: Store, root: Path, owner_id: str) -> projects_mod.Project | None:
    """An attached project already pointing at this folder, if there is one.

    Reopening the same folder must land you back in the same project, or a
    second pass would start from zero and none of your review state would be
    there. Compared resolved, so ``.\\data`` and the absolute path match.
    """
    for project in projects_mod.listing(store, owner_id):
        if not project.is_attached or not project.root:
            continue
        try:
            if Path(project.root).resolve() == root:
                return project
        except OSError:
            continue
    return None


def _import_existing(
    store: Store,
    project: projects_mod.Project,
    root: Path,
    registered: list[images_mod.Image],
) -> tuple[int, list[str]]:
    """Read whatever labels are already in the folder into the project."""
    notes: list[str] = []
    try:
        dataset = formats.read_dataset(root, "auto", recursive=True)
    except AnnotationError:
        return 0, ["no existing annotations were read"]

    notes.extend(dataset.warnings[:10])

    # An existing class order is already baked into files on disk. Inherit it
    # rather than inventing one, or every .txt in the folder changes meaning.
    if dataset.classes:
        try:
            labels_mod.adopt(store, project, list(dataset.classes))
        except StoreError as exc:
            notes.append(f"could not adopt the existing class order: {exc}")

    by_stem: dict[str, Annotation] = {}
    for item in dataset.items:
        by_stem.setdefault(item.stem, item.annotation)

    imported = 0
    matched = 0
    for image in registered:
        annotation = by_stem.get(Path(image.rel_path).stem)
        if annotation is None or not annotation.shapes:
            continue
        from_model = any(s.confidence is not None for s in annotation.shapes)
        try:
            shapes_mod.replace_all(
                store, project, image, list(annotation.shapes),
                origin="model" if from_model else "human",
            )
        except StoreError as exc:
            notes.append(f"{image.rel_path}: {exc}")
            continue
        images_mod.set_state(store, project, image, "predicted" if from_model else "reviewed")
        imported += len(annotation.shapes)
        matched += 1

    if dataset.items and matched < len(dataset.items):
        # Never silent: names not lining up is the usual reason an import
        # appears to have done nothing.
        notes.append(
            f"{matched} image(s) matched a label file; "
            f"{len(dataset.items) - matched} label file(s) matched no image"
        )
    return imported, notes
