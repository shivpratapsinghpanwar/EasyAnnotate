"""Export an annotation project into any of the five dataset formats.

This is the single bridge between the annotation world and the batch world.
Annotation is stateful CRUD and cannot be expressed as an operation; *export*
genuinely is folder-out, so it becomes one -- and gets the job manager, the
progress UI, the dry run and the recipe support for free.

Two rules are enforced here rather than left to the caller:

**The class order comes from the project's label table.** Not from sorting the
names that happen to appear in this export. The label table allocates an index
once and never moves it, so exporting today and exporting after adding a class
produce the same indices for the classes that existed before. Getting this
wrong is silent -- the ``.txt`` files still parse, they just mean something
else -- which is why the ordering lives in a table with a test rather than in
whichever call site remembered.

**Only reviewed images are exported by default.** Feeding a model's own
unreviewed predictions back in as ground truth is how a model poisons itself,
and it does so without a single error message. Widening the selection is
possible, explicit, and warned about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..annotations import formats
from ..annotations.model import Annotation
from ..registry import Param, operation
from ..types import Context, OpResult

#: The states this operation trusts unless told otherwise.
DEFAULT_STATES = "reviewed"

#: Anything outside this earns a warning: it is either unfinished work or an
#: explicit "I could not tell", and neither belongs in a training set by accident.
TRUSTED_STATES = frozenset({"reviewed"})

EXPORT_PARAMS = (
    Param(
        "database",
        "file",
        "The annotations.db holding the project.",
        required=True,
        must_exist=True,
    ),
    Param("project", "str", "Project id to export.", required=True),
    Param("output", "dir", "Where the exported dataset is written.", required=True),
    Param("to", "choice", "Target format.", default="yolo", choices=formats.FORMATS),
    Param(
        "images",
        "dir",
        "Folder holding the image files. Defaults to the project's own root.",
        default=None,
    ),
    Param(
        "states",
        "strs",
        "Image states to include. Default is reviewed only, on purpose.",
        default=DEFAULT_STATES,
    ),
    Param(
        "classes",
        "strs",
        "Override the project's class order. Rarely correct -- see the notes.",
        default=None,
    ),
    Param("owner", "str", "Owner the project belongs to.", default="local"),
    Param("copy_images", "bool", "Copy the image files alongside the labels.", default=True),
    Param(
        "segmentation",
        "bool",
        "Keep polygon outlines where the target format supports them.",
        default=True,
    ),
    Param(
        "include_empty",
        "bool",
        "Include images with no objects, as hard negatives.",
        default=True,
    ),
)


@operation(
    "annotate.export",
    title="Export an annotation project",
    category="export",
    summary="Turn a project's annotations into a VOC, YOLO, COCO, LabelMe or CSV dataset.",
    params=EXPORT_PARAMS,
    destructive=True,
    hidden_from_web=True,
    notes=(
        "The class order is the project's own label order, and label indices\n"
        "never move: renaming a class keeps its index, retiring one keeps its\n"
        "slot, and a new class is appended. Export twice with a class added in\n"
        "between and the original indices are identical -- which is what makes\n"
        "the .txt files you exported last month still mean what they said.\n"
        "\n"
        "'classes' overrides that order. It exists for the case where you are\n"
        "matching a class list someone else fixed, and it is otherwise a way to\n"
        "reintroduce exactly the bug the label table removes. Any class present\n"
        "but unlisted is appended rather than inserted.\n"
        "\n"
        "'states' defaults to 'reviewed' because unreviewed images may hold raw\n"
        "model predictions. Exporting those back into training teaches a model\n"
        "its own mistakes, and nothing about the output looks wrong."
    ),
)
def annotate_export(params: dict[str, Any], ctx: Context) -> OpResult:
    # Imported here, not at module scope, so `core` keeps its own import graph
    # free of the annotation package -- and so a build without it still lists
    # every other operation instead of failing to import ops entirely.
    from ...annotate import images as images_mod
    from ...annotate import labels as labels_mod
    from ...annotate import projects as projects_mod
    from ...annotate import shapes as shapes_mod
    from ...annotate.store import Store, StoreError

    result = OpResult(op="annotate.export", dry_run=ctx.dry_run)
    output: Path = params["output"]
    target: str = params["to"]

    try:
        store = Store(params["database"])
        project = projects_mod.get(store, params["project"], params["owner"])
    except StoreError as exc:
        # The commonest way to get here is running the shipped recipe without
        # setting the project, so say what to do rather than only what broke.
        result.fail(f"{exc}. {_project_hint(store, params)}")
        return result

    try:
        return _export(
            params, ctx, result, store, project, output, target,
            images_mod, labels_mod, shapes_mod,
        )
    finally:
        store.close()


def _default_image_root(database: Path, project: Any) -> Path | None:
    """Where this project's images live, when the caller did not say.

    An *attached* project knows: it is the folder the user opened. A *managed*
    project's images sit beside the database, at ``projects/<id>/`` -- derived
    from the database path rather than imported from the service layer, so
    ``core`` keeps its own import graph and the two cannot drift into
    disagreeing about the layout by accident.

    Without this, exporting a managed project quietly produced labels and no
    images, which reads as "the export is broken" rather than "you did not say
    where the pictures were".
    """
    if project.root:
        return Path(project.root)
    candidate = Path(database).parent / "projects" / project.id
    return candidate if candidate.is_dir() else None


def _project_hint(store: Any, params: dict[str, Any]) -> str:
    """Name the projects that *are* in this database, if any."""
    from ...annotate import projects as projects_mod

    try:
        found = projects_mod.listing(store, params["owner"])
    except Exception:
        return ""
    if not found:
        return (
            f"There are no projects in {params['database']} yet -- run "
            f"'datafactory annotate <folder>' to make one."
        )
    listed = ", ".join(f"{p.name} ({p.id})" for p in found[:5])
    return f"Projects here: {listed}"


def _export(
    params, ctx, result, store, project, output, target,
    images_mod, labels_mod, shapes_mod,
) -> OpResult:
    from ...annotate.store import StoreError

    wanted = [s.strip() for s in (params["states"] or []) if s.strip()]
    unknown = [s for s in wanted if s not in images_mod.STATES]
    if unknown:
        result.fail(
            f"unknown image state(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(images_mod.STATES)}"
        )
        return result
    if not wanted:
        wanted = [DEFAULT_STATES]

    risky = sorted(set(wanted) - TRUSTED_STATES)
    if risky:
        result.warnings.append(
            f"exporting unreviewed images ({', '.join(risky)}): any model "
            f"predictions among them have not been checked by a person"
        )

    # The class order. The project's own unless explicitly overridden.
    order = [c.strip() for c in (params["classes"] or []) if c.strip()]
    if order:
        result.warnings.append(
            "'classes' overrode the project's label order; indices in this "
            "export may not match previous ones"
        )
    else:
        order = labels_mod.class_list(store, project)
    if not order:
        result.fail("this project has no labels yet, so there is nothing to export")
        return result

    supplied = params["images"]
    if supplied is not None and not Path(supplied).is_dir():
        # Asked for explicitly and not there. Falling back would produce a
        # labels-only export that looks like the exporter is broken, when in
        # fact the path was wrong -- so it fails, and says which path.
        result.fail(f"images: no such folder: {supplied}")
        return result

    root = supplied or _default_image_root(params["database"], project)
    copy_images = bool(params["copy_images"]) and root is not None
    if params["copy_images"] and root is None:
        result.warnings.append(
            "no image folder was found for this project, so only labels were "
            "written. Pass 'images' to point at the pictures."
        )

    items: list[formats.Item] = []
    missing_files = 0
    skipped_empty = 0
    selected = [img for state in wanted for img in _all_in_state(store, project, images_mod, state)]
    selected.sort(key=lambda i: i.seq)

    total = len(selected)
    ctx.emit(0, total, f"{project.name} -> {target}")

    for done, image in enumerate(selected, start=1):
        ctx.check_cancelled()
        try:
            annotation: Annotation = shapes_mod.annotation_for(store, project, image)
        except StoreError as exc:
            result.warnings.append(f"{image.rel_path}: {exc}")
            continue

        if not annotation.shapes and not params["include_empty"]:
            skipped_empty += 1
            continue

        image_path: Path | None = None
        if root is not None:
            candidate = Path(root) / image.rel_path
            if candidate.is_file():
                image_path = candidate
            else:
                missing_files += 1

        items.append(formats.Item(annotation=annotation, image=image_path))
        if done % 200 == 0 or done == total:
            ctx.emit(done, total, image.name)

    if not items:
        result.fail(
            f"no images matched state(s) {', '.join(wanted)}. "
            f"Counts: {images_mod.counts(store, project)}"
        )
        return result

    dataset = formats.Dataset(items=items, source_format="annotate", classes=order)
    report = formats.write_dataset(
        dataset,
        output,
        target,
        ctx=ctx,
        copy_images=copy_images,
        segmentation=params["segmentation"],
        on_action=lambda action: result.actions.append(action),
    )

    stats = dataset.stats()
    result.counts.update(
        {
            "images": stats["images"],
            "objects": stats["objects"],
            "classes": len(order),
            "backgrounds": stats["backgrounds"],
            "labels_written": int(report["written"]),
            "images_copied": int(report["images"]),
            "skipped_empty": skipped_empty,
        }
    )
    if missing_files:
        result.warnings.append(
            f"{missing_files} image file(s) named in the project were not found "
            f"under {root}; their labels were still written"
        )
    result.warnings.extend(str(w) for w in report["warnings"])  # type: ignore[union-attr]
    result.data.update(
        {
            "project": project.name,
            "project_id": project.id,
            "target_format": target,
            "output": output,
            "classes": order,
            "states": wanted,
        }
    )
    return result


def _all_in_state(store, project, images_mod, state: str) -> list[Any]:
    """Every image in one state, paged so a 50,000-image project does not
    build one enormous result set."""
    out: list[Any] = []
    offset = 0
    page = 2000
    while True:
        batch = images_mod.listing(store, project, state=state, limit=page, offset=offset)
        out.extend(batch)
        if len(batch) < page:
            return out
        offset += page
