"""Turning a web request into operation parameters, without trusting paths.

Operations take real filesystem paths. A browser must never supply one, so this
module is the translation layer: the client picks *what* to run on by logical
name -- ``"source"``, or the name of an earlier output -- and every actual path
is derived from the dataset sandbox here.

The rule enforced below is blunt on purpose: **any path-kind parameter sent by
a client is rejected outright**, rather than sanitised. Sanitising invites an
arms race over encodings, symlinks and UNC paths; refusing does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.registry import Operation, ParamError
from .workspace import Dataset, Workspace, safe_segment

#: Parameter kinds that name a location on disk. None may come from a client.
PATH_KINDS = frozenset({"dir", "file", "path", "dirs"})

#: Parameters that name where an operation writes, rather than what it reads.
OUTPUT_NAMES = frozenset({"output", "quarantine"})


class BindingError(Exception):
    """Raised when a request cannot be turned into a safe parameter set."""


def resolve_input(dataset: Dataset, reference: str | None, workspace: Workspace) -> Path:
    """Map a logical input name onto a real directory inside the sandbox.

    ``None`` or ``"source"`` means the uploaded dataset; anything else names an
    output produced by an earlier step, which is what makes chaining possible
    without ever exposing a path.
    """
    if reference in (None, "", "source"):
        return workspace.assert_inside(dataset.source)
    candidate = dataset.output(reference)
    resolved = workspace.assert_inside(candidate)
    if not resolved.is_dir():
        raise BindingError(
            f"no step has produced {reference!r} yet. Available: "
            f"{', '.join(available_inputs(dataset)) or 'source'}"
        )
    return resolved


def available_inputs(dataset: Dataset) -> list[str]:
    """Everything that can currently be used as an input, for the UI to offer."""
    names = ["source"] if any(dataset.source.iterdir()) else []
    if dataset.outputs.is_dir():
        names.extend(sorted(p.name for p in dataset.outputs.iterdir() if p.is_dir()))
    return names


def bind(
    operation: Operation,
    supplied: dict[str, Any],
    *,
    dataset: Dataset,
    workspace: Workspace,
    input_ref: str | None = None,
    output_name: str = "",
) -> tuple[dict[str, Any], Path | None]:
    """Build a safe, fully-populated parameter set for one run.

    Returns the parameters and the output directory, if the operation writes
    one. Raises :class:`BindingError` for anything a client should not be able
    to ask for, and lets :class:`ParamError` through for ordinary validation --
    so the web API reports bad values exactly as the CLI does.
    """
    offered = dict(supplied or {})

    # Refuse rather than sanitise. A rejected request is a bug report; a
    # silently rewritten one is a vulnerability waiting for an encoding trick.
    for spec in operation.params:
        if spec.kind in PATH_KINDS and spec.name in offered:
            raise BindingError(
                f"{spec.name!r} is a path and cannot be set directly. Choose an "
                f"input with 'input', and name the result with 'output_name'."
            )

    output_dir: Path | None = None
    for spec in operation.params:
        if spec.kind not in PATH_KINDS:
            continue

        if spec.name in OUTPUT_NAMES:
            target = dataset.output(output_name or f"{operation.name.replace('.', '_')}")
            output_dir = workspace.assert_inside(target)
            if spec.kind == "file":
                # report.html writes a file, not a directory.
                output_dir.mkdir(parents=True, exist_ok=True)
                offered[spec.name] = str(output_dir / _filename_for(operation))
            else:
                offered[spec.name] = str(output_dir)
            continue

        source = resolve_input(dataset, input_ref, workspace)
        if spec.kind == "dirs":
            offered[spec.name] = [str(source)]
        elif spec.kind == "file":
            offered[spec.name] = str(_find_file(source, operation))
        else:
            offered[spec.name] = str(source)

    bound = operation.bind(offered)
    return bound, output_dir


def _filename_for(operation: Operation) -> str:
    return "report.html" if operation.name == "report.html" else "result.json"


def _find_file(folder: Path, operation: Operation) -> Path:
    """Locate the single file a file-kind input wants.

    Only ``inspect.coco`` needs this, and it wants an instances.json; guessing
    is acceptable because the alternative is asking a browser for a path.
    """
    for pattern in ("**/instances*.json", "**/_annotations.coco.json", "**/*.json"):
        for found in sorted(folder.glob(pattern)):
            if found.is_file():
                return found
    raise BindingError(f"{operation.name} needs a json file and none was found in the dataset")


def describe_operation(operation: Operation) -> dict[str, Any]:
    """The operation as a web form should render it.

    Path parameters are stripped: the client cannot set them, so showing them
    would only invite a request the service will refuse.
    """
    payload = operation.as_dict()
    payload["params"] = [p for p in payload["params"] if p["kind"] not in PATH_KINDS]
    payload["writes_output"] = any(
        p.name in OUTPUT_NAMES for p in operation.params
    )
    return payload


def public_schema() -> dict[str, Any]:
    """Every operation a browser can usefully render a form for.

    The ``hidden_from_web`` ones are filtered here and only here -- they stay
    in ``datafactory list``, in ``schema``, and in recipes. An operation whose
    real input is an annotation project rather than a folder has nothing the
    batch panel could ask for, so offering it would only produce a refusal.
    """
    from ..core import registry

    return {
        "version": 1,
        "operations": [
            describe_operation(op)
            for op in registry.all_operations()
            if not op.hidden_from_web
        ],
    }


__all__ = [
    "BindingError",
    "ParamError",
    "available_inputs",
    "bind",
    "describe_operation",
    "public_schema",
    "resolve_input",
    "safe_segment",
]
