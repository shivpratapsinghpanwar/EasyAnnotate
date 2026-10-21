"""Run a recipe: an ordered list of operations described in YAML or JSON.

A dataset build is rarely one command. Writing the sequence down makes it
reproducible and reviewable, and means the same build can be re-run later, by
someone else, or from a web backend, without anyone remembering the order.

    name: door-v4
    vars:
      root: ./datasets/doors/working
    steps:
      - op: clean.orphans
        with: {folder: "${root}", scope: labels, action: delete}
      - op: labels.rename
        with: {folder: "${root}", fold_case: true}
      - op: dataset.split
        with: {folder: "${root}", output: "${root}/../splits"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import registry
from .core.types import Context, OpResult


class RecipeError(ValueError):
    """Raised when a recipe file is malformed or names an unknown operation."""


@dataclass
class Step:
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    skip: bool = False

    @property
    def label(self) -> str:
        return self.name or self.op


@dataclass
class Recipe:
    name: str
    steps: list[Step]
    description: str = ""
    variables: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str, overrides: dict[str, str] | None = None) -> Recipe:
        # Accept a plain string too: the CLI always hands over a Path, but a
        # library caller reasonably writes Recipe.load("build.yaml").
        path = Path(path)
        document = _load_document(path)
        return cls.from_dict(document, source=path, overrides=overrides)

    @classmethod
    def from_dict(
        cls,
        document: dict[str, Any],
        source: Path | None = None,
        overrides: dict[str, str] | None = None,
    ) -> Recipe:
        if not isinstance(document, dict):
            raise RecipeError("a recipe must be a mapping at the top level")
        raw_steps = document.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise RecipeError("a recipe needs a non-empty 'steps' list")

        # Overrides win over the file's own vars, so one recipe can be pointed
        # at a different dataset without being edited.
        variables = {str(k): str(v) for k, v in (document.get("vars") or {}).items()}
        variables.update({str(k): str(v) for k, v in (overrides or {}).items()})
        steps: list[Step] = []
        for position, entry in enumerate(raw_steps, start=1):
            if not isinstance(entry, dict) or "op" not in entry:
                raise RecipeError(f"step {position}: each step needs an 'op' key")
            op_name = str(entry["op"])
            try:
                operation = registry.get(op_name)
            except KeyError as exc:
                raise RecipeError(f"step {position}: {exc}") from None

            params = entry.get("with") or entry.get("params") or {}
            if not isinstance(params, dict):
                raise RecipeError(f"step {position} ({op_name}): 'with' must be a mapping")
            params = {k: _substitute(v, variables) for k, v in params.items()}

            # Validate now rather than three steps into a long run -- but do not
            # check that paths exist, since a step's input is often the previous
            # step's output and will not be there until the run reaches it.
            try:
                operation.bind(params, check_paths=False)
            except registry.ParamError as exc:
                raise RecipeError(f"step {position} ({op_name}): {exc}") from None

            steps.append(
                Step(op=op_name, params=params, name=str(entry.get("name", "")),
                     skip=bool(entry.get("skip", False)))
            )

        return cls(
            name=str(document.get("name") or (source.stem if source else "recipe")),
            description=str(document.get("description", "")),
            variables=variables,
            steps=steps,
        )


def _substitute(value: Any, variables: dict[str, str]) -> Any:
    """Expand ``${name}`` references, recursing through lists and mappings."""
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("${" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [_substitute(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, variables) for k, v in value.items()}
    return value


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RecipeError(
                f"reading {path.name} needs PyYAML installed, or write the recipe as .json ({exc})"
            ) from None
        return yaml.safe_load(text) or {}
    return json.loads(text)


@dataclass
class RunReport:
    """Outcome of a whole recipe, one entry per executed step."""

    recipe: str
    results: list[OpResult] = field(default_factory=list)
    stopped_at: str | None = None

    @property
    def ok(self) -> bool:
        return self.stopped_at is None and all(r.ok for r in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "ok": self.ok,
            "stopped_at": self.stopped_at,
            "steps": [r.as_dict(include_actions=False) for r in self.results],
        }


def run(
    recipe: Recipe,
    ctx: Context | None = None,
    *,
    continue_on_error: bool = False,
    on_step: Any = None,
) -> RunReport:
    """Execute every step in order.

    Stops at the first failure by default: later steps usually assume the
    earlier ones succeeded, and carrying on with a half-cleaned folder does more
    damage than stopping.
    """
    ctx = ctx or Context()
    report = RunReport(recipe=recipe.name)

    for position, step in enumerate(recipe.steps, start=1):
        if step.skip:
            continue
        ctx.check_cancelled()
        if on_step is not None:
            on_step(position, len(recipe.steps), step)

        operation = registry.get(step.op)
        result = operation.run(step.params, ctx)
        report.results.append(result)

        if not result.ok and not continue_on_error:
            report.stopped_at = step.label
            break

    return report
