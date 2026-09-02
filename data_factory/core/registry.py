"""The operation registry: one declaration, four front-ends.

Every operation states its own parameters once. From that single declaration we
generate the argparse subcommand, the interactive menu prompts, the YAML step
validator, and the JSON schema a web app turns into a form. Adding an operation
therefore means writing one file -- it appears everywhere else for free, and no
front-end has to be edited.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .types import Context, OpResult

#: Parameter kinds. ``dir``/``file``/``path`` are all strings at runtime but
#: carry different intent -- a web form renders a folder picker for ``dir`` and
#: the CLI can validate existence before an operation starts.
PARAM_KINDS = (
    "dir",
    "file",
    "path",
    "str",
    "int",
    "float",
    "bool",
    "choice",
    "dirs",
    "strs",
    "mapping",
)


class ParamError(ValueError):
    """Raised when supplied parameters are missing, unknown, or ill-typed."""


@dataclass(slots=True)
class Param:
    name: str
    kind: str
    help: str
    default: Any = None
    required: bool = False
    choices: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    must_exist: bool = False

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"{self.name}: unknown param kind {self.kind!r}")
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"{self.name}: choice params need choices")

    # -- coercion ----------------------------------------------------------
    def coerce(self, raw: Any, *, check_exists: bool = True) -> Any:
        """Turn a string (CLI/YAML/HTTP form) into the real Python value.

        ``check_exists`` is switched off when validating a recipe up front: a
        later step's input folder is legitimately created by an earlier step,
        so it will not exist yet at validation time.
        """
        if raw is None:
            return None
        verify = self.must_exist and check_exists
        if self.kind in ("dir", "file", "path"):
            path = Path(str(raw)).expanduser()
            if verify and not path.exists():
                raise ParamError(f"{self.name}: path does not exist: {path}")
            if verify and self.kind == "dir" and not path.is_dir():
                raise ParamError(f"{self.name}: not a directory: {path}")
            return path
        if self.kind == "dirs":
            items = raw if isinstance(raw, (list, tuple)) else _split_list(str(raw))
            paths = [Path(str(i)).expanduser() for i in items]
            if verify:
                missing = [str(p) for p in paths if not p.is_dir()]
                if missing:
                    raise ParamError(f"{self.name}: missing folders: {', '.join(missing)}")
            return paths
        if self.kind == "strs":
            if isinstance(raw, (list, tuple)):
                return [str(i) for i in raw]
            return _split_list(str(raw))
        if self.kind == "mapping":
            return _coerce_mapping(self.name, raw)
        if self.kind == "int":
            value = int(raw)
            self._check_range(value)
            return value
        if self.kind == "float":
            number = float(raw)
            self._check_range(number)
            return number
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off"):
                return False
            raise ParamError(f"{self.name}: expected a boolean, got {raw!r}")
        if self.kind == "choice":
            text = str(raw)
            assert self.choices is not None
            if text not in self.choices:
                raise ParamError(
                    f"{self.name}: {text!r} is not one of {', '.join(self.choices)}"
                )
            return text
        return str(raw)

    def _check_range(self, value: float) -> None:
        if self.minimum is not None and value < self.minimum:
            raise ParamError(f"{self.name}: must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ParamError(f"{self.name}: must be <= {self.maximum}")

    # -- introspection -----------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "help": self.help,
            "required": self.required,
            "default": str(self.default) if isinstance(self.default, Path) else self.default,
        }
        if self.choices:
            out["choices"] = list(self.choices)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        return out


def _split_list(text: str) -> list[str]:
    """Split a comma- or semicolon-separated list, tolerating stray spaces."""
    parts = [p.strip() for chunk in text.split(";") for p in chunk.split(",")]
    return [p for p in parts if p]


def _coerce_mapping(name: str, raw: Any) -> dict[str, str]:
    """Accept a dict (YAML/JSON) or an ``old=new,old2=new2`` string (CLI)."""
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    mapping: dict[str, str] = {}
    for item in _split_list(str(raw)):
        if "=" not in item:
            raise ParamError(f"{name}: expected old=new pairs, got {item!r}")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


OpFunc = Callable[[dict[str, Any], Context], OpResult]


@dataclass(slots=True)
class Operation:
    """A single unit of work, plus everything a front-end needs to drive it."""

    name: str
    title: str
    category: str
    summary: str
    params: tuple[Param, ...]
    func: OpFunc
    destructive: bool = False
    #: Longer prose shown by ``datafactory explain <op>`` and web tooltips.
    notes: str = ""
    #: Given bound parameters, return True when this particular run changes
    #: nothing -- ``--action report`` on an otherwise destructive operation, say.
    safe_when: Callable[[dict[str, Any]], bool] | None = None

    def destructive_for(self, params: dict[str, Any]) -> bool:
        """Whether *this* invocation, with these parameters, writes anything."""
        if not self.destructive:
            return False
        if self.safe_when is not None and self.safe_when(params):
            return False
        return True

    def param(self, name: str) -> Param:
        for candidate in self.params:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def bind(self, supplied: dict[str, Any], *, check_paths: bool = True) -> dict[str, Any]:
        """Validate supplied values, apply defaults, and coerce types.

        Every front-end funnels through here, so a YAML recipe and an HTTP POST
        get byte-identical validation to a typed command line.

        ``check_paths=False`` validates everything except whether paths exist --
        what a recipe needs at load time, since step 7's input is step 6's output.
        """
        known = {p.name for p in self.params}
        unknown = set(supplied) - known
        if unknown:
            raise ParamError(
                f"{self.name}: unknown parameter(s): {', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(sorted(known))}"
            )
        bound: dict[str, Any] = {}
        for spec in self.params:
            raw = supplied.get(spec.name)
            if raw is None or raw == "":
                if spec.required:
                    raise ParamError(
                        f"{self.name}: missing required parameter {spec.name!r} ({spec.help})"
                    )
                bound[spec.name] = (
                    spec.coerce(spec.default, check_exists=check_paths)
                    if spec.default is not None
                    else spec.default
                )
            else:
                bound[spec.name] = spec.coerce(raw, check_exists=check_paths)
        return bound

    def run(self, supplied: dict[str, Any], ctx: Context | None = None) -> OpResult:
        return self.func(self.bind(supplied), ctx or Context())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "category": self.category,
            "summary": self.summary,
            "notes": self.notes,
            "destructive": self.destructive,
            "params": [p.as_dict() for p in self.params],
        }


_REGISTRY: dict[str, Operation] = {}


def operation(
    name: str,
    *,
    title: str,
    category: str,
    summary: str,
    params: Iterable[Param] = (),
    destructive: bool = False,
    notes: str = "",
    safe_when: Callable[[dict[str, Any]], bool] | None = None,
) -> Callable[[OpFunc], OpFunc]:
    """Decorator registering a function as a Data Factory operation."""

    def decorate(func: OpFunc) -> OpFunc:
        if name in _REGISTRY:
            raise ValueError(f"duplicate operation name: {name}")
        _REGISTRY[name] = Operation(
            name=name,
            title=title,
            category=category,
            summary=summary,
            params=tuple(params),
            func=func,
            destructive=destructive,
            notes=notes,
            safe_when=safe_when,
        )
        return func

    return decorate


def get(name: str) -> Operation:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown operation {name!r}. Known: {', '.join(sorted(_REGISTRY))}"
        ) from None


def all_operations() -> list[Operation]:
    return sorted(_REGISTRY.values(), key=lambda op: (op.category, op.name))


def categories() -> dict[str, list[Operation]]:
    grouped: dict[str, list[Operation]] = {}
    for op in all_operations():
        grouped.setdefault(op.category, []).append(op)
    return grouped


def schema() -> dict[str, Any]:
    """Machine-readable description of the whole tool.

    ``datafactory schema`` writes this to a file; a web front-end reads it and
    renders a form per operation without importing any Python.
    """
    return {"version": 1, "operations": [op.as_dict() for op in all_operations()]}
