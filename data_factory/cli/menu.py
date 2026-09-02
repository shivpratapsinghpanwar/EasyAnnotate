"""The interactive menu shown when Data Factory is run with no arguments.

Built from the same registry the CLI uses, so it can never drift out of sync
with the available operations. It exists because the tool it replaces was a
folder of scripts you edited before running -- typing a path at a prompt is a
strictly better version of that, and it is what someone new to the tool can
actually use.
"""

from __future__ import annotations

from typing import Any, TextIO

from ..core import registry
from ..core.registry import Operation, ParamError
from ..core.types import Cancelled, Context
from . import render

BANNER = r"""
  ____        _         _____         _
 |  _ \  __ _| |_ __ _ |  ___|_ _  __| |_ ___  _ __ _   _
 | | | |/ _` | __/ _` || |_ / _` |/ _| __/ _ \| '__| | | |
 | |_| | (_| | || (_| ||  _| (_| | (_| || (_) | |  | |_| |
 |____/ \__,_|\__\__,_||_|  \__,_|\___|\__\___/|_|   \__, |
                                                     |___/
"""


def run_menu(out: TextIO, err: TextIO) -> int:
    style = render.Style(render._supports_colour(out))
    out.write(style.cyan(BANNER))
    out.write("  Dataset preparation for computer vision.\n")
    out.write(style.dim("  Every step previews before it changes anything.\n"))

    while True:
        operations = registry.all_operations()
        _print_menu(operations, out, style)
        choice = _prompt("\n  Choose an operation (or q to quit): ", out)
        if choice is None or choice.lower() in ("q", "quit", "exit"):
            out.write("\n  Bye.\n\n")
            return 0
        if not choice.strip():
            continue

        operation = _resolve(choice.strip(), operations)
        if operation is None:
            out.write(style.yellow(f"  No operation matches {choice!r}.\n"))
            continue

        try:
            _run_interactive(operation, out, err, style)
        except (KeyboardInterrupt, EOFError):
            out.write("\n  Cancelled, back to the menu.\n")


def _print_menu(operations: list[Operation], out: TextIO, style: render.Style) -> None:
    out.write("\n")
    current = ""
    for number, operation in enumerate(operations, start=1):
        if operation.category != current:
            current = operation.category
            out.write(style.bold(f"\n  {current.upper()}\n"))
        marker = style.yellow("!") if operation.destructive else " "
        out.write(f"   {number:>2}) {marker} {style.cyan(operation.name):<30} {operation.summary[:60]}\n")


def _resolve(choice: str, operations: list[Operation]) -> Operation | None:
    if choice.isdigit():
        index = int(choice) - 1
        return operations[index] if 0 <= index < len(operations) else None
    for operation in operations:
        if operation.name == choice:
            return operation
    matches = [o for o in operations if choice in o.name]
    return matches[0] if len(matches) == 1 else None


def _run_interactive(
    operation: Operation, out: TextIO, err: TextIO, style: render.Style
) -> None:
    out.write(f"\n{style.bold(operation.title)} -- {operation.summary}\n")
    if operation.notes:
        out.write(style.dim("\n" + operation.notes + "\n"))
    out.write(style.dim("\n  Press Enter to accept the default shown in brackets.\n\n"))

    supplied: dict[str, Any] = {}
    for param in operation.params:
        value = _ask_param(param, out, style)
        if value is None:
            out.write("  Cancelled.\n")
            return
        if value != "":
            supplied[param.name] = value

    try:
        bound = operation.bind(supplied)
    except ParamError as exc:
        out.write(style.red(f"\n  {exc}\n"))
        return

    # Destructive work previews first, always. Seeing the plan before agreeing
    # to it is the whole safety story of this tool.
    if operation.destructive_for(bound):
        out.write(style.yellow("\n  Previewing (nothing will be changed yet)...\n"))
        preview = operation.run(supplied, Context(dry_run=True))
        render.render_result(preview, out)
        if not preview.ok:
            return
        answer = _prompt("  Apply these changes for real? [y/N] ", out)
        if answer is None or answer.strip().lower() not in ("y", "yes"):
            out.write("  Left untouched.\n")
            return

    try:
        result = operation.run(supplied, Context(dry_run=False))
    except Cancelled:
        out.write("\n  Cancelled.\n")
        return
    except (RuntimeError, OSError) as exc:
        out.write(style.red(f"\n  error: {exc}\n"))
        return

    render.render_result(result, out)
    _show_equivalent_command(operation, supplied, out, style)


def _ask_param(param: registry.Param, out: TextIO, style: render.Style) -> Any:
    """Prompt until the value is acceptable, or the user gives up."""
    label = param.name.replace("_", " ")
    hint = f" [{param.default}]" if param.default is not None else ""
    if param.required:
        hint = style.red(" (required)")
    if param.choices:
        hint += style.dim(f" {{{'|'.join(param.choices)}}}")

    while True:
        out.write(style.dim(f"  {param.help}\n"))
        raw = _prompt(f"  {label}{hint}: ", out)
        if raw is None:
            return None
        raw = raw.strip()
        if not raw:
            if param.required:
                out.write(style.yellow("    This one is required.\n"))
                continue
            return ""
        try:
            param.coerce(raw)
        except (ParamError, ValueError) as exc:
            out.write(style.yellow(f"    {exc}\n"))
            continue
        return raw


def _show_equivalent_command(
    operation: Operation, supplied: dict[str, Any], out: TextIO, style: render.Style
) -> None:
    """Teach the CLI form, so the menu is a ramp rather than a dead end."""
    parts = ["datafactory", operation.name]
    for name, value in supplied.items():
        flag = "--" + name.replace("_", "-")
        text = str(value)
        parts.append(flag if text.lower() == "true" else f'{flag} "{text}"')
    out.write(style.dim(f"  Same thing from a script:\n    {' '.join(parts)}\n\n"))


def _prompt(message: str, out: TextIO) -> str | None:
    out.write(message)
    out.flush()
    try:
        return input()
    except (EOFError, KeyboardInterrupt):
        return None
