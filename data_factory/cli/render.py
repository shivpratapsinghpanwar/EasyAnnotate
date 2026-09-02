"""Turning an OpResult into terminal output.

All presentation lives here. The core returns data; this module is the only
place that decides how it looks, which is what keeps a web front-end free to
render the same result completely differently.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from ..core.registry import Operation
from ..core.types import OpResult

BULLET = "-"
RULE = "=" * 66


def _supports_colour(stream: TextIO) -> bool:
    return hasattr(stream, "isatty") and stream.isatty() and sys.platform != "emscripten"


class Style:
    """ANSI codes, disabled when the output is redirected."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap(text, "1")

    def dim(self, text: str) -> str:
        return self._wrap(text, "2")

    def red(self, text: str) -> str:
        return self._wrap(text, "31")

    def yellow(self, text: str) -> str:
        return self._wrap(text, "33")

    def green(self, text: str) -> str:
        return self._wrap(text, "32")

    def cyan(self, text: str) -> str:
        return self._wrap(text, "36")


def render_result(
    result: OpResult,
    stream: TextIO,
    *,
    as_json: bool = False,
    show_actions: int = 12,
    verbose: bool = False,
) -> None:
    if as_json:
        json.dump(result.as_dict(include_actions=verbose), stream, indent=2)
        stream.write("\n")
        return

    style = Style(_supports_colour(stream))
    header = style.bold(result.op)
    if result.dry_run:
        header += style.yellow("  [dry run - nothing was changed]")
    stream.write(f"\n{header}\n{RULE}\n")

    if result.counts:
        width = max(len(k) for k in result.counts)
        for key, value in result.counts.items():
            stream.write(f"  {key.replace('_', ' '):<{width}}  {value:>9,}\n")

    _render_data(result.data, stream, style)

    for warning in result.warnings:
        stream.write(style.yellow(f"  ! {warning}\n"))
    for error in result.errors:
        stream.write(style.red(f"  x {error}\n"))

    if result.actions and show_actions:
        shown = result.actions if verbose else result.actions[:show_actions]
        stream.write(style.dim(f"\n  actions ({len(result.actions)} total):\n"))
        for action in shown:
            target = f" -> {action.dst}" if action.dst else ""
            stream.write(style.dim(f"    {action.action.value:<7} {action.src.name}{target}\n"))
        if len(result.actions) > len(shown):
            stream.write(style.dim(f"    ... {len(result.actions) - len(shown)} more (use --verbose)\n"))

    verdict = style.green("done") if result.ok else style.red("failed")
    stream.write(f"\n  {verdict}\n\n")


def _render_data(data: dict[str, Any], stream: TextIO, style: Style) -> None:
    """Show the handful of data keys that are worth reading in a terminal."""
    issues = data.get("issues")
    if issues:
        stream.write(style.bold("\n  findings\n"))
        for issue in issues:
            severity = issue.get("severity", "")
            colour = {
                "high": style.red,
                "medium": style.yellow,
                "low": style.dim,
            }.get(severity, style.green)
            stream.write(f"    {colour(severity.upper() or 'OK'):<6} {issue['issue']}\n")
            if issue.get("fix"):
                stream.write(style.dim(f"           fix: {issue['fix']}\n"))

    for key in ("per_label", "per_category", "per_split_labels", "case_variants", "case_collisions"):
        section = data.get(key)
        if not section:
            continue
        stream.write(style.bold(f"\n  {key.replace('_', ' ')}\n"))
        for name, value in section.items():
            if isinstance(value, dict):
                inner = ", ".join(f"{k}={v}" for k, v in value.items())
                stream.write(f"    {name}: {inner}\n")
            elif isinstance(value, list):
                stream.write(f"    {name}: {' / '.join(map(str, value))}\n")
            else:
                stream.write(f"    {name:<24} {value:>8,}\n")

    for key in ("output", "quarantine", "target"):
        if data.get(key):
            stream.write(f"\n  {key}: {style.cyan(str(data[key]))}\n")


def render_operation_list(operations: list[Operation], stream: TextIO) -> None:
    style = Style(_supports_colour(stream))
    current = ""
    for op in operations:
        if op.category != current:
            current = op.category
            stream.write(style.bold(f"\n{current}\n"))
        marker = style.yellow(" ! ") if op.destructive else "   "
        stream.write(f"{marker}{style.cyan(op.name):<32} {op.summary}\n")
    stream.write(style.dim("\n  ! writes or removes files. Add --dry-run to preview any of them.\n\n"))


def render_operation_help(op: Operation, stream: TextIO) -> None:
    style = Style(_supports_colour(stream))
    stream.write(f"\n{style.bold(op.name)} -- {op.title}\n{RULE}\n{op.summary}\n")
    if op.destructive:
        stream.write(style.yellow("\nThis operation writes or removes files. Preview it with --dry-run.\n"))
    if op.notes:
        stream.write(f"\n{op.notes}\n")

    stream.write(style.bold("\nparameters\n"))
    for param in op.params:
        flag = "--" + param.name.replace("_", "-")
        tag = style.red("required") if param.required else style.dim(f"default: {param.default!r}")
        stream.write(f"  {style.cyan(flag):<34} {param.help}\n")
        detail = f"      {param.kind}, {tag}"
        if param.choices:
            detail += style.dim(f", one of: {', '.join(param.choices)}")
        stream.write(detail + "\n")
    stream.write("\n")
