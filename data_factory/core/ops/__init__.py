"""Importing this package registers every operation.

Modules are imported for their decorator side effects. A new operation module
added here becomes available to the CLI, the menu, the YAML runner, and the web
schema at once -- no front-end needs editing.
"""

from . import (
    analyze,
    annotate_export,
    audit,
    augment,
    collect,
    combine,
    convert,
    dataset,
    dedup,
    export,
    image,
    labels,
    pairing,
    rename,
    report,
    visualize,
)

__all__ = [
    "analyze",
    "annotate_export",
    "audit",
    "augment",
    "collect",
    "combine",
    "convert",
    "dataset",
    "dedup",
    "export",
    "image",
    "labels",
    "pairing",
    "rename",
    "report",
    "visualize",
]
