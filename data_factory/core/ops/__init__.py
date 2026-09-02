"""Importing this package registers every operation.

Modules are imported for their decorator side effects. A new operation module
added here becomes available to the CLI, the menu, the YAML runner, and the web
schema at once -- no front-end needs editing.
"""

from . import audit, collect, dataset, dedup, export, labels, pairing, rename, visualize

__all__ = [
    "audit",
    "collect",
    "dataset",
    "dedup",
    "export",
    "labels",
    "pairing",
    "rename",
    "visualize",
]
