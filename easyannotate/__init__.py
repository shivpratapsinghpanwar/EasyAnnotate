"""EasyAnnotate as a library: headless, scriptable, no server, no browser.

    import easyannotate as ea

    project = ea.open_folder("./doors")          # labels already there are loaded
    print(project.counts())                      # {'todo': 812, 'reviewed': 200, ...}
    project.import_predictions("runs/predict/labels")
    project.export("./train-set", to="yolo")     # reviewed images only, stable class indices

    ea.run("inspect.health", {"folder": "./doors"})   # any of the dataset operations

The same database the editor and the CLI use, so a script, a terminal and a
browser can take turns on one project. Training and prediction need the
model tools (``easyannotate tools install``); everything else needs nothing.

The implementation lives in ``data_factory``, the package's name before the
editor became the point; this module is the stable surface over it.
"""

from __future__ import annotations

from data_factory import (
    Context,
    OpResult,
    __version__,
    get_operation,
    list_operations,
    run,
    schema,
)

from .project import Project, database_path, list_projects, open_folder, open_project

__all__ = [
    "Context",
    "OpResult",
    "Project",
    "__version__",
    "database_path",
    "get_operation",
    "list_operations",
    "list_projects",
    "open_folder",
    "open_project",
    "run",
    "schema",
]
