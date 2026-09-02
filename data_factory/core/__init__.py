"""Pure logic for Data Factory.

Nothing here prints, parses argv, or exits. Operations take parameters and a
:class:`~data_factory.core.types.Context` and return an
:class:`~data_factory.core.types.OpResult`. That contract is what lets the same
code serve the CLI, a YAML pipeline, and an HTTP endpoint unchanged.
"""

from .types import Action, Cancelled, Context, FileAction, OpResult

__all__ = ["Action", "Cancelled", "Context", "FileAction", "OpResult"]
