"""Entry point for the frozen desktop build.

Kept separate from ``cli/main.py`` because a double-clicked application should
do the obvious thing with no arguments -- start the interface and open a
browser -- while the command line should not.

It also picks a free port. A fixed one is fine until someone already has 8765
in use, at which point the app dies at startup with a stack trace, which is a
poor first impression for a double-clicked icon.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

PREFERRED_PORT = 8765


def free_port(preferred: int) -> int:
    """The preferred port if it is free, otherwise whatever the OS hands us."""
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
                return probe.getsockname()[1]
            except OSError:
                continue
    return preferred


def workspace() -> Path:
    """Somewhere writable that survives an app update.

    Never beside the executable: on Windows that is Program Files, which is
    read-only for a normal user, and the first upload would fail.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    path = base / "DataFactory" / "workspaces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    os.environ.setdefault("DATAFACTORY_WORKSPACE", str(workspace()))
    port = free_port(PREFERRED_PORT)
    url = f"http://127.0.0.1:{port}"

    print("Data Factory")
    print(f"  interface : {url}")
    print(f"  datasets  : {os.environ['DATAFACTORY_WORKSPACE']}")
    print("  close this window to quit")
    print()

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    from data_factory.service.app import app

    # The app object rather than an import string: the frozen build has no
    # importable module path for uvicorn's reloader to resolve.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
