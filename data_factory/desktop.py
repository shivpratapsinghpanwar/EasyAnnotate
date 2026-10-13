"""EasyAnnotate as a desktop application: a native window over the local server.

Same server, same pages, same code as ``easyannotate serve`` -- the window is
the only new part, and it is a window and nothing more. This is how the
binary launches, and what ``easyannotate app`` runs from a pip install.

Two things only the desktop can do, and this is where they live:

* **Open a folder with the system's own dialog.** The rule everywhere else is
  that a client never sends a filesystem path over HTTP. Here the path comes
  from the operating system's folder picker, inside this process, and never
  crosses the network -- the one place that rule can be honoured and a
  folder still opened by clicking.
* **Pick a port, not demand one.** A fixed port dies at startup when it is
  taken, with a stack trace, which is a poor first impression for a
  double-clicked icon.

When no window can be made -- pywebview not installed, no WebView2 runtime on
an old Windows, ``EASYANNOTATE_NO_WINDOW=1`` in CI -- it does what ``serve``
does: opens the browser. Degrading, not failing.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

PREFERRED_PORT = 8765
TITLE = "EasyAnnotate"


def free_port(preferred: int = PREFERRED_PORT) -> int:
    """The preferred port if it is free, otherwise whatever the OS hands us."""
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
                return probe.getsockname()[1]
            except OSError:
                continue
    return preferred


def default_workspace() -> Path:
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
    path = base / "EasyAnnotate" / "workspaces"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Bridge:
    """What the page may ask the desktop for.

    Exposed to the page as ``window.pywebview.api``. Deliberately tiny: every
    method here is a capability the browser version does not have, and each
    one is a line the sandbox otherwise draws.
    """

    def __init__(self) -> None:
        # Underscored on purpose. pywebview exposes every *public* attribute
        # of this object to the page, walking into it recursively -- and the
        # window holds native COM objects that may only be touched from the
        # UI thread. Making it public flooded the log with errors and starved
        # the server. Found by running the binary, not by reading it.
        self._window: Any = None

    def attach(self, window: Any) -> None:
        self._window = window

    def pick_folder(self) -> dict[str, Any]:
        """Native folder dialog, then open that folder as an attached project."""
        import webview

        if self._window is None:
            return {"error": "the window is not ready yet"}
        chosen = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if not chosen:
            return {"cancelled": True}
        folder = Path(chosen[0] if isinstance(chosen, (list, tuple)) else chosen)

        from .annotate import attach
        from .annotate.store import StoreError
        from .service.deps import get_store

        try:
            summary = attach.open_folder(get_store(), folder)
        except StoreError as exc:
            return {"error": str(exc)}
        project = summary["project"]
        return {
            "project": project.id,
            "name": project.name,
            "images": summary["images"],
            "imported": summary["shapes_imported"],
            "writes_back": summary["writes_back"],
        }

    def version(self) -> str:
        from . import __version__

        return __version__


def start_server(port: int) -> Any:
    """The service, on its own thread, listening before this returns."""
    import uvicorn

    from .service.app import app

    # The app object rather than an import string: the frozen build has no
    # importable module path for uvicorn's reloader to resolve.
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="easyannotate-server")
    thread.start()
    deadline = time.time() + 20
    while not server.started and thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("the server did not start")
    server._thread = thread  # noqa: SLF001 -- kept for the fallback's join
    return server


def open_window(url: str) -> bool:
    """Show the native window; return False if none could be made."""
    if os.environ.get("EASYANNOTATE_NO_WINDOW"):
        return False
    try:
        import webview
    except ImportError:
        return False

    bridge = Bridge()
    try:
        window = webview.create_window(
            TITLE,
            url,
            width=1440,
            height=900,
            min_size=(960, 600),
            js_api=bridge,
            text_select=False,
        )
        bridge.attach(window)
        print("  window    : native", flush=True)
        webview.start()  # blocks until the window closes
    except Exception as exc:  # no WebView2, no GTK, a display that is not there
        print(f"  window    : unavailable ({exc}); using the browser", flush=True)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    os.environ.setdefault("DATAFACTORY_WORKSPACE", str(default_workspace()))
    port = free_port()
    url = f"http://127.0.0.1:{port}"

    # Flushed: when the console is a pipe the banner would otherwise sit in a
    # buffer until exit, and the one thing a launcher must do is say its URL.
    print(TITLE, flush=True)
    print(f"  interface : {url}", flush=True)
    print(f"  datasets  : {os.environ['DATAFACTORY_WORKSPACE']}", flush=True)

    server = start_server(port)

    if "--browser" not in argv and open_window(url):
        server.should_exit = True
        return 0

    print("  close this window to quit", flush=True)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server._thread.join()  # noqa: SLF001
    except KeyboardInterrupt:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
