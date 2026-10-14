"""The local link: the browser opens folders on this computer -- safely.

Everywhere else in the service a client may never name a filesystem path,
because a web server that reads paths from requests is a file-reading
primitive for anyone who can reach it. On your own machine that is still
true of *other* websites: any page you visit can make your browser send a
request to ``localhost:8765``.

So opening a folder by path is the one exception, and it is fenced four ways:

1. **A per-launch secret.** The CLI (or desktop launcher) mints a token and
   puts it in the URL's ``#fragment`` -- which a browser never sends to any
   server, never logs, never leaks in a Referer. The page moves it into
   session storage and sends it as ``X-Local-Token``. No other site has it.
2. **Host must be this machine.** ``localhost``, ``127.0.0.1`` or ``::1``.
   A DNS-rebinding attack arrives with the attacker's hostname, and stops here.
3. **No foreign Origin.** A browser stamps cross-site requests with the
   page's origin; anything that is not a local origin is refused. (The custom
   header also forces a CORS preflight, which this server never grants.)
4. **Local mode at all.** Only the CLI and desktop launchers set the token,
   only when bound to this machine with no API key -- the hosted and Docker
   paths never do, and ``DATAFACTORY_ALLOW_ATTACHED=0`` turns it off.

And it grants one capability: a folder becomes a project, and the page gets
the project's *id*. Operations still run through the sandbox; the editor
still never sees a path. That is the same thing the desktop window's
Open-folder button does, for a browser.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from ...annotate import attach
from ...annotate import projects as projects_mod
from ...annotate.store import StoreError
from ..deps import get_store

TOKEN_ENV = "EASYANNOTATE_LOCAL_TOKEN"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

router = APIRouter(prefix="/api/local", tags=["local"])


def mint_token() -> str:
    """Called by a launcher that is serving this machine only."""
    token = secrets.token_urlsafe(24)
    os.environ[TOKEN_ENV] = token
    return token


def local_mode_enabled() -> bool:
    return bool(os.environ.get(TOKEN_ENV)) and projects_mod.attached_allowed()


def _hostname(value: str) -> str:
    """The host part of a Host header or an origin, without port or brackets."""
    if not value:
        return ""
    if "://" not in value:
        value = "http://" + value
    try:
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


def require_local(request: Request, x_local_token: str = Header(default="")) -> None:
    expected = os.environ.get(TOKEN_ENV, "")
    if not expected or not projects_mod.attached_allowed():
        raise HTTPException(404, "local mode is off -- start EasyAnnotate on this computer")
    if not x_local_token or not secrets.compare_digest(x_local_token, expected):
        raise HTTPException(403, "this page was not opened by EasyAnnotate on this computer")
    if _hostname(request.headers.get("host", "")) not in LOCAL_HOSTS:
        raise HTTPException(403, "local requests must be addressed to this computer")
    origin = request.headers.get("origin")
    if origin is not None and _hostname(origin) not in LOCAL_HOSTS:
        raise HTTPException(403, "requests from other sites are refused")


class OpenIn(BaseModel):
    path: str


@router.get("/status", dependencies=[Depends(require_local)])
def status() -> dict[str, Any]:
    """The token works. Says whether a native folder dialog can be offered."""
    return {"ok": True, "dialog": _dialog_available()}


@router.post("/open", dependencies=[Depends(require_local)])
def open_path(body: OpenIn, store=Depends(get_store)) -> dict[str, Any]:
    folder = Path(body.path.strip().strip('"')).expanduser()
    if not folder.is_absolute():
        raise HTTPException(400, "give the full path to a folder, e.g. D:\\photos\\doors")
    if not folder.is_dir():
        raise HTTPException(400, f"there is no folder at {folder}")
    return _opened(store, folder)


@router.post("/pick", dependencies=[Depends(require_local)])
def pick(store=Depends(get_store)) -> dict[str, Any]:
    """The operating system's own folder dialog, on this computer."""
    if not _dialog_available():
        raise HTTPException(501, "no folder dialog here -- type the path instead")
    chosen = _run_dialog()
    if not chosen:
        return {"cancelled": True}
    return _opened(store, Path(chosen))


def _opened(store, folder: Path) -> dict[str, Any]:
    try:
        summary = attach.open_folder(store, folder)
    except StoreError as exc:
        raise HTTPException(400, str(exc)) from None
    project = summary["project"]
    return {
        "project": project.id,
        "name": project.name,
        "images": summary["images"],
        "imported": summary["shapes_imported"],
        "writes_back": summary["writes_back"],
    }


def _dialog_available() -> bool:
    # A frozen build has no interpreter to run the dialog in (and the desktop
    # window has its own); a headless Linux box has no display to show it on.
    if getattr(sys, "frozen", False):
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


_DIALOG = (
    "import tkinter, tkinter.filedialog as d\n"
    "r = tkinter.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "print(d.askdirectory(title='Open a folder in EasyAnnotate') or '', end='')\n"
)


def _run_dialog() -> str:
    """In a child process: Tk wants its own main thread, which a request
    handler is not, and a dialog left open must not wedge the server."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", _DIALOG], capture_output=True, text=True, timeout=600
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip()
