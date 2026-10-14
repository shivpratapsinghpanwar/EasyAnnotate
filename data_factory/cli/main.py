"""The command line, generated from the operation registry.

Every subcommand and every flag below is derived from what an operation
declares about itself. Nothing here knows what dedup or splitting actually do,
which is why adding an operation never means editing this file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from .. import __version__
from ..core import (
    ops,  # noqa: F401  -- imported for registration side effects
    registry,
)
from ..core.registry import Operation, ParamError
from ..core.types import Cancelled, Context
from . import render

PROGRAM = "easyannotate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "EasyAnnotate -- annotate images with nothing to set up first, review a "
            "model's predictions fast, and get a dataset out that is correct. "
            f"Try '{PROGRAM} demo'. Run '{PROGRAM}' with no arguments for a menu."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            f"  {PROGRAM} demo\n"
            f"  {PROGRAM} annotate ./my-images\n"
            f"  {PROGRAM} inspect.dataset --folder ./data\n"
            f"  {PROGRAM} clean.dedup --folders ./a ./b --method phash --dry-run\n"
            f"  {PROGRAM} export.coco --folder ./voc --output ./coco\n"
            f"  {PROGRAM} run recipes/door.yaml\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    listing = subparsers.add_parser("list", help="list every available operation")
    listing.set_defaults(_handler=_cmd_list)

    explain = subparsers.add_parser("explain", help="show full help for one operation")
    explain.add_argument("operation", help="operation name, e.g. clean.dedup")
    explain.set_defaults(_handler=_cmd_explain)

    schema = subparsers.add_parser(
        "schema", help="dump the machine-readable operation schema (for a web UI)"
    )
    schema.add_argument("--output", type=Path, help="write to this file instead of stdout")
    schema.set_defaults(_handler=_cmd_schema)

    recipe = subparsers.add_parser("run", help="run a YAML or JSON recipe of steps")
    recipe.add_argument("recipe", type=Path, help="path to the recipe file")
    _add_common_flags(recipe)
    recipe.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override a recipe variable, e.g. --set root=D:/data (repeatable)",
    )
    recipe.add_argument(
        "--continue-on-error", action="store_true", help="keep going after a failing step"
    )
    recipe.set_defaults(_handler=_cmd_run_recipe)

    menu = subparsers.add_parser("menu", help="interactive menu")
    menu.set_defaults(_handler=_cmd_menu)

    serve = subparsers.add_parser(
        "serve", help="run the local web interface (this is also the desktop app)"
    )
    serve.add_argument("--host", default="127.0.0.1",
                       help="bind address; 127.0.0.1 keeps it on this machine only")
    serve.add_argument("--port", type=int, default=8765, help="port to listen on")
    serve.add_argument("--workspace", help="where uploaded datasets are stored")
    serve.add_argument("--no-browser", action="store_true", help="do not open a browser")
    serve.add_argument(
        "--headless",
        action="store_true",
        help="for a remote or GPU machine: no browser, and print how to reach it "
             "from your own (an SSH tunnel keeps it off the network)",
    )
    serve.add_argument("--api-key", help="require this key in the X-API-Key header")
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    serve.set_defaults(_handler=_cmd_serve)

    annotate = subparsers.add_parser(
        "annotate",
        help="open a folder in the annotation editor, editing its labels in place",
    )
    annotate.add_argument("folder", help="folder of images to annotate")
    annotate.add_argument("--name", default="", help="project name; defaults to the folder name")
    annotate.add_argument("--port", type=int, default=8765, help="port to listen on")
    annotate.add_argument("--host", default="127.0.0.1", help="bind address")
    annotate.add_argument("--workspace", help="where the annotation database lives")
    annotate.add_argument("--no-browser", action="store_true", help="do not open a browser")
    annotate.add_argument(
        "--no-import",
        action="store_true",
        help="ignore label files already in the folder instead of loading them",
    )
    annotate.set_defaults(_handler=_cmd_annotate)

    demo = subparsers.add_parser(
        "demo",
        help="open a ready-made sample project: some labelled, some predicted, some to do",
    )
    demo.add_argument("--folder", help="where to build it; defaults beside the database")
    demo.add_argument("--port", type=int, default=8765, help="port to listen on")
    demo.add_argument("--host", default="127.0.0.1", help="bind address")
    demo.add_argument("--workspace", help="where the annotation database lives")
    demo.add_argument("--no-browser", action="store_true", help="do not open a browser")
    demo.set_defaults(_handler=_cmd_demo)

    tools = subparsers.add_parser(
        "tools", help="model-backed helpers: SAM outlines and text-prompted detection"
    )
    tools.set_defaults(_handler=_cmd_tools_help)
    tools_sub = tools.add_subparsers(dest="tools_command")
    install = tools_sub.add_parser(
        "install", help="make ~/.datafactory/tools and install easyannotate-tools into it"
    )
    install.add_argument("--gpu", action="store_true",
                         help="require the GPU build (default: chosen from this machine's hardware)")
    install.add_argument("--cpu", action="store_true", help="install for CPU even if there is a GPU")
    install.add_argument(
        "--source", help="install from this checkout instead of PyPI (for developers)"
    )
    install.set_defaults(_handler=_cmd_tools_install)
    status = tools_sub.add_parser("status", help="what the editor will find")
    status.set_defaults(_handler=_cmd_tools_status)

    predict = subparsers.add_parser(
        "predict",
        help="run a trained model over a project's unreviewed images, as predictions to review",
    )
    predict.add_argument("target", help="the folder you annotate, or a project id")
    predict.add_argument("--model", help="an Ultralytics .pt; default: the project's latest trained model")
    predict.add_argument("--conf", type=float, default=0.25, help="drop predictions below this confidence")
    predict.add_argument("--workspace", help="where the annotation database lives")
    predict.set_defaults(_handler=_cmd_predict)

    trainer = subparsers.add_parser(
        "train",
        help="train a detector on the reviewed images, then predict the rest",
    )
    trainer.add_argument("target", help="the folder you annotate, or a project id")
    trainer.add_argument("--epochs", type=int, default=30)
    trainer.add_argument("--imgsz", type=int, default=640)
    trainer.add_argument("--base", default="yolov8n.pt", help="the model to start from")
    trainer.add_argument("--conf", type=float, default=0.25, help="for the predictions that follow")
    trainer.add_argument("--no-predict", action="store_true", help="train only")
    trainer.add_argument("--workspace", help="where the annotation database lives")
    trainer.set_defaults(_handler=_cmd_train)

    desktop = subparsers.add_parser(
        "app", help="open EasyAnnotate in its own window (falls back to the browser)"
    )
    desktop.add_argument("--browser", action="store_true", help="use the browser, not a window")
    desktop.set_defaults(_handler=_cmd_app)

    for operation in registry.all_operations():
        _add_operation_parser(subparsers, operation)

    return parser


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would happen without changing anything"
    )
    parser.add_argument("--json", action="store_true", help="emit the result as JSON")
    parser.add_argument("--verbose", action="store_true", help="list every action taken")
    parser.add_argument("--quiet", action="store_true", help="suppress the progress display")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="skip the confirmation prompt for destructive runs"
    )


def _add_operation_parser(subparsers: Any, operation: Operation) -> None:
    parser = subparsers.add_parser(
        operation.name,
        help=operation.summary,
        description=f"{operation.title} -- {operation.summary}",
        epilog=operation.notes,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for param in operation.params:
        _add_param_argument(parser, param)
    _add_common_flags(parser)
    parser.set_defaults(_handler=_cmd_run_operation, _operation=operation)


def _add_param_argument(parser: argparse.ArgumentParser, param: registry.Param) -> None:
    flag = "--" + param.name.replace("_", "-")
    help_text = param.help
    if param.default is not None:
        help_text += f"  (default: {param.default})"

    if param.kind == "bool":
        # Both polarities are always available, so a default of True is just as
        # easy to turn off as a default of False is to turn on.
        parser.add_argument(flag, dest=param.name, action="store_true", default=None, help=help_text)
        parser.add_argument(
            "--no-" + param.name.replace("_", "-"),
            dest=param.name,
            action="store_false",
            default=None,
            help=argparse.SUPPRESS,
        )
        return

    if param.kind in ("dirs", "strs"):
        parser.add_argument(flag, dest=param.name, nargs="+", default=None, help=help_text)
        return

    parser.add_argument(
        flag,
        dest=param.name,
        default=None,
        choices=list(param.choices) if param.choices else None,
        help=help_text,
    )


# -- commands ---------------------------------------------------------------

def _cmd_list(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    render.render_operation_list(registry.all_operations(), out)
    return 0


def _cmd_explain(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    try:
        operation = registry.get(args.operation)
    except KeyError as exc:
        err.write(f"{exc}\n")
        return 2
    render.render_operation_help(operation, out)
    return 0


def _cmd_schema(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    document = json.dumps(registry.schema(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
        out.write(f"schema written to {args.output}\n")
    else:
        out.write(document + "\n")
    return 0


def _cmd_menu(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from .menu import run_menu

    return run_menu(out, err)


def _local_fragment(args: argparse.Namespace) -> str:
    """``#token=...`` when this server is for this machine only -- the key that
    lets the browser open local folders (see service/routers/local.py). Never
    minted for a server reachable from the network or behind an API key."""
    import os

    from ..annotate import projects
    from ..service.routers.local import LOOPBACK_ENV, TOKEN_ENV, mint_token

    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    # Both decided here, by this launch -- never inherited from whatever
    # started this process.
    if loopback:
        os.environ[LOOPBACK_ENV] = "1"
    else:
        os.environ.pop(LOOPBACK_ENV, None)
    os.environ.pop(TOKEN_ENV, None)
    if not loopback or not projects.attached_allowed():
        return ""
    if getattr(args, "api_key", None) or os.environ.get("DATAFACTORY_API_KEY"):
        return ""
    return f"#token={mint_token()}"


def _url_host(host: str) -> str:
    """What to put in a URL for a bind address: a name, or bracketed IPv6."""
    if host in ("127.0.0.1", "0.0.0.0"):
        return "localhost"
    return f"[{host}]" if ":" in host else host


def _cmd_serve(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Start the local web interface.

    This doubles as the desktop application: a local server plus the system
    browser, which is how Jupyter and Netron ship. It avoids bundling a second
    browser engine, works identically on Windows, macOS and Linux, and the
    same code is what runs on a server -- so there is one thing to maintain
    rather than a desktop build and a web build that drift.
    """
    import os

    if args.workspace:
        os.environ["DATAFACTORY_WORKSPACE"] = str(Path(args.workspace).expanduser())
    if args.api_key:
        os.environ["DATAFACTORY_API_KEY"] = args.api_key

    try:
        import uvicorn
    except ImportError:
        err.write(
            "the web interface could not import uvicorn -- reinstall:\n"
            "    pip install --force-reinstall easyannotate\n"
        )
        return 2

    url = f"http://{_url_host(args.host)}:{args.port}"
    url += _local_fragment(args)
    out.write(f"EasyAnnotate is at {url}\n")
    if args.host == "0.0.0.0" and not args.api_key:
        err.write(
            "warning: binding to 0.0.0.0 exposes this to your whole network with no "
            "authentication. Pass --api-key, or bind to 127.0.0.1.\n"
        )
    out.write("Press Ctrl+C to stop.\n")
    out.flush()

    if args.headless and args.host not in ("127.0.0.1", "localhost", "::1"):
        args.no_browser = True
        out.write(
            f"headless, listening on {args.host}: reachable from the network.\n"
            "Protect it with --api-key, or bind to 127.0.0.1 and use an SSH tunnel.\n"
        )
        out.flush()
    elif args.headless:
        args.no_browser = True
        out.write(
            "headless: nothing opens here. From your own computer:\n"
            f"    ssh -L {args.port}:localhost:{args.port} <this-machine>\n"
            f"    then open http://localhost:{args.port}\n"
            "The server stays bound to this machine; the tunnel carries the editor to you.\n"
        )
        out.flush()

    if not args.no_browser:
        import threading
        import webbrowser

        # Opened on a delay so the server is listening by the time the tab
        # loads; otherwise the first request races the startup and fails.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(
        "data_factory.service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


def _cmd_annotate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Open a folder for annotation, editing its label files in place.

    This is the only way to create an attached project, and that is deliberate:
    it takes a path, and no HTTP route is ever allowed to. The path comes from
    a person typing it into their own terminal, not from a request.
    """
    import os

    if args.workspace:
        os.environ["DATAFACTORY_WORKSPACE"] = str(Path(args.workspace).expanduser())

    from ..annotate import attach
    from ..annotate.store import StoreError
    from ..service.deps import get_store, store_path

    folder = Path(args.folder).expanduser()
    try:
        summary = attach.open_folder(
            get_store(),
            folder,
            name=args.name,
            import_annotations=not args.no_import,
        )
    except StoreError as exc:
        err.write(f"{exc}\n")
        return 1

    project = summary["project"]
    out.write(f"\n{project.name}\n")
    out.write(f"  folder     {project.root}\n")
    out.write(
        f"  images     {summary['images']}"
        + (f" ({summary['new_images']} new)" if summary["reused"] else "")
        + "\n"
    )
    if summary["shapes_imported"]:
        out.write(f"  imported   {summary['shapes_imported']} existing shapes\n")
    if summary["writes_back"]:
        out.write(
            f"  writes to  {summary['sidecar_format'].upper()} files in that folder, "
            f"as you draw\n"
        )
    else:
        out.write(
            "  writes to  nothing on disk -- this format keeps one file for the\n"
            "             whole dataset, so use annotate.export to get it back\n"
        )
    out.write(f"  database   {store_path()}\n")
    for warning in summary["warnings"]:
        out.write(f"  note       {warning}\n")

    return _serve_editor(project.id, args, out)


def _serve_editor(project_id: str, args: argparse.Namespace, out: TextIO) -> int:
    """Start the server and open the editor on one project."""
    url = f"http://{_url_host(args.host)}:{args.port}/annotate?project={project_id}" + _local_fragment(args)
    out.write(f"\n{url}\n")
    out.write("Press Ctrl+C to stop.\n")
    out.flush()

    if not args.no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(
        "data_factory.service.app:app",
        host=args.host,
        port=args.port,
        log_level="warning",
    )
    return 0


def _cmd_tools_help(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    out.write("usage: easyannotate tools install [--gpu] [--source DIR]\n"
              "       easyannotate tools status\n")
    return 2


def _cmd_tools_install(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """A separate environment for the heavy tools, chosen for this machine --
    the CUDA build its driver supports, Metal, or CPU -- where torch's version
    matrix is resolved on its own and can never break the editor's install."""
    from ..annotate.tools import install as installer

    device = "cpu" if args.cpu else ("cuda" if args.gpu else None)
    try:
        installer.install(
            source=Path(args.source).expanduser() if args.source else None,
            device=device,
            on_line=lambda line: (out.write(f"  {line}\n"), out.flush()),
        )
    except installer.InstallError as exc:
        err.write(f"{exc}\n")
        return 1
    out.write("\nModel weights download on first use.\n")
    return _cmd_tools_status(args, out, err)


def _cmd_tools_status(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from ..annotate.tools import install as installer
    from ..annotate.tools.registry import REGISTRY

    state = installer.status()
    plan = state["plan"]
    out.write(f"this machine: {plan['summary']}\n")
    for warning in plan.get("warnings") or []:
        out.write(f"  note: {warning}\n")
    if state["installed"]:
        profile = state["profile"]
        out.write(
            f"installed:    {profile.get('summary', '?')} "
            f"(PyTorch sees {profile.get('verified_device', '?')})\n"
        )
    else:
        out.write(f"installed:    no -- easyannotate tools install (about {plan['download_gb']:g} GB)\n")
        return 1

    report = REGISTRY.as_dict()
    tools = report.get("tools") or []
    if not tools:
        health = report.get("health") or {}
        out.write(f"no tools: {report.get('reason') or health.get('detail') or 'none found'}\n")
        return 1
    for tool in tools:
        out.write(f"  {tool['name']:<14} {tool.get('title', '')}  [{tool.get('version', '')}]\n")
    REGISTRY.close()
    return 0


def _loop_target(args: argparse.Namespace, err: TextIO):
    """A folder (opened, or reopened, as an attached project) or a project id.
    Returns (store, project, image_root) or None after explaining why not."""
    import os

    if args.workspace:
        os.environ["DATAFACTORY_WORKSPACE"] = str(Path(args.workspace).expanduser())
    from ..annotate import attach
    from ..annotate import projects as projects_mod
    from ..annotate.store import StoreError
    from ..service.deps import get_store, media_root

    store = get_store()
    target = Path(args.target).expanduser()
    try:
        if target.is_dir():
            project = attach.open_folder(store, target)["project"]
        else:
            project = projects_mod.get(store, args.target, "local")
    except StoreError as exc:
        err.write(f"{exc}\n")
        return None
    root = Path(project.root) if project.is_attached and project.root else media_root(project.id)
    return store, project, root


def _progress_line(out: TextIO):
    last = [0.0]

    def report(done: int, total: int, name: str) -> None:
        now = time.time()
        if done == total or now - last[0] > 0.5:
            last[0] = now
            out.write(f"\r  {done}/{total}  {name[:50]:<50}")
            out.flush()
        if done == total:
            out.write("\n")

    return report


def _summarise_predictions(report: dict, project, out: TextIO) -> int:
    out.write(
        f"  {report['shapes']} prediction(s) on {report['predicted']} of {report['images']} "
        f"unreviewed image(s) in {report['seconds']}s"
        + (f"; {report['failed']} failed" if report["failed"] else "")
        + "\n"
    )
    if report.get("first_error"):
        out.write(f"  first failure: {report['first_error']}\n")
    if report["predicted"] == 0 and report["failed"]:
        return 1
    out.write(
        "  review them hardest first: open the editor and pick\n"
        "  'predicted, least sure first' in the image list.\n"
    )
    return 0


def _cmd_predict(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from ..annotate import loop

    found = _loop_target(args, err)
    if found is None:
        return 1
    store, project, root = found
    model = Path(args.model).expanduser() if args.model else loop.latest_model(project)
    if model is None:
        err.write("no model given and none trained for this project yet -- pass --model, or run train\n")
        return 1
    out.write(f"{project.name}: predicting with {model.name}\n")
    try:
        report = loop.predict(store, project, root, model, conf=args.conf, on_progress=_progress_line(out))
    except loop.LoopError as exc:
        err.write(f"{exc}\n")
        return 1
    return _summarise_predictions(report, project, out)


def _cmd_train(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from ..annotate import loop

    found = _loop_target(args, err)
    if found is None:
        return 1
    store, project, root = found
    out.write(f"{project.name}\n")
    try:
        trained = loop.train(
            store, project, epochs=args.epochs, imgsz=args.imgsz, base=args.base,
            on_line=lambda line: (out.write(f"  {line}\n"), out.flush()),
        )
    except loop.LoopError as exc:
        err.write(f"{exc}\n")
        return 1
    metrics = trained["metrics"]
    score = metrics.get("metrics/mAP50(B)")
    out.write(
        f"  model saved: {trained['model']}"
        + (f"  (mAP50 {score:.3f})" if isinstance(score, (int, float)) else "")
        + "\n"
    )
    if args.no_predict:
        return 0
    out.write("predicting the unreviewed images with it\n")
    try:
        report = loop.predict(store, project, root, trained["model"], conf=args.conf,
                              on_progress=_progress_line(out))
    except loop.LoopError as exc:
        err.write(f"{exc}\n")
        return 1
    return _summarise_predictions(report, project, out)


def _cmd_app(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from .. import desktop

    return desktop.main(["--browser"] if args.browser else [])


def _cmd_demo(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Build (once) and open the sample project.

    The first minute with a tool decides whether there is a second. This is
    one command from install to an editor with labelled images, predictions
    to review, and images still to do.
    """
    import os
    import tempfile

    if args.workspace:
        os.environ["DATAFACTORY_WORKSPACE"] = str(Path(args.workspace).expanduser())

    from ..annotate import attach, demo, review
    from ..service.deps import data_root, get_store

    folder = Path(args.folder).expanduser() if args.folder else data_root() / "demo"
    built = demo.build(folder)
    store = get_store()
    summary = attach.open_folder(store, folder, name="EasyAnnotate demo")
    project = summary["project"]

    if built["fresh"]:
        with tempfile.TemporaryDirectory() as holding:
            run = demo.write_predictions(built["predictions"], Path(holding) / "run")
            report = review.import_predictions(store, project, run, name="demo model")
        out.write(
            f"\nEasyAnnotate demo -- {summary['images']} images in {folder}\n"
            f"  {summary['shapes_imported']} shapes labelled by hand\n"
            f"  {report['shapes_imported']} model predictions to review: "
            f"A accepts an image, the slider hides weak guesses\n"
        )
    else:
        out.write(f"\nEasyAnnotate demo -- reopening {folder}\n")
    return _serve_editor(project.id, args, out)


def _cmd_run_operation(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    operation: Operation = args._operation
    supplied = {
        param.name: getattr(args, param.name)
        for param in operation.params
        if getattr(args, param.name, None) is not None
    }

    try:
        bound = operation.bind(supplied)
    except ParamError as exc:
        err.write(f"error: {exc}\n")
        return 2

    if not _confirm(operation, bound, args, err):
        return 1

    ctx = _make_context(args, err)
    try:
        result = operation.run(supplied, ctx)
    except Cancelled:
        err.write("\ncancelled\n")
        return 130
    except (RuntimeError, OSError) as exc:
        err.write(f"error: {exc}\n")
        return 1
    finally:
        _clear_progress(args, err)

    render.render_result(result, out, as_json=args.json, verbose=args.verbose)
    return 0 if result.ok else 1


def _cmd_run_recipe(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from ..pipeline import Recipe, RecipeError, run

    overrides: dict[str, str] = {}
    for item in args.overrides or []:
        if "=" not in item:
            err.write(f"error: --set expects KEY=VALUE, got {item!r}\n")
            return 2
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()

    try:
        recipe = Recipe.load(args.recipe, overrides=overrides)
    except (RecipeError, OSError, ValueError) as exc:
        err.write(f"error: {exc}\n")
        return 2

    destructive = [
        s
        for s in recipe.steps
        if not s.skip
        and registry.get(s.op).destructive_for(registry.get(s.op).bind(s.params, check_paths=False))
    ]
    if destructive and not args.dry_run and not args.yes:
        names = ", ".join(sorted({s.op for s in destructive}))
        if not _ask(f"Recipe '{recipe.name}' runs destructive steps ({names}). Continue?", err):
            err.write("aborted\n")
            return 1

    ctx = _make_context(args, err)

    def announce(position: int, total: int, step: Any) -> None:
        if not args.quiet and not args.json:
            out.write(f"\n[{position}/{total}] {step.label}\n")

    try:
        report = run(recipe, ctx, continue_on_error=args.continue_on_error, on_step=announce)
    except Cancelled:
        err.write("\ncancelled\n")
        return 130
    finally:
        _clear_progress(args, err)

    if args.json:
        json.dump(report.as_dict(), out, indent=2)
        out.write("\n")
    else:
        for result in report.results:
            render.render_result(result, out, verbose=args.verbose)
        if report.stopped_at:
            err.write(f"recipe stopped at step '{report.stopped_at}'\n")
    return 0 if report.ok else 1


# -- shared helpers ---------------------------------------------------------

def _make_context(args: argparse.Namespace, err: TextIO) -> Context:
    show_progress = not args.quiet and not args.json and hasattr(err, "isatty") and err.isatty()
    return Context(dry_run=args.dry_run, progress=_progress_printer(err) if show_progress else None)


def _progress_printer(err: TextIO) -> Any:
    """A single rewriting status line, throttled so it never dominates the run."""
    state = {"last": 0.0}

    def report(done: int, total: int, message: str = "") -> None:
        now = time.monotonic()
        if now - state["last"] < 0.08 and done != total:
            return
        state["last"] = now
        share = f"{done}/{total}" if total else str(done)
        line = f"  {share}  {message}"[:110]
        err.write("\r\033[K" + line)
        err.flush()

    return report


def _clear_progress(args: argparse.Namespace, err: TextIO) -> None:
    if not args.quiet and not args.json and hasattr(err, "isatty") and err.isatty():
        err.write("\r\033[K")
        err.flush()


def _confirm(
    operation: Operation, bound: dict[str, Any], args: argparse.Namespace, err: TextIO
) -> bool:
    # Judged from the actual parameters: --action report writes nothing, so it
    # should not demand a confirmation the way --action delete must.
    if not operation.destructive_for(bound) or args.dry_run or args.yes:
        return True
    if not sys.stdin.isatty():
        err.write(
            f"error: {operation.name} changes files. Re-run with --dry-run to preview, "
            "or --yes to confirm in a non-interactive session.\n"
        )
        return False
    return _ask(f"{operation.name} will modify files on disk. Continue?", err)


def _ask(question: str, err: TextIO) -> bool:
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        err.write("\n")
        return False
    return answer in ("y", "yes")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # Bare invocation is the friendly path: show the menu rather than a usage error.
    if not argv:
        from .menu import run_menu

        return run_menu(sys.stdout, sys.stderr)

    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 0

    try:
        return handler(args, sys.stdout, sys.stderr)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
