# PyInstaller spec for a standalone EasyAnnotate desktop build.
#
#   pip install pyinstaller easyannotate
#   pyinstaller packaging/easyannotate.spec
#
# Produces dist/EasyAnnotate/EasyAnnotate.exe (or the platform equivalent), which
# launches the local server and opens a browser -- no Python needed on the
# target machine.
#
# For "Install AI tools" to work in the built app, build the two wheels first:
#
#   uv build --wheel --out-dir build/wheels
#   uv build --wheel --directory tools --out-dir ../build/wheels
#
# The app then carries uv (which fetches a Python for the tools environment)
# and both wheels. PyTorch and the models are never bundled: they are the
# gigabytes, and they download only when someone asks.

import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent

# uvicorn and its protocol backends resolve implementations by string at
# runtime, so nothing static can see them; without this the frozen binary
# starts and then fails to find its own HTTP protocol handler.
hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("anyio")
    + [
        "data_factory.service.app",
        "data_factory.desktop",
        # pywebview ships its own PyInstaller hooks for the platform backends;
        # the package itself still has to be reachable from the entry script.
        "webview",
        "encodings.idna",
        # Registered by decorator side effect, so an import scan misses them.
        "data_factory.core.ops",
        "yaml",
        "imagehash",
        "PIL.Image",
        "PIL._tkinter_finder",
    ]
)

datas = [
    # The built-in UI is a data file, not a module -- easy to forget, and the
    # failure is a 404 on the front page of the app.
    (str(ROOT / "data_factory" / "service" / "static"), "data_factory/service/static"),
]
datas += collect_data_files("data_factory")
# The version is read from package metadata at runtime. Without the dist-info
# the frozen build reports 0.0.0+source from --version and /api/health.
datas += copy_metadata("easyannotate")

# What "Install AI tools" needs on a machine with no Python. Found at
# sys._MEIPASS/bin and /wheels by data_factory.annotate.tools.install.
binaries = []
uv = shutil.which("uv")
if uv:
    binaries.append((uv, "bin"))
else:
    print("WARNING: uv not on PATH -- the built app cannot install the AI tools")
wheels = sorted((ROOT / "build" / "wheels").glob("*.whl"))
datas += [(str(wheel), "wheels") for wheel in wheels]
if len(wheels) < 2:
    print("WARNING: build/wheels lacks the easyannotate and easyannotate-tools wheels --"
          " the built app will fetch them from PyPI instead")

analysis = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    # Trimming what a dataset tool never needs. torch is excluded on purpose:
    # the SAM extra is not part of the desktop build, and including it would
    # roughly quadruple the download.
    excludes=["torch", "torchvision", "matplotlib", "tkinter", "pytest", "IPython"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="EasyAnnotate",
    debug=False,
    strip=False,
    upx=False,          # UPX-packed binaries trip antivirus heuristics
    console=True,       # keep the console: it is where progress and errors go
)

COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="EasyAnnotate",
)
