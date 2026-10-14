"""Installing the model tools: one environment, chosen for this machine.

Shared by ``easyannotate tools install`` and the desktop/local web app's
"Install AI tools" button. Nothing heavy is downloaded until someone asks.

The environment lives at ``~/.datafactory/tools`` (``DATAFACTORY_TOOLS_HOME``),
apart from the editor's: torch's version matrix is resolved there alone, and
a broken install there breaks nothing but the AI buttons.

Where the pieces come from, in order of preference:

* **uv** -- bundled inside the desktop app, else on PATH. It creates the
  environment and fetches a Python for it, so a frozen app with no Python of
  its own can still install. Without uv, a pip install falls back to venv+pip.
* **easyannotate + easyannotate-tools** -- wheels bundled with the desktop app
  (``EASYANNOTATE_WHEELS`` or the app's ``wheels/`` folder), else a source
  checkout (``--source``), else PyPI.
* **PyTorch** -- the build ``hardware.plan()`` picked: the CUDA version this
  machine's *driver* supports, Metal on Apple silicon, or CPU.

A ``profile.json`` beside the environment records the device and model sizes,
and the sidecar reads it, so the same code runs SAM tiny on a laptop CPU and
SAM base on a 12 GB card.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ... import hardware
from .local import child_environment, python_for_tools

PROFILE = "profile.json"
PYTHON_VERSION = "3.12"


class InstallError(RuntimeError):
    pass


def tools_home() -> Path:
    return Path(os.environ.get("DATAFACTORY_TOOLS_HOME") or Path.home() / ".datafactory" / "tools")


def tools_python(home: Path | None = None) -> Path:
    home = home or tools_home()
    return home / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def read_profile(home: Path | None = None) -> dict[str, Any]:
    path = (home or tools_home()) / PROFILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def status() -> dict[str, Any]:
    """Installed or not, and what this machine would get."""
    hw = hardware.detect()
    chosen = hardware.plan(hw)
    profile = read_profile()
    return {
        "installed": tools_python().is_file() and bool(profile),
        "profile": profile,
        "hardware": hw.as_dict(),
        "plan": chosen.as_dict(),
        "uv": bool(find_uv()),
    }


def find_uv() -> str | None:
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "bin" / ("uv.exe" if sys.platform == "win32" else "uv")
    if getattr(sys, "_MEIPASS", None) and bundled.is_file():
        return str(bundled)
    return shutil.which("uv")


def bundled_wheels() -> list[Path]:
    folders = [os.environ.get("EASYANNOTATE_WHEELS", "")]
    if getattr(sys, "_MEIPASS", None):
        folders.append(str(Path(sys._MEIPASS) / "wheels"))  # noqa: SLF001
    for folder in folders:
        if folder and Path(folder).is_dir():
            found = sorted(Path(folder).glob("*.whl"))
            if found:
                return found
    return []


def install(
    *,
    source: Path | None = None,
    device: str | None = None,
    on_line: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Create or update the tools environment. Returns the written profile.

    ``device`` forces "cuda", "mps" or "cpu" instead of the detected choice.
    """
    hw = hardware.detect()
    chosen = hardware.plan(hw)
    if device == "cpu" and chosen.device != "cpu":
        chosen = hardware.plan(hardware.Hardware(os=hw.os, cpu_count=hw.cpu_count, ram_gb=hw.ram_gb))
    elif device in ("cuda", "mps") and chosen.device != device:
        raise InstallError(f"asked for {device}, but this machine looks like: {chosen.summary}")
    for warning in chosen.warnings:
        on_line(f"note: {warning}")
    on_line(f"plan: {chosen.summary} (about {chosen.download_gb:g} GB)")

    home = tools_home()
    python = tools_python(home)
    uv = find_uv()

    def run(command: list[str]) -> None:
        on_line("$ " + " ".join(str(part) for part in command))
        process = subprocess.Popen(
            [str(part) for part in command],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env=child_environment(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                on_line(line)
        if process.wait() != 0:
            raise InstallError(f"failed: {' '.join(str(part) for part in command[:4])} ...")

    if not python.is_file():
        on_line(f"creating {home}")
        if uv:
            run([uv, "venv", str(home), "--python", PYTHON_VERSION])
        elif not getattr(sys, "frozen", False):
            run([sys.executable, "-m", "venv", str(home)])
        else:
            raise InstallError("this build has no uv to create the environment with")
    pip = [uv, "pip", "install", "--python", str(python)] if uv else [str(python), "-m", "pip", "install"]

    if chosen.torch_index:
        on_line("installing PyTorch -- the large part of the download")
        # Reinstall, not install: a PyTorch already there -- the CPU build,
        # say -- "satisfies" plain `torch`, so asking for the CUDA build would
        # quietly change nothing. Measured: that is exactly what happened.
        force = (
            ["--reinstall-package", "torch", "--reinstall-package", "torchvision"]
            if uv else ["--force-reinstall"]
        )
        run(pip + ["torch", "torchvision", "--index-url", chosen.torch_index, *force])

    wheels = bundled_wheels()
    if wheels:
        packages = [str(w) for w in wheels]
    elif source is not None:
        packages = ["-e", str(source), "-e", str(Path(source) / "tools")]
    else:
        packages = ["easyannotate-tools"]
    on_line("installing the EasyAnnotate tools and Ultralytics")
    run(pip + packages)

    profile = {
        "device": chosen.device,
        "sam": chosen.sam,
        "world": chosen.world,
        "summary": chosen.summary,
        "hardware": hw.as_dict(),
    }
    probe = subprocess.run(
        [str(python), "-c", "import torch; print('cuda' if torch.cuda.is_available() else "
         "('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))"],
        capture_output=True, text=True, timeout=300, env=child_environment(),
    )
    seen = (probe.stdout or "").strip() or "unknown"
    profile["verified_device"] = seen
    if chosen.device != "cpu" and seen != chosen.device:
        on_line(f"warning: PyTorch sees {seen}, not {chosen.device} -- the tools will run on the CPU")
        # Then use the sizes a CPU can carry, and say what really happened.
        cpu = hardware.plan(hardware.Hardware(os=hw.os, cpu_count=hw.cpu_count, ram_gb=hw.ram_gb))
        profile.update({
            "device": "cpu", "sam": cpu.sam, "world": cpu.world,
            "summary": f"{cpu.summary} (the GPU build did not load -- see the log)",
        })
    (home / PROFILE).write_text(json.dumps(profile, indent=2), encoding="utf-8")
    on_line(f"done: {profile['summary']} (PyTorch sees: {seen})")
    return profile


__all__ = ["InstallError", "install", "python_for_tools", "read_profile", "status", "tools_home"]
