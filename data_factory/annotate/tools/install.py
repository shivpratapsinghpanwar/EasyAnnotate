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

#: Run in the new environment. Real work on the device, not
#: torch.cuda.is_available(): measured on a 529 driver with the cu126 build,
#: that said True while every allocation failed ("device busy or unavailable").
_PROBE = '''
import json, sys
out = {"device": "cpu", "error": ""}
import torch
out["torch"] = torch.__version__
def works(device):
    x = torch.randn(128, 128, device=device)
    float((x @ x).sum())
try:
    if torch.cuda.is_available():
        works("cuda:%d" % int(sys.argv[1]))
        out["device"] = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        works("mps")
        out["device"] = "mps"
except Exception as exc:
    out["error"] = (type(exc).__name__ + ": " + str(exc)).splitlines()[0][:300]
print("PROBE " + json.dumps(out))
'''


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
    # Gone until this install has verified a working PyTorch: a half-finished
    # install must not still read as "installed" with the old device.
    (home / PROFILE).unlink(missing_ok=True)

    def run(command: list[str]) -> None:
        on_line("$ " + " ".join(str(part) for part in command))
        try:
            process = subprocess.Popen(
                [str(part) for part in command],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", env=child_environment(),
            )
        except OSError as exc:
            raise InstallError(f"could not start {command[0]}: {exc}") from None
        assert process.stdout is not None
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    on_line(line)
            code = process.wait()
        finally:
            # Cancelled, or the server stopping: never leave pip running on.
            if process.poll() is None:
                process.kill()
                process.wait()
        if code != 0:
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
        packages = list(chosen.torch_packages) or ["torch", "torchvision"]
        run(pip + [*packages, "--index-url", chosen.torch_index, *force])

    wheels = bundled_wheels()
    if wheels:
        packages = [str(w) for w in wheels]
    elif source is not None:
        packages = ["-e", str(source), "-e", str(Path(source) / "tools")]
    else:
        # Pinned to this version: the two are released together, and an
        # unpinned name is whatever PyPI says it is today.
        from ... import __version__

        packages = [f"easyannotate-tools=={__version__}"]
    on_line("installing the EasyAnnotate tools and Ultralytics")
    run(pip + packages)

    profile = {
        "device": chosen.device,
        "sam": chosen.sam,
        "world": chosen.world,
        "summary": chosen.summary,
        "gpu_index": chosen.gpu_index,
        "hardware": hw.as_dict(),
    }
    on_line("checking that PyTorch runs on this machine")
    try:
        probe = subprocess.run(
            [str(python), "-I", "-c", _PROBE, str(chosen.gpu_index)],
            capture_output=True, text=True, timeout=600, env=child_environment(),
        )
    except subprocess.TimeoutExpired:
        raise InstallError("PyTorch took over 10 minutes to start -- try again, or `tools install --cpu`") from None
    found = [line for line in (probe.stdout or "").splitlines() if line.startswith("PROBE ")]
    if probe.returncode != 0 or not found:
        tail = " ".join((probe.stderr or "").strip().splitlines()[-2:])
        raise InstallError(f"PyTorch does not start in {home}: {tail or 'no output'}")
    result = json.loads(found[-1][len("PROBE "):])
    seen = result.get("device") or "cpu"
    profile["verified_device"] = seen
    profile["torch"] = result.get("torch", "")
    if chosen.device != "cpu" and seen != chosen.device:
        why = result.get("error") or f"PyTorch sees {seen}"
        on_line(f"warning: the {chosen.device} build does not run here ({why}) -- the tools will use the CPU")
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
