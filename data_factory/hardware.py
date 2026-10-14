"""What this machine can run: GPU, VRAM, driver, RAM -- standard library only.

Used to decide, before anything heavy is downloaded, which PyTorch build to
install and which model sizes to use. Getting it wrong is expensive in a
specific way: a CUDA build newer than the driver installs fine (2 GB later)
and then silently runs on the CPU, and a model too big for the card fails
with an out-of-memory error on the first click.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Gpu:
    name: str
    vram_gb: float
    driver: str = ""
    #: The newest CUDA the *driver* supports -- what nvidia-smi's header says.
    cuda: str = ""
    #: CUDA's index for it, so the models run on the card they were sized for.
    index: int = 0


@dataclass
class Hardware:
    os: str
    cpu_count: int
    ram_gb: float
    gpus: list[Gpu] = field(default_factory=list)
    apple_silicon: bool = False

    @property
    def best_gpu(self) -> Gpu | None:
        return max(self.gpus, key=lambda g: g.vram_gb) if self.gpus else None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["apple_silicon"] = self.apple_silicon
        return out


def detect() -> Hardware:
    return Hardware(
        os=platform.system(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=round(_ram_bytes() / 2**30, 1),
        gpus=_nvidia_gpus(),
        apple_silicon=sys.platform == "darwin" and platform.machine() == "arm64",
    )


# ---------------------------------------------------------------- the choice

#: PyTorch CUDA builds, newest first: the oldest driver each needs
#: (Windows, Linux), and the packages to ask for (empty: the newest).
#:
#: The minimums are each toolkit's *own* driver, not CUDA's minor-version
#: compatibility floor. Measured on an RTX 2050 with driver 529.04 (CUDA
#: 12.0): the cu126 build said torch.cuda.is_available() and then failed
#: every allocation ("device busy or unavailable"); PyTorch 2.7.1+cu118 ran.
#: 2.7 is the last PyTorch with a cu118 build, so older drivers get it.
CUDA_BUILDS = (
    ("cu128", ("570.65", "570.26"), ()),
    ("cu126", ("560.76", "560.28.03"), ()),
    ("cu118", ("522.06", "520.61.05"), ("torch==2.7.1", "torchvision==0.22.1")),
)
TORCH_INDEX = "https://download.pytorch.org/whl/{build}"


@dataclass
class Plan:
    device: str                  # "cuda", "mps" or "cpu"
    torch_index: str | None      # None: PyPI's default build
    sam: str                     # SAM weights file
    world: str                   # YOLO-World weights file
    download_gb: float
    summary: str
    warnings: list[str] = field(default_factory=list)
    gpu_index: int = 0
    #: What to install from ``torch_index``; empty means the newest torch.
    torch_packages: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan(hardware: Hardware | None = None) -> Plan:
    """What to install for this machine, and why, in words a person can read."""
    hw = hardware or detect()
    warnings: list[str] = []
    gpu = hw.best_gpu

    if gpu is not None:
        build = _cuda_build_for(gpu.driver, hw.os)
        if build is None:
            warnings.append(
                f"{gpu.name}: driver {gpu.driver or 'unknown'} is too old for PyTorch's "
                "CUDA builds -- update the NVIDIA driver to use the GPU. Installing for CPU."
            )
        elif build == "cu118":
            warnings.append(
                f"driver {gpu.driver} predates current PyTorch CUDA builds: installing PyTorch 2.7 "
                "with CUDA 11.8. An NVIDIA driver update (560 or newer) gets the current one."
            )
        elif gpu.vram_gb < 3.5:
            warnings.append(f"{gpu.name} has {gpu.vram_gb:g} GB: too little for the models. Installing for CPU.")
            build = None
        if build is not None:
            large = gpu.vram_gb >= 8
            return Plan(
                device="cuda",
                torch_index=TORCH_INDEX.format(build=build),
                torch_packages=_packages_for(build),
                sam="sam2.1_b.pt" if large else "sam2.1_s.pt",
                world="yolov8l-worldv2.pt" if large else "yolov8m-worldv2.pt",
                download_gb=3.4 if large else 3.1,
                summary=(
                    f"{gpu.name}, {gpu.vram_gb:g} GB: PyTorch with CUDA ({build}), "
                    f"SAM 2.1 {'base' if large else 'small'}, YOLO-World {'large' if large else 'medium'}"
                ),
                warnings=warnings,
                gpu_index=gpu.index,
            )

    if hw.apple_silicon:
        return Plan(
            device="mps", torch_index=None, sam="sam2.1_s.pt", world="yolov8m-worldv2.pt",
            download_gb=1.1,
            summary="Apple silicon: PyTorch with Metal, SAM 2.1 small, YOLO-World medium",
            warnings=warnings,
        )

    return Plan(
        device="cpu",
        torch_index="https://download.pytorch.org/whl/cpu",
        sam="sam2.1_t.pt",
        world="yolov8s-worldv2.pt",
        download_gb=0.9,
        summary=(
            f"CPU, {hw.ram_gb:g} GB RAM: PyTorch for CPU, SAM 2.1 tiny, YOLO-World small "
            "-- about 0.2 s per click once an image is encoded"
        ),
        warnings=warnings + (
            [f"{hw.ram_gb:g} GB RAM is tight for the models; close other apps while using them."]
            if hw.ram_gb and hw.ram_gb < 8 else []
        ),
    )


def _cuda_build_for(driver: str, os_name: str) -> str | None:
    # As a tuple, never a float: 525.105 is newer than 525.60, but as decimal
    # numbers it is smaller -- and a 12 GB card was sent to the CPU.
    version = _version(driver)
    if not version:
        return None
    column = 0 if os_name == "Windows" else 1
    for build, minimums, _packages in CUDA_BUILDS:
        if version >= _version(minimums[column]):
            return build
    return None


def _packages_for(build: str) -> tuple[str, ...]:
    return next((packages for name, _minimums, packages in CUDA_BUILDS if name == build), ())


def _version(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in text.strip().split("."))
    except ValueError:
        return ()


# ---------------------------------------------------------------- probing


def _nvidia_gpus() -> list[Gpu]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        listing = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        header = subprocess.run([smi], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return []
    match = re.search(r"CUDA Version:\s*([\d.]+)", header.stdout or "")
    cuda = match.group(1) if match else ""
    gpus = []
    # nvidia-smi lists in CUDA index order (both default to PCI bus order).
    for index, line in enumerate((listing.stdout or "").splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            vram = round(float(parts[1]) / 1024, 1)
        except ValueError:
            continue
        gpus.append(Gpu(name=parts[0], vram_gb=vram, driver=parts[2], cuda=cuda, index=index))
    return gpus


def _ram_bytes() -> int:
    try:
        if sys.platform == "win32":
            class Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys)
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip())
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0
