"""Getting raw frames into the factory: from video files, or from a camera.

The webcam grabber existed only as a three-line unfinished stub; it is
implemented here properly, with the same result contract as everything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import fs
from ..registry import Param, operation
from ..types import Action, Context, OpResult

FRAME_PARAMS = (
    Param("source", "path", "A video file, or a folder of videos.", required=True, must_exist=True),
    Param("output", "dir", "Folder to write frames into.", required=True),
    Param("every_nth", "int", "Keep one frame out of every N.", default=1, minimum=1),
    Param("max_frames", "int", "Stop after this many saved frames (0 = no limit).",
          default=0, minimum=0),
    Param("prefix", "str", "Filename prefix for saved frames.", default="frame"),
    Param("pad", "int", "Zero-padding width for the frame number.", default=6, minimum=1, maximum=12),
    Param("quality", "int", "JPEG quality.", default=95, minimum=1, maximum=100),
    Param("resize_width", "int", "Downscale to this width, keeping aspect (0 = keep original).",
          default=0, minimum=0),
    Param("start_index", "int", "First frame number.", default=0, minimum=0),
)


@operation(
    "collect.frames",
    title="Extract frames from video",
    category="collect",
    summary="Turn one video or a whole folder of them into numbered JPEG frames.",
    params=FRAME_PARAMS,
    notes=(
        "every_nth is the knob that matters. Consecutive frames of a video are near\n"
        "identical, so extracting all of them mostly manufactures duplicates for\n"
        "clean.dedup to find again. At 30fps, every_nth=15 gives two frames a second."
    ),
)
def frames(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="collect.frames", dry_run=ctx.dry_run)
    cv2 = _load_cv2()

    source: Path = params["source"]
    output: Path = params["output"]
    videos = (
        [source]
        if source.is_file()
        else list(fs.iter_files(source, recursive=False, exts=fs.VIDEO_EXTS))
    )
    if not videos:
        result.fail(f"no video files found at {source}")
        return result

    fs.ensure_dir(output, ctx)
    saved = int(params["start_index"])
    limit = int(params["max_frames"])
    every = int(params["every_nth"])
    pad = int(params["pad"])

    for video_path in videos:
        ctx.check_cancelled()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            result.warnings.append(f"cannot open {video_path.name}")
            result.bump("videos_failed")
            continue

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        read = 0
        try:
            while True:
                ctx.check_cancelled()
                ok, frame = capture.read()
                if not ok:
                    break
                if read % every == 0:
                    if params["resize_width"]:
                        frame = _resize_to_width(cv2, frame, int(params["resize_width"]))
                    destination = output / f"{params['prefix']}_{saved:0{pad}d}.jpg"
                    if not ctx.dry_run:
                        cv2.imwrite(
                            str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, int(params["quality"])]
                        )
                    result.record(Action.WRITE, video_path, destination, f"frame {read}")
                    saved += 1
                    result.bump("frames_saved")
                    if limit and result.counts.get("frames_saved", 0) >= limit:
                        break
                read += 1
                if total:
                    ctx.emit(read, total, video_path.name)
        finally:
            capture.release()

        result.bump("videos_processed")
        result.counts[f"frames_read:{video_path.name}"] = read
        if limit and result.counts.get("frames_saved", 0) >= limit:
            result.warnings.append(f"stopped at the {limit}-frame limit")
            break

    result.data["output"] = output
    return result


WEBCAM_PARAMS = (
    Param("output", "dir", "Folder to write captures into.", required=True),
    Param("camera", "int", "Camera index.", default=0, minimum=0),
    Param("count", "int", "How many images to capture (0 = until you quit).",
          default=0, minimum=0),
    Param("interval", "float", "Seconds between automatic captures (0 = manual, press SPACE).",
          default=0.0, minimum=0.0),
    Param("prefix", "str", "Filename prefix.", default="cam"),
    Param("pad", "int", "Zero-padding width.", default=5, minimum=1, maximum=12),
    Param("preview", "bool", "Show a live preview window.", default=True),
)


@operation(
    "collect.webcam",
    title="Capture images from a camera",
    category="collect",
    summary="Grab stills from a connected camera, manually or on a timer.",
    params=WEBCAM_PARAMS,
    notes=(
        "With preview on: SPACE captures, Q or ESC quits. Set interval to capture\n"
        "automatically instead. preview=False runs headless, which is what you want\n"
        "over SSH or from a web backend -- pair it with count and interval."
    ),
)
def webcam(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="collect.webcam", dry_run=ctx.dry_run)
    cv2 = _load_cv2()
    import time

    output: Path = params["output"]
    fs.ensure_dir(output, ctx)

    capture = cv2.VideoCapture(int(params["camera"]))
    if not capture.isOpened():
        result.fail(f"cannot open camera {params['camera']}")
        return result

    target = int(params["count"])
    interval = float(params["interval"])
    preview = bool(params["preview"]) and not ctx.dry_run
    pad = int(params["pad"])
    saved = 0
    last_capture = 0.0
    window = "Data Factory - capture"

    if preview and not _has_gui(cv2):
        # The default install is opencv-python-headless, which has no window
        # support at all. Falling back beats an unhandled cv2.error, and the
        # timer path does the same job without a screen.
        preview = False
        result.warnings.append(
            "no GUI support in this OpenCV build, so the preview window was skipped. "
            "Install the 'gui' extra for a live preview, or pass --interval to "
            "capture on a timer."
        )
        if interval <= 0:
            interval = 1.0

    try:
        if preview:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            ctx.check_cancelled()
            ok, frame = capture.read()
            if not ok:
                result.warnings.append("camera stopped returning frames")
                break

            now = time.monotonic()
            take = False
            if interval > 0 and now - last_capture >= interval:
                take = True
            if preview:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    take = True
            elif interval <= 0:
                # Headless with no timer would spin forever; one shot and out.
                take = True

            if take:
                destination = output / f"{params['prefix']}_{saved:0{pad}d}.jpg"
                if not ctx.dry_run:
                    cv2.imwrite(str(destination), frame)
                result.record(Action.WRITE, Path(f"camera:{params['camera']}"), destination, "capture")
                saved += 1
                last_capture = now
                result.bump("captured")
                ctx.emit(saved, target or saved, f"captured {destination.name}")
                if preview is False and interval <= 0:
                    break
            if target and saved >= target:
                break
    finally:
        capture.release()
        if preview:
            cv2.destroyAllWindows()

    result.data["output"] = output
    return result


def _has_gui(cv2: Any) -> bool:
    """Whether this OpenCV build can open a window.

    The headless wheel ships the same API surface but raises at call time, so
    the only reliable check is the build information string.
    """
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    for backend in ("GTK", "QT", "WIN32 UI", "COCOA"):
        if f"{backend}:" in info and "YES" in info.split(f"{backend}:")[1][:40]:
            return True
    return False


def _resize_to_width(cv2: Any, frame: Any, width: int) -> Any:
    height = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, max(1, height)), interpolation=cv2.INTER_AREA)


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"this operation needs opencv-python installed ({exc})") from exc
    return cv2
