"""A tool that does nothing useful, on purpose.

Two jobs:

* **It is the test fixture for the protocol.** Every behaviour the real SAM
  sidecar must have -- slow work, cold state, crashes, exceptions, silence --
  is reachable here without installing PyTorch, so the boundary is verified in
  a second rather than a minute and in CI rather than only on a GPU box.
* **It is the reference implementation.** A tool author copies this file,
  replaces ``invoke``, and gets framing, correlation and error replies for
  free.

Run it as ``python -m data_factory.annotate.tools.fake``.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .protocol import ToolError, ToolNotReady, ToolSpec
from .serve import serve

#: Milliseconds of pretend work, so a test can assert on timeouts.
WORK_MS = int(os.environ.get("DFX_FAKE_WORK_MS", "20"))

SPECS = [
    ToolSpec(
        name="fake.box",
        title="Fake box tool",
        summary="Returns a fixed shape. Exists to prove the sidecar boundary.",
        kind="interactive",
        inputs={
            "type": "object",
            "properties": {"box": {"type": "array", "items": {"type": "number"}}},
            "required": ["box"],
        },
        version="1",
        requires=("prepare",),
    ),
]


class FakeTool:
    def __init__(self) -> None:
        self.prepared: set[str] = set()

    def describe(self) -> list[dict[str, Any]]:
        return [spec.as_dict() for spec in SPECS]

    def health(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"state": "ready", "detail": f"fake tool, {WORK_MS}ms per call"}

    def prepare(self, params: dict[str, Any]) -> None:
        time.sleep(WORK_MS / 1000.0)
        self.prepared.add(str(params.get("image_id") or ""))

    def invoke(self, params: dict[str, Any]) -> dict[str, Any]:
        image_id = str(params.get("image_id") or "")

        # Behaviours a test can ask for by name, each mirroring a real failure
        # mode of a GPU-backed tool.
        behaviour = str(params.get("behaviour") or "")
        if behaviour == "raise":
            raise RuntimeError("the tool blew up")
        if behaviour == "exit":
            # os._exit, not sys.exit: serve() catches BaseException and always
            # replies, so a SystemExit would be *handled* rather than fatal.
            # A real segfault or CUDA OOM does not unwind, and this is the only
            # honest way to simulate one.
            os._exit(9)
        if behaviour == "hang":
            time.sleep(3600)                  # never replies; the caller must time out
        if behaviour == "silent":
            raise SystemExit                  # a BaseException, not an Exception
        if behaviour == "unknown_tool":
            raise ToolError("no such tool", code="unknown_tool")

        if image_id and image_id not in self.prepared:
            # Exactly how the real SAM tool behaves on a cold image: not an
            # error, just not yet -- the UI retries rather than showing a
            # failure the user can do nothing about.
            raise ToolNotReady(f"{image_id} has not been prepared", retry_after_ms=250)

        time.sleep(WORK_MS / 1000.0)
        box = params.get("box") or [0, 0, 10, 10]
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        return {
            "shapes": [
                {
                    "label": str(params.get("label") or "object"),
                    "bbox": [x1, y1, x2, y2],
                    "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "confidence": 0.99,
                }
            ],
            "ms": WORK_MS,
        }


def main() -> int:
    return serve(FakeTool())


if __name__ == "__main__":
    raise SystemExit(main())
