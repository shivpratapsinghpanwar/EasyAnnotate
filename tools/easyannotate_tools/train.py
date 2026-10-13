"""Train a detector on an exported project. Run by ``easyannotate train``.

Runs in the tools environment as its own process, so torch never enters the
editor's, and a training run that dies takes nothing else with it. Speaks to
its parent in two kinds of line among Ultralytics' own output:

    EASYANNOTATE_EPOCH 3/30  mAP50 0.412  box_loss 1.23
    EASYANNOTATE_RESULT {"best": "...", "metrics": {...}}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="easyannotate_tools.train")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    os.environ.setdefault("YOLO_VERBOSE", "False")
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)

    def on_epoch(trainer) -> None:
        # A progress line must never be able to kill a training run, and
        # Ultralytics changes these internals between releases (tloss went
        # from a tensor to a dict), so every field is read defensively.
        epoch, epochs = getattr(trainer, "epoch", 0) + 1, getattr(trainer, "epochs", 0)
        if epochs and epoch > epochs:
            return  # the final validation of the best weights fires this again
        parts = [f"{epoch}/{epochs or '?'}"]
        try:
            mAP = (trainer.metrics or {}).get("metrics/mAP50(B)")
            if mAP is not None:
                parts.append(f"mAP50 {float(mAP):.3f}")
        except Exception:
            pass
        try:
            loss = trainer.tloss
            if isinstance(loss, dict):
                loss = sum(float(v) for v in loss.values())
            elif hasattr(loss, "sum"):
                loss = float(loss.sum())
            if loss is not None:
                parts.append(f"loss {float(loss):.3f}")
        except Exception:
            pass
        print("EASYANNOTATE_EPOCH " + "  ".join(parts), flush=True)

    model.add_callback("on_fit_epoch_end", on_epoch)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.out,
        name=args.name,
        exist_ok=True,
        device=device,
        # Windows cannot fork; dataloader workers there cost more than they buy
        # on the small sets a first model is trained on.
        workers=0 if sys.platform == "win32" else 2,
        plots=False,
        verbose=False,
    )
    best = Path(model.trainer.best)
    final = {k: round(float(v), 4) for k, v in (model.trainer.metrics or {}).items()}
    print("EASYANNOTATE_RESULT " + json.dumps({"best": str(best), "metrics": final}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
