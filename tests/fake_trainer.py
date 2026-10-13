"""Stands in for easyannotate_tools.train in the test suite: no torch.

Checks it was handed a real training set, then behaves like the trainer --
epoch lines, a best.pt, a result line.
"""

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    for flag in ("--data", "--model", "--epochs", "--imgsz", "--out", "--name"):
        parser.add_argument(flag)
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(spec["path"])
    images = sorted((root / spec["train"]).glob("*.*"))
    labels = sorted((root / spec["train"].replace("images", "labels", 1)).glob("*.txt"))
    print(f"FAKE saw {len(images)} images, {len(labels)} labels, names {spec['names']}", flush=True)
    for epoch in range(int(args.epochs)):
        print(f"EASYANNOTATE_EPOCH {epoch + 1}/{args.epochs}  mAP50 0.5", flush=True)
    best = Path(args.out) / args.name / "weights" / "best.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_bytes(b"not really a model")
    print("EASYANNOTATE_RESULT " + json.dumps({
        "best": str(best),
        "metrics": {"metrics/mAP50(B)": 0.5, "images": len(images), "names": spec["names"]},
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
