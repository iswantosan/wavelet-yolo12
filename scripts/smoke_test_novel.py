"""Smoke test for novel YOLOv12 variants: Strip, HS-FPN, DSConv.

Builds each model from YAML at scale 's', runs a dummy forward at 640x640,
and prints param count + output shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel

VARIANTS = [
    ("Strip",    "ultralytics/cfg/models/v12/yolov12s-strip.yaml"),
    ("HS-FPN",   "ultralytics/cfg/models/v12/yolov12s-hsfpn.yaml"),
    ("DSConv",   "ultralytics/cfg/models/v12/yolov12s-dsconv.yaml"),
]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"torch : {torch.__version__}")
    print()

    x = torch.randn(1, 3, 640, 640, device=device)
    failed = []

    for name, cfg in VARIANTS:
        print(f"=== {name} ({cfg}) ===")
        try:
            model = DetectionModel(cfg, nc=1, ch=3, verbose=False).to(device).eval()
            n_params = sum(p.numel() for p in model.parameters())
            with torch.no_grad():
                y = model(x)
            shapes = [tuple(t.shape) for t in (y if isinstance(y, list) else [y])]
            print(f"  params   : {n_params/1e6:.2f}M")
            print(f"  outputs  : {shapes}")
            print(f"  status   : OK")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"  status   : FAILED — {type(e).__name__}: {e}")
        print()

    if failed:
        print("FAILED variants:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    print("All variants built and ran a forward pass successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
