"""Smoke test: build Wavelet-YOLOv12 and run a dummy forward pass.

Run: python scripts/smoke_test_wavelet.py
"""
import sys
from pathlib import Path

# Ensure local ultralytics is used.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.nn.tasks import DetectionModel


def build_and_probe(cfg: str, scale: str = "n"):
    print(f"\n=== {cfg} (scale={scale}) ===")
    model = DetectionModel(cfg=cfg, ch=3, nc=2, verbose=False)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model(x)

    # DetectionModel returns list of tensors at training, single tensor at eval.
    if isinstance(y, (list, tuple)):
        shapes = [tuple(t.shape) for t in y if torch.is_tensor(t)]
    else:
        shapes = tuple(y.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params : {n_params/1e6:.2f} M")
    print(f"  output : {shapes}")


if __name__ == "__main__":
    build_and_probe("ultralytics/cfg/models/v12/yolov12s.yaml")
    build_and_probe("ultralytics/cfg/models/v12/yolov12s-wavelet-p3.yaml")
    build_and_probe("ultralytics/cfg/models/v12/yolov12s-wavelet.yaml")
    print("\nOK")
