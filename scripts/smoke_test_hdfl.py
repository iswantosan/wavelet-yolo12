"""Smoke test: build HDFL / AuxSeg / P3P4 model variants and run dummy forward.

Run: python scripts/smoke_test_hdfl.py
"""
import sys
from pathlib import Path

# Use repo-local ultralytics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.nn.tasks import DetectionModel


def build_and_probe(cfg: str, scale: str = "s"):
    print(f"\n=== {cfg} (scale={scale}) ===")
    model = DetectionModel(cfg=cfg, ch=3, nc=2, verbose=False)
    model.eval()

    head = model.model[-1]
    print(f"  head class       : {type(head).__name__}")
    print(f"  reg_max          : {head.reg_max}")
    if hasattr(head, "fine_max"):
        print(f"  fine_max         : {head.fine_max}")
        print(f"  reg_total (=eff) : {head.reg_total}")
    print(f"  no  (per anchor) : {head.no}")
    print(f"  nl  (#scales)    : {head.nl}")
    print(f"  stride           : {head.stride.tolist()}")

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model(x)

    if isinstance(y, (list, tuple)):
        shapes = [tuple(t.shape) for t in y if torch.is_tensor(t)]
        print(f"  output shapes    : {shapes}")
    else:
        print(f"  output shape     : {tuple(y.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params total     : {n_params/1e6:.3f} M")
    print(f"  params trainable : {n_train/1e6:.3f} M")


def _check_loss_compat(cfg: str):
    """Quick test: build with task='detect' + train mode, run forward + a fake loss step."""
    print(f"  loss-compat check ...")
    model = DetectionModel(cfg=cfg, ch=3, nc=2, verbose=False)
    model.train()
    head = model.model[-1]

    # Build dummy batch with batch_idx / cls / bboxes (xywh normalized).
    imgsz = 640
    bs = 2
    x = torch.randn(bs, 3, imgsz, imgsz)
    feats = model(x)  # train mode returns list of feature maps

    # Construct dummy targets: 3 boxes total across batch. Include
    # batch["img"] because AuxSeg loss needs the image tensor for
    # the LAB a*-channel pseudo-mask.
    batch = {
        "img": x.clamp(0, 1),  # normalised to [0,1] for LAB a* approx
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0], [0.0]]),
        "bboxes": torch.tensor([
            [0.3, 0.3, 0.1, 0.1],
            [0.6, 0.6, 0.15, 0.2],
            [0.5, 0.5, 0.2, 0.2],
        ]),
    }

    from ultralytics.utils.loss import v8DetectionLoss
    # Need model.args (hyperparameters) — fake minimally.
    class _H:
        box = 7.5
        cls = 0.5
        dfl = 1.5
        # Auxiliary hyperparameters used by AuxSeg / Contrastive heads.
        auxseg_weight = 1.0
        contrast_weight = 0.3
        contrast_temp = 0.1
        contrast_n_neg = 16
        def get(self, k, default=None): return getattr(self, k, default)
    model.args = _H()

    crit = v8DetectionLoss(model)
    print(f"    crit.reg_max={crit.reg_max} fine_max={crit.fine_max} reg_total={crit.reg_total} no={crit.no}")

    loss, items = crit(feats, batch)
    print(f"    loss = {loss.item():.4f}   items (box, cls, dfl) = {items.tolist()}")
    loss.backward()
    print(f"    backward OK")


if __name__ == "__main__":
    CFGS = (
        "ultralytics/cfg/models/v12/yolov12s.yaml",
        "ultralytics/cfg/models/v12/yolov12s-p3p4.yaml",
        "ultralytics/cfg/models/v12/yolov12s-hdfl.yaml",
        "ultralytics/cfg/models/v12/yolov12s-hdfl-p3p4.yaml",
        "ultralytics/cfg/models/v12/yolov12s-auxseg.yaml",
        "ultralytics/cfg/models/v12/yolov12s-auxseg-p3p4.yaml",
        "ultralytics/cfg/models/v12/yolov12s-contrast.yaml",
        "ultralytics/cfg/models/v12/yolov12s-contrast-auxseg.yaml",
    )

    # Tier 1: structural build + forward (eval mode)
    for cfg in CFGS:
        build_and_probe(cfg)

    # Tier 2: loss compatibility (train mode + backward)
    print("\n=== LOSS COMPATIBILITY ===")
    for cfg in CFGS:
        print(f"\n--- {cfg} ---")
        try:
            _check_loss_compat(cfg)
        except Exception as e:
            print(f"    FAIL: {type(e).__name__}: {e}")

    print("\nOK")
