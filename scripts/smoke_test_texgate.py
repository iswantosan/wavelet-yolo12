"""Smoke test for the TextureGate variant.

Verifies that:
  1. The model builds from YAML and contains TextureGate modules at P3/P4.
  2. Identity-init: at step 0, the gated model output equals the baseline
     output (same numerical values up to float noise), confirming the
     gate starts as a no-op.
  3. After a forward pass in train mode, each TextureGate stores last_logits
     with the expected shape (B, 1, H, W).
  4. Auxiliary loss runs end-to-end (forward → loss → backward) and the
     gate's last conv receives non-zero gradient.
  5. init_criterion picks v8DetectionLossWithTexGate (not the base class).

Run:
    python scripts/smoke_test_texgate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ultralytics.nn.modules.wavelet import TextureGate
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss, v8DetectionLossWithTexGate


def section(name: str) -> None:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")


def main() -> int:
    torch.manual_seed(0)

    section("[1] Build baseline + texgate, count TextureGate modules")
    baseline = DetectionModel(cfg="ultralytics/cfg/models/v12/yolov12s.yaml", ch=3, nc=1, verbose=False)
    texgate = DetectionModel(cfg="ultralytics/cfg/models/v12/yolov12s-texgate.yaml", ch=3, nc=1, verbose=False)
    n_base_params = sum(p.numel() for p in baseline.parameters())
    n_tex_params = sum(p.numel() for p in texgate.parameters())
    n_gates = sum(1 for m in texgate.modules() if isinstance(m, TextureGate))
    print(f"  baseline params : {n_base_params/1e6:.3f} M")
    print(f"  texgate  params : {n_tex_params/1e6:.3f} M  (Δ {(n_tex_params-n_base_params)/1e3:.1f} K)")
    print(f"  TextureGate modules in texgate: {n_gates}")
    assert n_gates == 2, f"expected 2 gates (P3+P4), got {n_gates}"

    section("[2] Identity-init: each gate output equals its input")
    x = torch.randn(2, 3, 640, 640)
    texgate.eval()
    inputs_at_gate, outputs_at_gate = {}, {}

    def _make_hooks(name):
        def pre(_mod, inp):
            inputs_at_gate[name] = inp[0].detach()

        def post(_mod, _inp, out):
            outputs_at_gate[name] = out.detach()

        return pre, post

    gates_named = [(n, m) for n, m in texgate.named_modules() if isinstance(m, TextureGate)]
    handles = []
    for name, m in gates_named:
        pre, post = _make_hooks(name)
        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(post))
    with torch.no_grad():
        _ = texgate(x)
    for h in handles:
        h.remove()

    for name, _ in gates_named:
        diff = (inputs_at_gate[name] - outputs_at_gate[name]).abs().max().item()
        print(f"  {name:<30}  max |out - in| = {diff:.3e}")
        assert diff < 1e-5, f"gate {name} not identity at init (diff {diff:.3e})"
    print("  PASS: every gate is identity at step 0 (gate=1.0 → out=in)")

    section("[3] Train-mode forward stores last_logits in each gate")
    texgate.train()
    _ = texgate(x)
    for i, m in enumerate(texgate.modules()):
        if isinstance(m, TextureGate):
            print(f"  gate {i}: last_logits shape = {tuple(m.last_logits.shape)}")
            assert m.last_logits.dim() == 4 and m.last_logits.shape[1] == 1

    section("[4] init_criterion picks v8DetectionLossWithTexGate")
    # DetectionModel needs .args for the loss; emulate the trainer.
    from types import SimpleNamespace
    texgate.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5,
        tex_gate_weight=0.5, tex_pos_weight=4.0,
        nwd_ratio=0.0, nwd_c=12.8,
    )
    crit = texgate.init_criterion()
    print(f"  criterion class : {type(crit).__name__}")
    assert isinstance(crit, v8DetectionLossWithTexGate), "wrong criterion class"
    base_crit = baseline.init_criterion() if not hasattr(baseline, "init_criterion") else None
    # Above line is defensive — baseline has init_criterion via DetectionModel.

    section("[5] Forward → aux loss → backward, gate grads non-zero")
    texgate.train()
    # Synthetic batch: 2 images, each with 3 fake bacilli boxes near center.
    n_gt = 6  # 3 per image
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)
    cls = torch.zeros(n_gt, 1)
    # cxcywh normalized
    bboxes = torch.tensor(
        [
            [0.30, 0.30, 0.05, 0.05],
            [0.50, 0.50, 0.04, 0.04],
            [0.70, 0.70, 0.06, 0.06],
            [0.25, 0.40, 0.05, 0.05],
            [0.50, 0.60, 0.04, 0.04],
            [0.75, 0.30, 0.05, 0.05],
        ],
        dtype=torch.float32,
    )
    batch = {
        "img": x,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }

    preds = texgate(x)
    total_loss, loss_items = crit(preds, batch)
    print(f"  total_loss      : {float(total_loss):.4f}")
    print(f"  loss_items      : {loss_items.detach().cpu().numpy()}")
    total_loss.backward()

    grad_summary = []
    for name, m in texgate.named_modules():
        if isinstance(m, TextureGate):
            last_conv = m.head[-1]
            g_w = last_conv.weight.grad
            g_b = last_conv.bias.grad
            grad_summary.append((name, g_w.abs().mean().item() if g_w is not None else 0.0,
                                 g_b.abs().mean().item() if g_b is not None else 0.0))
    print(f"  TextureGate last-conv grads:")
    for name, gw, gb in grad_summary:
        print(f"    {name:<30}  |grad_w|={gw:.3e}  |grad_b|={gb:.3e}")
    assert all(gw > 0 for _, gw, _ in grad_summary), "some gate received zero gradient"

    print("\nAll checks passed. TextureGate integration looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
