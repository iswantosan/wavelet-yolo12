"""Quick isolated debug for HaarDWT / WaveDown — no training, no dataset.

Run:
    python scripts/debug_wavelet.py

Checks:
1. HaarDWT output shape + ordering (LL/LH/HL/HH)
2. Energy preservation (orthonormal Haar property)
3. Reconstruction sanity (inverse Haar)
4. WaveDown forward shape + finite + variance
5. Gradient flow (no zero/NaN grads through learnable parts)
6. Full Wavelet-YOLOv12 build + forward + backward
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.wavelet import HaarDWT, WaveDown

torch.manual_seed(0)


def section(name: str) -> None:
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


# ---------------------------------------------------------------------------
section("1. HaarDWT output shape & ordering")
C, H, W = 3, 8, 8
x = torch.randn(2, C, H, W)
dwt = HaarDWT(C)
y = dwt(x)
print(f"  input  : {tuple(x.shape)}")
print(f"  output : {tuple(y.shape)}  (expected: 2, 4, {C}, {H//2}, {W//2})")
assert y.shape == (2, 4, C, H // 2, W // 2), "DWT output shape mismatch"

# Manual check on a tiny known patch
xs = torch.zeros(1, 1, 2, 2)
xs[0, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ys = HaarDWT(1)(xs)
# Orthonormal Haar 2D:
#   LL = (1+2+3+4)/2 = 5.0
#   LH = (1+2-3-4)/2 = -2.0
#   HL = (1-2+3-4)/2 = -1.0
#   HH = (1-2-3+4)/2 = 0.0
ll, lh, hl, hh = ys[0, 0, 0, 0, 0], ys[0, 1, 0, 0, 0], ys[0, 2, 0, 0, 0], ys[0, 3, 0, 0, 0]
print(f"  LL={ll.item():+.2f}  LH={lh.item():+.2f}  HL={hl.item():+.2f}  HH={hh.item():+.2f}")
print(f"  expect : LL=+5.00  LH=-2.00  HL=-1.00  HH=+0.00")
assert abs(ll.item() - 5.0) < 1e-5, "LL wrong"
assert abs(lh.item() + 2.0) < 1e-5, "LH wrong"
assert abs(hl.item() + 1.0) < 1e-5, "HL wrong"
assert abs(hh.item() - 0.0) < 1e-5, "HH wrong"
print("  OK: Haar basis values match orthonormal formula")


# ---------------------------------------------------------------------------
section("2. Energy preservation (orthonormal Haar)")
x = torch.randn(4, 16, 32, 32)
y = HaarDWT(16)(x)
e_in = (x ** 2).sum().item()
e_out = (y ** 2).sum().item()
print(f"  input  energy : {e_in:.4f}")
print(f"  output energy : {e_out:.4f}")
print(f"  ratio         : {e_out/e_in:.6f}  (orthonormal Haar -> 1.0)")
assert abs(e_out / e_in - 1.0) < 1e-4, "energy not preserved -> not orthonormal"
print("  OK: energy preserved -> DWT is correctly orthonormal")


# ---------------------------------------------------------------------------
section("3. Reconstruction (inverse Haar)")
# Inverse Haar 2D using the same kernels (orthonormal -> inverse = transpose conv)
def inverse_haar(y: torch.Tensor) -> torch.Tensor:
    # y: (B, 4, C, H, W) -> (B, C, 2H, 2W)
    b, _, c, h, w = y.shape
    # Re-pack to (B, 4C, H, W) in interleaved order (same as conv output)
    yi = y.permute(0, 2, 1, 3, 4).contiguous().view(b, 4 * c, h, w)
    # Build conv_transpose weight using same kernels per channel
    from ultralytics.nn.modules.wavelet import _haar_kernels
    k = _haar_kernels()                       # (4, 2, 2)
    w_t = k.repeat(c, 1, 1).unsqueeze(1)      # (4C, 1, 2, 2)
    return F.conv_transpose2d(yi, w_t, stride=2, groups=c)


x = torch.randn(2, 4, 16, 16)
y = HaarDWT(4)(x)
x_rec = inverse_haar(y)
err = (x - x_rec).abs().max().item()
print(f"  max |x - x_reconstructed| = {err:.2e}  (expect ~1e-6)")
assert err < 1e-4, "Reconstruction error too large -> DWT/IDWT not consistent"
print("  OK: DWT is invertible (information preserving)")


# ---------------------------------------------------------------------------
section("4. WaveDown forward")
for c1, c2 in [(256, 256), (512, 512), (1024, 1024)]:
    m = WaveDown(c1, c2)
    x = torch.randn(2, c1, 32, 32)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  WaveDown({c1:4d}->{c2:4d}) : x{tuple(x.shape)} -> y{tuple(y.shape)}  "
          f"params={n_params/1e3:.1f}K  mean={y.mean().item():+.3f}  std={y.std().item():.3f}")
    assert y.shape == (2, c2, 16, 16), "WaveDown output shape wrong"
    assert torch.isfinite(y).all(), "WaveDown produced non-finite values"


# ---------------------------------------------------------------------------
section("5. Gradient flow")
m = WaveDown(64, 64)
x = torch.randn(2, 64, 32, 32, requires_grad=True)
y = m(x)
loss = y.mean()
loss.backward()

# Check all learnable params have non-zero, finite gradients
for name, p in m.named_parameters():
    g = p.grad
    assert g is not None, f"No grad for {name}"
    g_abs = g.abs()
    nz = (g_abs > 0).float().mean().item()
    print(f"  {name:30s}  grad_mean={g_abs.mean().item():.3e}  "
          f"grad_max={g_abs.max().item():.3e}  nonzero_frac={nz:.2f}")
    assert torch.isfinite(g).all(), f"Non-finite grad in {name}"
    assert g_abs.max().item() > 0, f"All-zero grad in {name}"
print("  OK: gradients flow through proj_ll, proj_hf, fuse")


# ---------------------------------------------------------------------------
section("6. Full Wavelet-YOLOv12 model — build / forward / backward")
from ultralytics.nn.tasks import DetectionModel
for cfg in [
    "ultralytics/cfg/models/v12/yolov12.yaml",
    "ultralytics/cfg/models/v12/yolov12-wavelet-p3.yaml",
    "ultralytics/cfg/models/v12/yolov12-wavelet.yaml",
]:
    name = Path(cfg).stem
    model = DetectionModel(cfg=cfg, ch=3, nc=1, verbose=False)
    n_params = sum(p.numel() for p in model.parameters())
    model.train()
    x = torch.randn(2, 3, 640, 640)
    y = model(x)
    # In train mode, model returns list of P3/P4/P5 raw outputs
    loss = sum(yi.float().mean() for yi in y if torch.is_tensor(yi))
    loss.backward()
    n_zero_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0)
    print(f"  {name:30s}  params={n_params/1e6:.2f}M  "
          f"out_shapes={[tuple(yi.shape) for yi in y if torch.is_tensor(yi)]}")
    print(f"  {'':30s}  loss={loss.item():+.3e}  zero-grad-params={n_zero_grad}")


print("\nAll checks passed. WaveDown / HaarDWT are mathematically correct.\n"
      "If training mAP=0 persists past epoch ~10, the issue is in YAML routing or\n"
      "channel mismatch with the rest of YOLOv12, not in WaveDown internals.\n")
