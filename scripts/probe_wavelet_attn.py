"""Probe a trained Wavelet-YOLOv12 checkpoint to see whether the wavelet
attention branch is contributing anything or quietly dead.

Three checks:
  [1] Branch parameters
      v1: learned alpha  (≈0 → branch dead, zero-grad-trap confirmed)
      v2: final hf-branch weight magnitude
      v3: gate weight/bias  (≈0 → gate stuck at 1.0 = identity)
  [2] Residual / main-path magnitude ratio (on real images)
      |out − y_main| / |y_main|. <0.5% = dead, <5% = tiny, >50% = dominant.
  [3] Pretrained-transfer sanity
      Build a fresh wavelet model, load yolov12s.pt by name, check that the
      main-path stride-2 conv looks like a trained kernel (not Kaiming).

Colab usage:
    !python scripts/probe_wavelet_attn.py \\
        --ckpt /content/runs/wavelet_chen/<run>/weights/best.pt \\
        --data /content/chen_split/data.yaml \\
        --pretrained /content/yolov12s.pt \\
        --n-images 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from ultralytics.nn.modules.wavelet import (
    WaveAttnDown,
    WaveAttnDownV2,
    WaveAttnDownV3,
)

WAVE_ATTN_CLASSES = (WaveAttnDown, WaveAttnDownV2, WaveAttnDownV3)


def hr(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def find_wave_attn_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    out = []
    for name, m in model.named_modules():
        if isinstance(m, WAVE_ATTN_CLASSES):
            out.append((name, m))
    return out


# ---------------------------------------------------------------------------
# [1] Branch parameters
# ---------------------------------------------------------------------------

def parameter_report(name: str, m: torch.nn.Module) -> None:
    cls = type(m).__name__
    print(f"\n  [{name}]  ({cls})")

    if isinstance(m, WaveAttnDown):
        a = float(m.alpha.detach().cpu())
        hf_w_abs = m.hf_proj.weight.detach().abs().mean().item()
        print(f"    learned alpha            : {a:+.5f}")
        print(f"    hf_proj  |W| mean        : {hf_w_abs:.3e}")
        if abs(a) < 1e-3:
            print("    >>> alpha ≈ 0 — residual = alpha * attn * y_main ≈ 0 → branch DEAD")
        elif abs(a) < 0.05:
            print("    >>> alpha small (<0.05) — branch contributing very weakly")
        else:
            print("    >>> alpha grew — branch is active, check [2] for magnitude")

    elif isinstance(m, WaveAttnDownV2):
        final = m.hf_branch[3]  # Conv2d(c_mid → c2)
        w_abs = final.weight.detach().abs().mean().item()
        w_std = final.weight.detach().std().item()
        c_in = final.in_channels
        # Init was default kaiming_uniform_(a=√5) * 0.05.
        # That gives std ≈ 0.05 · √(2/(3·c_in))
        init_std = 0.05 * (2.0 / (3 * c_in)) ** 0.5
        ratio = w_std / max(init_std, 1e-12)
        print(f"    hf_branch final |W| mean : {w_abs:.3e}")
        print(f"    hf_branch final  W  std  : {w_std:.3e}")
        print(f"    expected init std        : {init_std:.3e}")
        print(f"    std grew by              : {ratio:.2f}× since init")
        if ratio < 2:
            print("    >>> weights barely moved → branch likely suppressed (check [2])")

    elif isinstance(m, WaveAttnDownV3):
        gw = m.gate.weight.detach()
        gb = m.gate.bias.detach()
        print(f"    gate.weight              : mean={gw.mean().item():+.3e}  std={gw.std().item():.3e}")
        print(f"    gate.bias                : {gb.item():+.3e}")
        if gw.abs().max().item() < 1e-3 and abs(gb.item()) < 1e-3:
            print("    >>> gate ≈ 0 → 2·σ(0) = 1.0 → identity gate → branch DEAD")


# ---------------------------------------------------------------------------
# [2] Magnitude probe (residual vs main path on real inputs)
# ---------------------------------------------------------------------------

class BranchProbe:
    """Captures the input tensor at each WaveAttn module via forward_pre_hook,
    then decomposes the output into (main, residual) for analysis."""

    def __init__(self):
        self.x_in: Dict[str, torch.Tensor] = {}
        self.handles = []

    def attach(self, named_modules):
        for name, m in named_modules:
            def make(n):
                def hook(_mod, inp):
                    self.x_in[n] = inp[0].detach()
                return hook
            self.handles.append(m.register_forward_pre_hook(make(name)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def decompose(self, name: str, m: torch.nn.Module):
        x = self.x_in[name]
        with torch.no_grad():
            if isinstance(m, WaveAttnDown):
                y_main = m.act(m.bn(m.conv(x)))
                hf = m.dwt(x)[:, 1:].flatten(1, 2)
                attn = torch.sigmoid(m.attn_bn(m.hf_proj(hf)))
                out = y_main * (1 + m.alpha * attn)
            elif isinstance(m, WaveAttnDownV2):
                y_main = m.act(m.bn(m.conv(x)))
                hf = m.dwt(x)[:, 1:].flatten(1, 2)
                residual = m.hf_branch(hf)
                out = y_main + residual
            elif isinstance(m, WaveAttnDownV3):
                y_main = m.act(m.bn(m.conv(x)))
                hf = m.dwt(x)[:, 1:]
                energy = hf.pow(2).mean(dim=(1, 2)).unsqueeze(1)
                mu = energy.mean(dim=(2, 3), keepdim=True)
                sigma = energy.std(dim=(2, 3), keepdim=True) + 1e-6
                energy = (energy - mu) / sigma
                gate = 2 * torch.sigmoid(m.gate(energy))
                out = y_main * gate
            else:
                raise TypeError(type(m))
        residual = out - y_main
        return out, y_main, residual


def load_sample_images(data_yaml: Path, n: int, imgsz: int, device, split_pref="val"):
    from PIL import Image
    import torchvision.transforms.functional as TF

    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", data_yaml.parent))
    split = data.get(split_pref) or data.get("val") or data.get("test") or data.get("train")
    if isinstance(split, list):
        split = split[0]
    img_dir = base / split if (base / split).is_dir() else Path(split)
    img_paths = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    )[:n]
    if not img_paths:
        raise RuntimeError(f"No images found under {img_dir}")
    tensors = []
    for p in img_paths:
        img = Image.open(p).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
        tensors.append(TF.to_tensor(img))
    return torch.stack(tensors).to(device), img_paths


def magnitude_report(model, modules, n_images, imgsz, data_yaml, device) -> None:
    print("\n  Loading sample images...")
    x_batch, paths = load_sample_images(Path(data_yaml), n_images, imgsz, device)
    print(f"  using {len(paths)} images, batch shape = {tuple(x_batch.shape)}")

    probe = BranchProbe()
    probe.attach(modules)
    model.eval()
    with torch.no_grad():
        _ = model(x_batch)

    print()
    print(f"  {'module':<28}{'|y_main|':>12}{'|residual|':>14}{'ratio':>10}   verdict")
    print(f"  {'-' * 28}{'-' * 12}{'-' * 14}{'-' * 10}   {'-' * 12}")
    for name, m in modules:
        out, y_main, residual = probe.decompose(name, m)
        y_mag = y_main.abs().mean().item()
        r_mag = residual.abs().mean().item()
        ratio = r_mag / max(y_mag, 1e-12)
        if ratio < 0.005:
            verdict = "DEAD"
        elif ratio < 0.05:
            verdict = "tiny"
        elif ratio < 0.5:
            verdict = "active"
        else:
            verdict = "dominant"
        print(f"  {name:<28}{y_mag:>12.3e}{r_mag:>14.3e}{ratio:>10.4f}   {verdict}")
    probe.detach()


# ---------------------------------------------------------------------------
# [3] Pretrained-transfer check
# ---------------------------------------------------------------------------

def pretrained_transfer_check(cfg_used, pretrained_pt) -> None:
    """Build a fresh model from the same YAML, load pretrained yolov12s.pt by
    name, then inspect the main-path conv at each WaveAttn index. Random init
    would give std ≈ √(2 / fan_in); a successful transfer is usually smaller
    AND has a non-trivial absolute mean (pretrained kernels are not zero-mean
    in the way fresh init is)."""
    print(f"\n  YAML used         : {cfg_used}")
    print(f"  Pretrained source : {pretrained_pt}")
    fresh = YOLO(str(cfg_used))
    try:
        fresh.load(str(pretrained_pt))
    except Exception as e:
        print(f"  [warn] load failed: {e}")
        return

    sd = fresh.model.state_dict()
    targets = [n for n, m in fresh.model.named_modules() if isinstance(m, WAVE_ATTN_CLASSES)]
    if not targets:
        print("  No WaveAttn modules in fresh model — config mismatch?")
        return

    print()
    print(f"  {'module':<28}{'|W| mean':>12}{'W std':>12}{'kaiming std':>14}   verdict")
    print(f"  {'-' * 28}{'-' * 12}{'-' * 12}{'-' * 14}   {'-' * 20}")
    for name in targets:
        key = f"{name}.conv.weight"
        if key not in sd:
            print(f"  {name:<28}{'(missing key)':>40}")
            continue
        w = sd[key]
        m_abs = w.abs().mean().item()
        m_std = w.std().item()
        fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        kaiming = (2.0 / fan_in) ** 0.5
        # Random kaiming std lands near sqrt(2/fan_in); trained weights are
        # usually 3-10× smaller AND shifted (non-zero mean component).
        is_random_like = abs(m_std - kaiming) / kaiming < 0.15
        verdict = "looks RANDOM" if is_random_like else "looks loaded (OK)"
        print(f"  {name:<28}{m_abs:>12.3e}{m_std:>12.3e}{kaiming:>14.3e}   {verdict}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Trained .pt (best.pt or last.pt)")
    ap.add_argument("--data", required=True, help="data.yaml for sample images")
    ap.add_argument("--pretrained", default=None, help="yolov12s.pt to verify transfer")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-images", type=int, default=32)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    hr("Loading model")
    y = YOLO(args.ckpt)
    model = y.model.to(device).eval()
    cfg_used = (
        getattr(model, "yaml_file", None)
        or (y.ckpt.get("model_cfg") if isinstance(y.ckpt, dict) else None)
        or (model.yaml.get("yaml_file") if isinstance(getattr(model, "yaml", None), dict) else None)
    )
    print(f"  ckpt : {args.ckpt}")
    print(f"  cfg  : {cfg_used}")

    modules = find_wave_attn_modules(model)
    print(f"  found {len(modules)} WaveAttn module(s)")
    if not modules:
        print("\nNo WaveAttn modules in this checkpoint — nothing to probe.")
        return 1

    hr("[1] Branch parameters")
    for name, m in modules:
        parameter_report(name, m)

    hr("[2] Residual / main-path magnitude ratio")
    magnitude_report(model, modules, args.n_images, args.imgsz, args.data, device)

    if args.pretrained:
        hr("[3] Pretrained transfer check")
        if not cfg_used:
            print("  Could not recover model cfg from checkpoint — pass --cfg manually if needed.")
        else:
            pretrained_transfer_check(cfg_used, args.pretrained)
    else:
        print("\n(skip [3] — no --pretrained given)")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
