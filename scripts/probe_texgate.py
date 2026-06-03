"""Probe a trained TextureGate model: did the gate actually learn to suppress
background-texture regions, or did it stay at identity (= dead, output ≡
baseline)?

Three checks (analog of probe_wavelet_attn.py for wavelet variants):

  [1] Gate parameters — head[-1] last-conv weights moved from zero-init?
      Zero std + zero bias = DEAD (gate stuck at 2·sigmoid(0) = 1.0).
  [2] Gate output statistics on real images:
        - global mean of gate (close to 1.0 = no net effect)
        - mean *inside* GT boxes vs *outside*
        - ratio inside/outside: > 1.2 = gate suppresses background ✓
                                ≈ 1.0 = identity ✗
                                < 0.8 = gate inverted (bug) ✗
  [3] Per-image heatmap visualization saved as PNG.

Colab usage:
    !python scripts/probe_texgate.py \\
        --ckpt /content/runs/.../weights/best.pt \\
        --data /content/tb_kfold/fold0/data.yaml \\
        --split val --n-images 12 \\
        --out /content/texgate_probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from ultralytics.nn.modules.wavelet import TextureGate


def load_labels(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.empty((0, 5))
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append([float(x) for x in parts[:5]])
    return np.array(rows) if rows else np.empty((0, 5))


def yolo_to_xyxy(b, W, H):
    if b.size == 0:
        return np.empty((0, 4))
    cx, cy, bw, bh = b[:, 1] * W, b[:, 2] * H, b[:, 3] * W, b[:, 4] * H
    return np.stack(
        [
            np.clip(cx - bw / 2, 0, W - 1),
            np.clip(cy - bh / 2, 0, H - 1),
            np.clip(cx + bw / 2, 0, W - 1),
            np.clip(cy + bh / 2, 0, H - 1),
        ],
        axis=1,
    )


def box_mask(boxes_xyxy, H, W):
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy.astype(int):
        mask[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = True
    return mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="paper/figs/texgate_probe")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from PIL import Image

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"\n  Loading {args.ckpt} ...")
    y = YOLO(args.ckpt)
    model = y.model.to(device).eval()

    gates = [(n, m) for n, m in model.named_modules() if isinstance(m, TextureGate)]
    print(f"  Found {len(gates)} TextureGate module(s)")
    if not gates:
        print("\n  No TextureGate in this checkpoint — wrong file?")
        return 1

    # ---------- [1] Gate parameters ------------------------------------
    print(f"\n{'=' * 64}\n  [1] Gate parameters (init: zeros → gate = 1.0)\n{'=' * 64}")
    print(f"  {'gate':<28}{'|W| mean':>14}{'W std':>14}{'bias':>16}{'verdict':>14}")
    print(f"  {'-' * 28}{'-' * 14}{'-' * 14}{'-' * 16}{'-' * 14}")
    for name, m in gates:
        last_conv = m.head[-1]
        w_abs = last_conv.weight.detach().abs().mean().item()
        w_std = last_conv.weight.detach().std().item()
        b_val = float(last_conv.bias.detach().mean().item())
        if w_std < 1e-4 and abs(b_val) < 1e-4:
            verdict = "DEAD"
        elif w_std < 1e-3:
            verdict = "barely-moved"
        else:
            verdict = "active"
        print(f"  {name:<28}{w_abs:>14.3e}{w_std:>14.3e}{b_val:>+16.3e}{verdict:>14}")

    # ---------- Sample images ------------------------------------------
    with open(args.data) as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", Path(args.data).parent))
    split_rel = data.get(args.split)
    if isinstance(split_rel, list):
        split_rel = split_rel[0]
    img_dir = base / split_rel if (base / split_rel).is_dir() else Path(split_rel)
    label_dir = Path(str(img_dir).replace("images", "labels"))
    print(f"\n  images: {img_dir}")
    print(f"  labels: {label_dir}")

    all_imgs = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    )
    with_labels = [p for p in all_imgs if load_labels(label_dir / (p.stem + ".txt")).size > 0]
    sample = with_labels[: args.n_images]
    if not sample:
        print("  No labelled images found.")
        return 1
    print(f"  using {len(sample)} labelled images\n")

    # Build batch + capture gate inputs via forward hook (we'll re-derive
    # the gate map by calling m.head(input) since the captured output is
    # input * gate, not the gate itself).
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(_mod, inp, _out):
            captured[name] = inp[0].detach()
        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in gates]

    tensors = []
    box_info = []
    for p in sample:
        img_pil = Image.open(p).convert("RGB")
        img_rs = img_pil.resize((args.imgsz, args.imgsz), Image.BILINEAR)
        tensors.append(TF.to_tensor(img_rs))
        gt_norm = load_labels(label_dir / (p.stem + ".txt"))
        gt_xyxy = yolo_to_xyxy(gt_norm, args.imgsz, args.imgsz)
        box_info.append((p, img_rs, gt_xyxy))
    x = torch.stack(tensors).to(device)

    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()

    # ---------- [2] Gate output statistics ----------------------------
    print(f"{'=' * 64}\n  [2] Gate output stats on real images\n{'=' * 64}")
    print(f"  {'gate':<28}{'global mean':>14}{'inside-box':>14}{'outside-box':>14}{'in/out':>10}{'verdict':>14}")
    print(f"  {'-' * 28}{'-' * 14}{'-' * 14}{'-' * 14}{'-' * 10}{'-' * 14}")

    summary = {}
    for name, m in gates:
        feat_in = captured[name].to(device)                      # (N, C, Hf, Wf)
        with torch.no_grad():
            logits = m.head(feat_in)                             # (N, 1, Hf, Wf)
            gate = (2 * torch.sigmoid(logits)).cpu().numpy()     # (N, 1, Hf, Wf)
        gate = gate[:, 0]                                        # (N, Hf, Wf)
        Hf, Wf = gate.shape[1:]

        ins_means, out_means, glob_means = [], [], []
        for i in range(gate.shape[0]):
            gmap = gate[i]
            glob_means.append(float(gmap.mean()))
            _, _, gt_xyxy = box_info[i]
            if len(gt_xyxy) == 0:
                continue
            scale_h = Hf / args.imgsz
            scale_w = Wf / args.imgsz
            gt_f = gt_xyxy.copy()
            gt_f[:, [0, 2]] *= scale_w
            gt_f[:, [1, 3]] *= scale_h
            mask = box_mask(gt_f, Hf, Wf)
            if mask.any():
                ins_means.append(float(gmap[mask].mean()))
            if (~mask).any():
                out_means.append(float(gmap[~mask].mean()))

        g_mean = float(np.mean(glob_means))
        in_mean = float(np.mean(ins_means)) if ins_means else float("nan")
        out_mean = float(np.mean(out_means)) if out_means else float("nan")
        ratio = in_mean / max(out_mean, 1e-9)
        if ratio > 1.2:
            verdict = "suppress-bg"
        elif 0.8 <= ratio <= 1.2:
            verdict = "near-identity"
        else:
            verdict = "INVERTED"
        print(f"  {name:<28}{g_mean:>14.4f}{in_mean:>14.4f}{out_mean:>14.4f}{ratio:>10.3f}{verdict:>14}")
        summary[name] = {
            "gate_maps": gate,
            "in_mean": in_mean,
            "out_mean": out_mean,
            "ratio": ratio,
        }

    # ---------- [3] Heatmap visualization per gate --------------------
    print(f"\n{'=' * 64}\n  [3] Saving per-gate heatmap visualizations\n{'=' * 64}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_show = min(8, len(sample))
    for name, _ in gates:
        gate_maps = summary[name]["gate_maps"]
        cols, rows = 4, (n_show + 3) // 4
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_1d(axes).flatten()
        for i in range(n_show):
            ax = axes[i]
            _, img_rs, gt_xyxy = box_info[i]
            ax.imshow(img_rs)
            # Upsample gate map to image size with np.kron
            gmap = gate_maps[i]
            scale_h = max(1, args.imgsz // gmap.shape[0])
            scale_w = max(1, args.imgsz // gmap.shape[1])
            gate_up = np.kron(gmap, np.ones((scale_h, scale_w)))[: args.imgsz, : args.imgsz]
            ax.imshow(
                gate_up, cmap="RdBu_r", alpha=0.45,
                vmin=0, vmax=2, extent=(0, args.imgsz, args.imgsz, 0),
            )
            for x1, y1, x2, y2 in gt_xyxy:
                ax.add_patch(
                    patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.2, ec="lime", fc="none")
                )
            ax.set_title(f"{box_info[i][0].name}\nblue<1 (suppress)  red>1 (amplify)")
            ax.axis("off")
        for ax in axes[n_show:]:
            ax.axis("off")
        plt.suptitle(
            f"{name}    in/out = {summary[name]['ratio']:.3f}    "
            f"(in={summary[name]['in_mean']:.3f}, out={summary[name]['out_mean']:.3f})",
            y=1.0,
        )
        plt.tight_layout()
        fname = f"gate_heatmap_{name.replace('.', '_')}.png"
        plt.savefig(out_dir / fname, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  saved {out_dir / fname}")

    print(f"\n  Interpretation cheat sheet:")
    print(f"    [1] DEAD       → gate never moved from zero-init, aux loss likely silent")
    print(f"    [2] in/out > 1.2 → gate suppresses background as designed ✓")
    print(f"    [2] in/out ≈ 1   → gate stayed near identity, branch contributing little")
    print(f"    [2] in/out < 0.8 → gate INVERTED, hurts (check loss sign / pos_weight)")
    print(f"    [3] visual heatmaps: blue regions outside boxes = correct behaviour")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
