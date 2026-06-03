"""Visualise HaarDWT subbands on AFB images and measure HF energy *inside*
GT bounding boxes vs *outside* them.

This is the empirical test for the wavelet prior: if bacilli carry a strong
HF signature, the in-box HF energy should be substantially higher than the
background. A ratio near 1.0 means HF is uniform noise — the wavelet branch
has nothing task-relevant to offer, no architecture choice can save it.

Outputs per image (into --out/):
    <stem>_subbands.png   — original+boxes, LL, |LH|, |HL|, |HH|, HF energy map
And one CSV:
    hf_energy_ratio.csv   — per-image inside/outside means and their ratio

Colab usage:
    !python scripts/visualize_hf_subbands.py \\
        --data /content/chen_split/data.yaml \\
        --split val --n-images 12 \\
        --out /content/hf_debug
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.nn.modules.wavelet import HaarDWT


def load_yolo_labels(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.empty((0, 5))
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append([float(x) for x in parts[:5]])
    return np.array(rows) if rows else np.empty((0, 5))


def yolo_to_xyxy(boxes_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    if boxes_norm.size == 0:
        return np.empty((0, 4))
    cx = boxes_norm[:, 1] * w
    cy = boxes_norm[:, 2] * h
    bw = boxes_norm[:, 3] * w
    bh = boxes_norm[:, 4] * h
    x1 = np.clip(cx - bw / 2, 0, w - 1)
    y1 = np.clip(cy - bh / 2, 0, h - 1)
    x2 = np.clip(cx + bw / 2, 0, w - 1)
    y2 = np.clip(cy + bh / 2, 0, h - 1)
    return np.stack([x1, y1, x2, y2], axis=1)


def box_mask(boxes_xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy.astype(int):
        mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


def resolve_image_dir(data_yaml: Path, split: str) -> Path:
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", data_yaml.parent))
    split_rel = data.get(split)
    if isinstance(split_rel, list):
        split_rel = split_rel[0]
    cand = base / split_rel
    return cand if cand.is_dir() else Path(split_rel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="paper/figs/hf_debug")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    img_dir = resolve_image_dir(Path(args.data), args.split)
    # Standard YOLO layout: replace 'images' segment with 'labels'.
    label_dir = Path(str(img_dir).replace("images", "labels"))
    print(f"images: {img_dir}")
    print(f"labels: {label_dir}")

    all_imgs = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    )
    with_labels = [p for p in all_imgs if load_yolo_labels(label_dir / (p.stem + ".txt")).size > 0]
    if not with_labels:
        print(f"No labelled images in {img_dir}")
        return 1
    print(f"  {len(all_imgs)} total / {len(with_labels)} with labels")

    rng = np.random.default_rng(args.seed)
    sample = list(rng.choice(with_labels, size=min(args.n_images, len(with_labels)), replace=False))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dwt = HaarDWT(1)
    rows = []

    for img_path in sample:
        img = Image.open(img_path).convert("RGB").resize((args.imgsz, args.imgsz), Image.BILINEAR)
        gray = np.array(img.convert("L"), dtype=np.float32)
        H, W = gray.shape

        boxes_norm = load_yolo_labels(label_dir / (img_path.stem + ".txt"))
        boxes_xyxy = yolo_to_xyxy(boxes_norm, W, H)
        in_mask = box_mask(boxes_xyxy, H, W)

        t = torch.from_numpy(gray)[None, None]                # (1, 1, H, W)
        sub = dwt(t)[0, :, 0]                                 # (4, H/2, W/2)
        ll, lh, hl, hh = (sub[i].numpy() for i in range(4))
        hf_energy = lh ** 2 + hl ** 2 + hh ** 2               # (H/2, W/2)

        # Upsample HF map to full res for box-aligned inside/outside stats.
        hf_up = np.kron(hf_energy, np.ones((2, 2), dtype=np.float32))[:H, :W]
        e_in = float(hf_up[in_mask].mean()) if in_mask.any() else float("nan")
        e_out = float(hf_up[~in_mask].mean()) if (~in_mask).any() else float("nan")
        ratio = e_in / max(e_out, 1e-12)
        rows.append(
            {
                "img": img_path.name,
                "n_boxes": int(len(boxes_xyxy)),
                "energy_inside": e_in,
                "energy_outside": e_out,
                "ratio_in_over_out": ratio,
            }
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f"original + GT (n={len(boxes_xyxy)})")
        for x1, y1, x2, y2 in boxes_xyxy:
            axes[0, 0].add_patch(
                patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.2, ec="lime", fc="none")
            )
        axes[0, 1].imshow(ll, cmap="gray")
        axes[0, 1].set_title("LL  (approximation)")
        axes[0, 2].imshow(np.abs(lh), cmap="hot")
        axes[0, 2].set_title("|LH|  (horizontal HF)")
        axes[1, 0].imshow(np.abs(hl), cmap="hot")
        axes[1, 0].set_title("|HL|  (vertical HF)")
        axes[1, 1].imshow(np.abs(hh), cmap="hot")
        axes[1, 1].set_title("|HH|  (diagonal HF)")
        axes[1, 2].imshow(hf_energy, cmap="hot")
        axes[1, 2].set_title(f"HF energy   in/out = {ratio:.2f}")
        for ax in axes.flatten():
            ax.axis("off")
        plt.suptitle(
            f"{img_path.name}    inside={e_in:.3e}  outside={e_out:.3e}", y=0.99
        )
        plt.tight_layout()
        out_png = out_dir / f"{img_path.stem}_subbands.png"
        plt.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  {img_path.name}  in/out = {ratio:.3f}   -> {out_png.name}")

    csv_path = out_dir / "hf_energy_ratio.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ratios = [r["ratio_in_over_out"] for r in rows]
    print("\nSummary (HF energy inside-box / outside-box):")
    print(f"  mean ratio   : {np.mean(ratios):.3f}")
    print(f"  median ratio : {np.median(ratios):.3f}")
    print(f"  min / max    : {np.min(ratios):.3f} / {np.max(ratios):.3f}")
    print("    ratio >> 1  → bacilli light up in HF → wavelet prior is real")
    print("    ratio ≈ 1   → HF is uniform noise → wavelet prior gives no signal")
    print(f"\n  CSV : {csv_path}")
    print(f"  PNGs: {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
