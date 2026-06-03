"""Diagnose WHERE the baseline YOLOv12s is stuck on the AFB Chen split.

Runs the trained ckpt on val/test, matches predictions to GT greedily by IoU,
then prints / plots:

  [1] Overall TP/FP/FN at conf={0.001, 0.25, 0.5}
  [2] Per-size recall (by GT box diameter = sqrt(area) in pixels)
  [3] TP IoU distribution           — diagnoses box-regression quality
  [4] TP vs FP confidence histogram — diagnoses classifier separability
  [5] Per-image recall histogram    — diagnoses image-level failure spread
  [6] Worst-recall N images (viz)   — qualitative inspection of failure modes

Interpretation guide:
  - Per-size recall drops sharply at small sizes  → small-object problem
                                                    (NWD / scale-aware loss
                                                    has theoretical fit)
  - TP IoU median ~0.55, clustered near threshold → box regression is weak
                                                    (NWD again, or refined head)
  - TP IoU median >0.75                            → box quality fine, focus on
                                                    recall (FN reduction)
  - TP/FP score histograms overlap heavily        → classifier weak; try
                                                    threshold sweep / hard-neg
  - Per-image recall bimodal                       → some images systematically
                                                    fail → inspect worst-N viz

Colab usage:
    !python scripts/diagnose_baseline.py \\
        --ckpt /content/runs/.../weights/best.pt \\
        --data /content/chen_split/data.yaml \\
        --split val --out /content/diag
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

from ultralytics import YOLO


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


def yolo_to_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
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


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter = (
        np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
        * np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.clip(union, 1e-9, None)


def greedy_match(pred_xyxy, pred_scores, gt_xyxy, iou_thresh):
    """Greedy IoU matching. Returns (tp_mask, tp_iou, fn_mask, matched_gt_idx)."""
    P, G = len(pred_xyxy), len(gt_xyxy)
    tp_mask = np.zeros(P, dtype=bool)
    tp_iou = np.zeros(P)
    matched = -np.ones(P, dtype=int)
    used_gt = np.zeros(G, dtype=bool)
    if P == 0 or G == 0:
        return tp_mask, tp_iou, ~used_gt, matched
    order = np.argsort(-pred_scores)
    ious = iou_matrix(pred_xyxy[order], gt_xyxy)
    for rank, p_orig in enumerate(order):
        row = ious[rank] * (~used_gt)
        g = int(np.argmax(row))
        if row[g] >= iou_thresh:
            tp_mask[p_orig] = True
            tp_iou[p_orig] = float(row[g])
            matched[p_orig] = g
            used_gt[g] = True
    return tp_mask, tp_iou, ~used_gt, matched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="paper/figs/diag_baseline")
    ap.add_argument("--worst-n", type=int, default=8)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    with open(args.data) as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", Path(args.data).parent))
    split_rel = data.get(args.split)
    if isinstance(split_rel, list):
        split_rel = split_rel[0]
    img_dir = base / split_rel if (base / split_rel).is_dir() else Path(split_rel)
    label_dir = Path(str(img_dir).replace("images", "labels"))
    print(f"  images: {img_dir}")
    print(f"  labels: {label_dir}")

    img_paths = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    )
    print(f"  {len(img_paths)} images")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.ckpt)
    print(f"  loaded {args.ckpt}\n  running inference...")

    all_tp_iou, all_tp_scores, all_fp_scores = [], [], []
    all_gt_diam = []
    per_size_gt = {"<8": 0, "8-16": 0, "16-24": 0, "24-32": 0, ">32": 0}
    per_size_tp = {k: 0 for k in per_size_gt}

    def size_bin(d):
        if d < 8:
            return "<8"
        if d < 16:
            return "8-16"
        if d < 24:
            return "16-24"
        if d < 32:
            return "24-32"
        return ">32"

    per_image = []
    for img_path in img_paths:
        img_pil = Image.open(img_path).convert("RGB")
        W, H = img_pil.size

        gt_norm = load_labels(label_dir / (img_path.stem + ".txt"))
        gt_xyxy = yolo_to_xyxy(gt_norm, W, H)
        gt_diam = (
            np.sqrt((gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1]))
            if len(gt_xyxy)
            else np.empty(0)
        )

        res = model.predict(
            img_path, imgsz=args.imgsz, conf=args.conf, iou=0.6,
            device=args.device, verbose=False,
        )[0]
        if len(res.boxes):
            pred_xyxy = res.boxes.xyxy.cpu().numpy()
            pred_scores = res.boxes.conf.cpu().numpy()
        else:
            pred_xyxy = np.empty((0, 4))
            pred_scores = np.empty(0)

        tp_mask, tp_iou, fn_mask, _ = greedy_match(
            pred_xyxy, pred_scores, gt_xyxy, args.iou_thresh
        )
        gt_matched = ~fn_mask

        all_tp_iou.extend(tp_iou[tp_mask].tolist())
        all_tp_scores.extend(pred_scores[tp_mask].tolist())
        all_fp_scores.extend(pred_scores[~tp_mask].tolist())
        all_gt_diam.extend(gt_diam.tolist())
        for d, m in zip(gt_diam, gt_matched):
            b = size_bin(float(d))
            per_size_gt[b] += 1
            if m:
                per_size_tp[b] += 1

        per_image.append(
            {
                "img": img_path.name,
                "img_path": str(img_path),
                "n_gt": int(len(gt_xyxy)),
                "n_pred": int(len(pred_xyxy)),
                "n_tp": int(tp_mask.sum()),
                "n_fp": int((~tp_mask).sum()),
                "n_fn": int(fn_mask.sum()),
                "recall": float(tp_mask.sum()) / max(len(gt_xyxy), 1),
                "gt_xyxy": gt_xyxy,
                "pred_xyxy": pred_xyxy,
                "pred_scores": pred_scores,
                "tp_mask": tp_mask,
            }
        )

    # ---------- [1] overall counts --------------------------------------
    n_gt = sum(s["n_gt"] for s in per_image)
    n_tp = sum(s["n_tp"] for s in per_image)
    n_fp = sum(s["n_fp"] for s in per_image)
    print(f"\n  [1] At conf≥{args.conf}, IoU≥{args.iou_thresh}:")
    print(f"      GT={n_gt}  TP={n_tp}  FP={n_fp}  FN={n_gt-n_tp}")
    print(f"      Recall={n_tp/max(n_gt,1):.4f}  Precision={n_tp/max(n_tp+n_fp,1):.4f}")

    tp_scores_arr = np.array(all_tp_scores)
    fp_scores_arr = np.array(all_fp_scores)
    for ct in (0.25, 0.5):
        tp_at = int((tp_scores_arr >= ct).sum())
        fp_at = int((fp_scores_arr >= ct).sum())
        rec = tp_at / max(n_gt, 1)
        prec = tp_at / max(tp_at + fp_at, 1)
        print(f"      conf≥{ct:.2f}: TP={tp_at}  FP={fp_at}  Recall={rec:.4f}  Precision={prec:.4f}")

    # ---------- [2] per-size recall ------------------------------------
    print(f"\n  [2] Per-size GT recall (box diameter = sqrt(area) in px):")
    print(f"      {'size':<10}{'nGT':>8}{'nMatched':>10}{'recall':>10}")
    print(f"      {'-'*10}{'-'*8}{'-'*10}{'-'*10}")
    for k in ["<8", "8-16", "16-24", "24-32", ">32"]:
        ng = per_size_gt[k]
        nm = per_size_tp[k]
        rec = nm / ng if ng else float("nan")
        print(f"      {k:<10}{ng:>8}{nm:>10}{rec:>10.4f}")

    # ---------- [3-5] histograms ---------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    if all_tp_iou:
        axes[0, 0].hist(all_tp_iou, bins=20, range=(0.5, 1.0), color="seagreen", edgecolor="white")
        axes[0, 0].axvline(np.median(all_tp_iou), color="black", ls="--", lw=1)
        axes[0, 0].set_title(
            f"[3] TP IoU dist (n={len(all_tp_iou)})\n"
            f"median={np.median(all_tp_iou):.3f}  mean={np.mean(all_tp_iou):.3f}"
        )
        axes[0, 0].set_xlabel("IoU")
    if all_tp_scores or all_fp_scores:
        axes[0, 1].hist(all_tp_scores, bins=30, range=(0, 1), alpha=0.6,
                        label=f"TP (n={len(all_tp_scores)})", color="seagreen")
        axes[0, 1].hist(all_fp_scores, bins=30, range=(0, 1), alpha=0.5,
                        label=f"FP (n={len(all_fp_scores)})", color="firebrick")
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("[4] Confidence: TP vs FP")
        axes[0, 1].set_xlabel("conf")
        axes[0, 1].legend()
    if all_gt_diam:
        axes[1, 0].hist(all_gt_diam, bins=30, color="steelblue", edgecolor="white")
        axes[1, 0].axvline(np.median(all_gt_diam), color="black", ls="--", lw=1)
        axes[1, 0].set_title(
            f"GT box diameter (sqrt(area))\n"
            f"median={np.median(all_gt_diam):.1f}px  mean={np.mean(all_gt_diam):.1f}px"
        )
        axes[1, 0].set_xlabel("px")
    recs = [s["recall"] for s in per_image if s["n_gt"] > 0]
    if recs:
        axes[1, 1].hist(recs, bins=20, range=(0, 1), color="goldenrod", edgecolor="white")
        axes[1, 1].axvline(np.median(recs), color="black", ls="--", lw=1)
        axes[1, 1].set_title(
            f"[5] Per-image recall (n={len(recs)})\n"
            f"median={np.median(recs):.3f}  mean={np.mean(recs):.3f}"
        )
        axes[1, 1].set_xlabel("recall")
    plt.tight_layout()
    plt.savefig(out_dir / "diag_overview.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\n  saved {out_dir/'diag_overview.png'}")

    # ---------- [6] worst-N viz ----------------------------------------
    worst = sorted([s for s in per_image if s["n_gt"] > 0], key=lambda s: s["recall"])[: args.worst_n]
    if worst:
        cols = 2
        rows = (len(worst) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 6))
        axes = np.atleast_2d(axes).flatten()
        for ax, s in zip(axes, worst):
            ax.imshow(Image.open(s["img_path"]).convert("RGB"))
            for x1, y1, x2, y2 in s["gt_xyxy"]:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.0, ec="lime", fc="none"))
            for (x1, y1, x2, y2), sc in zip(s["pred_xyxy"], s["pred_scores"]):
                if sc >= 0.25:
                    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.0, ec="red", fc="none"))
            ax.set_title(
                f'{s["img"]}  GT={s["n_gt"]} TP={s["n_tp"]} FN={s["n_fn"]} '
                f'recall={s["recall"]:.2f}'
            )
            ax.axis("off")
        for ax in axes[len(worst):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "worst_recall.png", dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  saved {out_dir/'worst_recall.png'}")

    # CSV
    csv_path = out_dir / "per_image.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img", "n_gt", "n_pred", "n_tp", "n_fp", "n_fn", "recall"])
        for s in per_image:
            w.writerow([s["img"], s["n_gt"], s["n_pred"], s["n_tp"], s["n_fp"], s["n_fn"], f"{s['recall']:.4f}"])
    print(f"  saved {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
