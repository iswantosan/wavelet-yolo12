"""Sweep the confidence threshold on val and report the F1-max operating point.

Cheap, no retrain. Inference once at conf=0.001, then sweep thresholds and
compute precision / recall / F1 at each. Output tells you exactly which conf
to use at deployment / test eval.

Usage:
    !python scripts/sweep_conf.py \\
        --ckpt /content/runs/.../weights/best.pt \\
        --data /content/chen_split/data.yaml \\
        --split val \\
        --out /content/conf_sweep
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
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


def iou_matrix(a, b):
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


def greedy_match(pred_xyxy, pred_scores, gt_xyxy, iou_thresh=0.5):
    P, G = len(pred_xyxy), len(gt_xyxy)
    tp_mask = np.zeros(P, dtype=bool)
    if P == 0 or G == 0:
        return tp_mask
    order = np.argsort(-pred_scores)
    ious = iou_matrix(pred_xyxy[order], gt_xyxy)
    used = np.zeros(G, dtype=bool)
    for rank, p_orig in enumerate(order):
        row = ious[rank] * (~used)
        g = int(np.argmax(row))
        if row[g] >= iou_thresh:
            tp_mask[p_orig] = True
            used[g] = True
    return tp_mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="paper/figs/conf_sweep")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
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
    print(f"  loaded {args.ckpt}\n  running inference at conf=0.001...")

    all_scores: list = []
    all_tp_flags: list = []
    n_gt_total = 0
    for img_path in img_paths:
        W, H = Image.open(img_path).size
        gt_norm = load_labels(label_dir / (img_path.stem + ".txt"))
        gt_xyxy = yolo_to_xyxy(gt_norm, W, H)
        n_gt_total += len(gt_xyxy)
        res = model.predict(
            img_path, imgsz=args.imgsz, conf=0.001, iou=0.6,
            device=args.device, verbose=False,
        )[0]
        if len(res.boxes) == 0:
            continue
        pred_xyxy = res.boxes.xyxy.cpu().numpy()
        pred_scores = res.boxes.conf.cpu().numpy()
        tp_mask = greedy_match(pred_xyxy, pred_scores, gt_xyxy, args.iou_thresh)
        all_scores.extend(pred_scores.tolist())
        all_tp_flags.extend(tp_mask.tolist())

    scores = np.array(all_scores)
    tp_flags = np.array(all_tp_flags, dtype=bool)
    print(f"  total preds: {len(scores)}, GT: {n_gt_total}\n")

    # Sweep
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for conf in thresholds:
        mask = scores >= conf
        tp = int(tp_flags[mask].sum())
        fp = int((~tp_flags[mask]).sum())
        fn = n_gt_total - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(n_gt_total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append((float(conf), tp, fp, fn, precision, recall, f1))

    best = max(rows, key=lambda r: r[6])
    default_row = min(rows, key=lambda r: abs(r[0] - 0.25))

    # Print table at 0.05 step
    print(f"  {'conf':<8}{'TP':>6}{'FP':>6}{'FN':>6}{'precision':>12}{'recall':>10}{'F1':>10}")
    print(f"  {'-' * 8}{'-' * 6}{'-' * 6}{'-' * 6}{'-' * 12}{'-' * 10}{'-' * 10}")
    for r in rows[::5]:
        print(f"  {r[0]:<8.2f}{r[1]:>6}{r[2]:>6}{r[3]:>6}{r[4]:>12.4f}{r[5]:>10.4f}{r[6]:>10.4f}")

    print(f"\n  >>> BEST F1: conf={best[0]:.3f}  P={best[4]:.4f}  R={best[5]:.4f}  F1={best[6]:.4f}")
    print(f"      TP={best[1]}  FP={best[2]}  FN={best[3]}")
    print(f"      default conf=0.25 :   P={default_row[4]:.4f}  R={default_row[5]:.4f}  F1={default_row[6]:.4f}")
    delta_f1 = best[6] - default_row[6]
    pct = 100 * delta_f1 / max(default_row[6], 1e-9)
    print(f"      F1 gain over default: {delta_f1:+.4f}  ({pct:+.2f}%)")

    # CSV
    csv_path = out_dir / "conf_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conf", "TP", "FP", "FN", "precision", "recall", "F1"])
        for r in rows:
            w.writerow(
                [f"{r[0]:.2f}", r[1], r[2], r[3], f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}"]
            )

    # Plot
    confs = np.array([r[0] for r in rows])
    precisions = np.array([r[4] for r in rows])
    recalls = np.array([r[5] for r in rows])
    f1s = np.array([r[6] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(confs, precisions, label="precision", color="firebrick")
    axes[0].plot(confs, recalls, label="recall", color="steelblue")
    axes[0].plot(confs, f1s, label="F1", color="seagreen", lw=2.5)
    axes[0].axvline(best[0], ls="--", color="black", lw=1, label=f"F1-max @ {best[0]:.2f}")
    axes[0].axvline(0.25, ls=":", color="gray", lw=1, label="default 0.25")
    axes[0].set_xlabel("conf threshold")
    axes[0].set_ylabel("score")
    axes[0].set_title(
        f"P / R / F1 vs conf\n"
        f"best F1={best[6]:.3f} @ conf={best[0]:.2f} "
        f"(default → F1={default_row[6]:.3f})"
    )
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[1].plot(recalls, precisions, color="purple", lw=2)
    axes[1].scatter([best[5]], [best[4]], color="black", s=80, zorder=10,
                    label=f"F1-max @ conf={best[0]:.2f}")
    axes[1].scatter([default_row[5]], [default_row[4]], color="gray", s=80, zorder=10,
                    label="default conf=0.25")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("PR curve (each point = one conf threshold)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_dir / "conf_sweep.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\n  saved {csv_path}")
    print(f"  saved {out_dir / 'conf_sweep.png'}")
    print(f"\n  USE THIS: pass conf={best[0]:.2f} to model.val() / model.predict() at test time")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
