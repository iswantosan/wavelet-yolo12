"""Architectural decision diagnostic.

Runs 1 checkpoint on a split, then computes 4 analyses that diagnose_baseline
does NOT cover but are needed to choose an architecture lever:

  [A] Per-IoU-threshold mAP curve  — diagnoses if box-reg or recall is bottleneck
  [B] FP-to-nearest-GT distance    — diagnoses if FPs are near-miss vs hard-neg
  [C] Confidence threshold F1 sweep — quantifies "is conf optimal?"
  [D] Size-binned TP attribution    — proxies which FPN scale matters

Then prints a DECISION MATRIX that maps observations → recommended arch levers.

Usage (1-seed, local or Colab):
    python scripts/diagnose_for_arch.py \\
        --ckpt paper/runs-baseline-nwd/content/runs/wavelet_chen/yolov12s_seed1050_60ep_kf5_fold0/weights/best.pt \\
        --data /content/tb_chen_split/data.yaml \\
        --split test \\
        --out diag_arch_fold0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def greedy_match_with_iou(iou_mat: np.ndarray, scores: np.ndarray, thresh: float):
    """Greedy match using precomputed IoU matrix. Returns tp_mask, tp_iou."""
    P, G = iou_mat.shape
    tp_mask = np.zeros(P, dtype=bool)
    tp_iou = np.zeros(P)
    if P == 0 or G == 0:
        return tp_mask, tp_iou
    used_gt = np.zeros(G, dtype=bool)
    order = np.argsort(-scores)
    for p in order:
        row = iou_mat[p] * (~used_gt)
        g = int(np.argmax(row))
        if row[g] >= thresh:
            tp_mask[p] = True
            tp_iou[p] = float(row[g])
            used_gt[g] = True
    return tp_mask, tp_iou


def compute_ap(tp_mask: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO-style 101-point AP from sorted-by-score tp_mask."""
    if n_gt == 0 or len(tp_mask) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp_sorted = tp_mask[order].astype(np.float32)
    fp_sorted = 1.0 - tp_sorted
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # 101-point interpolation
    ap = 0.0
    for r_target in np.linspace(0, 1, 101):
        prec_at = precision[recall >= r_target]
        ap += (prec_at.max() if len(prec_at) else 0.0) / 101
    return float(ap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="inference threshold (low to catch all)")
    ap.add_argument("--nms-iou", type=float, default=0.6)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="diag_arch")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data) as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", Path(args.data).parent))
    split_rel = data.get(args.split)
    if isinstance(split_rel, list):
        split_rel = split_rel[0]
    img_dir = base / split_rel if (base / split_rel).is_dir() else Path(split_rel)
    label_dir = Path(str(img_dir).replace("images", "labels"))
    print(f"  ckpt   : {args.ckpt}")
    print(f"  images : {img_dir}")
    print(f"  labels : {label_dir}")

    img_paths = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    )
    print(f"  {len(img_paths)} images")

    model = YOLO(args.ckpt)
    print(f"  running inference (conf={args.conf})...")

    # ──────────────────────────────────────────────────────────────────
    # Collect per-image: pred boxes/scores, gt boxes, IoU matrix
    # ──────────────────────────────────────────────────────────────────
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
            img_path, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
            device=args.device, verbose=False,
        )[0]
        if len(res.boxes):
            pred_xyxy = res.boxes.xyxy.cpu().numpy()
            pred_scores = res.boxes.conf.cpu().numpy()
        else:
            pred_xyxy = np.empty((0, 4))
            pred_scores = np.empty(0)

        ious = iou_matrix(pred_xyxy, gt_xyxy)
        per_image.append({
            "img": img_path.name,
            "W": W, "H": H,
            "gt_xyxy": gt_xyxy,
            "gt_diam": gt_diam,
            "pred_xyxy": pred_xyxy,
            "pred_scores": pred_scores,
            "iou_mat": ious,
        })

    n_gt_total = sum(len(s["gt_xyxy"]) for s in per_image)
    n_pred_total = sum(len(s["pred_xyxy"]) for s in per_image)
    print(f"  total GT={n_gt_total}  total pred={n_pred_total}")

    # ──────────────────────────────────────────────────────────────────
    # [A] Per-IoU-threshold mAP curve
    # ──────────────────────────────────────────────────────────────────
    iou_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ap_by_iou = {}
    for t in iou_grid:
        all_tp, all_score = [], []
        for s in per_image:
            tp_mask, _ = greedy_match_with_iou(s["iou_mat"], s["pred_scores"], t)
            all_tp.extend(tp_mask.tolist())
            all_score.extend(s["pred_scores"].tolist())
        ap_by_iou[t] = compute_ap(np.array(all_tp, dtype=bool), np.array(all_score), n_gt_total)

    map_5095 = float(np.mean(list(ap_by_iou.values())))

    # ──────────────────────────────────────────────────────────────────
    # [B] FP-to-nearest-GT distance (at IoU=0.5 threshold)
    # ──────────────────────────────────────────────────────────────────
    fp_dist_norm = []   # distance / GT diameter (proxy for "near-miss")
    fp_iou_best = []    # max IoU each FP has with any GT (irrespective of matching)
    for s in per_image:
        if not len(s["pred_xyxy"]) or not len(s["gt_xyxy"]):
            continue
        tp_mask, _ = greedy_match_with_iou(s["iou_mat"], s["pred_scores"], 0.5)
        fp_idx = np.where(~tp_mask)[0]
        if not len(fp_idx):
            continue
        pred_cx = (s["pred_xyxy"][fp_idx, 0] + s["pred_xyxy"][fp_idx, 2]) / 2
        pred_cy = (s["pred_xyxy"][fp_idx, 1] + s["pred_xyxy"][fp_idx, 3]) / 2
        gt_cx = (s["gt_xyxy"][:, 0] + s["gt_xyxy"][:, 2]) / 2
        gt_cy = (s["gt_xyxy"][:, 1] + s["gt_xyxy"][:, 3]) / 2
        for k, p in enumerate(fp_idx):
            dx = gt_cx - pred_cx[k]
            dy = gt_cy - pred_cy[k]
            d = np.sqrt(dx * dx + dy * dy)
            i = int(np.argmin(d))
            fp_dist_norm.append(float(d[i] / max(s["gt_diam"][i], 1.0)))
            fp_iou_best.append(float(s["iou_mat"][p].max()) if s["iou_mat"].shape[1] else 0.0)

    fp_dist_norm = np.array(fp_dist_norm)
    fp_iou_best = np.array(fp_iou_best)
    n_fp = len(fp_dist_norm)
    n_fp_near = int((fp_dist_norm < 0.5).sum())         # likely duplicate / low-IoU
    n_fp_close = int(((fp_dist_norm >= 0.5) & (fp_dist_norm < 2.0)).sum())
    n_fp_far = int((fp_dist_norm >= 2.0).sum())          # likely hard-negative
    n_fp_low_iou = int(((fp_iou_best >= 0.3) & (fp_iou_best < 0.5)).sum())  # had IoU 0.3-0.5 → almost TP

    # ──────────────────────────────────────────────────────────────────
    # [C] Confidence threshold F1 sweep (at IoU=0.5)
    # ──────────────────────────────────────────────────────────────────
    # Re-derive TP / FP scores at IoU=0.5
    tp_scores, fp_scores = [], []
    for s in per_image:
        tp_mask, _ = greedy_match_with_iou(s["iou_mat"], s["pred_scores"], 0.5)
        tp_scores.extend(s["pred_scores"][tp_mask].tolist())
        fp_scores.extend(s["pred_scores"][~tp_mask].tolist())
    tp_scores = np.array(tp_scores)
    fp_scores = np.array(fp_scores)

    conf_grid = np.linspace(0.05, 0.95, 19)
    f1_curve = []
    for ct in conf_grid:
        tp = int((tp_scores >= ct).sum())
        fp = int((fp_scores >= ct).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(n_gt_total, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        f1_curve.append((float(ct), p, r, f1))
    opt_ct, opt_p, opt_r, opt_f1 = max(f1_curve, key=lambda x: x[3])
    # F1 at default 0.25
    def_tp = int((tp_scores >= 0.25).sum())
    def_fp = int((fp_scores >= 0.25).sum())
    def_p = def_tp / max(def_tp + def_fp, 1)
    def_r = def_tp / max(n_gt_total, 1)
    def_f1 = 2 * def_p * def_r / max(def_p + def_r, 1e-9)

    # ──────────────────────────────────────────────────────────────────
    # [D] Size-binned TP attribution (FPN-scale proxy)
    # ──────────────────────────────────────────────────────────────────
    # YOLOv12s strides: P3=8, P4=16, P5=32. TAL is dynamic but objects
    # cluster by size to natural FPN level:
    #   d ≤ 48  → P3-typical (stride 8)
    #   48 < d ≤ 96 → P4-typical (stride 16)
    #   d > 96  → P5-typical (stride 32)
    tp_diams = []
    fn_diams = []
    for s in per_image:
        if not len(s["gt_xyxy"]):
            continue
        tp_mask, _ = greedy_match_with_iou(s["iou_mat"], s["pred_scores"], 0.5)
        if len(s["pred_xyxy"]):
            order = np.argsort(-s["pred_scores"])
            ious_sorted = s["iou_mat"][order]
            used_gt = np.zeros(len(s["gt_xyxy"]), dtype=bool)
            matched = -np.ones(len(s["pred_xyxy"]), dtype=int)
            for rank, p_orig in enumerate(order):
                row = ious_sorted[rank] * (~used_gt)
                g = int(np.argmax(row)) if len(row) else -1
                if g >= 0 and row[g] >= 0.5:
                    matched[p_orig] = g
                    used_gt[g] = True
            tp_gt_idx = matched[tp_mask]
            tp_diams.extend(s["gt_diam"][tp_gt_idx].tolist())
            fn_diams.extend(s["gt_diam"][~used_gt].tolist())
        else:
            fn_diams.extend(s["gt_diam"].tolist())
    tp_diams = np.array(tp_diams)
    fn_diams = np.array(fn_diams)

    def fpn_bin(d):
        if d <= 48: return "P3"
        if d <= 96: return "P4"
        return "P5"
    tp_by_scale = {"P3": 0, "P4": 0, "P5": 0}
    fn_by_scale = {"P3": 0, "P4": 0, "P5": 0}
    for d in tp_diams: tp_by_scale[fpn_bin(d)] += 1
    for d in fn_diams: fn_by_scale[fpn_bin(d)] += 1
    total_gt_by_scale = {k: tp_by_scale[k] + fn_by_scale[k] for k in tp_by_scale}
    recall_by_scale = {
        k: (tp_by_scale[k] / total_gt_by_scale[k] if total_gt_by_scale[k] else float("nan"))
        for k in tp_by_scale
    }

    # ──────────────────────────────────────────────────────────────────
    # Print summary
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[A] Per-IoU-threshold mAP")
    print("-" * 78)
    print(f"  {'IoU':>6}", end="")
    for t in iou_grid: print(f"{t:>7.2f}", end="")
    print(f"  | mAP50-95")
    print(f"  {'mAP':>6}", end="")
    for t in iou_grid: print(f"{ap_by_iou[t]:>7.3f}", end="")
    print(f"  | {map_5095:.3f}")
    drop_50_to_70 = ap_by_iou[0.5] - ap_by_iou[0.7]
    drop_70_to_90 = ap_by_iou[0.7] - ap_by_iou[0.9]
    print(f"\n  drop 0.5→0.7 = {drop_50_to_70:.3f}   drop 0.7→0.9 = {drop_70_to_90:.3f}")

    print("\n" + "=" * 78)
    print("[B] FP analysis (at IoU=0.5)")
    print("-" * 78)
    print(f"  total FP = {n_fp}")
    if n_fp:
        print(f"  near GT  (<0.5 box-diam):    {n_fp_near:5d}  ({100*n_fp_near/n_fp:5.1f}%)  ← duplicates / low-IoU near-miss")
        print(f"  close    (0.5–2 box-diam):  {n_fp_close:5d}  ({100*n_fp_close/n_fp:5.1f}%)  ← ambiguous")
        print(f"  far      (≥2 box-diam):     {n_fp_far:5d}  ({100*n_fp_far/n_fp:5.1f}%)  ← hard-negatives (background)")
        print(f"  FPs with IoU 0.3–0.5 to GT: {n_fp_low_iou:5d}  ({100*n_fp_low_iou/n_fp:5.1f}%)  ← would-be-TPs if box reg better")

    print("\n" + "=" * 78)
    print("[C] Confidence-threshold F1 sweep")
    print("-" * 78)
    print(f"  default conf=0.25:  P={def_p:.4f}  R={def_r:.4f}  F1={def_f1:.4f}")
    print(f"  optimal conf*={opt_ct:.2f}:  P={opt_p:.4f}  R={opt_r:.4f}  F1={opt_f1:.4f}")
    print(f"  F1 gain by tuning: {opt_f1 - def_f1:+.4f}")

    print("\n" + "=" * 78)
    print("[D] Size-binned TP attribution (FPN-scale proxy)")
    print("-" * 78)
    print(f"  {'Scale':>6} {'n_GT':>8} {'n_TP':>8} {'n_FN':>8} {'recall':>10}")
    for k in ("P3", "P4", "P5"):
        ng = total_gt_by_scale[k]
        ntp = tp_by_scale[k]
        nfn = fn_by_scale[k]
        rec = recall_by_scale[k]
        print(f"  {k:>6} {ng:>8d} {ntp:>8d} {nfn:>8d} {rec:>10.4f}")
    median_gt_diam = float(np.median(np.concatenate([tp_diams, fn_diams]))) if (len(tp_diams) + len(fn_diams)) else 0.0
    p3_share = total_gt_by_scale["P3"] / max(n_gt_total, 1)
    p5_share = total_gt_by_scale["P5"] / max(n_gt_total, 1)
    print(f"  median GT diameter = {median_gt_diam:.1f} px")
    print(f"  P3-typical share = {100*p3_share:.1f}%   P5-typical share = {100*p5_share:.1f}%")

    # ──────────────────────────────────────────────────────────────────
    # DECISION MATRIX (data-driven)
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  DECISION MATRIX")
    print("=" * 78)

    rec = []

    # Box-reg verdict
    if drop_50_to_70 > 0.15 or ap_by_iou[0.9] < 0.05:
        rec.append(("BOX-REG", "STRONG",
                    f"mAP@0.5={ap_by_iou[0.5]:.3f} → mAP@0.7={ap_by_iou[0.7]:.3f} (drop {drop_50_to_70:.3f}); mAP@0.9={ap_by_iou[0.9]:.3f}",
                    ["SIoU/EIoU/WiseIoU loss", "DFL weight ↑ (try 5-8)", "imgsz=1280", "auxiliary refinement head"]))
    elif drop_50_to_70 > 0.08:
        rec.append(("BOX-REG", "MODERATE", f"drop 0.5→0.7 = {drop_50_to_70:.3f}",
                    ["SIoU loss", "DFL weight ↑ (try 3-5)"]))

    # FP nature verdict
    if n_fp:
        near_pct = 100 * n_fp_near / n_fp
        far_pct = 100 * n_fp_far / n_fp
        low_iou_pct = 100 * n_fp_low_iou / n_fp
        if low_iou_pct > 25:
            rec.append(("NEAR-MISS-FP", "HIGH",
                        f"{low_iou_pct:.1f}% of FPs have 0.3≤IoU<0.5 → would be TPs if box reg better",
                        ["confirms box-reg fix above; soft-NMS at inference"]))
        if far_pct > 50:
            rec.append(("HARD-NEG", "HIGH",
                        f"{far_pct:.1f}% of FPs are >2 box-diam from any GT (background mistakes)",
                        ["focal loss / VFL", "stain-aware augmentation", "hard-neg mining"]))
        elif near_pct > 50:
            rec.append(("DUPLICATES/NEAR-MISS", "HIGH",
                        f"{near_pct:.1f}% of FPs are <0.5 box-diam from GT",
                        ["soft-NMS", "NMS IoU threshold tuning", "box-reg fix"]))

    # Conf calibration verdict
    if abs(opt_ct - 0.25) > 0.05 and (opt_f1 - def_f1) > 0.005:
        rec.append(("CONF-CAL", "MODERATE",
                    f"optimal conf*={opt_ct:.2f} vs default 0.25; F1 gain {opt_f1-def_f1:+.4f}",
                    [f"use conf={opt_ct:.2f} at inference"]))

    # FPN scale verdict
    if p5_share < 0.03 and total_gt_by_scale["P5"] < 0.05 * n_gt_total:
        rec.append(("DROP-P5", "RECOMMENDED",
                    f"P5-typical objects = {100*p5_share:.1f}% of GT — P5 head is mostly dead weight",
                    ["use 2-scale head (P3+P4 only) → smaller, faster, mAP-neutral"]))
    if p3_share > 0.6:
        rec.append(("P3-FOCUSED", "RECOMMENDED",
                    f"P3-typical objects = {100*p3_share:.1f}% — capacity should be at P3",
                    ["add C3k2 blocks at P3 neck", "add P2 head (stride 4) if median <40px"]))

    # Small-object verdict (size-stratified recall)
    if recall_by_scale["P3"] < 0.7 and total_gt_by_scale["P3"] > 30:
        rec.append(("SMALL-OBJ", "MODERATE",
                    f"P3 recall = {recall_by_scale['P3']:.3f} — small objects underperforming",
                    ["NWD loss", "imgsz=1280", "P2 head (stride 4)"]))

    if not rec:
        print("\n  (no strong signal — possibly already near-optimal; investigate qualitative failures)")
    else:
        for i, (tag, sev, reason, fixes) in enumerate(rec, 1):
            print(f"\n  [{i}] {tag} ({sev})")
            print(f"      Why: {reason}")
            print(f"      Recommended fixes:")
            for f in fixes:
                print(f"        - {f}")

    # ──────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # [A]
    ts = list(ap_by_iou.keys())
    aps = [ap_by_iou[t] for t in ts]
    axes[0, 0].plot(ts, aps, "o-", color="steelblue")
    axes[0, 0].set_xlabel("IoU threshold")
    axes[0, 0].set_ylabel("AP")
    axes[0, 0].set_title(f"[A] Per-IoU-threshold AP   (mAP50-95 = {map_5095:.3f})")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # [B]
    if n_fp:
        axes[0, 1].hist(fp_dist_norm[fp_dist_norm < 5], bins=40, range=(0, 5), color="firebrick", edgecolor="white")
        axes[0, 1].axvline(0.5, color="black", ls="--", lw=1, label="near (<0.5)")
        axes[0, 1].axvline(2.0, color="black", ls=":", lw=1, label="far (≥2)")
        axes[0, 1].set_title(f"[B] FP distance to nearest GT (norm by GT diam)  n={n_fp}")
        axes[0, 1].set_xlabel("distance / GT diameter")
        axes[0, 1].legend()

    # [C]
    cs = [x[0] for x in f1_curve]
    f1s = [x[3] for x in f1_curve]
    ps = [x[1] for x in f1_curve]
    rs = [x[2] for x in f1_curve]
    axes[1, 0].plot(cs, f1s, "o-", label="F1", color="purple")
    axes[1, 0].plot(cs, ps, "--", label="P", color="seagreen")
    axes[1, 0].plot(cs, rs, "--", label="R", color="firebrick")
    axes[1, 0].axvline(0.25, color="grey", ls=":", lw=1, label="default 0.25")
    axes[1, 0].axvline(opt_ct, color="black", ls="--", lw=1, label=f"opt {opt_ct:.2f}")
    axes[1, 0].set_xlabel("conf threshold")
    axes[1, 0].set_title(f"[C] Conf sweep — opt={opt_ct:.2f}, F1={opt_f1:.3f} (vs default {def_f1:.3f})")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # [D]
    if len(tp_diams) + len(fn_diams):
        axes[1, 1].hist([tp_diams, fn_diams], bins=20, stacked=True,
                        label=["TP", "FN"], color=["seagreen", "firebrick"], edgecolor="white")
        axes[1, 1].axvline(48, color="black", ls="--", lw=1, label="P3|P4 boundary")
        axes[1, 1].axvline(96, color="black", ls=":", lw=1, label="P4|P5 boundary")
        axes[1, 1].set_xlabel("GT diameter (px)")
        axes[1, 1].set_title(f"[D] Size-binned outcome (median GT={median_gt_diam:.0f}px)")
        axes[1, 1].legend()
    plt.tight_layout()
    fig_path = out_dir / "decision_matrix.png"
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\n  saved {fig_path}")

    # ──────────────────────────────────────────────────────────────────
    # JSON dump
    # ──────────────────────────────────────────────────────────────────
    summary = {
        "ckpt": args.ckpt,
        "split": args.split,
        "n_images": len(per_image),
        "n_gt": n_gt_total,
        "n_pred": n_pred_total,
        "map_by_iou": {f"{t:.2f}": ap_by_iou[t] for t in iou_grid},
        "map_5095": map_5095,
        "drop_50_to_70": drop_50_to_70,
        "drop_70_to_90": drop_70_to_90,
        "fp": {
            "total": int(n_fp),
            "near_lt_0.5": int(n_fp_near),
            "close_0.5_to_2": int(n_fp_close),
            "far_ge_2": int(n_fp_far),
            "would_be_tp_iou_0.3_to_0.5": int(n_fp_low_iou),
        },
        "conf_sweep": {
            "optimal_conf": float(opt_ct),
            "optimal_p": float(opt_p),
            "optimal_r": float(opt_r),
            "optimal_f1": float(opt_f1),
            "default_conf": 0.25,
            "default_p": float(def_p),
            "default_r": float(def_r),
            "default_f1": float(def_f1),
        },
        "fpn_attribution": {
            "tp_by_scale": tp_by_scale,
            "fn_by_scale": fn_by_scale,
            "recall_by_scale": recall_by_scale,
            "median_gt_diameter": median_gt_diam,
            "p3_share": float(p3_share),
            "p5_share": float(p5_share),
        },
        "recommendations": [
            {"tag": t, "severity": s, "reason": r, "fixes": f}
            for t, s, r, f in rec
        ],
    }
    json_path = out_dir / "decision_matrix.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
