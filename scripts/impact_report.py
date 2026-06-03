"""Compare 2+ trained ckpts side-by-side and report the impact.

Protocol (proper "tuned-vs-tuned" comparison):
  For each ckpt:
    1. Run inference on --val-split at conf=0.001 to capture all preds.
    2. Sweep conf 0.05–0.95 and find F1-max on val.
    3. Evaluate on --test-split at that tuned conf via Ultralytics
       model.val() → get mAP50, mAP50-95, P, R.
    4. Also report the default-conf (0.25) numbers for the same model.

The first ckpt is treated as the baseline; delta columns show the gain (or
loss) for the others.

Colab usage:
    !python scripts/impact_report.py \\
        --ckpts /content/best_baseline.pt /content/best_texgate.pt \\
        --names baseline texgate \\
        --data /content/tb_kfold/fold0/data.yaml \\
        --val-split val --test-split test \\
        --out /content/impact
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


def resolve_image_dir(data_yaml: Path, split: str) -> tuple[Path, Path]:
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", data_yaml.parent))
    split_rel = data.get(split)
    if isinstance(split_rel, list):
        split_rel = split_rel[0]
    img_dir = base / split_rel if (base / split_rel).is_dir() else Path(split_rel)
    label_dir = Path(str(img_dir).replace("images", "labels"))
    return img_dir, label_dir


def find_f1max_conf(model: YOLO, img_paths, label_dir: Path, imgsz: int,
                    iou_thresh: float, device) -> tuple[float, dict]:
    """Return (best_conf, metrics_dict_at_best)."""
    from PIL import Image

    all_scores, all_tp = [], []
    n_gt_total = 0
    for img_path in img_paths:
        W, H = Image.open(img_path).size
        gt_norm = load_labels(label_dir / (img_path.stem + ".txt"))
        gt_xyxy = yolo_to_xyxy(gt_norm, W, H)
        n_gt_total += len(gt_xyxy)
        res = model.predict(
            img_path, imgsz=imgsz, conf=0.001, iou=0.6,
            device=device, verbose=False,
        )[0]
        if len(res.boxes) == 0:
            continue
        pred_xyxy = res.boxes.xyxy.cpu().numpy()
        pred_scores = res.boxes.conf.cpu().numpy()
        tp_mask = greedy_match(pred_xyxy, pred_scores, gt_xyxy, iou_thresh)
        all_scores.extend(pred_scores.tolist())
        all_tp.extend(tp_mask.tolist())
    scores = np.array(all_scores)
    tp_flags = np.array(all_tp, dtype=bool)

    best = None
    for conf in np.linspace(0.05, 0.95, 91):
        mask = scores >= conf
        tp = int(tp_flags[mask].sum())
        fp = int((~tp_flags[mask]).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(n_gt_total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if best is None or f1 > best["F1"]:
            best = {"conf": float(conf), "TP": tp, "FP": fp,
                    "P": precision, "R": recall, "F1": f1, "nGT": n_gt_total}
    return best["conf"], best


def eval_test(ckpt_path: str, data_yaml: str, split: str, imgsz: int,
              conf: float, iou: float, device) -> dict:
    """Run Ultralytics model.val() at the given conf, return mAP/P/R."""
    model = YOLO(ckpt_path)
    r = model.val(data=data_yaml, split=split, imgsz=imgsz,
                  conf=conf, iou=iou, device=device, verbose=False)
    p = float(r.box.mp)
    rec = float(r.box.mr)
    f1 = 2 * p * rec / max(p + rec, 1e-12)
    return {
        "conf": conf,
        "mAP50": float(r.box.map50),
        "mAP50-95": float(r.box.map),
        "P": p,
        "R": rec,
        "F1": f1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="2+ ckpt paths; first is treated as baseline")
    ap.add_argument("--names", nargs="+", default=None,
                    help="Display names (same length as --ckpts). Defaults to ckpt stems.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--test-split", default=None,
                    help="If omitted, uses --val-split for the final eval too "
                         "(in which case you tuned and reported on the same data — note in writeup).")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--default-conf", type=float, default=0.25)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="paper/figs/impact")
    args = ap.parse_args()

    if args.names is None:
        args.names = [Path(c).parent.parent.name or Path(c).stem for c in args.ckpts]
    if len(args.names) != len(args.ckpts):
        raise ValueError("--names must match --ckpts length")
    if len(args.ckpts) < 2:
        print("Pass at least 2 ckpts (baseline + variant).")
        return 1
    test_split = args.test_split or args.val_split

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_img_dir, val_label_dir = resolve_image_dir(Path(args.data), args.val_split)
    val_imgs = sorted(
        list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpeg"))
    )
    print(f"  val split: {len(val_imgs)} images at {val_img_dir}")
    if not val_imgs:
        print("  no images found.")
        return 1

    # Sweep + eval for each ckpt
    print(f"\n  {'='*70}")
    print(f"  Step 1/3: sweep conf on val to find each ckpt's F1-max operating point")
    print(f"  {'='*70}\n")
    tuned_confs = {}
    val_metrics = {}
    for name, ckpt in zip(args.names, args.ckpts):
        print(f"  [{name}] inference + sweep ...")
        model = YOLO(ckpt)
        conf, metrics_val = find_f1max_conf(
            model, val_imgs, val_label_dir, args.imgsz, args.iou_thresh, args.device
        )
        tuned_confs[name] = conf
        val_metrics[name] = metrics_val
        print(f"    val F1-max @ conf={conf:.2f}: "
              f"P={metrics_val['P']:.4f}  R={metrics_val['R']:.4f}  F1={metrics_val['F1']:.4f}")

    # Eval on test at default conf + tuned conf
    print(f"\n  {'='*70}")
    print(f"  Step 2/3: eval each ckpt on {test_split} split")
    print(f"  (a) at default conf={args.default_conf}")
    print(f"  (b) at val-tuned conf")
    print(f"  {'='*70}\n")
    default_results = {}
    tuned_results = {}
    for name, ckpt in zip(args.names, args.ckpts):
        print(f"  [{name}]:")
        default_results[name] = eval_test(
            ckpt, args.data, test_split, args.imgsz,
            args.default_conf, args.iou_thresh, args.device,
        )
        print(f"    default conf={args.default_conf}: "
              f"mAP50={default_results[name]['mAP50']:.4f}  "
              f"P={default_results[name]['P']:.4f}  R={default_results[name]['R']:.4f}  "
              f"F1={default_results[name]['F1']:.4f}")
        tuned_results[name] = eval_test(
            ckpt, args.data, test_split, args.imgsz,
            tuned_confs[name], args.iou_thresh, args.device,
        )
        print(f"    tuned   conf={tuned_confs[name]:.2f}: "
              f"mAP50={tuned_results[name]['mAP50']:.4f}  "
              f"P={tuned_results[name]['P']:.4f}  R={tuned_results[name]['R']:.4f}  "
              f"F1={tuned_results[name]['F1']:.4f}")

    # Step 3: side-by-side table with deltas
    print(f"\n  {'='*70}")
    print(f"  Step 3/3: side-by-side comparison (deltas vs {args.names[0]})")
    print(f"  {'='*70}\n")

    baseline_name = args.names[0]
    headers = ["model", "conf", "mAP50", "Δ mAP50", "mAP50-95", "Δ mAP50-95",
               "P", "R", "F1", "Δ F1"]
    rows = []
    for label, results_dict in [("DEFAULT conf", default_results),
                                ("TUNED   conf", tuned_results)]:
        rows.append([f"--- {label} on {test_split} ---", "", "", "", "", "", "", "", "", ""])
        base = results_dict[baseline_name]
        for name in args.names:
            r = results_dict[name]
            d_m50 = r["mAP50"] - base["mAP50"]
            d_m = r["mAP50-95"] - base["mAP50-95"]
            d_f1 = r["F1"] - base["F1"]
            rows.append([
                name, f"{r['conf']:.2f}",
                f"{r['mAP50']:.4f}", f"{d_m50:+.4f}",
                f"{r['mAP50-95']:.4f}", f"{d_m:+.4f}",
                f"{r['P']:.4f}", f"{r['R']:.4f}", f"{r['F1']:.4f}", f"{d_f1:+.4f}",
            ])

    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))

    # CSV
    csv_path = out_dir / f"impact_{test_split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)
    print(f"\n  saved {csv_path}")

    print(f"\n  Bottom line:")
    bn = args.names[0]
    for n in args.names[1:]:
        dm50_def = default_results[n]["mAP50"] - default_results[bn]["mAP50"]
        dm50_tun = tuned_results[n]["mAP50"] - tuned_results[bn]["mAP50"]
        df1_tun = tuned_results[n]["F1"] - tuned_results[bn]["F1"]
        verdict_mAP = "WIN" if dm50_tun > 0.005 else ("TIE" if abs(dm50_tun) <= 0.005 else "LOSS")
        verdict_F1 = "WIN" if df1_tun > 0.005 else ("TIE" if abs(df1_tun) <= 0.005 else "LOSS")
        print(f"    {n} vs {bn}: "
              f"Δ mAP50 = {dm50_def:+.4f} (default) / {dm50_tun:+.4f} (tuned) → mAP {verdict_mAP}, F1 {verdict_F1}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
