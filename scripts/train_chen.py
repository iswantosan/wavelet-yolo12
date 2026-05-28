"""Train one model on the Chen-style TB6208 split, log to W&B, eval on test.

Pretty much the thesis training routine, kept end-to-end (no staged freeze)
because end-to-end 60 epochs has consistently beaten 50-freeze + 200-unfreeze
on this dataset/backbone.

Designed to be called multiple times from the Colab notebook so we can sweep
across (model_cfg × seed).
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# Stable SDP kernel on Ampere/Ada — avoid flash/mem-efficient mismatch.
os.environ.setdefault("PYTORCH_SDP_KERNEL", "math")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update({"wandb": False})  # we drive W&B manually, not via Ultralytics callback


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_tag(cfg_path: str) -> str:
    return Path(cfg_path).stem  # e.g. yolov12-wavelet-p3


def train_one(
    model_cfg: str,
    data_yaml: str,
    pretrained: str | None,
    seed: int,
    epochs: int,
    imgsz: int,
    batch: int,
    device: int | str,
    project: str,
    run_name: str | None,
    wandb_project: str | None,
) -> dict:
    """Train one (model, seed) combination, log to W&B, eval on test split."""
    import pandas as pd  # local import — keeps script lightweight when not installed

    use_wandb = bool(wandb_project)
    wandb = None
    if use_wandb:
        import wandb  # noqa: WPS433

    tag = model_tag(model_cfg)
    name = run_name or f"{tag}_seed{seed}_{epochs}ep"

    run = None
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            name=name,
            reinit=True,
            config=dict(
                model_cfg=model_cfg, data_yaml=data_yaml, pretrained=pretrained,
                seed=seed, epochs=epochs, imgsz=imgsz, batch=batch,
                optimizer="SGD", lr0=0.01, momentum=0.937, cos_lr=True,
            ),
            tags=[tag, f"seed{seed}", "chen_split"],
        )
        wandb.define_metric("epoch")
        for k in [
            "train/box_loss", "train/cls_loss", "train/dfl_loss", "train/total_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss", "val/total_loss",
            "val/mAP50", "val/mAP50-95", "val/precision", "val/recall", "lr/pg0",
        ]:
            wandb.define_metric(k, step_metric="epoch")

    set_seeds(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    model = YOLO(model_cfg)
    if pretrained:
        try:
            model.load(pretrained)
        except Exception as e:
            print(f"  [warn] could not load pretrained '{pretrained}': {e}")

    t0 = time.time()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        cos_lr=True,
        patience=0,
        amp=True,
        deterministic=True,
        seed=seed,
        workers=8,
        # augmentation — mirror the thesis settings
        hsv_h=0.1, hsv_s=0.3, hsv_v=0.3,
        degrees=10, translate=0.05, scale=0.3, shear=0.0, perspective=0.0,
        flipud=0.5, fliplr=0.5, mosaic=0.5, mixup=0.3, auto_augment=None,
        # output
        project=project,
        name=f"{name}_train",
        exist_ok=True,
        save=True,
        verbose=True,
        device=device,
    )
    train_secs = time.time() - t0

    # Re-log per-epoch curves into W&B from results.csv
    csv_path = Path(results.save_dir) / "results.csv"
    if use_wandb and csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        col_map = [
            ("train/box_loss", "train/box_loss"),
            ("train/cls_loss", "train/cls_loss"),
            ("train/dfl_loss", "train/dfl_loss"),
            ("val/box_loss", "val/box_loss"),
            ("val/cls_loss", "val/cls_loss"),
            ("val/dfl_loss", "val/dfl_loss"),
            ("metrics/mAP50(B)", "val/mAP50"),
            ("metrics/mAP50-95(B)", "val/mAP50-95"),
            ("metrics/precision(B)", "val/precision"),
            ("metrics/recall(B)", "val/recall"),
            ("lr/pg0", "lr/pg0"),
        ]
        for _, row in df.iterrows():
            try:
                ep = int(row.get("epoch", 0))
            except (ValueError, TypeError):
                continue
            log = {"epoch": ep}
            for src, dst in col_map:
                if src in df.columns:
                    try:
                        log[dst] = float(row[src])
                    except (ValueError, TypeError):
                        pass
            tb, tc, td = log.get("train/box_loss"), log.get("train/cls_loss"), log.get("train/dfl_loss")
            if None not in (tb, tc, td):
                log["train/total_loss"] = tb + tc + td
            vb, vc, vd = log.get("val/box_loss"), log.get("val/cls_loss"), log.get("val/dfl_loss")
            if None not in (vb, vc, vd):
                log["val/total_loss"] = vb + vc + vd
            run.log(log)

    # Test eval on the held-out 101-image test split
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    map50 = map5095 = map_at_09 = precision = recall = float("nan")
    try:
        eval_model = YOLO(str(best_pt))
        eva = eval_model.val(data=data_yaml, split="test", imgsz=imgsz, device=device, verbose=False)
        map50 = float(getattr(eva.box, "map50", float("nan")))
        map5095 = float(getattr(eva.box, "map", float("nan")))
        try:
            precision = float(np.mean(np.atleast_1d(eva.box.p)))
        except Exception:
            pass
        try:
            recall = float(np.mean(np.atleast_1d(eva.box.r)))
        except Exception:
            pass
        try:
            ap_all = eva.box.all_ap
            if ap_all is not None and len(ap_all):
                ap = ap_all.mean(axis=0) if hasattr(ap_all, "ndim") and ap_all.ndim == 2 else ap_all
                if len(ap) >= 9:
                    map_at_09 = float(ap[8])
        except Exception as e:
            print(f"  (mAP@0.9 extract failed: {e})")
    except Exception as e:
        print(f"  [warn] test eval failed: {e}")

    summary = {
        "model": tag,
        "seed": seed,
        "test_mAP50": map50,
        "test_mAP50-95": map5095,
        "test_mAP@0.9": map_at_09,
        "test_precision": precision,
        "test_recall": recall,
        "train_min": train_secs / 60,
        "save_dir": str(results.save_dir),
        "best": str(best_pt),
    }

    if use_wandb and run is not None:
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                run.summary[f"test/{k}"] = v
        for img in Path(results.save_dir).glob("*.png"):
            if any(t in img.stem.lower() for t in ("results", "confusion", "f1_curve", "pr_curve", "p_curve", "r_curve")):
                try:
                    run.log({f"plots/{img.stem}": wandb.Image(str(img))})
                except Exception:
                    pass
        run.finish()

    print(
        f"  {tag} seed{seed}: TEST mAP50={map50:.4f}  mAP50-95={map5095:.4f}  "
        f"mAP@0.9={map_at_09:.4f}  ({train_secs/60:.1f} min)"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Model YAML, e.g. ultralytics/cfg/models/v12/yolov12-wavelet.yaml")
    ap.add_argument("--data", required=True, help="Chen-split data.yaml path")
    ap.add_argument("--pretrained", default=None, help="Optional .pt to warm-start (e.g. yolov12s.pt)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0)
    ap.add_argument("--project", default="runs/wavelet_chen")
    ap.add_argument("--name", default=None)
    ap.add_argument("--wandb-project", default="wavelet_yolo12_chen")
    args = ap.parse_args()

    train_one(
        model_cfg=args.cfg,
        data_yaml=args.data,
        pretrained=args.pretrained,
        seed=args.seed,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        run_name=args.name,
        wandb_project=args.wandb_project,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
