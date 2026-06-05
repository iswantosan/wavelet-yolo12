"""Derive pseudo-OBB labels for Chen-TB6208 from existing axis-aligned YOLO labels.

Per object:
1. Crop the axis-aligned box from the image (small padding).
2. Pick LAB a* channel (red-green axis — ZN-stained bacilli read pink/red).
3. Otsu threshold + morph close → bacillus mask.
4. Constrain mask to the original GT box (no leak to background).
5. Largest connected component → cv2.minAreaRect → rotated rectangle.
6. Sanity gates:
   - fill fraction in (0.05, 0.95)
   - OBB area ≤ 1.05× axis-aligned box area  (prevents "too round" leak)
   - aspect ratio ≥ 1.3  (bacilli are elongated)
7. If any gate fails → fall back to axis-aligned (angle=0).
8. Write YOLO-OBB label format: ``class x1 y1 x2 y2 x3 y3 x4 y4`` (normalized).

Output: mirrors the input split directory structure under ``--out``, with
images symlinked and new ``labels/`` containing OBB labels. A data.yaml
is written with ``task: obb``.

Usage:
    python scripts/build_obb_split.py \
        --src /content/tb_chen_split \
        --out /content/tb_chen_split_obb
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def derive_obb_corners(img: np.ndarray, box_xyxy: tuple, pad: int = 4) -> tuple[np.ndarray, str]:
    """Return ((4,2) corner array in pixel coords, status_string).

    Falls back to the axis-aligned box if any sanity gate fails. The fallback
    OBB always has 4 right-angle corners coinciding with the axis-aligned box.
    """
    x1, y1, x2, y2 = box_xyxy
    box_area = max((x2 - x1) * (y2 - y1), 1)
    fallback = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )

    H, W = img.shape[:2]
    xa, ya = max(0, x1 - pad), max(0, y1 - pad)
    xb, yb = min(W, x2 + pad), min(H, y2 + pad)
    crop = img[ya:yb, xa:xb]
    if crop.size == 0:
        return fallback, "fallback empty_crop"

    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    a_star = lab[:, :, 1]
    a_norm = cv2.normalize(a_star, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(a_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    box_mask = np.zeros_like(mask)
    box_mask[(y1 - ya) : (y2 - ya), (x1 - xa) : (x2 - xa)] = 255
    mask = cv2.bitwise_and(mask, box_mask)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return fallback, "fallback no_component"
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp_mask = (labels == largest).astype(np.uint8) * 255
    comp_area = int(stats[largest, cv2.CC_STAT_AREA])
    fill = comp_area / box_area
    if fill < 0.05 or fill > 0.95:
        return fallback, f"fallback fill={fill:.2f}"

    ys, xs = np.where(comp_mask > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) < 5:
        return fallback, "fallback too_few_pts"
    rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
    (cx_c, cy_c), (w, h), ang = rect
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect < 1.3:
        return fallback, f"fallback aspect={aspect:.2f}"
    obb_area = w * h
    if obb_area > 1.05 * box_area:
        return fallback, f"fallback area_leak={obb_area/box_area:.2f}"

    # 4 corners in crop coords → translate to original image coords
    corners = cv2.boxPoints(rect)
    corners[:, 0] += xa
    corners[:, 1] += ya
    return corners.astype(np.float32), "ok"


def process_split(split_root: Path, out_root: Path, copy_images: bool) -> dict:
    """Process one split (train/val/test). Returns stats dict."""
    img_dir = split_root / "images"
    lbl_dir = split_root / "labels"
    out_img_dir = out_root / "images"
    out_lbl_dir = out_root / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "objects": 0, "obb_ok": 0, "fallback": 0, "fallback_reasons": {}}
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    for ip in img_paths:
        stats["images"] += 1
        out_ip = out_img_dir / ip.name
        if not out_ip.exists():
            if copy_images:
                shutil.copy2(ip, out_ip)
            else:
                try:
                    out_ip.symlink_to(ip.resolve())
                except OSError:
                    shutil.copy2(ip, out_ip)

        lp = lbl_dir / (ip.stem + ".txt")
        out_lp = out_lbl_dir / (ip.stem + ".txt")
        if not lp.exists() or lp.stat().st_size == 0:
            out_lp.write_text("")  # empty label = background
            continue

        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        gt = np.loadtxt(lp).reshape(-1, 5)

        lines = []
        for c, cx, cy, bw, bh in gt:
            stats["objects"] += 1
            x1 = int(max(0, (cx - bw / 2) * W))
            y1 = int(max(0, (cy - bh / 2) * H))
            x2 = int(min(W, (cx + bw / 2) * W))
            y2 = int(min(H, (cy + bh / 2) * H))
            if x2 - x1 < 6 or y2 - y1 < 6:
                stats["fallback"] += 1
                stats["fallback_reasons"]["too_small"] = stats["fallback_reasons"].get("too_small", 0) + 1
                corners = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
            else:
                corners, status = derive_obb_corners(img, (x1, y1, x2, y2))
                if status == "ok":
                    stats["obb_ok"] += 1
                else:
                    stats["fallback"] += 1
                    reason = status.split(" ", 1)[1] if " " in status else status
                    stats["fallback_reasons"][reason] = stats["fallback_reasons"].get(reason, 0) + 1

            # Normalize to (0,1) — YOLO-OBB format
            norm = corners.copy()
            norm[:, 0] /= W
            norm[:, 1] /= H
            norm = norm.clip(0.0, 1.0)
            flat = " ".join(f"{v:.6f}" for v in norm.flatten().tolist())
            lines.append(f"{int(c)} {flat}")

        out_lp.write_text("\n".join(lines) + ("\n" if lines else ""))

    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root of axis-aligned YOLO split (has train/val/test subdirs)")
    ap.add_argument("--out", required=True, help="Output root for OBB split")
    ap.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    with open(src / "data.yaml") as f:
        src_yaml = yaml.safe_load(f)
    names = src_yaml.get("names", ["bacilli"])

    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        sp = src / split
        if not sp.exists():
            print(f"  skip {split} (not found)")
            continue
        stats = process_split(sp, out / split, copy_images=args.copy_images)
        ok_rate = stats["obb_ok"] / max(stats["objects"], 1)
        print(f"\n[{split}] {stats['images']} images, {stats['objects']} objects")
        print(f"  pseudo-OBB success: {stats['obb_ok']} ({ok_rate:.1%})")
        print(f"  fallback (axis-aligned): {stats['fallback']}")
        for r, c in sorted(stats["fallback_reasons"].items(), key=lambda x: -x[1]):
            print(f"    - {r}: {c}")

    # Write new data.yaml — same paths but with task=obb
    out_yaml = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names if isinstance(names, dict) else {i: n for i, n in enumerate(names)},
        "task": "obb",
    }
    (out / "data.yaml").write_text(yaml.safe_dump(out_yaml, sort_keys=False))
    print(f"\nWrote {out / 'data.yaml'}")
    print(f"Use: model = YOLO('yolov12s-obb-afb.yaml', task='obb')")
    print(f"     model.train(data='{out / 'data.yaml'}', ...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
