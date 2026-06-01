"""Generate Figure 2 for the wavelet-YOLO paper.

Applies the Haar 2D DWT (equivalent to ultralytics/nn/modules/wavelet.py
HaarDWT) to a real bacilli microscopy image and renders the input alongside
the four sub-band outputs (LL, LH, HL, HH) as a 1x5 panel grid.

The numpy implementation here is mathematically identical to the depthwise
stride-2 conv used in the PyTorch HaarDWT module: both apply the same four
2x2 Haar kernels scaled by 1/2.

Usage:
    python scripts/make_figure2.py                  # default image 0888
    python scripts/make_figure2.py 12               # custom ID
    python scripts/make_figure2.py 888 --crop       # crop around bacilli
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("D:/project/yolov12/Tuberculosis6208/tuberculosis-phonecamera")
OUT = REPO_ROOT / "paper" / "figure2_subbands.png"


def img_for(idx: int) -> Path:
    return ROOT / f"tuberculosis-phone-{idx:04d}.jpg"


def parse_boxes(xml_path: Path) -> list[tuple[int, int, int, int]]:
    if not xml_path.exists():
        return []
    out: list[tuple[int, int, int, int]] = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        out.append(
            (
                int(float(bnd.findtext("xmin"))),
                int(float(bnd.findtext("ymin"))),
                int(float(bnd.findtext("xmax"))),
                int(float(bnd.findtext("ymax"))),
            )
        )
    return out


def bbox_crop(img: Image.Image, boxes, pad: int = 60, aspect: float = 4 / 3) -> Image.Image:
    """Crop region tightly around all bounding boxes plus padding, expanded to a
    target aspect ratio (width / height). Default 4:3 matches the original image."""
    if not boxes:
        return img
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    x0, x1 = max(0, min(xs) - pad), min(img.width, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(img.height, max(ys) + pad)

    # Expand to target aspect ratio without exceeding image bounds.
    w = x1 - x0
    h = y1 - y0
    if w / h < aspect:
        # too tall — widen
        target_w = int(h * aspect)
        extra = target_w - w
        x0 = max(0, x0 - extra // 2)
        x1 = min(img.width, x0 + target_w)
        x0 = max(0, x1 - target_w)
    else:
        # too wide — heighten
        target_h = int(w / aspect)
        extra = target_h - h
        y0 = max(0, y0 - extra // 2)
        y1 = min(img.height, y0 + target_h)
        y0 = max(0, y1 - target_h)
    return img.crop((x0, y0, x1, y1))


def haar_2d_dwt(arr: np.ndarray) -> tuple[np.ndarray, ...]:
    """Apply Haar 2D DWT identical to the HaarDWT module (per-channel stride-2 conv).

    For 2x2 block [[a,b],[c,d]] and the four Haar kernels (each scaled by 1/2):
        LL = (a + b + c + d) / 2     (average / low-pass)
        LH = (a + b - c - d) / 2     (horizontal edge / vertical diff)
        HL = (a - b + c - d) / 2     (vertical edge / horizontal diff)
        HH = (a - b - c + d) / 2     (diagonal edge)
    """
    # ensure even dimensions for stride-2
    h, w = arr.shape[-2:]
    h -= h % 2
    w -= w % 2
    arr = arr[..., :h, :w]

    a = arr[..., 0::2, 0::2]  # top-left of each 2x2 block
    b = arr[..., 0::2, 1::2]  # top-right
    c = arr[..., 1::2, 0::2]  # bottom-left
    d = arr[..., 1::2, 1::2]  # bottom-right
    LL = (a + b + c + d) / 2.0
    LH = (a + b - c - d) / 2.0
    HL = (a - b + c - d) / 2.0
    HH = (a - b - c + d) / 2.0
    return LL, LH, HL, HH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", nargs="?", type=int, default=888)
    ap.add_argument("--crop", action="store_true", help="crop tightly around bbox region")
    args = ap.parse_args()

    img_path = img_for(args.idx)
    xml_path = img_path.with_suffix(".xml")
    if not img_path.exists():
        raise SystemExit(f"image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    boxes = parse_boxes(xml_path)
    print(f"Image  : {img_path.name}  ({img.size[0]} x {img.size[1]})")
    print(f"Bacilli: {len(boxes)} bounding boxes")

    if args.crop:
        img = bbox_crop(img, boxes, pad=60)
        print(f"Cropped: {img.size[0]} x {img.size[1]}")

    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    LL, LH, HL, HH = haar_2d_dwt(gray)

    labels = [
        ("(a) Input", img, None),
        ("(b) $S_{LL}$  (approximation)", LL, "gray"),
        ("(c) $S_{LH}$  (horizontal edges)", LH, "RdBu_r"),
        ("(d) $S_{HL}$  (vertical edges)", HL, "RdBu_r"),
        ("(e) $S_{HH}$  (diagonal edges)", HH, "RdBu_r"),
    ]

    # Match Figure 1's per-panel figsize: 5 wide x 4 tall
    fig, axes = plt.subplots(1, 5, figsize=(5 * 5, 4), dpi=200)
    for ax, (title, content, cmap) in zip(axes, labels):
        if isinstance(content, Image.Image):
            ax.imshow(content)
        elif cmap == "gray":
            ax.imshow(content, cmap="gray")
        else:
            vmax = float(np.abs(content).max())
            ax.imshow(content, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    plt.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
