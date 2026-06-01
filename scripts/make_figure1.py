"""Generate Figure 1 for the wavelet-YOLO paper.

Renders multiple sputum smear microscopy images side-by-side with overlaid
ground-truth bounding boxes parsed from the Pascal VOC XML annotations.

Usage:
    python scripts/make_figure1.py                       # default 3 samples
    python scripts/make_figure1.py 100 500 888           # custom IDs
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("D:/project/yolov12/Tuberculosis6208/tuberculosis-phonecamera")
OUT = Path(__file__).resolve().parents[1] / "paper" / "figure1_dataset_sample.png"
DEFAULT_IDS = [250, 700, 888]


def img_for(idx: int) -> Path:
    return ROOT / f"tuberculosis-phone-{idx:04d}.jpg"


def parse_boxes(xml_path: Path) -> list[tuple[int, int, int, int]]:
    if not xml_path.exists():
        return []
    root = ET.parse(xml_path).getroot()
    out: list[tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
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


def main() -> None:
    ids = [int(x) for x in sys.argv[1:]] or DEFAULT_IDS

    fig, axes = plt.subplots(1, len(ids), figsize=(5 * len(ids), 4), dpi=200)
    if len(ids) == 1:
        axes = [axes]

    panel_letters = "abcdefghij"
    for ax, idx, letter in zip(axes, ids, panel_letters):
        img_path = img_for(idx)
        xml_path = img_path.with_suffix(".xml")
        if not img_path.exists():
            print(f"[warn] missing image for id={idx}: {img_path}")
            ax.set_visible(False)
            continue

        img = Image.open(img_path).convert("RGB")
        boxes = parse_boxes(xml_path)
        print(
            f"({letter})  id={idx:04d}  size={img.size[0]}x{img.size[1]}  "
            f"bacilli={len(boxes)}"
        )

        ax.imshow(img)
        for xmin, ymin, xmax, ymax in boxes:
            ax.add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    linewidth=1.6,
                    edgecolor="#ff2d2d",
                    facecolor="none",
                )
            )
        ax.set_title(f"({letter})", fontsize=12, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    plt.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
