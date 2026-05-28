"""Build the Chen-style train/val/test split (1024/140/101) for Tuberculosis6208.

Pipeline:
1. Optional: extract Tuberculosis6208.zip if not yet extracted.
2. Convert Pascal-VOC XML annotations to YOLO format (class `bacilli`, id 0).
3. Deterministic shuffle with SPLIT_SEED=42 → 1024 / 140 / 101.
4. Copy (or symlink) images + .txt labels into <out>/{train,val,test}/{images,labels}.
5. Write `data.yaml` for Ultralytics.

Matches the split used in the thesis Colab so results are comparable to the
Chen et al. (IJAI 2024) baseline.

Usage:
    python scripts/build_chen_split.py \
        --src /content/Tuberculosis6208/tuberculosis-phonecamera \
        --out /content/tb_chen_split
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image

CLASS_NAME = "bacilli"          # single foreground class in TB6208
CLASS_ID = 0
N_TRAIN, N_VAL, N_TEST = 1024, 140, 101
SPLIT_SEED = 42                 # do not change — must match Chen et al. baseline


def maybe_extract(zip_path: Path, extract_dir: Path, expected: Path) -> None:
    if expected.exists() and any(expected.iterdir()):
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Source not extracted and zip missing: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def voc_box_to_yolo(bnd: ET.Element, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xmin = float(bnd.findtext("xmin"))
    ymin = float(bnd.findtext("ymin"))
    xmax = float(bnd.findtext("xmax"))
    ymax = float(bnd.findtext("ymax"))
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    ww = (xmax - xmin) / img_w
    hh = (ymax - ymin) / img_h
    return xc, yc, ww, hh


def write_yolo_label(xml_path: Path, img_path: Path, out_lbl: Path) -> int:
    """Convert one VOC XML → YOLO .txt. Returns number of objects."""
    with Image.open(img_path) as im:
        w, h = im.size
    lines = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xc, yc, ww, hh = voc_box_to_yolo(bnd, w, h)
        lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.write_text("\n".join(lines))
    return len(lines)


def place(img: Path, lbl_txt: Path, split_dir: Path, use_symlink: bool) -> None:
    dst_img = split_dir / "images" / img.name
    dst_lbl = split_dir / "labels" / f"{img.stem}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        if dst_img.exists():
            dst_img.unlink()
        if dst_lbl.exists():
            dst_lbl.unlink()
        dst_img.symlink_to(img.resolve())
        dst_lbl.symlink_to(lbl_txt.resolve())
    else:
        shutil.copy2(img, dst_img)
        shutil.copy2(lbl_txt, dst_lbl)


def build_split(src: Path, out: Path, use_symlink: bool, fresh: bool) -> Path:
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    pairs = []
    for img in images:
        xml = img.with_suffix(".xml")
        if xml.exists():
            pairs.append((img, xml))
    print(f"Image+XML pairs: {len(pairs)} (target {N_TRAIN + N_VAL + N_TEST})")
    if len(pairs) < N_TRAIN + N_VAL + N_TEST:
        print(f"WARNING: only {len(pairs)} pairs found, ratios will be scaled.")

    # Convert all XML → YOLO once into a staging dir, then we just copy/symlink them.
    stage = out.parent / f"{out.name}_yolo_labels"
    stage.mkdir(parents=True, exist_ok=True)
    img_to_lbl: dict[Path, Path] = {}
    for img, xml in pairs:
        lbl = stage / f"{img.stem}.txt"
        if not lbl.exists():
            write_yolo_label(xml, img, lbl)
        img_to_lbl[img] = lbl

    # Deterministic shuffle with the seed locked to Chen split.
    rng = random.Random(SPLIT_SEED)
    shuffled = [img for img, _ in pairs]
    rng.shuffle(shuffled)

    total = len(shuffled)
    if total >= N_TRAIN + N_VAL + N_TEST:
        n_tr, n_va, n_te = N_TRAIN, N_VAL, N_TEST
    else:
        # proportional fallback
        n_tr = int(total * N_TRAIN / (N_TRAIN + N_VAL + N_TEST))
        n_va = int(total * N_VAL / (N_TRAIN + N_VAL + N_TEST))
        n_te = total - n_tr - n_va

    train = shuffled[:n_tr]
    val = shuffled[n_tr : n_tr + n_va]
    test = shuffled[n_tr + n_va : n_tr + n_va + n_te]
    print(f"Split: train={len(train)}  val={len(val)}  test={len(test)}  seed={SPLIT_SEED}")

    if fresh and out.exists():
        shutil.rmtree(out)
    for sp in ("train", "val", "test"):
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)

    for name, group in (("train", train), ("val", val), ("test", test)):
        for img in group:
            place(img, img_to_lbl[img], out / name, use_symlink)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "# Chen-style split (Chen et al. IJAI 2024) — 1024/140/101\n"
        f"# Split seed: {SPLIT_SEED} (deterministic)\n"
        f"path: {out.resolve()}\n"
        "train: train/images\n"
        "val:   val/images\n"
        "test:  test/images\n"
        "nc: 1\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    print(f"\nWrote {data_yaml}")
    return data_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing TB images + matching XML.")
    ap.add_argument("--out", required=True, help="Output split dir.")
    ap.add_argument("--zip", default=None, help="Optional zip path to extract if --src is missing.")
    ap.add_argument("--extract-dir", default=None, help="Where to extract the zip (parent of --src).")
    ap.add_argument("--symlink", action="store_true", help="Symlink instead of copy.")
    ap.add_argument("--keep", action="store_true", help="Do not wipe --out before building.")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if args.zip and args.extract_dir:
        maybe_extract(Path(args.zip), Path(args.extract_dir), src)

    if not src.exists():
        print(f"ERROR: src does not exist: {src}", file=sys.stderr)
        return 1

    build_split(src, out, use_symlink=args.symlink, fresh=not args.keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
