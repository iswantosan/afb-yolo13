"""Build Chen-style split (1024 / 140 / 101) untuk Tuberculosis6208.

Replikasi logic dari wavelet-yolo12/scripts/build_chen_split.py:
- Pascal-VOC XML -> YOLO .txt (cx, cy, w, h normalized)
- Image dimensions di-read dari JPG (XML Makerere tidak punya <size> tag)
- Deterministic shuffle: np.random.RandomState(seed).permutation pada sorted file list
- Output: <out>/{train,val,test}/{images,labels}/  + data.yaml

Class mapping: TBbacillus -> 0 (single class: "bacilli")
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_MAP = {"TBbacillus": 0}
CLASS_NAMES = ["bacilli"]


def parse_voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert one VOC XML to YOLO lines. Returns list of "cls cx cy w h" strings."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines: list[str] = []
    for obj in root.iter("object"):
        # NOTE: jangan pakai `find("label") or find("name")` -- ElementTree
        # Element punya __bool__ berbasis jumlah child, jadi <label>TBbacillus</label>
        # (no children, text only) evaluasi False. Pakai explicit None check.
        label_el = obj.find("label")
        if label_el is None:
            label_el = obj.find("name")
        if label_el is None or label_el.text is None:
            continue
        label = label_el.text.strip()
        if label not in CLASS_MAP:
            continue
        cls = CLASS_MAP[label]
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))
        # Clamp to image bounds
        xmin = max(0.0, min(img_w, xmin))
        xmax = max(0.0, min(img_w, xmax))
        ymin = max(0.0, min(img_h, ymin))
        ymax = max(0.0, min(img_h, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def get_image_size(img_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as im:
        return im.size  # (W, H)


def maybe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    if not zip_path.exists():
        return
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print(f"Extracted {zip_path} -> {extract_dir}")


def collect_pairs(src_dir: Path) -> list[tuple[Path, Path]]:
    """Find all image+xml pairs. Sorted alphabetically for deterministic baseline."""
    pairs: list[tuple[Path, Path]] = []
    for jpg in sorted(src_dir.glob("*.jpg")):
        xml = jpg.with_suffix(".xml")
        if xml.exists():
            pairs.append((jpg, xml))
    return pairs


def write_split(pairs: list[tuple[Path, Path]], out_split: Path) -> tuple[int, int, int]:
    """Copy images + write YOLO labels for one split.

    Returns (n_pairs, n_total_boxes, n_empty_labels).
    """
    img_out = out_split / "images"
    lbl_out = out_split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_total_boxes = 0
    n_empty = 0
    for jpg, xml in pairs:
        # Copy image
        dst_img = img_out / jpg.name
        if not dst_img.exists():
            shutil.copy2(jpg, dst_img)
        # Convert XML -> YOLO txt
        try:
            W, H = get_image_size(jpg)
        except Exception as e:
            print(f"[warn] cannot open {jpg.name}: {e}", file=sys.stderr)
            continue
        lines = parse_voc_to_yolo(xml, W, H)
        lbl_path = lbl_out / (jpg.stem + ".txt")
        lbl_path.write_text("\n".join(lines), encoding="utf-8")
        n_total_boxes += len(lines)
        if not lines:
            n_empty += 1
    return len(pairs), n_total_boxes, n_empty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="optional zip to extract first")
    ap.add_argument("--extract-dir", default=None, help="where to extract zip")
    ap.add_argument("--src", required=True, help="dir with *.jpg + *.xml pairs")
    ap.add_argument("--out", required=True, help="output split dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-n", type=int, default=1024)
    ap.add_argument("--val-n", type=int, default=140)
    ap.add_argument("--test-n", type=int, default=101)
    args = ap.parse_args()

    # 1. Extract zip if provided
    if args.zip:
        zip_path = Path(args.zip)
        extract_dir = Path(args.extract_dir or "dataset_raw")
        maybe_extract_zip(zip_path, extract_dir)

    # 2. Collect pairs
    src = Path(args.src)
    if not src.is_dir():
        sys.exit(f"src dir not found: {src}")
    pairs = collect_pairs(src)
    target_n = args.train_n + args.val_n + args.test_n
    print(f"Image+XML pairs: {len(pairs)} (target {target_n})")
    if len(pairs) < target_n:
        sys.exit(f"Not enough pairs ({len(pairs)} < {target_n})")

    # 3. Deterministic shuffle
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(pairs))
    pairs_shuffled = [pairs[i] for i in perm]

    train_pairs = pairs_shuffled[: args.train_n]
    val_pairs = pairs_shuffled[args.train_n : args.train_n + args.val_n]
    test_pairs = pairs_shuffled[
        args.train_n + args.val_n : args.train_n + args.val_n + args.test_n
    ]
    print(
        f"Split: train={len(train_pairs)}  "
        f"val={len(val_pairs)}  test={len(test_pairs)}  seed={args.seed}"
    )

    # 4. Write splits
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, ps in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        n_p, n_box, n_emp = write_split(ps, out / name)
        stats[name] = (n_p, n_box, n_emp)

    # Sanity: kalau ada empty labels banyak -> bug (mis. XML tag mismatch).
    print("\nLabels written:")
    for name, (n_p, n_box, n_emp) in stats.items():
        avg = n_box / max(n_p, 1)
        print(f"  {name:5s}: pairs={n_p:5d}  total_boxes={n_box:6d}  "
              f"avg/img={avg:5.1f}  empty_labels={n_emp}")
    if any(stats[s][2] > 0.5 * stats[s][0] for s in stats):
        print("\n[WARN] >50% labels are empty - likely XML parse bug. "
              "Check class names di CLASS_MAP.")
    if any(stats[s][1] == 0 for s in stats):
        sys.exit("\n[FATAL] one or more split has 0 boxes - aborting. "
                 "Fix the XML parse logic and re-run.")

    # 5. data.yaml
    yaml_path = out / "data.yaml"
    yaml_text = (
        f"# Chen-style split (Chen et al. IJAI 2024) - "
        f"{args.train_n}/{args.val_n}/{args.test_n}\n"
        f"# Split seed: {args.seed} (deterministic)\n"
        f"path: {out.as_posix()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
    )
    for i, n in enumerate(CLASS_NAMES):
        yaml_text += f"  {i}: {n}\n"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\nWrote {yaml_path}")

    # 6. Sanity check: print first 5 train + val filenames
    print("\nFirst 5 train images:")
    for p, _ in train_pairs[:5]:
        print(f"  {p.name}")
    print("First 5 val images:")
    for p, _ in val_pairs[:5]:
        print(f"  {p.name}")
    print("First 5 test images:")
    for p, _ in test_pairs[:5]:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
