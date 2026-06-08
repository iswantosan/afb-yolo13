"""Self-training pseudo-label augmentation for AFB detection.

Following Noisy Student paradigm (Xie et al. CVPR 2020) and pseudo-label
refinement literature (STAC 2020, Soft Teacher ICCV 2021):

  Given baseline best.pt as TEACHER (trained on original GT), predict on
  TRAIN images, filter high-confidence predictions far from existing GT,
  treat as pseudo-positives, write augmented label files. Then STUDENT
  is trained on augmented dataset via standard training pipeline.

KEY safety guarantees (for journal defensibility):
  - VAL labels are NEVER touched. Only TRAIN labels are augmented.
  - Conservative confidence threshold (default 0.7).
  - Distance filter prevents double-labeling existing GT.
  - Output dataset is SEPARATE dir (original preserved).
  - All filter thresholds reported in summary JSON.

Paper framing (do NOT claim labels are wrong):
  "We apply pseudo-label refinement as auxiliary training signal. The
   baseline model serves as teacher to generate high-confidence
   pseudo-positives on training images, augmenting original GT for
   student training."

Usage:
    python self_train.py --src-split /content/tb_5fold_seed42/fold_0 \\
        --teacher-ckpt /content/runs/.../fold0_train/weights/best.pt \\
        --out /content/tb_5fold_self_iter1/fold_0 \\
        --conf 0.7 --dist 1.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def predict_pseudo_labels(model_path: Path, img_dir: Path, label_dir: Path,
                          conf_threshold: float, dist_threshold: float,
                          device: str = "0"):
    """Run teacher on images, return filtered pseudo-positives + stats."""
    from ultralytics import YOLO
    teacher = YOLO(str(model_path))

    pseudo_labels: dict[str, list] = {}
    n_total = 0
    n_drop_size = 0
    n_drop_overlap = 0
    n_kept = 0

    img_list = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"  Teacher predicting on {len(img_list)} train images (conf>={conf_threshold})...")

    for idx, img_path in enumerate(img_list):
        if (idx + 1) % 200 == 0:
            print(f"    {idx+1}/{len(img_list)} processed, {n_kept} pseudo-positives kept")

        with Image.open(img_path) as im:
            W, H = im.size

        res = teacher.predict(str(img_path), conf=conf_threshold, iou=0.5,
                              verbose=False, device=device)[0]
        if not len(res.boxes):
            continue

        preds_xyxy = res.boxes.xyxy.cpu().numpy()
        preds_conf = res.boxes.conf.cpu().numpy()
        n_total += len(preds_xyxy)

        # Load existing GT (absolute xyxy / px coords)
        lbl_path = label_dir / (img_path.stem + ".txt")
        gt_centers, gt_diams = [], []
        if lbl_path.exists():
            for ln in lbl_path.read_text().strip().splitlines():
                parts = ln.split()
                if len(parts) >= 5:
                    _, cx, cy, w, h = map(float, parts[:5])
                    gt_centers.append([cx * W, cy * H])
                    gt_diams.append(float(np.sqrt(w * W * h * H)))
        gt_c = np.array(gt_centers) if gt_centers else np.empty((0, 2))
        gt_d = np.array(gt_diams) if gt_diams else np.empty(0)

        kept_for_img = []
        for box, conf in zip(preds_xyxy, preds_conf):
            x1, y1, x2, y2 = box
            pred_w = x2 - x1
            pred_h = y2 - y1
            # Size sanity: not too tiny, not absurdly large
            if pred_w < 5 or pred_h < 5:
                n_drop_size += 1
                continue
            if pred_w > W * 0.5 or pred_h > H * 0.5:
                n_drop_size += 1
                continue
            pred_cx = (x1 + x2) / 2
            pred_cy = (y1 + y2) / 2
            # Distance from nearest GT in box-diameters
            if len(gt_c) > 0:
                d = np.linalg.norm(gt_c - np.array([pred_cx, pred_cy]), axis=1)
                j = int(d.argmin())
                d_norm = float(d[j] / max(gt_d[j], 1.0))
                if d_norm < dist_threshold:
                    n_drop_overlap += 1
                    continue
            # Keep: convert to YOLO normalized
            cx_n = max(0.0, min(1.0, pred_cx / W))
            cy_n = max(0.0, min(1.0, pred_cy / H))
            w_n = max(0.001, min(1.0, pred_w / W))
            h_n = max(0.001, min(1.0, pred_h / H))
            kept_for_img.append((0, cx_n, cy_n, w_n, h_n))  # class 0 = bacilli
            n_kept += 1

        if kept_for_img:
            pseudo_labels[img_path.stem] = kept_for_img

    stats = dict(
        n_images=len(img_list),
        n_predictions_above_conf=n_total,
        n_dropped_size=n_drop_size,
        n_dropped_overlap_with_gt=n_drop_overlap,
        n_kept_pseudo=n_kept,
        n_imgs_with_pseudo=len(pseudo_labels),
    )
    return pseudo_labels, stats


def build_augmented_split(src_split: Path, dst_split: Path,
                          pseudo_labels: dict) -> Path:
    """Replicate src split, augment ONLY TRAIN labels. Val untouched."""
    src = Path(src_split)
    dst = Path(dst_split)
    dst.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        src_img_dir = src / split / "images"
        src_lbl_dir = src / split / "labels"
        if not src_img_dir.is_dir():
            continue
        dst_img_dir = dst / split / "images"
        dst_lbl_dir = dst / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Images: try hardlink (cheap), fallback copy
        for img in src_img_dir.glob("*.jpg"):
            dst_img = dst_img_dir / img.name
            if dst_img.exists():
                continue
            try:
                dst_img.hardlink_to(img)
            except (OSError, AttributeError, NotImplementedError):
                shutil.copy2(img, dst_img)

        # Labels: copy original, AUGMENT only train
        for lbl in src_lbl_dir.glob("*.txt"):
            stem = lbl.stem
            dst_lbl = dst_lbl_dir / lbl.name
            orig_lines = [ln for ln in lbl.read_text().strip().splitlines() if ln.strip()]
            if split == "train" and stem in pseudo_labels:
                pseudo_lines = [
                    f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    for cls, cx, cy, w, h in pseudo_labels[stem]
                ]
                all_lines = orig_lines + pseudo_lines
                dst_lbl.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
            else:
                dst_lbl.write_text(
                    ("\n".join(orig_lines) + "\n") if orig_lines else "",
                    encoding="utf-8",
                )

    yaml_text = (
        "# Augmented split (self-training iter)\n"
        "# train labels = original GT + pseudo-positives\n"
        "# val labels = ORIGINAL untouched (no data leakage)\n"
        f"path: {dst.as_posix()}\n"
        "train: train/images\n"
        "val:   val/images\n"
        "nc: 1\n"
        "names:\n  0: bacilli\n"
    )
    yaml_path = dst / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def count_labels(label_dir: Path):
    """Return (n_imgs_with_labels, n_boxes_total)."""
    n_i, n_b = 0, 0
    for lbl in label_dir.glob("*.txt"):
        lines = [ln for ln in lbl.read_text().strip().splitlines() if ln.strip()]
        if lines:
            n_i += 1
            n_b += len(lines)
    return n_i, n_b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-split", required=True)
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.7,
                    help="confidence threshold (default 0.7, conservative)")
    ap.add_argument("--dist", type=float, default=1.0,
                    help="min distance from existing GT in box-diameters (default 1.0)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    src = Path(args.src_split)
    teacher = Path(args.teacher_ckpt)
    out = Path(args.out)

    if not src.is_dir():
        sys.exit(f"src split not found: {src}")
    if not teacher.exists():
        sys.exit(f"teacher ckpt not found: {teacher}")

    train_img = src / "train" / "images"
    train_lbl = src / "train" / "labels"
    if not train_img.is_dir():
        sys.exit(f"missing train/images in {src}")

    print(f"\n{'=' * 70}")
    print(f"  SELF-TRAINING PSEUDO-LABEL GENERATION")
    print(f"{'=' * 70}")
    print(f"  src split    : {src}")
    print(f"  teacher ckpt : {teacher}")
    print(f"  out          : {out}")
    print(f"  conf >=      : {args.conf}")
    print(f"  dist >=      : {args.dist} box-diameters")

    n_orig_imgs, n_orig_boxes = count_labels(train_lbl)
    print(f"\n  Original train: {n_orig_imgs} imgs with GT, "
          f"{n_orig_boxes} boxes (avg {n_orig_boxes/max(n_orig_imgs,1):.1f}/img)")

    print(f"\nPhase 1: pseudo-label generation")
    pseudo_labels, stats = predict_pseudo_labels(
        teacher, train_img, train_lbl, args.conf, args.dist, args.device
    )

    print(f"\n  Predictions kept             : {stats['n_kept_pseudo']}")
    print(f"  Predictions dropped (size)   : {stats['n_dropped_size']}")
    print(f"  Predictions dropped (overlap): {stats['n_dropped_overlap_with_gt']}")
    print(f"  Imgs with new pseudo-labels  : {stats['n_imgs_with_pseudo']} / {stats['n_images']}")

    ratio = stats["n_kept_pseudo"] / max(n_orig_boxes, 1)
    print(f"\n  Pseudo:GT box ratio = {ratio:.2%}")
    if ratio > 1.0:
        print(f"  [WARN] More pseudo than GT (ratio>{1.0:.0%}). Threshold may be too low.")
    elif ratio < 0.05:
        print(f"  [WARN] Very few pseudo-positives ({ratio:.1%}). Threshold may be too high; "
              "self-training effect may be minimal.")
    else:
        print(f"  [OK] Reasonable pseudo-label volume.")

    print(f"\nPhase 2: writing augmented split to {out}")
    yaml_path = build_augmented_split(src, out, pseudo_labels)

    aug_train_i, aug_train_b = count_labels(out / "train" / "labels")
    aug_val_i, aug_val_b = count_labels(out / "val" / "labels")
    print(f"\n  Augmented train: {aug_train_i} imgs, {aug_train_b} total boxes "
          f"(was {n_orig_boxes}, +{aug_train_b - n_orig_boxes} pseudo)")
    print(f"  Val UNCHANGED  : {aug_val_i} imgs, {aug_val_b} boxes (no data leakage)")
    print(f"\n  data.yaml: {yaml_path}")

    summary = dict(
        teacher_ckpt=str(teacher),
        src_split=str(src),
        out_split=str(out),
        conf_threshold=args.conf,
        dist_threshold=args.dist,
        original_train_boxes=n_orig_boxes,
        augmented_train_boxes=aug_train_b,
        pseudo_to_gt_ratio=ratio,
        **stats,
    )
    (out / "self_train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"  summary saved: {out / 'self_train_summary.json'}")
    print(f"\nNext: train student on this data.yaml via standard training cell.")


if __name__ == "__main__":
    main()
