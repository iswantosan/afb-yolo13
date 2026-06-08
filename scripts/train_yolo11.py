"""YOLOv11 + AFB architecture ablation runner on Tuberculosis6208 (Chen split).

Trains one or more YOLO11s variants with identical hyperparameters mirroring
the v13 baseline (so v11 results are apples-to-apples vs v13 numbers).

Variants supported (file basename in configs/):
  - baseline  -> yolo11s.yaml (stock, no custom YAML needed)
  - p2        -> yolo11s-p2.yaml      (add P2 head, keep P5)
  - pc        -> yolo11s-pc.yaml      (PC-YOLO11s: P2 + noP5 + CSA)
  - spd       -> yolo11s-spd.yaml     (SPDConv backbone, Sunkara 2022)
  - wtconv    -> yolo11s-wtconv.yaml  (Haar wavelet, MS-YOLOv11 / ECCV 2024)
  - csa       -> yolo11s-csa.yaml     (CSA-only ablation)

Pretrained: yolo11s.pt (Ultralytics auto-download).

Usage (single):
  python scripts/train_yolo11.py --data .../data.yaml --variant pc

Usage (sweep all variants sequentially):
  python scripts/train_yolo11.py --data .../data.yaml --variant all

Hyperparameters match scripts/train_baseline.py for direct comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


VARIANT_TO_YAML = {
    "baseline": "yolo11s.yaml",                         # stock Ultralytics
    "p2":       "configs/yolo11s-p2.yaml",
    "pc":       "configs/yolo11s-pc.yaml",
    "spd":      "configs/yolo11s-spd.yaml",
    "wtconv":   "configs/yolo11s-wtconv.yaml",
    "csa":      "configs/yolo11s-csa.yaml",
}

DEFAULT_PRETRAINED = "yolo11s.pt"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTORCH_SDP_KERNEL"] = "math"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def resolve_yaml(variant: str, repo_root: Path) -> str:
    rel = VARIANT_TO_YAML.get(variant)
    if rel is None:
        sys.exit(f"Unknown variant '{variant}'. Choices: {list(VARIANT_TO_YAML)}")
    if variant == "baseline":
        return rel  # Ultralytics resolves stock yolo11s.yaml internally
    p = repo_root / rel
    if not p.exists():
        sys.exit(f"Config not found: {p}")
    return str(p)


def train_one(
    variant: str,
    data_path: Path,
    repo_root: Path,
    *,
    seed: int,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    pretrained: str,
    nwd_ratio: float = 0.0,
    wiou_ratio: float = 0.0,
) -> dict:
    """Train a single variant, return {variant, map50, map5095, run_dir}."""

    model_yaml = resolve_yaml(variant, repo_root)
    run_name = f"yolo11s-{variant}_seed{seed}_{epochs}ep"
    print(f"\n========== Variant: {variant} ==========")
    print(f"  model_yaml = {model_yaml}")
    print(f"  pretrained = {pretrained}")
    print(f"  run_name   = {run_name}")

    set_global_seed(seed)

    from ultralytics import YOLO

    # Quiet built-in W&B
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"wandb": False})
    except Exception:
        pass

    # Build model from YAML, then load pretrained weights (partial-load OK)
    model = YOLO(model_yaml)
    try:
        model.load(pretrained)  # Ultralytics auto-downloads yolo11s.pt
        print(f"Loaded pretrained: {pretrained}")
    except Exception as e:
        print(f"[warn] could not load pretrained: {e}")

    t0 = time.time()
    model.train(
        data=str(data_path),
        freeze=2,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.1,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=30,
        translate=0.05,
        scale=0.1,
        flipud=0.3,
        mosaic=0.2,
        mixup=0.2,
        patience=0,
        amp=True,
        deterministic=True,
        seed=seed,
        workers=8,
        project=project,
        name=f"{run_name}_train",
        exist_ok=True,
        save=True,
        verbose=True,
        nwd_ratio=nwd_ratio,
        wiou_ratio=wiou_ratio,
    )
    train_dt = time.time() - t0

    # Validate on val split, capture mAP50/mAP50-95
    metrics = model.val(
        data=str(data_path),
        imgsz=imgsz,
        batch=batch,
        device=device,
        split="val",
        project=project,
        name=f"{run_name}_val",
        exist_ok=True,
        verbose=False,
    )
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    run_dir = Path(project) / f"{run_name}_train"
    print(f"[done] {variant}: val mAP50={map50:.4f}  mAP50-95={map5095:.4f}  ({train_dt:.0f}s)")

    return {
        "variant": variant,
        "model_yaml": model_yaml,
        "val_map50": map50,
        "val_map5095": map5095,
        "train_time_sec": train_dt,
        "run_dir": str(run_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", required=True,
                    help="single variant name OR 'all' for sweep")
    ap.add_argument("--repo-root", default=".",
                    help="repo root for resolving configs/ (default: cwd)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/afb_yolo11")
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                    help="pretrained weights path/name (auto-download)")
    ap.add_argument("--nwd-ratio", type=float, default=0.0,
                    help="NWD loss mix (0=CIoU only). Universal across YOLO families.")
    ap.add_argument("--wiou-ratio", type=float, default=0.0)
    ap.add_argument("--summary-out", default="runs/afb_yolo11/ablation_summary.json")
    args = ap.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        sys.exit(f"data.yaml not found: {data_path}")

    repo_root = Path(args.repo_root).resolve()
    if args.variant == "all":
        variants = list(VARIANT_TO_YAML.keys())
    else:
        if args.variant not in VARIANT_TO_YAML:
            sys.exit(f"Unknown variant '{args.variant}'. "
                     f"Choices: {list(VARIANT_TO_YAML)} or 'all'")
        variants = [args.variant]

    results = []
    for v in variants:
        try:
            r = train_one(
                variant=v,
                data_path=data_path,
                repo_root=repo_root,
                seed=args.seed,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=args.project,
                pretrained=args.pretrained,
                nwd_ratio=args.nwd_ratio,
                wiou_ratio=args.wiou_ratio,
            )
            results.append(r)
        except Exception as e:
            import traceback
            print(f"[ERROR] variant {v} failed: {e}")
            traceback.print_exc()
            results.append({"variant": v, "error": str(e)})

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Variant':<12} {'mAP50':>8} {'mAP50-95':>10} {'time(s)':>10}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<12} ERROR: {r['error'][:40]}")
        else:
            print(f"{r['variant']:<12} {r['val_map50']:>8.4f} "
                  f"{r['val_map5095']:>10.4f} {r['train_time_sec']:>10.0f}")
    print("=" * 60)

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written to: {out}")


if __name__ == "__main__":
    main()
