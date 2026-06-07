"""Baseline YOLOv13s training on Tuberculosis6208 (Chen split).

Mirror hyperparam dari yolo12.ipynb supaya apples-to-apples vs YOLOv12s baseline:
- optimizer SGD, lr0=0.01, lrf=0.01, mom=0.937, wd=5e-4, cos_lr=True
- imgsz=640, batch=16, epochs=60, freeze=2, patience=0
- augmentation: hsv 0.1/0.3/0.3, degrees=30, translate=0.05, scale=0.1,
                flipud=0.3, mosaic=0.2, mixup=0.2, close_mosaic=10
- amp=True, deterministic=True, workers=8

Pretrained `yolov13s.pt` auto-download dari iMoonLab releases jika belum ada.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch


YOLOV13_WEIGHT_URLS = {
    "yolov13n": "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt",
    "yolov13s": "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13s.pt",
    "yolov13l": "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt",
    "yolov13x": "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13x.pt",
}


def ensure_pretrained(model_key: str, weights_dir: Path) -> Path:
    """Download pretrained .pt if missing. Returns local path."""
    if model_key not in YOLOV13_WEIGHT_URLS:
        sys.exit(
            f"Unknown model {model_key}. "
            f"Pick one of: {list(YOLOV13_WEIGHT_URLS.keys())}"
        )
    weights_dir.mkdir(parents=True, exist_ok=True)
    pt_path = weights_dir / f"{model_key}.pt"
    if pt_path.exists():
        print(f"Pretrained found: {pt_path}")
        return pt_path
    url = YOLOV13_WEIGHT_URLS[model_key]
    print(f"Downloading {url} -> {pt_path}")
    urlretrieve(url, pt_path)
    print(f"Done.")
    return pt_path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Stable SDP kernel (mirror yolo12.ipynb)
    os.environ["PYTORCH_SDP_KERNEL"] = "math"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml from build_chen_split")
    ap.add_argument("--model", default="yolov13s",
                    choices=list(YOLOV13_WEIGHT_URLS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/afb_yolov13")
    ap.add_argument("--name", default=None)
    ap.add_argument("--weights-dir", default="weights",
                    help="dir untuk simpan yolov13s.pt etc")
    ap.add_argument("--wandb-project", default=None,
                    help="enable W&B logging dgn project name")
    args = ap.parse_args()

    # Validate
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"data.yaml not found: {data_path}")

    # Auto run name
    run_name = args.name or f"{args.model}_seed{args.seed}_{args.epochs}ep"
    print(f"\n=== Run config ===")
    print(f"  model    : {args.model}")
    print(f"  data     : {data_path}")
    print(f"  seed     : {args.seed}")
    print(f"  epochs   : {args.epochs}")
    print(f"  imgsz    : {args.imgsz}")
    print(f"  batch    : {args.batch}")
    print(f"  device   : {args.device}")
    print(f"  run_name : {run_name}")

    # Seed
    set_global_seed(args.seed)

    # Pretrained
    pt_path = ensure_pretrained(args.model, Path(args.weights_dir))

    # Lazy import after seed set
    from ultralytics import YOLO

    # Disable Ultralytics built-in W&B (we'll log manually if needed)
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"wandb": False})
    except Exception:
        pass

    # Optional W&B
    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            reinit=True,
            config=dict(
                model=args.model, data=str(data_path), pretrained=str(pt_path),
                seed=args.seed, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, optimizer="SGD", lr0=0.01, momentum=0.937,
                cos_lr=True, split="chen_1024_140_101", split_seed=42,
            ),
            tags=[args.model, f"seed{args.seed}", "chen_split", "baseline"],
        )
        print(f"W&B run: {run.url}")

    # Build model from YAML + load pretrained
    # YOLOv13 yaml: ultralytics/cfg/models/v13/yolov13.yaml (scale ditentukan dari
    # nama pretrained: yolov13s.pt -> scale 's').
    model_yaml = f"{args.model}.yaml"
    model = YOLO(model_yaml)
    try:
        model.load(str(pt_path))
        print(f"Loaded pretrained: {pt_path}")
    except Exception as e:
        print(f"[warn] could not load pretrained: {e}")

    # Train (mirror yolo12.ipynb hyperparams)
    t0 = time.time()
    results = model.train(
        data=str(data_path),
        freeze=2,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        # optimizer
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        # augmentation - mirror yolo12.ipynb
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
        # control
        patience=0,
        amp=True,
        deterministic=True,
        seed=args.seed,
        workers=8,
        # output
        project=args.project,
        name=f"{run_name}_train",
        exist_ok=True,
        save=True,
        verbose=True,
    )
    train_secs = time.time() - t0
    print(f"\nTrain time: {train_secs/60:.1f} min")
    print(f"Save dir  : {results.save_dir}")

    # === Test eval on 101-image holdout ===
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[warn] best.pt not found at {best_pt}")
        return
    print(f"\nBest ckpt: {best_pt}")
    eval_model = YOLO(str(best_pt))
    eva = eval_model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    map50 = float(eva.box.map50)
    map5095 = float(eva.box.map)
    precision = float(np.mean(np.atleast_1d(eva.box.p)))
    recall = float(np.mean(np.atleast_1d(eva.box.r)))

    # mAP@0.9
    map_at_09 = float("nan")
    try:
        ap_all = eva.box.all_ap
        if ap_all is not None and len(ap_all):
            if hasattr(ap_all, "ndim") and ap_all.ndim == 2:
                ap = ap_all.mean(axis=0)
            else:
                ap = ap_all
            if len(ap) >= 9:
                map_at_09 = float(ap[8])
    except Exception as e:
        print(f"(mAP@0.9 extract failed: {e})")

    print(f"\n=== TEST RESULTS ({run_name}) ===")
    print(f"  mAP50    : {map50:.4f}")
    print(f"  mAP50-95 : {map5095:.4f}")
    print(f"  mAP@0.9  : {map_at_09:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  recall   : {recall:.4f}")
    print(f"  train_min: {train_secs/60:.1f}")

    # W&B summary
    if run is not None:
        run.summary["test/mAP50"] = map50
        run.summary["test/mAP50-95"] = map5095
        run.summary["test/mAP@0.9"] = map_at_09
        run.summary["test/precision"] = precision
        run.summary["test/recall"] = recall
        run.summary["train/time_min"] = train_secs / 60
        # upload key plots
        try:
            import wandb
            for img in Path(results.save_dir).glob("*.png"):
                tag = img.stem.lower()
                if any(t in tag for t in ("results", "confusion",
                                          "f1_curve", "pr_curve",
                                          "p_curve", "r_curve")):
                    run.log({f"plots/{img.stem}": wandb.Image(str(img))})
        except Exception:
            pass
        run.finish()
        print(f"\nW&B run finalised: {run_name}")


if __name__ == "__main__":
    main()
