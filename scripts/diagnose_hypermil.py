"""Diagnose HyperMIL after training: visualize attention maps + count vs GT.

For a trained HyperMIL checkpoint, this script:
  1. Loads the model, re-installs HyperMIL (head weights stored in ckpt).
  2. Iterates val set, captures per-image (predicted_count, GT_count, attn_map).
  3. Visualizes top-K attention maps overlaid on images for qualitative
     verification that MIL learned to focus on bacilli regions.
  4. Reports count correlation (pearson r) and MAE between predicted and GT.

Usage:
    python diagnose_hypermil.py --ckpt path/to/best.pt --data data.yaml \\
        --split val --out diag_hypermil_val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="diag_hypermil")
    ap.add_argument("--n-viz", type=int, default=12,
                    help="number of high-attention examples to visualize")
    ap.add_argument("--mil-weight", type=float, default=0.5)
    ap.add_argument("--mil-hidden", type=int, default=128)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.ckpt}")
    from ultralytics import YOLO
    yolo = YOLO(args.ckpt)
    model = yolo.model
    if torch.cuda.is_available() and str(args.device) != "cpu":
        model = model.cuda().eval()

    # Re-install HyperMIL to attach head from state_dict + hook
    # (Ultralytics restores model.mil_head state from ckpt automatically if
    # the head was registered when ckpt was saved.)
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from afb_yolov13 import install_hypermil
    install_hypermil(model, mil_weight=args.mil_weight, mil_hidden=args.mil_hidden)

    # Load split images + GT labels
    import yaml
    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    base = Path(data_cfg.get("path", Path(args.data).parent))
    img_dir = (base / data_cfg.get(args.split, f"{args.split}/images")).resolve()
    lbl_dir = Path(str(img_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"Found {len(img_paths)} images in split '{args.split}'")

    from PIL import Image

    pred_counts = []
    gt_counts = []
    attn_records = []  # (img_path, gt_count, pred_count, attn[H,W])

    model.eval()
    for img_path in img_paths:
        # Load + preprocess to 640x640 (same as training preprocess)
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB").resize((args.imgsz, args.imgsz))
        x = torch.from_numpy(np.asarray(im_rgb, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available() and str(args.device) != "cpu":
            x = x.cuda()

        # Forward to populate _hyperace_out via hook
        with torch.no_grad():
            _ = model(x)
        feat = model._hyperace_out
        if feat is None:
            continue
        with torch.no_grad():
            count, attn = model.mil_head(feat)

        # attn is [B, N] flat; reshape to [H, W]
        _, _, H, W = feat.shape
        attn_map = attn[0].view(H, W).cpu().numpy()

        # GT count from YOLO label file
        lbl = lbl_dir / (img_path.stem + ".txt")
        n_gt = 0
        if lbl.exists():
            for ln in lbl.read_text().strip().splitlines():
                if ln.strip():
                    n_gt += 1

        pred_counts.append(float(count[0].item()))
        gt_counts.append(n_gt)
        attn_records.append((img_path, n_gt, float(count[0].item()), attn_map))

    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    n = len(pred_counts)

    if n == 0:
        sys.exit("No predictions; abort.")

    # Stats
    mae = float(np.mean(np.abs(pred_counts - gt_counts)))
    bias = float(np.mean(pred_counts - gt_counts))
    if pred_counts.std() > 1e-6 and gt_counts.std() > 1e-6:
        r = float(np.corrcoef(pred_counts, gt_counts)[0, 1])
    else:
        r = float("nan")

    print("\n=== HyperMIL count summary ===")
    print(f"  n_images     : {n}")
    print(f"  GT count     : mean={gt_counts.mean():.2f} std={gt_counts.std():.2f} "
          f"min={gt_counts.min()} max={gt_counts.max()}")
    print(f"  Pred count   : mean={pred_counts.mean():.2f} std={pred_counts.std():.2f} "
          f"min={pred_counts.min():.2f} max={pred_counts.max():.2f}")
    print(f"  MAE          : {mae:.3f}")
    print(f"  Bias (P - GT): {bias:+.3f}")
    print(f"  Pearson r    : {r:.4f}")

    # Visualize top-K
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping visualization")
        return

    # Pick: highest-count images (likely most bacilli)
    sorted_records = sorted(attn_records, key=lambda x: -x[1])[: args.n_viz]
    cols = 4
    rows = (args.n_viz + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (img_path, n_gt, pred, attn) in enumerate(sorted_records):
        with Image.open(img_path) as im:
            im_rgb = np.asarray(im.convert("RGB").resize((args.imgsz, args.imgsz)))
        ax_img = axes[2 * i]
        ax_atn = axes[2 * i + 1]
        ax_img.imshow(im_rgb)
        ax_img.set_title(f"{img_path.name}\nGT={n_gt}", fontsize=8)
        ax_img.axis("off")
        # Upsample attention to image size
        attn_up = np.array(Image.fromarray(attn).resize((args.imgsz, args.imgsz),
                                                       Image.BILINEAR))
        ax_atn.imshow(im_rgb)
        ax_atn.imshow(attn_up, cmap="hot", alpha=0.5)
        ax_atn.set_title(f"pred={pred:.1f}", fontsize=8)
        ax_atn.axis("off")
    for ax in axes[2 * len(sorted_records):]:
        ax.axis("off")
    plt.tight_layout()
    viz_path = out_dir / "mil_attention.png"
    plt.savefig(viz_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved attention viz: {viz_path}")

    # Save scatter plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt_counts, pred_counts, alpha=0.5)
    lim = max(gt_counts.max(), pred_counts.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel("GT bacilli count")
    ax.set_ylabel("MIL predicted count")
    ax.set_title(f"MIL count vs GT  (MAE={mae:.2f}, r={r:.3f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    scatter_path = out_dir / "mil_count_scatter.png"
    plt.savefig(scatter_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot : {scatter_path}")

    # Save JSON summary
    import json
    summary = dict(
        n_images=n, mae=mae, bias=bias, pearson_r=r,
        gt_count_mean=float(gt_counts.mean()), gt_count_std=float(gt_counts.std()),
        pred_count_mean=float(pred_counts.mean()), pred_count_std=float(pred_counts.std()),
        ckpt=str(args.ckpt), split=args.split, imgsz=args.imgsz,
    )
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary JSON : {json_path}")


if __name__ == "__main__":
    main()
