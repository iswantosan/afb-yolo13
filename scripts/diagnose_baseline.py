"""Diagnose YOLOv13 baseline untuk inform arsitektur improvement decision.

Outputs (per ckpt):
  1. Per-IoU mAP (0.5, 0.6, 0.7, 0.8, 0.9) - mengukur localization quality.
  2. FP composition - near / close / far / would-be-TP (IoU 0.3-0.5).
  3. Conf sweep -> optimal F1 conf.
  4. FullPAD_Tunnel gate values (HyperACE pathway active or alpha-trap).
  5. HyperACE output magnitude per layer.
  6. Recommendations berdasar pola di atas.

Decision matrix (text-only summary, no matplotlib dependency for portability).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_data_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_split_dirs(data_yaml_path: Path, split: str) -> tuple[Path, Path]:
    """Return (images_dir, labels_dir) for given split."""
    cfg = load_data_yaml(data_yaml_path)
    base = Path(cfg.get("path", data_yaml_path.parent))
    img_rel = cfg.get(split, f"{split}/images")
    img_dir = (base / img_rel).resolve()
    lbl_dir = Path(str(img_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
    return img_dir, lbl_dir


def read_yolo_label(lbl_path: Path) -> np.ndarray:
    """Return GT boxes as (N, 5) [cls, cx, cy, w, h] in normalized coords."""
    if not lbl_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for ln in lbl_path.read_text().strip().splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[:5]])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5))


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) boxes in xyxy. Returns (N, M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a = a[:, None, :]   # (N,1,4)
    b = b[None, :, :]   # (1,M,4)
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def diagnose_predictions(
    model_predict_fn,
    img_paths: list[Path],
    lbl_dir: Path,
    imgsz: int = 640,
    iou_thresholds: list[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict:
    """Iterate images, get preds, compute aggregate metrics."""
    from PIL import Image

    n_gt_total = 0
    n_pred_total = 0
    # For per-IoU mAP we use simple precision-recall style: for each IoU thresh,
    # count TPs/FPs/FNs across all preds (no per-class since nc=1).
    tp_per_iou = {t: 0 for t in iou_thresholds}
    fp_per_iou = {t: 0 for t in iou_thresholds}
    fn_per_iou = {t: 0 for t in iou_thresholds}

    # FP composition: distance from nearest GT in box-diameters
    fp_near = 0   # < 0.5 d
    fp_close = 0  # 0.5 - 2 d
    fp_far = 0    # >= 2 d
    fp_would_be_tp = 0  # IoU in [0.3, 0.5)

    # For conf sweep: collect all (conf, is_tp_at_0.5)
    all_preds = []  # list of (conf, is_tp_iou_0.5)

    for img_path in img_paths:
        with Image.open(img_path) as im:
            W, H = im.size

        # GT in absolute xyxy
        gt_yolo = read_yolo_label(lbl_dir / (img_path.stem + ".txt"))
        if len(gt_yolo):
            cx, cy, gw, gh = gt_yolo[:, 1] * W, gt_yolo[:, 2] * H, gt_yolo[:, 3] * W, gt_yolo[:, 4] * H
            gt_xyxy = np.stack([cx - gw/2, cy - gh/2, cx + gw/2, cy + gh/2], axis=1)
            gt_diam = np.sqrt(gw * gh)
            gt_centers = np.stack([cx, cy], axis=1)
        else:
            gt_xyxy = np.zeros((0, 4))
            gt_diam = np.zeros(0)
            gt_centers = np.zeros((0, 2))

        n_gt_total += len(gt_xyxy)

        # Predict (conf=0.001 to capture full PR curve)
        pred_xyxy, pred_conf = model_predict_fn(img_path, imgsz)
        n_pred_total += len(pred_xyxy)

        # Compute IoU matrix
        if len(pred_xyxy) and len(gt_xyxy):
            iou_mat = box_iou_xyxy(pred_xyxy, gt_xyxy)
            best_iou_per_pred = iou_mat.max(axis=1)
            best_gt_per_pred = iou_mat.argmax(axis=1)
        else:
            iou_mat = np.zeros((len(pred_xyxy), len(gt_xyxy)))
            best_iou_per_pred = np.zeros(len(pred_xyxy))
            best_gt_per_pred = np.zeros(len(pred_xyxy), dtype=int)

        # Greedy match per IoU threshold for TP/FP/FN
        for t in iou_thresholds:
            matched_gt = set()
            for i in np.argsort(-pred_conf):  # high conf first
                if best_iou_per_pred[i] >= t and best_gt_per_pred[i] not in matched_gt:
                    tp_per_iou[t] += 1
                    matched_gt.add(int(best_gt_per_pred[i]))
                else:
                    fp_per_iou[t] += 1
            fn_per_iou[t] += len(gt_xyxy) - len(matched_gt)

        # FP composition at IoU 0.5
        matched_gt_05 = set()
        is_tp_05 = np.zeros(len(pred_xyxy), dtype=bool)
        for i in np.argsort(-pred_conf):
            if best_iou_per_pred[i] >= 0.5 and best_gt_per_pred[i] not in matched_gt_05:
                matched_gt_05.add(int(best_gt_per_pred[i]))
                is_tp_05[i] = True

        for i, (b, c) in enumerate(zip(pred_xyxy, pred_conf)):
            all_preds.append((float(c), bool(is_tp_05[i])))
            if is_tp_05[i]:
                continue
            # FP - bucket by distance to nearest GT
            pc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            if len(gt_centers) == 0:
                fp_far += 1
                continue
            d = np.linalg.norm(gt_centers - pc, axis=1)
            j = d.argmin()
            d_norm = float(d[j] / max(gt_diam[j], 1))
            # would-be-TP check
            if 0.3 <= best_iou_per_pred[i] < 0.5:
                fp_would_be_tp += 1
            if d_norm < 0.5:
                fp_near += 1
            elif d_norm < 2.0:
                fp_close += 1
            else:
                fp_far += 1

    # Aggregate per-IoU precision/recall/AP-like
    map_by_iou = {}
    for t in iou_thresholds:
        tp = tp_per_iou[t]
        fp = fp_per_iou[t]
        fn = fn_per_iou[t]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        # Simple F1 as proxy for AP at this single conf (model.predict default)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        map_by_iou[f"{t:.2f}"] = float(f1)

    # Conf sweep for optimal F1 at IoU=0.5
    all_preds.sort(key=lambda x: -x[0])
    tp_cum = 0
    fp_cum = 0
    best_f1 = 0.0
    best_conf = 0.0
    for k, (conf, is_tp) in enumerate(all_preds, 1):
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        prec = tp_cum / k
        rec = tp_cum / max(n_gt_total, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf

    n_fp = max(fp_near + fp_close + fp_far, 1)
    return dict(
        n_images=len(img_paths),
        n_gt=n_gt_total,
        n_pred=n_pred_total,
        map_by_iou=map_by_iou,
        drop_50_to_70=float(map_by_iou["0.50"] - map_by_iou["0.70"]),
        fp=dict(
            total=fp_near + fp_close + fp_far,
            near_lt_0_5=fp_near,
            close_0_5_to_2=fp_close,
            far_ge_2=fp_far,
            would_be_tp_iou_0_3_to_0_5=fp_would_be_tp,
            pct_near=round(100 * fp_near / n_fp, 1),
            pct_close=round(100 * fp_close / n_fp, 1),
            pct_far=round(100 * fp_far / n_fp, 1),
            pct_would_be_tp=round(100 * fp_would_be_tp / n_fp, 1),
        ),
        conf_sweep=dict(optimal_conf=float(best_conf), optimal_f1=float(best_f1)),
    )


def probe_yolov13_internals(model) -> dict:
    """Probe FullPAD_Tunnel gates + HyperACE output magnitudes."""
    out = dict(fullpad_gates=[], hyperace_magnitudes={})
    try:
        from ultralytics.nn.modules.block import FullPAD_Tunnel, HyperACE
    except Exception as e:
        out["error"] = f"could not import YOLOv13 modules: {e}"
        return out

    m = model.model
    # FullPAD gate
    idx = 0
    for mod in m.modules():
        if isinstance(mod, FullPAD_Tunnel):
            idx += 1
            g = float(mod.gate.detach().cpu().item())
            status = ("ACTIVE" if abs(g) > 0.05 else
                      ("marginal" if abs(g) > 0.01 else "DEAD"))
            out["fullpad_gates"].append(dict(idx=idx, gate=g, status=status))

    # HyperACE output magnitude (with dummy input)
    try:
        hyperace_outs = {}
        handles = []

        def make_hook(name):
            def fn(module, inp, _out):
                hyperace_outs[name] = float(_out.detach().abs().mean().item())
            return fn

        for i, mod in enumerate(m.model):
            if isinstance(mod, HyperACE):
                handles.append(mod.register_forward_hook(make_hook(f"layer{i}")))

        device = next(m.parameters()).device
        dummy = torch.randn(1, 3, 640, 640, device=device)
        m.eval()
        with torch.no_grad():
            _ = m(dummy)
        for h in handles:
            h.remove()
        out["hyperace_magnitudes"] = hyperace_outs
    except Exception as e:
        out["hyperace_error"] = str(e)
    return out


def build_recommendations(diag: dict, internals: dict) -> list[dict]:
    """Produce ranked recommendations based on diagnose stats."""
    recs = []
    fp = diag["fp"]
    drop = diag["drop_50_to_70"]
    map50 = diag["map_by_iou"]["0.50"]
    map70 = diag["map_by_iou"]["0.70"]
    map90 = diag["map_by_iou"]["0.90"]

    # Localization issue
    if drop > 0.10:
        recs.append(dict(
            severity="HIGH", tag="LOSS_LOCALIZATION",
            reason=(f"mAP drops {drop:.3f} from IoU 0.5 -> 0.7. "
                    "Many predictions barely overlap GT. "
                    "Try: NWD loss + box loss reweight, Inner-CIoU."),
        ))

    # FP composition
    if fp["pct_far"] > 50:
        recs.append(dict(
            severity="HIGH", tag="HARD_NEG_FP",
            reason=(f"{fp['pct_far']:.1f}% FPs are far (>=2d) from GT. "
                    "Likely smear/debris hallucination. "
                    "Try: harder neg mining, contrastive aux loss, "
                    "or label-quality probe (mungkin sebenarnya unlabeled GT)."),
        ))
    if fp["pct_would_be_tp"] > 30:
        recs.append(dict(
            severity="MEDIUM", tag="LOCALIZATION_BORDERLINE",
            reason=(f"{fp['pct_would_be_tp']:.1f}% FPs would be TP at IoU 0.3. "
                    "Indicates localization drift. NWD or Shape-IoU."),
        ))
    if fp["pct_near"] > 30:
        recs.append(dict(
            severity="MEDIUM", tag="DUPLICATE_NEAR_FP",
            reason=(f"{fp['pct_near']:.1f}% FPs are very near GT. "
                    "Adjust NMS IoU threshold or use Soft-NMS."),
        ))

    # Ceiling at high IoU
    if map90 < 0.10 and map50 > 0.80:
        recs.append(dict(
            severity="MEDIUM", tag="HIGH_IOU_CEILING",
            reason=(f"mAP@0.5={map50:.3f} but mAP@0.9={map90:.3f}. "
                    "Severely lossy localization. Mask-guided regression or "
                    "explicit edge supervision could help (e.g. Label-HyperYOLO style)."),
        ))

    # FullPAD gates
    dead_gates = [g for g in internals.get("fullpad_gates", []) if g["status"] == "DEAD"]
    if dead_gates:
        recs.append(dict(
            severity="HIGH", tag="FULLPAD_ALPHA_TRAP",
            reason=(f"{len(dead_gates)}/{len(internals['fullpad_gates'])} FullPAD_Tunnel "
                    "gates DEAD (|g|<0.01). HyperACE pathway not contributing. "
                    "Try: increase gate init, replace scalar gate dgn spatial gate, "
                    "atau add aux loss on gate magnitude."),
        ))

    if not recs:
        recs.append(dict(
            severity="LOW", tag="ALL_GOOD_NO_OBVIOUS_BOTTLENECK",
            reason=(f"mAP@0.5={map50:.3f}, drop->0.7 small. "
                    "Untuk gain >0.5 mAP perlu novelty (NWD/SAHI/copy-paste/arch)."),
        ))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="diag_out")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {args.ckpt}")
    from ultralytics import YOLO
    model = YOLO(args.ckpt)
    if torch.cuda.is_available() and str(args.device) != "cpu":
        model.model = model.model.cuda().eval()

    # Image paths
    img_dir, lbl_dir = get_split_dirs(Path(args.data), args.split)
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"Found {len(img_paths)} images in split '{args.split}'")

    # Predict function (conf=0.001 for full sweep)
    def predict_fn(img_path: Path, imgsz: int):
        res = model.predict(
            str(img_path), conf=0.001, iou=0.6,
            imgsz=imgsz, device=args.device, verbose=False,
        )[0]
        if not len(res.boxes):
            return np.zeros((0, 4)), np.zeros(0)
        return (
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
        )

    print(f"Diagnosing predictions ... (this may take a few minutes)")
    diag = diagnose_predictions(predict_fn, img_paths, lbl_dir, imgsz=args.imgsz)
    print(f"Probing YOLOv13 internals (FullPAD gates + HyperACE) ...")
    internals = probe_yolov13_internals(model)
    recs = build_recommendations(diag, internals)

    # === Print summary ===
    print("\n" + "=" * 64)
    print(f"  DIAGNOSE - {Path(args.ckpt).name} ({args.split} split)")
    print("=" * 64)
    print(f"  n_images           : {diag['n_images']}")
    print(f"  n_GT / n_pred      : {diag['n_gt']} / {diag['n_pred']}")
    print(f"  F1 by IoU thresh   :")
    for k, v in diag["map_by_iou"].items():
        print(f"    @IoU={k}        : {v:.4f}")
    print(f"  drop 0.5 -> 0.7    : {diag['drop_50_to_70']:.4f}")
    fp = diag["fp"]
    print(f"  FP composition (total={fp['total']}):")
    print(f"    near (<0.5d)     : {fp['near_lt_0_5']:5d} ({fp['pct_near']}%)")
    print(f"    close (0.5-2d)   : {fp['close_0_5_to_2']:5d} ({fp['pct_close']}%)")
    print(f"    far (>=2d)       : {fp['far_ge_2']:5d} ({fp['pct_far']}%)")
    print(f"    would-be-TP      : {fp['would_be_tp_iou_0_3_to_0_5']:5d} ({fp['pct_would_be_tp']}%)")
    print(f"  Conf optimal       : {diag['conf_sweep']['optimal_conf']:.3f} "
          f"(F1={diag['conf_sweep']['optimal_f1']:.4f})")

    print(f"\n  FullPAD_Tunnel gates:")
    for g in internals.get("fullpad_gates", []):
        print(f"    #{g['idx']:2d}  gate={g['gate']:+.6f}  [{g['status']}]")
    if "hyperace_magnitudes" in internals and internals["hyperace_magnitudes"]:
        print(f"  HyperACE |output|_mean:")
        for k, v in internals["hyperace_magnitudes"].items():
            print(f"    {k}: {v:.4e}")

    print(f"\n  RECOMMENDATIONS:")
    for r in recs:
        print(f"    [{r['severity']:>6}] {r['tag']}: {r['reason']}")

    # === Save JSON ===
    json_path = out_dir / "diagnose.json"
    json_payload = dict(diag=diag, internals=internals, recommendations=recs,
                        ckpt=str(args.ckpt), split=args.split, imgsz=args.imgsz)
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
