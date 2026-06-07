"""Apply AFB-YOLOv13 custom module patches to a YOLOv13 fork.

Copies 3 modified files into the YOLOv13 source tree to enable L3 variants:
  - ultralytics/nn/modules/block.py     (adds RodDSBottleneck, RodDSC3k2,
                                         SpatialFullPAD_Tunnel,
                                         ScaleAwareFuseModule, HyperACEScale)
  - ultralytics/nn/modules/__init__.py  (registers new classes)
  - ultralytics/nn/tasks.py             (parse_model handling for new classes)

Idempotent: skips files that already match the patched version. Backs up
originals to *.orig on first patch.

Usage (CLI):
    python apply_yolov13_patches.py /path/to/yolov13

Usage (Python):
    from scripts.apply_yolov13_patches import apply
    apply(Path('/content/yolov13'))
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


PATCHES_DIR = Path(__file__).resolve().parent.parent / "yolov13_patches"

# (target_rel_path, patch_src_filename)
PATCH_MAP = [
    ("ultralytics/nn/modules/block.py",     "block.py"),
    ("ultralytics/nn/modules/__init__.py",  "modules_init.py"),
    ("ultralytics/nn/tasks.py",             "tasks.py"),
    ("ultralytics/utils/loss.py",           "loss.py"),         # NWD loss
    ("ultralytics/cfg/default.yaml",        "default.yaml"),    # nwd_ratio/nwd_c defaults
    ("ultralytics/cfg/__init__.py",         "cfg_init.py"),     # validator keys
]


def apply(yolov13_dir: Path) -> int:
    """Apply patches. Returns number of files actually patched."""
    yolov13_dir = Path(yolov13_dir).resolve()
    if not yolov13_dir.is_dir():
        print(f"[FAIL] yolov13 dir not found: {yolov13_dir}", file=sys.stderr)
        return -1
    if not PATCHES_DIR.is_dir():
        print(f"[FAIL] patches dir not found: {PATCHES_DIR}", file=sys.stderr)
        return -1

    print(f"Applying AFB-YOLOv13 patches to: {yolov13_dir}")
    n_patched = 0
    for rel, src_name in PATCH_MAP:
        dst = yolov13_dir / rel
        src = PATCHES_DIR / src_name
        if not src.exists():
            print(f"  [FAIL] patch source missing: {src}")
            continue
        if not dst.exists():
            print(f"  [FAIL] target not found: {dst}")
            continue
        if dst.read_bytes() == src.read_bytes():
            print(f"  [skip] {rel} already patched")
            continue
        # Backup once
        backup = dst.with_suffix(dst.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(dst, backup)
            print(f"  [backup] {rel} -> {backup.name}")
        shutil.copy2(src, dst)
        print(f"  [done] patched {rel}")
        n_patched += 1

    if n_patched == 0:
        print("\nAll files already patched. Nothing to do.")
    else:
        print(f"\nPatched {n_patched} file(s). Reinstall YOLOv13:")
        print(f"    pip install -e {yolov13_dir}")
    return n_patched


def revert(yolov13_dir: Path) -> int:
    """Revert all patched files from *.orig backups."""
    yolov13_dir = Path(yolov13_dir).resolve()
    print(f"Reverting AFB-YOLOv13 patches in: {yolov13_dir}")
    n_reverted = 0
    for rel, _ in PATCH_MAP:
        dst = yolov13_dir / rel
        backup = dst.with_suffix(dst.suffix + ".orig")
        if not backup.exists():
            print(f"  [skip] {rel} (no backup)")
            continue
        shutil.copy2(backup, dst)
        backup.unlink()
        print(f"  [done] reverted {rel}")
        n_reverted += 1
    return n_reverted


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("yolov13_dir", help="path to YOLOv13 source (e.g. /content/yolov13)")
    ap.add_argument("--revert", action="store_true",
                    help="revert from *.orig backups instead of applying")
    args = ap.parse_args()
    if args.revert:
        revert(Path(args.yolov13_dir))
    else:
        rc = apply(Path(args.yolov13_dir))
        if rc < 0:
            sys.exit(1)
