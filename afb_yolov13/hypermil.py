"""HyperMIL — Hypergraph-backed Multiple Instance Learning aux head for YOLOv13.

Adds an image-level bacilli count prediction head reading from HyperACE output
(layer 9 of YOLOv13). The MIL count loss is added to the base detection loss as
a regularizer that exploits image-level supervision robustness against missing
instance-level annotations.

All hook and class machinery is module-level / pickleable so that
torch.save(model) (used by Ultralytics for ckpt snapshots) succeeds.

Usage:
    >>> from ultralytics import YOLO
    >>> from afb_yolov13 import make_hypermil_callback
    >>> model = YOLO('yolov13s.yaml')
    >>> model.load('yolov13s.pt')
    >>> model.add_callback('on_pretrain_routine_start',
    ...                    make_hypermil_callback(mil_weight=0.5))
    >>> model.train(data=...)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Modules
# ============================================================================

class AttnMILPool(nn.Module):
    """Gated attention pooling (Ilse et al. 2018) over spatial features."""

    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.attn_V = nn.Linear(channels, hidden)
        self.attn_U = nn.Linear(channels, hidden)
        self.attn_w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)        # [B, N, C], N = H*W
        u = torch.sigmoid(self.attn_U(z))
        v = torch.tanh(self.attn_V(z))
        a = self.attn_w(u * v).squeeze(-1)       # [B, N]
        a = F.softmax(a, dim=1)
        pooled = (a.unsqueeze(2) * z).sum(dim=1) # [B, C]
        return pooled, a


class HyperMILHead(nn.Module):
    """MIL count prediction: AttnMILPool + MLP -> non-negative count."""

    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.pool = AttnMILPool(channels, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, attn = self.pool(x)
        raw = self.mlp(pooled).squeeze(-1)       # [B]
        count = F.softplus(raw)                  # non-negative
        return count, attn


# ============================================================================
# Module-level state (pickle-safe -- not stored on model instance)
# ============================================================================

# Maps id(HyperACE module) -> last forward output. Hooks write here, loss
# reads here. Module-level so it does NOT get pickled inside the model state
# (each new Python process gets a fresh dict).
_HYPERACE_OUTPUTS: dict[int, torch.Tensor] = {}


def _hypermil_capture_hook(module, inputs, output):
    """Top-level forward hook (pickleable). Stores output keyed by module id.

    Top-level (not a closure) so that nn.Module._forward_hooks can be pickled
    when Ultralytics calls torch.save(model) for checkpointing.
    """
    _HYPERACE_OUTPUTS[id(module)] = output


# ============================================================================
# Helpers
# ============================================================================

def _find_hyperace_layer(detection_model):
    try:
        from ultralytics.nn.modules.block import HyperACE, HyperACEScale
        targets = (HyperACE, HyperACEScale)
    except ImportError:
        from ultralytics.nn.modules.block import HyperACE
        targets = (HyperACE,)
    for i, m in enumerate(detection_model.model):
        if isinstance(m, targets):
            return i, m
    return None, None


def _hyperace_out_channels(layer: nn.Module) -> int:
    return layer.cv2.conv.out_channels


# ============================================================================
# Loss wrapper
# ============================================================================

class HyperMILLoss:
    """Wraps base v8DetectionLoss with MIL count regression term.

    Pickle considerations: this is a top-level class. The instance stores
    `self.base` (v8DetectionLoss instance) and `self.model_ref` (DetectionModel
    instance). The model in turn stores `self.criterion = this` -- so the cycle
    is `model.criterion.model_ref -> model`. Pickle handles cycles via memo,
    so this is fine as long as nothing referenced is a local closure.
    """

    def __init__(self, base_criterion, model_ref):
        self.base = base_criterion
        self.model_ref = model_ref
        self.device = base_criterion.device
        self.hyp = base_criterion.hyp
        if hasattr(base_criterion, "stride"):
            self.stride = base_criterion.stride

    def __call__(self, preds, batch):
        # 1. Base detection loss (unchanged)
        base_total, base_items = self.base(preds, batch)

        # 2. Hard skip MIL during validation/eval (no_grad context)
        if not torch.is_grad_enabled():
            return base_total, base_items

        # 3. Retrieve cached HyperACE output via module-level dict
        hid = getattr(self.model_ref, "_hypermil_hyperace_id", None)
        if hid is None:
            return base_total, base_items
        feat = _HYPERACE_OUTPUTS.get(hid)
        if feat is None:
            return base_total, base_items

        # 4. Guard against batch-size mismatch
        img = batch.get("img") if isinstance(batch, dict) else None
        if img is not None and feat.shape[0] != img.shape[0]:
            return base_total, base_items

        # 5. MIL forward
        mil_count, _ = self.model_ref.mil_head(feat)
        B = feat.shape[0]
        device = mil_count.device

        # 6. Target count per image
        if "batch_idx" not in batch:
            return base_total, base_items
        bidx = batch["batch_idx"].view(-1).to(device)
        target_count = torch.zeros(B, device=device, dtype=mil_count.dtype)
        ones = torch.ones_like(bidx, dtype=mil_count.dtype)
        target_count.scatter_add_(0, bidx.long(), ones)

        # 7. MIL count loss
        mil_loss = F.smooth_l1_loss(mil_count, target_count, beta=1.0)

        # 8. Optional consistency loss
        consist_w = getattr(self.model_ref, "consist_weight", 0.0)
        consist_loss = torch.tensor(0.0, device=device)
        if consist_w > 0:
            try:
                feats = preds[1] if isinstance(preds, tuple) else preds
                nc = self.base.nc
                cls_sums = []
                for f in feats:
                    cls = f[:, -nc:]
                    cls = cls.sigmoid().flatten(2).sum(dim=2).sum(dim=1)
                    cls_sums.append(cls)
                soft_count = sum(cls_sums)
                consist_loss = F.smooth_l1_loss(soft_count, mil_count.detach(),
                                                beta=1.0)
            except Exception:
                consist_loss = torch.tensor(0.0, device=device)

        # 9. Combine. base_total = sum(box, cls, dfl) * batch_size. Scale
        #    mil similarly for consistent weighting.
        mil_w = self.model_ref.mil_weight
        mil_scaled = mil_loss * B * mil_w
        consist_scaled = consist_loss * B * consist_w

        total = base_total + mil_scaled + consist_scaled

        # 10. Track stats
        self.model_ref._last_mil_loss = float(mil_loss.detach())
        self.model_ref._last_mil_count_mean = float(mil_count.mean().detach())
        self.model_ref._last_mil_target_mean = float(target_count.mean().detach())
        if consist_w > 0:
            self.model_ref._last_consist_loss = float(consist_loss.detach())

        return total, base_items


# ============================================================================
# DetectionModel subclass (pickle-safe init_criterion override)
# ============================================================================

# Lazy import / late binding for the parent class so this module can be
# imported even if ultralytics isn't installed.
def _get_hypermil_detection_class():
    """Return HyperMILDetectionModel subclass of ultralytics DetectionModel.

    Defined lazily because we need ultralytics imported. The returned class is
    cached so all instances refer to the same class (important for pickle).
    """
    global _HyperMILDetectionModel
    try:
        return _HyperMILDetectionModel
    except NameError:
        pass
    from ultralytics.nn.tasks import DetectionModel

    class HyperMILDetectionModel(DetectionModel):
        """DetectionModel that wraps init_criterion with HyperMILLoss.

        Top-level class (defined at module import via _get_hypermil_detection_class).
        Pickleable because it has a stable fully-qualified name once cached as
        `afb_yolov13.hypermil._HyperMILDetectionModel`.
        """

        def init_criterion(self):
            base = super().init_criterion()
            return HyperMILLoss(base, self)

    _HyperMILDetectionModel = HyperMILDetectionModel
    # Make accessible by qualified name for pickle
    HyperMILDetectionModel.__module__ = __name__
    HyperMILDetectionModel.__qualname__ = "_HyperMILDetectionModel"
    globals()["_HyperMILDetectionModel"] = HyperMILDetectionModel
    return HyperMILDetectionModel


# ============================================================================
# Install function
# ============================================================================

def install_hypermil(detection_model, mil_weight: float = 0.5,
                    mil_hidden: int = 128, consist_weight: float = 0.0):
    """Install HyperMIL aux head + loss on a YOLOv13 DetectionModel.

    Pickle-safe: all stored attributes are either top-level objects or basic
    types. No closures, no instance-attributed methods.

    Effects:
      - detection_model.mil_head: HyperMILHead submodule.
      - detection_model.mil_weight / consist_weight: floats.
      - detection_model._hypermil_hyperace_id: int (id of HyperACE module).
      - detection_model.__class__: swapped to HyperMILDetectionModel which
        overrides init_criterion to wrap with HyperMILLoss at first .loss() call.
      - HyperACE forward hook (top-level fn) writes to module-level
        _HYPERACE_OUTPUTS dict, read by the loss wrapper.

    Returns:
        detection_model (modified in-place).
    """
    hidx, hlayer = _find_hyperace_layer(detection_model)
    if hlayer is None:
        raise ValueError(
            "[HyperMIL] No HyperACE / HyperACEScale layer found in model."
        )
    c_out = _hyperace_out_channels(hlayer)
    device = next(detection_model.parameters()).device

    # 1. MIL head (registered submodule -> optimizer picks it up)
    detection_model.mil_head = HyperMILHead(c_out, mil_hidden).to(device)
    detection_model.mil_weight = float(mil_weight)
    detection_model.consist_weight = float(consist_weight)
    detection_model._hypermil_hyperace_id = id(hlayer)
    detection_model._last_mil_loss = 0.0
    detection_model._last_mil_count_mean = 0.0
    detection_model._last_mil_target_mean = 0.0

    # 2. Forward hook -- TOP-LEVEL FN, pickleable
    hlayer.register_forward_hook(_hypermil_capture_hook)

    # 3. Swap class so init_criterion is overridden (pickle-safe vs assigning
    #    a closure to detection_model.init_criterion).
    HyperMILDetectionModel = _get_hypermil_detection_class()
    if not isinstance(detection_model, HyperMILDetectionModel):
        # Check that base class is compatible (DetectionModel)
        from ultralytics.nn.tasks import DetectionModel
        if not isinstance(detection_model, DetectionModel):
            raise TypeError(
                f"[HyperMIL] Expected DetectionModel, got {type(detection_model).__name__}"
            )
        detection_model.__class__ = HyperMILDetectionModel

    # 4. Reset any cached criterion so next .loss() re-builds via overridden
    #    init_criterion (HyperMILLoss).
    detection_model.criterion = None

    n_mil_params = sum(p.numel() for p in detection_model.mil_head.parameters())
    print(
        f"[HyperMIL] installed (pickle-safe):\n"
        f"  HyperACE layer index : {hidx}  (id={detection_model._hypermil_hyperace_id})\n"
        f"  HyperACE out channels: {c_out}\n"
        f"  MIL head params      : {n_mil_params:,}\n"
        f"  mil_weight           : {mil_weight}\n"
        f"  mil_hidden           : {mil_hidden}\n"
        f"  consist_weight       : {consist_weight}\n"
        f"  model class -> {type(detection_model).__name__}\n"
    )
    return detection_model


# ============================================================================
# Callback factory
# ============================================================================

def make_hypermil_callback(mil_weight: float = 0.5, mil_hidden: int = 128,
                           consist_weight: float = 0.0):
    """Create an Ultralytics callback that installs HyperMIL pre-training.

    Use:
        model.add_callback('on_pretrain_routine_start',
                           make_hypermil_callback(mil_weight=0.5))

    Fires before optimizer build so MIL head params are picked up.
    """

    def _cb(trainer):
        model = trainer.model if hasattr(trainer, "model") else trainer
        install_hypermil(model, mil_weight=mil_weight, mil_hidden=mil_hidden,
                         consist_weight=consist_weight)

    return _cb
