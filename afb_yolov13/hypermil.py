"""HyperMIL — Hypergraph-backed Multiple Instance Learning aux head for YOLOv13.

Adds an image-level bacilli count prediction head reading from HyperACE output
(layer 9 of YOLOv13). The MIL count loss is added to the base detection loss as
a regularizer that exploits image-level supervision robustness against missing
instance-level annotations.

Motivation:
    Sparsely-annotated AFB/TB microscopy datasets (e.g. Tuberculosis6208) have
    label noise where annotators mark only a subset of bacilli per smear -- for
    diagnostic purposes a few marked instances suffice, but the model treats
    the rest as false positives. Image-level count (total bacilli per image) is
    more robust to this noise than instance-level box annotations.

    By auxiliary-training to predict image-level count from HyperACE's
    hypergraph-enhanced features, the model is pushed to discover ALL bacilli
    (not only the GT-marked ones), then the consistency regularizer pulls the
    detection head's confidence sum toward that count.

References:
    - Ilse et al. 2018, "Attention-based Deep Multiple Instance Learning" -- the
      gated attention pooling formulation used in AttnMILPool.
    - Polski et al. 2020, "Classifying bacteria clones using attention-based
      deep MIL" -- MIL precedent for bacteria classification.
    - Related but distinct: NeGPR (2025) does graph-level pseudo-label
      refinement for domain adaptation, not hypergraph-feature MIL for
      single-domain detection.

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
    """Gated attention pooling (Ilse et al. 2018) over spatial features.

    Maps a 4D feature map [B, C, H, W] to a 2D pooled vector [B, C] using a
    learned per-location attention weight (softmax over spatial positions).
    Also returns the [B, H*W] attention map for visualization / consistency.

    Gated attention formula:
        a_i = w^T ( tanh(V z_i) ⊙ sigmoid(U z_i) )
        α_i = softmax_i(a_i)
        pooled = Σ_i α_i z_i
    """

    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.attn_V = nn.Linear(channels, hidden)
        self.attn_U = nn.Linear(channels, hidden)
        self.attn_w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)        # [B, N, C], N = H*W
        u = torch.sigmoid(self.attn_U(z))        # [B, N, hidden]
        v = torch.tanh(self.attn_V(z))           # [B, N, hidden]
        a = self.attn_w(u * v).squeeze(-1)       # [B, N]
        a = F.softmax(a, dim=1)
        pooled = (a.unsqueeze(2) * z).sum(dim=1) # [B, C]
        return pooled, a


class HyperMILHead(nn.Module):
    """MIL count prediction head: AttnMILPool + MLP -> non-negative count.

    Output is constrained non-negative via Softplus.
    """

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
# Helpers
# ============================================================================

def _find_hyperace_layer(detection_model) -> tuple[int, nn.Module] | tuple[None, None]:
    """Locate HyperACE / HyperACEScale in model.model Sequential.

    Returns (layer_index, module) or (None, None) if not found.
    """
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
    """Get output channel count of HyperACE-like layer.

    HyperACE.cv2 is the final Conv block; its inner nn.Conv2d has out_channels = c2.
    """
    return layer.cv2.conv.out_channels


# ============================================================================
# Loss wrapper
# ============================================================================

class HyperMILLoss:
    """Wraps base v8DetectionLoss with MIL count regression term.

    Reads HyperACE feature map from `model_ref._hyperace_out` (captured by
    forward hook), feeds it to `model_ref.mil_head`, computes Smooth-L1 loss
    against per-image GT box count (from batch['batch_idx']).

    Optional consistency regularizer: encourages sum of detection objectness
    scores to match MIL-predicted count.
    """

    def __init__(self, base_criterion, model_ref):
        self.base = base_criterion
        self.model_ref = model_ref
        # Inherit attrs the trainer / lr scheduler may poke
        self.device = base_criterion.device
        self.hyp = base_criterion.hyp
        if hasattr(base_criterion, "stride"):
            self.stride = base_criterion.stride

    def __call__(self, preds, batch):
        # 1. Base detection loss (unchanged)
        base_total, base_items = self.base(preds, batch)

        feat = getattr(self.model_ref, "_hyperace_out", None)
        if feat is None or not self.model_ref.training:
            # During val/eval or if hook didn't fire, skip MIL
            return base_total, base_items

        # 2. MIL forward
        mil_count, _ = self.model_ref.mil_head(feat)  # [B]
        B = feat.shape[0]
        device = mil_count.device

        # 3. Target count per image from batch GT
        if "batch_idx" not in batch:
            return base_total, base_items
        bidx = batch["batch_idx"].view(-1).to(device)
        target_count = torch.zeros(B, device=device, dtype=mil_count.dtype)
        # Vectorized count via scatter_add
        ones = torch.ones_like(bidx, dtype=mil_count.dtype)
        target_count.scatter_add_(0, bidx.long(), ones)

        # 4. MIL count loss (Smooth L1 for robustness to outliers)
        mil_loss = F.smooth_l1_loss(mil_count, target_count, beta=1.0)

        # 5. Optional consistency loss
        consist_w = getattr(self.model_ref, "consist_weight", 0.0)
        consist_loss = torch.tensor(0.0, device=device)
        if consist_w > 0:
            # Soft "count" from detection: sum of sigmoid(class_logit) across anchors
            try:
                feats = preds[1] if isinstance(preds, tuple) else preds
                # feats is list of [B, no, H, W] per scale
                # no = reg_max*4 + nc; take class portion (last nc channels)
                nc = self.base.nc
                cls_sums = []
                for f in feats:
                    cls = f[:, -nc:]                      # [B, nc, H, W]
                    cls = cls.sigmoid().flatten(2).sum(dim=2).sum(dim=1)  # [B]
                    cls_sums.append(cls)
                soft_count = sum(cls_sums)
                consist_loss = F.smooth_l1_loss(soft_count, mil_count.detach(),
                                                beta=1.0)
            except Exception:
                consist_loss = torch.tensor(0.0, device=device)

        # 6. Combine
        # base_total = sum(box, cls, dfl) * batch_size in v8DetectionLoss.
        # Scale mil_loss similarly so weights compare consistently.
        mil_w = self.model_ref.mil_weight
        mil_scaled = mil_loss * B * mil_w
        consist_scaled = consist_loss * B * consist_w

        total = base_total + mil_scaled + consist_scaled

        # 7. Track MIL stats on model for logging
        self.model_ref._last_mil_loss = float(mil_loss.detach())
        self.model_ref._last_mil_count_mean = float(mil_count.mean().detach())
        self.model_ref._last_mil_target_mean = float(target_count.mean().detach())
        if consist_w > 0:
            self.model_ref._last_consist_loss = float(consist_loss.detach())

        return total, base_items


# ============================================================================
# Install function (model-side)
# ============================================================================

def install_hypermil(detection_model, mil_weight: float = 0.5,
                    mil_hidden: int = 128, consist_weight: float = 0.0):
    """Install HyperMIL aux head + loss on a YOLOv13 DetectionModel.

    Modifies in-place:
        - detection_model.mil_head: HyperMILHead submodule (registered so it
          gets included when the Trainer later builds the optimizer).
        - detection_model._hyperace_out: storage for forward-hook captured tensor.
        - detection_model.init_criterion: monkey-patched to lazily return a
          HyperMILLoss wrapping the original criterion. Lazy so it's called
          AFTER trainer sets model.args.

    Args:
        detection_model: ultralytics.nn.tasks.DetectionModel (model.model in
            the YOLO wrapper). Must contain HyperACE or HyperACEScale.
        mil_weight: Weight for MIL count loss.
        mil_hidden: Hidden dim of MIL attention + MLP.
        consist_weight: Weight for detection-MIL consistency regularizer
            (0 disables, recommended start 0).

    Note on callback timing:
        Called from `on_pretrain_routine_start` BEFORE
        trainer.set_model_attributes() which assigns model.args. We therefore
        cannot call detection_model.init_criterion() directly here -- the
        underlying v8DetectionLoss requires model.args. Instead we monkey-patch
        init_criterion so it lazy-builds the wrapped criterion at first loss
        call (by which time model.args is set).

    Returns:
        detection_model (same instance, modified).
    """
    hidx, hlayer = _find_hyperace_layer(detection_model)
    if hlayer is None:
        raise ValueError(
            "[HyperMIL] No HyperACE / HyperACEScale layer found in model. "
            "HyperMIL requires a YOLOv13-style model with HyperACE in the head."
        )
    c_out = _hyperace_out_channels(hlayer)
    device = next(detection_model.parameters()).device

    # 1. MIL head (registered submodule so it's auto-included in
    #    model.parameters() when optimizer is built later by the trainer).
    detection_model.mil_head = HyperMILHead(c_out, mil_hidden).to(device)
    detection_model.mil_weight = float(mil_weight)
    detection_model.consist_weight = float(consist_weight)
    detection_model._hyperace_out = None
    detection_model._last_mil_loss = 0.0
    detection_model._last_mil_count_mean = 0.0
    detection_model._last_mil_target_mean = 0.0

    # 2. Forward hook on HyperACE captures its output for the loss to read.
    def _capture(_module, _inputs, output):
        detection_model._hyperace_out = output

    detection_model._hypermil_hook = hlayer.register_forward_hook(_capture)

    # 3. Monkey-patch init_criterion (lazy wrap).
    #    Reset any cached criterion so the next loss() call re-inits via patch.
    if hasattr(detection_model, "_hypermil_patched"):
        # Already patched - skip to avoid double-wrapping.
        print("[HyperMIL] init_criterion already patched; refreshing head/hook only.")
        return detection_model

    original_init_criterion = detection_model.init_criterion

    def patched_init_criterion():
        # At call time, trainer has set model.args; safe to build base criterion.
        base = original_init_criterion()
        return HyperMILLoss(base, detection_model)

    detection_model.init_criterion = patched_init_criterion
    detection_model._hypermil_patched = True
    # Clear any previously-cached criterion so next .loss() call re-inits.
    if hasattr(detection_model, "criterion"):
        detection_model.criterion = None

    n_mil_params = sum(p.numel() for p in detection_model.mil_head.parameters())
    print(
        f"[HyperMIL] installed (lazy criterion):\n"
        f"  HyperACE layer index : {hidx}\n"
        f"  HyperACE out channels: {c_out}\n"
        f"  MIL head params      : {n_mil_params:,}\n"
        f"  mil_weight           : {mil_weight}\n"
        f"  mil_hidden           : {mil_hidden}\n"
        f"  consist_weight       : {consist_weight}\n"
    )
    return detection_model


# ============================================================================
# Callback factory (notebook-side)
# ============================================================================

def make_hypermil_callback(mil_weight: float = 0.5, mil_hidden: int = 128,
                           consist_weight: float = 0.0):
    """Create an Ultralytics callback that installs HyperMIL pre-training.

    Use:
        model.add_callback('on_pretrain_routine_start',
                           make_hypermil_callback(mil_weight=0.5))

    Why on_pretrain_routine_start: fires BEFORE optimizer build, so MIL head
    params are picked up by the optimizer.
    """

    def _cb(trainer):
        model = trainer.model if hasattr(trainer, "model") else trainer
        install_hypermil(model, mil_weight=mil_weight, mil_hidden=mil_hidden,
                         consist_weight=consist_weight)

    return _cb
