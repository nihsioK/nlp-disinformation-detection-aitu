"""Loss functions used by the LIAR classifiers.

LIAR labels are ordinal (`pants-fire < false < barely-true < half-true <
mostly-true < true`) but the standard cross-entropy loss treats them as
nominal. Predicting `true` for a `pants-fire` statement should cost more than
predicting `false` for it. This module exposes:

- `emd_loss`: squared Earth Mover's Distance between the predicted softmax
  distribution and a (smoothed) one-hot target, computed via cumulative
  distribution differences. Penalises predictions further from the true
  ordinal class more than predictions one bin away (Hou et al., 2017).
- `OrdinalAwareLoss`: convex combination of weighted CE and EMD with optional
  label smoothing. Falls back to plain CE when `emd_weight == 0`, which keeps
  the previous training behaviour reproducible from configs.

Class weights (e.g., the inverse-sqrt-frequency weights used elsewhere in the
project) are applied to the CE term only — EMD already encodes ordinal
distance and double-weighting tends to over-correct toward the rare classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def emd_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Squared Earth Mover's Distance loss for ordinal classification.

    Args:
        logits: `(batch, num_classes)` raw logits.
        labels: `(batch,)` integer class targets.
        num_classes: Number of ordinal classes.
        label_smoothing: Optional uniform smoothing applied to the target
            distribution before the CDF difference is computed.

    Returns:
        Mean squared distance between the predicted and target cumulative
        distributions.
    """

    probs = F.softmax(logits, dim=1)
    if label_smoothing > 0:
        smooth = label_smoothing / num_classes
        target = torch.full_like(probs, smooth)
        target.scatter_(1, labels.unsqueeze(1), 1 - label_smoothing + smooth)
    else:
        target = F.one_hot(labels, num_classes=num_classes).to(probs.dtype)

    cdf_pred = torch.cumsum(probs, dim=1)
    cdf_true = torch.cumsum(target, dim=1)
    return torch.mean(torch.sum((cdf_pred - cdf_true) ** 2, dim=1))


class OrdinalAwareLoss(nn.Module):
    """Combines weighted cross-entropy with squared-EMD for ordinal labels."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        emd_weight: float = 0.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if ce_weight <= 0 and emd_weight <= 0:
            raise ValueError("At least one of ce_weight / emd_weight must be > 0.")
        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.emd_weight = float(emd_weight)
        self.label_smoothing = float(label_smoothing)
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss = logits.new_zeros(())
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * F.cross_entropy(
                logits,
                labels,
                weight=weight,
                label_smoothing=self.label_smoothing,
            )
        if self.emd_weight > 0:
            loss = loss + self.emd_weight * emd_loss(
                logits,
                labels,
                num_classes=self.num_classes,
                label_smoothing=self.label_smoothing,
            )
        return loss

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, ce_weight={self.ce_weight}, "
            f"emd_weight={self.emd_weight}, label_smoothing={self.label_smoothing}"
        )


def build_loss_module(
    loss_cfg: dict | None,
    num_classes: int,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> OrdinalAwareLoss:
    """Construct an `OrdinalAwareLoss` from a loss-section config dict.

    The config may be `None` or contain:
        type: "ce" | "ordinal"
        ce_weight: float (default 1.0)
        emd_weight: float (default 0.0 for ce, 0.5 for ordinal)

    Returns a loss module that subsumes plain CE when `type == "ce"`.
    """

    cfg = dict(loss_cfg or {})
    loss_type = str(cfg.get("type", "ce")).lower()
    if loss_type not in {"ce", "ordinal"}:
        raise ValueError(f"Unknown loss type: {loss_type!r}. Use 'ce' or 'ordinal'.")
    ce_weight = float(cfg.get("ce_weight", 1.0))
    if loss_type == "ordinal":
        emd_weight = float(cfg.get("emd_weight", 0.5))
    else:
        emd_weight = float(cfg.get("emd_weight", 0.0))
    return OrdinalAwareLoss(
        num_classes=num_classes,
        ce_weight=ce_weight,
        emd_weight=emd_weight,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )


__all__ = ["emd_loss", "OrdinalAwareLoss", "build_loss_module"]
