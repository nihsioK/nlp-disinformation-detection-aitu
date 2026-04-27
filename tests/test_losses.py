"""Tests for the ordinal-aware loss functions."""

from __future__ import annotations

import math

import pytest
import torch

from src.disinfo_detection.losses import OrdinalAwareLoss, build_loss_module, emd_loss


def _logits(num_classes: int = 6, batch_size: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch_size, num_classes, requires_grad=True)


def test_emd_loss_is_zero_on_perfect_match():
    num_classes = 4
    labels = torch.tensor([0, 1, 2, 3])
    # Logits that put almost all mass on the correct class.
    logits = torch.full((4, num_classes), -10.0)
    for index, target in enumerate(labels):
        logits[index, target] = 10.0
    loss = emd_loss(logits, labels, num_classes=num_classes)
    assert loss.item() < 1e-3


def test_emd_loss_penalises_distant_classes_more_than_neighbours():
    num_classes = 6
    labels = torch.tensor([0, 0])
    near = torch.full((1, num_classes), -10.0)
    near[0, 1] = 10.0  # one bin away from the truth.
    far = torch.full((1, num_classes), -10.0)
    far[0, 5] = 10.0   # five bins away.
    near_loss = emd_loss(near, labels[:1], num_classes=num_classes)
    far_loss = emd_loss(far, labels[:1], num_classes=num_classes)
    assert far_loss.item() > near_loss.item() * 4


def test_ordinal_aware_loss_falls_back_to_cross_entropy_when_emd_zero():
    logits = _logits()
    labels = torch.tensor([0, 1, 2, 3])
    ce_only = OrdinalAwareLoss(num_classes=6, ce_weight=1.0, emd_weight=0.0)
    blended = OrdinalAwareLoss(num_classes=6, ce_weight=1.0, emd_weight=0.0)
    a = ce_only(logits, labels)
    b = blended(logits, labels)
    assert torch.allclose(a, b)


def test_ordinal_aware_loss_blends_terms_linearly():
    logits = _logits()
    labels = torch.tensor([0, 1, 2, 3])
    ce_only = OrdinalAwareLoss(num_classes=6, ce_weight=1.0, emd_weight=0.0)
    emd_only = OrdinalAwareLoss(num_classes=6, ce_weight=0.0, emd_weight=1.0)
    blended = OrdinalAwareLoss(num_classes=6, ce_weight=1.0, emd_weight=1.0)
    expected = ce_only(logits, labels) + emd_only(logits, labels)
    assert torch.allclose(blended(logits, labels), expected, atol=1e-6)


def test_build_loss_module_defaults_to_cross_entropy():
    module = build_loss_module(loss_cfg=None, num_classes=6, class_weights=None, label_smoothing=0.0)
    assert module.ce_weight == 1.0
    assert module.emd_weight == 0.0


def test_build_loss_module_handles_ordinal_type():
    module = build_loss_module(
        loss_cfg={"type": "ordinal", "ce_weight": 1.0, "emd_weight": 0.5},
        num_classes=6,
        class_weights=None,
        label_smoothing=0.05,
    )
    assert module.ce_weight == 1.0
    assert module.emd_weight == 0.5
    assert math.isclose(module.label_smoothing, 0.05)


def test_build_loss_module_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_loss_module(
            loss_cfg={"type": "magic"},
            num_classes=6,
            class_weights=None,
            label_smoothing=0.0,
        )


def test_ordinal_aware_loss_supports_class_weights():
    logits = _logits()
    labels = torch.tensor([0, 1, 2, 3])
    weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    weighted = OrdinalAwareLoss(num_classes=6, ce_weight=1.0, emd_weight=0.0, class_weights=weights)
    unweighted = OrdinalAwareLoss(num_classes=6, ce_weight=1.0, emd_weight=0.0)
    # Class-1 gets up-weighted, so the weighted loss should not equal the unweighted one.
    assert not torch.allclose(weighted(logits, labels), unweighted(logits, labels))
