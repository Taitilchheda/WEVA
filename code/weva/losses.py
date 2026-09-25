"""Loss functions for WEVA.

Primary: MultimodalContrastiveLoss (InfoNCE over all ordered modality pairs).
Side:    OrbitalEnergyRegularizer (OER — paper appendix only).
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _info_nce_one_way(
    a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07,
) -> torch.Tensor:
    """Standard symmetric InfoNCE between two batches of normalized
    embeddings (B, d).
    """
    logits = (a @ b.T) / temperature                # (B, B)
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


class MultimodalContrastiveLoss(nn.Module):
    """Symmetric InfoNCE averaged over all ordered modality pairs.

    Given a batch with up to 4 modalities (text, image, audio, video), this
    loss treats each modality's embeddings as a "view" of the same item and
    pulls corresponding items together while pushing non-corresponding items
    apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = list(embeddings.keys())
        if len(keys) < 2:
            return torch.tensor(0.0, device=next(iter(embeddings.values())).device)
        total = 0.0
        pairs = 0
        for i, j in permutations(keys, 2):
            total = total + _info_nce_one_way(
                embeddings[i], embeddings[j], self.temperature
            )
            pairs += 1
        return total / max(pairs, 1)


class OrbitalEnergyRegularizer(nn.Module):
    """OER v2 — soft-min over energy shells. (Historical label; v2 refers to the soft-min variant — current code.)

    Original (v1) formulation used `min_n |E(θ) - E_n*|` which is non-
    differentiable. The soft-min variant uses negative log-sum-exp:

        L_OER(θ) = -τ * log( Σ_n exp(-|E(θ) - E_n*|² / τ) )

    As τ → 0, this approximates the v1 hard-min. As τ → ∞, it averages
    over shells.

    Energy is computed from gradient norm (training activity) plus a small
    weight-norm term (magnitude of model parameters).
    """

    def __init__(self, E0: float = 1.0, n_shells: int = 5,
                 tau: float = 0.1, weight_lambda: float = 0.1):
        super().__init__()
        self.E0 = E0
        self.n_shells = n_shells
        self.tau = tau
        self.weight_lambda = weight_lambda
        # Pre-compute shell targets E_n* = E0 / n²
        self.register_buffer(
            "shells", torch.tensor([E0 / (n ** 2) for n in range(1, n_shells + 1)])
        )

    def energy(self, model: nn.Module) -> torch.Tensor:
        """Compute the current model energy.

        Uses the running gradient norms (if available) and weight norms.
        """
        grad_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq = grad_sq + p.grad.detach().pow(2).sum()
        weight_sq = sum(p.detach().pow(2).sum() for p in model.parameters())
        return grad_sq + self.weight_lambda * weight_sq

    def forward(self, model: nn.Module) -> torch.Tensor:
        E = self.energy(model)
        diffs = (E - self.shells) ** 2
        # Soft-min: -τ * log( Σ exp(-diffs/τ) )
        return -self.tau * torch.log(torch.exp(-diffs / self.tau).sum())
