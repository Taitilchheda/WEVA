"""Feedforward Resonance Network (FRN).

A sinusoidal-activation feedforward block in the SIREN style
(Sitzmann et al. 2020). The sin activation is a natural fit for WEVA
because representations are already wave-structured.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class FeedforwardResonanceNetwork(nn.Module):
    """Two-layer MLP with sinusoidal hidden activation.

    FRN(x) = W2 @ sin(omega_0 * (W1 @ x + b1)) + b2
    """

    def __init__(self, d_model: int, d_ff: int, omega_0: float = 30.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        # omega_0 as a learnable scalar (per SIREN convention)
        self.omega_0 = nn.Parameter(torch.tensor(float(omega_0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model) real
        h = torch.sin(self.omega_0 * self.linear1(x))
        return self.linear2(h)
