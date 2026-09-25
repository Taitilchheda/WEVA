"""Spectral Wave Mixer (SWM) and multi-head SWM.

The SWM is the WEVA replacement for attention. It operates on a complex-
valued input, applies FFT along the sequence dimension, separately
transforms the amplitude and phase spectra with learned linear maps, and
inverts with IFFT.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """Cast real tensor to complex (imag = 0)."""
    if x.is_complex():
        return x
    return x.to(torch.complex64 if x.dtype == torch.float32 else torch.complex128)


class SpectralWaveMixer(nn.Module):
    """Single SWM block. Operates on complex-valued (B, N, d) inputs.

    Components:
        1) FFT along the sequence dim.
        2) Decompose into amplitude and phase.
        3) Apply learned linear maps W_amp, W_phase.
        4) Add a learnable frequency bias (broadcast across sequence).
        5) Reconstruct complex spectrum and inverse FFT.
        6) Residual connection.
    """

    def __init__(self, d_model: int, init_std: float = 0.02, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.W_amp = nn.Parameter(torch.randn(d_model, d_model) * init_std)
        self.W_phase = nn.Parameter(torch.randn(d_model, d_model) * init_std)
        self.b_freq = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, d) complex
        z_in = z
        # 1) FFT along the sequence dim
        z_freq = torch.fft.fft(z, dim=1)            # (B, N, d) complex
        # 2) Amplitude and phase
        amp = z_freq.abs()                           # (B, N, d) real
        phase = z_freq.angle()                       # (B, N, d) real
        # 3) Learned linear maps
        amp_out = amp @ self.W_amp + self.b_freq     # (B, N, d) real
        phase_out = phase @ self.W_phase             # (B, N, d) real
        # 4) Reconstruct complex spectrum
        z_freq_out = amp_out * torch.exp(1j * phase_out)
        # 5) Inverse FFT
        z_out = torch.fft.ifft(z_freq_out, dim=1)    # (B, N, d) complex
        # 6) Output projection (real linear) + residual
        z_real = z_out.real.to(z_out.real.dtype)
        out = self.out_proj(z_real) + z_in.real
        return self.dropout(out)


class MultiHeadSWM(nn.Module):
    """H independent SWMs over d/H-dimensional subspaces.

    Equivalent in spirit to multi-head attention: each head sees a slice of
    the channel dimension. Outputs are concatenated and linearly projected.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.heads = nn.ModuleList([
            SpectralWaveMixer(self.d_head, dropout=dropout) for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, d_model) complex
        B, N, _ = z.shape
        # Split channel dim into heads
        z_split = z.view(B, N, self.n_heads, self.d_head)
        # Process each head
        head_outs = []
        for h, head in enumerate(self.heads):
            z_h = z_split[:, :, h, :].contiguous()
            head_outs.append(head(z_h))
        out = torch.cat(head_outs, dim=-1)           # (B, N, d_model) real
        return self.out_proj(out)


class SWMBlock(nn.Module):
    """A full SWM block: Multi-Head SWM + FRN, with residuals and norms.

    Pre-norm style for stability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, omega_0: float,
                 dropout: float = 0.0):
        super().__init__()
        from .frn import FeedforwardResonanceNetwork
        self.norm1 = nn.LayerNorm(d_model)
        self.mhswm = MultiHeadSWM(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.frn = FeedforwardResonanceNetwork(d_model, d_ff, omega_0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, d) complex; we work on the real part through the
        # pre-norm, then cast back.
        z_real = z.real
        # Sub-block 1: Multi-Head SWM
        h = self.norm1(z_real)
        h = self.mhswm(_to_complex(h))
        h = self.dropout(h)
        z_real = z_real + h
        # Sub-block 2: FRN
        h2 = self.norm2(z_real)
        h2 = self.frn(h2)
        h2 = self.dropout(h2)
        z_real = z_real + h2
        return _to_complex(z_real)
