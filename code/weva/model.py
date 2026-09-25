"""The complete WEVA model: USE + SWM backbone + per-modality heads."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import WEVAConfig
from .encoder import UniversalSpectralEncoder, Modality
from .mixer import SWMBlock


class _ProjectionHead(nn.Module):
    """A small per-modality projection head.

    Takes the SWM-backbone output for one modality, mean-pools over the
    sequence dimension, and projects to a normalized proj_dim vector.
    """

    def __init__(self, d_model: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, proj_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, d_model) real
        pooled = z.mean(dim=1)              # (B, d_model)
        out = self.proj(pooled)             # (B, proj_dim)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


class WEVAModel(nn.Module):
    """The full WEVA model.

    Architecture:
        USE (modality-agnostic) -> SWM backbone (modality-agnostic) -> per-modality
        projection heads.

    The USE and SWM backbone are *exactly the same modules* for all four
    modalities. Only the input frontend (text/image/audio/video → real
    embedding) and the output head differ.
    """

    def __init__(self, config: WEVAConfig):
        super().__init__()
        self.config = config
        self.use = UniversalSpectralEncoder(config)
        self.backbone = nn.ModuleList([
            SWMBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                omega_0=config.omega_0,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.heads = nn.ModuleDict({
            Modality.TEXT.value: _ProjectionHead(config.d_model, config.proj_dim),
            Modality.IMAGE.value: _ProjectionHead(config.d_model, config.proj_dim),
            Modality.AUDIO.value: _ProjectionHead(config.d_model, config.proj_dim),
            Modality.VIDEO.value: _ProjectionHead(config.d_model, config.proj_dim),
        })

    def encode(self, modality: Modality, x: torch.Tensor) -> torch.Tensor:
        """Encode one modality through USE + SWM backbone, returning real
        output embeddings (B, N+1, d_model) ready for the projection head.
        """
        z = self.use(modality, x)                  # complex (B, N+1, d)
        for block in self.backbone:
            z = block(z)
        z_real = z.real
        z_real = self.final_norm(z_real)
        return z_real

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        modalities_to_run: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the model on a batch containing one or more modalities.

        Args:
            batch: Dict mapping modality name -> tensor of that modality's
                   inputs. Only modalities present in this dict are encoded.
            modalities_to_run: optional subset of modalities to encode. If
                                None, encode all keys in `batch`.
        Returns:
            Dict mapping modality name -> (B, proj_dim) L2-normalized
            embedding.
        """
        if modalities_to_run is None:
            modalities_to_run = list(batch.keys())
        out: Dict[str, torch.Tensor] = {}
        for m in modalities_to_run:
            modality = Modality(m)
            z = self.encode(modality, batch[m])
            out[m] = self.heads[m](z)
        return out

    def num_parameters(self, only_trainable: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not only_trainable) or p.requires_grad
        )
