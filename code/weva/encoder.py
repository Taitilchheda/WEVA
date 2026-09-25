"""Universal Spectral Encoder (USE).

The USE projects any modality into a shared complex-valued wave-primitive
representation Z ∈ C^{N×d}. The projection that maps the per-modality
embedding (A_i, phi_i) is *shared across all four modalities*. This is the
"modality-agnostic" claim of WEVA.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class _TextFrontend(nn.Module):
    """Token embedding + learned positional embedding."""

    def __init__(self, vocab_size: int, max_len: int, d_model: int, dropout: float):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # input_ids: (B, L)
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token(input_ids) + self.position(pos)
        return self.dropout(x)  # (B, L, d)


class _ImageFrontend(nn.Module):
    """ViT-style 16x16 patch projection."""

    def __init__(self, image_size: int, patch_size: int, d_model: int, dropout: float):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.position = nn.Embedding(self.n_patches, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W)
        B = images.size(0)
        x = self.proj(images)              # (B, d, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)    # (B, N, d)
        pos = torch.arange(self.n_patches, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.position(pos)
        return self.dropout(x)


class _AudioFrontend(nn.Module):
    """STFT-based audio frontend.

    Computes the magnitude spectrogram, applies a log compression, then
    linear-projects each frame to d_model.
    """

    def __init__(self, n_fft: int, hop: int, d_model: int, max_frames: int, dropout: float):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.n_freq = n_fft // 2 + 1
        self.proj = nn.Linear(self.n_freq, d_model)
        self.position = nn.Embedding(max_frames, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_frames = max_frames

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (B, T_audio)
        window = torch.hann_window(self.n_fft, device=waveforms.device)
        spec = torch.stft(
            waveforms, n_fft=self.n_fft, hop_length=self.hop, window=window,
            return_complex=True, center=True,
        )                                       # (B, n_freq, T_frames)
        spec = spec.transpose(1, 2)             # (B, T_frames, n_freq)
        mag = spec.abs().clamp_min(1e-6).log()  # log-magnitude
        # Truncate or pad to max_frames
        T = mag.size(1)
        if T > self.max_frames:
            mag = mag[:, : self.max_frames, :]
        elif T < self.max_frames:
            mag = F.pad(mag, (0, 0, 0, self.max_frames - T))
        x = self.proj(mag)                      # (B, max_frames, d)
        pos = torch.arange(self.max_frames, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.position(pos)
        return self.dropout(x)


class _VideoFrontend(nn.Module):
    """Per-frame patch projection with temporal downsample.

    Treats a video as a sequence of frames; each frame is encoded by the
    image frontend (which already adds positional embeddings), then a 1D
    conv along the time axis downsamples 2x.
    """

    def __init__(self, image_size: int, patch_size: int, d_model: int,
                 max_frames: int, dropout: float):
        super().__init__()
        self.image_frontend = _ImageFrontend(image_size, patch_size, d_model, dropout)
        self.temporal_pool = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)
        self.max_frames = max_frames
        self.dropout = nn.Dropout(dropout)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, 3, H, W)
        B, T, _, _, _ = video.shape
        flat = video.reshape(B * T, *video.shape[2:])
        x = self.image_frontend(flat)            # (B*T, N_per_frame, d)
        # Pool the spatial dimension, then arrange as a sequence over time
        x = x.mean(dim=1)                        # (B*T, d)
        x = x.view(B, T, -1)                     # (B, T, d)
        # Temporal conv: (B, d, T) -> (B, d, T/2)
        x = x.transpose(1, 2)
        x = self.temporal_pool(x)
        x = x.transpose(1, 2)                    # (B, T/2, d)
        # Truncate or pad to max_frames
        Tnew = x.size(1)
        if Tnew > self.max_frames:
            x = x[:, : self.max_frames, :]
        elif Tnew < self.max_frames:
            x = F.pad(x, (0, 0, 0, self.max_frames - Tnew))
        return self.dropout(x)


class _SharedWaveProjection(nn.Module):
    """The single shared projection that maps per-modality embeddings to
    (amplitude, phase) wave packets. This is the heart of modality-agnosticism.

    Input: real-valued embedding (B, N, d_model) from any modality.
    Output: complex-valued wave packet (B, N, d_model).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.to_amp = nn.Linear(d_model, d_model)
        self.to_phase = nn.Linear(d_model, d_model)
        self.proj_residual = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model) real
        amp = F.softplus(self.to_amp(x))        # positivity by softplus
        phase = self.to_phase(x)                # unbounded
        # Residual: add a real linear projection of the original signal
        residual = self.proj_residual(x)
        # Build complex wave packet: A * exp(i*phi) + residual (real)
        wave = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))
        wave = wave + residual.to(wave.dtype if wave.is_complex() else torch.float32)
        return wave


class _ModalityToken(nn.Module):
    """A learned embedding prepended so the backbone can disambiguate
    modalities. Modality index: 0=text, 1=image, 2=audio, 3=video.
    """

    def __init__(self, n_modalities: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_modalities, d_model)

    def forward(self, x: torch.Tensor, modality: Modality) -> torch.Tensor:
        # x: (B, N, d) complex
        mod_idx = list(Modality).index(Modality(modality))
        idx = torch.full((x.size(0),), mod_idx, dtype=torch.long, device=x.device)
        token = self.emb(idx)
        # Build a complex token (real-valued, zero imag) for concat.
        token_c = torch.complex(token, torch.zeros_like(token))
        return torch.cat([token_c.unsqueeze(1), x], dim=1)


class UniversalSpectralEncoder(nn.Module):
    """The complete USE. Encodes any modality into a shared complex-valued
    wave representation, prepends a modality token, and returns.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_frontend = _TextFrontend(
            config.text_vocab_size, config.text_max_len, config.d_model, config.dropout,
        )
        self.image_frontend = _ImageFrontend(
            config.image_size, config.image_patch_size, config.d_model, config.dropout,
        )
        self.audio_frontend = _AudioFrontend(
            config.audio_n_fft, config.audio_hop, config.d_model,
            config.audio_max_frames, config.dropout,
        )
        self.video_frontend = _VideoFrontend(
            config.image_size, config.image_patch_size, config.d_model,
            config.video_max_frames, config.dropout,
        )
        self.shared_wave_projection = _SharedWaveProjection(config.d_model)
        self.modality_token = _ModalityToken(config.n_modalities, config.d_model)
        self.modality_index = {m.value: i for i, m in enumerate(Modality)}

    def encode(self, modality: Modality, x: torch.Tensor) -> torch.Tensor:
        if modality == Modality.TEXT:
            return self.text_frontend(x)
        if modality == Modality.IMAGE:
            return self.image_frontend(x)
        if modality == Modality.AUDIO:
            return self.audio_frontend(x)
        if modality == Modality.VIDEO:
            return self.video_frontend(x)
        raise ValueError(f"Unknown modality: {modality}")

    def forward(self, modality: Modality, x: torch.Tensor) -> torch.Tensor:
        # 1) Per-modality frontend -> real embedding (B, N, d_model)
        emb = self.encode(modality, x)
        # 2) Shared projection -> complex wave packet (B, N, d_model)
        wave = self.shared_wave_projection(emb)
        # 3) Prepend modality token (real-valued, complex zero-imag part)
        wave = self.modality_token(wave, modality)
        return wave
