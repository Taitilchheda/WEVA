"""Dataset loaders for the WEVA multimodal training.

To keep the implementation self-contained and runnable on a single RTX
3060 (12GB) without external downloads, we provide:

- `SyntheticMultimodalDataset`: produces synthetic (text, image, audio,
  video) tuples on the fly. Used for end-to-end testing and for the demo
  when no real data is available. Each modality is sampled from a simple
  generative process with controllable "ground-truth" similarity.

- `CocoCaptionsSubset`: a thin wrapper around HuggingFace `datasets` for
  COCO Captions. Loads only the first N items so we fit in 12GB VRAM.
  Optional — only used if the user installs `datasets` and downloads the
  data.

For the demo and quick verification, we use the synthetic dataset. For
real training, the user can plug in their own data using the same
interface.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .encoder import Modality
from .config import WEVAConfig


# ── Synthetic dataset ──────────────────────────────────────────────────────


def _synthetic_text(item_id: int, vocab_size: int, max_len: int) -> torch.LongTensor:
    """Generate a synthetic text sequence that is *similar* across items in
    the same class. item_id encodes the class.
    """
    # First token = class index (mod vocab_size). The rest is random padding.
    cls = (item_id * 7 + 3) % 1024
    seq = [cls] + [
        (cls + i * 13 + item_id * 5) % vocab_size for i in range(max_len - 1)
    ]
    return torch.tensor(seq, dtype=torch.long)


def _synthetic_image(item_id: int, image_size: int) -> torch.Tensor:
    """Synthetic image: 3xHxW with a deterministic low-frequency pattern
    that varies with the item id.
    """
    H = W = image_size
    yy = torch.linspace(0, 2 * math.pi, H).unsqueeze(1).expand(H, W)
    xx = torch.linspace(0, 2 * math.pi, W).unsqueeze(0).expand(H, W)
    phase = (item_id % 16) * (math.pi / 8)
    pattern = (
        torch.sin(xx + phase) * torch.cos(yy + phase / 2)
        + 0.5 * torch.sin(2 * xx + 2 * phase)
    )
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
    rgb = torch.stack([pattern, pattern * 0.8, pattern * 0.6], dim=0)
    return rgb  # (3, H, W)


def _synthetic_audio(item_id: int, n_samples: int = 16000 * 2,
                     sr: int = 16000) -> torch.Tensor:
    """Synthetic audio: a sum of two sinusoids whose frequencies encode
    the item id.
    """
    t = torch.linspace(0, n_samples / sr, n_samples)
    f1 = 220 + (item_id % 12) * 30
    f2 = 440 + (item_id % 7) * 50
    wave = (
        0.5 * torch.sin(2 * math.pi * f1 * t)
        + 0.3 * torch.sin(2 * math.pi * f2 * t)
    )
    return wave  # (n_samples,)


def _synthetic_video(item_id: int, n_frames: int, image_size: int) -> torch.Tensor:
    """Synthetic video: a stack of n_frames synthetic images with a temporal
    rotation of the pattern phase.
    """
    frames = []
    for t in range(n_frames):
        # Reuse _synthetic_image with a phase shift
        frames.append(_synthetic_image(item_id * 7 + t, image_size))
    return torch.stack(frames, dim=0)  # (T, 3, H, W)


class SyntheticMultimodalDataset(Dataset):
    """A self-contained synthetic dataset for end-to-end testing.

    Each item contains all four modalities: text, image, audio, video.
    The item_id is the same for all four modalities in a single sample, so
    the contrastive loss can pair them correctly.

    Args:
        n_samples: number of distinct items.
        config: WEVA config (used for sizes).
    """

    def __init__(self, n_samples: int, config: WEVAConfig, audio_seconds: float = 2.0):
        self.n_samples = n_samples
        self.config = config
        self.audio_seconds = audio_seconds

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cfg = self.config
        return {
            Modality.TEXT.value: _synthetic_text(idx, cfg.text_vocab_size, cfg.text_max_len),
            Modality.IMAGE.value: _synthetic_image(idx, cfg.image_size),
            Modality.AUDIO.value: _synthetic_audio(
                idx, n_samples=int(16000 * self.audio_seconds)
            ),
            Modality.VIDEO.value: _synthetic_video(idx, cfg.video_max_frames, cfg.image_size),
        }


# ── Optional real-data wrapper ─────────────────────────────────────────────


class CocoCaptionsSubset(Dataset):
    """Optional thin wrapper for COCO Captions via HuggingFace `datasets`.

    Only used if the user has installed `datasets` and downloaded the data.
    The first caption per image is used. Audio and video are not used for
    this dataset — pass an empty dict for those.

    For full multimodal training, swap in your own dataset that yields
    (text, image, audio, video) tuples.
    """

    def __init__(self, n_samples: int, image_size: int, text_max_len: int):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Please install `datasets` to use CocoCaptionsSubset: "
                "pip install datasets"
            ) from e
        self.ds = load_dataset(
            "yerevann/coco-karpathy", split="train",
        ).select(range(n_samples))
        self.image_size = image_size
        self.text_max_len = text_max_len
        from torchvision import transforms
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.ds[idx]
        # Caption -> token ids (use a simple whitespace split for demo;
        # replace with a real tokenizer for serious training).
        text = item["caption"][0] if isinstance(item["caption"], list) else item["caption"]
        tokens = [hash(w) % self.text_max_len for w in text.split()[: self.text_max_len]]
        tokens = tokens + [0] * (self.text_max_len - len(tokens))
        return {
            Modality.TEXT.value: torch.tensor(tokens, dtype=torch.long),
            Modality.IMAGE.value: self.tf(item["image"].convert("RGB")),
        }


# ── Collate ───────────────────────────────────────────────────────────────


def multimodal_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-item dicts into a batched dict.

    All tensors are assumed to have the same shape per modality across the
    batch (we use synthetic data, so this holds). For variable-length
    inputs (e.g., real text), pad before calling this function.
    """
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
