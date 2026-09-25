"""Model configurations for WEVA at three scales.

Total parameter counts (approximate, including projections and heads):
  - WEVA_TINY:    ~15M
  - WEVA_SMALL:   ~35M
  - WEVA_MEDIUM:  ~60M
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class WEVAConfig:
    """Configuration for a WEVA model.

    Attributes:
        d_model: Embedding dimension used across all modalities.
        n_layers: Number of SWM blocks in the shared backbone.
        n_heads: Number of SWM heads per block.
        d_ff: Hidden dimension of the FRN (SIREN FFN).
        omega_0: Initial frequency scale for SIREN activations.
        n_modalities: Number of modalities (4 for the headline result).
        proj_dim: Output projection dimension for contrastive loss.
        text_vocab_size: Size of the BPE vocabulary.
        text_max_len: Maximum text sequence length.
        image_size: Input image side length.
        image_patch_size: Patch size for image tokenizer.
        audio_n_fft: FFT size for STFT frontend.
        audio_hop: Hop length for STFT frontend.
        audio_max_frames: Maximum number of STFT frames per audio clip.
        video_max_frames: Maximum number of frames per video clip.
        dropout: Dropout probability throughout the backbone.
        initializer_range: Std of the truncated normal initializer.
    """

    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1536
    omega_0: float = 30.0
    n_modalities: int = 4
    proj_dim: int = 256

    text_vocab_size: int = 50272
    text_max_len: int = 77
    image_size: int = 224
    image_patch_size: int = 16
    audio_n_fft: int = 400
    audio_hop: int = 160
    audio_max_frames: int = 256
    video_max_frames: int = 8

    dropout: float = 0.1
    initializer_range: float = 0.02


WEVA_TINY = WEVAConfig(
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_ff=1024,
    text_max_len=64,
    image_size=128,
    audio_max_frames=128,
    video_max_frames=4,
)


WEVA_SMALL = WEVAConfig(
    d_model=384,
    n_layers=8,
    n_heads=6,
    d_ff=1536,
    text_max_len=77,
    image_size=224,
    audio_max_frames=256,
    video_max_frames=8,
)


WEVA_MEDIUM = WEVAConfig(
    d_model=512,
    n_layers=12,
    n_heads=8,
    d_ff=2048,
    text_max_len=96,
    image_size=224,
    audio_max_frames=256,
    video_max_frames=8,
)
