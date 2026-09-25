"""WEVA: Wave-Energy Vibrational Architecture.

A unified multimodal architecture that processes text, image, audio, and
video through one shared spectral encoder and one shared Spectral Wave
Mixer (SWM) backbone.

Reference: WEVA Design Document (Proposal A), 2026-06-11.
"""

from .config import WEVAConfig, WEVA_TINY, WEVA_SMALL, WEVA_MEDIUM
from .encoder import UniversalSpectralEncoder, Modality
from .mixer import SpectralWaveMixer, MultiHeadSWM
from .frn import FeedforwardResonanceNetwork
from .model import WEVAModel
from .losses import MultimodalContrastiveLoss, OrbitalEnergyRegularizer

__all__ = [
    "WEVAConfig",
    "WEVA_TINY",
    "WEVA_SMALL",
    "WEVA_MEDIUM",
    "UniversalSpectralEncoder",
    "Modality",
    "SpectralWaveMixer",
    "MultiHeadSWM",
    "FeedforwardResonanceNetwork",
    "WEVAModel",
    "MultimodalContrastiveLoss",
    "OrbitalEnergyRegularizer",
]
