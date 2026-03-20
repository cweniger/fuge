"""Spectral signal analysis: STFT, peak finding, noise estimation, tokenization."""

from fuge.spectral.core import (
    DechirpSTFT, PeakFinder, NoiseModel,
    ChirpTokenizer, ToneTokenizer,
)
from fuge.spectral.embedding import ChirpTokenEmbedding, ToneTokenEmbedding

__all__ = [
    "DechirpSTFT", "PeakFinder", "NoiseModel",
    "ChirpTokenizer", "ChirpTokenEmbedding",
    # Backwards compatibility
    "ToneTokenizer", "ToneTokenEmbedding",
]
