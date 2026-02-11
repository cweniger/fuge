"""fuge: Scientific signal embeddings."""

from fuge.spectral import DechirpSTFT, ToneTokenizer, ToneTokenEmbedding
from fuge.nn import TransformerEmbedding
from fuge.emri import emri_signal

__all__ = [
    "DechirpSTFT", "ToneTokenizer", "ToneTokenEmbedding",
    "TransformerEmbedding",
    "emri_signal",
]
