"""Token embedding and transformer backbone for spectral tokens.

TokenEmbedding: raw (B, W, K, 5) tokens -> embedded (B, W*K, n_embed)
TransformerEmbedding: raw tokens -> fixed-size (B, d_model) summary vector
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Embed raw spectral peak tokens into model-ready features.

    Applies cos/sin to phases, log1p to amplitude, then z-score normalizes.
    Each peak becomes an independent token in the output sequence.

    Parameters
    ----------
    phase_mode : str
        "center": use (phase_start + phase_end) / 2  -> n_embed = 5
        "boundary": keep both phase endpoints         -> n_embed = 7
    mask_phases : bool
        Zero out phase features (for ablation studies).
    """

    N_EMBED = {"center": 5, "boundary": 7}

    def __init__(self, phase_mode="center", mask_phases=False):
        super().__init__()
        self.phase_mode = phase_mode
        self.mask_phases = mask_phases
        self.n_embed = self.N_EMBED[phase_mode]
        self.register_buffer("mean", torch.zeros(self.n_embed))
        self.register_buffer("std", torch.ones(self.n_embed))

    def compute_normalization(self, raw_tokens):
        """Compute z-score stats from training tokens.

        Parameters
        ----------
        raw_tokens : Tensor, shape (B, W, K, 5)
        """
        embedded = self._embed(raw_tokens)
        flat = embedded.reshape(-1, self.n_embed)
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0).clamp(min=1e-8)

    def _embed(self, raw_tokens):
        """Apply feature transforms (before z-scoring).

        (B, W, K, 5) -> (B, W, K, n_embed)
        """
        freq = raw_tokens[..., 0]
        dlnf = raw_tokens[..., 1]
        amp = torch.log1p(raw_tokens[..., 2])
        ps = raw_tokens[..., 3]
        pe = raw_tokens[..., 4]

        if self.phase_mode == "center":
            phi = (ps + pe) / 2
            out = torch.stack([freq, dlnf, amp,
                               torch.cos(phi), torch.sin(phi)], dim=-1)
        else:
            out = torch.stack([freq, dlnf, amp,
                               torch.cos(ps), torch.sin(ps),
                               torch.cos(pe), torch.sin(pe)], dim=-1)

        if self.mask_phases:
            out[..., 3:] = 0.0

        return out

    def forward(self, raw_tokens):
        """Embed, normalize, and flatten peaks into sequence.

        Parameters
        ----------
        raw_tokens : Tensor, shape (B, W, K, 5)

        Returns
        -------
        embedded : Tensor, shape (B, W*K, n_embed)
        n_windows : int
        n_peaks : int
        """
        B, W, K, _ = raw_tokens.shape
        embedded = self._embed(raw_tokens)
        normalized = (embedded - self.mean) / self.std
        return normalized.reshape(B, W * K, self.n_embed), W, K


class TransformerEmbedding(nn.Module):
    """Transformer backbone: raw tokens -> fixed-size summary vector.

    Chains TokenEmbedding -> linear projection -> time-only positional
    encoding -> TransformerEncoder -> global average pool.

    Output is (B, d_model) — a fixed-size embedding suitable for
    downstream tasks (SBI posterior network, regression head, etc.).

    Parameters
    ----------
    token_embedding : TokenEmbedding
        Handles raw token -> embedded feature conversion.
    n_windows : int
        Number of time windows in the token sequence.
    n_peaks : int
        Number of peaks per time window.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, token_embedding, n_windows, n_peaks, d_model=64,
                 n_heads=4, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.token_embedding = token_embedding
        self.n_peaks = n_peaks
        self.input_proj = nn.Linear(token_embedding.n_embed, d_model)
        self.d_model = d_model

        # Time-only positional encoding: shared across K peaks per window
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_windows, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """Map raw tokens to a fixed-size embedding.

        Parameters
        ----------
        x : Tensor, shape (B, W, K, 5)
            Raw spectral tokens from SpectralTokenizer.

        Returns
        -------
        embedding : Tensor, shape (B, d_model)
        """
        B, W, K, _ = x.shape
        embedded, _, _ = self.token_embedding(x)    # (B, W*K, n_embed)
        projected = self.input_proj(embedded)        # (B, W*K, d_model)
        # Add time-only positional encoding (broadcast over K peaks)
        projected = projected.reshape(B, W, K, -1)
        projected = projected + self.pos_encoding
        projected = projected.reshape(B, W * K, -1)
        x = self.encoder(projected)
        return x.mean(dim=1)                         # (B, d_model)
