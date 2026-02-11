"""Generic neural network building blocks for signal embeddings."""

import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    """Transformer backbone: pre-embedded tokens -> fixed-size summary vector.

    Linear projection -> positional encoding -> TransformerEncoder ->
    global average pool.

    Output is (B, d_model) -- a fixed-size embedding suitable for
    downstream tasks (SBI posterior network, regression head, etc.).

    Parameters
    ----------
    d_in : int
        Input feature dimension per token.
    seq_len : int
        Total sequence length (e.g. n_windows * n_peaks).
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

    def __init__(self, d_in, seq_len, d_model=64,
                 n_heads=4, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.d_model = d_model

        # Learnable positional encoding over the full sequence
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """Map pre-embedded tokens to a fixed-size embedding.

        Parameters
        ----------
        x : Tensor, shape (B, seq_len, d_in)
            Pre-embedded token features.

        Returns
        -------
        embedding : Tensor, shape (B, d_model)
        """
        projected = self.input_proj(x)           # (B, seq_len, d_model)
        projected = projected + self.pos_encoding
        x = self.encoder(projected)
        return x.mean(dim=1)                     # (B, d_model)
