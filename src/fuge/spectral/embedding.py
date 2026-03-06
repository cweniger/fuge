"""Tone token embedding: raw (B, W, K, 5) tokens -> (B, W*K, n_embed).

Transforms raw tone features (f_start, f_end, amp, phase_start,
phase_end) into model-ready embedded features with z-score normalization.

Input token format (from ToneTokenizer):
  f_start, f_end: normalized frequency in [-1, 1]
  amp: amplitude (positive, unbounded; log1p applied here)
  phase_start, phase_end: wrapped to [-pi, pi]
"""

import torch
import torch.nn as nn


class ToneTokenEmbedding(nn.Module):
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
        f_start = raw_tokens[..., 0]
        f_end = raw_tokens[..., 1]
        amp = torch.log1p(raw_tokens[..., 2])
        ps = raw_tokens[..., 3]
        pe = raw_tokens[..., 4]

        if self.phase_mode == "center":
            phi = (ps + pe) / 2
            out = torch.stack([f_start, f_end, amp,
                               torch.cos(phi), torch.sin(phi)], dim=-1)
        else:
            out = torch.stack([f_start, f_end, amp,
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
