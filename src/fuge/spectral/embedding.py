"""Tone token embedding: raw (B, W, K, 9) tokens -> (B, W*K, n_embed).

Transforms raw tone features (snr, t_start, t_end, f_start, f_end,
A_start, A_end, phase_start, phase_end) into model-ready embedded
features with z-score normalization.

Input token format (from ToneTokenizer):
  snr: peak amplitude / noise std (log1p applied here)
  t_start, t_end: normalized time in [-1, 1]
  f_start, f_end: normalized frequency in [-1, 1]
  A_start, A_end: boundary amplitudes (positive, unbounded; log1p applied here)
  phase_start, phase_end: wrapped to [-pi, pi]
"""

import torch
import torch.nn as nn


class ToneTokenEmbedding(nn.Module):
    """Embed raw spectral peak tokens into model-ready features.

    Applies log1p to snr and amplitudes, cos/sin to phases, passes time
    and frequency through directly, then z-score normalizes.
    Each peak becomes an independent token in the output sequence.

    Parameters
    ----------
    phase_mode : str
        "center": use (phase_start + phase_end) / 2  -> n_embed = 9
        "boundary": keep both phase endpoints         -> n_embed = 11
    mask_phases : bool
        Zero out phase features (for ablation studies).
    """

    N_EMBED = {"center": 9, "boundary": 11}

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
        raw_tokens : Tensor, shape (B, W, K, 9)
        """
        embedded = self._embed(raw_tokens)
        flat = embedded.reshape(-1, self.n_embed)
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0).clamp(min=1e-8)

    def _embed(self, raw_tokens):
        """Apply feature transforms (before z-scoring).

        (B, W, K, 9) -> (B, W, K, n_embed)
        """
        snr = torch.log1p(raw_tokens[..., 0])
        t_start = raw_tokens[..., 1]
        t_end = raw_tokens[..., 2]
        f_start = raw_tokens[..., 3]
        f_end = raw_tokens[..., 4]
        a_start = torch.log1p(raw_tokens[..., 5])
        a_end = torch.log1p(raw_tokens[..., 6])
        ps = raw_tokens[..., 7]
        pe = raw_tokens[..., 8]

        if self.phase_mode == "center":
            phi = (ps + pe) / 2
            out = torch.stack([snr, t_start, t_end, f_start, f_end,
                               a_start, a_end,
                               torch.cos(phi), torch.sin(phi)], dim=-1)
        else:
            out = torch.stack([snr, t_start, t_end, f_start, f_end,
                               a_start, a_end,
                               torch.cos(ps), torch.sin(ps),
                               torch.cos(pe), torch.sin(pe)], dim=-1)

        if self.mask_phases:
            out[..., 7:] = 0.0

        return out

    def forward(self, raw_tokens):
        """Embed, normalize, and flatten peaks into sequence.

        Parameters
        ----------
        raw_tokens : Tensor, shape (B, W, K, 9)

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
