"""Structured chirp token container.

Thin wrapper around a (B, W, K, C) tensor with named field access.
The underlying tensor stays contiguous and GPU-compatible.
"""

import torch


# Field indices for the base 9-field token format.
SNR = 0
T_START = 1
T_END = 2
F_START = 3
F_END = 4
A_START = 5
A_END = 6
PHASE_START = 7
PHASE_END = 8
CHAIN_ID = 9

N_BASE = 9
N_LINKED = 10


class ChirpTokens:
    """Structured wrapper around chirp token tensor.

    Fields (per token):
        0: snr          — peak amplitude (or accumulated SNR after linking)
        1: t_start      — sample index at token start (t = -1/2)
        2: t_end        — sample index at token end (t = +1/2)
        3: f_start      — frequency at start (cycles/sample)
        4: f_end        — frequency at end (cycles/sample)
        5: A_start      — amplitude at start boundary
        6: A_end        — amplitude at end boundary
        7: phase_start  — phase at start boundary
        8: phase_end    — phase at end boundary
        9: chain_id     — linked chain ID (-1 = unlinked), added by ChirpLinker

    Parameters
    ----------
    data : Tensor, shape (B, W, K, C)
        C >= 9 (base tokens) or C >= 10 (after linking).
    """

    def __init__(self, data: torch.Tensor):
        assert data.ndim == 4 and data.shape[-1] >= N_BASE
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def snr(self) -> torch.Tensor:
        return self.data[..., SNR]

    @property
    def t_start(self) -> torch.Tensor:
        return self.data[..., T_START]

    @property
    def t_end(self) -> torch.Tensor:
        return self.data[..., T_END]

    @property
    def f_start(self) -> torch.Tensor:
        return self.data[..., F_START]

    @property
    def f_end(self) -> torch.Tensor:
        return self.data[..., F_END]

    @property
    def A_start(self) -> torch.Tensor:
        return self.data[..., A_START]

    @property
    def A_end(self) -> torch.Tensor:
        return self.data[..., A_END]

    @property
    def phase_start(self) -> torch.Tensor:
        return self.data[..., PHASE_START]

    @property
    def phase_end(self) -> torch.Tensor:
        return self.data[..., PHASE_END]

    @property
    def chain_id(self) -> torch.Tensor:
        """Chain ID (-1 = unlinked). Only available after linking."""
        assert self.data.shape[-1] >= N_LINKED, \
            "chain_id not available — run ChirpLinker first"
        return self.data[..., CHAIN_ID]

    @property
    def is_linked(self) -> bool:
        return self.data.shape[-1] >= N_LINKED

    def __repr__(self):
        B, W, K, C = self.data.shape
        linked = ", linked" if self.is_linked else ""
        return f"ChirpTokens(B={B}, W={W}, K={K}, C={C}{linked})"
