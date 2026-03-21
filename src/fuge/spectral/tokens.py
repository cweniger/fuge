"""Structured chirp token containers.

ChirpTokens: thin wrapper around a (B, W, K, 9) tensor with named field access.
LinkedChirpTokens: subclass adding a separate (B, W, K) long tensor for chain IDs.
Both keep the underlying data contiguous and GPU-compatible.
"""

import torch


# Field indices for the 9-field token format.
SNR = 0
T_START = 1
T_END = 2
F_START = 3
F_END = 4
A_START = 5
A_END = 6
PHASE_START = 7
PHASE_END = 8

N_BASE = 9


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

    Parameters
    ----------
    data : Tensor, shape (B, W, K, 9)
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

    def __repr__(self):
        B, W, K, C = self.data.shape
        return f"ChirpTokens(B={B}, W={W}, K={K})"


class LinkedChirpTokens(ChirpTokens):
    """Chirp tokens with chain linking information.

    Extends ChirpTokens with a separate integer chain_id tensor.
    Tokens in the same chain share a chain_id >= 0; unlinked tokens
    have chain_id = -1.

    Parameters
    ----------
    data : Tensor, shape (B, W, K, 9)
        Token data (same 9 fields, with updated SNR/boundaries after linking).
    chain_id : LongTensor, shape (B, W, K)
        Chain assignment per token. -1 = unlinked.
    """

    def __init__(self, data: torch.Tensor, chain_id: torch.Tensor):
        super().__init__(data)
        assert chain_id.shape == data.shape[:3]
        assert chain_id.dtype == torch.long
        self._chain_id = chain_id

    @property
    def chain_id(self) -> torch.Tensor:
        return self._chain_id

    @property
    def n_chains(self) -> int:
        """Number of distinct chains (excluding unlinked tokens)."""
        return int((self._chain_id >= 0).any(dim=-1).any(dim=-1).sum()
                    if self._chain_id.numel() > 0
                    else 0)

    def __repr__(self):
        B, W, K, _ = self.data.shape
        ids = self._chain_id.unique()
        nc = (ids >= 0).sum().item()
        return f"LinkedChirpTokens(B={B}, W={W}, K={K}, chains={nc})"
