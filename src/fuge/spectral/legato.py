"""Legato: smooth linking of chirp tokens across window boundaries.

Finds matching tokens in adjacent windows (by frequency, phase, and
amplitude continuity), links them into chains, and enriches the tokens:
- Boundary frequencies and amplitudes are averaged to agree.
- Boundary phases are split-corrected for coherence.
- SNR is replaced with accumulated chain SNR: sqrt(sum s_i^2).
- A chain ID is assigned to each token.

The output has the same (B, W, K) layout as the input, with one extra
field (chain_id), making it directly usable by downstream transformers.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from fuge.spectral.tokens import ChirpTokens, N_BASE


@dataclass
class ChirpLinkConfig:
    """Matching thresholds for token linking.

    Parameters
    ----------
    max_df : float
        Max relative frequency mismatch |f_end[w] - f_start[w+1]| / f_mean.
    max_dphi : float
        Max boundary phase residual |wrap(φ_start[w+1] - φ_end[w])| in radians.
    max_dA : float
        Max relative amplitude mismatch |A_end[w] - A_start[w+1]| / max(A).
    """
    max_df: float = 0.05
    max_dphi: float = 0.5
    max_dA: float = 0.5


def _wrap(x: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π]."""
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def _build_dag(tokens: torch.Tensor, cfg: ChirpLinkConfig
               ) -> tuple[list[list[tuple[int, int]]], dict]:
    """Build DAG of compatible tokens and resolve into chains.

    Parameters
    ----------
    tokens : (W, K, C) tensor for a single batch element (C >= 9).
    cfg : matching thresholds.

    Returns
    -------
    edges : list of list of (k_prev, k_next) per window boundary.
    chains : dict mapping (w, k) -> (total_snr, [path]).
    """
    W, K = tokens.shape[0], tokens.shape[1]

    snr = tokens[..., 0]
    f_start = tokens[..., 3]
    f_end = tokens[..., 4]
    A_start = tokens[..., 5]
    A_end = tokens[..., 6]
    ps = tokens[..., 7]
    pe = tokens[..., 8]

    edges = []
    for w in range(W - 1):
        window_edges = []
        for kp in range(K):
            if snr[w, kp] <= 0:
                continue
            for kn in range(K):
                if snr[w + 1, kn] <= 0:
                    continue
                fe = f_end[w, kp]
                fs = f_start[w + 1, kn]
                f_mean = (fe + fs) / 2
                if f_mean > 0 and abs(fe - fs) / f_mean > cfg.max_df:
                    continue
                dphi = _wrap(ps[w + 1, kn] - pe[w, kp])
                if abs(dphi) > cfg.max_dphi:
                    continue
                ae = A_end[w, kp]
                an = A_start[w + 1, kn]
                a_max = max(ae, an)
                if a_max > 0 and abs(ae - an) / a_max > cfg.max_dA:
                    continue
                window_edges.append((kp, kn))
        edges.append(window_edges)

    # DP: best chain ending at each (w, k).
    chains: dict[tuple[int, int], tuple[float, list[int]]] = {}

    for k_idx in range(K):
        if snr[0, k_idx] > 0:
            chains[(0, k_idx)] = (snr[0, k_idx].item(), [k_idx])

    for w in range(W - 1):
        for kp, kn in edges[w]:
            if (w, kp) not in chains:
                continue
            prev_snr, prev_path = chains[(w, kp)]
            new_snr = prev_snr + snr[w + 1, kn].item()
            if (w + 1, kn) not in chains or chains[(w + 1, kn)][0] < new_snr:
                chains[(w + 1, kn)] = (new_snr, prev_path + [kn])

    # Seed unreached tokens.
    for w in range(1, W):
        for k_idx in range(K):
            if snr[w, k_idx] > 0 and (w, k_idx) not in chains:
                chains[(w, k_idx)] = (snr[w, k_idx].item(), [k_idx])

    return edges, chains


def _greedy_assign(chains: dict, W: int) -> list[tuple[int, list[int]]]:
    """Greedily assign non-overlapping chains by total SNR.

    Returns list of (w_start, path) tuples.
    """
    all_chains = []
    for (w, k_idx), (total_snr, path) in chains.items():
        w_start = w - len(path) + 1
        all_chains.append((total_snr, w_start, path))

    all_chains.sort(key=lambda x: x[0], reverse=True)

    used = set()
    result = []
    for total_snr, w_start, path in all_chains:
        slots = [(w_start + i, path[i]) for i in range(len(path))]
        if any(s in used for s in slots):
            continue
        for s in slots:
            used.add(s)
        result.append((w_start, path))

    return result


class ChirpLinker(nn.Module):
    """Link chirp tokens across windows with boundary smoothing.

    Parameters
    ----------
    config : ChirpLinkConfig or None
        Matching thresholds (max_df, max_dphi, max_dA).
    min_length : int
        Minimum chain length to enrich. Shorter chains are left as-is
        with chain_id = -1.
    """

    def __init__(self, config: ChirpLinkConfig | None = None,
                 min_length: int = 2):
        super().__init__()
        self.config = config or ChirpLinkConfig()
        self.min_length = min_length

    @torch.no_grad()
    def forward(self, tokens: ChirpTokens) -> ChirpTokens:
        """Link tokens and return enriched ChirpTokens.

        Parameters
        ----------
        tokens : ChirpTokens with shape (B, W, K, 9)

        Returns
        -------
        enriched : ChirpTokens with shape (B, W, K, 10)
            Same fields plus chain_id at index 9.
        """
        B, W, K, C = tokens.shape
        assert C >= N_BASE

        # Append chain_id column initialized to -1.
        chain_col = torch.full(
            (B, W, K, 1), -1.0, device=tokens.device, dtype=tokens.dtype)
        enriched = torch.cat([tokens.data[..., :N_BASE], chain_col], dim=-1)

        chain_counter = 0
        for b in range(B):
            chain_counter = self._enrich_single(
                enriched[b], chain_counter)

        return ChirpTokens(enriched)

    def _enrich_single(self, tokens: torch.Tensor,
                       chain_counter: int) -> int:
        """Enrich one batch element in-place: (W, K, 10).

        Returns updated chain_counter.
        """
        W, K, _ = tokens.shape

        # Use the first 9 fields for DAG building.
        edges, chains = _build_dag(tokens[..., :N_BASE], self.config)
        assignments = _greedy_assign(chains, W)

        for w_start, path in assignments:
            V = len(path)
            if V < self.min_length:
                continue

            # Assign chain ID.
            for i in range(V):
                tokens[w_start + i, path[i], 9] = chain_counter
            chain_counter += 1

            # Accumulated SNR: sqrt(sum s_i^2).
            snr_sq_sum = 0.0
            for i in range(V):
                snr_sq_sum += tokens[w_start + i, path[i], 0].item() ** 2
            snr_combined = snr_sq_sum ** 0.5

            for i in range(V):
                tokens[w_start + i, path[i], 0] = snr_combined

            # Smooth boundaries between consecutive tokens.
            for i in range(V - 1):
                wp, kp = w_start + i, path[i]
                wn, kn = w_start + i + 1, path[i + 1]

                # Average frequency at boundary.
                f_avg = (tokens[wp, kp, 4] + tokens[wn, kn, 3]) / 2
                tokens[wp, kp, 4] = f_avg
                tokens[wn, kn, 3] = f_avg

                # Average amplitude at boundary.
                A_avg = (tokens[wp, kp, 6] + tokens[wn, kn, 5]) / 2
                tokens[wp, kp, 6] = A_avg
                tokens[wn, kn, 5] = A_avg

                # Phase: split boundary correction evenly.
                pe_prev = tokens[wp, kp, 8]
                ps_next = tokens[wn, kn, 7]
                correction = _wrap(ps_next - pe_prev)
                half_corr = correction / 2
                tokens[wp, kp, 8] = pe_prev + half_corr
                tokens[wn, kn, 7] = ps_next - half_corr

        return chain_counter
