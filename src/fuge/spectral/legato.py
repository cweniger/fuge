"""Legato: smooth linking of chirp tokens across window boundaries.

Finds matching tokens in adjacent windows (by frequency, phase, and
amplitude continuity), links them into chains, and enriches the tokens:
- Boundary frequencies and amplitudes are averaged to agree.
- Boundary phases are split-corrected for coherence.
- score is replaced with accumulated chain score: sqrt(sum s_i^2).
- A chain ID is assigned to each token.

Input and output are flat (B, N, 9) ChirpTokens.  Window structure is
recovered from t_start values (tokens from the same window share the
same t_start).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from fuge.spectral.tokens import ChirpTokens, LinkedChirpTokens, N_BASE


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


def _group_by_window(tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Group flat tokens (N, C) by window using t_start values.

    Returns
    -------
    window_times : sorted unique t_start values
    groups : list of 1D index tensors, one per window
    """
    t_start = tokens[:, 1]
    window_times, inverse = t_start.unique(sorted=True, return_inverse=True)
    groups = []
    for w in range(len(window_times)):
        groups.append((inverse == w).nonzero(as_tuple=True)[0])
    return window_times, groups


def _build_dag(tokens: torch.Tensor, groups: list[torch.Tensor],
               cfg: ChirpLinkConfig
               ) -> tuple[list[list[tuple[int, int]]], dict]:
    """Build DAG of compatible tokens and resolve into chains.

    Parameters
    ----------
    tokens : (N, C) tensor for a single batch element (C >= 9).
    groups : list of index tensors per window.
    cfg : matching thresholds.

    Returns
    -------
    edges : list of list of (idx_prev, idx_next) per window boundary.
        Indices are flat into the (N,) dimension.
    chains : dict mapping flat_idx -> (total_snr_sq, [path of flat indices]).
    """
    W = len(groups)

    score = tokens[:, 0]
    f_start = tokens[:, 3]
    f_end = tokens[:, 4]
    A_start = tokens[:, 5]
    A_end = tokens[:, 6]
    ps = tokens[:, 7]
    pe = tokens[:, 8]

    edges = []
    for w in range(W - 1):
        window_edges = []
        for ip in groups[w]:
            ip = ip.item()
            if score[ip] <= 0:
                continue
            for in_ in groups[w + 1]:
                in_ = in_.item()
                if score[in_] <= 0:
                    continue
                fe = f_end[ip]
                fs = f_start[in_]
                f_mean = (fe + fs) / 2
                if f_mean > 0 and abs(fe - fs) / f_mean > cfg.max_df:
                    continue
                dphi = _wrap(ps[in_] - pe[ip])
                if abs(dphi) > cfg.max_dphi:
                    continue
                ae = A_end[ip]
                an = A_start[in_]
                a_max = max(ae, an)
                if a_max > 0 and abs(ae - an) / a_max > cfg.max_dA:
                    continue
                window_edges.append((ip, in_))
        edges.append(window_edges)

    # DP: best chain ending at each flat index, accumulating score².
    # Using score² so greedy selection maximizes sqrt(Σ s_i²),
    # consistent with the enrichment step.
    chains: dict[int, tuple[float, list[int]]] = {}

    for idx in groups[0]:
        idx = idx.item()
        if score[idx] > 0:
            s = score[idx].item()
            chains[idx] = (s * s, [idx])

    for w in range(W - 1):
        for ip, in_ in edges[w]:
            if ip not in chains:
                continue
            prev_score_sq, prev_path = chains[ip]
            s = score[in_].item()
            new_score_sq = prev_score_sq + s * s
            if in_ not in chains or chains[in_][0] < new_score_sq:
                chains[in_] = (new_score_sq, prev_path + [in_])

    # Seed unreached tokens.
    for w in range(1, W):
        for idx in groups[w]:
            idx = idx.item()
            if score[idx] > 0 and idx not in chains:
                s = score[idx].item()
                chains[idx] = (s * s, [idx])

    return edges, chains


def _greedy_assign(chains: dict) -> list[list[int]]:
    """Greedily assign non-overlapping chains by total score².

    Returns list of paths (each path is a list of flat token indices).
    """
    all_chains = []
    for end_idx, (total_snr_sq, path) in chains.items():
        all_chains.append((total_snr_sq, path))

    all_chains.sort(key=lambda x: x[0], reverse=True)

    used = set()
    result = []
    for total_snr_sq, path in all_chains:
        if any(idx in used for idx in path):
            continue
        for idx in path:
            used.add(idx)
        result.append(path)

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
    def forward(self, tokens: ChirpTokens) -> LinkedChirpTokens:
        """Link tokens and return LinkedChirpTokens.

        Parameters
        ----------
        tokens : ChirpTokens with shape (B, N, 9)

        Returns
        -------
        linked : LinkedChirpTokens
            Same 9-field data (with updated score/boundaries) plus
            a separate chain_id LongTensor (B, N).
        """
        B, N, C = tokens.shape
        assert C >= N_BASE

        data = tokens.data[..., :N_BASE].clone()
        chain_id = torch.full(
            (B, N), -1, device=tokens.device, dtype=torch.long)

        chain_counter = 0
        for b in range(B):
            chain_counter = self._enrich_single(
                data[b], chain_id[b], chain_counter)

        return LinkedChirpTokens(data, chain_id)

    def _enrich_single(self, tokens: torch.Tensor,
                       chain_id: torch.Tensor,
                       chain_counter: int) -> int:
        """Enrich one batch element in-place.

        Parameters
        ----------
        tokens : (N, 9) float tensor, modified in-place.
        chain_id : (N,) long tensor, modified in-place.
        chain_counter : current chain ID counter.

        Returns updated chain_counter.
        """
        _, groups = _group_by_window(tokens)
        edges, chains = _build_dag(tokens, groups, self.config)
        assignments = _greedy_assign(chains)

        for path in assignments:
            V = len(path)
            if V < self.min_length:
                continue

            # Assign chain ID.
            for idx in path:
                chain_id[idx] = chain_counter
            chain_counter += 1

            # Accumulated score: sqrt(sum s_i^2).
            score_sq_sum = 0.0
            for idx in path:
                score_sq_sum += tokens[idx, 0].item() ** 2
            score_combined = score_sq_sum ** 0.5

            for idx in path:
                tokens[idx, 0] = score_combined

            # Smooth boundaries between consecutive tokens in the chain.
            for i in range(V - 1):
                ip, in_ = path[i], path[i + 1]

                # Average frequency at boundary.
                f_avg = (tokens[ip, 4] + tokens[in_, 3]) / 2
                tokens[ip, 4] = f_avg
                tokens[in_, 3] = f_avg

                # Average amplitude at boundary.
                A_avg = (tokens[ip, 6] + tokens[in_, 5]) / 2
                tokens[ip, 6] = A_avg
                tokens[in_, 5] = A_avg

                # Phase: split boundary correction evenly.
                pe_prev = tokens[ip, 8]
                ps_next = tokens[in_, 7]
                correction = _wrap(ps_next - pe_prev)
                half_corr = correction / 2
                tokens[ip, 8] = pe_prev + half_corr
                tokens[in_, 7] = ps_next - half_corr

        return chain_counter
