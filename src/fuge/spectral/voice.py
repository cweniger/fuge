"""Voice stitching: connect chirp tokens into phase-coherent voices.

Given chirp tokens (B, W, K, 9) from ChirpTokenizer, this module:
1. Builds a DAG of compatible tokens across adjacent windows.
2. Extracts voices as longest high-SNR paths through the DAG.
3. Either stitches into anchor-point voices (VoiceStitcher.forward)
   or enriches tokens via ChirpLinker (see legato.py).

Shared helpers (_wrap, _build_dag, _greedy_assign) are used by both
VoiceStitcher and ChirpLinker.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from fuge.spectral.tokens import ChirpTokens


@dataclass
class VoiceStitchConfig:
    """Matching thresholds for voice construction.

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


def _build_dag(tokens: torch.Tensor, cfg: VoiceStitchConfig
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


class VoiceStitcher(nn.Module):
    """Stitch chirp tokens into phase-coherent voices.

    Takes ChirpTokens (B, W, K, C) and produces anchor-point voices
    for analysis and visualization.

    The algorithm:
    1. For each pair of adjacent windows, find compatible token pairs
       (edges in a DAG).
    2. Resolve branching by keeping the highest-SNR path.
    3. Stitch phase coherently using within-window advances and
       boundary corrections.
    """

    def __init__(self, config: VoiceStitchConfig | None = None,
                 min_length: int = 2):
        super().__init__()
        self.config = config or VoiceStitchConfig()
        self.min_length = min_length

    @torch.no_grad()
    def forward(self, tokens: ChirpTokens | torch.Tensor
                ) -> list[list[torch.Tensor]]:
        """Stitch tokens into anchor-point voices.

        Parameters
        ----------
        tokens : ChirpTokens or Tensor, shape (B, W, K, C)

        Returns
        -------
        voices : list[list[Tensor]]
            voices[b] is a list of (V+1, 4) tensors: [A, t, phi, f].
        """
        data = tokens.data if isinstance(tokens, ChirpTokens) else tokens
        B, W, K, _ = data.shape
        result = []
        for b in range(B):
            edges, chains = _build_dag(data[b], self.config)
            assignments = _greedy_assign(chains, W)
            voices = []
            for w_start, path in assignments:
                if len(path) < self.min_length:
                    continue
                anchors = self._build_anchors(data[b], w_start, path)
                if anchors is not None:
                    voices.append(anchors)
            result.append(voices)
        return result

    def _build_anchors(
        self, tokens: torch.Tensor, w_start: int, path: list[int]
    ) -> torch.Tensor | None:
        """Build (V+1, 4) anchor points [A, t, phi, f] for a voice."""
        V = len(path)
        if V == 0:
            return None

        device = tokens.device
        dtype = tokens.dtype

        ws = torch.arange(w_start, w_start + V, device=device)
        ks = torch.tensor(path, device=device)
        tok = tokens[ws, ks]  # (V, C)

        t_s, t_e = tok[:, 1], tok[:, 2]
        f_s, f_e = tok[:, 3], tok[:, 4]
        A_s, A_e = tok[:, 5], tok[:, 6]
        ps, pe = tok[:, 7], tok[:, 8]

        anchors = torch.zeros(V + 1, 4, device=device, dtype=dtype)

        # Time
        anchors[0, 1] = t_s[0]
        anchors[V, 1] = t_e[V - 1]
        for i in range(1, V):
            anchors[i, 1] = (t_e[i - 1] + t_s[i]) / 2

        # Frequency
        anchors[0, 3] = f_s[0]
        anchors[V, 3] = f_e[V - 1]
        for i in range(1, V):
            anchors[i, 3] = (f_e[i - 1] + f_s[i]) / 2

        # Amplitude
        anchors[0, 0] = A_s[0]
        anchors[V, 0] = A_e[V - 1]
        for i in range(1, V):
            anchors[i, 0] = (A_e[i - 1] + A_s[i]) / 2

        # Phase: coherent stitching
        anchors[0, 2] = ps[0]
        if V >= 1:
            anchors[1, 2] = pe[0]
        for i in range(1, V):
            boundary_correction = _wrap(ps[i] - pe[i - 1])
            within_advance = pe[i] - ps[i]
            anchors[i + 1, 2] = anchors[i, 2] + boundary_correction + within_advance

        return anchors
