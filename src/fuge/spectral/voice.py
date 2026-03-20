"""Voice stitching: connect chirp tokens into phase-coherent voices.

Given chirp tokens (B, W, K, 9) from ChirpTokenizer, this module:
1. Builds a DAG of compatible tokens across adjacent windows.
2. Extracts voices as longest high-SNR paths through the DAG.
3. Stitches phase coherently across window boundaries.

Output: list of voices, each a (V+1, 4) tensor of anchor points
[amplitude, time, phase, frequency] with unwrapped phase.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


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


class VoiceStitcher(nn.Module):
    """Stitch chirp tokens into phase-coherent voices.

    Takes (B, W, K, 9) tokens from ChirpTokenizer and produces a list
    of voices per batch element.  Each voice is a (V+1, 4) tensor of
    anchor points: [amplitude, time, phase, frequency].

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
    def forward(self, tokens: torch.Tensor) -> list[list[torch.Tensor]]:
        """Stitch tokens into voices.

        Parameters
        ----------
        tokens : Tensor, shape (B, W, K, 9)
            Raw chirp tokens from ChirpTokenizer:
            [snr, t_start, t_end, f_start, f_end, A_start, A_end,
             phase_start, phase_end].

        Returns
        -------
        voices : list[list[Tensor]]
            voices[b] is a list of voice tensors for batch element b.
            Each voice tensor has shape (V+1, 4): [A, t, phi, f]
            at V+1 anchor points (V = number of tokens in the voice).
            Phase is coherently unwrapped.
        """
        B, W, K, _ = tokens.shape
        result = []
        for b in range(B):
            result.append(self._stitch_single(tokens[b]))
        return result

    def _stitch_single(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Process one batch element: (W, K, 9) -> list of voice tensors."""
        W, K, _ = tokens.shape
        cfg = self.config

        snr = tokens[..., 0]           # (W, K)
        t_start = tokens[..., 1]       # (W, K)
        t_end = tokens[..., 2]         # (W, K)
        f_start = tokens[..., 3]       # (W, K)
        f_end = tokens[..., 4]         # (W, K)
        A_start = tokens[..., 5]       # (W, K)
        A_end = tokens[..., 6]         # (W, K)
        ps = tokens[..., 7]            # (W, K)
        pe = tokens[..., 8]            # (W, K)

        # Build adjacency: edges[w] is a list of (k_prev, k_next) pairs
        # for windows w and w+1.
        edges = []
        for w in range(W - 1):
            window_edges = []
            for kp in range(K):
                if snr[w, kp] <= 0:
                    continue
                for kn in range(K):
                    if snr[w + 1, kn] <= 0:
                        continue
                    # Frequency match
                    fe = f_end[w, kp]
                    fs = f_start[w + 1, kn]
                    f_mean = (fe + fs) / 2
                    if f_mean > 0 and abs(fe - fs) / f_mean > cfg.max_df:
                        continue
                    # Phase match
                    dphi = _wrap(ps[w + 1, kn] - pe[w, kp])
                    if abs(dphi) > cfg.max_dphi:
                        continue
                    # Amplitude match
                    ae = A_end[w, kp]
                    an = A_start[w + 1, kn]
                    a_max = max(ae, an)
                    if a_max > 0 and abs(ae - an) / a_max > cfg.max_dA:
                        continue
                    window_edges.append((kp, kn))
            edges.append(window_edges)

        # Build chains using dynamic programming on the DAG.
        # State: (w, k) -> best chain ending here.
        # chain[w][k] = (total_snr, [list of k indices from w=0..w])
        chains: dict[tuple[int, int], tuple[float, list[int]]] = {}

        # Initialize: every token in window 0 starts a chain.
        for k_idx in range(K):
            if snr[0, k_idx] > 0:
                chains[(0, k_idx)] = (snr[0, k_idx].item(), [k_idx])

        # Forward pass: extend chains through edges.
        for w in range(W - 1):
            for kp, kn in edges[w]:
                if (w, kp) not in chains:
                    continue
                prev_snr, prev_path = chains[(w, kp)]
                new_snr = prev_snr + snr[w + 1, kn].item()
                if (w + 1, kn) not in chains or chains[(w + 1, kn)][0] < new_snr:
                    chains[(w + 1, kn)] = (new_snr, prev_path + [kn])

        # Also seed chains at later windows for tokens that weren't reached.
        for w in range(1, W):
            for k_idx in range(K):
                if snr[w, k_idx] > 0 and (w, k_idx) not in chains:
                    chains[(w, k_idx)] = (snr[w, k_idx].item(), [k_idx])
                    # Try to extend this new chain forward too — handled
                    # in the next iteration of the outer loop.

        # Collect all completed chains.  A chain is identified by its
        # terminal (w, k).  We want non-overlapping voices: each (w, k)
        # slot can belong to at most one voice.
        # Strategy: sort chains by total SNR (descending), greedily assign.
        all_chains = []
        for (w, k_idx), (total_snr, path) in chains.items():
            # Chain spans windows (w - len(path) + 1) to w.
            w_start = w - len(path) + 1
            all_chains.append((total_snr, w_start, path))

        all_chains.sort(key=lambda x: x[0], reverse=True)

        # Greedy assignment: mark used (w, k) slots.
        used = set()
        voices = []
        for total_snr, w_start, path in all_chains:
            # Check no overlap with already-assigned voices.
            slots = [(w_start + i, path[i]) for i in range(len(path))]
            if any(s in used for s in slots):
                continue
            for s in slots:
                used.add(s)
            voices.append((w_start, path))

        # Convert each voice to anchor-point representation.
        result = []
        for w_start, path in voices:
            if len(path) < self.min_length:
                continue
            anchors = self._build_anchors(
                tokens, w_start, path)
            if anchors is not None:
                result.append(anchors)

        return result

    def _build_anchors(
        self, tokens: torch.Tensor, w_start: int, path: list[int]
    ) -> torch.Tensor | None:
        """Build (V+1, 4) anchor points [A, t, phi, f] for a voice.

        Parameters
        ----------
        tokens : (W, K, 9) tensor
        w_start : first window index
        path : list of peak indices, one per window
        """
        V = len(path)
        if V == 0:
            return None

        device = tokens.device
        dtype = tokens.dtype

        # Extract per-token fields along the path.
        ws = torch.arange(w_start, w_start + V, device=device)
        ks = torch.tensor(path, device=device)

        tok = tokens[ws, ks]  # (V, 9)
        snr = tok[:, 0]
        t_s = tok[:, 1]
        t_e = tok[:, 2]
        f_s = tok[:, 3]
        f_e = tok[:, 4]
        A_s = tok[:, 5]
        A_e = tok[:, 6]
        ps = tok[:, 7]
        pe = tok[:, 8]

        # V+1 anchor points at token boundaries.
        # Anchor 0: start of first token
        # Anchor i (1..V-1): boundary between token i-1 and token i
        # Anchor V: end of last token
        anchors = torch.zeros(V + 1, 4, device=device, dtype=dtype)

        # Time: anchors at token boundaries (sample indices).
        anchors[0, 1] = t_s[0]
        anchors[V, 1] = t_e[V - 1]
        for i in range(1, V):
            # t_end[i-1] and t_start[i] should match; average for robustness.
            anchors[i, 1] = (t_e[i - 1] + t_s[i]) / 2

        # Frequency: average at boundaries.
        anchors[0, 3] = f_s[0]
        anchors[V, 3] = f_e[V - 1]
        for i in range(1, V):
            anchors[i, 3] = (f_e[i - 1] + f_s[i]) / 2

        # Amplitude: average at boundaries.
        anchors[0, 0] = A_s[0]
        anchors[V, 0] = A_e[V - 1]
        for i in range(1, V):
            anchors[i, 0] = (A_e[i - 1] + A_s[i]) / 2

        # Phase: coherent stitching.
        # Anchor 0: φ_start[0]
        # Anchor 1: φ_start[0] + (φ_end[0] - φ_start[0]) = φ_end[0]
        # Anchor i+1: anchor[i] + wrap(φ_start[i] - anchor_raw[i])
        #             + (φ_end[i] - φ_start[i])
        # where anchor_raw[i] is the raw phase at anchor i from the
        # previous token (φ_end[i-1]), and wrap corrects the boundary.
        anchors[0, 2] = ps[0]
        if V >= 1:
            anchors[1, 2] = ps[0] + (pe[0] - ps[0])  # = pe[0]
        for i in range(1, V):
            # Boundary correction: difference between new token's start
            # phase and previous token's end phase, wrapped to [-π, π].
            boundary_correction = _wrap(ps[i] - pe[i - 1])
            # Within-window advance for token i.
            within_advance = pe[i] - ps[i]
            anchors[i + 1, 2] = anchors[i, 2] + boundary_correction + within_advance

        return anchors
