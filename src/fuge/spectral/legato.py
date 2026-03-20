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

from fuge.spectral.tokens import ChirpTokens, N_BASE
from fuge.spectral.voice import VoiceStitchConfig, _build_dag, _greedy_assign, _wrap


class ChirpLinker(nn.Module):
    """Link chirp tokens across windows with boundary smoothing.

    Parameters
    ----------
    config : VoiceStitchConfig or None
        Matching thresholds (max_df, max_dphi, max_dA).
    min_length : int
        Minimum chain length to enrich. Shorter chains are left as-is
        with chain_id = -1.
    """

    def __init__(self, config: VoiceStitchConfig | None = None,
                 min_length: int = 2):
        super().__init__()
        self.config = config or VoiceStitchConfig()
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
