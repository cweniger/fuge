"""Chirp token embedding: raw (B, W, K, 9) tokens -> (B, W*K, n_embed).

Transforms raw chirp token features via harmonic embeddings — multi-scale
sin/cos representations that give the network sensitivity at all relevant
scales.  Four parameter types (time, frequency, amplitude, phase), each
embedded at both boundaries (_start, _end), plus log1p(SNR) as a scalar.

Two embedding types:
  HarmonicEmbedding: for bounded scalar parameters (time, frequency,
      amplitude).  Log-uniformly spaced frequencies from 2pi/range to
      2pi/resolution.
  HarmonicPhaseEmbedding: for phase (unwrapped radians).  Anchored at
      sin(phi), cos(phi) with powers of 2 extending down (for unwrapped
      phase comparison) and up (for fine wrapped-phase matching).

Input token format (from ChirpTokenizer):
  snr: peak amplitude / noise std (log1p scalar, not harmonically embedded)
  t_start, t_end: absolute sample indices
  f_start, f_end: frequency in cycles/sample (0 to 0.5)
  A_start, A_end: boundary amplitudes
  phase_start, phase_end: unwrapped radians
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from fuge.spectral.tokens import ChirpTokens


def _to_tensor(x):
    """Accept ChirpTokens or raw Tensor, return the underlying tensor."""
    return x.data if isinstance(x, ChirpTokens) else x


# ── Harmonic embedding for bounded scalars ─────────────────────────────

@dataclass
class HarmonicEmbeddingConfig:
    """Configuration for a bounded-scalar harmonic embedding.

    Parameters
    ----------
    v_min : float
        Lower bound of parameter range.
    v_max : float
        Upper bound of parameter range.
    resolution : float
        Finest scale to resolve.  Sets the highest embedding frequency.
    modes_per_octave : int
        Number of sin/cos modes per octave of frequency range.
        Default 1 (one mode per octave = Nyquist baseline).
        Higher values give incommensurate frequencies and cleaner sidelobes.
    """
    v_min: float
    v_max: float
    resolution: float
    modes_per_octave: int = 1

    @property
    def n_harmonics(self):
        R = self.v_max - self.v_min
        n_octaves = math.log2(R / self.resolution)
        return max(1, math.ceil(self.modes_per_octave * n_octaves) + 1)

    @property
    def n_embed(self):
        return 2 * self.n_harmonics


class HarmonicEmbedding(nn.Module):
    """Multi-scale sin/cos embedding for a bounded scalar parameter.

    Maps v -> [sin(w_0*v'), cos(w_0*v'), ..., sin(w_{n-1}*v'), cos(w_{n-1}*v')]
    where v' = v - v_min.

    Frequencies are log-uniformly spaced from 2*pi/R to 2*pi/resolution,
    giving n_harmonics modes.  With modes_per_octave > 1, the frequencies
    are incommensurate (not powers of 2), producing smoother dot-product
    kernels.
    """

    def __init__(self, config: HarmonicEmbeddingConfig):
        super().__init__()
        n = config.n_harmonics
        f_min = 2 * math.pi / (config.v_max - config.v_min)
        f_max = 2 * math.pi / config.resolution
        if n == 1:
            freqs = torch.tensor([f_min])
        else:
            freqs = f_min * torch.pow(
                f_max / f_min,
                torch.arange(n).float() / (n - 1),
            )
        self.register_buffer('freqs', freqs)
        self.v_min = config.v_min

    def forward(self, v):
        """Embed scalar values.

        Parameters
        ----------
        v : Tensor, arbitrary shape (...)

        Returns
        -------
        Tensor, shape (..., 2 * n_harmonics)
        """
        angles = (v - self.v_min).unsqueeze(-1) * self.freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


# ── Harmonic phase embedding for unwrapped angles ──────────────────────

@dataclass
class HarmonicPhaseEmbeddingConfig:
    """Configuration for a phase harmonic embedding.

    The embedding is anchored at the natural period: sin(phi), cos(phi).
    Powers of 2 extend downward (sub-harmonics for unwrapped phase
    comparison across many cycles) and upward (super-harmonics for fine
    wrapped-phase matching).

    Parameters
    ----------
    phi_max : float
        Maximum unwrapped phase extent (radians).  Sets the lowest
        embedding frequency: f_min = 2*pi / phi_max.  Determines how
        many sub-harmonic octaves below the anchor.
    phi_resolution : float
        Finest phase resolution (radians).  Sets the highest embedding
        frequency: f_max = 2*pi / phi_resolution.  Determines how many
        super-harmonic octaves above the anchor.
    """
    phi_max: float
    phi_resolution: float

    @property
    def n_low(self):
        """Number of sub-harmonic modes below the anchor."""
        return max(0, math.ceil(math.log2(self.phi_max / (2 * math.pi))))

    @property
    def n_high(self):
        """Number of super-harmonic modes above the anchor."""
        return max(0, math.ceil(math.log2(2 * math.pi / self.phi_resolution)))

    @property
    def n_harmonics(self):
        return self.n_low + 1 + self.n_high  # +1 for the anchor

    @property
    def n_embed(self):
        return 2 * self.n_harmonics


class HarmonicPhaseEmbedding(nn.Module):
    """Multi-scale sin/cos embedding for unwrapped phase.

    Anchored at sin(phi), cos(phi) (the natural period).  Extends with
    powers of 2 downward (for unwrapped phase spanning many cycles) and
    upward (for fine wrapped-phase matching).

    The anchor mode is the only one that exactly respects 2*pi periodicity.
    Sub-harmonics below it enable comparison of accumulated phase across
    long chains; super-harmonics above it resolve sub-radian differences.
    """

    def __init__(self, config: HarmonicPhaseEmbeddingConfig):
        super().__init__()
        n_low = config.n_low
        n_high = config.n_high
        # Powers of 2: ..., 2^{-2}, 2^{-1}, 2^0 (anchor), 2^1, 2^2, ...
        exponents = torch.arange(-n_low, n_high + 1).float()
        freqs = torch.pow(2.0, exponents)
        self.register_buffer('freqs', freqs)

    def forward(self, phi):
        """Embed phase values.

        Parameters
        ----------
        phi : Tensor, arbitrary shape (...), unwrapped radians

        Returns
        -------
        Tensor, shape (..., 2 * n_harmonics)
        """
        angles = phi.unsqueeze(-1) * self.freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


# ── Chirp token embedding ──────────────────────────────────────────────

class ChirpTokenEmbedding(nn.Module):
    """Embed raw chirp tokens via harmonic embeddings.

    Three bounded-scalar parameter types (time, freq, amp) each get a
    HarmonicEmbedding at both boundaries.  Phase gets a
    HarmonicPhaseEmbedding at both boundaries.  Plus log1p(SNR) as a
    scalar.  Total: 6 harmonic + 2 phase embeddings + 1 scalar.

    Parameters
    ----------
    time : HarmonicEmbeddingConfig
        Range and resolution for sample-index time values.
    freq : HarmonicEmbeddingConfig
        Range and resolution for frequency (cycles/sample).
    amp : HarmonicEmbeddingConfig
        Range and resolution for boundary amplitudes.
    phase : HarmonicPhaseEmbeddingConfig
        Phase extent and resolution for unwrapped phase.
    """

    def __init__(self, time, freq, amp, phase):
        super().__init__()
        self.embed_time = HarmonicEmbedding(time)
        self.embed_freq = HarmonicEmbedding(freq)
        self.embed_amp = HarmonicEmbedding(amp)
        self.embed_phase = HarmonicPhaseEmbedding(phase)

        self.n_embed = (
            time.n_embed * 2     # t_start, t_end
            + freq.n_embed * 2   # f_start, f_end
            + amp.n_embed * 2    # A_start, A_end
            + phase.n_embed * 2  # phase_start, phase_end
            + 1                  # log1p(snr)
        )

    def forward(self, raw_tokens):
        """Embed and flatten peaks into sequence.

        Parameters
        ----------
        raw_tokens : ChirpTokens or Tensor, shape (B, W, K, C)

        Returns
        -------
        embedded : Tensor, shape (B, W*K, n_embed)
        n_windows : int
        n_peaks : int
        """
        raw_tokens = _to_tensor(raw_tokens)
        B, W, K, _ = raw_tokens.shape

        snr = torch.log1p(raw_tokens[..., 0:1])

        t_s = self.embed_time(raw_tokens[..., 1])
        t_e = self.embed_time(raw_tokens[..., 2])

        f_s = self.embed_freq(raw_tokens[..., 3])
        f_e = self.embed_freq(raw_tokens[..., 4])

        a_s = self.embed_amp(raw_tokens[..., 5])
        a_e = self.embed_amp(raw_tokens[..., 6])

        ph_s = self.embed_phase(raw_tokens[..., 7])
        ph_e = self.embed_phase(raw_tokens[..., 8])

        embedded = torch.cat(
            [snr, t_s, t_e, f_s, f_e, a_s, a_e, ph_s, ph_e],
            dim=-1,
        )
        return embedded.reshape(B, W * K, self.n_embed), W, K
