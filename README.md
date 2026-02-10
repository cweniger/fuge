# fuge

Spectral tokenization toolkit for gravitational wave signals. Converts time-domain data into compact spectral tokens suitable for transformer-based inference, designed as an embedding layer for Simulation-Based Inference (SBI).

## Installation

```bash
pip install -e .
```

Requires Python 3.10+, PyTorch, JAX, NumPy, Matplotlib.

## Quick start

```python
import torch
from fuge import SpectralTokenizer, TokenEmbedding, TransformerEmbedding

# 1. Tokenize: time-domain signal -> raw spectral tokens
tokenizer = SpectralTokenizer(k=1024, n_peaks=3).to(device)
tokens = tokenizer(signals)  # (B, N) -> (B, W, K, 5)

# 2. Embed: raw tokens -> fixed-size vector for downstream tasks
token_emb = TokenEmbedding(phase_mode="center")
token_emb.compute_normalization(train_tokens)

backbone = TransformerEmbedding(
    token_emb, n_windows=W, n_peaks=3, d_model=64,
)
embedding = backbone(tokens)  # (B, W, K, 5) -> (B, d_model)

# Use `embedding` as input to SBI posterior network, regression head, etc.
```

## Core modules

### `SpectralTokenizer` — Signal to tokens

Chains de-chirped STFT, peak finding, and phase extraction into a single batched `forward()` call.

```
(B, N) signal -> (B, W, K, 5) raw tokens
```

Each token represents one spectral peak in one time window, with 5 raw features:

| Feature | Description |
|---|---|
| `freq` | Fractional frequency bin index (parabolic-interpolated) |
| `dlnf` | Relative chirp rate (d ln f / dt, interpolated) |
| `amp` | Peak amplitude (or SNR when whitening is active) |
| `phase_start` | Phase at half-window start (t = -0.5) |
| `phase_end` | Phase at half-window end (t = +0.5) |

Phase boundaries overlap between adjacent windows: `phase_end[w] = phase_start[w+1]` for clean signals.

**Constructor parameters:**
- `k` — Window size / FFT size (default 1024)
- `n_peaks` — Peaks per time window (default 3)
- `radius` — Peak suppression radius for local-max detection (default 2)
- `n_dlnf`, `dlnf_min`, `dlnf_max` — De-chirp grid (default 11 points, 0.0 to 0.05)
- `psd` — Pre-computed noise PSD, shape `(W, Fk)` where `Fk = k // 2 + 1` (default None = no whitening)

**PSD whitening:** Divides the STFT by `sqrt(PSD)` before peak detection, so amplitudes become SNR-like. Two ways to set the PSD:

```python
# Option 1: Pre-computed PSD
tokenizer = SpectralTokenizer(k=1024, psd=my_psd)  # my_psd: (W, Fk)

# Option 2: Streaming EMA from data
tokenizer = SpectralTokenizer(k=1024)
tokenizer.update_psd(noise_batch)          # first call sets PSD
tokenizer.update_psd(noise_batch2, momentum=0.99)  # subsequent calls EMA-update
```

### `TokenEmbedding` — Feature transforms + normalization

Transforms raw token values into model-ready features: `log1p` on amplitude, `cos/sin` on phases, then z-score normalization. Each peak becomes an independent token in the sequence.

```
(B, W, K, 5) raw tokens -> (B, W*K, n_embed) embedded features
```

- `phase_mode="center"`: uses `(phase_start + phase_end) / 2` -> 5 embedded features
- `phase_mode="boundary"`: keeps both endpoints -> 7 embedded features

Call `compute_normalization(train_tokens)` once on training data before use.

### `TransformerEmbedding` — Tokens to fixed-size vector

Transformer encoder backbone that maps raw tokens to a fixed-size summary vector. Uses **time-only positional encoding** shared across all peaks within the same window — peak identity comes from its features (freq, dlnf, amp), not from ordering.

```
(B, W, K, 5) raw tokens -> (B, d_model) embedding vector
```

Designed as a drop-in embedding network for SBI frameworks.

### `SpectralDecomposer` — Low-level STFT

The underlying STFT engine with two de-chirp modes:

- **Phase de-chirp** (`a`): `exp(-i a t^2)` multiplication removes constant absolute chirp rate
- **Resample de-chirp** (`dlnf`): exponential time-grid warping removes constant relative chirp rate, de-chirping all harmonics simultaneously

### `emri_signal` — Synthetic waveform generator

JAX-based post-Newtonian EMRI waveform generator, differentiable w.r.t. all continuous parameters. Used for generating training data and Fisher information analysis.

## Examples

```bash
# Transformer parameter estimation demo (3 params, ~2x CRB)
JAX_PLATFORMS=cpu python examples/transformer_demo.py

# Spectral decomposition visualization
python examples/spectral_demo.py

# Fisher information / Cramer-Rao bound analysis
JAX_PLATFORMS=cpu python examples/fisher_demo.py

# EMRI waveform generator demo
JAX_PLATFORMS=cpu python examples/emri_demo.py
```

## Package structure

```
fuge/
├── src/fuge/
│   ├── spectral.py      # SpectralDecomposer, SpectralTokenizer
│   ├── embedding.py     # TokenEmbedding, TransformerEmbedding
│   └── emri.py          # emri_signal (JAX)
└── examples/
    ├── transformer_demo.py   # End-to-end parameter estimation
    ├── spectral_demo.py      # STFT + peak visualization
    ├── fisher_demo.py        # Fisher matrix / CRB analysis
    └── emri_demo.py          # Waveform generator demo
```

## Architecture notes

**Dual-framework design:** PyTorch for signal analysis (STFT, tokenization, transformer), JAX for signal synthesis (waveform generation with autodiff). This split is intentional — PyTorch handles efficient batched GPU tensor operations, while JAX provides automatic differentiation through the waveform model for Fisher information computation.

**Token design:** Phases are defined at half-window boundaries so they tile the signal without gaps. The `phase_center` can be recovered as `(phase_start + phase_end) / 2`. With 50% Hann window overlap, `phase_end[w]` coincides exactly with `phase_start[w+1]` for noiseless signals, enabling coherent phase tracking across windows.
