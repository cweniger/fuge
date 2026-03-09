# fuge

Scientific signal embeddings toolkit. Converts time-domain data into compact representations suitable for neural network inference, designed as an embedding layer for Simulation-Based Inference (SBI).

## Installation

```bash
pip install -e .
```

Requires Python 3.10+, PyTorch, JAX, NumPy, Matplotlib.

## Quick start

```python
import torch
from fuge.spectral import ToneTokenizer, ToneTokenEmbedding
from fuge.nn import TransformerEmbedding

# 1. Tokenize: time-domain signal -> raw spectral tokens
tokenizer = ToneTokenizer(k=1024, n_peaks=3).to(device)
tokens = tokenizer(signals)  # (B, N) -> (B, W, K, 5)

# 2. Embed: raw tokens -> model-ready features
token_emb = ToneTokenEmbedding(phase_mode="center")
token_emb.compute_normalization(train_tokens)
embedded, n_windows, n_peaks = token_emb(tokens)  # (B, W*K, n_embed)

# 3. Encode: embedded features -> fixed-size vector
backbone = TransformerEmbedding(
    d_in=token_emb.n_embed, seq_len=W * K, d_model=64,
)
embedding = backbone(embedded)  # (B, seq_len, d_in) -> (B, d_model)

# Use `embedding` as input to SBI posterior network, regression head, etc.
```

## Core modules

### `fuge.spectral.ToneTokenizer` — Signal to tokens

Chains de-chirped STFT, peak finding, and phase extraction into a single batched `forward()` call.

```
(B, N) signal -> (B, W, K, 5) raw tokens
```

Each token represents one tone in one time window, with 5 raw features:

| Feature | Description |
|---|---|
| `f_start` | Frequency bin index at half-window start (t = -0.5) |
| `f_end` | Frequency bin index at half-window end (t = +0.5) |
| `amp` | Peak amplitude (or SNR when whitening is active) |
| `phase_start` | Phase at half-window start (t = -0.5) |
| `phase_end` | Phase at half-window end (t = +0.5) |

All boundary values tile across adjacent windows: `f_end[w] = f_start[w+1]` and `phase_end[w] = phase_start[w+1]` for clean signals.

**Constructor parameters:**
- `k` — Window size / FFT size (default 1024)
- `n_peaks` — Peaks per time window (default 3)
- `radius` — Peak suppression radius for local-max detection (default 2)
- `n_dlnf`, `dlnf_min`, `dlnf_max` — De-chirp grid (default 11 points, 0.0 to 0.05)
- `noise_std` — Pre-computed noise std per bin, shape `(W, Fk)` where `Fk = k // 2 + 1` (default None = no whitening)

**Whitening:** Divides the STFT by noise std before peak detection, so amplitudes become SNR-like. Two ways to set the noise std:

```python
# Option 1: Pre-computed noise std
tokenizer = ToneTokenizer(k=1024, noise_std=my_std)  # my_std: (W, Fk)

# Option 2: Streaming EMA from data
tokenizer = ToneTokenizer(k=1024)
tokenizer.update_noise_std(noise_batch)          # first call sets noise_std
tokenizer.update_noise_std(noise_batch2, momentum=0.99)  # subsequent calls EMA-update
```

### `fuge.spectral.ToneTokenEmbedding` — Feature transforms + normalization

Transforms raw token values into model-ready features: `log1p` on amplitude, `cos/sin` on phases, then z-score normalization. Each peak becomes an independent token in the sequence.

```
(B, W, K, 5) raw tokens -> (B, W*K, n_embed) embedded features
```

- `phase_mode="center"`: uses `(phase_start + phase_end) / 2` -> 5 embedded features
- `phase_mode="boundary"`: keeps both endpoints -> 7 embedded features

Call `compute_normalization(train_tokens)` once on training data before use.

### `fuge.svd.StreamingPCA` — Streaming PCA with Procrustes stabilization

Streaming dimensionality reduction for whitened signals. Momentum-blended covariance updates via single SVD, Procrustes alignment for output stability, diagonal Wiener filter for denoising.

```python
from fuge.svd import StreamingPCA

pca = StreamingPCA(n_components=32, buffer_size=256, momentum=0.1)
pca.update(whitened_batch)       # streaming update
coeffs = pca(whitened_x)         # (B, D) -> (B, 32), zeros before first update
```

### `fuge.nn.TransformerEmbedding` — Embedded tokens to fixed-size vector

Generic transformer encoder backbone that maps pre-embedded tokens to a fixed-size summary vector. Accepts any `(B, seq_len, d_in)` input — not coupled to any specific embedding type.

```
(B, seq_len, d_in) embedded tokens -> (B, d_model) embedding vector
```

Designed as a drop-in embedding network for SBI frameworks.

### `fuge.spectral.DechirpSTFT` — Low-level STFT

The underlying STFT engine with two de-chirp modes:

- **Phase de-chirp** (`a`): `exp(-i a t^2)` multiplication removes constant absolute chirp rate
- **Resample de-chirp** (`dlnf`): exponential time-grid warping removes constant relative chirp rate, de-chirping all harmonics simultaneously

## Examples

```bash
# Transformer parameter estimation demo
JAX_PLATFORMS=cpu python examples/transformer_demo.py

# Spectral decomposition visualization
python examples/spectral_demo.py

# Streaming PCA demo (Procrustes stability + Wiener filter verification)
python examples/svd_demo.py

# Fisher information / Cramer-Rao bound analysis
JAX_PLATFORMS=cpu python examples/fisher_demo.py

# Chirp signal generator demo
JAX_PLATFORMS=cpu python examples/chirp_demo.py
```

## Package structure

```
fuge/
├── src/fuge/
│   ├── __init__.py              # package docstring, no flat re-exports
│   ├── nn.py                    # TransformerEmbedding (generic)
│   ├── spectral/
│   │   ├── core.py              # DechirpSTFT, ToneTokenizer
│   │   └── embedding.py         # ToneTokenEmbedding
│   └── svd/
│       └── core.py              # StreamingPCA
└── examples/
    ├── chirp.py                  # Test signal generator (JAX)
    ├── chirp_demo.py
    ├── spectral_demo.py
    ├── transformer_demo.py
    ├── psd_whitening_demo.py
    ├── fisher_demo.py
    └── svd_demo.py
```

## Architecture notes

**Dual-framework design:** PyTorch for signal analysis (STFT, tokenization, transformer), JAX for signal synthesis (waveform generation with autodiff). This split is intentional — PyTorch handles efficient batched GPU tensor operations, while JAX provides automatic differentiation through the waveform model for Fisher information computation.

**Modular embedding design:** Each embedding type lives in its own subpackage (e.g. `fuge.spectral`, `fuge.svd`). Generic neural network components live in `fuge.nn` and accept pre-embedded tensors of any dimension, making them reusable across embedding types. Import via explicit subpackage: `fuge.spectral.*`, `fuge.svd.*`, `fuge.nn.*`.

**Token design:** Both frequency and phase are defined at half-window boundaries so they tile the signal without gaps. With 50% Hann window overlap, `f_end[w] ≈ f_start[w+1]` and `phase_end[w] ≈ phase_start[w+1]` for noiseless signals, enabling coherent tracking across windows. Center values can be recovered as `(start + end) / 2`.
