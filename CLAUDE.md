# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific signal embeddings toolkit, installable as the `fuge` Python package. Modular architecture supports multiple embedding types (spectral, streaming SVD, etc.) with shared neural network infrastructure.

- **`src/fuge/spectral/`** — Spectral analysis subpackage: STFT with de-chirping, chirp tokenization, and chirp token embedding
- **`src/fuge/svd/`** — Streaming PCA with Procrustes-stabilized output (StreamingPCA)
- **`src/fuge/nn.py`** — Generic neural network building blocks (TransformerEmbedding)
- **`docs/spectral_math.md`** — Mathematical reference for the chirp tokenization pipeline

## Installation

```bash
pip install -e .
```

## Running

```bash
# Run spectral decomposition demo (generates spectral_demo.png, peaks_demo.png)
python examples/spectral_demo.py

# Run chirp signal generator demo (generates chirp_demo.png)
python examples/chirp_demo.py

# Run voice stitching demo (generates voice_demo.png)
python examples/voice_demo.py

# Run streaming PCA demo (generates svd_demo_procrustes.png, svd_demo_wiener.png)
python examples/svd_demo.py
```

There is no formal test suite. Demo scripts live in `examples/`. No build system, linting, or CI is configured beyond `pyproject.toml`.

## Architecture

**Dual-framework design:** PyTorch for signal analysis, JAX for signal synthesis. This split is intentional — PyTorch handles efficient GPU tensor operations for STFT windowing, while JAX provides automatic differentiation through the waveform model for parameter optimization.

**Modular embedding design:** Each embedding type lives in its own subpackage under `src/fuge/`. Generic neural network components (e.g. `TransformerEmbedding`) live in `src/fuge/nn.py` and accept pre-embedded tensors of any `d_in` dimension. Import via explicit subpackage: `fuge.spectral.*`, `fuge.svd.*`, `fuge.nn.*`.

**Terminology:** The tokenizer extracts **chirp tokens** — short spectral components with frequency, amplitude, and phase at window boundaries. Phase-coherent sequences of chirp tokens stitched across windows form **voices**. The full set of voices in a signal is the **choir**. (The package name `fuge` alludes to the musical fugue.)

### `src/fuge/spectral/core.py` — `DechirpSTFT`, `PeakFinder`, `NoiseModel`, `ChirpTokenizer`

Four classes with separated concerns. See `docs/spectral_math.md` for the full mathematical reference.

**Coordinate convention:** `t ∈ [-1, 1]` across each window, with `n(t) = k/2 · (t + 1)`. Discrete samples at `t_n = 2n/k - 1`. Token boundaries at `t = ±½` (samples k/4 and 3k/4). Periodic Hann window with zero at n=0, peak at n=k/2. Requires `k % 4 == 0`.

- **`DechirpSTFT(nn.Module)`**: STFT with half-overlapping periodic Hann windows (hop = k/2). De-chirps via resampling with Jacobian correction (`β = 2·dlnf`, warps time grid for constant relative chirp rate `f(t) = f_center · exp(β·t)`). With `n_hann_splits=2`, returns boundary FFTs `(X_start, X_end)` from `((1-t)/2)·hann` and `((1+t)/2)·hann` weighted sub-windows; standard FFT is `X = X_start + X_end`. Input: `(B, N)` tensor. Output: complex `(B, N_WINDOWS, D, Fk)` tensor where `Fk = k/2 + 1`.

- **`PeakFinder(nn.Module)`**: Finds top-K peaks in the (dlnf, freq) plane via max-pool suppression, refines positions via parabolic interpolation (with Hann bias correction), extracts phases at token boundaries t = ±½ using the periodic Hann phase anchor (`φ_center = arg(X[m]) + π·m`, exact and δ-independent) with forward-warp propagation, and recovers boundary amplitudes from weighted FFTs (with scalloping correction and mixing matrix inversion).

- **`NoiseModel(nn.Module)`**: Streaming noise PSD estimator. Holds a reference to a `DechirpSTFT`, maintains EMA-updated noise std per (window, freq) bin from pure noise signals. Provides `whiten()` for SNR-based peak detection.

- **`ChirpTokenizer(nn.Module)`**: Thin orchestrator composing `DechirpSTFT`, `PeakFinder`, and optionally `NoiseModel`. Returns `ChirpTokens` with 9 fields: `[snr, t_start, t_end, f_start, f_end, A_start, A_end, phase_start, phase_end]`. Time in sample indices, frequency in cycles/sample (0–0.5), phases unwrapped (pe − ps = phase advance per hop). Accepts `start` parameter for dyadic multi-resolution alignment. Adjacent tokens share boundaries for voice formation.

The `dlnf` parameter is per-hop; `β = 2·dlnf` is the total log-frequency change across the full window. Resampling uses linear interpolation on an exponentially warped time grid: `τ(t) = [exp(β·t) − exp(−β)] / sinh(β) − 1`. |dlnf| ≤ 0.5 supported.

### `src/fuge/spectral/tokens.py` — `ChirpTokens`

Structured wrapper around the (B, W, K, C) chirp token tensor with named field access (`.snr`, `.f_start`, `.phase_end`, `.chain_id`, etc.).  The underlying tensor stays contiguous and GPU-compatible.  Base tokens have C=9; after linking, C=10 (adds `chain_id`).

### `src/fuge/spectral/legato.py` — `ChirpLinker(nn.Module)`

Links chirp tokens across windows with boundary smoothing.  Shares the DAG-building and chain-resolution logic with `VoiceStitcher`.  For each matched chain: boundary frequencies and amplitudes are averaged to agree, boundary phases are split-corrected for coherence, SNR is replaced with accumulated chain SNR (`sqrt(Σ s_i²)`), and a chain ID is assigned.  Output is `ChirpTokens` with shape (B, W, K, 10) — same layout as input plus `chain_id`, directly usable by downstream transformers.

### `src/fuge/spectral/voice.py` — `VoiceStitcher(nn.Module)`, `VoiceStitchConfig`

Stitches chirp tokens into phase-coherent voices.  Builds a DAG of compatible tokens across adjacent windows (matching on frequency, phase, and amplitude), resolves branching via greedy highest-SNR path selection, and produces anchor-point sequences with coherently unwrapped phase.  Each voice is a `(V+1, 4)` tensor of `[amplitude, time, phase, frequency]` at boundary anchor points.  Phase stitching uses exact within-window advances and wrapped boundary corrections: `φ[i+1] = φ[i] + wrap(φ_start[i] − φ_end[i−1]) + (φ_end[i] − φ_start[i])`.

### `src/fuge/spectral/embedding.py` — `ChirpTokenEmbedding(nn.Module)`

Transforms raw chirp tokens (snr, t_start, t_end, f_start, f_end, A_start, A_end, phase_start, phase_end) into model-ready embedded features with z-score normalization. SNR is peak amplitude from the (optionally whitened) STFT. Time and frequency boundaries tile the signal; boundary amplitudes are recovered via weighted FFTs with complementary time weights. (Was `ToneTokenEmbedding`.)

### `src/fuge/svd/core.py` — `StreamingPCA(nn.Module)`

Streaming PCA with momentum-blended covariance updates via single SVD. Procrustes alignment stabilizes output for neural networks. Wiener filter provides optimal denoising assuming unit noise variance.

### `src/fuge/nn.py` — `TransformerEmbedding(nn.Module)`

Generic transformer encoder backbone: accepts pre-embedded `(B, seq_len, d_in)` tensors, projects to `d_model`, adds learnable positional encoding, runs through transformer encoder layers, and returns a fixed-size `(B, d_model)` summary via global average pooling. Not coupled to any specific embedding type.

## Package structure

```
fuge/
├── pyproject.toml
├── CLAUDE.md
├── LICENSE
├── .gitignore
├── docs/
│   ├── spectral_math.md         # mathematical reference for chirp tokenization
│   └── PLAN.md                  # implementation plan (may be outdated)
├── src/
│   └── fuge/
│       ├── __init__.py              # package docstring, no flat re-exports
│       ├── nn.py                    # TransformerEmbedding (generic)
│       ├── spectral/
│       │   ├── __init__.py          # re-exports all public classes
│       │   ├── tokens.py            # ChirpTokens
│       │   ├── core.py              # DechirpSTFT, PeakFinder, NoiseModel, ChirpTokenizer
│       │   ├── legato.py            # ChirpLinker
│       │   ├── voice.py             # VoiceStitcher, VoiceStitchConfig
│       │   └── embedding.py         # ChirpTokenEmbedding
│       └── svd/
│           ├── __init__.py          # re-exports: StreamingPCA
│           └── core.py              # StreamingPCA
└── examples/
    ├── chirp.py                     # test signal generator (JAX)
    ├── chirp_demo.py
    ├── spectral_demo.py
    ├── transformer_demo.py
    ├── psd_whitening_demo.py
    ├── fisher_demo.py
    ├── voice_demo.py
    └── svd_demo.py
```

## Dependencies

Requires: `torch`, `jax`, `jaxlib`, `numpy`, `matplotlib`, `scipy`. Python 3.10+. GPU (CUDA) supported by both frameworks but not required.
