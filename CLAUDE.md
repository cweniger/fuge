# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific signal embeddings toolkit, installable as the `fuge` Python package. Modular architecture supports multiple embedding types (spectral, streaming SVD, etc.) with shared neural network infrastructure.

- **`src/fuge/spectral/`** — Spectral analysis subpackage: STFT with de-chirping, tokenization, and spectral token embedding
- **`src/fuge/svd/`** — Streaming PCA with Procrustes-stabilized output (StreamingPCA)
- **`src/fuge/nn.py`** — Generic neural network building blocks (TransformerEmbedding)

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

# Run streaming PCA demo (generates svd_demo_procrustes.png, svd_demo_wiener.png)
python examples/svd_demo.py
```

There is no formal test suite. Demo scripts live in `examples/`. No build system, linting, or CI is configured beyond `pyproject.toml`.

## Architecture

**Dual-framework design:** PyTorch for signal analysis, JAX for signal synthesis. This split is intentional — PyTorch handles efficient GPU tensor operations for STFT windowing, while JAX provides automatic differentiation through the waveform model for parameter optimization.

**Modular embedding design:** Each embedding type lives in its own subpackage under `src/fuge/`. Generic neural network components (e.g. `TransformerEmbedding`) live in `src/fuge/nn.py` and accept pre-embedded tensors of any `d_in` dimension. Import via explicit subpackage: `fuge.spectral.*`, `fuge.svd.*`, `fuge.nn.*`.

### `src/fuge/spectral/core.py` — `DechirpSTFT`, `PeakFinder`, `NoiseModel`, `ToneTokenizer`

Four classes with separated concerns:

- **`DechirpSTFT(nn.Module)`**: STFT with half-overlapping Hann windows (hop = k/2). Two de-chirp modes: phase (`a`, multiplies by `exp(-i*a*t²)`) and resample (`dlnf`, warps time grid for constant relative chirp rate). Optionally returns weighted FFTs (`(1-t)*hann`, `t*hann`) for boundary amplitude estimation. Input: `(N,)` or `(B, N)` tensor. Output: complex `(N_WINDOWS, k)` tensor.

- **`PeakFinder(nn.Module)`**: Finds top-K peaks in the (dlnf, freq) plane via max-pool suppression, refines positions via parabolic interpolation (with Hann bias correction), extracts phases at half-window boundaries (with dechirp-aware warping), and recovers boundary amplitudes from weighted FFTs (with scalloping correction and mixing matrix inversion).

- **`NoiseModel(nn.Module)`**: Streaming noise PSD estimator. Holds a reference to a `DechirpSTFT`, maintains EMA-updated noise std per (window, freq) bin from pure noise signals. Provides `whiten()` for SNR-based peak detection.

- **`ToneTokenizer(nn.Module)`**: Thin orchestrator composing `DechirpSTFT`, `PeakFinder`, and optionally `NoiseModel`. Outputs 9-field tokens: `[snr, t_start, t_end, f_start, f_end, A_start, A_end, phase_start, phase_end]` with normalized frequencies and wrapped phases.

The `dlnf` parameter is per-hop and internally scaled by 2 for the full window. Resampling uses linear interpolation on an exponentially warped time grid: `τ(t) = (exp(βt) - 1) / (exp(β) - 1)`.

### `src/fuge/spectral/embedding.py` — `ToneTokenEmbedding(nn.Module)`

Transforms raw tone tokens (snr, t_start, t_end, f_start, f_end, A_start, A_end, phase_start, phase_end) into model-ready embedded features with z-score normalization. SNR is peak amplitude from the (optionally whitened) STFT. Time and frequency boundaries tile the signal; boundary amplitudes are recovered via weighted FFTs with complementary time weights.

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
├── src/
│   └── fuge/
│       ├── __init__.py              # package docstring, no flat re-exports
│       ├── nn.py                    # TransformerEmbedding (generic)
│       ├── spectral/
│       │   ├── __init__.py          # re-exports: DechirpSTFT, PeakFinder, NoiseModel, ToneTokenizer, ToneTokenEmbedding
│       │   ├── core.py              # DechirpSTFT, PeakFinder, NoiseModel, ToneTokenizer
│       │   └── embedding.py         # ToneTokenEmbedding
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
    └── svd_demo.py
```

## Dependencies

Requires: `torch`, `jax`, `jaxlib`, `numpy`, `matplotlib`, `scipy`. Python 3.10+. GPU (CUDA) supported by both frameworks but not required.
