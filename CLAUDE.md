# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gravitational wave signal analysis toolkit with two standalone modules:

- **`spectral.py`** — PyTorch-based Short-Time Fourier Transform (STFT) with de-chirping for time-frequency analysis
- **`emri.py`** — JAX-based synthetic EMRI (Extreme Mass Ratio Inspiral) waveform generator with autodiff support

## Running

```bash
# Run spectral decomposition demo (generates spectral_demo.png)
python spectral.py

# Run EMRI signal generator demo (generates emri_demo.png)
python emri.py
```

There is no formal test suite. Each module has an inline demo in its `if __name__ == "__main__"` block. No build system, linting, or CI is configured.

## Architecture

**Dual-framework design:** PyTorch for signal analysis, JAX for signal synthesis. This split is intentional — PyTorch handles efficient GPU tensor operations for STFT windowing, while JAX provides automatic differentiation through the waveform model for parameter optimization.

### `spectral.py` — `SpectralDecomposer(nn.Module)`

STFT with half-overlapping Hann windows (hop = k/2). Two de-chirp modes that can be combined:

- **Phase de-chirp** (`a`): Multiplies each window by `exp(-i * a * t²)` to remove constant absolute chirp rate
- **Resample de-chirp** (`dlnf`): Resamples onto warped time grid to remove constant *relative* chirp rate (fdot/f = const), de-chirping all harmonics simultaneously

Input: `(N,)` or `(B, N)` tensor. Output: complex `(N_WINDOWS, k)` tensor.

The `dlnf` parameter is per-hop and internally scaled by 2 for the full window. Resampling uses linear interpolation on an exponentially warped time grid: `τ(t) = (exp(βt) - 1) / (exp(β) - 1)`.

### `emri.py` — `emri_signal()` / `_emri_impl()`

Post-Newtonian inspired waveform: `f(t) = f₀(1 - t/t_c)^(-3/8 · chirp_mass)`, amplitude `A(t) = A₀(1 - t/t_c)^(-1/4)`, with harmonic sum `h(t) = Σ A_k(t) cos(k φ(t))`.

Phase accumulated via trapezoidal integration (O(dt²)). Harmonics summed with `jax.lax.scan`. Core function `_emri_impl` is JIT-compiled with `n_harmonics` and `N` as static arguments.

## Dependencies

Requires: `torch`, `jax`, `jaxlib`, `numpy`, `matplotlib`, `scipy`. Environment: Python 3.10+ via Conda (`emri_few_timm`). GPU (CUDA) supported by both frameworks but not required.
