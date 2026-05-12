# Demo consolidation plan

Consolidate `notebooks/` (7 loosely related `.py` files) into `demos/` (3 focused demos).
Move superseded files to `_graveyard/`. Update `CLAUDE.md`.

## Target structure

```
demos/
  01_tokenize.py       ← spectral_demo.py + token_continuity_demo.py
  02_reconstruct.py    ← merger_reconstruction_demo.py + multiresolution_demo.py
  03_voices.py         ← emri_demo.py (lightly cleaned up)
```

## Dropped files (to _graveyard/)

- `token_reconstruction_demo.py` — same algorithm as merger_reconstruction, less compelling signal
- `voice_demo.py` — uses linear frequency drift (`fdot`) which mismatches the tokenizer's
  exponential de-chirping model; physically wrong, confusing

## File-by-file plan

### `demos/01_tokenize.py`

Two sections in one `main()`:

1. **Noise sweep** (from `spectral_demo.py`): DechirpSTFT spectrogram at 4 noise levels,
   PeakFinder overlaid cyan slope lines. Saves `01_noise_sweep.png`.
2. **Token boundary continuity** (from `token_continuity_demo.py`): ChirpTokenizer on a
   chirp signal, plots f/A/phase boundary mismatches as time series + histograms.
   Saves `01_token_continuity.png`.

Shared signal: PN-inspired chirp (same parameters as `spectral_demo.py`).
Drop the separate phase-continuity demo from `spectral_demo.py` (it duplicates what
`token_continuity_demo.py` shows at the token level; keep only the spectrogram sweep).

### `demos/02_reconstruct.py`

Two sections sharing `make_merger_signal()`:

1. **Single-resolution reconstruction** (from `merger_reconstruction_demo.py`):
   tokenize at one `k`, reconstruct via overlap-add, plot signal + residual + STFT.
   Saves `02_single_resolution.png`.
2. **Multi-resolution reconstruction** (from `multiresolution_demo.py`):
   tokenize at `k ∈ [64, 128, 256, 512, 1024, 2048, 4096]`, greedy resolution
   selection, cross-faded overlap-add, resolution map panel.
   Saves `02_multiresolution.png`.

Functions to lift verbatim from `multiresolution_demo.py`:
`make_merger_signal`, `tokenize_multi`, `greedy_select`,
`_synthesize_window`, `_reconstruct_single_k`, `reconstruct_multiresolution`.

### `demos/03_voices.py`

Essentially `emri_demo.py` with minor cleanup:
- Parameterized via `argparse` (already present).
- Multi-source, multi-harmonic, exponential chirp rate — correct model.
- Plots: clean signal, noisy signal, chirp tokens, linked chains, accumulated phase.
- Saves to `--output` (default `03_voices.png`).

## Files to move to _graveyard/

From `notebooks/`:
- `spectral_demo.py`, `token_continuity_demo.py`
- `merger_reconstruction_demo.py`, `multiresolution_demo.py`
- `token_reconstruction_demo.py`, `voice_demo.py`
- `emri_demo.py`
- All `*.png` output files

## CLAUDE.md changes

- Replace `notebooks/` section with `demos/` in the package structure tree.
- Update the "Running" section: replace the 4 notebook commands with:
  ```
  python demos/01_tokenize.py
  python demos/02_reconstruct.py
  python demos/03_voices.py [--snr N] [--n-harmonics N] [--n-sources N]
  ```
- Note that `_graveyard/` is untracked (in `.gitignore`).

## Implementation steps

1. `mkdir demos/`
2. Write `demos/01_tokenize.py`, `demos/02_reconstruct.py`, `demos/03_voices.py`
3. `mv notebooks/*.py notebooks/*.png _graveyard/`
4. `rmdir notebooks/`
5. Edit `CLAUDE.md`
6. `git add demos/ && git rm -r notebooks/` (or stage manually)
