# t_root / phi_accum refactor plan

Enrich ChirpTokens with two DP-propagated fields per token:
- **`t_root`** — `t_start` of the oldest coherent ancestor on the best forward-DP path
- **`phi_accum`** — accumulated phase from `t_root` to the end of this token

These replace discrete chain IDs as the voice-grouping signal and enable stable
sub-harmonic phase embeddings for the downstream transformer.

See also: `docs/spectral_math.md` for coordinate conventions,
memory `project_token_embedding_design.md` for the full feature design.

---

## 1. ChirpTokens — extend the token tensor

**File:** `src/fuge/spectral/tokens.py`

- Extend the data tensor from `(B, N, 9)` to `(B, N, 11)`.
- New fields at indices 9, 10: `t_root`, `phi_accum`.
- Add named accessors `.t_root` and `.phi_accum` alongside the existing ones.
- Update docstring and field list.

Backward-compatibility note: callers that index by column number will break;
callers that use named accessors are fine. Audit `legato.py`, `embedding.py`,
and all demos before merging.

---

## 2. ChirpLinker — propagate t_root and phi_accum in forward DP

**File:** `src/fuge/spectral/legato.py`

### 2a. Forward DP changes (`_build_dag`)

The existing forward DP computes `chains[idx] = (snr_sq, path)`.
Replace with propagated scalars alongside the SNR accumulation:

```python
# existing: chains[idx] = (best_snr_sq, best_path)
# add:      t_root[idx], phi_accum[idx]

# For isolated token (no valid predecessor):
t_root[idx]    = tokens[idx, 1]   # t_start of this token
phi_accum[idx] = 0.0

# For token with best predecessor p:
delta_within   = tokens[idx, 8] - tokens[idx, 7]          # pe - ps of current
boundary_resid = _wrap(tokens[p, 8] - tokens[idx, 7])     # phase continuity residual
phi_accum[idx] = phi_accum[p] + boundary_resid + delta_within
t_root[idx]    = t_root[p]
```

`_wrap(x)` is the existing `((x + π) % 2π) − π` helper.

### 2b. DAG coherence score (companion to this refactor)

The forward DP gives `snr_sq_fwd[idx]` (best path ending here).
A backward DP gives `snr_sq_bwd[idx]` (best path starting here).
The per-token coherence score:

```python
score[idx] = sqrt(snr_sq_fwd[idx] + snr_sq_bwd[idx] - snr[idx]**2)
```

(Subtracts the token's own contribution counted twice.)

This replaces the greedy chain-assignment step `_greedy_assign`.
`LinkedChirpTokens` no longer needs `chain_id`; the `score` is written into
`tokens[:, 0]` in place of the raw per-window SNR.

### 2c. `_enrich_single` changes

Current enrichment steps:
1. Boundary f/A averaging — keep (reconstruction quality)
2. Split phase correction — keep (reconstruction quality)
3. Accumulated SNR → `tokens[:, 0]` — replace with coherence score
4. Chain ID assignment — **drop** (replaced by t_root)
5. **New:** write `t_root` and `phi_accum` into columns 9, 10

### 2d. Remove `_greedy_assign`

No longer needed once chain IDs are gone.
`LinkedChirpTokens` can be retired or kept with `chain_id` field set to -1 everywhere
(for backward compat) while `t_root` becomes the primary grouping signal.

---

## 3. Embedding — add phi_accum + t_root, wire sigma damping

**File:** `src/fuge/spectral/embedding.py`

### 3a. SNR-damped harmonics

Add optional `sigma: float | None = None` to:
- `HarmonicEmbedding.forward(x, sigma=None)`
- `HarmonicPhaseEmbedding.forward(x, sigma=None)`

When provided, multiply each output feature by `exp(-0.5 * (freqs * sigma)**2)`,
broadcast over sin/cos pairs. No-op when `sigma=None`.

Floor: clamp the damping factor from below at 0.05 so very-low-SNR tokens
don't vanish entirely.

### 3b. HarmonicPhaseEmbeddingConfig — sub-harmonic levels for phi_accum

`phi_accum` can reach O(thousands) of radians for long voices.
Add a second config preset `HarmonicPhaseEmbeddingConfig.for_accum(phi_max)` where
`phi_max` is set to e.g. `2π × f_max × k/2 × max_windows`.
This config uses sub-harmonics (modes with ω < 1) down to period ~phi_max,
plus the standard base + super-harmonics.

Contrast with the config for raw `ps`/`pe` which uses **no sub-harmonics**
(unstable reference; sub-harmonics not meaningful for raw window phase).

### 3c. ChirpTokenEmbedding — new fields

Extend to embed `t_root` (via `HarmonicEmbedding`) and `phi_accum`
(via `HarmonicPhaseEmbedding.for_accum`).

Updated feature table:

| Field | Embedding | sigma |
|---|---|---|
| t_start, t_end | HarmonicEmbedding | — |
| f_start, f_end | HarmonicEmbedding | 1/score |
| A_start, A_end | HarmonicEmbedding | σ_A from NoiseModel |
| ps, pe | HarmonicPhaseEmbedding (base+super only) | 1/score |
| phi_accum | HarmonicPhaseEmbedding (base+sub+super) | 1/score |
| t_root | HarmonicEmbedding | — |
| log(score) | scalar | — |
| log(k) | scalar | — |

Drop: `pe − ps` (redundant with f), `log(snr)` (replaced by `log(score)`).

---

## 4. Vectorization (prerequisite / companion — see Task #1)

The current `_build_dag` has Python loops with `.item()` GPU syncs — O(10–100 s)
at 10k+ tokens. The forward DP propagation of `phi_accum`/`t_root` adds two more
scalars per token, but does **not** increase the asymptotic complexity.

Before implementing this refactor at scale, the DP should be rewritten with
`scatter_reduce` / `torch.gather` (see Task #1). The propagation of
`phi_accum` and `t_root` fits naturally into the vectorized forward pass as
two additional scatter-accumulated tensors.

---

## Implementation order

1. **Extend `ChirpTokens`** (tokens.py) — add fields 9, 10 with accessors
2. **Vectorize `_build_dag`** (legato.py) — prerequisite for scale (Task #1)
3. **Propagate `phi_accum`, `t_root`, coherence score** in forward+backward DP
4. **Drop `_greedy_assign`**, update `_enrich_single`
5. **Add `sigma` damping** to HarmonicEmbedding / HarmonicPhaseEmbedding
6. **Add `for_accum` config** to HarmonicPhaseEmbeddingConfig
7. **Extend ChirpTokenEmbedding** with new fields + sigma plumbing
8. **Update demos** (03_voices.py) to show coherence score coloring
