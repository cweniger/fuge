# Implementation Plan: Align `core.py` with `spectral_math.md`

This plan brings the code in `src/fuge/spectral/core.py` (and
`embedding.py` where affected) into alignment with the conventions
defined in `docs/spectral_math.md`.

No behavioral changes to the external API (class names, `dlnf` grid
parameter, token output format).  Internal math and coordinate
conventions change.

---

## Step 1: Add `k % 4 == 0` assertion

**File:** `DechirpSTFT.__init__`, `PeakFinder.__init__`

Add `assert k % 4 == 0` so that all landmark positions (t = −1, −½,
0, +½, +1) map to integer bin centers via `n(t) = k/2 · (t + 1)`.

Add warp resolution `R: int = 1` to `DechirpSTFT.__init__`.
Store `self.R = R`, `self.k_tau = R * k`,
`self.Fk_tau = R * k // 2 + 1`.  `R` is the warp resolution:
k_tau = R·k τ-samples, giving R·k/2+1 frequency bins
(default R = 1).

---

## Step 2: Switch to periodic Hann explicitly

**File:** `DechirpSTFT.__init__`

Current code uses `torch.hann_window(k)` which defaults to
`periodic=True`.  Make this explicit: `torch.hann_window(k, periodic=True)`.
Same for all `hann_window` calls in `PeakFinder._init_*` methods.

---

## Step 3: Update weighted sub-windows

**File:** `DechirpSTFT.__init__`

**Current:**
```python
t_unit = torch.linspace(0, 1, k)
self.window_start = (1 - t_unit) * self.window
self.window_end = t_unit * self.window
```

**New (using t ∈ [−1, 1]):**
```python
t = torch.linspace(-1, 1, k)  # NOTE: k points from -1 to +1
self.window_start = ((1 - t) / 2) * self.window
self.window_end = ((1 + t) / 2) * self.window
```

Note: `torch.linspace(-1, 1, k)` produces k points including both
endpoints.  With the mapping `n(t) = k/2·(t+1)`, we need sample n
to correspond to `t = 2n/k - 1`, i.e., `t[0] = -1` and
`t[k-1] = 2(k-1)/k - 1 = 1 - 2/k`.  So we should use:
```python
t = 2 * torch.arange(k) / k - 1
```
This gives t = -1 at n = 0 and t = 1 - 2/k at n = k-1, consistent
with `n(t) = k/2·(t+1)` evaluated at discrete sample positions.

---

## Step 4: Rewrite `_resample_dechirp_batched`

**File:** `DechirpSTFT._resample_dechirp_batched`

Changes:
1. Add warp resolution `R` (int, default 1).
   `k_tau = R * k` destination samples in τ-space.
2. Replace `tau_uniform ∈ [0, 1]` with `τ ∈ [-1, 1]` (k_tau discrete
   points via `2 * torch.arange(k_tau) / k_tau - 1`).
3. Inverse warp formula:
   `t_source = ln[1 + ((τ+1)/2)·(exp(2β)−1)] / β − 1`
   with β = betas (already the full-window parameter).
4. Identity fallback: `t_source = τ` for |β| < eps.
5. Map to index space on the original k-sample grid:
   `idx = k/2 · (t_source + 1)` (instead of
   `(t_source + 1) * 0.5 * (k - 1)`).
6. **Add Jacobian correction:** After interpolation, multiply by
   `exp(−β · (t_source − t_source_center))` where
   `t_source_center = t_source` at the τ = 0 position (or at the
   midpoint of the τ grid).
7. Linear interpolation and gather remain the same.
8. Enforce `|dlnf| ≤ 0.5` (linear interpolation is adequate in
   this range; for larger values, an NUFFT backend would be needed).
9. FFT is now k_tau-point, giving k_tau/2 + 1 positive frequency
   bins.  Output shape changes from (B, W, D, Fk) to
   (B, W, D, Fk_tau) where Fk_tau = k_tau/2 + 1 = R*k/2 + 1.

---

## Step 5: Convert `dlnf` → `β` once at entry point

**File:** `DechirpSTFT.forward`

**Current:**
```python
windowed = self._resample_dechirp_batched(windowed, 2.0 * dlnf)
```

**New:**
```python
beta = 2.0 * dlnf  # convert once
windowed = self._resample_dechirp_batched(windowed, beta)
```

Rename `betas` parameter in `_resample_dechirp_batched` to `beta`
(it already receives the full-window value).  Add a comment:
`# β = 2·dlnf, total log-frequency change across the full window`.

---

## Step 6: Update phase extraction

**File:** `PeakFinder.peak_phases`

**Current (anchor at τ = 0, with fractional-bin correction):**
```python
phi_0 = X_peak.angle() - torch.pi * f_delta * (self.k - 1) / self.k
```

**New (anchor at window center, no fractional-bin correction):**
```python
phi_center = X_peak.angle() + torch.pi * freq_idx.float() / R
```
where R is the warp resolution (R = 1 reduces to `+ π * freq_idx`).

Then propagate to boundaries using the forward warp (§3 of the doc):
```python
tau_start = forward_warp(t=-0.5, beta=beta)
tau_end = forward_warp(t=+0.5, beta=beta)
phase_start = phi_center + torch.pi * freq_ref * tau_start
phase_end = phi_center + torch.pi * freq_ref * tau_end
```

This replaces the current `_phase_at` helper which anchors at τ = 0.

---

## Step 7: Update amplitude unmixing basis

**File:** `PeakFinder._init_amplitude_unmix`

**Current:**
```python
t_unit = torch.linspace(0, 1, k)
basis_start = 1 - t_unit
basis_end = t_unit
```

**New:**
```python
t = 2 * torch.arange(k, dtype=torch.float32) / k - 1
basis_start = 0.5 - t
basis_end = 0.5 + t
```

These equal A_start at t = −½ and A_end at t = +½ (§7 of doc).
The mixing matrix M and its inverse change numerically, but the
structure is the same.

---

## Step 8: Update scalloping and parabolic LUTs

**File:** `PeakFinder._init_scalloping_lut`, `_init_parabolic_lut`

These use `torch.hann_window(k)` internally.  Make `periodic=True`
explicit.  The LUT values will change slightly (periodic vs symmetric
Hann have different sidelobe structure), but the code structure is
unchanged.

---

## Step 9: Update token boundary positions

**File:** `ToneTokenizer.forward`

**Current:**
```python
t_s = w_idx * hop + k // 4
t_e = w_idx * hop + 3 * k // 4
```

**New:**
```python
t_s = w_idx * hop + k // 4       # n(−½) = k/4, unchanged numerically
t_e = w_idx * hop + 3 * k // 4   # n(+½) = 3k/4, unchanged numerically
```

These are actually the same values (k//4 = k/4 when 4∣k).  Just
update the comment to reference the new convention.

Frequency boundary computation is also unchanged:
```python
f_start = freq * torch.exp(-dlnf / 2)  # = f_center · exp(−β/2) ✓
f_end = freq * torch.exp(dlnf / 2)     # = f_center · exp(+β/2) ✓
```
(since dlnf here is per-hop and the boundaries span ±½ hop from
center, but β = 2·dlnf and the formula uses β/2 = dlnf).

---

## Step 10: Update `embedding.py` if needed

**File:** `src/fuge/spectral/embedding.py`

Check whether `ToneTokenEmbedding` depends on any internal conventions
that changed.  It consumes the 9-element token output, which has the
same format.  Likely no changes needed, but verify.

---

## Step 11: Rename Tone → Chirp

Rename classes and files:
- `ToneTokenizer` → `ChirpTokenizer`
- `ToneTokenEmbedding` → `ChirpTokenEmbedding`
- Keep old names as deprecated aliases in `__init__.py` for
  backwards compatibility (one-line `ChirpTokenizer = ...`).

Phase-coherent sequences of stitched tokens = **voices**.
The full set of voices in a signal = the **choir**.

---

## Step 12: Update docstrings and CLAUDE.md

Update all docstrings in `core.py` to reference the t ∈ [−1, 1]
convention, β instead of `2*dlnf` in formulas, periodic Hann,
Jacobian correction, and chirp/track terminology.  Update CLAUDE.md
architecture section to match.

---

## Step 13: Verify with demos

Run `examples/spectral_demo.py` and `examples/psd_whitening_demo.py`
to confirm outputs are reasonable.  There is no formal test suite,
so visual inspection of demo plots is the primary validation.

---

## Order and dependencies

Steps 1–2 are independent prerequisites.
Steps 3–5 modify `DechirpSTFT` (do together).
Steps 6–8 modify `PeakFinder` (do together, after 3–5).
Step 9 modifies `ToneTokenizer` (after 6–8).
Steps 10–13 are follow-up.
