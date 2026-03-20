# Chirp Tokenization: Mathematical Definitions

This document defines the conventions and mathematics for the chirp
tokenization pipeline (`fuge.spectral`).  It serves as the reference for
the code in `core.py` and `embedding.py`.

**Terminology:** The tokenizer extracts **chirp tokens** — short
spectral components with frequency, amplitude, and phase at window
boundaries.  Phase-coherent sequences of chirp tokens stitched across
windows form **voices**.  The full set of voices in a signal is the
**choir**.  (The package name `fuge` alludes to the musical fugue,
where independent voices enter, evolve, and interweave.)

---

## 1. Window coordinate: t ∈ [−1, 1]

Every STFT window of length `k` samples is parameterized by a
dimensionless coordinate

    t ∈ [−1, 1]

### Sample model

Each sample is a bin-averaged value occupying an interval of width 1.
Sample n occupies [n − ½, n + ½), with its center at integer position n.
The k samples cover bins centered at n = 0, 1, …, k − 1.

### Mapping

    n(t) = k/2 · (t + 1)

The slope k/2 equals the hop size, which ensures exact tiling of
adjacent windows.

### Discrete sample grid

The k samples at integer positions n = 0, 1, …, k − 1 correspond to
t-values:

    t_n = 2n/k − 1

i.e. `t = 2 * torch.arange(k) / k - 1`.  Note: this is NOT the same
as `torch.linspace(-1, 1, k)`, which would place the last sample at
t = +1 (= n = k, outside the array).  The discrete grid has t[0] = −1
and t[k−1] = 1 − 2/k, consistent with `n(t) = k/2·(t + 1)` evaluated
at integer n.

Landmark points:

| t    | n      | Meaning                                             |
|------|--------|-----------------------------------------------------|
| −1   | 0      | Center of first bin (periodic Hann zero)            |
| −½   | k/4    | Token start boundary (integer bin center when 4∣k)  |
|  0   | k/2    | Periodic Hann peak (exact)                          |
| +½   | 3k/4   | Token end boundary (integer bin center when 4∣k)    |
| +1   | k      | Virtual periodic zero (½ bin past last bin's right edge) |

The window center t = 0 aligns exactly with the periodic Hann peak at
sample k/2.  The window edges t = ±1 map to n = 0 (first sample, Hann
zero) and n = k (no physical sample — the periodic Hann's "virtual"
zero).  The ½-bin asymmetry at the edges reflects the periodic Hann
window, which has its zero at n = 0 but not at n = k − 1.

### Tiling

With 50 % overlap (hop = k/2), adjacent windows share the boundary:
**t = +½ of window w** coincides with **t = −½ of window w+1**.
Both map to the same bin center, because the slope equals the hop:

    n_w(+½) = w · hop + 3k/4
    n_{w+1}(−½) = (w + 1) · hop + k/4 = w · hop + 3k/4   ✓

All token parameters (f, A, φ) are defined at the boundaries **t = ±½**.

### Constraint: k must be a multiple of 4

Requiring 4 ∣ k ensures that all five landmark positions (t = −1, −½,
0, +½, +1) map to integer bin centers.  Typical FFT sizes (256, 512,
1024, …) satisfy this automatically.

---

## 2. Chirp model

Within each window the instantaneous frequency follows an exponential
chirp:

    f(t) = f_center · exp(β · t)

where

- **f_center = f(0)** is the frequency at the window center,
- **β** is the chirp parameter (dimensionless log-frequency rate per
  unit of t).

Boundary frequencies:

    f_start ≡ f(−½) = f_center · exp(−β/2)
    f_end   ≡ f(+½) = f_center · exp(+β/2)

Derived relations:

    f_center = √(f_start · f_end)        (geometric mean)
    β = ln(f_end / f_start)

The FFT bin index corresponds to f_center.

### Relation to `dlnf`

The user-facing grid parameter `dlnf` is the log-frequency change per
hop (= k/2 samples = ½ unit of t).  The full window spans 2 units of t
(from −1 to +1), so:

    β = 2 · dlnf    (total log-frequency change across the full window)

And across the token span (t = −½ to +½):

    ln(f_end / f_start) = β = 2 · dlnf

In physical units, dlnf relates to the frequency derivative as:

    dlnf = (ḟ / f) · T_hop

where T_hop = (k/2) / f_s is the hop duration and f_s the sample rate.

---

## 3. Dechirp warp: τ(t)

De-chirping resamples the windowed signal on a non-uniform time grid so
that a chirped tone becomes a pure tone in the warped domain.  The FFT
of the resampled signal then shows a sharp peak.

Both t and τ live in [−1, 1].

### Forward warp

The warp is defined by requiring uniform phase accumulation in τ-space:

    dτ/dt ∝ f(t) = f_center · exp(β·t)

With boundary conditions τ(−1) = −1, τ(+1) = +1, integrating gives:

    τ(t) = 2 · [exp(β·t) − exp(−β)] / [exp(β) − exp(−β)] − 1

which simplifies to:

    τ(t) = [exp(β·t) − exp(−β)] / sinh(β) − 1

For β → 0 this reduces to the identity τ(t) = t.

Note: τ(0) = [1 − exp(−β)] / sinh(β) − 1 = −tanh(β/2) ≠ 0 in general.

### Inverse warp (used in code)

Given a uniform destination grid τ ∈ [−1, 1], solve for the source
position t(τ):

    t(τ) = ln[exp(−β) + ((τ + 1) / 2) · (exp(β) − exp(−β))] / β

or equivalently:

    t(τ) = ln[1 + ((τ + 1) / 2) · (exp(2β) − 1)] / β − 1

This is what `_resample_dechirp_batched` computes: for each uniformly
spaced τ, find the physical time t to read from, then linearly
interpolate the windowed signal.

### Jacobian correction

The warp changes the density of samples: regions where f(t) is high
(fast phase accumulation) map to more τ-samples than regions where
f(t) is low.  The FFT treats all τ-samples equally, so without
correction a constant-amplitude chirp would appear brighter on the
high-frequency side.

The correction factor is the Jacobian of the inverse warp:

    J(τ) = |dt/dτ| = 1 / (dτ/dt)|_{t=t(τ)}

Since dτ/dt ∝ exp(β·t), we have:

    J(τ) ∝ exp(−β · t(τ))

After resampling, each value is multiplied by J(τ) (normalized so
that J = 1 at τ = 0, or equivalently so that the total window energy
is preserved):

    resampled(τ) *= J(τ) / J(0) = exp(−β · (t(τ) − t(0)))

For β → 0 this reduces to 1 (no correction).  The correction is
computed from the already-available t_source values, so it adds
negligible cost.

### Chirped matched filter interpretation

The dechirp warp followed by FFT computes a matched filter against
chirped templates.  For a given β, the output at frequency bin f is:

    X(f, β) = Σ_n w[n] · x[n] · exp(−2πi · f · τ_n / k)

where τ_n = τ(t_n) is the warped position of sample n.  This is the
inner product of the windowed signal with a chirped complex exponential
at every center frequency f simultaneously — a chirped convolution.

### Why constant dlnf (not constant df)

The chirp model f(t) = f_center · exp(βt) has constant *relative*
chirp rate d(ln f)/dt.  The key advantage: the warp τ(t) depends
only on β, not on f_center.  A single warp + FFT covers all center
frequencies at once.

The alternative — constant df/dt (linear chirp, f(t) = f_center + αt)
— would require a frequency-dependent warp or phase correction, since
the relative chirp rate α/f varies across bins.  This is natural for
radar/sonar but poorly suited to a single FFT.

### Relation to standard methods

Mathematically, the chirped matched filter is a type-2 NUFFT:
uniform output frequencies f, non-uniform input positions τ_n.
Several methods can compute it:

**Warp + FFT (current approach).**  Resample (grid) the windowed
signal onto a uniform τ-grid, then standard FFT.  The resampling is
a gridding step with a linear interpolation kernel.  Cost: O(k log k)
per β value.  Adequate for |dlnf| ≤ 0.5, where the non-uniformity
of τ_n is mild and linear interpolation suffices.  For larger β, a
wider interpolation kernel (Kaiser–Bessel, Gaussian) would improve
accuracy — this is exactly what NUFFT libraries do internally.

**Type-2 NUFFT.**  The exact version of the same operation: gridding
with an optimized kernel + FFT + deapodization.  Same O(k log k) cost,
better accuracy for large β.  Well-suited to our context because
the non-uniform positions τ_n depend only on β (not f), so the
gridding setup is reused across all frequencies.  Available in
libraries (finufft, torchkbnufft).

**Chirp-z transform (CZT).**  Computes the z-transform along a spiral
in the z-plane via the Bluestein identity (three FFTs).  Designed for
*linear* chirps (constant df/dt), where the quadratic phase factor
decomposes cleanly.  Not directly applicable to our exponential chirp
model — the phase is not quadratic in n.

**Direct summation.**  Exact, O(k²) per β value.  Only practical for
small k or as a reference for testing.

### Upgrade path for |dlnf| > 0.5

The current warp + FFT with linear interpolation is adequate for
|dlnf| ≤ 0.5.  To support larger chirp rates, replace the gridding
step with a type-2 NUFFT backend.  Recommended libraries (all support
GPU + PyTorch autograd):

- **torchkbnufft** — pure PyTorch, no compiled dependencies, easiest
  to integrate.  `pip install torchkbnufft`.
- **pytorch-finufft** — wraps cuFINUFFT (Flatiron Institute),
  10–100× faster on GPU, but requires compiled CUDA library.
- **jax-finufft** — relevant for the JAX side of this project
  (signal synthesis); full autodiff support.

### Sample-index form

In both t-space and τ-space, samples sit at t_n = 2n/k − 1 for
n = 0, …, k − 1.  Both map to source positions via n(·) = k/2 · (· + 1).
The resampling reads the windowed signal at source positions n(t(τ_n)),
interpolating on the original k-sample grid, then applies the
Jacobian correction.

Note: the τ-grid has the same k samples as the source grid.
Increasing the number of τ-samples beyond k does not improve
frequency resolution — the information content is limited by the
k source samples (k/2 + 1 independent frequency bins).  For finer
frequency resolution, use a larger window (larger k).

---

## 4. Hann window and weighted sub-windows

The periodic Hann window `torch.hann_window(k)` has k samples at
n = 0, 1, …, k − 1, with a zero at n = 0, peak at n = k/2, and
a virtual periodic zero at n = k.  In the t-coordinate:

    w(t) = hann(n(t))    where n(t) = k/2 · (t + 1)

The weighted sub-windows for boundary amplitude estimation are:

    w_start(t) = ((1 − t) / 2) · w(t)
    w_end(t)   = ((1 + t) / 2) · w(t)

These satisfy w_start + w_end = w and provide linear ramps across the
window for amplitude deconvolution.

---

## 5. Peak detection

Peaks are found in the 2-D (dlnf, frequency) plane from the magnitude
of the full Hann FFT:

    X = X_start + X_end

(optionally whitened by the noise model).

1. **Max-pool suppression:** A pixel is a peak iff it equals the max
   in its (2r+1) × (2r+1) neighborhood.
2. **Dlnf deduplication:** For each frequency bin, only the dlnf index
   with highest amplitude survives.
3. **Top-K selection:** The K strongest peaks per window are retained.
4. **Parabolic interpolation:** Refines both frequency and dlnf
   positions to sub-bin accuracy, with Hann bias correction via LUT.

### TODO: Joint amplitude detection

Instead of detecting on |X|, one could detect on a joint statistic from
|X_start| and |X_end| by solving for the best-fit (A_start, A_end) at
each bin via the inverse mixing matrix, then using
SNR² = A_start² + A_end² as the detection statistic.  This would give
higher sensitivity to signals with rapidly evolving amplitude (large
|dA/dt|).  The math is the same as in §7 (amplitude recovery), just
applied before peak selection.

---

## 6. Phase extraction

### Natural anchor: window center in τ-space

The periodic Hann window is exactly symmetric about sample k/2, and
its value at sample 0 is exactly zero (w[0] = 0).  These two facts
together mean that the DTFT of the centered window is purely real,
giving a clean phase relationship.

For the periodic Hann of length k, the FFT phase at integer bin m
relates to the phase at the window center (sample k/2, t = 0) by:

    φ_center = arg(X[m]) + π · m

This is exact and **does not depend on the fractional bin offset δ**.
The proof: shifting the DFT sum to center at k/2 pulls out a factor
exp(−iπm); the remaining sum over the symmetric window is real
(the asymmetric boundary term at n = 0 vanishes because w[0] = 0).

The anchor point is the window center in τ-space (sample k/2 of the
dechirped grid), which corresponds to physical time
t = −tanh(β/2), close to but not exactly t = 0 for β ≠ 0.

### Propagation to token boundaries

From the anchor φ_center, propagate to the token boundaries at physical
times t = ±½ through the dechirp warp.

In the dechirped domain the signal is a pure tone at frequency f_ref
(in bins), so the phase advance from the anchor to any dechirped
position τ is proportional to the sample-index difference:

    φ(τ) = φ_center + 2π · f_ref · (n(τ) − n(0)) / k

where n(τ) = k/2 · (τ + 1) and n(0) = k/2, giving
n(τ) − n(0) = k/2 · τ.

To find the phase at physical time t = ±½, compute the warped
positions from the forward warp (§3):

    τ_start = τ(t = −½)
    τ_end   = τ(t = +½)

Then:

    phase_start = φ_center + π · f_ref · τ_start
    phase_end   = φ_center + π · f_ref · τ_end

Note: τ_start and τ_end are not symmetric about 0 for β ≠ 0, so the
phase errors from propagation are inherently asymmetric.

Final phases are wrapped to [−π, π].

---

## 7. Boundary amplitude recovery

Assuming amplitude varies linearly within the window, parameterized at
the token boundaries t = ±½:

    A(t) = A_start · (½ − t) + A_end · (½ + t)

(equal to A_start at t = −½, A_end at t = +½), the weighted FFT
magnitudes relate to the true boundary amplitudes via:

    [|X_start|]         [A_start]
    [         ] = M  ·  [       ]
    [|X_end|  ]         [A_end  ]

where the 2 × 2 mixing matrix M is:

    M_ij = Σ_n  w_i(t_n) · basis_j(t_n)

with w_i ∈ {w_start, w_end} and basis_j ∈ {(½ − t), (½ + t)}.

Recovery:

    [A_start, A_end]^T = 2 · M⁻¹ · [|X_start|, |X_end|]^T

The factor of 2 corrects for the cos → complex exponential factor of
½ in the FFT.  A scalloping correction (LUT-based) is applied to the
magnitudes before inversion to account for fractional bin offset.

---

## 8. Chirp token output

Each peak produces a 9-element token:

| Index | Field       | Definition                      | Range        |
|-------|-------------|---------------------------------|--------------|
| 0     | snr         | Peak amplitude (whitened)       | [0, ∞)       |
| 1     | t_start     | Time at t = −½, normalized      | [−1, 1]      |
| 2     | t_end       | Time at t = +½, normalized      | [−1, 1]      |
| 3     | f_start     | f(−½), normalized               | [−1, 1]      |
| 4     | f_end       | f(+½), normalized               | [−1, 1]      |
| 5     | A_start     | Amplitude at t = −½             | [0, ∞)       |
| 6     | A_end       | Amplitude at t = +½             | [0, ∞)       |
| 7     | phase_start | Phase at t = −½                 | [−π, π]      |
| 8     | phase_end   | Phase at t = +½                 | [−π, π]      |

Time normalization: absolute sample position mapped to [−1, 1] over the
signal length.

Frequency normalization: bin index mapped to [−1, 1] over [0, Fk−1]
where Fk = k/2 + 1.

Adjacent windows share boundaries: t_end[w] = t_start[w+1],
and for clean signals f_end[w] ≈ f_start[w+1],
phase_end[w] ≈ phase_start[w+1].

---

## 9. Motivation: sparse phase-coherent representation for SBI

### The problem

Simulation-based inference (SBI) for gravitational wave (GW) data
analysis requires a compact data representation that a neural network
can consume.  Standard representations lose information or are too
large:

- **Raw time series**: very high dimensional (N samples).
- **Power spectrograms**: discard phase — lose the √N_segments
  sensitivity gain from coherent combination.
- **Q-transform / wavelets**: better time-frequency resolution, but
  still dense and phase-free.
- **Learned embeddings**: may not preserve physically meaningful
  structure; must be retrained per problem.

### Chirp tokens as sparse sufficient statistics

The token output (§8) is designed to be a sparse, phase-coherent
representation suitable for downstream inference:

- **Sparse**: K peaks per window (typically 3–10), not Fk frequency
  bins.  Orders of magnitude smaller than a spectrogram.
- **Phase-coherent**: boundary phases enable stitching across windows.
  A neural network can learn to accumulate coherent phase along
  connected tracks, recovering the ~√N sensitivity gain that
  power-based methods discard.
- **Chirp-aware**: f_start ≠ f_end directly encodes the local chirp
  rate.  No need for the network to infer it from adjacent windows.
- **Model-agnostic**: does not assume post-Newtonian or any specific
  waveform model.  Works for compact binary coalescences, continuous
  waves, bursts — anything tonal.
- **Differentiable**: the entire tokenization pipeline is PyTorch
  and can sit inside an end-to-end training loop.

The key idea: physics-informed compression (peak finding, phase
extraction, boundary matching) is done up front, so the neural
network operates on a low-dimensional, information-rich input and
can focus on inference rather than feature extraction.

### Voice formation

The boundary compatibility of the token format —

    phase_end[w] ≈ phase_start[w+1]
    f_end[w]     ≈ f_start[w+1]

— enables stitching chirp tokens across windows into phase-coherent
voices, each representing a single evolving spectral component.  The
full set of voices — the choir — is a sparse, phase-coherent
decomposition of the signal.

The total accumulated phase along a voice provides a detection
statistic with coherent sensitivity, obtained from short-window FFTs
at semi-coherent computational cost.

### Relation to existing GW methods

Voice formation is closely related to existing techniques:

- **Phase-coherent SFT stitching** [arXiv:1510.06820]: carefully
  aligning short FFT segments to preserve phase continuity for
  semi-coherent CW searches.
- **HMM phase tracking** [arXiv:2107.12822, Suvorova et al.]: Hidden
  Markov model with phase-sensitive B-statistic that tracks both
  frequency and rotational phase via Viterbi decoding.
- **SOAP** [arXiv:1903.12614, Bayley & Woan]: Viterbi algorithm on
  spectrograms for CW frequency tracking (power-based; the phase-
  tracking HMM extends this).
- **Loosely coherent methods** [Prix, Dhurandhar]: intermediate
  between fully coherent and semi-coherent, maintaining phase
  coherence over a controllable length scale.

The chirp-token approach differs from these in that it is
model-agnostic: the stitching is based on boundary compatibility of
measured quantities, not on a parametric signal model.
