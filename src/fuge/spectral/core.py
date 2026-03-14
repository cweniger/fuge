"""Short-time Fourier transform via half-overlapping Hann windows (PyTorch).

Supports two de-chirp modes:
  - "phase" (a): multiply by exp(-i * a * t^2), removes constant absolute chirp rate
  - "resample" (dlnf): resample onto warped time grid,
    removes constant *relative* chirp rate (fdot/f = const), de-chirping
    all harmonics simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DechirpSTFT(nn.Module):
    """STFT with half-overlapping Hann windows and optional de-chirping.

    Parameters
    ----------
    k : int
        Window size in bins.  Also the FFT size per window.

    Output shape: (N_WINDOWS, k)  -- complex-valued.
    N_WINDOWS = (N - k) // (k // 2) + 1   (with hop = k // 2).
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.hop = k // 2
        self.register_buffer("window", torch.hann_window(k))
        # Normalized time coordinate: t in [-1, +1] across the window
        self.register_buffer("t_norm", torch.linspace(-1.0, 1.0, k))

        # Weighted windows for amplitude-at-boundary estimation
        t_unit = torch.linspace(0, 1, k)
        window = torch.hann_window(k)
        self.register_buffer("window_start", (1 - t_unit) * window)
        self.register_buffer("window_end", t_unit * window)

        # Precompute 2x2 mixing matrix and its inverse.
        # The mixing matrix relates weighted FFT magnitudes to boundary
        # amplitudes, assuming linear A(t) within the window.
        basis_start = 1 - t_unit
        basis_end = t_unit
        M = torch.tensor([
            [(self.window_start * basis_start).sum(), (self.window_start * basis_end).sum()],
            [(self.window_end * basis_start).sum(), (self.window_end * basis_end).sum()],
        ])
        self.register_buffer("amp_unmix", torch.linalg.inv(M))

        # Precompute scalloping correction lookup table for weighted windows.
        # At fractional bin offset delta, the FFT magnitude is reduced by
        # |DTFT(w, delta)| / |DTFT(w, 0)|.  We store the inverse of this
        # ratio on a dense grid for fast interpolation at runtime.
        # w_start and w_end have nearly identical scalloping (Hann symmetry),
        # so we compute from w_start and use for both.
        n_lut = 64  # half-bin resolution: 0 to 0.5
        ns = torch.arange(k, dtype=torch.float64)
        delta_lut = torch.linspace(0, 0.5, n_lut)
        w_ref = self.window_start.double()
        dtft_mag = torch.zeros(n_lut)
        for i, d in enumerate(delta_lut):
            phasor = torch.exp(2j * torch.pi * d * ns / k)
            dtft_mag[i] = (w_ref * phasor).sum().abs()
        scallop_lut = dtft_mag[0] / dtft_mag.clamp(min=1e-12)
        self.register_buffer("_scallop_lut", scallop_lut.float())

        # Precompute parabolic interpolation correction LUT.
        # Parabolic interpolation on Hann-windowed data systematically
        # underestimates fractional bin offsets (e.g. true 0.4 -> estimated 0.36).
        # We precompute the exact forward mapping true_delta -> para_delta using
        # the known Hann DTFT, then store the inverse for runtime correction.
        n_corr = 1000  # dense forward mapping
        true_d = torch.linspace(0, 0.5, n_corr, dtype=torch.float64)
        para_d = torch.zeros(n_corr, dtype=torch.float64)
        w_hann = self.window.double()
        for i, td in enumerate(true_d):
            mags = []
            for off in [-1, 0, 1]:
                phasor = torch.exp(2j * torch.pi * (off - td) * ns / k)
                mags.append((w_hann * phasor).sum().abs())
            ym, y0, yp = mags
            denom = ym - 2.0 * y0 + yp
            para_d[i] = 0.5 * (ym - yp) / denom if denom.abs() > 1e-12 else 0.0
        # Invert: uniform grid over parabolic delta -> true delta
        n_inv = 64
        para_grid = torch.linspace(0, 0.5, n_inv, dtype=torch.float64)
        # numpy interp for inversion (monotonic)
        import numpy as _np
        true_inv = _np.interp(
            para_grid.numpy(), para_d.numpy(), true_d.numpy())
        self.register_buffer(
            "_para_corr_lut", torch.from_numpy(true_inv).float())

    def forward(self, x: torch.Tensor, a: float = 0.0, dlnf=0.0,
                return_weighted: bool = False):
        """Compute the (de-chirped) windowed FFT.

        Parameters
        ----------
        x : Tensor, shape (N,) or (B, N)
            Time-domain signal.  If 1-D, a batch dim is added and removed.
        a : float
            Absolute chirp rate parameter.  Multiplies each window by
            exp(-i * a * t^2) where t in [-1, +1].  a = 0 disables.
        dlnf : float or Tensor of shape (D,)
            Relative chirp parameter: change in ln(f) per hop step,
            i.e. dlnf = (fdot/f) * T_hop  (dimensionless).
            If a 1-D Tensor, computes the STFT for each value in parallel;
            output gains a leading D dimension.
            dlnf = 0 disables.
        return_weighted : bool
            If True, also return weighted FFTs for amplitude boundary
            estimation.  Returns (X, X_start, X_end).

        Returns
        -------
        X : complex Tensor
            Scalar dlnf: shape (N_WINDOWS, k) or (B, N_WINDOWS, k).
            Batched dlnf: shape (D, N_WINDOWS, k) or (D, B, N_WINDOWS, k).
        X_start, X_end : complex Tensor (only if return_weighted=True)
            Same shape as X, from (1-t)*hann and t*hann weighted windows.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)  # (1, N)

        # Unfold into overlapping windows: (B, N_WINDOWS, k)
        windows = x.unfold(dimension=1, size=self.k, step=self.hop)

        # Apply Hann window
        windowed = windows * self.window

        # Weighted windows for amplitude boundary estimation
        if return_weighted:
            windowed_start = windows * self.window_start
            windowed_end = windows * self.window_end

        # --- Relative de-chirp via time-grid resampling ---
        # dlnf is per hop; full window spans 2 hops
        batched = isinstance(dlnf, torch.Tensor) and dlnf.dim() >= 1
        if batched:
            windowed = self._resample_dechirp_batched(windowed, 2.0 * dlnf)
            if return_weighted:
                windowed_start = self._resample_dechirp_batched(windowed_start, 2.0 * dlnf)
                windowed_end = self._resample_dechirp_batched(windowed_end, 2.0 * dlnf)
        elif dlnf != 0.0:
            windowed = self._resample_dechirp(windowed, 2.0 * dlnf)
            if return_weighted:
                windowed_start = self._resample_dechirp(windowed_start, 2.0 * dlnf)
                windowed_end = self._resample_dechirp(windowed_end, 2.0 * dlnf)

        # --- Absolute de-chirp via phase multiplication ---
        if a != 0.0:
            chirp_kernel = torch.exp(-1j * a * self.t_norm ** 2)
            windowed = windowed * chirp_kernel
            if return_weighted:
                windowed_start = windowed_start * chirp_kernel
                windowed_end = windowed_end * chirp_kernel

        X = torch.fft.fft(windowed, n=self.k, dim=-1)

        if return_weighted:
            X_start = torch.fft.fft(windowed_start, n=self.k, dim=-1)
            X_end = torch.fft.fft(windowed_end, n=self.k, dim=-1)

        if squeeze:
            X = X.squeeze(1) if batched else X.squeeze(0)
            if return_weighted:
                X_start = X_start.squeeze(1) if batched else X_start.squeeze(0)
                X_end = X_end.squeeze(1) if batched else X_end.squeeze(0)

        if return_weighted:
            return X, X_start, X_end
        return X

    def _resample_dechirp(self, windowed: torch.Tensor, beta: float) -> torch.Tensor:
        """Resample windowed segments onto a warped time grid.

        Parameters
        ----------
        beta : float
            Total ln(f) change over the full window (= 2 * dlnf).

        The mapping: given uniform tau in [0, 1], compute the source
        time t such that tau(t) = (exp(beta*t) - 1) / (exp(beta) - 1).
        Then interpolate the signal at those source positions.

        For beta -> 0 this reduces to the identity (no resampling).
        """
        tau_uniform = torch.linspace(0.0, 1.0, self.k, device=windowed.device)
        eb = torch.exp(torch.tensor(beta, device=windowed.device))
        # Source positions in [-1, 1] for interpolation
        t_source = (2.0 / beta) * torch.log(1.0 + tau_uniform * (eb - 1.0)) - 1.0

        # Linear interpolation: t_source is in [-1, 1], map to [0, k-1] index space
        idx = (t_source + 1.0) * 0.5 * (self.k - 1)  # [0, k-1]
        idx_lo = idx.long().clamp(0, self.k - 2)
        idx_hi = idx_lo + 1
        frac = (idx - idx_lo.float()).unsqueeze(0).unsqueeze(0)  # (1, 1, k)

        # Gather and interpolate: windowed is (B, N_WINDOWS, k)
        lo_vals = torch.gather(windowed, 2, idx_lo.unsqueeze(0).unsqueeze(0).expand_as(windowed))
        hi_vals = torch.gather(windowed, 2, idx_hi.unsqueeze(0).unsqueeze(0).expand_as(windowed))
        resampled = lo_vals * (1.0 - frac) + hi_vals * frac

        return resampled

    def _resample_dechirp_batched(self, windowed: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        """Resample windowed segments for multiple chirp rates at once.

        Parameters
        ----------
        windowed : Tensor, shape (B, N_WINDOWS, k)
        betas : Tensor, shape (D,)
            Total ln(f) change over the full window for each trial.

        Returns
        -------
        resampled : Tensor, shape (D, B, N_WINDOWS, k)
        """
        D = betas.shape[0]
        tau_uniform = torch.linspace(0.0, 1.0, self.k, device=windowed.device)  # (k,)

        # Replace near-zero betas to avoid division by zero;
        # for |beta| < eps the warped grid is effectively identity.
        safe = betas.abs() < 1e-8
        betas_safe = torch.where(safe, torch.ones_like(betas) * 1e-8, betas)

        eb = torch.exp(betas_safe)  # (D,)
        # Source positions: (D, k)
        t_source = (2.0 / betas_safe.unsqueeze(1)) * torch.log(
            1.0 + tau_uniform.unsqueeze(0) * (eb.unsqueeze(1) - 1.0)
        ) - 1.0
        # For near-zero betas, use identity grid
        identity = torch.linspace(-1.0, 1.0, self.k, device=windowed.device)
        t_source = torch.where(safe.unsqueeze(1), identity.unsqueeze(0), t_source)

        # Map to index space [0, k-1]
        idx = (t_source + 1.0) * 0.5 * (self.k - 1)  # (D, k)
        idx_lo = idx.long().clamp(0, self.k - 2)       # (D, k)
        idx_hi = idx_lo + 1
        frac = (idx - idx_lo.float())                   # (D, k)

        # Expand windowed (B, W, k) -> (D, B, W, k) for gathering
        windowed_exp = windowed.unsqueeze(0).expand(D, -1, -1, -1)
        idx_lo_exp = idx_lo[:, None, None, :].expand_as(windowed_exp)
        idx_hi_exp = idx_hi[:, None, None, :].expand_as(windowed_exp)
        frac_exp = frac[:, None, None, :]  # broadcasts over B, W

        lo_vals = torch.gather(windowed_exp, 3, idx_lo_exp)
        hi_vals = torch.gather(windowed_exp, 3, idx_hi_exp)
        resampled = lo_vals * (1.0 - frac_exp) + hi_vals * frac_exp

        return resampled

    def find_peaks(self, X: torch.Tensor, K: int, dlnf_grid: torch.Tensor,
                   radius: int = 2,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find top K peaks in the (dlnf, freq) plane per time window.

        A pixel is a peak iff it equals the max in its (2r+1)x(2r+1)
        neighborhood (via max-pool), then the K strongest peaks are returned.
        Both frequency and dlnf positions are refined via parabolic
        interpolation on the amplitude and its immediate neighbours.

        Supports both single and batched input.

        Parameters
        ----------
        X : complex Tensor, shape (D, N_WINDOWS, k) or (D, B, N_WINDOWS, k)
            STFT output from forward() with a 1-D dlnf tensor.
        K : int
            Number of peaks to return per time window.
        dlnf_grid : Tensor, shape (D,)
            The dlnf values used to produce X (assumed linearly spaced).
        radius : int
            Suppression radius (default 2): peak must be the maximum in
            a (2*radius+1) x (2*radius+1) window in the (dlnf, freq) plane.

        Returns
        -------
        peaks : LongTensor, shape ([B,] N_WINDOWS, K, 2)
            Integer (dlnf_idx, freq_idx) of each peak.
        freq_refined : Tensor, shape ([B,] N_WINDOWS, K)
            Parabolic-interpolated fractional frequency bin index.
        dlnf_refined : Tensor, shape ([B,] N_WINDOWS, K)
            Parabolic-interpolated dlnf value.
        values : Tensor, shape ([B,] N_WINDOWS, K)
            Amplitude at integer peak location.
        """
        batched = X.dim() == 4
        if not batched:
            X = X.unsqueeze(1)  # (D, 1, W, k)

        D, B, W, k_full = X.shape
        Fk = self.k // 2 + 1

        # Merge batch and window dims: all ops are per-window
        amp = X[..., :Fk].abs()           # (D, B, W, F)
        amp = amp.permute(1, 2, 0, 3)     # (B, W, D, F)
        amp = amp.reshape(B * W, D, Fk)   # (BW, D, F)
        BW = B * W

        # Max pool to find local maxima
        kernel = 2 * radius + 1
        amp_4d = amp.unsqueeze(1)          # (BW, 1, D, F)
        pooled = F.max_pool2d(amp_4d, kernel_size=kernel, stride=1, padding=radius)
        is_peak = (amp_4d == pooled).squeeze(1)  # (BW, D, F)

        # Collapse duplicate peaks along the dlnf axis: for each frequency
        # bin, keep only the dlnf index with the highest amplitude.  This
        # prevents a slowly-chirping signal (flat dlnf response) from
        # producing multiple peaks at the same frequency.
        best_d = amp.argmax(dim=1, keepdim=True)  # (BW, 1, Fk)
        dlnf_mask = torch.zeros_like(is_peak)
        dlnf_mask.scatter_(1, best_d, 1)
        is_peak = is_peak & dlnf_mask.bool()

        # Keep only peak amplitudes, flatten spatial dims
        amp_peaks = torch.where(is_peak, amp, torch.zeros_like(amp))
        amp_flat = amp_peaks.reshape(BW, -1)  # (BW, D*F)

        topk_vals, topk_idx = amp_flat.topk(K, dim=1)  # (BW, K)

        # Unravel flat index -> (dlnf_idx, freq_idx)
        dlnf_idx = topk_idx // Fk
        freq_idx = topk_idx % Fk
        peaks = torch.stack([dlnf_idx, freq_idx], dim=-1)  # (BW, K, 2)

        amp_2d = amp.reshape(BW, -1)  # (BW, D*Fk)

        # --- Parabolic interpolation along frequency axis ---
        fi = freq_idx.clamp(1, Fk - 2)
        f_flat_m = dlnf_idx * Fk + (fi - 1)
        f_flat_0 = dlnf_idx * Fk + fi
        f_flat_p = dlnf_idx * Fk + (fi + 1)
        yf_m = amp_2d.gather(1, f_flat_m)
        yf_0 = amp_2d.gather(1, f_flat_0)
        yf_p = amp_2d.gather(1, f_flat_p)

        f_denom = yf_m - 2.0 * yf_0 + yf_p
        f_delta = 0.5 * (yf_m - yf_p) / f_denom
        f_delta = torch.where(f_denom.abs() < 1e-12, torch.zeros_like(f_delta), f_delta)
        f_delta = f_delta.clamp(-0.5, 0.5)
        f_delta = self._correct_parabolic(f_delta)
        freq_refined = fi.float() + f_delta

        # --- Parabolic interpolation along dlnf axis ---
        di = dlnf_idx.clamp(1, D - 2)
        d_flat_m = (di - 1) * Fk + freq_idx
        d_flat_0 = di * Fk + freq_idx
        d_flat_p = (di + 1) * Fk + freq_idx
        yd_m = amp_2d.gather(1, d_flat_m)
        yd_0 = amp_2d.gather(1, d_flat_0)
        yd_p = amp_2d.gather(1, d_flat_p)

        d_denom = yd_m - 2.0 * yd_0 + yd_p
        d_delta = 0.5 * (yd_m - yd_p) / d_denom
        d_delta = torch.where(d_denom.abs() < 1e-12, torch.zeros_like(d_delta), d_delta)
        d_delta = d_delta.clamp(-0.5, 0.5)

        dlnf_step = dlnf_grid[1] - dlnf_grid[0] if D > 1 else torch.ones(1, device=X.device)
        dlnf_refined = dlnf_grid[di] + d_delta * dlnf_step

        # Restore (B, W, ...) shape
        peaks = peaks.reshape(B, W, K, 2)
        freq_refined = freq_refined.reshape(B, W, K)
        dlnf_refined = dlnf_refined.reshape(B, W, K)
        topk_vals = topk_vals.reshape(B, W, K)

        if not batched:
            peaks = peaks.squeeze(0)
            freq_refined = freq_refined.squeeze(0)
            dlnf_refined = dlnf_refined.squeeze(0)
            topk_vals = topk_vals.squeeze(0)

        return peaks, freq_refined, dlnf_refined, topk_vals

    def peak_phases(self, X: torch.Tensor, peaks: torch.Tensor,
                    freq_refined: torch.Tensor, dlnf_refined: torch.Tensor,
                    dlnf_grid: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate phase at half-window boundaries for each peak.

        Returns phase_start and phase_end at the ±0.5 points of the
        Hann window (samples k/4 and 3k/4).  With 50% overlap,
        phase_end[w] coincides with phase_start[w+1], so the phases
        tile the signal without gaps.

        phase_center (at the window midpoint) can be recovered as
        (phase_start + phase_end) / 2.

        Supports both single and batched input.

        Parameters
        ----------
        X : complex Tensor, shape (D, N_WINDOWS, k) or (D, B, N_WINDOWS, k)
        peaks : LongTensor, shape ([B,] N_WINDOWS, K, 2)
            Integer (dlnf_idx, freq_idx) from find_peaks.
        freq_refined : Tensor, shape ([B,] N_WINDOWS, K)
            Fractional frequency bin index (in de-chirped domain).
        dlnf_refined : Tensor, shape ([B,] N_WINDOWS, K)
            Interpolated dlnf value (true chirp rate).
        dlnf_grid : Tensor, shape (D,)
            The dlnf grid used for de-chirping.

        Returns
        -------
        phase_start : Tensor, shape ([B,] N_WINDOWS, K)
            Phase at sample k/4 (t = -0.5 in Hann window coords).
        phase_end : Tensor, shape ([B,] N_WINDOWS, K)
            Phase at sample 3k/4 (t = +0.5 in Hann window coords).
            phase_end[w] = phase_start[w+1] for noiseless signals.
        """
        batched = X.dim() == 4
        if not batched:
            X = X.unsqueeze(1)      # (D, 1, W, k)
            peaks = peaks.unsqueeze(0)
            freq_refined = freq_refined.unsqueeze(0)

        D, B, W, k_full = X.shape
        K = peaks.shape[2]

        # Flatten batch*window for per-window gather
        dlnf_idx = peaks.reshape(B * W, K, 2)[:, :, 0]   # (BW, K)
        freq_idx = peaks.reshape(B * W, K, 2)[:, :, 1]    # (BW, K)
        freq_ref = freq_refined.reshape(B * W, K)          # (BW, K)

        # Rearrange X: (D, B, W, k) -> (B, W, D, k) -> (BW, D*k)
        Xp = X.permute(1, 2, 0, 3).reshape(B * W, D * k_full)
        flat_idx = dlnf_idx * k_full + freq_idx
        X_peak = Xp.gather(1, flat_idx)  # (BW, K) complex

        # Phase at window start (sample 0), corrected for fractional bin offset.
        f_delta = freq_ref - freq_idx.float()
        phi_0 = X_peak.angle() - torch.pi * f_delta * (self.k - 1) / self.k

        # Advance to half-window boundaries via the dechirp warping.
        #
        # phi_0 is the phase at sample 0, extracted from the dechirped FFT.
        # In the dechirped domain, the signal is a pure tone at freq_ref,
        # so the phase at dechirped position tau in [0, 1] is:
        #   phi_dechirp(tau) = phi_0 + 2*pi * freq_ref * tau
        #
        # The dechirp resampling maps original sample position n to
        # dechirped position:
        #   tau(n) = (exp(beta * n/k) - 1) / (exp(beta) - 1)
        # where beta = 2 * dlnf_applied (the grid value used for dechirping).
        #
        # For beta -> 0 this reduces to tau = n/k (identity), giving the
        # linear formula phi_0 + 2*pi*freq*n/k.
        dlnf_applied = dlnf_grid[dlnf_idx]  # (BW, K) — the actual dechirp used
        beta = 2.0 * dlnf_applied
        small = beta.abs() < 1e-8
        beta_safe = torch.where(small, torch.ones_like(beta), beta)
        eb = torch.exp(beta_safe)

        def _phase_at(n_frac):
            """Phase at original sample n_frac * k, via dechirp mapping."""
            tau = (torch.exp(beta_safe * n_frac) - 1.0) / (eb - 1.0)
            warped_adv = 2.0 * torch.pi * freq_ref * tau
            linear_adv = 2.0 * torch.pi * freq_ref * n_frac
            return torch.where(small, linear_adv, warped_adv)

        phase_start = (phi_0 + _phase_at(0.25)).reshape(B, W, K)
        phase_end = (phi_0 + _phase_at(0.75)).reshape(B, W, K)

        if not batched:
            phase_start = phase_start.squeeze(0)
            phase_end = phase_end.squeeze(0)

        return phase_start, phase_end

    def _correct_parabolic(self, delta: torch.Tensor) -> torch.Tensor:
        """Correct parabolic interpolation bias using precomputed LUT.

        Parabolic interpolation systematically underestimates fractional
        bin offsets for Hann windows.  This method maps the biased estimate
        to the true offset using an exact inverse mapping.

        Parameters
        ----------
        delta : Tensor
            Signed parabolic estimate of fractional bin offset.

        Returns
        -------
        corrected : Tensor, same shape as delta.
        """
        lut = self._para_corr_lut  # (n_inv,) on [0, 0.5]
        n_inv = lut.shape[0]
        sign = delta.sign()
        ad = delta.abs().clamp(max=0.5)
        t = ad * (n_inv - 1) / 0.5
        idx_lo = t.long().clamp(max=n_inv - 2)
        frac = t - idx_lo.float()
        corrected = lut[idx_lo] * (1 - frac) + lut[idx_lo + 1] * frac
        return sign * corrected

    def _scallop_correction(self, delta: torch.Tensor) -> torch.Tensor:
        """Look up scalloping correction factor from precomputed table.

        Parameters
        ----------
        delta : Tensor
            Fractional bin offset (signed); only |delta| matters.

        Returns
        -------
        correction : Tensor, same shape as delta.
            Multiply FFT magnitude at integer bin by this to recover
            the on-bin magnitude.
        """
        lut = self._scallop_lut  # (n_lut,) on [0, 0.5]
        n_lut = lut.shape[0]
        # Map |delta| to LUT index space [0, n_lut-1]
        t = delta.abs().clamp(max=0.5) * (n_lut - 1) / 0.5
        idx_lo = t.long().clamp(max=n_lut - 2)
        frac = t - idx_lo.float()
        return lut[idx_lo] * (1 - frac) + lut[idx_lo + 1] * frac

    def peak_amplitudes(self, X_start: torch.Tensor, X_end: torch.Tensor,
                        peaks: torch.Tensor, freq_refined: torch.Tensor,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract boundary amplitudes (A_start, A_end) from weighted FFTs.

        Uses the precomputed inverse mixing matrix to deconvolve the
        weighted FFT magnitudes into true boundary amplitudes, assuming
        linear amplitude variation within each window.  Corrects for
        Hann-window scalloping loss at fractional frequency bin offsets.

        The output amplitudes are true time-domain amplitudes (the 1/2
        factor from the cosine→complex exponential decomposition in the
        FFT is corrected for).

        Parameters
        ----------
        X_start, X_end : complex Tensor, shape (D, N_WINDOWS, k) or (D, B, N_WINDOWS, k)
            Weighted FFTs from forward(..., return_weighted=True).
        peaks : LongTensor, shape ([B,] N_WINDOWS, K, 2)
            Integer (dlnf_idx, freq_idx) from find_peaks.
        freq_refined : Tensor, shape ([B,] N_WINDOWS, K)
            Parabolic-interpolated fractional frequency bin index from
            find_peaks, used to correct for scalloping loss.

        Returns
        -------
        A_start, A_end : Tensor, shape ([B,] N_WINDOWS, K)
            Amplitude at window start and end boundaries, clamped >= 0.
        """
        batched = X_start.dim() == 4
        if not batched:
            X_start = X_start.unsqueeze(1)
            X_end = X_end.unsqueeze(1)
            peaks = peaks.unsqueeze(0)
            freq_refined = freq_refined.unsqueeze(0)

        D, B, W, k_full = X_start.shape
        Fk = self.k // 2 + 1
        K = peaks.shape[2]

        # Flatten batch*window for gathering
        dlnf_idx = peaks.reshape(B * W, K, 2)[:, :, 0]
        freq_idx = peaks.reshape(B * W, K, 2)[:, :, 1]

        # Rearrange: (D, B, W, k) -> (B, W, D, Fk) -> (BW, D*Fk)
        amp_s = X_start[..., :Fk].abs().permute(1, 2, 0, 3).reshape(B * W, D * Fk)
        amp_e = X_end[..., :Fk].abs().permute(1, 2, 0, 3).reshape(B * W, D * Fk)

        flat_idx = dlnf_idx * Fk + freq_idx  # (BW, K)
        mag_s = amp_s.gather(1, flat_idx)  # (BW, K)
        mag_e = amp_e.gather(1, flat_idx)  # (BW, K)

        # Correct for scalloping loss at fractional bin offset
        delta = freq_refined.reshape(B * W, K) - freq_idx.float()
        scallop_corr = self._scallop_correction(delta)
        mag_s = mag_s * scallop_corr
        mag_e = mag_e * scallop_corr

        # Apply inverse mixing matrix: [A_start, A_end] = amp_unmix @ [mag_s, mag_e]
        # Then multiply by 2 to correct for the cosine→complex exponential
        # factor of 1/2 in the FFT magnitudes.
        M_inv = self.amp_unmix  # (2, 2)
        A_start = 2.0 * (M_inv[0, 0] * mag_s + M_inv[0, 1] * mag_e)
        A_end = 2.0 * (M_inv[1, 0] * mag_s + M_inv[1, 1] * mag_e)

        A_start = A_start.clamp(min=0.0).reshape(B, W, K)
        A_end = A_end.clamp(min=0.0).reshape(B, W, K)

        if not batched:
            A_start = A_start.squeeze(0)
            A_end = A_end.squeeze(0)

        return A_start, A_end


class ToneTokenizer(nn.Module):
    """Tokenize time-domain signals into spectral peak features.

    Wraps DechirpSTFT and chains STFT -> peak finding -> phase
    extraction into a single batched forward pass.  Optionally whitens
    the STFT by estimated noise std before peak detection.

    Parameters
    ----------
    k : int
        Window size (FFT size) in samples.
    n_peaks : int
        Number of peaks to extract per time window.
    radius : int
        Peak suppression radius for max-pool peak detection.
    n_dlnf : int
        Number of dlnf (relative chirp rate) grid points.
    dlnf_min, dlnf_max : float
        Range of dlnf grid.
    noise_std : Tensor or None
        Pre-computed noise std per bin, shape (W, Fk) where Fk = k // 2 + 1.
        If None, no whitening is applied (can be set later via update_noise_std).

    Output
    ------
    forward(x) returns raw tokens of shape (B, N_WINDOWS, n_peaks, 9)
    with 9 values per peak: [snr, t_start, t_end, f_start, f_end, A_start,
    A_end, phase_start, phase_end].
    snr is the peak amplitude from the (optionally whitened) STFT — when
    whitening is active this is a true signal-to-noise ratio.
    t_start and t_end are sample positions at half-window boundaries (k/4 and
    3k/4), normalized to [-1, 1] over the signal length.  For adjacent windows,
    t_end[w] = t_start[w+1].
    f_start and f_end are normalized to [-1, 1] (mapping from [0, Fk-1] where
    Fk = k // 2 + 1).  A_start and A_end are boundary amplitudes recovered
    via weighted FFTs and mixing matrix inversion.  phase_start and phase_end
    are wrapped to [-pi, pi].  For adjacent windows, f_end[w] ≈ f_start[w+1]
    for clean signals.
    """

    def __init__(self, k: int = 1024, n_peaks: int = 3, radius: int = 2,
                 n_dlnf: int = 11, dlnf_min: float = 0.0, dlnf_max: float = 0.05,
                 noise_std: torch.Tensor = None):
        super().__init__()
        self.decomposer = DechirpSTFT(k=k)
        self.n_peaks = n_peaks
        self.radius = radius
        self.register_buffer(
            "dlnf_grid", torch.linspace(dlnf_min, dlnf_max, n_dlnf))
        # Noise std per bin for whitening: (W, Fk). None = no whitening.
        self.register_buffer("noise_std", noise_std)

    @property
    def k(self):
        return self.decomposer.k

    @property
    def n_raw(self):
        """Number of raw features per peak token (always 9)."""
        return 9

    @torch.no_grad()
    def update_noise_std(self, x: torch.Tensor, momentum: float = 0.99) -> None:
        """Update internal noise std estimate from a batch of signals.

        Computes a non-de-chirped STFT and estimates the standard deviation
        of amplitudes per (window, freq) bin.  On the first call (when
        ``self.noise_std is None``), the estimate is set directly.  On
        subsequent calls it is updated via exponential moving average::

            noise_std = momentum * noise_std + (1 - momentum) * std_new

        Parameters
        ----------
        x : Tensor, shape (B, N) or (N,)
            Time-domain signals (typically noise-dominated).
        momentum : float
            EMA smoothing factor (default 0.99).  Higher values give a
            slower, more stable estimate.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Non-de-chirped STFT — noise std is chirp-independent
        X = self.decomposer(x)                          # (B, W, k)
        Fk = self.decomposer.k // 2 + 1
        std_new = X[..., :Fk].abs().std(dim=0)           # (W, Fk)

        if self.noise_std is None:
            self.noise_std = std_new
        else:
            self.noise_std = momentum * self.noise_std + (1 - momentum) * std_new

    def _whiten(self, X: torch.Tensor) -> torch.Tensor:
        """Divide STFT positive-frequency bins by noise std.

        Parameters
        ----------
        X : complex Tensor, shape (D, B, W, k) or (B, W, k)

        Returns
        -------
        X_w : complex Tensor, same shape as X.
        """
        Fk = self.decomposer.k // 2 + 1
        # noise_std: (W, Fk) — broadcasts over leading (D, B) or (B,) dims
        scale = 1.0 / self.noise_std.clamp(min=1e-12)     # (W, Fk)
        X_w = X.clone()
        X_w[..., :Fk] = X_w[..., :Fk] * scale
        return X_w

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize batched time-domain signals.

        Parameters
        ----------
        x : Tensor, shape (B, N) or (N,)

        Returns
        -------
        tokens : Tensor, shape (B, W, K, 9) or (W, K, 9)
            Values per peak: [snr, t_start, t_end, f_start, f_end, A_start,
            A_end, phase_start, phase_end].
            snr is peak amplitude (SNR when whitened).
            t_start/t_end are normalized to [-1, 1] over signal length.
            f_start/f_end are normalized to [-1, 1].
            A_start/A_end are boundary amplitudes.
            phase_start/phase_end are wrapped to [-pi, pi].
            W = number of time windows, K = n_peaks.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        B, N = x.shape

        X, X_start, X_end = self.decomposer(
            x, dlnf=self.dlnf_grid, return_weighted=True)  # (D, B, W, k)

        if self.noise_std is not None:
            X = self._whiten(X)
            X_start = self._whiten(X_start)
            X_end = self._whiten(X_end)

        peaks, freq, dlnf, snr = self.decomposer.find_peaks(
            X, K=self.n_peaks, dlnf_grid=self.dlnf_grid, radius=self.radius)
        ps, pe = self.decomposer.peak_phases(
            X, peaks, freq, dlnf, self.dlnf_grid)
        A_start, A_end = self.decomposer.peak_amplitudes(
            X_start, X_end, peaks, freq)

        # Time at half-window boundaries: sample k/4 and 3k/4 per window
        k = self.decomposer.k
        hop = self.decomposer.hop
        W = peaks.shape[-3]
        w_idx = torch.arange(W, device=x.device, dtype=x.dtype)
        # Absolute sample positions of boundaries
        t_s = w_idx * hop + k // 4       # (W,)
        t_e = w_idx * hop + 3 * k // 4   # (W,)
        # Normalize to [-1, 1] over signal length
        t_s = 2.0 * t_s / (N - 1) - 1.0
        t_e = 2.0 * t_e / (N - 1) - 1.0
        # Broadcast to match peak dims: (W,) -> (W, 1) -> matches (B, W, K)
        t_start = t_s.unsqueeze(-1).expand_as(freq)
        t_end = t_e.unsqueeze(-1).expand_as(freq)

        # Frequency at half-window boundaries (dlnf is per hop,
        # boundaries are ±0.5 hops from center)
        f_start = freq * torch.exp(-dlnf / 2)
        f_end = freq * torch.exp(dlnf / 2)

        # Normalize frequencies to [-1, 1] from [0, Fk-1]
        Fk = self.decomposer.k // 2 + 1
        f_start = 2.0 * f_start / (Fk - 1) - 1.0
        f_end = 2.0 * f_end / (Fk - 1) - 1.0

        # Wrap phases to [-pi, pi]
        ps = (ps + torch.pi) % (2 * torch.pi) - torch.pi
        pe = (pe + torch.pi) % (2 * torch.pi) - torch.pi

        tokens = torch.stack(
            [snr, t_start, t_end, f_start, f_end, A_start, A_end, ps, pe], dim=-1)

        if squeeze:
            tokens = tokens.squeeze(0)
        return tokens
