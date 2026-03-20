"""Spectral analysis: STFT, peak finding, noise estimation, tokenization.

Classes
-------
DechirpSTFT      — STFT with half-overlapping Hann windows and resample de-chirping.
PeakFinder       — Find and characterize spectral peaks (frequency, phase, amplitude).
NoiseModel       — Streaming noise PSD estimator for whitening.
ChirpTokenizer   — Orchestrates STFT → whitening → peak finding → token output.

Coordinate convention (see docs/spectral_math.md):
    t ∈ [-1, 1] across the window, n(t) = k/2 · (t + 1).
    Discrete samples: t_n = 2n/k - 1 for n = 0, …, k-1.
    Token boundaries at t = ±½ → samples k/4 and 3k/4.
    Periodic Hann window: zero at n=0, peak at n=k/2.
    β = 2·dlnf (total log-frequency change across the full window).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DechirpSTFT(nn.Module):
    """STFT with half-overlapping Hann windows and de-chirping.

    Parameters
    ----------
    k : int
        Window size in samples.  Must be a multiple of 4.
    R : int
        Warp resolution: the τ-grid has R·k samples, giving R·k/2+1
        frequency bins.  Default 1 (no extra resolution).

    Output shape: (B, N_WINDOWS, D, Fk) complex-valued,
    where Fk = R·k // 2 + 1.
    N_WINDOWS = (N - k) // (k // 2) + 1  (with hop = k // 2).
    """

    def __init__(self, k: int, R: int = 1):
        super().__init__()
        assert k % 4 == 0, f"k must be a multiple of 4, got {k}"
        self.k = k
        self.hop = k // 2
        self.R = R
        self.k_tau = R * k
        self.Fk = R * k // 2 + 1
        self.register_buffer("window", torch.hann_window(k, periodic=True))

        # Weighted sub-windows for boundary amplitude estimation.
        # t ∈ [-1, 1] across the window; w_start = ((1-t)/2)·hann,
        # w_end = ((1+t)/2)·hann.  w_start + w_end = hann.
        t = 2 * torch.arange(k, dtype=torch.float32) / k - 1  # t_n = 2n/k - 1
        self.register_buffer("window_start", ((1 - t) / 2) * self.window)
        self.register_buffer("window_end", ((1 + t) / 2) * self.window)

    def forward(self, x: torch.Tensor, dlnf: torch.Tensor = None,
                n_hann_splits: int = 1):
        """Compute the de-chirped windowed FFT.

        The chirp model is f(t) = f_center · exp(β·t) with t ∈ [-1, 1]
        and β = 2·dlnf.

        Parameters
        ----------
        x : Tensor, shape (B, N)
            Batched time-domain signals.
        dlnf : Tensor of shape (D,), or None
            Relative chirp parameter: change in ln(f) per hop step,
            i.e. dlnf = (fdot/f) · T_hop  (dimensionless).
            |dlnf| ≤ 0.5 supported (linear interpolation adequate).
            Computes the STFT for each value in parallel; output has a
            leading D dimension.  Default None is equivalent to [0.].
        n_hann_splits : int
            1 (default): return the standard Hann-windowed FFT.
            2: return (X_start, X_end) from weighted sub-windows.

        Returns
        -------
        X : complex Tensor, shape (B, N_WINDOWS, D, Fk) (if n_hann_splits=1)
            Fk = R·k // 2 + 1 positive-frequency bins (via rfft).
        (X_start, X_end) : tuple of complex Tensors (if n_hann_splits=2)
            Each has shape (B, N_WINDOWS, D, Fk).
        """
        if dlnf is None:
            dlnf = torch.zeros(1, device=x.device)

        # β = 2·dlnf: total log-frequency change across the full window
        beta = 2.0 * dlnf

        # Unfold into overlapping windows: (B, N_WINDOWS, k)
        raw_windows = x.unfold(dimension=1, size=self.k, step=self.hop)

        # Build list of windows to process
        if n_hann_splits == 2:
            win_funcs = [self.window_start, self.window_end]
        elif n_hann_splits == 1:
            win_funcs = [self.window]
        else:
            raise ValueError(f"n_hann_splits must be 1 or 2, got {n_hann_splits}")

        results = []
        for wf in win_funcs:
            windowed = raw_windows * wf

            # Resample via dechirp warp: (B, W, k) -> (D, B, W, k_tau)
            windowed = self._resample_dechirp_batched(windowed, beta)
            # Permute to (B, W, D, k_tau)
            windowed = windowed.permute(1, 2, 0, 3)

            results.append(torch.fft.rfft(windowed, n=self.k_tau, dim=-1))

        if n_hann_splits == 1:
            return results[0]
        return tuple(results)

    def _resample_dechirp_batched(self, windowed: torch.Tensor,
                                  beta: torch.Tensor) -> torch.Tensor:
        """Resample windowed segments for multiple chirp rates at once.

        Computes the chirped matched filter by resampling (gridding) the
        windowed signal onto a uniform τ-grid, with Jacobian correction
        for amplitude fidelity.

        Parameters
        ----------
        windowed : Tensor, shape (B, N_WINDOWS, k)
        beta : Tensor, shape (D,)
            β = 2·dlnf, total log-frequency change across the full window.

        Returns
        -------
        resampled : Tensor, shape (D, B, N_WINDOWS, k_tau)
        """
        D = beta.shape[0]
        k = self.k
        k_tau = self.k_tau

        # Destination grid in τ-space: k_tau samples at τ_n = 2n/k_tau - 1
        tau = (2 * torch.arange(k_tau, device=windowed.device, dtype=torch.float32)
               / k_tau - 1)  # (k_tau,)

        # Replace near-zero betas to avoid division by zero;
        # for |beta| < eps the warped grid is effectively identity.
        small = beta.abs() < 1e-8
        beta_safe = torch.where(small, torch.ones_like(beta) * 1e-8, beta)

        # Inverse warp: t(τ) = ln[1 + ((τ+1)/2)·(exp(2β)-1)] / β - 1
        # (see docs/spectral_math.md §3)
        e2b = torch.exp(2.0 * beta_safe)  # (D,)
        t_source = torch.log(
            1.0 + ((tau.unsqueeze(0) + 1.0) / 2.0)
            * (e2b.unsqueeze(1) - 1.0)
        ) / beta_safe.unsqueeze(1) - 1.0  # (D, k_tau)

        # For near-zero beta, use identity grid
        t_source = torch.where(
            small.unsqueeze(1),
            tau.unsqueeze(0),
            t_source)

        # Jacobian correction: exp(-β · (t(τ) - t(0)))
        # t(0) for the midpoint of τ-grid: t at τ = 2*(k_tau//2)/k_tau - 1
        tau_mid = 2.0 * (k_tau // 2) / k_tau - 1.0
        t_mid = torch.log(
            1.0 + ((tau_mid + 1.0) / 2.0) * (e2b - 1.0)
        ) / beta_safe - 1.0  # (D,)
        t_mid = torch.where(small, torch.tensor(tau_mid, device=beta.device), t_mid)
        jacobian = torch.exp(-beta_safe.unsqueeze(1) * (t_source - t_mid.unsqueeze(1)))
        jacobian = torch.where(small.unsqueeze(1), torch.ones_like(jacobian), jacobian)

        # Map source positions to index space [0, k-1] on the original grid
        # n(t) = k/2 · (t + 1)
        idx = (k / 2.0) * (t_source + 1.0)  # (D, k_tau)
        idx_lo = idx.long().clamp(0, k - 2)
        idx_hi = idx_lo + 1
        frac = idx - idx_lo.float()

        # Expand windowed (B, W, k) -> (D, B, W, k_tau) for gathering
        windowed_exp = windowed.unsqueeze(0).expand(D, -1, -1, -1)
        idx_lo_exp = idx_lo[:, None, None, :].expand(D, windowed.shape[0], windowed.shape[1], k_tau)
        idx_hi_exp = idx_hi[:, None, None, :].expand_as(idx_lo_exp)
        frac_exp = frac[:, None, None, :]

        lo_vals = torch.gather(windowed_exp, 3, idx_lo_exp)
        hi_vals = torch.gather(windowed_exp, 3, idx_hi_exp)
        resampled = lo_vals * (1.0 - frac_exp) + hi_vals * frac_exp

        # Apply Jacobian correction
        resampled = resampled * jacobian[:, None, None, :]

        return resampled


class PeakFinder(nn.Module):
    """Find and characterize spectral peaks from STFT output.

    Extracts peak locations via max-pool suppression, refines positions
    via parabolic interpolation, and recovers phases and boundary
    amplitudes at found peak locations.

    Parameters
    ----------
    k : int
        Window size (must match the DechirpSTFT that produced the input).
        Must be a multiple of 4.
    R : int
        Warp resolution (must match DechirpSTFT).
    correct_parabolic : bool
        Apply Hann-window parabolic interpolation bias correction (default True).
    correct_scalloping : bool
        Apply scalloping loss correction for boundary amplitude recovery (default True).
    """

    def __init__(self, k: int, R: int = 1, correct_parabolic: bool = True,
                 correct_scalloping: bool = True):
        super().__init__()
        assert k % 4 == 0, f"k must be a multiple of 4, got {k}"
        self.k = k
        self.R = R
        self.k_tau = R * k
        self.Fk = R * k // 2 + 1
        self.correct_parabolic = correct_parabolic
        self.correct_scalloping = correct_scalloping

        self._init_amplitude_unmix(k)
        if correct_scalloping:
            self._init_scalloping_lut(k)
        if correct_parabolic:
            self._init_parabolic_lut(k)

    def _init_amplitude_unmix(self, k: int):
        """Precompute the 2x2 mixing matrix inverse for boundary amplitudes.

        Linear amplitude model A(t) = A_start·(½-t) + A_end·(½+t),
        evaluated at the token boundaries t = ±½.
        """
        t = 2 * torch.arange(k, dtype=torch.float32) / k - 1
        window = torch.hann_window(k, periodic=True)
        window_start = ((1 - t) / 2) * window
        window_end = ((1 + t) / 2) * window
        basis_start = 0.5 - t
        basis_end = 0.5 + t
        M = torch.tensor([
            [(window_start * basis_start).sum(), (window_start * basis_end).sum()],
            [(window_end * basis_start).sum(), (window_end * basis_end).sum()],
        ])
        self.register_buffer("amp_unmix", torch.linalg.inv(M))

    def _init_scalloping_lut(self, k: int):
        """Precompute scalloping correction lookup table for weighted windows.

        At fractional bin offset delta, the FFT magnitude is reduced by
        |DTFT(w, delta)| / |DTFT(w, 0)|.  We store the inverse of this
        ratio on a dense grid for fast interpolation at runtime.
        """
        n_lut = 64
        ns = torch.arange(k, dtype=torch.float64)
        delta_lut = torch.linspace(0, 0.5, n_lut)
        t = 2 * torch.arange(k, dtype=torch.float64) / k - 1
        w_ref = ((1 - t) / 2 * torch.hann_window(k, periodic=True).double())
        dtft_mag = torch.zeros(n_lut)
        for i, d in enumerate(delta_lut):
            phasor = torch.exp(2j * torch.pi * d * ns / k)
            dtft_mag[i] = (w_ref * phasor).sum().abs()
        scallop_lut = dtft_mag[0] / dtft_mag.clamp(min=1e-12)
        self.register_buffer("_scallop_lut", scallop_lut.float())

    def _init_parabolic_lut(self, k: int):
        """Precompute parabolic interpolation correction LUT.

        Parabolic interpolation on Hann-windowed data systematically
        underestimates fractional bin offsets.  We precompute the exact
        forward mapping true_delta -> para_delta using the known Hann DTFT,
        then store the inverse for runtime correction.
        """
        n_corr = 1000
        ns = torch.arange(k, dtype=torch.float64)
        true_d = torch.linspace(0, 0.5, n_corr, dtype=torch.float64)
        para_d = torch.zeros(n_corr, dtype=torch.float64)
        w_hann = torch.hann_window(k, periodic=True).double()
        for i, td in enumerate(true_d):
            mags = []
            for off in [-1, 0, 1]:
                phasor = torch.exp(2j * torch.pi * (off - td) * ns / k)
                mags.append((w_hann * phasor).sum().abs())
            ym, y0, yp = mags
            denom = ym - 2.0 * y0 + yp
            para_d[i] = 0.5 * (ym - yp) / denom if denom.abs() > 1e-12 else 0.0
        n_inv = 64
        para_grid = torch.linspace(0, 0.5, n_inv, dtype=torch.float64)
        import numpy as _np
        true_inv = _np.interp(
            para_grid.numpy(), para_d.numpy(), true_d.numpy())
        self.register_buffer(
            "_para_corr_lut", torch.from_numpy(true_inv).float())

    def find_peaks(self, X: torch.Tensor, K: int, dlnf_grid: torch.Tensor,
                   radius: int = 2,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find top K peaks in the (dlnf, freq) plane per time window.

        Parameters
        ----------
        X : complex Tensor, shape (B, N_WINDOWS, D, Fk)
        K : int
            Number of peaks to return per time window.
        dlnf_grid : Tensor, shape (D,)
        radius : int
            Suppression radius (default 2).

        Returns
        -------
        peaks : LongTensor, shape (B, N_WINDOWS, K, 2)
            Integer (dlnf_idx, freq_idx) of each peak.
        freq_refined : Tensor, shape (B, N_WINDOWS, K)
            Parabolic-interpolated fractional frequency bin index.
        dlnf_refined : Tensor, shape (B, N_WINDOWS, K)
            Parabolic-interpolated dlnf value.
        values : Tensor, shape (B, N_WINDOWS, K)
            Amplitude at integer peak location.
        """
        B, W, D, Fk = X.shape

        amp = X.abs()
        amp = amp.reshape(B * W, D, Fk)
        BW = B * W

        kernel = 2 * radius + 1
        amp_4d = amp.unsqueeze(1)
        pooled = F.max_pool2d(amp_4d, kernel_size=kernel, stride=1, padding=radius)
        is_peak = (amp_4d == pooled).squeeze(1)

        best_d = amp.argmax(dim=1, keepdim=True)
        dlnf_mask = torch.zeros_like(is_peak)
        dlnf_mask.scatter_(1, best_d, 1)
        is_peak = is_peak & dlnf_mask.bool()

        amp_peaks = torch.where(is_peak, amp, torch.zeros_like(amp))
        amp_flat = amp_peaks.reshape(BW, -1)

        topk_vals, topk_idx = amp_flat.topk(K, dim=1)

        dlnf_idx = topk_idx // Fk
        freq_idx = topk_idx % Fk
        peaks = torch.stack([dlnf_idx, freq_idx], dim=-1)

        amp_2d = amp.reshape(BW, -1)

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
        if self.correct_parabolic:
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

        peaks = peaks.reshape(B, W, K, 2)
        freq_refined = freq_refined.reshape(B, W, K)
        dlnf_refined = dlnf_refined.reshape(B, W, K)
        topk_vals = topk_vals.reshape(B, W, K)

        return peaks, freq_refined, dlnf_refined, topk_vals

    def _forward_warp(self, t: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Forward warp τ(t) for given physical time t and chirp rate β.

        τ(t) = [exp(β·t) - exp(-β)] / sinh(β) - 1

        Parameters
        ----------
        t : float or Tensor
        beta : Tensor, shape (BW, K)

        Returns
        -------
        tau : Tensor, same shape as beta
        """
        small = beta.abs() < 1e-8
        beta_safe = torch.where(small, torch.ones_like(beta), beta)
        sinh_b = torch.sinh(beta_safe)
        tau = (torch.exp(beta_safe * t) - torch.exp(-beta_safe)) / sinh_b - 1.0
        tau = torch.where(small, torch.full_like(tau, float(t)), tau)
        return tau

    def peak_phases(self, X: torch.Tensor, peaks: torch.Tensor,
                    freq_refined: torch.Tensor, dlnf_refined: torch.Tensor,
                    dlnf_grid: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate phase at token boundaries (t = ±½) for each peak.

        Uses the periodic Hann phase anchor (§6 of spectral_math.md):
        φ_center = arg(X[m]) + π·m/R, exact and δ-independent.
        Then propagates to t = ±½ via the forward warp.

        Parameters
        ----------
        X : complex Tensor, shape (B, N_WINDOWS, D, Fk)
        peaks : LongTensor, shape (B, N_WINDOWS, K, 2)
        freq_refined : Tensor, shape (B, N_WINDOWS, K)
        dlnf_refined : Tensor, shape (B, N_WINDOWS, K)
        dlnf_grid : Tensor, shape (D,)

        Returns
        -------
        phase_start : Tensor, shape (B, N_WINDOWS, K)
            Phase at t = -½ (token start boundary).
        phase_end : Tensor, shape (B, N_WINDOWS, K)
            Phase at t = +½ (token end boundary).
        """
        B, W, D, Fk = X.shape
        K = peaks.shape[2]
        R = self.R

        dlnf_idx = peaks.reshape(B * W, K, 2)[:, :, 0]
        freq_idx = peaks.reshape(B * W, K, 2)[:, :, 1]
        freq_ref = freq_refined.reshape(B * W, K)

        Xp = X.reshape(B * W, D * Fk)
        flat_idx = dlnf_idx * Fk + freq_idx
        X_peak = Xp.gather(1, flat_idx)

        # Phase anchor at window center in τ-space (sample k/2):
        # φ_center = arg(X[m]) + π·m/R
        phi_center = X_peak.angle() + torch.pi * freq_idx.float() / R

        # Propagate to token boundaries via forward warp.
        # Phase advance: φ(τ) = φ_center + π·f_ref·τ
        # (since n(τ)-n(0) = k/2·τ, and advance = 2π·f_ref·(k/2·τ)/k = π·f_ref·τ)
        dlnf_applied = dlnf_grid[dlnf_idx]
        beta = 2.0 * dlnf_applied

        tau_start = self._forward_warp(-0.5, beta)
        tau_end = self._forward_warp(0.5, beta)

        phase_start = (phi_center + torch.pi * freq_ref * tau_start).reshape(B, W, K)
        phase_end = (phi_center + torch.pi * freq_ref * tau_end).reshape(B, W, K)

        return phase_start, phase_end

    def peak_amplitudes(self, X_start: torch.Tensor, X_end: torch.Tensor,
                        peaks: torch.Tensor, freq_refined: torch.Tensor,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract boundary amplitudes (A_start, A_end) from weighted FFTs.

        Uses the precomputed inverse mixing matrix to deconvolve the
        weighted FFT magnitudes into true boundary amplitudes at t = ±½.

        Parameters
        ----------
        X_start, X_end : complex Tensor, shape (B, N_WINDOWS, D, Fk)
        peaks : LongTensor, shape (B, N_WINDOWS, K, 2)
        freq_refined : Tensor, shape (B, N_WINDOWS, K)

        Returns
        -------
        A_start, A_end : Tensor, shape (B, N_WINDOWS, K)
            Amplitude at token boundaries t = ±½, clamped >= 0.
        """
        B, W, D, Fk = X_start.shape
        K = peaks.shape[2]

        dlnf_idx = peaks.reshape(B * W, K, 2)[:, :, 0]
        freq_idx = peaks.reshape(B * W, K, 2)[:, :, 1]

        amp_s = X_start.abs().reshape(B * W, D * Fk)
        amp_e = X_end.abs().reshape(B * W, D * Fk)

        flat_idx = dlnf_idx * Fk + freq_idx
        mag_s = amp_s.gather(1, flat_idx)
        mag_e = amp_e.gather(1, flat_idx)

        if self.correct_scalloping:
            delta = freq_refined.reshape(B * W, K) - freq_idx.float()
            scallop_corr = self._scallop_correction(delta)
            mag_s = mag_s * scallop_corr
            mag_e = mag_e * scallop_corr

        M_inv = self.amp_unmix
        A_start = 2.0 * (M_inv[0, 0] * mag_s + M_inv[0, 1] * mag_e)
        A_end = 2.0 * (M_inv[1, 0] * mag_s + M_inv[1, 1] * mag_e)

        A_start = A_start.clamp(min=0.0).reshape(B, W, K)
        A_end = A_end.clamp(min=0.0).reshape(B, W, K)

        return A_start, A_end

    def _correct_parabolic(self, delta: torch.Tensor) -> torch.Tensor:
        """Correct parabolic interpolation bias using precomputed LUT."""
        lut = self._para_corr_lut
        n_inv = lut.shape[0]
        sign = delta.sign()
        ad = delta.abs().clamp(max=0.5)
        t = ad * (n_inv - 1) / 0.5
        idx_lo = t.long().clamp(max=n_inv - 2)
        frac = t - idx_lo.float()
        corrected = lut[idx_lo] * (1 - frac) + lut[idx_lo + 1] * frac
        return sign * corrected

    def _scallop_correction(self, delta: torch.Tensor) -> torch.Tensor:
        """Look up scalloping correction factor from precomputed table."""
        lut = self._scallop_lut
        n_lut = lut.shape[0]
        t = delta.abs().clamp(max=0.5) * (n_lut - 1) / 0.5
        idx_lo = t.long().clamp(max=n_lut - 2)
        frac = t - idx_lo.float()
        return lut[idx_lo] * (1 - frac) + lut[idx_lo + 1] * frac


class NoiseModel(nn.Module):
    """Streaming noise PSD estimator for whitening STFT output.

    Holds a reference to a DechirpSTFT and maintains a running estimate
    of the noise standard deviation per (window, frequency) bin.

    Parameters
    ----------
    stft : DechirpSTFT
        The STFT instance used for signal analysis (same window, same k).
    momentum : float
        EMA smoothing factor for noise std updates (default 0.99).
    """

    def __init__(self, stft: DechirpSTFT, momentum: float = 0.99):
        super().__init__()
        self.stft = stft
        self.momentum = momentum
        self.register_buffer("noise_std", None)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update noise std estimate from a batch of pure-noise signals.

        Parameters
        ----------
        x : Tensor, shape (B, N)
        """
        X = self.stft(x)[:, :, 0, :]    # (B, W, D=1, Fk) -> (B, W, Fk)
        std_new = X.abs().std(dim=0)     # (W, Fk)

        if self.noise_std is None:
            self.noise_std = std_new
        else:
            self.noise_std = self.momentum * self.noise_std + (1 - self.momentum) * std_new

    def whiten(self, X: torch.Tensor) -> torch.Tensor:
        """Divide STFT bins by noise std.

        Parameters
        ----------
        X : complex Tensor, shape (B, W, D, Fk)

        Returns
        -------
        X_w : complex Tensor, same shape as X.
        """
        scale = 1.0 / self.noise_std.clamp(min=1e-12)
        return X * scale.unsqueeze(0).unsqueeze(2)


# --- Chirp tokenizer (renamed from ToneTokenizer) ---

class ChirpTokenizer(nn.Module):
    """Tokenize time-domain signals into chirp tokens.

    Orchestrates DechirpSTFT, PeakFinder, and (optionally) NoiseModel
    into a single forward pass: STFT → whitening → peak finding →
    normalization → chirp token output.

    Chirp tokens are the building blocks for voice formation: adjacent
    tokens with compatible boundary values (f_end[w] ≈ f_start[w+1],
    phase_end[w] ≈ phase_start[w+1]) can be stitched into phase-coherent
    voices.

    Parameters
    ----------
    k : int
        Window size in samples.  Must be a multiple of 4.
    R : int
        Warp resolution (default 1).
    n_peaks : int
        Number of peaks to extract per time window.
    radius : int
        Peak suppression radius for max-pool peak detection.
    n_dlnf : int
        Number of dlnf (relative chirp rate) grid points.
    dlnf_min, dlnf_max : float
        Range of dlnf grid.  |dlnf| ≤ 0.5 recommended.
    noise_model : NoiseModel or None
        Pre-configured noise model for whitening.

    Output
    ------
    forward(x) returns chirp tokens of shape (B, W, K, 9):
    [snr, t_start, t_end, f_start, f_end, A_start, A_end,
     phase_start, phase_end].
    All boundary quantities are at t = ±½ (samples k/4 and 3k/4).
    """

    def __init__(self, k: int = 1024, R: int = 1, n_peaks: int = 3,
                 radius: int = 2, n_dlnf: int = 11,
                 dlnf_min: float = 0.0, dlnf_max: float = 0.05,
                 noise_model: 'NoiseModel | None' = None):
        super().__init__()
        self.stft = DechirpSTFT(k=k, R=R)
        self.peak_finder = PeakFinder(k=k, R=R)
        self.noise_model = noise_model
        self.n_peaks = n_peaks
        self.radius = radius
        self.register_buffer(
            "dlnf_grid", torch.linspace(dlnf_min, dlnf_max, n_dlnf))

    @property
    def k(self):
        return self.stft.k

    @property
    def n_raw(self):
        """Number of raw features per chirp token (always 9)."""
        return 9

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize batched time-domain signals into chirp tokens.

        Parameters
        ----------
        x : Tensor, shape (B, N)

        Returns
        -------
        tokens : Tensor, shape (B, W, K, 9)
            [snr, t_start, t_end, f_start, f_end, A_start, A_end,
             phase_start, phase_end].
        """
        B, N = x.shape

        X_start, X_end = self.stft(
            x, dlnf=self.dlnf_grid, n_hann_splits=2)
        X = X_start + X_end

        if self.noise_model is not None:
            X_w = self.noise_model.whiten(X)
        else:
            X_w = X

        peaks, freq, dlnf, snr = self.peak_finder.find_peaks(
            X_w, K=self.n_peaks, dlnf_grid=self.dlnf_grid, radius=self.radius)
        ps, pe = self.peak_finder.peak_phases(
            X_w, peaks, freq, dlnf, self.dlnf_grid)
        A_start, A_end = self.peak_finder.peak_amplitudes(
            X_start, X_end, peaks, freq)

        # Time at token boundaries: n(±½) = k/4 and 3k/4
        k = self.stft.k
        hop = self.stft.hop
        W = peaks.shape[-3]
        w_idx = torch.arange(W, device=x.device, dtype=x.dtype)
        t_s = w_idx * hop + k // 4
        t_e = w_idx * hop + 3 * k // 4
        # Normalize to [-1, 1] over signal length
        t_s = 2.0 * t_s / (N - 1) - 1.0
        t_e = 2.0 * t_e / (N - 1) - 1.0
        t_start = t_s.unsqueeze(-1).expand_as(freq)
        t_end = t_e.unsqueeze(-1).expand_as(freq)

        # Frequency at token boundaries (dlnf is per hop = β/2,
        # boundaries span ±½ in t, so ±β/2 in log-frequency from center)
        f_start = freq * torch.exp(-dlnf / 2)
        f_end = freq * torch.exp(dlnf / 2)

        # Normalize frequencies to [-1, 1] from [0, Fk-1]
        Fk = self.stft.Fk
        f_start = 2.0 * f_start / (Fk - 1) - 1.0
        f_end = 2.0 * f_end / (Fk - 1) - 1.0

        # Wrap phases to [-pi, pi]
        ps = (ps + torch.pi) % (2 * torch.pi) - torch.pi
        pe = (pe + torch.pi) % (2 * torch.pi) - torch.pi

        tokens = torch.stack(
            [snr, t_start, t_end, f_start, f_end, A_start, A_end, ps, pe], dim=-1)

        return tokens


# Backwards compatibility alias
ToneTokenizer = ChirpTokenizer
