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


class SpectralDecomposer(nn.Module):
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

    def forward(self, x: torch.Tensor, a: float = 0.0, dlnf=0.0) -> torch.Tensor:
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

        Returns
        -------
        X : complex Tensor
            Scalar dlnf: shape (N_WINDOWS, k) or (B, N_WINDOWS, k).
            Batched dlnf: shape (D, N_WINDOWS, k) or (D, B, N_WINDOWS, k).
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)  # (1, N)

        # Unfold into overlapping windows: (B, N_WINDOWS, k)
        windows = x.unfold(dimension=1, size=self.k, step=self.hop)

        # Apply Hann window
        windowed = windows * self.window

        # --- Relative de-chirp via time-grid resampling ---
        # dlnf is per hop; full window spans 2 hops
        batched = isinstance(dlnf, torch.Tensor) and dlnf.dim() >= 1
        if batched:
            windowed = self._resample_dechirp_batched(windowed, 2.0 * dlnf)
            # (D, B, N_WINDOWS, k)
        elif dlnf != 0.0:
            windowed = self._resample_dechirp(windowed, 2.0 * dlnf)

        # --- Absolute de-chirp via phase multiplication ---
        if a != 0.0:
            chirp_kernel = torch.exp(-1j * a * self.t_norm ** 2)
            windowed = windowed * chirp_kernel

        X = torch.fft.fft(windowed, n=self.k, dim=-1)

        if squeeze:
            X = X.squeeze(1) if batched else X.squeeze(0)
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

        Parameters
        ----------
        X : complex Tensor, shape (D, N_WINDOWS, k)
            Batched STFT output from forward() with a 1-D dlnf tensor.
        K : int
            Number of peaks to return per time window.
        dlnf_grid : Tensor, shape (D,)
            The dlnf values used to produce X (assumed linearly spaced).
        radius : int
            Suppression radius (default 2): peak must be the maximum in
            a (2*radius+1) x (2*radius+1) window in the (dlnf, freq) plane.

        Returns
        -------
        peaks : LongTensor, shape (N_WINDOWS, K, 2)
            Integer (dlnf_idx, freq_idx) of each peak.
        freq_refined : Tensor, shape (N_WINDOWS, K)
            Parabolic-interpolated fractional frequency bin index.
        dlnf_refined : Tensor, shape (N_WINDOWS, K)
            Parabolic-interpolated dlnf value.
        values : Tensor, shape (N_WINDOWS, K)
            Amplitude at integer peak location.
        """
        # Positive frequencies only
        amp = X[:, :, :self.k // 2 + 1].abs()  # (D, N_WINDOWS, F)
        amp = amp.permute(1, 0, 2)              # (N_WINDOWS, D, F)
        W, D, Fk = amp.shape

        # Max pool to find local maxima
        kernel = 2 * radius + 1
        amp_4d = amp.unsqueeze(1)               # (W, 1, D, F)
        pooled = F.max_pool2d(amp_4d, kernel_size=kernel, stride=1, padding=radius)
        is_peak = (amp_4d == pooled).squeeze(1)  # (W, D, F)

        # Keep only peak amplitudes, flatten spatial dims
        amp_peaks = torch.where(is_peak, amp, torch.zeros_like(amp))
        amp_flat = amp_peaks.reshape(W, -1)      # (W, D*F)

        topk_vals, topk_idx = amp_flat.topk(K, dim=1)  # (W, K)

        # Unravel flat index -> (dlnf_idx, freq_idx)
        dlnf_idx = topk_idx // Fk
        freq_idx = topk_idx % Fk
        peaks = torch.stack([dlnf_idx, freq_idx], dim=-1)  # (W, K, 2)

        amp_2d = amp.reshape(W, -1)  # (W, D*Fk)

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

        return peaks, freq_refined, dlnf_refined, topk_vals

    def peak_phases(self, X: torch.Tensor, peaks: torch.Tensor,
                    freq_refined: torch.Tensor, dlnf_refined: torch.Tensor,
                    dlnf_grid: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate phase at hop boundaries for each detected peak.

        Uses the complex STFT value at the integer peak bin, corrects for
        the fractional-bin offset (Hann-window phase centre), then
        propagates the phase forward by one hop accounting for the chirp.

        The de-chirp resampling shifts the apparent frequency:
            f_dechirped = f_original * (exp(2*dlnf) - 1) / (2*dlnf)
        so freq_refined must be converted back to f_original before
        computing the phase increment.

        Parameters
        ----------
        X : complex Tensor, shape (D, N_WINDOWS, k)
        peaks : LongTensor, shape (N_WINDOWS, K, 2)
            Integer (dlnf_idx, freq_idx) from find_peaks.
        freq_refined : Tensor, shape (N_WINDOWS, K)
            Fractional frequency bin index (in de-chirped domain).
        dlnf_refined : Tensor, shape (N_WINDOWS, K)
            Interpolated dlnf value (true chirp rate).
        dlnf_grid : Tensor, shape (D,)
            The dlnf grid used for de-chirping.

        Returns
        -------
        phi_0 : Tensor, shape (N_WINDOWS, K)
            Phase at the start of each window.
        phi_1 : Tensor, shape (N_WINDOWS, K)
            Phase propagated forward by one hop (= start of next window).
        """
        dlnf_idx = peaks[:, :, 0]
        freq_idx = peaks[:, :, 1]

        # Gather complex STFT at integer peak bins
        Xp = X.permute(1, 0, 2).reshape(X.shape[1], -1)   # (W, D*k)
        flat_idx = dlnf_idx * X.shape[2] + freq_idx        # (W, K)
        X_peak = Xp.gather(1, flat_idx)                     # (W, K) complex

        # Phase at window start, corrected for fractional bin offset.
        # For Hann window with centre at sample (k-1)/2, a fractional
        # offset delta introduces phase pi * delta * (k-1) / k.
        f_delta = freq_refined - freq_idx.float()
        phi_start = X_peak.angle() - torch.pi * f_delta * (self.k - 1) / self.k

        # --- Convert de-chirped freq to original freq at window start ---
        # De-chirping with beta = 2*dlnf_used maps f_original to
        # f_dechirp = f_original * (exp(beta) - 1) / beta
        # so f_original = f_dechirp * beta / (exp(beta) - 1)
        dlnf_used = dlnf_grid[dlnf_idx]                     # (W, K)
        beta = 2.0 * dlnf_used
        safe_beta = torch.where(beta.abs() < 1e-8,
                                 torch.ones_like(beta) * 1e-8, beta)
        freq_correction = safe_beta / (torch.exp(safe_beta) - 1.0)
        freq_correction = torch.where(beta.abs() < 1e-8,
                                       torch.ones_like(freq_correction),
                                       freq_correction)
        f0_bin = freq_refined * freq_correction              # original freq in bins

        # --- Phase increment over one hop ---
        # integral_0^{T_hop} 2*pi * f0 * exp(dlnf_true * t/T_hop) dt
        # = pi * f0_bin * (exp(dlnf_true) - 1) / dlnf_true
        # (using f0_hz * T_hop = f0_bin / 2)
        dl = dlnf_refined
        safe_dl = torch.where(dl.abs() < 1e-10,
                               torch.ones_like(dl) * 1e-10, dl)
        chirp_factor = (torch.exp(dl) - 1.0) / safe_dl
        chirp_factor = torch.where(dl.abs() < 1e-10,
                                    torch.ones_like(chirp_factor), chirp_factor)

        phase_inc = torch.pi * f0_bin * chirp_factor

        return phi_start, phi_start + phase_inc
