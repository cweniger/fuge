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

        eb = torch.exp(betas)  # (D,)
        # Source positions: (D, k)
        t_source = (2.0 / betas.unsqueeze(1)) * torch.log(
            1.0 + tau_uniform.unsqueeze(0) * (eb.unsqueeze(1) - 1.0)
        ) - 1.0

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

    def find_peaks(self, X: torch.Tensor, K: int, radius: int = 2
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find top K peaks in the (dlnf, freq) plane per time window.

        A pixel is a peak iff it equals the max in its (2r+1)x(2r+1)
        neighborhood (via max-pool), then the K strongest peaks are returned.
        Frequency positions are refined via parabolic interpolation on the
        amplitude of the peak bin and its two frequency neighbours.

        Parameters
        ----------
        X : complex Tensor, shape (D, N_WINDOWS, k)
            Batched STFT output from forward() with a 1-D dlnf tensor.
        K : int
            Number of peaks to return per time window.
        radius : int
            Suppression radius (default 2): peak must be the maximum in
            a (2*radius+1) x (2*radius+1) window in the (dlnf, freq) plane.

        Returns
        -------
        peaks : LongTensor, shape (N_WINDOWS, K, 2)
            Integer (dlnf_idx, freq_idx) of each peak.
        freq_refined : Tensor, shape (N_WINDOWS, K)
            Parabolic-interpolated fractional frequency bin index.
        values : Tensor, shape (N_WINDOWS, K)
            Parabolic-interpolated amplitude at each peak.
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

        # --- Parabolic interpolation along frequency axis ---
        fi = freq_idx.clamp(1, Fk - 2)  # ensure neighbours exist
        amp_2d = amp.reshape(W, -1)      # (W, D*Fk)
        flat_m = dlnf_idx * Fk + (fi - 1)
        flat_0 = dlnf_idx * Fk + fi
        flat_p = dlnf_idx * Fk + (fi + 1)
        y_m = amp_2d.gather(1, flat_m)   # (W, K)
        y_0 = amp_2d.gather(1, flat_0)
        y_p = amp_2d.gather(1, flat_p)

        denom = y_m - 2.0 * y_0 + y_p
        delta = 0.5 * (y_m - y_p) / denom
        delta = torch.where(denom.abs() < 1e-12, torch.zeros_like(delta), delta)
        delta = delta.clamp(-0.5, 0.5)

        freq_refined = fi.float() + delta               # fractional bin index
        values_refined = y_0 - 0.25 * (y_m - y_p) * delta  # interpolated amplitude

        return peaks, freq_refined, values_refined


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Inline test signal (PN-inspired chirp, no JAX dependency)
    params = dict(f0=1e-3, t_c=1e6, A0=5.0, n_harmonics=4, N=100_000)
    T_obs = 0.9 * params["t_c"]
    N = params["N"]
    t = np.linspace(0, T_obs, N)
    dt = t[1] - t[0]
    tau = 1 - t / params["t_c"]
    f_t = params["f0"] * tau ** (-3.0 / 8)
    A_t = params["A0"] * tau ** (-0.25)
    trapz = (f_t[:-1] + f_t[1:]) * 0.5 * dt
    phase = np.concatenate([[0.0], np.cumsum(trapz)]) * 2 * np.pi
    h = np.zeros(N)
    for k_harm in range(1, params["n_harmonics"] + 1):
        h += A_t * np.exp(-1.5 * (k_harm - 1)) * np.cos(k_harm * phase)
    fs = N / T_obs

    # Add unit-variance white noise
    noise = np.random.default_rng(42).standard_normal(N)
    x_noisy = h + noise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x_noisy).float().to(device)

    k = 4096
    decomposer = SpectralDecomposer(k=k).to(device)
    noise_rms = np.sqrt(k * 3.0 / 8.0)

    dlnf_values = [-0.1, -0.05, 0.0, 0.05, 0.1]

    # Batched call: all non-zero dlnf values in one forward pass
    dlnf_nonzero = [v for v in dlnf_values if v != 0.0]
    dlnf_tensor = torch.tensor(dlnf_nonzero, device=device)
    X_batched = decomposer(x, dlnf=dlnf_tensor)  # (D, N_WINDOWS, k)
    X_zero = decomposer(x, dlnf=0.0)              # (N_WINDOWS, k)

    # Reassemble in original order
    results = {}
    j = 0
    for v in dlnf_values:
        if v == 0.0:
            results[v] = X_zero
        else:
            results[v] = X_batched[j]
            j += 1

    fig, axes = plt.subplots(len(dlnf_values), 1, figsize=(12, 3 * len(dlnf_values)),
                              sharex=True, sharey=True)

    for ax, dlnf_val in zip(axes, dlnf_values):
        X = results[dlnf_val]
        amplitude = X.abs().cpu().numpy()[:, :k // 2 + 1]
        snr = amplitude / noise_rms

        n_windows = snr.shape[0]
        t_centers = (np.arange(n_windows) * decomposer.hop + k / 2) / fs
        f_bins = np.arange(k // 2 + 1) * fs / k

        im = ax.pcolormesh(t_centers, f_bins, snr.T, shading="nearest",
                            cmap="inferno", vmin=0)
        ax.set_yscale("log")
        ax.set_ylim(params["f0"] * 0.5,
                     params["f0"] * (params["n_harmonics"] + 1) * 5)
        ax.set_ylabel("f (Hz)")
        ax.set_title(f"dlnf = {dlnf_val}")
        fig.colorbar(im, ax=ax, label="SNR")

        # Overlay expected track: ln(f) increases by dlnf per window step
        if dlnf_val != 0.0:
            f_start = params["f0"] * 1.05
            win_idx = np.arange(n_windows)
            f_track = f_start * np.exp(dlnf_val * win_idx)
            ax.plot(t_centers, f_track, '--', color='cyan', lw=1, alpha=0.7,
                    label=f'slope = dlnf')
            ax.legend(loc='lower right', fontsize=8)

    axes[-1].set_xlabel("t (s)")
    plt.tight_layout()
    plt.savefig("spectral_demo.png", dpi=150)
    print("Saved spectral_demo.png")

    # --- Peak finder plot: spectrogram (dlnf=0) with top-3 peaks overlaid ---
    X_dlnf0 = X_zero.unsqueeze(0)  # (1, N_WINDOWS, k)
    peaks, freq_refined, peak_vals = decomposer.find_peaks(X_dlnf0, K=3)

    amplitude = X_zero.abs().cpu().numpy()[:, :k // 2 + 1]
    snr = amplitude / noise_rms
    n_windows = snr.shape[0]
    t_centers = (np.arange(n_windows) * decomposer.hop + k / 2) / fs
    f_bins = np.arange(k // 2 + 1) * fs / k

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    im = ax2.pcolormesh(t_centers, f_bins, snr.T, shading="nearest",
                         cmap="inferno", vmin=0)
    ax2.set_yscale("log")
    ax2.set_ylim(params["f0"] * 0.5,
                  params["f0"] * (params["n_harmonics"] + 1) * 5)

    # Scatter top-3 peaks (parabolic-interpolated) for every time window
    f_peak = freq_refined.cpu().numpy() * fs / k         # (N_WINDOWS, 3) in Hz
    t_peak = np.repeat(t_centers[:, None], 3, axis=1)    # (N_WINDOWS, 3)
    ax2.scatter(t_peak.ravel(), f_peak.ravel(), color="cyan", s=8,
                edgecolors="white", linewidths=0.3, zorder=5, label="top-3 peaks")

    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("f (Hz)")
    ax2.set_title("dlnf = 0, with detected peaks (parabolic interpolation)")
    ax2.legend(loc="lower right")
    fig2.colorbar(im, ax=ax2, label="SNR")
    plt.tight_layout()
    plt.savefig("peaks_demo.png", dpi=150)
    print("Saved peaks_demo.png")
