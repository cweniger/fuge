"""Spectral decomposition demo: noise sweep and phase continuity."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import torch

from fuge.spectral import DechirpSTFT, PeakFinder

if __name__ == "__main__":
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = 4096
    decomposer = DechirpSTFT(k=k).to(device)
    peak_finder = PeakFinder(k=k).to(device)
    noise_rms = np.sqrt(k * 3.0 / 8.0)

    # --- dlnf grid hyperparameters ---
    dlnf_min, dlnf_max, n_dlnf = 0.0, 0.05, 21
    dlnf_grid = torch.linspace(dlnf_min, dlnf_max, n_dlnf, device=device)

    # --- Noise sweep (spectral_demo.png) ---
    noise_sigmas = [1.0, 5.0, 15.0, 40.0]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(len(noise_sigmas), 1,
                              figsize=(12, 3.5 * len(noise_sigmas)),
                              sharex=True, sharey=True)

    for ax, sigma in zip(axes, noise_sigmas):
        noise = rng.standard_normal(N) * sigma
        x_noisy = h + noise
        x = torch.from_numpy(x_noisy).float().to(device).unsqueeze(0)  # (1, N)

        X_grid = decomposer(x, dlnf=dlnf_grid)  # (D, 1, W, k)
        amp_zero = X_grid[0, 0].abs().cpu().numpy()[:, :k // 2 + 1]
        snr = amp_zero / (noise_rms * sigma)
        n_windows = snr.shape[0]
        t_centers = (np.arange(n_windows) * decomposer.hop + k / 2) / fs
        f_bins = np.arange(k // 2 + 1) * fs / k

        im = ax.pcolormesh(t_centers, f_bins, snr.T, shading="nearest",
                            cmap="inferno", vmin=0, vmax=30)
        ax.set_yscale("log")
        ax.set_ylim(params["f0"] * 0.5,
                     params["f0"] * (params["n_harmonics"] + 1) * 5)

        peaks, freq_refined, dlnf_refined, peak_vals = peak_finder.find_peaks(
            X_grid, K=3, dlnf_grid=dlnf_grid)
        # Remove batch dim (B=1)
        f_peak = freq_refined[0].cpu().numpy() * fs / k
        dlnf_peak = dlnf_refined[0].cpu().numpy()
        dt_half = decomposer.hop / fs / 2

        for wi in range(n_windows):
            for ki in range(3):
                fc, dl, tc = f_peak[wi, ki], dlnf_peak[wi, ki], t_centers[wi]
                ax.plot([tc - dt_half, tc + dt_half],
                        [fc * np.exp(-dl / 2), fc * np.exp(dl / 2)],
                        color="cyan", lw=0.8, solid_capstyle="round")

        ax.set_ylabel("f (Hz)")
        ax.set_title(f"noise σ = {sigma}")
        fig.colorbar(im, ax=ax, label="SNR")

    axes[-1].set_xlabel("t (s)")
    plt.tight_layout()
    plt.savefig("spectral_demo.png", dpi=150)
    print("Saved spectral_demo.png")

    # --- Phase continuity demo (peaks_demo.png) ---
    phase_sigmas = [1.0, 5.0]
    n_windows = (N - k) // (k // 2) + 1
    t_centers = (np.arange(n_windows) * decomposer.hop + k / 2) / fs
    f_bins = np.arange(k // 2 + 1) * fs / k
    dt_half = decomposer.hop / fs / 2
    norm = Normalize(vmin=0, vmax=np.pi)
    cmap_phase = plt.cm.RdYlGn_r
    labels = ["fundamental", "2nd harmonic", "3rd harmonic"]
    colors_line = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig2, axes2 = plt.subplots(
        len(phase_sigmas) * 2, 1, figsize=(12, 5 * len(phase_sigmas)),
        height_ratios=[2, 1] * len(phase_sigmas),
        sharex=True)

    for si, sigma in enumerate(phase_sigmas):
        ax_spec = axes2[si * 2]
        ax_res = axes2[si * 2 + 1]

        noise = np.random.default_rng(42).standard_normal(N) * sigma
        x = torch.from_numpy((h + noise)).float().to(device).unsqueeze(0)  # (1, N)
        X_grid = decomposer(x, dlnf=dlnf_grid)  # (D, 1, W, k)

        amp_zero = X_grid[0, 0].abs().cpu().numpy()[:, :k // 2 + 1]
        snr = amp_zero / (noise_rms * sigma)

        peaks, freq_refined, dlnf_refined, peak_vals = peak_finder.find_peaks(
            X_grid, K=3, dlnf_grid=dlnf_grid)
        phase_start, phase_end = peak_finder.peak_phases(
            X_grid, peaks, freq_refined, dlnf_refined, dlnf_grid)

        # Remove batch dim (B=1) for plotting
        phase_start = phase_start[0]
        phase_end = phase_end[0]

        # Phase residual: wrap(phase_start[w+1] - phase_end[w])
        residual = phase_start[1:] - phase_end[:-1]
        residual = ((residual + torch.pi) % (2 * torch.pi)) - torch.pi
        residual_np = residual.cpu().numpy()
        t_res = 0.5 * (t_centers[:-1] + t_centers[1:])

        f_peak = freq_refined[0].cpu().numpy() * fs / k
        dlnf_peak = dlnf_refined[0].cpu().numpy()

        res_abs = np.abs(residual_np)
        res_pad = np.concatenate([res_abs, res_abs[-1:]], axis=0)

        # -- Spectrogram with coloured slope lines --
        im = ax_spec.pcolormesh(t_centers, f_bins, snr.T, shading="nearest",
                                 cmap="inferno", vmin=0, vmax=30)
        ax_spec.set_yscale("log")
        ax_spec.set_ylim(params["f0"] * 0.5,
                          params["f0"] * (params["n_harmonics"] + 1) * 5)

        for ki in range(3):
            segments, colors = [], []
            for wi in range(n_windows):
                fc, dl, tc = f_peak[wi, ki], dlnf_peak[wi, ki], t_centers[wi]
                segments.append([(tc - dt_half, fc * np.exp(-dl / 2)),
                                 (tc + dt_half, fc * np.exp(dl / 2))])
                colors.append(cmap_phase(norm(res_pad[wi, ki])))
            lc = LineCollection(segments, colors=colors, linewidths=1.0,
                                capstyle="round")
            ax_spec.add_collection(lc)

        sm = plt.cm.ScalarMappable(cmap=cmap_phase, norm=norm)
        fig2.colorbar(sm, ax=ax_spec, label="|Δφ| (rad)")
        fig2.colorbar(im, ax=ax_spec, label="SNR", location="left")
        ax_spec.set_ylabel("f (Hz)")
        ax_spec.set_title(f"Phase continuity (σ = {sigma}): "
                           "green = coherent, red = lost")

        # -- Phase residual scatter --
        for ki in range(3):
            ax_res.scatter(t_res, residual_np[:, ki], s=3, alpha=0.6,
                            color=colors_line[ki], label=labels[ki])
        ax_res.axhline(0, color="gray", lw=0.5, ls="--")
        ax_res.set_ylim(-np.pi * 1.1, np.pi * 1.1)
        ax_res.set_ylabel("Δφ (rad)")
        ax_res.set_title(f"Phase residual (σ = {sigma}):  φ₀(w+1) − φ₁(w)")
        if si == 0:
            ax_res.legend(loc="upper left", fontsize=8, ncol=3)

    axes2[-1].set_xlabel("t (s)")
    plt.tight_layout()
    plt.savefig("peaks_demo.png", dpi=150)
    print("Saved peaks_demo.png")
