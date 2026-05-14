"""Score calibration demo.

Verifies that token.score correctly tracks signal strength relative to the
local noise floor for a single exponentially-chirping sinusoid buried in
coloured (1/f) noise.

Signal: amplitude A chirp from f0 to f1 over 10 000 samples.
Noise:  1/f coloured noise, variance varying 10x across the frequency band.

With noise model active (noise= passed to forward), high-score tokens should
align with the chirp track regardless of the noise colour.  Without it, raw
amplitude peaks cluster at the loud low-frequency end instead.

Usage:
    python notebooks/score_demo.py [--amplitude A] [--snr SNR]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from fuge.spectral import ChirpTokenizer


# ---------------------------------------------------------------------------
# Signal and noise generators
# ---------------------------------------------------------------------------

def make_chirp(N: int, f0: float, f1: float, amplitude: float) -> np.ndarray:
    """Exponential chirp from f0 to f1 over N samples."""
    t = np.arange(N)
    log_ratio = np.log(f1 / f0)
    inst_freq = f0 * np.exp(log_ratio * t / N)
    phase = 2 * np.pi * np.cumsum(inst_freq)
    return amplitude * np.sin(phase).astype(np.float32), inst_freq


def make_colored_noise(N: int, alpha: float, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """1/f^alpha noise with overall std sigma."""
    white = rng.standard_normal(N)
    F = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1.0                   # avoid div-by-zero at DC
    F *= freqs ** (-alpha / 2)
    noise = np.fft.irfft(F, n=N)
    return (noise / noise.std() * sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplitude", type=float, default=1.0,
                        help="Chirp amplitude")
    parser.add_argument("--snr", type=float, default=5.0,
                        help="Approximate per-window SNR target (sets noise std)")
    parser.add_argument("--output", default="notebooks/score_demo.png")
    args = parser.parse_args()

    # Parameters
    N = 10_000
    k = 1024
    f0, f1 = 0.05, 0.20    # cycles/sample
    alpha = 0.01             # noise colour exponent (1 = pink, 2 = brown)

    # The Hann-window peak amplitude scales as ~k/2.
    # Noise floor per bin scales as ~sigma * sqrt(k * 3/8).
    # So theoretical per-bin SNR ≈ A * sqrt(k/6) / sigma.
    # Invert to set sigma for the requested SNR.
    sigma = args.amplitude * np.sqrt(k / 6) / args.snr

    rng = np.random.default_rng(42)
    signal, inst_freq = make_chirp(N, f0, f1, args.amplitude)
    noise = make_colored_noise(N, alpha, sigma, rng)
    noisy = signal + noise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChirpTokenizer(k=k, n_peaks=5, dlnf_max=0.15, n_dlnf=11).to(device)

    def to_dev(arr):
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    x        = to_dev(noisy)
    x_signal = to_dev(signal)

    # Build a stable noise floor from B_noise independent realisations.
    B_noise = 64
    noise_batch = np.stack(
        [make_colored_noise(N, alpha, sigma, rng) for _ in range(B_noise)])
    x_noise_batch = torch.from_numpy(noise_batch).to(device)
    tokenizer.noise_model.update(x_noise_batch)

    tokens_calibrated = tokenizer(x)        # measured (signal + noise)

    # --- Without noise model (fresh tokenizer, no noise update) ---
    tok_raw = ChirpTokenizer(k=k, n_peaks=5, dlnf_max=0.15, n_dlnf=11).to(device)
    tokens_raw = tok_raw(x)

    # Unpack for plotting
    def unpack(toks):
        d = toks.data[0].cpu().numpy()   # (N_tok, 9)
        t_c = (d[:, 1] + d[:, 2]) / 2
        f_c = (d[:, 3] + d[:, 4]) / 2
        sc  = d[:, 0]
        return t_c, f_c, sc

    t_cal, f_cal, sc_cal = unpack(tokens_calibrated)
    t_raw, f_raw, sc_raw = unpack(tokens_raw)

    # Expected score at each measured token position.
    # Compute the signal-only STFT amplitude at the same (window, dlnf, freq_bin)
    # where each measured peak was found — no peak matching needed.
    with torch.no_grad():
        Xs, Xe = tokenizer.stft(
            x_signal, dlnf=tokenizer.dlnf_grid, n_hann_splits=2,
            start=tokenizer.start)
        X_sig_amp = (Xs + Xe).abs()[0].cpu().numpy()   # (W, D, Fk)

    noise_floor = tokenizer.noise_model.noise_std.cpu().numpy()  # (W, Fk)
    W_nf, Fk = noise_floor.shape
    D = X_sig_amp.shape[1]

    # Map each measured token to its window index via t_start
    d_cal_raw = tokens_calibrated.data[0].cpu().numpy()
    t_start_vals = d_cal_raw[:, 1]
    unique_ts = np.sort(np.unique(t_start_vals))
    w_idx = np.searchsorted(unique_ts, t_start_vals).clip(0, W_nf - 1)

    # Integer frequency bin and dlnf index from stored token fields
    f_bin_idx = np.round(f_cal * k).astype(int).clip(0, Fk - 1)
    dlnf_tok = np.log((d_cal_raw[:, 4] / d_cal_raw[:, 3]).clip(1e-9))
    dlnf_grid_np = tokenizer.dlnf_grid.cpu().numpy()
    d_idx = np.argmin(np.abs(dlnf_tok[:, None] - dlnf_grid_np[None, :]), axis=1)

    # expected_sc: |STFT(signal)| / noise_floor at each token's peak bin
    expected_sc = (X_sig_amp[w_idx, d_idx, f_bin_idx]
                   / noise_floor[w_idx, f_bin_idx].clip(1e-12))

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Score calibration demo  |  A={args.amplitude:.2f}, σ={sigma:.3f}, "
        f"target SNR≈{args.snr:.1f}, 1/f^{alpha:.0f} noise",
        fontsize=12)

    t_ax = np.arange(N)

    # Top-left: waveform
    ax = axes[0, 0]
    ax.plot(t_ax, noisy, lw=0.4, alpha=0.6, color="steelblue", label="noisy")
    ax.plot(t_ax, signal, lw=1.2, color="C1", label="signal")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal + coloured noise")
    ax.legend(fontsize=8)

    # Top-right: calibrated scores
    ax = axes[0, 1]
    sc_max = np.percentile(sc_cal, 98)
    sc = ax.scatter(t_cal, f_cal, c=sc_cal, cmap="viridis",
                    s=15, vmin=0, vmax=sc_max, rasterized=True)
    fig.colorbar(sc, ax=ax, label="score (calibrated)")
    ax.plot(t_ax, inst_freq, "r--", lw=1.2, label="true f(t)")
    ax.set_xlim(0, N)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Frequency (cycles/sample)")
    ax.set_title("Token scores — with noise model")
    ax.legend(fontsize=8)

    # Bottom-left: raw scores (no noise model)
    ax = axes[1, 0]
    sc_max_raw = np.percentile(sc_raw, 98)
    sc2 = ax.scatter(t_raw, f_raw, c=sc_raw, cmap="viridis",
                     s=15, vmin=0, vmax=sc_max_raw, rasterized=True)
    fig.colorbar(sc2, ax=ax, label="score (raw amplitude)")
    ax.plot(t_ax, inst_freq, "r--", lw=1.2, label="true f(t)")
    ax.set_xlim(0, N)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Frequency (cycles/sample)")
    ax.set_title("Token scores — no noise model (raw amplitude)")
    ax.legend(fontsize=8)

    # Bottom-right: measured vs expected (signal STFT at measured peak bins).
    # Signal tokens (high expected score) should lie on the 1:1 line;
    # noise tokens (expected ≈ 0) scatter near y ≈ 1.
    ax = axes[1, 1]
    ax.scatter(expected_sc, sc_cal, s=12, alpha=0.5, rasterized=True)
    lim = max(expected_sc.max(), sc_cal.max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="score = expected")
    ax.set_xlabel("Expected score  |STFT(signal)| / noise_floor")
    ax.set_ylabel("Measured score  |STFT(signal+noise)| / noise_floor")
    ax.set_title("Score calibration")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
