"""Synthetic data generator: merger-like signal in LISA-like noise.

Generates the same toy black-hole-merger waveform as merger_reconstruction_demo.py
(linear-chirp inspiral + exponential ringdown), buries it in noise with a
rough three-component LISA-like ASD:

    ASD(f) ∝ sqrt( (f_acc/f)^4  +  1  +  (f/f_shot)^2 )

  - f < f_acc  : acceleration noise wall (steep low-f rise, ASD ∝ f^-2)
  - f_acc–f_shot : flat bucket (best sensitivity)
  - f > f_shot : photon shot noise (ASD ∝ f)

Plots: clean/noisy waveforms, estimated noise floor, raw and whitened STFTs,
and reconstructed multi-resolution chirp tokens overlaid on the time-domain signal.

Usage:
    python notebooks/synthetic_data.py [--snr SNR] [--output FILE]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from fuge.spectral import ChirpTokenizer, DyadicChirpTokenizer

# ── Signal parameters (same as merger_reconstruction_demo.py) ────────────────
N         = 10_000
T_C       = int(0.75 * N)          # coalescence at sample 7500

F0        = 0.001                   # initial frequency (cycles/sample)
F_RD      = 0.01                    # ringdown frequency (cycles/sample)
CHIRP_RATE = (F_RD - F0) / T_C     # constant df/dt (linear chirp)

A0        = 0.1                     # amplitude at t=0
AMP_EXP   = -1.0                    # power-law index (amplitude rises as coalescence)
A_PEAK    = 10.0                    # peak amplitude at coalescence
TAU_RD    = 120.0                   # ringdown e-folding time (samples)

# ── Single-resolution tokenizer parameters ────────────────────────────────────
K         = 1024
N_PEAKS   = 3
N_DLNF    = 21
DLNF_MAX  = 0.3

# ── Multi-resolution tokenizer parameters ────────────────────────────────────
K_MIN     = 128
K_MAX     = 1024
N_PEAKS_D = 5
N_DLNF_D  = 11
DLNF_MAX_D = 0.3


# ── Signal / noise generators ─────────────────────────────────────────────────

def make_merger_signal(N, t_c, f0, chirp_rate, f_rd, A0, amp_exp, A_peak, tau_rd):
    """Toy merger waveform: linear-chirp inspiral + exponential ringdown."""
    t = np.arange(N, dtype=np.float64)
    out = np.zeros(N)

    # Inspiral (t < t_c): f(t) = f0 + chirp_rate*t, power-law amplitude
    m = t < t_c
    phi_i = 2 * np.pi * (f0 * t[m] + 0.5 * chirp_rate * t[m] ** 2)
    tau   = np.maximum(1.0 - t[m] / t_c, 1e-6)
    A_i   = np.minimum(A0 * tau ** amp_exp, A_peak)
    out[m] = A_i * np.cos(phi_i)

    # Ringdown (t >= t_c): fixed frequency, exponential decay
    phi_c = 2 * np.pi * (f0 * t_c + 0.5 * chirp_rate * t_c ** 2)
    m2    = ~m
    t_rd  = t[m2] - t_c
    out[m2] = A_peak * np.exp(-t_rd / tau_rd) * np.cos(phi_c + 2 * np.pi * f_rd * t_rd)

    return out.astype(np.float32)


# ── LISA-like noise parameters ────────────────────────────────────────────────
# Signal lives in F0=0.001 – F_RD=0.01 cycles/sample.
# Bucket minimum sits roughly in the middle of that band.
F_ACC  = 0.002   # acceleration knee: noise rises steeply below this
F_SHOT = 0.015   # shot-noise knee: noise rises above this


def lisa_asd(f):
    """Rough three-component LISA-like amplitude spectral density (arbitrary units).

    ASD(f) = sqrt( (F_ACC/f)^4 + 1 + (f/F_SHOT)^2 )

    Matches the qualitative shape of Fig. 6 in arXiv:1803.01944:
    steep low-f acceleration wall, flat bucket, rising shot noise.
    """
    f = np.where(f > 0, f, np.nan)
    return np.sqrt(0.1*(F_ACC / f) ** 4 + 1.0 + (f / F_SHOT) ** 2)


def make_lisa_noise(N, sigma, rng):
    """Noise coloured by the LISA-like ASD, normalised to overall std = sigma."""
    white = rng.standard_normal(N)
    freqs = np.fft.rfftfreq(N)
    asd   = lisa_asd(np.where(freqs > 0, freqs, freqs[1]))  # avoid DC=0
    F     = np.fft.rfft(white) * asd
    noise = np.fft.irfft(F, n=N)
    return (noise / noise.std() * sigma).astype(np.float32)


def reconstruct_token(t_start, t_end, phase_start, phase_end, A_start, A_end):
    """Reconstruct a chirp token as a sinusoid via linear phase/amplitude interp."""
    t = np.arange(int(np.round(t_start)), int(np.round(t_end)) + 1)
    if len(t) < 2:
        return t, np.zeros(len(t), dtype=np.float32)
    frac = (t - t_start) / (t_end - t_start)
    phi = phase_start + (phase_end - phase_start) * frac
    A   = A_start + (A_end - A_start) * frac
    return t, (A * np.cos(phi)).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr",    type=float, default=20.0,
                        help="Target matched-filter SNR (sets noise sigma)")
    parser.add_argument("--output", default="notebooks/synthetic_data.png")
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Signal
    signal = make_merger_signal(
        N, T_C, F0, CHIRP_RATE, F_RD, A0, AMP_EXP, A_PEAK, TAU_RD)

    # Noise sigma: calibrate so matched-filter SNR ≈ args.snr.
    # The effective noise power seen by the signal depends on the LISA ASD
    # shape integrated over the signal band; use signal RMS as a proxy.
    signal_rms = np.sqrt(np.mean(signal ** 2))
    sigma = signal_rms * np.sqrt(N) / args.snr
    noise = make_lisa_noise(N, sigma, rng)
    noisy = signal + noise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dev(arr):
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    # Build noise floor from B independent realisations
    B_noise = 64
    noise_batch = np.stack(
        [make_lisa_noise(N, sigma, rng) for _ in range(B_noise)])
    noise_tensor = torch.from_numpy(noise_batch).to(device)

    # ── Single-resolution tokenizer (for STFT display) ───────────────────────
    tokenizer = ChirpTokenizer(
        k=K, n_peaks=N_PEAKS, n_dlnf=N_DLNF,
        dlnf_min=0.0, dlnf_max=DLNF_MAX,
    ).to(device)
    tokenizer.noise_model.update(noise_tensor)

    with torch.no_grad():
        X_raw = tokenizer.stft(to_dev(noisy), dlnf=0., start=tokenizer.start)
        X_raw = X_raw[0, :, 0, :].abs().cpu().numpy()     # (W, Fk)

    noise_floor = tokenizer.noise_model.noise_std.cpu().numpy()  # (W, Fk)
    X_white = X_raw / noise_floor.clip(1e-12)

    hop   = K // 2
    Fk    = K // 2 + 1
    t_win = tokenizer.start + np.arange(X_raw.shape[0]) * hop + K / 2
    f_ax  = np.arange(Fk) / K

    # ── Multi-resolution tokenizer ────────────────────────────────────────────
    dyadic_tok = DyadicChirpTokenizer(
        k_min=K_MIN, k_max=K_MAX,
        n_peaks=N_PEAKS_D, n_dlnf=N_DLNF_D, dlnf_max=DLNF_MAX_D,
        f_min=F0 / 2, f_max=F_RD * 3,
    ).to(device)
    dyadic_tok(to_dev(noisy), noise=noise_tensor)   # warm up noise model
    with torch.no_grad():
        dyadic_tokens = dyadic_tok(to_dev(noisy))

    d      = dyadic_tokens.data[0].cpu().numpy()    # (N_tok, 9)
    scores = d[:, 0]
    score_max = np.percentile(scores[scores > 0], 99) if (scores > 0).any() else 1.0

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    fig.suptitle(
        f"Synthetic merger data  |  SNR≈{args.snr:.0f}, LISA-like noise,  k={K}",
        fontsize=13)

    gs  = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.4, 1.4])
    t_s = np.arange(N)

    # Row 0 left: clean signal
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_s, signal, lw=0.7, color="C0")
    ax.axvline(T_C, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_ylabel("Amplitude")
    ax.set_title("Clean signal")
    ax.set_xlim(0, N)

    # Row 0 right: noisy signal
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_s, noisy, lw=0.4, alpha=0.7, color="steelblue")
    ax.plot(t_s, signal, lw=0.8, color="C1", alpha=0.8, label="signal")
    ax.axvline(T_C, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_title(f"Noisy signal  (σ={sigma:.3f})")
    ax.set_xlim(0, N)
    ax.legend(fontsize=8, loc="upper left")

    # Row 1 left: noise PSD (estimated vs theoretical)
    ax = fig.add_subplot(gs[1, 0])
    mean_floor = noise_floor.mean(axis=0)
    ax.semilogy(f_ax, mean_floor, lw=1.2, label="estimated noise floor")
    for w in range(0, noise_floor.shape[0], max(1, noise_floor.shape[0] // 5)):
        ax.semilogy(f_ax, noise_floor[w], lw=0.5, alpha=0.3, color="C0")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("|STFT| amplitude")
    ax.set_title("Noise floor per bin  (mean + individual windows)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.5)

    # Row 1 right: measured floor vs LISA ASD model
    ax = fig.add_subplot(gs[1, 1])
    f_plot = f_ax[1:]
    model  = lisa_asd(f_plot)
    model_norm = model / model.min()
    floor_norm = mean_floor[1:] / mean_floor[1:].min()
    ax.loglog(f_plot, floor_norm, lw=1.2, label="estimated noise floor")
    ax.loglog(f_plot, model_norm, ls="--", lw=1.2, label="LISA ASD model")
    ax.axvline(F_ACC,  color="k", ls=":", lw=0.8, alpha=0.6, label=f"f_acc={F_ACC}")
    ax.axvline(F_SHOT, color="k", ls="-.", lw=0.8, alpha=0.6, label=f"f_shot={F_SHOT}")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("Relative ASD (normalised to bucket)")
    ax.set_title("Noise shape: measured vs LISA model")
    ax.legend(fontsize=8)
    ax.set_xlim(f_plot[0], 0.5)

    # Row 2 left: raw STFT magnitude (log scale)
    f_max_plot = min(5 * F_RD, 0.45)
    ax = fig.add_subplot(gs[2, 0])
    im = ax.pcolormesh(t_win, f_ax, 20 * np.log10(X_raw.T + 1e-12),
                       cmap="magma", shading="nearest")
    fig.colorbar(im, ax=ax, label="dB")
    ax.axvline(T_C, color="w", ls=":", lw=1, alpha=0.7)
    ax.set_ylim(0, f_max_plot)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Frequency (cycles/sample)")
    ax.set_title("Raw STFT magnitude (log)")

    # Row 2 right: whitened STFT (SNR per bin)
    ax = fig.add_subplot(gs[2, 1])
    vmax = np.percentile(X_white, 99)
    im = ax.pcolormesh(t_win, f_ax, X_white.T,
                       cmap="magma", shading="nearest", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="score (amplitude / noise floor)")
    ax.axvline(T_C, color="w", ls=":", lw=1, alpha=0.7)
    ax.set_ylim(0, f_max_plot)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Frequency (cycles/sample)")
    ax.set_title("Whitened STFT (score per bin)")

    # Row 3 (full width): multi-resolution token reconstruction
    ax = fig.add_subplot(gs[3, :])
    ax.plot(t_s, noisy, lw=0.3, alpha=0.25, color="gray", zorder=1)
    ax.plot(t_s, signal, lw=1.0, color="C1", alpha=0.7, label="signal", zorder=2)
    ax.axvline(T_C, color="k", ls=":", lw=1, alpha=0.4)

    cmap = plt.cm.viridis
    score_thresh = np.percentile(scores[scores > 0], 30) if (scores > 0).any() else 0
    for i in np.argsort(scores):          # plot low-score first so high-score on top
        sc = scores[i]
        if sc < score_thresh:
            continue
        t_tok, y_tok = reconstruct_token(d[i, 1], d[i, 2], d[i, 7], d[i, 8],
                                         d[i, 5], d[i, 6])
        alpha = 0.3 + 0.7 * min(sc / score_max, 1.0)
        color = cmap(min(sc / score_max, 1.0))
        ax.plot(t_tok, y_tok, color=color, alpha=alpha, lw=0.8, zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=score_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="token score")

    ax.set_xlim(0, N)
    sig_peak = np.abs(signal).max()
    ax.set_ylim(-2 * sig_peak, 2 * sig_peak)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Multi-resolution token reconstruction  "
        f"(k={K_MIN}–{K_MAX}, top-70% by score, coloured by score)")
    ax.legend(fontsize=8, loc="upper left")

    plt.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
