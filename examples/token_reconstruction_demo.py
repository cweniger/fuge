"""Token reconstruction demo.

Generates a chirp signal, tokenizes it, reconstructs the signal from
tokens alone using overlap-add synthesis, and compares with the original.

Reconstruction per window per peak:
  1. Linearly interpolate phase from phase_start (at sample k/4)
     to phase_end (at sample 3k/4), extrapolate to window edges.
  2. Linearly interpolate amplitude from A_start to A_end similarly.
  3. Synthesize: s(n) = A(n) * cos(phi(n))
  4. Apply Hann window, overlap-add.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, ".")
from chirp import chirp_signal
from fuge.spectral import ToneTokenizer

# ── Signal parameters ────────────────────────────────────────────────
N = 100_000
F0 = 2.75e-3
CHIRP_MASS = 1.0
T_C = 1e6
A0 = 5.0
HARMONIC_DECAY = 1.5
N_HARMONICS = 4
NOISE_SIGMA = 0.0  # noiseless for clean reconstruction test

# ── Tokenizer parameters ─────────────────────────────────────────────
K_WINDOW = 1024
N_PEAKS = 4
N_DLNF = 11
DLNF_MIN = 0.0
DLNF_MAX = 0.05

SEED = 42


def reconstruct_from_tokens(tokens, k, N_signal):
    """Reconstruct a time-domain signal from tone tokens via overlap-add.

    Parameters
    ----------
    tokens : ndarray, shape (W, K, 9)
        Token array: [snr, t_start, t_end, f_start, f_end,
                       A_start, A_end, phase_start, phase_end].
    k : int
        Window size.
    N_signal : int
        Output signal length.

    Returns
    -------
    signal : ndarray, shape (N_signal,)
    """
    hop = k // 2
    W, K, _ = tokens.shape
    Fk = k // 2 + 1

    # Sample indices within a window
    n = np.arange(k)
    # Fractional position: 0 at k/4, 1 at 3k/4
    frac = (n - k / 4) / (k / 2)  # (k,)

    # Must match torch.hann_window (periodic=True by default)
    window = torch.hann_window(k, dtype=torch.float64).numpy()

    signal = np.zeros(N_signal)
    norm = np.zeros(N_signal)

    for w in range(W):
        start = w * hop
        end = start + k
        if end > N_signal:
            break

        window_signal = np.zeros(k)

        for p in range(K):
            snr = tokens[w, p, 0]
            if snr <= 0:
                continue

            # Denormalize frequencies from [-1, 1] to bin indices
            f_start_bin = (tokens[w, p, 3] + 1.0) / 2.0 * (Fk - 1)
            f_end_bin = (tokens[w, p, 4] + 1.0) / 2.0 * (Fk - 1)

            # Factor of 2: FFT of w(n)*A*cos(...) gives |X| = A*sum(w)/2
            # but the mixing matrix uses sum(w*basis) without the 1/2,
            # so recovered amplitudes are half the true time-domain value.
            A_s = tokens[w, p, 5] * 2.0
            A_e = tokens[w, p, 6] * 2.0
            ps = tokens[w, p, 7]
            pe = tokens[w, p, 8]

            # Linearly interpolate amplitude
            A_n = A_s + (A_e - A_s) * frac  # (k,)

            # Phase interpolation: unwrap the phase difference
            dphi = pe - ps
            # The total phase advance from k/4 to 3k/4 should be
            # close to 2*pi*f_center * (k/2) / k = pi*f_center_bin
            # Use frequency to resolve wrapping ambiguity
            f_center_bin = (f_start_bin + f_end_bin) / 2
            expected_dphi = np.pi * f_center_bin  # 2*pi*f*(k/2)/k
            # Find the multiple of 2*pi closest to expected
            n_wraps = np.round((expected_dphi - dphi) / (2 * np.pi))
            dphi_unwrapped = dphi + n_wraps * 2 * np.pi

            # Linear phase interpolation
            phi_n = ps + dphi_unwrapped * frac  # (k,)

            window_signal += A_n * np.cos(phi_n)

        # Overlap-add with Hann synthesis window.
        # COLA property: sum of shifted Hann windows = 1.0 in interior,
        # so we normalize by sum(window) not sum(window^2).
        signal[start:end] += window_signal * window
        norm[start:end] += window

    # Normalize by window overlap (avoid division by zero at edges)
    mask = norm > 1e-12
    signal[mask] /= norm[mask]

    return signal


def main():
    rng = np.random.default_rng(SEED)

    # Generate signal
    print("Generating chirp signal...")
    signal_clean = chirp_signal(
        f0=F0, chirp_mass=CHIRP_MASS, t_c=T_C, A0=A0,
        harmonic_decay=HARMONIC_DECAY, n_harmonics=N_HARMONICS, N=N,
    )
    noise = rng.standard_normal(N) * NOISE_SIGMA
    signal = signal_clean + noise
    x = torch.from_numpy(signal).unsqueeze(0)  # (1, N)

    # Tokenize
    print("Tokenizing...")
    tokenizer = ToneTokenizer(
        k=K_WINDOW, n_peaks=N_PEAKS, n_dlnf=N_DLNF,
        dlnf_min=DLNF_MIN, dlnf_max=DLNF_MAX,
    ).double()

    tokens = tokenizer(x)  # (1, W, K, 9)
    tokens = tokens[0].numpy()  # (W, K, 9)
    W, K, _ = tokens.shape
    print(f"  {W} windows, {K} peaks/window")

    # Reconstruct
    print("Reconstructing from tokens...")
    recon = reconstruct_from_tokens(tokens, K_WINDOW, N)

    # Compute residual
    residual = signal_clean - recon
    margin = K_WINDOW

    # SNR by region
    regions = {
        "full": slice(margin, N - margin),
        "first 90%": slice(margin, int(0.9 * N)),
        "last 10%": slice(int(0.9 * N), N - margin),
    }
    snr_by_region = {}
    print("  Reconstruction SNR by region:")
    for label, sl in regions.items():
        rms_s = np.sqrt(np.mean(signal_clean[sl] ** 2))
        rms_r = np.sqrt(np.mean(residual[sl] ** 2))
        snr = 20 * np.log10(rms_s / rms_r) if rms_r > 0 else np.inf
        snr_by_region[label] = snr
        print(f"    {label:12s}: {snr:.1f} dB  (signal RMS={rms_s:.3f}, residual RMS={rms_r:.4f})")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Token reconstruction  (k={K_WINDOW}, {N_PEAKS} peaks, "
        f"noise_sigma={NOISE_SIGMA})\n"
        f"SNR: full={snr_by_region['full']:.1f} dB, "
        f"first 90%={snr_by_region['first 90%']:.1f} dB, "
        f"last 10%={snr_by_region['last 10%']:.1f} dB",
        fontsize=13, y=1.0)

    t = np.arange(N)

    # Zoomed comparison — early (good region)
    ax = axes[0, 0]
    z1 = slice(N // 4, N // 4 + 6 * K_WINDOW)
    ax.plot(t[z1], signal_clean[z1], 'b-', lw=0.8, alpha=0.8, label='Original')
    ax.plot(t[z1], recon[z1], 'r--', lw=0.8, alpha=0.8, label='Reconstructed')
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title("Early region (6 windows at t=25%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Zoomed comparison — late (harder region)
    ax = axes[0, 1]
    z2_start = int(0.85 * N)
    z2 = slice(z2_start, z2_start + 6 * K_WINDOW)
    ax.plot(t[z2], signal_clean[z2], 'b-', lw=0.8, alpha=0.8, label='Original')
    ax.plot(t[z2], recon[z2], 'r--', lw=0.8, alpha=0.8, label='Reconstructed')
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title("Late region (6 windows at t=85%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Residual over time
    ax = axes[1, 0]
    ax.plot(t, residual, 'g-', lw=0.3, alpha=0.7)
    ax.axvline(0.9 * N, color='k', ls='--', lw=1, alpha=0.5, label='90% mark')
    ax.set_xlabel("Sample")
    ax.set_ylabel("Residual")
    ax.set_title("Residual (original - reconstructed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Residual spectrogram
    ax = axes[1, 1]
    ax.specgram(residual, NFFT=K_WINDOW, Fs=1.0, noverlap=K_WINDOW // 2,
                cmap='magma', scale='dB')
    ax.set_xlabel("Sample")
    ax.set_ylabel("Frequency (cycles/sample)")
    ax.set_title("Residual spectrogram")

    plt.tight_layout()
    out = "token_reconstruction_demo.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
