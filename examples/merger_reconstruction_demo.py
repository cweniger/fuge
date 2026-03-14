"""Merger reconstruction demo.

Generates a toy black-hole-merger-like signal (chirp inspiral with
power-law amplitude rise, followed by exponential ringdown), tokenizes
it, reconstructs from tokens, and compares with the original.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, ".")
from fuge.spectral import ToneTokenizer

# ── Signal parameters ────────────────────────────────────────────────
N = 10_000
T_C_FRAC = 0.75          # coalescence at 3/4 of signal
T_C = int(T_C_FRAC * N)  # sample index of coalescence

F0 = 0.001               # initial frequency (cycles/sample)
F_RD = 0.01              # ringdown frequency (cycles/sample)
CHIRP_RATE = (F_RD - F0) / T_C  # constant df/dt

A0 = 0.1                 # amplitude scale
AMP_EXPONENT = -1.00     # power-law index for inspiral amplitude
A_PEAK = 10.0             # peak amplitude at coalescence

TAU_RD = 120.0           # ringdown e-folding time (samples)
# ~50 oscillations in ringdown: 5*tau_rd * f_rd = 5*500*0.02 = 50

# ── Tokenizer parameters ─────────────────────────────────────────────
K_WINDOW = 256
N_PEAKS = 1
N_DLNF = 51
DLNF_MIN = 0.0
DLNF_MAX = 0.3


def make_merger_signal(N, t_c, f0, chirp_rate, f_rd, A0, amp_exp, A_peak, tau_rd):
    """Generate a merger-like signal: chirp inspiral + exponential ringdown."""
    t = np.arange(N, dtype=np.float64)
    signal = np.zeros(N)

    # --- Inspiral phase (t < t_c) ---
    mask_insp = t < t_c
    t_insp = t[mask_insp]

    # Frequency: f(t) = f0 + chirp_rate * t
    phi_insp = 2.0 * np.pi * (f0 * t_insp + 0.5 * chirp_rate * t_insp**2)

    # Amplitude: power-law rise, capped at A_peak
    # A(t) = A0 * (1 - t/t_c)^amp_exp, but clamp so it doesn't exceed A_peak
    tau = np.maximum(1.0 - t_insp / t_c, 1e-6)
    A_insp = np.minimum(A0 * tau ** amp_exp, A_peak)

    signal[mask_insp] = A_insp * np.cos(phi_insp)

    # Phase at coalescence
    phi_c = 2.0 * np.pi * (f0 * t_c + 0.5 * chirp_rate * t_c**2)

    # --- Ringdown phase (t >= t_c) ---
    mask_rd = t >= t_c
    t_rd = t[mask_rd] - t_c

    phi_rd = phi_c + 2.0 * np.pi * f_rd * t_rd
    A_rd = A_peak * np.exp(-t_rd / tau_rd)

    signal[mask_rd] = A_rd * np.cos(phi_rd)

    return signal


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
    """
    hop = k // 2
    W, K, _ = tokens.shape
    Fk = k // 2 + 1

    n = np.arange(k)
    frac = (n - k / 4) / (k / 2)

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

            f_start_bin = (tokens[w, p, 3] + 1.0) / 2.0 * (Fk - 1)
            f_end_bin = (tokens[w, p, 4] + 1.0) / 2.0 * (Fk - 1)

            A_s = tokens[w, p, 5]
            A_e = tokens[w, p, 6]
            ps = tokens[w, p, 7]
            pe = tokens[w, p, 8]

            A_n = A_s + (A_e - A_s) * frac

            dphi = pe - ps
            f_center_bin = (f_start_bin + f_end_bin) / 2
            expected_dphi = np.pi * f_center_bin
            n_wraps = np.round((expected_dphi - dphi) / (2 * np.pi))
            dphi_unwrapped = dphi + n_wraps * 2 * np.pi

            df = f_end_bin - f_start_bin
            phi_n = (ps + dphi_unwrapped * frac
                     + (np.pi / 2) * df * frac * (frac - 1))

            window_signal += A_n * np.cos(phi_n)

        signal[start:end] += window_signal * window
        norm[start:end] += window

    mask = norm > 1e-12
    signal[mask] /= norm[mask]

    return signal


def main():
    # Generate signal
    print("Generating merger signal...")
    signal = make_merger_signal(
        N, T_C, F0, CHIRP_RATE, F_RD, A0, AMP_EXPONENT, A_PEAK, TAU_RD,
    )
    x = torch.from_numpy(signal).unsqueeze(0)  # (1, N)

    # Tokenize
    print("Tokenizing...")
    tokenizer = ToneTokenizer(
        k=K_WINDOW, n_peaks=N_PEAKS, n_dlnf=N_DLNF,
        dlnf_min=DLNF_MIN, dlnf_max=DLNF_MAX,
    ).double()

    # Get the dlnf=0 STFT (same as what goes into peak search)
    X0 = tokenizer.stft(x, dlnf=0.0)  # (1, W, k)
    stft_mag = X0[0, :, :K_WINDOW // 2 + 1].abs().numpy()  # (W, Fk)

    tokens = tokenizer(x)  # (1, W, K, 9)
    tokens = tokens[0].numpy()  # (W, K, 9)
    W, K, _ = tokens.shape
    print(f"  {W} windows, {K} peaks/window")

    # Reconstruct
    print("Reconstructing from tokens...")
    recon = reconstruct_from_tokens(tokens, K_WINDOW, N)

    # Compute residual
    residual = signal - recon
    margin = K_WINDOW

    # SNR by region
    sl = slice(margin, N - margin)
    rms_s = np.sqrt(np.mean(signal[sl] ** 2))
    rms_r = np.sqrt(np.mean(residual[sl] ** 2))
    snr = 20 * np.log10(rms_s / rms_r) if rms_r > 0 else np.inf
    print(f"  Reconstruction SNR: {snr:.1f} dB")

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    fig.suptitle(
        f"Merger reconstruction  (k={K_WINDOW}, {N_PEAKS} peaks, "
        f"SNR={snr:.1f} dB)",
        fontsize=13)

    t = np.arange(N)

    # Top: full signal comparison (wide panel)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t, signal, 'b-', lw=0.6, alpha=0.8, label='Original')
    ax_top.plot(t, recon, 'r--', lw=0.6, alpha=0.8, label='Reconstructed')
    ax_top.axvline(T_C, color='k', ls=':', lw=1, alpha=0.5, label=f'Coalescence (t={T_C})')
    ax_top.set_xlabel("Sample")
    ax_top.set_ylabel("Amplitude")
    ax_top.set_title("Full signal: inspiral + merger + ringdown")
    ax_top.legend(fontsize=9, loc='upper left')
    ax_top.grid(True, alpha=0.3)

    # Bottom left: residual over time
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.plot(t, residual, 'g-', lw=0.3, alpha=0.7)
    ax_bl.axvline(T_C, color='k', ls=':', lw=1, alpha=0.5, label='Coalescence')
    ax_bl.set_xlabel("Sample")
    ax_bl.set_ylabel("Residual")
    ax_bl.set_title("Residual (original - reconstructed)")
    ax_bl.legend(fontsize=8)
    ax_bl.grid(True, alpha=0.3)

    # Bottom right: tokenizer STFT at dlnf=0
    ax_br = fig.add_subplot(gs[1, 1])
    hop = K_WINDOW // 2
    Fk = K_WINDOW // 2 + 1
    t_wins = np.arange(stft_mag.shape[0]) * hop + K_WINDOW / 2  # window centers
    f_bins = np.arange(Fk) / K_WINDOW  # cycles/sample
    ax_br.pcolormesh(t_wins, f_bins, 20 * np.log10(stft_mag.T + 1e-12),
                     cmap='magma', shading='nearest')
    ax_br.axvline(T_C, color='w', ls=':', lw=1, alpha=0.7)
    ax_br.set_ylim(0, 5 * F_RD)
    ax_br.set_xlabel("Sample")
    ax_br.set_ylabel("Frequency (cycles/sample)")
    ax_br.set_title("DechirpSTFT magnitude (dlnf=0)")

    out = "merger_reconstruction_demo.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
