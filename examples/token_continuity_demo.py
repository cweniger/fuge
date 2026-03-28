"""Token boundary continuity demo.

Generates a chirp signal, tokenizes it, and plots the continuity of
boundary quantities across adjacent windows:
  - f_end[w] vs f_start[w+1]
  - A_end[w] vs A_start[w+1]
  - phase_end[w] vs phase_start[w+1]

For a clean (noiseless) signal these should match closely.  The plots
show both the raw values over time and the boundary mismatch histograms.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, ".")
from chirp import chirp_signal
from fuge.spectral import ChirpTokenizer

# ── Signal parameters ────────────────────────────────────────────────
N = 100_000
F0 = 2.75e-3
CHIRP_MASS = 1.0
T_C = 1e6
A0 = 5.0
HARMONIC_DECAY = 1.5
N_HARMONICS = 4
NOISE_SIGMA = 0.5

# ── Tokenizer parameters ─────────────────────────────────────────────
K_WINDOW = 1024
N_PEAKS = 1       # single strongest peak per window for cleaner plots
N_DLNF = 11
DLNF_MIN = 0.0
DLNF_MAX = 0.05

SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    # Generate signal
    print("Generating chirp signal...")
    signal = chirp_signal(
        f0=F0, chirp_mass=CHIRP_MASS, t_c=T_C, A0=A0,
        harmonic_decay=HARMONIC_DECAY, n_harmonics=N_HARMONICS, N=N,
    )
    noise = rng.standard_normal(N) * NOISE_SIGMA
    x = torch.from_numpy(signal + noise).unsqueeze(0)  # (1, N)

    # Tokenize
    print("Tokenizing...")
    tokenizer = ChirpTokenizer(
        k=K_WINDOW, n_peaks=N_PEAKS, n_dlnf=N_DLNF,
        dlnf_min=DLNF_MIN, dlnf_max=DLNF_MAX,
    ).double()

    tokens = tokenizer(x)  # (1, W, K, 9)
    tokens = tokens.data[0, :, 0, :]  # (W, 9) — first (only) peak
    W = tokens.shape[0]
    print(f"  {W} windows, token shape per peak: {tokens.shape}")

    # Extract fields: [snr, t_start, t_end, f_start, f_end, A_start, A_end, ps, pe]
    snr = tokens[:, 0].numpy()
    t_start = tokens[:, 1].numpy()
    t_end = tokens[:, 2].numpy()
    f_start = tokens[:, 3].numpy()
    f_end = tokens[:, 4].numpy()
    A_start = tokens[:, 5].numpy()
    A_end = tokens[:, 6].numpy()
    ps = tokens[:, 7].numpy()
    pe = tokens[:, 8].numpy()

    # Window center time for x-axis
    t_center = (t_start + t_end) / 2

    # Boundary differences: end[w] - start[w+1]
    df = f_end[:-1] - f_start[1:]
    dA = A_end[:-1] - A_start[1:]
    # Phase difference mod 2*pi
    dphi = pe[:-1] - ps[1:]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    t_boundary = t_end[:-1]

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(
        f"Token boundary continuity  (k={K_WINDOW}, noise_sigma={NOISE_SIGMA})",
        fontsize=14, y=0.98)

    # --- Row 0: Frequency ---
    ax = axes[0, 0]
    ax.plot(t_center, f_start, '.', ms=2, alpha=0.5, label='f_start')
    ax.plot(t_center, f_end, '.', ms=2, alpha=0.5, label='f_end')
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Frequency (normalized)")
    ax.set_title("Frequency boundaries over time")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_boundary, df, '.', ms=2, alpha=0.5, color='C2')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("f_end[w] - f_start[w+1]")
    ax.set_title("Frequency boundary mismatch")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.hist(df, bins=50, color='C2', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("f_end[w] - f_start[w+1]")
    ax.set_ylabel("Count")
    ax.set_title(f"Freq mismatch histogram (std={np.std(df):.4f})")
    ax.grid(True, alpha=0.3)

    # --- Row 1: Amplitude ---
    ax = axes[1, 0]
    ax.plot(t_center, A_start, '.', ms=2, alpha=0.5, label='A_start')
    ax.plot(t_center, A_end, '.', ms=2, alpha=0.5, label='A_end')
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude boundaries over time")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_boundary, dA, '.', ms=2, alpha=0.5, color='C3')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("A_end[w] - A_start[w+1]")
    ax.set_title("Amplitude boundary mismatch")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.hist(dA, bins=50, color='C3', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("A_end[w] - A_start[w+1]")
    ax.set_ylabel("Count")
    ax.set_title(f"Amplitude mismatch histogram (std={np.std(dA):.4f})")
    ax.grid(True, alpha=0.3)

    # --- Row 2: Phase ---
    ax = axes[2, 0]
    ax.plot(t_center, ps, '.', ms=2, alpha=0.5, label='phase_start')
    ax.plot(t_center, pe, '.', ms=2, alpha=0.5, label='phase_end')
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Phase boundaries over time")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t_boundary, dphi, '.', ms=2, alpha=0.5, color='C4')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("phase_end[w] - phase_start[w+1]")
    ax.set_title("Phase boundary mismatch")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.hist(dphi, bins=50, color='C4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("phase_end[w] - phase_start[w+1] (rad)")
    ax.set_ylabel("Count")
    ax.set_title(f"Phase mismatch histogram (std={np.std(dphi):.4f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "token_continuity_demo.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
