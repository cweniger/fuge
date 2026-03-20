"""Voice stitching demo: tokenize a chirping signal and stitch into voices."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from fuge.spectral import ChirpTokenizer, VoiceStitcher, VoiceStitchConfig


def make_test_signal(N=50_000, f0=0.05, fdot=1e-6, A=5.0, noise_sigma=1.0, seed=42):
    """Chirping sinusoid with linear frequency drift.

    Returns (noisy_signal, clean_signal, f_true1, f_true2).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64)
    f_t = f0 + fdot * t
    phase = 2 * np.pi * np.cumsum(f_t)
    signal = A * np.cos(phase)
    # Add a second, weaker voice
    f2 = 0.12 + 0.5e-6 * t
    phase2 = 2 * np.pi * np.cumsum(f2)
    signal += 0.4 * A * np.cos(phase2)
    noise = rng.standard_normal(N) * noise_sigma
    return (signal + noise).astype(np.float32), signal.astype(np.float32), f_t, f2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice stitching demo")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise std dev")
    parser.add_argument("--N", type=int, default=50_000, help="Signal length in samples")
    parser.add_argument("--k", type=int, default=1024, help="Window size")
    parser.add_argument("--n-peaks", type=int, default=5, help="Peaks per window")
    parser.add_argument("--min-length", type=int, default=3, help="Min tokens per voice")
    parser.add_argument("--max-df", type=float, default=0.1, help="Frequency match threshold")
    parser.add_argument("--max-dphi", type=float, default=1.0, help="Phase match threshold (rad)")
    parser.add_argument("--max-dA", type=float, default=0.8, help="Amplitude match threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-o", "--output", type=str, default="voice_demo.png", help="Output file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noisy, clean, f_true1, f_true2 = make_test_signal(
        N=args.N, noise_sigma=args.sigma, seed=args.seed)
    x = torch.from_numpy(noisy).unsqueeze(0).to(device)  # (1, N)

    # Tokenize
    tokenizer = ChirpTokenizer(
        k=args.k, n_peaks=args.n_peaks, dlnf_min=0.0, dlnf_max=0.02, n_dlnf=11)
    tokenizer = tokenizer.to(device)
    tokens = tokenizer(x)  # (1, W, K, 9)
    print(f"Tokens shape: {tokens.shape}")

    # Stitch into voices
    config = VoiceStitchConfig(
        max_df=args.max_df, max_dphi=args.max_dphi, max_dA=args.max_dA)
    stitcher = VoiceStitcher(config=config, min_length=args.min_length)
    voices = stitcher(tokens)

    print(f"Found {len(voices[0])} voices")
    for i, v in enumerate(voices[0]):
        n_anchors = v.shape[0]
        print(f"  Voice {i}: {n_anchors} anchors, "
              f"f range [{v[:, 3].min():.4f}, {v[:, 3].max():.4f}] cycles/sample, "
              f"total phase advance: {v[-1, 2] - v[0, 2]:.1f} rad")

    # Plot
    N = args.N
    t_samples = np.arange(N)
    fig, axes = plt.subplots(5, 1, figsize=(14, 16),
                             sharex=True,
                             height_ratios=[1, 1, 1.2, 1.2, 1])

    # Panel 1: Clean signal
    ax = axes[0]
    ax.plot(t_samples, clean, lw=0.3, color='#1f77b4')
    ax.set_ylabel("amplitude")
    ax.set_title("Clean signal (two chirping sinusoids)")

    # Panel 2: Signal + noise
    ax = axes[1]
    ax.plot(t_samples, noisy, lw=0.3, color='#1f77b4', alpha=0.7)
    ax.set_ylabel("amplitude")
    ax.set_title(f"Signal + noise (σ = {args.sigma})")

    # Panel 3: Token spectrogram
    ax = axes[2]
    tok = tokens[0].cpu()  # (W, K, 9)
    W, K, _ = tok.shape
    for ki in range(K):
        t_mid = (tok[:, ki, 1] + tok[:, ki, 2]) / 2
        f_mid = (tok[:, ki, 3] + tok[:, ki, 4]) / 2
        snr = tok[:, ki, 0]
        mask = snr > 0
        sc = ax.scatter(t_mid[mask], f_mid[mask], c=snr[mask], s=3,
                        cmap="inferno", vmin=0)
    ax.set_ylabel("f (cycles/sample)")
    ax.set_title("Chirp tokens (color = peak amplitude)")
    fig.colorbar(sc, ax=ax, label="peak amplitude")

    # Panel 4: Voices in time-frequency
    ax = axes[3]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(voices[0]), 1)))
    for i, v in enumerate(voices[0]):
        v_np = v.cpu().numpy()
        ax.plot(v_np[:, 1], v_np[:, 3], '-o', color=colors[i % len(colors)],
                ms=2, lw=1.2, label=f"voice {i}")
    # True frequencies
    ax.plot(t_samples, f_true1, '--', color='gray', lw=0.5, alpha=0.7, label='true f1')
    ax.plot(t_samples, f_true2, '--', color='silver', lw=0.5, alpha=0.7, label='true f2')
    ax.set_ylabel("f (cycles/sample)")
    ax.set_title("Stitched voices")
    ax.legend(fontsize=7, ncol=4)

    # Panel 5: Unwrapped phase for each voice
    ax = axes[4]
    for i, v in enumerate(voices[0]):
        v_np = v.cpu().numpy()
        # Remove linear trend for visibility
        phi = v_np[:, 2]
        t_anchor = v_np[:, 1]
        if len(t_anchor) > 1:
            slope = (phi[-1] - phi[0]) / (t_anchor[-1] - t_anchor[0])
            phi_detrend = phi - slope * (t_anchor - t_anchor[0]) - phi[0]
        else:
            phi_detrend = phi - phi[0]
        ax.plot(t_anchor, phi_detrend, '-', color=colors[i % len(colors)],
                lw=1, label=f"voice {i}")
    ax.set_ylabel("φ − linear trend (rad)")
    ax.set_xlabel("sample index")
    ax.set_title("Coherent phase (detrended)")
    ax.legend(fontsize=7, ncol=4)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
