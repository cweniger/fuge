"""EMRI-inspired voice stitching demo.

Simulates a single EMRI harmonic with exponential frequency evolution,
buried in noise at realistic total SNR (~30).  Motivated by semi-coherent
EMRI search strategies (e.g. arXiv:2510.20891) that split the signal
across ~10 time-frequency windows.

Usage:
    python examples/emri_demo.py
    python examples/emri_demo.py --snr 20 --n-windows 10
    python examples/emri_demo.py --snr 50 --n-harmonics 3
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from fuge.spectral import ChirpTokenizer, VoiceStitcher, VoiceStitchConfig


def make_emri_signal(N, k, f0, dlnf_per_hop, A, n_harmonics=1,
                     noise_sigma=1.0, seed=42):
    """Generate EMRI-like chirping harmonics in noise.

    Parameters
    ----------
    N : int
        Signal length in samples.
    k : int
        Window size (for computing hop = k/2).
    f0 : float
        Fundamental frequency at t=0 in cycles/sample.
    dlnf_per_hop : float
        Log-frequency drift per hop (exponential chirp rate).
    A : float
        Amplitude of the fundamental.
    n_harmonics : int
        Number of harmonics (1 = fundamental only).
    noise_sigma : float
        Noise standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    noisy : float32 array (N,)
    clean : float32 array (N,)
    f_harmonics : list of float64 arrays, instantaneous frequency per harmonic
    """
    rng = np.random.default_rng(seed)
    hop = k // 2
    t = np.arange(N, dtype=np.float64)

    clean = np.zeros(N, dtype=np.float64)
    f_harmonics = []

    for h in range(1, n_harmonics + 1):
        f_t = h * f0 * np.exp(dlnf_per_hop * t / hop)
        phase = 2 * np.pi * np.cumsum(f_t)
        # Amplitude falls off with harmonic number
        A_h = A / h
        clean += A_h * np.cos(phase)
        f_harmonics.append(f_t)

    noise = rng.standard_normal(N) * noise_sigma
    noisy = (clean + noise).astype(np.float32)
    return noisy, clean.astype(np.float32), f_harmonics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EMRI-inspired voice stitching demo")
    parser.add_argument("--snr", type=float, default=30.0,
                        help="Target total matched-filter SNR")
    parser.add_argument("--n-windows", type=int, default=10,
                        help="Number of time-frequency windows")
    parser.add_argument("--k", type=int, default=2048,
                        help="Window size in samples")
    parser.add_argument("--f0", type=float, default=0.1,
                        help="Fundamental frequency (cycles/sample)")
    parser.add_argument("--dlnf", type=float, default=0.005,
                        help="Log-frequency drift per hop")
    parser.add_argument("--n-harmonics", type=int, default=1,
                        help="Number of harmonics per source")
    parser.add_argument("--n-sources", type=int, default=1,
                        help="Number of overlapping EMRI sources")
    parser.add_argument("--n-peaks", type=int, default=3,
                        help="Peaks per window")
    parser.add_argument("--min-length", type=int, default=3,
                        help="Min tokens per voice")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("-o", "--output", type=str, default="emri_demo.png",
                        help="Output file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = args.k
    hop = k // 2
    W = args.n_windows
    N = k + (W - 1) * hop

    # Generate multiple overlapping sources
    sigma = 1.0
    rng_sources = np.random.default_rng(args.seed)

    # Each source gets a random f0 and dlnf, with target SNR
    source_params = []
    if args.n_sources == 1:
        source_params.append((args.f0, args.dlnf, args.snr))
    else:
        for s in range(args.n_sources):
            f0_s = 0.03 + 0.12 * rng_sources.random()  # 0.03–0.15 cyc/samp
            dlnf_s = 0.002 + 0.008 * rng_sources.random()  # mild chirps
            snr_s = args.snr * (0.5 + 0.5 * rng_sources.random())  # 50–100% of target
            source_params.append((f0_s, dlnf_s, snr_s))

    all_f_harmonics = []
    all_source_labels = []
    clean_total = np.zeros(N, dtype=np.float64)

    print(f"Signal: N={N}, k={k}, W={W}, σ={sigma:.1f}, sources={args.n_sources}")
    for s_idx, (f0_s, dlnf_s, snr_s) in enumerate(source_params):
        A_s = snr_s * sigma / np.sqrt(N / 2)
        _, clean_s, f_harmonics_s = make_emri_signal(
            N=N, k=k, f0=f0_s, dlnf_per_hop=dlnf_s, A=A_s,
            n_harmonics=args.n_harmonics, noise_sigma=0.0,
            seed=args.seed + s_idx + 1)
        clean_total += clean_s.astype(np.float64)

        print(f"  Source {s_idx}: f0={f0_s:.4f}, dlnf={dlnf_s:.4f}")
        t = np.arange(N, dtype=np.float64)
        for h_idx, f_h in enumerate(f_harmonics_s):
            A_h = A_s / (h_idx + 1)
            snr_h = np.sqrt(np.sum((A_h * np.cos(2 * np.pi * np.cumsum(f_h)))**2)) / sigma
            snr_per_w = A_h * np.sqrt(k / 2) / sigma
            print(f"    h{h_idx+1}: SNR={snr_h:.1f} (per-window≈{snr_per_w:.1f}), "
                  f"f=[{f_h[0]:.4f}, {f_h[-1]:.4f}]")
            all_f_harmonics.append(f_h)
            all_source_labels.append(f"s{s_idx}h{h_idx+1}")

    noise = rng_sources.standard_normal(N) * sigma
    noisy = (clean_total + noise).astype(np.float32)
    clean = clean_total.astype(np.float32)

    x = torch.from_numpy(noisy).unsqueeze(0).to(device)

    # Tokenize — dlnf range should cover all chirp rates
    max_dlnf_source = max(abs(p[1]) for p in source_params)
    dlnf_max = max(0.02, max_dlnf_source * 3)
    tokenizer = ChirpTokenizer(
        k=k, n_peaks=args.n_peaks,
        dlnf_min=0.0, dlnf_max=dlnf_max, n_dlnf=11)
    tokenizer = tokenizer.to(device)
    tokens = tokenizer(x)
    print(f"Tokens: {tokens.shape}")

    # Stitch
    config = VoiceStitchConfig(max_df=0.05, max_dphi=2.0, max_dA=0.8)
    stitcher = VoiceStitcher(config=config, min_length=args.min_length)
    voices = stitcher(tokens)

    print(f"Found {len(voices[0])} voices")
    for i, v in enumerate(voices[0]):
        print(f"  Voice {i}: {v.shape[0]} anchors, "
              f"f=[{v[:, 3].min():.4f}, {v[:, 3].max():.4f}], "
              f"Δφ_total={v[-1, 2] - v[0, 2]:.1f} rad")

    # --- Plot ---
    t_samples = np.arange(N)
    n_panels = 4 + (1 if len(voices[0]) > 0 else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels),
                             sharex=True)

    # Panel 1: Clean signal
    ax = axes[0]
    ax.plot(t_samples, clean, lw=0.5, color='#1f77b4')
    ax.set_ylabel("amplitude")
    ax.set_title(f"Clean EMRI signal ({args.n_sources} source"
                 f"{'s' if args.n_sources > 1 else ''}, "
                 f"{args.n_harmonics} harmonic"
                 f"{'s' if args.n_harmonics > 1 else ''})")

    # Panel 2: Noisy signal
    ax = axes[1]
    ax.plot(t_samples, noisy, lw=0.3, color='#1f77b4', alpha=0.7)
    ax.set_ylabel("amplitude")
    ax.set_title(f"Signal + noise (σ={sigma}, total SNR≈{args.snr:.0f})")

    # Panel 3: Chirp tokens
    ax = axes[2]
    tok = tokens[0].cpu()
    W_tok, K, _ = tok.shape
    for ki in range(K):
        t_mid = (tok[:, ki, 1] + tok[:, ki, 2]) / 2
        f_mid = (tok[:, ki, 3] + tok[:, ki, 4]) / 2
        amp = tok[:, ki, 0]
        mask = amp > 0
        sc = ax.scatter(t_mid[mask], f_mid[mask], c=amp[mask], s=8,
                        cmap="inferno", vmin=0)
    # True frequencies
    for h_idx, f_h in enumerate(all_f_harmonics):
        ax.plot(t_samples, f_h, '--', color='lime', lw=1, alpha=0.7,
                label=all_source_labels[h_idx])
    ax.set_ylabel("f (cycles/sample)")
    ax.set_title("Chirp tokens (color = peak amplitude)")
    ax.legend(fontsize=6, loc='upper left', ncol=3)
    fig.colorbar(sc, ax=ax, label="peak amplitude")

    # Panel 4: Stitched voices + true frequencies
    ax = axes[3]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(voices[0]), 1)))
    for i, v in enumerate(voices[0]):
        v_np = v.cpu().numpy()
        ax.plot(v_np[:, 1], v_np[:, 3], '-o', color=colors[i % len(colors)],
                ms=3, lw=1.5, label=f"voice {i}")
    for h_idx, f_h in enumerate(all_f_harmonics):
        ax.plot(t_samples, f_h, '--', color='gray', lw=1, alpha=0.5,
                label=all_source_labels[h_idx])
    ax.set_ylabel("f (cycles/sample)")
    ax.set_title("Stitched voices vs true frequency")
    ax.legend(fontsize=7, ncol=4)

    # Panel 5: Phase residual (if voices found)
    if len(voices[0]) > 0:
        ax = axes[4]
        for i, v in enumerate(voices[0]):
            v_np = v.cpu().numpy()
            phi = v_np[:, 2]
            t_anchor = v_np[:, 1]
            if len(t_anchor) > 1:
                slope = (phi[-1] - phi[0]) / (t_anchor[-1] - t_anchor[0])
                phi_detrend = phi - slope * (t_anchor - t_anchor[0]) - phi[0]
            else:
                phi_detrend = phi - phi[0]
            ax.plot(t_anchor, phi_detrend, '-o', color=colors[i % len(colors)],
                    ms=3, lw=1, label=f"voice {i}")
        ax.set_ylabel("φ − linear trend (rad)")
        ax.set_title("Coherent phase (detrended)")
        ax.legend(fontsize=7, ncol=4)

    axes[-1].set_xlabel("sample index")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
