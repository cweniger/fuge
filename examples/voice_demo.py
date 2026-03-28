"""Chirp linking demo: tokenize a chirping signal and link into chains."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from fuge.spectral import ChirpTokenizer, ChirpLinker, ChirpLinkConfig


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
    parser = argparse.ArgumentParser(description="Chirp linking demo")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise std dev")
    parser.add_argument("--N", type=int, default=50_000, help="Signal length in samples")
    parser.add_argument("--k", type=int, default=1024, help="Window size")
    parser.add_argument("--n-peaks", type=int, default=5, help="Peaks per window")
    parser.add_argument("--min-length", type=int, default=3, help="Min tokens per chain")
    parser.add_argument("--max-df", type=float, default=0.1, help="Frequency match threshold")
    parser.add_argument("--max-dphi", type=float, default=1.0, help="Phase match threshold (rad)")
    parser.add_argument("--max-dA", type=float, default=0.8, help="Amplitude match threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-o", "--output", type=str, default="voice_demo.png", help="Output file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    noisy, clean, f_true1, f_true2 = make_test_signal(
        N=args.N, noise_sigma=args.sigma, seed=args.seed)
    x = torch.from_numpy(noisy).unsqueeze(0).to(device)  # (1, N)

    # True matched-filter SNR for each component: sqrt(sum(h^2)) / sigma
    t = np.arange(args.N, dtype=np.float64)
    h1 = 5.0 * np.cos(2 * np.pi * np.cumsum(0.05 + 1e-6 * t))
    h2 = 2.0 * np.cos(2 * np.pi * np.cumsum(0.12 + 0.5e-6 * t))
    snr1 = np.sqrt(np.sum(h1**2)) / args.sigma
    snr2 = np.sqrt(np.sum(h2**2)) / args.sigma
    print(f"True matched-filter SNR: voice 1 = {snr1:.1f}, voice 2 = {snr2:.1f}")

    # Tokenize
    tokenizer = ChirpTokenizer(
        k=args.k, n_peaks=args.n_peaks, dlnf_min=0.0, dlnf_max=0.02, n_dlnf=11)
    tokenizer = tokenizer.to(device)
    tokens = tokenizer(x)
    print(f"Tokens: {tokens}")

    # Link tokens
    config = ChirpLinkConfig(
        max_df=args.max_df, max_dphi=args.max_dphi, max_dA=args.max_dA)
    linker = ChirpLinker(config=config, min_length=args.min_length)
    linked = linker(tokens)
    print(f"Linked: {linked}")

    chain_ids = linked.chain_id[0].cpu()
    unique_chains = chain_ids.unique()
    n_chains = (unique_chains >= 0).sum().item()
    print(f"Found {n_chains} chains")
    for cid in unique_chains:
        if cid < 0:
            continue
        mask = chain_ids == cid
        n_tok = mask.sum().item()
        f_vals = linked.f_start[0].cpu()[mask]
        snr_val = linked.snr[0].cpu()[mask][0].item()
        print(f"  Chain {int(cid)}: {n_tok} tokens, "
              f"f=[{f_vals.min():.4f}, {f_vals.max():.4f}], "
              f"accumulated SNR={snr_val:.1f}")

    # Plot
    N = args.N
    t_samples = np.arange(N)
    lt = linked.data[0].cpu()
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
    tok = tokens.data[0].cpu()
    t_mid = (tok[:, 1] + tok[:, 2]) / 2
    f_mid = (tok[:, 3] + tok[:, 4]) / 2
    snr_vals = tok[:, 0]
    mask = snr_vals > 0
    sc = ax.scatter(t_mid[mask], f_mid[mask], c=snr_vals[mask], s=3,
                    cmap="inferno", vmin=0)
    ax.set_ylabel("f (cycles/sample)")
    ax.set_title("Chirp tokens (color = peak amplitude)")
    fig.colorbar(sc, ax=ax, label="peak amplitude")

    # Panel 4: Linked chains in time-frequency
    ax = axes[3]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_chains, 1)))
    ci = 0
    for cid in unique_chains:
        if cid < 0:
            continue
        mask = chain_ids == cid
        idxs = torch.where(mask)[0]
        # Sort by t_start to get temporal order
        order = lt[idxs, 1].argsort()
        idxs = idxs[order]
        t_mid = ((lt[idxs, 1] + lt[idxs, 2]) / 2).numpy()
        f_mid = ((lt[idxs, 3] + lt[idxs, 4]) / 2).numpy()
        ax.plot(t_mid, f_mid, '-o', color=colors[ci % len(colors)],
                ms=2, lw=1.2, label=f"chain {int(cid)}")
        ci += 1
    ax.plot(t_samples, f_true1, '--', color='gray', lw=0.5, alpha=0.7, label='true f1')
    ax.plot(t_samples, f_true2, '--', color='silver', lw=0.5, alpha=0.7, label='true f2')
    ax.set_ylabel("f (cycles/sample)")
    ax.set_title("Linked chains")
    ax.legend(fontsize=7, ncol=4)

    # Panel 5: Accumulated phase (detrended)
    ax = axes[4]
    ci = 0
    for cid in unique_chains:
        if cid < 0:
            continue
        mask = chain_ids == cid
        idxs = torch.where(mask)[0]
        order = lt[idxs, 1].argsort()
        idxs = idxs[order]

        phi_accum = [lt[idxs[0], 7].item()]
        for j in range(len(idxs)):
            ps_j = lt[idxs[j], 7].item()
            pe_j = lt[idxs[j], 8].item()
            phi_accum.append(phi_accum[-1] + (pe_j - ps_j))
        phi_accum = np.array(phi_accum)

        t_anchors = [lt[idxs[0], 1].item()]
        for j in range(len(idxs)):
            t_anchors.append(lt[idxs[j], 2].item())
        t_anchors = np.array(t_anchors)

        if len(t_anchors) > 1:
            slope = (phi_accum[-1] - phi_accum[0]) / (t_anchors[-1] - t_anchors[0])
            phi_detrend = phi_accum - slope * (t_anchors - t_anchors[0]) - phi_accum[0]
        else:
            phi_detrend = phi_accum - phi_accum[0]
        ax.plot(t_anchors, phi_detrend, '-', color=colors[ci % len(colors)],
                lw=1, label=f"chain {int(cid)}")
        ci += 1
    ax.set_ylabel("φ − linear trend (rad)")
    ax.set_xlabel("sample index")
    ax.set_title("Coherent phase (detrended)")
    ax.legend(fontsize=7, ncol=4)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
