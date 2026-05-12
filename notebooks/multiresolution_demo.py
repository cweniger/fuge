"""Multi-resolution token reconstruction demo.

Generates a merger signal with noise, tokenizes at multiple window sizes
(64, 128, ..., 1024), greedily selects the best-SNR tokens across
resolutions, and reconstructs from the multi-resolution token set.

The greedy algorithm:
  1. Start from the finest resolution (k_min) — its tokens tile the full
     time axis via hop = k_min/2.
  2. For each coarser resolution k (in ascending order), each token covers
     k/k_min fine-grid slots.  If the coarse token's SNR exceeds the
     combined SNR of the sub-tokens it would replace (power addition:
     sqrt(sum(snr_i^2))), the coarse token wins and replaces them.
  3. The result is a multi-resolution tiling: short windows near
     coalescence (fast changes), long windows in the quiet inspiral
     (better frequency resolution).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import jax
jax.config.update("jax_enable_x64", True)

from fuge.spectral import ChirpTokenizer

# ── Signal parameters ────────────────────────────────────────────────
N = 8192
T_C_FRAC = 0.75
T_C = int(T_C_FRAC * N)

F0 = 0.001
F_RD = 0.01
CHIRP_RATE = (F_RD - F0) / T_C

A0 = 0.1
AMP_EXPONENT = -1.00
A_PEAK = 10.0

TAU_RD = 120.0

F_MAX = 2 * F_RD         # maximum frequency for peak selection

NOISE_SIGMA = 1.0

# ── Tokenizer parameters ─────────────────────────────────────────────
K_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
N_PEAKS = 1
N_DLNF = 51
DLNF_MIN = 0.0
DLNF_MAX = 0.3

SEED = 42


def make_merger_signal(N, t_c, f0, chirp_rate, f_rd, A0, amp_exp, A_peak, tau_rd):
    """Generate a merger-like signal: chirp inspiral + exponential ringdown."""
    t = np.arange(N, dtype=np.float64)
    signal = np.zeros(N)

    mask_insp = t < t_c
    t_insp = t[mask_insp]
    phi_insp = 2.0 * np.pi * (f0 * t_insp + 0.5 * chirp_rate * t_insp**2)
    tau = np.maximum(1.0 - t_insp / t_c, 1e-6)
    A_insp = np.minimum(A0 * tau ** amp_exp, A_peak)
    signal[mask_insp] = A_insp * np.cos(phi_insp)

    phi_c = 2.0 * np.pi * (f0 * t_c + 0.5 * chirp_rate * t_c**2)
    mask_rd = t >= t_c
    t_rd = t[mask_rd] - t_c
    phi_rd = phi_c + 2.0 * np.pi * f_rd * t_rd
    A_rd = A_peak * np.exp(-t_rd / tau_rd)
    signal[mask_rd] = A_rd * np.cos(phi_rd)

    return signal


def tokenize_multi(signal, k_values, n_peaks, n_dlnf, dlnf_min, dlnf_max,
                    f_max=None):
    """Tokenize at multiple window sizes. Returns dict k -> (W, K, 9).

    If f_max is given (in cycles/sample), tokens with center frequency
    above f_max have their SNR zeroed out so they are ignored.
    """
    x = torch.from_numpy(signal).unsqueeze(0)
    tokens_by_k = {}
    for k in k_values:
        tokenizer = ChirpTokenizer(
            k=k, n_peaks=n_peaks, n_dlnf=n_dlnf,
            dlnf_min=dlnf_min, dlnf_max=dlnf_max,
        ).double()
        tok = tokenizer(x).data[0].numpy()  # (N, 9)

        if f_max is not None:
            # f_start, f_end are in [-1, 1] mapping to [0, Fk-1] bins
            # Convert f_max to normalized token units
            Fk = k // 2 + 1
            f_max_bin = f_max * k  # cycles/sample -> bin index
            f_max_norm = 2.0 * f_max_bin / (Fk - 1) - 1.0
            # Zero SNR for peaks whose center frequency exceeds f_max
            f_center = (tok[:, 3] + tok[:, 4]) / 2
            above = f_center > f_max_norm
            tok[above, 0] = 0.0
            n_killed = np.sum(above)
            n_total = tok.shape[0]
            print(f"  k={k:4d}: {n_total} tokens "
                  f"({n_killed} above f_max={f_max:.4f})")
        else:
            print(f"  k={k:4d}: {tok.shape[0]} tokens")

        tokens_by_k[k] = tok
    return tokens_by_k


def greedy_select(tokens_by_k, k_values):
    """Greedy multi-resolution token selection.

    Returns a list of (k, window_idx, token_data) for each unique selected
    token, plus the assignment array mapping fine slots -> token index.
    """
    k_min = min(k_values)
    hop_min = k_min // 2

    # Group flat tokens by window (using t_start), pick best peak per window.
    def _best_per_window(tok):
        """Return (W, 9) array: best-SNR token per window from flat (N, 9)."""
        unique_t = np.unique(tok[:, 1])
        best = []
        for t_s in unique_t:
            group = tok[tok[:, 1] == t_s]
            best.append(group[np.argmax(group[:, 0])])
        return np.array(best)

    tok_fine = _best_per_window(tokens_by_k[k_min])  # (W_fine, 9)
    W_fine = tok_fine.shape[0]

    # Per fine slot: track (k, window_idx, snr, token_vector)
    slot_k = np.full(W_fine, k_min, dtype=int)
    slot_w = np.arange(W_fine, dtype=int)
    slot_snr = tok_fine[:, 0].copy()
    slot_token = tok_fine.copy()

    # Upgrade through coarser resolutions
    for k in sorted(k_values):
        if k == k_min:
            continue

        tok = _best_per_window(tokens_by_k[k])  # (W_coarse, 9)
        ratio = k // k_min  # fine slots per coarse window

        for w in range(tok.shape[0]):
            coarse_snr = tok[w, 0]

            fine_start = w * ratio
            fine_end = fine_start + ratio
            if fine_end > W_fine:
                continue

            # Combined SNR of current sub-tokens (power addition)
            combined_snr = np.sqrt(np.sum(slot_snr[fine_start:fine_end] ** 2))

            if coarse_snr > combined_snr:
                slot_k[fine_start:fine_end] = k
                slot_w[fine_start:fine_end] = w
                slot_snr[fine_start:fine_end] = coarse_snr
                slot_token[fine_start:fine_end] = tok[w, :]

    return slot_k, slot_w, slot_snr, slot_token, W_fine


def _synthesize_window(tok, k):
    """Synthesize a full k-sample waveform from a single token."""
    Fk = k // 2 + 1
    n = np.arange(k)
    frac = (n - k / 4) / (k / 2)

    if tok[0] <= 0:  # snr
        return np.zeros(k)

    f_start_bin = (tok[3] + 1.0) / 2.0 * (Fk - 1)
    f_end_bin = (tok[4] + 1.0) / 2.0 * (Fk - 1)
    A_s, A_e = tok[5], tok[6]
    ps, pe = tok[7], tok[8]

    A_n = A_s + (A_e - A_s) * frac

    dphi = pe - ps
    f_center_bin = (f_start_bin + f_end_bin) / 2
    expected_dphi = np.pi * f_center_bin
    n_wraps = np.round((expected_dphi - dphi) / (2 * np.pi))
    dphi_unwrapped = dphi + n_wraps * 2 * np.pi

    df = f_end_bin - f_start_bin
    phi_n = (ps + dphi_unwrapped * frac
             + (np.pi / 2) * df * frac * (frac - 1))

    return A_n * np.cos(phi_n)


def _reconstruct_single_k(tokens_k, k, N_signal):
    """Standard COLA overlap-add reconstruction for a single window size."""
    window = torch.hann_window(k, dtype=torch.float64).numpy()
    signal = np.zeros(N_signal)
    norm = np.zeros(N_signal)

    # Group flat tokens by window, pick best SNR per window
    unique_t = np.unique(tokens_k[:, 1])
    for t_s in unique_t:
        start = int(t_s - k / 4)
        end = start + k
        if end > N_signal:
            break
        group = tokens_k[tokens_k[:, 1] == t_s]
        best = group[np.argmax(group[:, 0])]
        wave = _synthesize_window(best, k)
        signal[start:end] += wave * window
        norm[start:end] += window

    mask = norm > 1e-12
    signal[mask] /= norm[mask]
    return signal


def reconstruct_multiresolution(slot_k, slot_w, slot_token, k_min, W_fine,
                                N_signal, tokens_by_k, k_values):
    """Reconstruct by blending per-resolution COLA reconstructions.

    1. Reconstruct each resolution independently via standard COLA.
    2. For each fine time slot, use the signal from the selected resolution.
    3. Cross-fade at resolution boundaries.
    """
    hop_min = k_min // 2

    # Full COLA reconstruction per k
    recon_by_k = {}
    for k in k_values:
        recon_by_k[k] = _reconstruct_single_k(tokens_by_k[k], k, N_signal)

    # Build output: pick from selected resolution per fine slot
    signal = np.zeros(N_signal)
    for i in range(W_fine):
        k = int(slot_k[i])
        out_start = i * hop_min
        out_end = min(out_start + hop_min, N_signal)
        signal[out_start:out_end] = recon_by_k[k][out_start:out_end]

    # Cross-fade at resolution boundaries
    for i in range(1, W_fine):
        if slot_k[i] == slot_k[i - 1]:
            continue
        boundary = i * hop_min
        fade_len = hop_min  # cross-fade over one fine hop
        b_start = max(boundary - fade_len // 2, 0)
        b_end = min(boundary + fade_len // 2, N_signal)
        n = b_end - b_start
        if n < 2:
            continue
        fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
        fade_out = 1 - fade_in
        k_left = int(slot_k[i - 1])
        k_right = int(slot_k[i])
        signal[b_start:b_end] = (fade_out * recon_by_k[k_left][b_start:b_end]
                                 + fade_in * recon_by_k[k_right][b_start:b_end])

    return signal


def main():
    rng = np.random.default_rng(SEED)

    # Generate signal
    print("Generating merger signal...")
    signal_clean = make_merger_signal(
        N, T_C, F0, CHIRP_RATE, F_RD, A0, AMP_EXPONENT, A_PEAK, TAU_RD,
    )
    noise = rng.standard_normal(N) * NOISE_SIGMA
    signal = signal_clean + noise

    # Tokenize at multiple resolutions
    print("Tokenizing at multiple resolutions...")
    tokens_by_k = tokenize_multi(
        signal, K_VALUES, N_PEAKS, N_DLNF, DLNF_MIN, DLNF_MAX,
        f_max=F_MAX,
    )

    # Greedy selection
    print("Greedy multi-resolution selection...")
    k_min = min(K_VALUES)
    slot_k, slot_w, slot_snr, slot_token, W_fine = greedy_select(
        tokens_by_k, K_VALUES,
    )

    # Report selection statistics
    for k in K_VALUES:
        count = np.sum(slot_k == k)
        print(f"  k={k:4d}: {count:3d} fine slots ({count * 100 / W_fine:.0f}%)")

    # Reconstruct
    print("Reconstructing from multi-resolution tokens...")
    recon = reconstruct_multiresolution(
        slot_k, slot_w, slot_token, k_min, W_fine, N,
        tokens_by_k, K_VALUES,
    )

    # Also reconstruct single-resolution for comparison (finest k)
    recon_fine = _reconstruct_single_k(tokens_by_k[k_min], k_min, N)

    # SNR
    margin = max(K_VALUES)
    sl = slice(margin, N - margin)

    residual = signal_clean - recon
    rms_s = np.sqrt(np.mean(signal_clean[sl] ** 2))
    rms_r = np.sqrt(np.mean(residual[sl] ** 2))
    snr_multi = 20 * np.log10(rms_s / rms_r) if rms_r > 0 else np.inf

    residual_fine = signal_clean - recon_fine
    rms_r_fine = np.sqrt(np.mean(residual_fine[sl] ** 2))
    snr_fine = 20 * np.log10(rms_s / rms_r_fine) if rms_r_fine > 0 else np.inf

    print(f"  Multi-resolution SNR: {snr_multi:.1f} dB")
    print(f"  Fine-only (k={k_min}) SNR: {snr_fine:.1f} dB")

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.5, 1])

    fig.suptitle(
        f"Multi-resolution reconstruction  "
        f"(k={K_VALUES}, noise_sigma={NOISE_SIGMA})\n"
        f"Multi-res SNR={snr_multi:.1f} dB vs "
        f"fine-only (k={k_min}) SNR={snr_fine:.1f} dB",
        fontsize=13)

    t = np.arange(N)

    # Row 0: full signal comparison (wide panel)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t, signal, color='0.75', lw=0.3, alpha=0.5, label='Noisy input')
    ax_top.plot(t, signal_clean, 'b-', lw=0.5, alpha=0.7, label='Original (clean)')
    ax_top.plot(t, recon, 'r-', lw=0.5, alpha=0.7, label='Multi-res reconstruction')
    ax_top.axvline(T_C, color='k', ls=':', lw=1, alpha=0.5, label=f'Coalescence')
    ymax = np.max(np.abs(signal_clean)) * 1.2
    ax_top.set_ylim(-ymax, ymax)
    ax_top.set_xlabel("Sample")
    ax_top.set_ylabel("Amplitude")
    ax_top.set_title("Signal and multi-resolution reconstruction")
    ax_top.legend(fontsize=9, loc='upper left')
    ax_top.grid(True, alpha=0.3)

    # Row 1: resolution map (wide panel)
    ax_res = fig.add_subplot(gs[1, :])
    hop_min = k_min // 2
    # Color-code by selected k
    k_colors = {k: plt.cm.viridis(i / (len(K_VALUES) - 1))
                for i, k in enumerate(K_VALUES)}
    for i in range(W_fine):
        k = slot_k[i]
        x0 = i * hop_min
        ax_res.barh(0, hop_min, left=x0, height=1,
                    color=k_colors[k], edgecolor='none')
    ax_res.axvline(T_C, color='k', ls=':', lw=1, alpha=0.5)
    ax_res.set_xlim(0, N)
    ax_res.set_yticks([])
    ax_res.set_xlabel("Sample")
    ax_res.set_title("Selected window size per time slot")
    patches = [mpatches.Patch(color=k_colors[k], label=f'k={k}')
               for k in K_VALUES]
    ax_res.legend(handles=patches, fontsize=8, loc='upper right', ncol=len(K_VALUES))

    # Row 2 left: residual comparison
    ax_bl = fig.add_subplot(gs[2, 0])
    ax_bl.plot(t, residual_fine, 'c-', lw=0.3, alpha=0.5,
               label=f'Fine-only (k={k_min})')
    ax_bl.plot(t, residual, 'g-', lw=0.3, alpha=0.7,
               label='Multi-resolution')
    ax_bl.axvline(T_C, color='k', ls=':', lw=1, alpha=0.5)
    ax_bl.set_xlabel("Sample")
    ax_bl.set_ylabel("Residual")
    ax_bl.set_title("Residual (original - reconstructed)")
    ax_bl.legend(fontsize=8)
    ax_bl.grid(True, alpha=0.3)

    # Row 2 right: SNR per fine slot
    ax_br = fig.add_subplot(gs[2, 1])
    t_slots = np.arange(W_fine) * hop_min + hop_min / 2
    ax_br.plot(t_slots, slot_snr, 'k-', lw=0.5, alpha=0.7)
    # Color the background by selected k
    for i in range(W_fine):
        k = slot_k[i]
        x0 = i * hop_min
        ax_br.axvspan(x0, x0 + hop_min, alpha=0.15, color=k_colors[k],
                      edgecolor='none')
    ax_br.axvline(T_C, color='k', ls=':', lw=1, alpha=0.5)
    ax_br.set_xlabel("Sample")
    ax_br.set_ylabel("Token SNR")
    ax_br.set_title("Selected token SNR (background = window size)")
    ax_br.grid(True, alpha=0.3)

    out = "multiresolution_demo.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
