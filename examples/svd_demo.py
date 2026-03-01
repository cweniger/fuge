"""
Streaming PCA demo: prior shrinkage and Wiener filter verification.

Figure 1: Component stability — prior shrinkage from non-linear to linear
regime. Left = raw eigenbasis (sign flips), right = Procrustes-stabilized.

Figure 2: Wiener filter verification — known signal structure with
controlled eigenvalues. Compares measured output variance and MSE against
analytical predictions:
    - Output variance = λ/(λ+1)
    - MSE vs true signal = 1/(λ+1)

Usage:
    python svd_demo.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from fuge.svd import StreamingPCA


def generate_signals(theta, t):
    """Non-linear signal: A * sin(2π f t + φ)."""
    A = theta[:, 0:1]
    f = theta[:, 1:2]
    phi = theta[:, 2:3]
    return A * torch.sin(2 * np.pi * f * t.unsqueeze(0) + phi)


# ── Figure 1: Procrustes stability ──────────────────────────────────────

def figure_procrustes():
    torch.manual_seed(42)

    D = 400
    k = 5
    batch_size = 128
    buffer_size = 256
    n_rounds = 40
    noise_sigma = 0.3

    t = torch.linspace(0, 2, D, dtype=torch.float64)
    A0, f0, phi0 = 3.0, 5.0, 1.0
    width_start = torch.tensor([2.0, 3.0, np.pi], dtype=torch.float64)
    width_end = torch.tensor([0.05, 0.05, 0.02], dtype=torch.float64)

    pca = StreamingPCA(n_components=k, buffer_size=buffer_size, momentum=0.15)
    history = []

    for r in range(n_rounds):
        frac = r / max(n_rounds - 1, 1)
        width = width_start * (width_end / width_start) ** frac
        theta_center = torch.tensor([[A0, f0, phi0]], dtype=torch.float64)
        theta = theta_center + width.unsqueeze(0) * torch.randn(
            batch_size, 3, dtype=torch.float64)
        signals = generate_signals(theta, t)
        x = signals + noise_sigma * torch.randn_like(signals)
        pca.update(x / noise_sigma)

        if pca.components is not None:
            raw = pca.components.clone().detach()
            stable = (pca._R @ pca.components).clone().detach()
            history.append((r, raw, stable))

    fig, axes = plt.subplots(k, 2, figsize=(14, 2.2 * k), sharex=True)
    cmap = plt.cm.RdYlBu_r
    n_traces = len(history)

    for comp_idx in range(k):
        for col, (label, key) in enumerate([
            ("Raw eigenbasis (no Procrustes)", 1),
            ("Procrustes-stabilized", 2),
        ]):
            ax = axes[comp_idx, col]
            for trace_idx, entry in enumerate(history):
                vecs = entry[key]
                color = cmap(trace_idx / max(n_traces - 1, 1))
                alpha = 0.3 + 0.7 * (trace_idx / max(n_traces - 1, 1))
                ax.plot(t.numpy(), vecs[comp_idx].numpy(),
                        color=color, alpha=alpha, linewidth=0.8)
            if col == 0:
                ax.set_ylabel(f"PC {comp_idx + 1}")
            ax.set_xlim(t[0].item(), t[-1].item())
            if comp_idx == 0:
                ax.set_title(label, fontsize=11)
            if col == 1 and pca.eigenvalues is not None:
                ev = pca.eigenvalues[comp_idx].item()
                ax.text(0.98, 0.92, f"\u03bb = {ev:.1f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.2",
                                              facecolor="white", alpha=0.8))

    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_rounds - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, aspect=40)
    cbar.set_label("Update step")
    fig.subplots_adjust(hspace=0.3, wspace=0.15, right=0.88)
    plt.savefig("svd_demo_procrustes.png", dpi=150, bbox_inches="tight")
    print("Saved svd_demo_procrustes.png")


# ── Figure 2: Wiener filter verification ────────────────────────────────

def figure_wiener():
    """Verify Wiener filter against analytical predictions.

    Setup: known orthogonal signal directions with controlled eigenvalues.
    Data = signal + unit white noise. PCA trained to convergence, then
    forward() applied to fresh test data.

    Analytical predictions (per component with eigenvalue λ):
        forward() does: c = (x @ V.T) * λ/(λ+1) / √λ  then rotates by R.

        If x = s + n with Var(s along v_i) = λ_i, Var(n along v_i) = 1:
        - Output variance:  Var[output_i] = λ/(λ+1)
        - MSE vs truth:     E[(output_i - s_i/√λ)²] = 1/(λ+1)

        where s_i/√λ is the unit-variance "true" signal coefficient.
    """
    torch.manual_seed(123)

    D = 300
    k = 8
    n_train = 5000
    n_test = 10000

    # Eigenvalues spanning 3 orders of magnitude
    eigenvalues_true = torch.tensor(
        [200.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.3],
        dtype=torch.float64,
    )

    # Random orthogonal signal directions
    V_true = torch.linalg.qr(torch.randn(D, k, dtype=torch.float64))[0].T[:k]

    # --- Train PCA to convergence ---
    pca = StreamingPCA(n_components=k, buffer_size=256, momentum=0.05,
                       shrinkage=True)

    for _ in range(n_train // 256 + 1):
        signal_coeffs = torch.randn(256, k, dtype=torch.float64) * torch.sqrt(eigenvalues_true)
        signal = signal_coeffs @ V_true
        noise = torch.randn(256, D, dtype=torch.float64)
        pca.update(signal + noise)

    # --- Test: generate data with known signal/noise split ---
    signal_coeffs_test = torch.randn(n_test, k, dtype=torch.float64) * torch.sqrt(eigenvalues_true)
    signal_test = signal_coeffs_test @ V_true
    noise_test = torch.randn(n_test, D, dtype=torch.float64)
    x_test = signal_test + noise_test

    # forward() output
    with torch.no_grad():
        output = pca(x_test)

    # True signal in the normalized eigenbasis, rotated to stable frame
    # forward() normalizes by /√λ and rotates by R, so the "truth" is:
    #   truth = (signal @ V.T) / √λ  @ R.T
    # which equals signal_coeffs / √λ_true projected through R,
    # but V_true ≠ pca.components exactly. Use pca's own components:
    true_coeffs_eigenbasis = signal_test @ pca.components.T  # (n_test, k)
    Λ = pca.eigenvalues.clamp(min=1e-12)
    true_normalized = (true_coeffs_eigenbasis / torch.sqrt(Λ).unsqueeze(0)) @ pca._R.T

    # --- Measured statistics ---
    measured_var = output.var(dim=0).numpy()
    measured_mse = ((output - true_normalized) ** 2).mean(dim=0).numpy()

    # --- Analytical predictions (using learned eigenvalues) ---
    Λ_np = pca.eigenvalues.numpy()
    predicted_var = Λ_np / (Λ_np + 1)
    predicted_mse = 1.0 / (Λ_np + 1)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x_pos = np.arange(k)
    bar_w = 0.35

    # Variance
    ax1.bar(x_pos - bar_w / 2, measured_var, bar_w, label="Measured", color="steelblue")
    ax1.bar(x_pos + bar_w / 2, predicted_var, bar_w, label="Analytical: \u03bb/(\u03bb+1)",
            color="coral", alpha=0.8)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Variance")
    ax1.set_title("Output variance of forward()")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{i+1}\n(\u03bb={Λ_np[i]:.1f})" for i in range(k)], fontsize=8)
    ax1.legend()
    ax1.set_ylim(0, 1.15)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.3)

    # MSE
    ax2.bar(x_pos - bar_w / 2, measured_mse, bar_w, label="Measured", color="steelblue")
    ax2.bar(x_pos + bar_w / 2, predicted_mse, bar_w, label="Analytical: 1/(\u03bb+1)",
            color="coral", alpha=0.8)
    ax2.set_xlabel("Component")
    ax2.set_ylabel("MSE")
    ax2.set_title("MSE vs true signal (normalized)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{i+1}\n(\u03bb={Λ_np[i]:.1f})" for i in range(k)], fontsize=8)
    ax2.legend()

    fig.suptitle("Wiener filter verification: measured vs analytical", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("svd_demo_wiener.png", dpi=150, bbox_inches="tight")
    print("Saved svd_demo_wiener.png")

    # Print numerical comparison
    print("\n  Component  |  λ (learned)  |  Var measured/predicted  |  MSE measured/predicted")
    print("  " + "-" * 75)
    for i in range(k):
        print(f"  PC {i+1:>2}      |  {Λ_np[i]:>10.1f}  |"
              f"  {measured_var[i]:.4f} / {predicted_var[i]:.4f}    |"
              f"  {measured_mse[i]:.4f} / {predicted_mse[i]:.4f}")


def main():
    figure_procrustes()
    figure_wiener()
    plt.show()


if __name__ == "__main__":
    main()
