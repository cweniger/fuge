"""Fisher information analysis: Cramér-Rao bound for f0 estimation.

Computes the theoretical minimum error for f0 using JAX autodiff,
and compares with the transformer's empirical performance.
"""

import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from chirp import _chirp_impl

# ── Signal parameters (matching transformer_demo.py) ────────────────
T_C = 1e6
A0 = 5.0
N_HARMONICS = 4
CHIRP_MASS = 1.0
HARMONIC_DECAY = 1.5
N = 100_000
T_OBS = 0.9 * T_C
SIGMA = 1.0

F0_CENTER = 2.75e-3
F0_HALF = 1.5e-8  # 30 nHz half-width (matching transformer_demo.py)
F0_MIN, F0_MAX = F0_CENTER - F0_HALF, F0_CENTER + F0_HALF


# =====================================================================
# Fisher information for f0 (scalar)
# =====================================================================

@jax.jit
def fisher_f0(f0):
    """Compute Fisher information for f0 (scalar, all other params fixed).

    F = (1/σ²) Σ_n (∂h_n/∂f0)²

    Returns F (scalar) and σ_CRB = 1/√F.
    """
    def signal(f0_):
        return _chirp_impl(f0_, CHIRP_MASS, T_C, A0, HARMONIC_DECAY,
                          N_HARMONICS, N, T_OBS)

    # Forward-mode AD: get gradient ∂h/∂f0 as a (N,) vector
    dh_df0 = jax.jacfwd(signal)(f0)  # (N,)

    # Fisher information = (1/σ²) * ||∂h/∂f0||²
    F = jnp.sum(dh_df0 ** 2) / SIGMA ** 2
    sigma_crb = 1.0 / jnp.sqrt(F)
    return F, sigma_crb


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    # --- Reconstruct validation f0 values from transformer_demo.py ---
    rng = np.random.default_rng(42)
    all_f0 = rng.uniform(F0_MIN, F0_MAX, size=5500)
    val_f0 = all_f0[5000:]  # 500 validation points

    # --- Compute CRB at each validation point ---
    print(f"Computing Fisher information at {len(val_f0)} f0 values...")
    t0 = time.time()

    crb_sigma = np.zeros(len(val_f0))
    crb_rel = np.zeros(len(val_f0))
    fisher_vals = np.zeros(len(val_f0))

    for i, f0 in enumerate(val_f0):
        F, sig = fisher_f0(jnp.float64(f0))
        fisher_vals[i] = float(F)
        crb_sigma[i] = float(sig)
        crb_rel[i] = float(sig) / f0
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(val_f0)}")

    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Cramér-Rao bound for f0 (single parameter, others fixed)")
    print("=" * 60)
    print(f"  f0 range: [{F0_MIN:.4e}, {F0_MAX:.4e}] Hz")
    print(f"  Prior width: {(F0_MAX - F0_MIN):.2e} Hz")
    print(f"  Noise σ: {SIGMA}")
    print(f"\n  Fisher information:    {np.median(fisher_vals):.4e}")
    print(f"  CRB σ(f0):             {np.median(crb_sigma):.4e} Hz")
    print(f"  CRB relative error:    {np.median(crb_rel):.6%}")

    # --- Compare with transformer ---
    # From latest transformer_demo.py run (30 nHz prior, k=1024, float64)
    transformer_median_abs = 2.58e-10  # Hz
    transformer_eff_sigma = transformer_median_abs / 0.6745  # Gaussian median|err| = 0.6745*σ

    print(f"\n  Transformer median |err|: {transformer_median_abs:.2e} Hz")
    print(f"  Transformer eff. σ:       {transformer_eff_sigma:.2e} Hz")
    print(f"  Ratio (transformer σ / CRB σ): {transformer_eff_sigma / np.median(crb_sigma):.1f}x")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # CRB σ vs f0
    ax = axes[0]
    ax.scatter(val_f0 * 1e3, crb_sigma, s=3, alpha=0.5)
    ax.set_xlabel("$f_0$ (mHz)")
    ax.set_ylabel("CRB $\\sigma(f_0)$ (Hz)")
    ax.set_title("CRB absolute error vs $f_0$")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    # CRB relative error distribution
    ax = axes[1]
    ax.hist(crb_rel * 100, bins=40, alpha=0.7, color="steelblue",
            label="CRB")
    crb_med = np.median(crb_rel) * 100
    ax.axvline(crb_med, color="steelblue", ls="--", lw=2,
               label=f"CRB median: {crb_med:.2e}%")
    transformer_median_rel = transformer_eff_sigma / np.median(val_f0)
    ax.axvline(transformer_median_rel * 100, color="orangered", ls="-", lw=2,
               label=f"Transformer: {transformer_median_rel * 100:.6f}%")
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Count")
    ax.set_title("CRB vs Transformer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Fisher information vs f0
    ax = axes[2]
    ax.scatter(val_f0 * 1e3, fisher_vals, s=3, alpha=0.5)
    ax.set_xlabel("$f_0$ (mHz)")
    ax.set_ylabel("Fisher information $F(f_0)$")
    ax.set_title("Fisher information vs $f_0$")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cramér-Rao bound for $f_0$  (σ_noise={SIGMA}, N={N:,})",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("fisher_demo.png", dpi=150)
    print("\nSaved fisher_demo.png")
