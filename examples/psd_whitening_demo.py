"""PSD whitening demo: colored noise + Fisher CRB + NN estimation.

Generates EMRI signals with colored noise (mild 1/f^2 spectrum), computes
the Cramér-Rao bound using the noise-weighted inner product, and trains
three transformer models to compare:
  1. White noise (sigma=1) — baseline
  2. Colored noise, no whitening — shows degradation
  3. Colored noise + PSD whitening — should recover near-CRB performance
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from fuge import ToneTokenizer, ToneTokenEmbedding, TransformerEmbedding

# ── Signal parameters ────────────────────────────────────────────────
N = 100_000
T_C = 1e6
T_OBS = 0.9 * T_C
A0 = 5.0
N_HARMONICS = 4
SEED = 42

# ── Dataset sizes ────────────────────────────────────────────────────
N_TRAIN = 5000
N_VAL = 500

# ── Spectral decomposition ──────────────────────────────────────────
K_WINDOW = 1024
N_PEAKS = 3
N_DLNF = 11
DLNF_MIN = 0.0
DLNF_MAX = 0.05

# ── Predicted parameters (prior = ~10x CRB width) ───────────────────
PARAM_NAMES = ["f0", "chirp_mass", "harmonic_decay"]
PARAM_CENTERS = np.array([2.75e-3, 1.0, 1.5])
# Prior half-widths: ~10x colored CRB so NN can learn in all cases
PARAM_HALF = np.array([2.5e-8, 3.5e-5, 1.0e-1])
PARAM_MIN = PARAM_CENTERS - PARAM_HALF
PARAM_MAX = PARAM_CENTERS + PARAM_HALF
N_PARAMS = len(PARAM_NAMES)

# ── Transformer architecture ────────────────────────────────────────
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
DROPOUT = 0.1
PHASE_MODE = "center"

# ── Training ─────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 1e-3
N_EPOCHS = 120
TOKENIZE_BATCH = 64


# =====================================================================
# Colored noise PSD
# =====================================================================

def colored_psd(f, f_bump=3e-3, bump_width=0.3, bump_height=30.0):
    """Colored noise PSD with a bump near the signal frequency.

        S_n(f) = S_white * (1 + A * exp(-0.5 * ((log10 f - log10 f_bump) / w)^2))

    where S_white = 2*T_OBS/N is the white-noise PSD for sigma=1.
    The bump adds ~30x noise at f_bump relative to the flat floor,
    concentrated within ~1 decade of f_bump.  This mimics confusion
    noise from unresolved galactic binaries in LISA.

    At f_bump=3 mHz (near our signal f0=2.75 mHz), this selectively
    degrades estimation at the signal frequency while leaving high-f
    bins relatively clean.  Total noise std ≈ 2.1, CRB degradation ~4x.

    Parameters
    ----------
    f : array-like
        Frequency array (Hz).  f=0 returns 0.
    f_bump : float
        Center frequency of the noise bump.
    bump_width : float
        Width of bump in decades (log10 space).
    bump_height : float
        Peak height of bump relative to flat floor.

    Returns
    -------
    S_n : array, same shape as f.
    """
    f = np.asarray(f, dtype=np.float64)
    S_white = 2.0 * T_OBS / N  # flat PSD for sigma=1
    S_n = np.zeros_like(f)
    mask = f > 0
    log_ratio = (np.log10(f[mask]) - np.log10(f_bump)) / bump_width
    S_n[mask] = S_white * (1.0 + bump_height * np.exp(-0.5 * log_ratio ** 2))
    return S_n


def generate_colored_noise(n_signals, N, T_obs, psd_func, rng):
    """Generate colored Gaussian noise with a given PSD.

    Generates noise in frequency domain as sqrt(S_n) * complex_normal,
    then IFFTs to time domain.
    """
    freqs = np.fft.rfftfreq(N, d=T_obs / N)  # (N//2+1,)
    S_n = psd_func(freqs)
    S_n[0] = 0.0  # no DC

    # rfft amplitude so that time-domain variance = ∫ S_n(f) df:
    # E[|x|^2] = (1/N^2) * 4 * Σ A^2 = Σ S_n * df  =>  A = (N/2)*sqrt(S_n/T_obs)
    amp = (N / 2.0) * np.sqrt(S_n / T_obs)

    Nf = len(freqs)
    noise = np.zeros((n_signals, N))
    for i in range(n_signals):
        z = rng.standard_normal(Nf) + 1j * rng.standard_normal(Nf)
        z[0] = z[0].real
        if N % 2 == 0:
            z[-1] = z[-1].real
        noise[i] = np.fft.irfft(amp * z, n=N)

    return noise


# =====================================================================
# Signal generation (pure numpy, no JAX dependency)
# =====================================================================

def _emri_numpy(f0, chirp_mass, t_c, A0, harmonic_decay, n_harmonics, N, T_obs):
    """Generate EMRI signal in numpy (matching fuge.emri but without JAX)."""
    t = np.linspace(0, T_obs, N)
    dt = T_obs / (N - 1)
    tau = 1.0 - t / t_c
    f_t = f0 * tau ** (-3.0 / 8.0 * chirp_mass)
    A_t = A0 * tau ** (-0.25)
    trap = (f_t[:-1] + f_t[1:]) * 0.5 * dt
    phase = np.concatenate([[0.0], np.cumsum(trap)]) * 2.0 * np.pi
    h = np.zeros(N)
    for k in range(1, n_harmonics + 1):
        h += A_t * np.exp(-harmonic_decay * (k - 1)) * np.cos(k * phase)
    return h


# =====================================================================
# Fisher information (finite-difference Jacobian, pure numpy)
# =====================================================================

def _jacobian(params, N, T_obs, n_harmonics):
    """Compute (N, 3) Jacobian via central finite differences."""
    f0, cm, hd = params["f0"], params["chirp_mass"], params["harmonic_decay"]
    p_list = [f0, cm, hd]
    eps = [1e-12, 1e-8, 1e-6]  # tuned per parameter scale

    h0 = _emri_numpy(f0, cm, T_C, A0, hd, n_harmonics, N, T_obs)
    jac = np.zeros((N, 3))
    for i, e in enumerate(eps):
        args_p = list(p_list); args_p[i] += e
        args_m = list(p_list); args_m[i] -= e
        h_p = _emri_numpy(args_p[0], args_p[1], T_C, A0, args_p[2],
                          n_harmonics, N, T_obs)
        h_m = _emri_numpy(args_m[0], args_m[1], T_C, A0, args_m[2],
                          n_harmonics, N, T_obs)
        jac[:, i] = (h_p - h_m) / (2.0 * e)
    return jac


def fisher_matrix_colored(params, psd_func, N, T_obs, n_harmonics):
    """3x3 Fisher matrix with noise-weighted inner product.

    F_ij = 4*(T_obs/N^2) * Σ_k c_k * Re[dH_i*dH_j*] / S_n
    """
    jac = _jacobian(params, N, T_obs, n_harmonics)

    freqs = np.fft.rfftfreq(N, d=T_obs / N)
    S_n = psd_func(freqs)
    S_n[0] = np.inf

    weight = np.where(S_n > 0, 1.0 / S_n, 0.0)
    dH = np.column_stack([np.fft.rfft(jac[:, i]) for i in range(3)])

    c_k = np.ones(len(freqs))
    c_k[0] = 0.5
    if N % 2 == 0:
        c_k[-1] = 0.5

    overall = 4.0 * T_obs / N ** 2
    F_mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            val = (dH[:, i] * np.conj(dH[:, j]) * weight * c_k).real.sum()
            F_mat[i, j] = F_mat[j, i] = val * overall

    crb_sigma = np.sqrt(np.diag(np.linalg.inv(F_mat)))
    return F_mat, crb_sigma


def fisher_matrix_white(params, sigma, N, T_obs, n_harmonics):
    """3x3 Fisher matrix for white noise: F_ij = (1/σ²) * Σ ∂h_i ∂h_j."""
    jac = _jacobian(params, N, T_obs, n_harmonics)
    F_mat = (jac.T @ jac) / sigma ** 2
    crb_sigma = np.sqrt(np.diag(np.linalg.inv(F_mat)))
    return F_mat, crb_sigma


# =====================================================================
# Data generation + tokenization
# =====================================================================

def generate_dataset(n_signals, rng):
    params = np.column_stack([
        rng.uniform(PARAM_MIN[i], PARAM_MAX[i], size=n_signals)
        for i in range(N_PARAMS)
    ])
    signals = np.zeros((n_signals, N))
    for i in range(n_signals):
        signals[i] = _emri_numpy(
            params[i, 0], params[i, 1], T_C, A0,
            params[i, 2], N_HARMONICS, N, T_OBS,
        )
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_signals}")
    return signals, params


def tokenize_signals(signals, tokenizer, device):
    all_tokens = []
    for start in range(0, len(signals), TOKENIZE_BATCH):
        batch = torch.from_numpy(signals[start:start + TOKENIZE_BATCH]).to(
            device=device, dtype=tokenizer.dlnf_grid.dtype)
        all_tokens.append(tokenizer(batch).cpu())
        done = min(start + TOKENIZE_BATCH, len(signals))
        if done % 1000 == 0 or done == len(signals):
            print(f"  {done}/{len(signals)}")
    return torch.cat(all_tokens)


# =====================================================================
# Dataset and Model
# =====================================================================

class EMRITokenDataset(Dataset):
    def __init__(self, tokens, targets):
        self.tokens = tokens
        self.targets = targets
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        return self.tokens[idx], self.targets[idx]


class EMRIModel(nn.Module):
    def __init__(self, token_emb, backbone, n_out, dropout=0.1):
        super().__init__()
        self.token_emb = token_emb
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(backbone.d_model),
            nn.Linear(backbone.d_model, backbone.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone.d_model, n_out),
            nn.Sigmoid(),
        )
    def forward(self, x):
        embedded, _, _ = self.token_emb(x)
        return self.head(self.backbone(embedded))


# =====================================================================
# Training and evaluation
# =====================================================================

def train(model, train_loader, val_loader, device, n_epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for tokens, targets in train_loader:
            tokens, targets = tokens.to(device), targets.to(device)
            pred = model(tokens)
            loss = F.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for tokens, targets in val_loader:
                tokens, targets = tokens.to(device), targets.to(device)
                val_loss += F.mse_loss(model(tokens), targets).item()
                n_val += 1
        val_losses.append(val_loss / n_val)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}"
                  f"  train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}")

    return train_losses, val_losses


def evaluate(model, val_loader, device):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for tokens, _ in val_loader:
            all_pred.append(model(tokens.to(device)).cpu())
    pred_norm = torch.cat(all_pred).numpy()
    return pred_norm * (PARAM_MAX - PARAM_MIN) + PARAM_MIN


def build_and_train(train_tokens, val_tokens, train_params, val_params,
                    device, label):
    token_emb = ToneTokenEmbedding(phase_mode=PHASE_MODE).double().to(device)
    token_emb.compute_normalization(train_tokens)

    train_targets = torch.from_numpy(
        (train_params - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)).double()
    val_targets = torch.from_numpy(
        (val_params - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)).double()

    train_loader = DataLoader(
        EMRITokenDataset(train_tokens.cpu(), train_targets),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        EMRITokenDataset(val_tokens.cpu(), val_targets),
        batch_size=BATCH_SIZE)

    n_windows = train_tokens.shape[1]
    seq_len = n_windows * N_PEAKS
    backbone = TransformerEmbedding(
        d_in=token_emb.n_embed, seq_len=seq_len,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, dropout=DROPOUT,
    ).double().to(device)
    model = EMRIModel(token_emb, backbone, n_out=N_PARAMS, dropout=DROPOUT
                      ).double().to(device)
    print(f"  [{label}] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"  [{label}] Training {N_EPOCHS} epochs...")
    tl, vl = train(model, train_loader, val_loader, device, N_EPOCHS, LR)

    pred = evaluate(model, val_loader, device)
    return pred, tl, vl


# =====================================================================
# Plotting
# =====================================================================

def plot_results(results, val_params, crb_white, crb_colored):
    """4-row plot: PSD + CRB, then one row per model."""

    labels = list(results.keys())
    n_models = len(labels)

    fig, axes = plt.subplots(1 + n_models, 3, figsize=(16, 4 * (1 + n_models)))

    # ── Row 0: PSD shape, CRB comparison, CRB ratio ──
    ax = axes[0, 0]
    freqs = np.logspace(-5, -1, 1000)
    S_n = colored_psd(freqs)
    ax.loglog(freqs * 1e3, S_n)
    f_sig = PARAM_CENTERS[0] * 1e3
    ax.axvline(f_sig, color="red", ls="--", lw=1, alpha=0.7,
               label=f"$f_0$ = {f_sig:.2f} mHz")
    ax.set_xlabel("Frequency (mHz)")
    ax.set_ylabel("$S_n(f)$")
    ax.set_title("Noise PSD")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    x_bar = np.arange(N_PARAMS)
    w = 0.35
    ax.bar(x_bar - w/2, crb_white, w, label="White ($\\sigma$=1)", alpha=0.8)
    ax.bar(x_bar + w/2, crb_colored, w, label="Colored", alpha=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(PARAM_NAMES, fontsize=9)
    ax.set_ylabel("CRB $\\sigma$")
    ax.set_title("Cramér-Rao bounds")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 2]
    ratio = crb_colored / crb_white
    bars = ax.bar(x_bar, ratio, color="steelblue", alpha=0.8)
    ax.bar_label(bars, fmt="%.1fx", fontsize=9)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(PARAM_NAMES, fontsize=9)
    ax.set_ylabel("$\\sigma_{colored} / \\sigma_{white}$")
    ax.set_title("CRB degradation")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Rows 1+: one per model ──
    colors = {"white": "C0", "colored (no whiten)": "C3",
              "colored + PSD whiten": "C1"}

    for row, label in enumerate(labels, start=1):
        pred, tl, vl = results[label]
        color = colors.get(label, f"C{row}")

        # Choose reference CRB
        crb_ref = crb_white if label == "white" else crb_colored

        # Loss curves
        ax = axes[row, 0]
        ax.semilogy(tl, label="train", color=color, alpha=0.8)
        ax.semilogy(vl, label="val", color=color, ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{label}: loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Scatter: f0 and chirp_mass
        for i in range(min(2, N_PARAMS)):
            ax = axes[row, i + 1]
            name = PARAM_NAMES[i]
            true_i = val_params[:, i]
            pred_i = pred[:, i]
            eff_sigma = np.median(np.abs(pred_i - true_i)) / 0.6745
            ratio_crb = eff_sigma / crb_ref[i]

            ax.scatter(true_i, pred_i, s=5, alpha=0.4, color=color)
            ax.plot([PARAM_MIN[i], PARAM_MAX[i]],
                    [PARAM_MIN[i], PARAM_MAX[i]], "k--", lw=1)
            ax.set_xlabel(f"True {name}")
            ax.set_ylabel(f"Predicted {name}")
            ax.set_title(f"{label}: {name} ({ratio_crb:.1f}x CRB)")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("psd_whitening_demo.png", dpi=150)
    print("Saved psd_whitening_demo.png")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rng = np.random.default_rng(SEED)

    # ── 1. Fisher information ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Fisher information at fiducial parameters")
    print("=" * 60)
    fid = dict(f0=PARAM_CENTERS[0], chirp_mass=PARAM_CENTERS[1],
               harmonic_decay=PARAM_CENTERS[2])

    print("  White-noise Fisher matrix...")
    t0 = time.time()
    _, crb_white = fisher_matrix_white(fid, sigma=1.0, N=N, T_obs=T_OBS,
                                       n_harmonics=N_HARMONICS)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("  Colored-noise Fisher matrix...")
    t0 = time.time()
    _, crb_colored = fisher_matrix_colored(fid, colored_psd, N=N,
                                           T_obs=T_OBS, n_harmonics=N_HARMONICS)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n  White noise CRB:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"    σ({name:16s}) = {crb_white[i]:.3e}")
    print("  Colored noise CRB:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"    σ({name:16s}) = {crb_colored[i]:.3e}")
    print("  Degradation (colored / white):")
    for i, name in enumerate(PARAM_NAMES):
        print(f"    {name:16s}: {crb_colored[i] / crb_white[i]:.1f}x")

    # ── 2. Generate clean signals ────────────────────────────────────
    print(f"\nGenerating {N_TRAIN + N_VAL} EMRI signals...")
    t0 = time.time()
    all_signals, all_params = generate_dataset(N_TRAIN + N_VAL, rng)
    print(f"  Done in {time.time() - t0:.1f}s")
    train_signals = all_signals[:N_TRAIN]
    val_signals = all_signals[N_TRAIN:]
    train_params = all_params[:N_TRAIN]
    val_params = all_params[N_TRAIN:]

    # ── 3. Add noise ─────────────────────────────────────────────────
    # White noise (sigma=1)
    train_white = train_signals + rng.standard_normal(train_signals.shape)
    val_white = val_signals + rng.standard_normal(val_signals.shape)

    # Colored noise
    print("Generating colored noise...")
    t0 = time.time()
    train_noise_c = generate_colored_noise(N_TRAIN, N, T_OBS, colored_psd, rng)
    val_noise_c = generate_colored_noise(N_VAL, N, T_OBS, colored_psd, rng)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Colored noise std: {train_noise_c.std():.3f}")
    train_colored = train_signals + train_noise_c
    val_colored = val_signals + val_noise_c

    # ── 4. Build tokenizers ──────────────────────────────────────────
    tok_plain = ToneTokenizer(
        k=K_WINDOW, n_peaks=N_PEAKS, n_dlnf=N_DLNF,
        dlnf_min=DLNF_MIN, dlnf_max=DLNF_MAX,
    ).double().to(device)

    tok_psd = ToneTokenizer(
        k=K_WINDOW, n_peaks=N_PEAKS, n_dlnf=N_DLNF,
        dlnf_min=DLNF_MIN, dlnf_max=DLNF_MAX,
    ).double().to(device)

    # Estimate noise std from pure noise batches
    print("Estimating noise std...")
    for _ in range(10):
        noise_batch = generate_colored_noise(64, N, T_OBS, colored_psd, rng)
        tok_psd.update_noise_std(
            torch.from_numpy(noise_batch).to(device, dtype=torch.float64),
            momentum=0.9)
    print(f"  noise_std shape: {tok_psd.noise_std.shape}")
    print(f"  noise_std range: [{tok_psd.noise_std.min():.2e}, {tok_psd.noise_std.max():.2e}]")

    # ── 5. Tokenize all three cases ──────────────────────────────────
    print("\nTokenizing: white noise (no whitening)...")
    t0 = time.time()
    train_tok_w = tokenize_signals(train_white, tok_plain, device)
    val_tok_w = tokenize_signals(val_white, tok_plain, device)
    print(f"  {time.time() - t0:.1f}s, shape: {train_tok_w.shape}")

    print("Tokenizing: colored noise (no whitening)...")
    t0 = time.time()
    train_tok_cn = tokenize_signals(train_colored, tok_plain, device)
    val_tok_cn = tokenize_signals(val_colored, tok_plain, device)
    print(f"  {time.time() - t0:.1f}s")

    print("Tokenizing: colored noise (PSD whitening)...")
    t0 = time.time()
    train_tok_cp = tokenize_signals(train_colored, tok_psd, device)
    val_tok_cp = tokenize_signals(val_colored, tok_psd, device)
    print(f"  {time.time() - t0:.1f}s")

    # ── 6. Train three models ────────────────────────────────────────
    results = {}

    print("\n" + "=" * 60)
    print("Case 1: White noise (sigma=1), no whitening")
    print("=" * 60)
    pred, tl, vl = build_and_train(
        train_tok_w, val_tok_w, train_params, val_params, device, "white")
    results["white"] = (pred, tl, vl)

    print("\n" + "=" * 60)
    print("Case 2: Colored noise, NO whitening")
    print("=" * 60)
    pred, tl, vl = build_and_train(
        train_tok_cn, val_tok_cn, train_params, val_params, device,
        "colored-no-whiten")
    results["colored (no whiten)"] = (pred, tl, vl)

    print("\n" + "=" * 60)
    print("Case 3: Colored noise + PSD whitening")
    print("=" * 60)
    pred, tl, vl = build_and_train(
        train_tok_cp, val_tok_cp, train_params, val_params, device,
        "colored+psd")
    results["colored + PSD whiten"] = (pred, tl, vl)

    # ── 7. Print comparison ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Results comparison")
    print("=" * 70)

    header = (f"  {'':25s} |"
              + "".join(f" {n:>14s}" for n in PARAM_NAMES))
    print(header)
    print("  " + "-" * (25 + 3 + 15 * N_PARAMS))

    # CRB lines
    print(f"  {'CRB (white σ=1)':25s} |"
          + "".join(f" {crb_white[i]:14.3e}" for i in range(N_PARAMS)))
    print(f"  {'CRB (colored)':25s} |"
          + "".join(f" {crb_colored[i]:14.3e}" for i in range(N_PARAMS)))
    print("  " + "-" * (25 + 3 + 15 * N_PARAMS))

    for label, (pred, _, _) in results.items():
        crb_ref = crb_white if label == "white" else crb_colored
        effs = []
        for i in range(N_PARAMS):
            eff = np.median(np.abs(pred[:, i] - val_params[:, i])) / 0.6745
            effs.append(eff)
        line = f"  {label:25s} |"
        for i in range(N_PARAMS):
            line += f"  {effs[i]:.3e} ({effs[i]/crb_ref[i]:.1f}x)"
        print(line)

    # ── 8. Plot ──────────────────────────────────────────────────────
    plot_results(results, val_params, crb_white, crb_colored)
