"""Transformer demo: predict EMRI f0 from spectral tokens."""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

from fuge import SpectralDecomposer, emri_signal

# ── Signal parameters ────────────────────────────────────────────────
N = 100_000
N_TRAIN = 5000
N_VAL = 500
NOISE_SIGMA = 1.0
SEED = 42

# ── Spectral decomposition ──────────────────────────────────────────
K_WINDOW = 1024
N_PEAKS = 3
N_DLNF = 11
DLNF_MIN = 0.0
DLNF_MAX = 0.05

# ── Fixed EMRI parameters ───────────────────────────────────────────
T_C = 1e6
A0 = 5.0
N_HARMONICS = 4
CHIRP_MASS = 1.0
HARMONIC_DECAY = 1.5

# ── Predicted parameter ─────────────────────────────────────────────
F0_CENTER = 2.75e-3
F0_HALF = 1.5e-8                          # 30 nHz half-width (near token info floor for k=1024)
F0_MIN, F0_MAX = F0_CENTER - F0_HALF, F0_CENTER + F0_HALF

# ── Transformer architecture ────────────────────────────────────────
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
DROPOUT = 0.1

# ── Embedding ────────────────────────────────────────────────────────
PHASE_MODE = "center"        # "center" or "boundary"
MASK_PHASES = False          # zero out phase features (for ablation)

# ── Training ────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 1e-3
N_EPOCHS = 200


# =====================================================================
# Data generation
# =====================================================================

def generate_dataset(n_signals, rng):
    """Generate EMRI signals with random f0.

    Returns (signals, f0s) where signals is (n, N) and f0s is (n,).
    """
    f0s = rng.uniform(F0_MIN, F0_MAX, size=n_signals)
    signals = np.zeros((n_signals, N))

    for i in range(n_signals):
        signals[i] = emri_signal(
            f0=f0s[i], chirp_mass=CHIRP_MASS, t_c=T_C, A0=A0,
            harmonic_decay=HARMONIC_DECAY, n_harmonics=N_HARMONICS, N=N,
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_signals}")

    return signals, f0s


# =====================================================================
# Tokenization
# =====================================================================

def tokenize_signals(signals, device):
    """Convert raw signals to raw token values.

    Returns 5 raw values per peak: [freq, dlnf, amp, phase_start, phase_end].
    No cos/sin or log transforms — those belong in the embedding layer.
    """
    decomposer = SpectralDecomposer(k=K_WINDOW).double().to(device)
    dlnf_grid = torch.linspace(DLNF_MIN, DLNF_MAX, N_DLNF, device=device, dtype=torch.float64)

    all_tokens = []

    for i in range(len(signals)):
        x = torch.from_numpy(signals[i]).double().to(device)

        X = decomposer(x, dlnf=dlnf_grid)  # (D, N_WINDOWS, k)

        peaks, freq_refined, dlnf_refined, peak_vals = decomposer.find_peaks(
            X, K=N_PEAKS, dlnf_grid=dlnf_grid)

        phase_start, phase_end = decomposer.peak_phases(
            X, peaks, freq_refined, dlnf_refined, dlnf_grid)

        features = torch.stack([
            freq_refined, dlnf_refined, peak_vals, phase_start, phase_end,
        ], dim=-1)  # (N_WINDOWS, K, 5)

        features = features.reshape(features.shape[0], -1)  # (N_WINDOWS, K*5)
        all_tokens.append(features)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(signals)}")

    return torch.stack(all_tokens)


# =====================================================================
# Dataset
# =====================================================================

class EMRITokenDataset(Dataset):
    def __init__(self, tokens, targets):
        self.tokens = tokens
        self.targets = targets

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.targets[idx]


# =====================================================================
# Embedding + Transformer model
# =====================================================================

class TokenEmbedding(nn.Module):
    """Embed raw token values into model-ready features.

    Raw tokens: 5 values per peak [freq, dlnf, amp, phase_start, phase_end].
    Applies cos/sin to phases, log1p to amplitude, then z-score normalizes.

    PHASE_MODE == "center":   uses (phase_start + phase_end) / 2 → 5 embedded features/peak
    PHASE_MODE == "boundary": uses phase_start and phase_end   → 7 embedded features/peak
    """

    N_FEAT = {"center": 5, "boundary": 7}

    def __init__(self, n_peaks, phase_mode="center", mask_phases=False):
        super().__init__()
        self.n_peaks = n_peaks
        self.phase_mode = phase_mode
        self.mask_phases = mask_phases
        self.n_feat = self.N_FEAT[phase_mode]
        self.n_out = n_peaks * self.n_feat
        # z-score parameters (set via set_normalization)
        self.register_buffer("mean", torch.zeros(self.n_out))
        self.register_buffer("std", torch.ones(self.n_out))

    def compute_normalization(self, raw_tokens):
        """Compute z-score stats from embedded training tokens."""
        embedded = self._embed(raw_tokens)
        flat = embedded.reshape(-1, embedded.shape[-1])
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0).clamp(min=1e-8)

    def _embed(self, raw_tokens):
        """Apply transforms to raw tokens (before z-scoring)."""
        # raw_tokens: (B, W, K*5)  →  (B, W, K, 5)
        B, W, _ = raw_tokens.shape
        raw = raw_tokens.reshape(B, W, self.n_peaks, 5)

        freq = raw[..., 0]
        dlnf = raw[..., 1]
        amp = torch.log1p(raw[..., 2])
        ps = raw[..., 3]  # phase_start
        pe = raw[..., 4]  # phase_end

        if self.phase_mode == "center":
            phi = (ps + pe) / 2
            features = torch.stack([freq, dlnf, amp,
                                    torch.cos(phi), torch.sin(phi)], dim=-1)
        else:
            features = torch.stack([freq, dlnf, amp,
                                    torch.cos(ps), torch.sin(ps),
                                    torch.cos(pe), torch.sin(pe)], dim=-1)

        out = features.reshape(B, W, -1)  # (B, W, K*n_feat)

        if self.mask_phases:
            for p in range(self.n_peaks):
                out[..., p * self.n_feat + 3 : p * self.n_feat + self.n_feat] = 0.0

        return out

    def forward(self, raw_tokens):
        return (self._embed(raw_tokens) - self.mean) / self.std


class EMRITransformer(nn.Module):
    """Embedding -> positional encoding -> TransformerEncoder ->
    global average pool -> MLP head -> Sigmoid (output in [0, 1]).
    """

    def __init__(self, embedding, seq_len, d_model, n_heads, n_layers,
                 d_ff, dropout=0.1):
        super().__init__()
        self.embedding = embedding
        self.input_proj = nn.Linear(embedding.n_out, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.input_proj(x) + self.pos_encoding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# =====================================================================
# Training
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

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}"
                  f"  train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}")

    return train_losses, val_losses


# =====================================================================
# Evaluation and plotting
# =====================================================================

def evaluate_and_plot(model, val_loader, val_f0, train_losses, val_losses, device):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for tokens, _ in val_loader:
            all_pred.append(model(tokens.to(device)).cpu())

    pred_norm = torch.cat(all_pred).numpy()
    pred_f0 = pred_norm * (F0_MAX - F0_MIN) + F0_MIN
    true_f0 = val_f0

    abs_err = np.abs(pred_f0 - true_f0)
    rel_err = abs_err / true_f0

    ss_res = np.sum((true_f0 - pred_f0) ** 2)
    ss_tot = np.sum((true_f0 - true_f0.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    med_rel = np.median(rel_err)
    med_abs = np.median(abs_err)

    print(f"\n  R² = {r2:.6f}")
    print(f"  Median absolute error: {med_abs:.2e} Hz")
    print(f"  Median relative error: {med_rel:.4%}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    ax = axes[0, 0]
    ax.semilogy(train_losses, label="train")
    ax.semilogy(val_losses, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predicted vs true
    ax = axes[0, 1]
    ax.scatter(true_f0 * 1e3, pred_f0 * 1e3, s=5, alpha=0.5)
    ax.plot([F0_MIN * 1e3, F0_MAX * 1e3], [F0_MIN * 1e3, F0_MAX * 1e3],
            "r--", lw=1, label="perfect")
    ax.set_xlabel("True $f_0$ (mHz)")
    ax.set_ylabel("Predicted $f_0$ (mHz)")
    ax.set_title(f"$f_0$:  R²={r2:.6f},  median rel. err={med_rel:.4%}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative error vs f0
    ax = axes[1, 0]
    ax.scatter(true_f0 * 1e3, rel_err * 100, s=5, alpha=0.5)
    ax.axhline(med_rel * 100, color="r", ls="--", lw=1,
               label=f"median: {med_rel:.4%}")
    ax.set_xlabel("True $f_0$ (mHz)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative error vs $f_0$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram of relative errors
    ax = axes[1, 1]
    ax.hist(rel_err * 100, bins=50, alpha=0.7, color="steelblue")
    ax.axvline(med_rel * 100, color="r", ls="--", lw=2,
               label=f"median: {med_rel:.4%}")
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("transformer_demo.png", dpi=150)
    print("Saved transformer_demo.png")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rng = np.random.default_rng(SEED)

    # 1. Generate signals
    print(f"Generating {N_TRAIN + N_VAL} EMRI signals...")
    t0 = time.time()
    all_signals, all_f0s = generate_dataset(N_TRAIN + N_VAL, rng)
    print(f"  Done in {time.time() - t0:.1f}s")

    train_signals, val_signals = all_signals[:N_TRAIN], all_signals[N_TRAIN:]
    train_f0, val_f0 = all_f0s[:N_TRAIN], all_f0s[N_TRAIN:]

    # 2. Add noise
    if NOISE_SIGMA > 0:
        train_signals = train_signals + rng.standard_normal(train_signals.shape) * NOISE_SIGMA
        val_signals = val_signals + rng.standard_normal(val_signals.shape) * NOISE_SIGMA

    # 3. Tokenize
    print("Tokenizing signals...")
    t0 = time.time()
    train_tokens = tokenize_signals(train_signals, device)
    val_tokens = tokenize_signals(val_signals, device)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Token shape: {train_tokens.shape}")

    # 4. Build embedding and compute normalization from training tokens
    embedding = TokenEmbedding(N_PEAKS, phase_mode=PHASE_MODE,
                               mask_phases=MASK_PHASES).double().to(device)
    embedding.compute_normalization(train_tokens)

    # Move raw tokens to CPU for DataLoader
    train_tokens = train_tokens.cpu()
    val_tokens = val_tokens.cpu()

    # Min-max normalize f0 to [0, 1]
    train_targets = torch.from_numpy(
        (train_f0 - F0_MIN) / (F0_MAX - F0_MIN)).double()
    val_targets = torch.from_numpy(
        (val_f0 - F0_MIN) / (F0_MAX - F0_MIN)).double()

    # 5. Data loaders
    train_loader = DataLoader(
        EMRITokenDataset(train_tokens, train_targets),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        EMRITokenDataset(val_tokens, val_targets),
        batch_size=BATCH_SIZE)

    # 6. Model
    seq_len = train_tokens.shape[1]
    model = EMRITransformer(
        embedding=embedding, seq_len=seq_len, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
    ).double().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 7. Train
    print(f"Training for {N_EPOCHS} epochs...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, device, N_EPOCHS, LR)

    # 8. Evaluate and plot
    evaluate_and_plot(model, val_loader, val_f0, train_losses, val_losses, device)
