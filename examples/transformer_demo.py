"""Transformer demo: predict EMRI parameters from spectral tokens."""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

from fuge import SpectralTokenizer, emri_signal

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

# ── Predicted parameters (prior = ~10x marginalized CRB width) ─────
# CRB from 3-parameter Fisher matrix at fiducial values:
#   σ(f0)=4.89e-10, σ(chirp_mass)=7.37e-7, σ(harmonic_decay)=2.94e-3
PARAM_NAMES = ["f0", "chirp_mass", "harmonic_decay"]
PARAM_CENTERS = np.array([2.75e-3, 1.0, 1.5])
PARAM_HALF = np.array([2.5e-9, 3.7e-6, 1.5e-2])  # ~5x CRB σ each side
PARAM_MIN = PARAM_CENTERS - PARAM_HALF
PARAM_MAX = PARAM_CENTERS + PARAM_HALF
N_PARAMS = len(PARAM_NAMES)

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
N_EPOCHS = 60


# =====================================================================
# Data generation
# =====================================================================

def generate_dataset(n_signals, rng):
    """Generate EMRI signals with random (f0, chirp_mass, harmonic_decay).

    Returns (signals, params) where signals is (n, N) and params is (n, 3).
    """
    params = np.column_stack([
        rng.uniform(PARAM_MIN[i], PARAM_MAX[i], size=n_signals)
        for i in range(N_PARAMS)
    ])
    signals = np.zeros((n_signals, N))

    for i in range(n_signals):
        signals[i] = emri_signal(
            f0=params[i, 0], chirp_mass=params[i, 1], t_c=T_C, A0=A0,
            harmonic_decay=params[i, 2], n_harmonics=N_HARMONICS, N=N,
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_signals}")

    return signals, params


# =====================================================================
# Tokenization
# =====================================================================

TOKENIZE_BATCH = 64  # signals per GPU batch during tokenization

def tokenize_signals(signals, tokenizer, device):
    """Tokenize signals in batches using SpectralTokenizer."""
    all_tokens = []
    for start in range(0, len(signals), TOKENIZE_BATCH):
        batch = torch.from_numpy(signals[start:start + TOKENIZE_BATCH]).to(
            device=device, dtype=tokenizer.dlnf_grid.dtype)
        all_tokens.append(tokenizer(batch).cpu())
        done = min(start + TOKENIZE_BATCH, len(signals))
        if done % 500 == 0 or done == len(signals):
            print(f"  {done}/{len(signals)}")
    return torch.cat(all_tokens)


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
    """Embed raw peak tokens into model-ready features.

    Input: (B, W, K, 5) raw values [freq, dlnf, amp, phase_start, phase_end].
    Output: (B, W*K, n_embed) z-score normalized embedded features.

    Each peak becomes an independent token in the sequence.  Applies
    cos/sin to phases, log1p to amplitude, then z-score normalizes.

    PHASE_MODE == "center":   (phase_start + phase_end) / 2  → n_embed = 5
    PHASE_MODE == "boundary": keep both endpoints            → n_embed = 7
    """

    N_EMBED = {"center": 5, "boundary": 7}

    def __init__(self, phase_mode="center", mask_phases=False):
        super().__init__()
        self.phase_mode = phase_mode
        self.mask_phases = mask_phases
        self.n_embed = self.N_EMBED[phase_mode]
        self.register_buffer("mean", torch.zeros(self.n_embed))
        self.register_buffer("std", torch.ones(self.n_embed))

    def compute_normalization(self, raw_tokens):
        """Compute z-score stats from training tokens.

        raw_tokens: (B, W, K, 5)
        """
        embedded = self._embed(raw_tokens)          # (B, W, K, n_embed)
        flat = embedded.reshape(-1, self.n_embed)
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0).clamp(min=1e-8)

    def _embed(self, raw_tokens):
        """Transform raw features (before z-scoring).

        raw_tokens: (B, W, K, 5) -> (B, W, K, n_embed)
        """
        freq = raw_tokens[..., 0]
        dlnf = raw_tokens[..., 1]
        amp = torch.log1p(raw_tokens[..., 2])
        ps = raw_tokens[..., 3]
        pe = raw_tokens[..., 4]

        if self.phase_mode == "center":
            phi = (ps + pe) / 2
            out = torch.stack([freq, dlnf, amp,
                               torch.cos(phi), torch.sin(phi)], dim=-1)
        else:
            out = torch.stack([freq, dlnf, amp,
                               torch.cos(ps), torch.sin(ps),
                               torch.cos(pe), torch.sin(pe)], dim=-1)

        if self.mask_phases:
            out[..., 3:] = 0.0

        return out

    def forward(self, raw_tokens):
        """Embed and normalize: (B, W, K, 5) -> (B, W*K, n_embed)."""
        B, W, K, _ = raw_tokens.shape
        embedded = self._embed(raw_tokens)                   # (B, W, K, n_embed)
        normalized = (embedded - self.mean) / self.std
        return normalized.reshape(B, W * K, self.n_embed), W, K


class EMRITransformer(nn.Module):
    """Embedding -> time-only positional encoding -> TransformerEncoder ->
    global average pool -> MLP head -> Sigmoid.

    Each peak is an independent token.  Positional encoding is shared
    across all K peaks within the same time window.
    """

    def __init__(self, embedding, n_windows, n_peaks, n_out, d_model,
                 n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = embedding
        self.n_peaks = n_peaks
        self.input_proj = nn.Linear(embedding.n_embed, d_model)
        # Time-only positional encoding: (1, W, 1, d_model), broadcast over K
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_windows, 1, d_model) * 0.02)

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
            nn.Linear(d_model, n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, W, K, 5) raw tokens."""
        B, W, K, _ = x.shape
        embedded, _, _ = self.embedding(x)        # (B, W*K, n_embed)
        projected = self.input_proj(embedded)      # (B, W*K, d_model)
        # Reshape to (B, W, K, d_model), add time pos encoding, flatten back
        projected = projected.reshape(B, W, K, -1)
        projected = projected + self.pos_encoding  # broadcasts over K
        projected = projected.reshape(B, W * K, -1)
        x = self.encoder(projected)
        x = x.mean(dim=1)
        return self.head(x)


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

def evaluate_and_plot(model, val_loader, val_params, train_losses, val_losses, device):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for tokens, _ in val_loader:
            all_pred.append(model(tokens.to(device)).cpu())

    pred_norm = torch.cat(all_pred).numpy()  # (N_VAL, 3)
    pred_params = pred_norm * (PARAM_MAX - PARAM_MIN) + PARAM_MIN
    true_params = val_params

    print()
    for i, name in enumerate(PARAM_NAMES):
        abs_err = np.abs(pred_params[:, i] - true_params[:, i])
        ss_res = np.sum((true_params[:, i] - pred_params[:, i]) ** 2)
        ss_tot = np.sum((true_params[:, i] - true_params[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        med_abs = np.median(abs_err)
        crb_sigma = np.array([4.89e-10, 7.37e-7, 2.94e-3])
        eff_sigma = med_abs / 0.6745
        ratio = eff_sigma / crb_sigma[i]
        print(f"  {name:16s}  R²={r2:.6f}  median|err|={med_abs:.2e}  "
              f"eff_σ={eff_sigma:.2e}  ({ratio:.1f}x CRB)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    ax = axes[0, 0]
    ax.semilogy(train_losses, label="train")
    ax.semilogy(val_losses, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plots for each parameter
    for i, (name, ax) in enumerate(zip(PARAM_NAMES, [axes[0, 1], axes[1, 0], axes[1, 1]])):
        true_i = true_params[:, i]
        pred_i = pred_params[:, i]
        abs_err = np.abs(pred_i - true_i)
        ss_res = np.sum((true_i - pred_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        med_abs = np.median(abs_err)

        ax.scatter(true_i, pred_i, s=5, alpha=0.5)
        ax.plot([PARAM_MIN[i], PARAM_MAX[i]], [PARAM_MIN[i], PARAM_MAX[i]],
                "r--", lw=1, label="perfect")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}:  R²={r2:.6f},  median|err|={med_abs:.2e}")
        ax.legend(fontsize=8)
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
    all_signals, all_params = generate_dataset(N_TRAIN + N_VAL, rng)
    print(f"  Done in {time.time() - t0:.1f}s")

    train_signals, val_signals = all_signals[:N_TRAIN], all_signals[N_TRAIN:]
    train_params, val_params = all_params[:N_TRAIN], all_params[N_TRAIN:]

    # 2. Add noise
    if NOISE_SIGMA > 0:
        train_signals = train_signals + rng.standard_normal(train_signals.shape) * NOISE_SIGMA
        val_signals = val_signals + rng.standard_normal(val_signals.shape) * NOISE_SIGMA

    # 3. Tokenize
    print("Tokenizing signals...")
    tokenizer = SpectralTokenizer(
        k=K_WINDOW, n_peaks=N_PEAKS, n_dlnf=N_DLNF,
        dlnf_min=DLNF_MIN, dlnf_max=DLNF_MAX,
    ).double().to(device)
    t0 = time.time()
    train_tokens = tokenize_signals(train_signals, tokenizer, device)
    val_tokens = tokenize_signals(val_signals, tokenizer, device)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Token shape: {train_tokens.shape}")

    # 4. Build embedding and compute normalization from training tokens
    embedding = TokenEmbedding(phase_mode=PHASE_MODE,
                               mask_phases=MASK_PHASES).double().to(device)
    embedding.compute_normalization(train_tokens)

    # Move raw tokens to CPU for DataLoader
    train_tokens = train_tokens.cpu()
    val_tokens = val_tokens.cpu()

    # Min-max normalize params to [0, 1]
    train_targets = torch.from_numpy(
        (train_params - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)).double()
    val_targets = torch.from_numpy(
        (val_params - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)).double()

    # 5. Data loaders
    train_loader = DataLoader(
        EMRITokenDataset(train_tokens, train_targets),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        EMRITokenDataset(val_tokens, val_targets),
        batch_size=BATCH_SIZE)

    # 6. Model
    n_windows = train_tokens.shape[1]
    model = EMRITransformer(
        embedding=embedding, n_windows=n_windows, n_peaks=N_PEAKS,
        n_out=N_PARAMS, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
    ).double().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 7. Train
    print(f"Training for {N_EPOCHS} epochs...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, device, N_EPOCHS, LR)

    # 8. Evaluate and plot
    evaluate_and_plot(model, val_loader, val_params, train_losses, val_losses, device)
