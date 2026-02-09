"""Transformer demo: predict EMRI parameters from spectral tokens."""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from spectral_tokens import SpectralDecomposer, emri_signal

# ── Signal parameters ────────────────────────────────────────────────
N = 100_000
N_TRAIN = 2000
N_VAL = 500
NOISE_SIGMA = 1.0
SEED = 42

# ── Spectral decomposition ──────────────────────────────────────────
K_WINDOW = 1024
N_PEAKS = 3
N_DLNF = 11
DLNF_MIN = 0.0
DLNF_MAX = 0.05

# ── Fixed EMRI parameters (not predicted) ───────────────────────────
T_C = 1e6
A0 = 5.0
N_HARMONICS = 4

# ── Predicted parameters and their ranges ───────────────────────────
PARAM_RANGES = {
    "f0": (5e-4, 5e-3),
    "chirp_mass": (0.5, 2.0),
    "harmonic_decay": (0.5, 3.0),
}

# ── Transformer architecture ────────────────────────────────────────
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
DROPOUT = 0.1

# ── Training ────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 1e-3
N_EPOCHS = 80


# =====================================================================
# Data generation
# =====================================================================

def generate_dataset(n_signals, rng):
    """Generate EMRI signals with random parameters.

    Returns (signals, params) where signals has shape (n, N) and
    params has shape (n, 3) with columns [f0, chirp_mass, harmonic_decay].
    """
    params = np.zeros((n_signals, 3))
    signals = np.zeros((n_signals, N))

    for i in range(n_signals):
        f0 = rng.uniform(*PARAM_RANGES["f0"])
        chirp_mass = rng.uniform(*PARAM_RANGES["chirp_mass"])
        harmonic_decay = rng.uniform(*PARAM_RANGES["harmonic_decay"])

        params[i] = [f0, chirp_mass, harmonic_decay]
        signals[i] = emri_signal(
            f0=f0, chirp_mass=chirp_mass, t_c=T_C, A0=A0,
            harmonic_decay=harmonic_decay, n_harmonics=N_HARMONICS, N=N,
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_signals}")

    return signals, params


# =====================================================================
# Tokenization
# =====================================================================

def tokenize_signals(signals, device):
    """Convert raw signals to token features.

    Returns Tensor of shape (n_signals, N_WINDOWS, N_PEAKS * 5).
    Features per peak: [freq, dlnf, log_amplitude, phi_0, phi_1].
    """
    decomposer = SpectralDecomposer(k=K_WINDOW).to(device)
    dlnf_grid = torch.linspace(DLNF_MIN, DLNF_MAX, N_DLNF, device=device)

    all_tokens = []

    for i in range(len(signals)):
        x = torch.from_numpy(signals[i]).float().to(device)

        X = decomposer(x, dlnf=dlnf_grid)  # (D, N_WINDOWS, k)

        peaks, freq_refined, dlnf_refined, peak_vals = decomposer.find_peaks(
            X, K=N_PEAKS, dlnf_grid=dlnf_grid)

        phi_0, phi_1 = decomposer.peak_phases(
            X, peaks, freq_refined, dlnf_refined, dlnf_grid)

        features = torch.stack([
            freq_refined,
            dlnf_refined,
            torch.log1p(peak_vals),
            torch.remainder(phi_0, 2 * torch.pi),
            torch.remainder(phi_1, 2 * torch.pi),
        ], dim=-1)  # (N_WINDOWS, K, 5)

        features = features.reshape(features.shape[0], -1)  # (N_WINDOWS, K*5)
        all_tokens.append(features)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(signals)}")

    return torch.stack(all_tokens)  # (n_signals, N_WINDOWS, K*5)


def compute_normalization(tokens):
    flat = tokens.reshape(-1, tokens.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp(min=1e-8)
    return mean, std


def normalize_tokens(tokens, mean, std):
    return (tokens - mean) / std


def normalize_params(params):
    mins = np.array([v[0] for v in PARAM_RANGES.values()])
    maxs = np.array([v[1] for v in PARAM_RANGES.values()])
    return (params - mins) / (maxs - mins)


def denormalize_params(params_norm):
    mins = np.array([v[0] for v in PARAM_RANGES.values()])
    maxs = np.array([v[1] for v in PARAM_RANGES.values()])
    return params_norm * (maxs - mins) + mins


# =====================================================================
# Dataset
# =====================================================================

class EMRITokenDataset(Dataset):
    def __init__(self, tokens, params):
        self.tokens = tokens
        self.params = params

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.params[idx]


# =====================================================================
# Transformer model
# =====================================================================

class EMRITransformer(nn.Module):
    """Encoder-only transformer for EMRI parameter regression.

    Input projection → positional encoding → TransformerEncoder →
    global average pool → MLP head → Sigmoid (outputs in [0, 1]).
    """

    def __init__(self, n_features, seq_len, d_model, n_heads, n_layers,
                 d_ff, n_params, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
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
            nn.Linear(d_model, n_params),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding
        x = self.encoder(x)
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
        for tokens, params in train_loader:
            tokens, params = tokens.to(device), params.to(device)
            pred = model(tokens)
            loss = F.mse_loss(pred, params)
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
            for tokens, params in val_loader:
                tokens, params = tokens.to(device), params.to(device)
                val_loss += F.mse_loss(model(tokens), params).item()
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

def evaluate_and_plot(model, val_loader, train_losses, val_losses, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for tokens, params in val_loader:
            all_pred.append(model(tokens.to(device)).cpu())
            all_true.append(params)

    pred = denormalize_params(torch.cat(all_pred).numpy())
    true = denormalize_params(torch.cat(all_true).numpy())

    param_labels = ["$f_0$ (Hz)", "chirp mass", "harmonic decay"]

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

    # Scatter plots
    for i, (ax, name, label) in enumerate(zip(
            [axes[0, 1], axes[1, 0], axes[1, 1]],
            PARAM_RANGES.keys(), param_labels)):
        ax.scatter(true[:, i], pred[:, i], s=5, alpha=0.5)
        lo, hi = PARAM_RANGES[name]
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect")

        ss_res = np.sum((true[:, i] - pred[:, i]) ** 2)
        ss_tot = np.sum((true[:, i] - true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        med_rel = np.median(np.abs(pred[:, i] - true[:, i]) / np.abs(true[:, i]))

        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}:  R²={r2:.3f},  median rel. err={med_rel:.1%}")
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
    t0 = time.time()
    train_tokens = tokenize_signals(train_signals, device)
    val_tokens = tokenize_signals(val_signals, device)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Token shape: {train_tokens.shape}")

    # 4. Normalize
    tok_mean, tok_std = compute_normalization(train_tokens)
    train_tokens = normalize_tokens(train_tokens, tok_mean, tok_std).cpu()
    val_tokens = normalize_tokens(val_tokens, tok_mean, tok_std).cpu()

    train_params_norm = torch.from_numpy(normalize_params(train_params)).float()
    val_params_norm = torch.from_numpy(normalize_params(val_params)).float()

    # 5. Data loaders
    train_loader = DataLoader(
        EMRITokenDataset(train_tokens, train_params_norm),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        EMRITokenDataset(val_tokens, val_params_norm),
        batch_size=BATCH_SIZE)

    # 6. Model
    seq_len = train_tokens.shape[1]
    model = EMRITransformer(
        n_features=N_PEAKS * 5, seq_len=seq_len, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        n_params=len(PARAM_RANGES), dropout=DROPOUT,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 7. Train
    print(f"Training for {N_EPOCHS} epochs...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, device, N_EPOCHS, LR)

    # 8. Evaluate and plot
    evaluate_and_plot(model, val_loader, train_losses, val_losses, device)
