"""Chirp signal generator demo: time/frequency domain plots and spectrogram."""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.signal import spectrogram

from chirp import chirp_signal, _chirp_impl

if __name__ == "__main__":
    params = dict(
        f0=1e-3,            # 1 mHz initial frequency
        chirp_mass=1.0,     # standard PN exponent
        t_c=1e6,            # coalescence at 1e6 s (~11.6 days)
        A0=1e-21,           # strain amplitude
        harmonic_decay=1.5, # higher harmonics decay quickly
        n_harmonics=4,
        N=100_000,          # fewer points for quick plot
    )

    h = chirp_signal(**params)
    T_obs = 0.9 * params["t_c"]
    t = np.linspace(0, T_obs, params["N"])

    # Test autodiff
    def loss(f0):
        return jnp.sum(_chirp_impl(f0, 1.0, 1e6, 1e-21, 1.5, 4, 10_000, 0.9e6))
    grad_f0 = jax.grad(loss)(1e-3)
    print(f"grad(sum(h)) w.r.t. f0 = {grad_f0:.6e}  (autodiff works)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # --- Top left: full time-domain waveform ---
    axes[0, 0].plot(t, h, linewidth=0.3)
    axes[0, 0].set_xlabel("t (s)")
    axes[0, 0].set_ylabel("h(t)")
    axes[0, 0].set_title("Full waveform")

    # --- Top right: zoomed-in near the plunge ---
    plunge_frac = 0.02
    t_start = T_obs * (1.0 - plunge_frac)
    mask = t >= t_start
    axes[0, 1].plot(t[mask], h[mask], linewidth=0.5)
    axes[0, 1].set_xlabel("t (s)")
    axes[0, 1].set_ylabel("h(t)")
    axes[0, 1].set_title(f"Plunge (last {plunge_frac*100:.0f}% of observation)")

    # --- Bottom left: frequency domain ---
    freqs = np.fft.rfftfreq(len(h), d=T_obs / len(h))
    H = np.fft.rfft(h)
    axes[1, 0].loglog(freqs[1:], np.abs(H[1:]), linewidth=0.3)
    axes[1, 0].set_xlabel("f (Hz)")
    axes[1, 0].set_ylabel("|H(f)|")
    axes[1, 0].set_title("Frequency domain")

    # --- Bottom right: spectrogram ---
    fs = len(h) / T_obs
    nperseg = min(2048, len(h) // 16)
    f_spec, t_spec, Sxx = spectrogram(h, fs=fs, nperseg=nperseg,
                                       noverlap=nperseg * 3 // 4)
    Sxx_db = 10 * np.log10(Sxx + 1e-300)
    im = axes[1, 1].pcolormesh(t_spec, f_spec, Sxx_db, shading="gouraud",
                                cmap="inferno")
    axes[1, 1].set_ylabel("f (Hz)")
    axes[1, 1].set_xlabel("t (s)")
    axes[1, 1].set_title("Spectrogram")
    axes[1, 1].set_yscale("log")
    f_max = params["f0"] * (params["n_harmonics"] + 1) * 5
    axes[1, 1].set_ylim(params["f0"] * 0.5, f_max)
    fig.colorbar(im, ax=axes[1, 1], label="PSD (dB)")

    plt.tight_layout()
    plt.savefig("chirp_demo.png", dpi=150)
    print("Saved chirp_demo.png")
