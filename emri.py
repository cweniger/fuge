"""Fake EMRI (Extreme Mass Ratio Inspiral) gravitational wave signal generator.

Auto-differentiable w.r.t. all continuous parameters via JAX.

Parameters
----------
f0 : initial GW frequency (Hz)
chirp_mass : controls how fast the frequency sweeps up (arbitrary units)
t_c : coalescence time (s); must be > T_obs
A0 : overall amplitude scale
harmonic_decay : exponential decay rate for higher harmonics;
    amplitude of k-th harmonic ~ exp(-harmonic_decay * (k - 1))

The waveform is built as:
    h(t) = sum_{k=1}^{n_harmonics} A_k(t) * cos(k * phi(t))

where phi(t) = 2*pi * cumulative_trapezoid(f(t)) and

    f(t) = f0 * (1 - t / t_c)^(-3/8 * chirp_mass)   (PN-inspired chirp)
    A(t) = A0 * (1 - t / t_c)^(-1/4)                  (PN-inspired amplitude growth)
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np


def emri_signal(
    f0: float,
    chirp_mass: float,
    t_c: float,
    A0: float,
    harmonic_decay: float,
    n_harmonics: int = 4,
    N: int = 1_000_000,
    T_obs: float | None = None,
) -> np.ndarray:
    """Generate a fake EMRI gravitational wave signal.

    Parameters
    ----------
    f0 : float
        Initial gravitational wave frequency (Hz).
    chirp_mass : float
        Chirp mass parameter controlling frequency evolution rate.
        Higher values produce faster chirps.  Units are arbitrary;
        it enters as f(t) = f0 * (1 - t / t_c)^(-3/8 * chirp_mass).
    t_c : float
        Time of coalescence (s).  Must exceed T_obs.
    A0 : float
        Overall amplitude scale.
    harmonic_decay : float
        Exponential decay factor for harmonic amplitudes.
        Harmonic k has amplitude A(t) * exp(-harmonic_decay * (k-1)).
    n_harmonics : int
        Number of harmonics to include (1 = fundamental only).
    N : int
        Number of time-domain samples.
    T_obs : float or None
        Observation duration (s).  Defaults to 0.9 * t_c.

    Returns
    -------
    h : numpy.ndarray, shape (N,)
        Time-domain gravitational wave strain.
    """
    if T_obs is None:
        T_obs = 0.9 * t_c
    h = _emri_impl(f0, chirp_mass, t_c, A0, harmonic_decay, n_harmonics, N, T_obs)
    return np.asarray(h)


@functools.partial(jax.jit, static_argnums=(5, 6))
def _emri_impl(f0, chirp_mass, t_c, A0, harmonic_decay, n_harmonics, N, T_obs):
    """JIT-compiled core.  n_harmonics and N are static (integers)."""
    t = jnp.linspace(0.0, T_obs, N)
    dt = T_obs / (N - 1)

    # Dimensionless time to coalescence: tau in (0, 1]
    tau = 1.0 - t / t_c

    # --- Frequency evolution (PN-inspired) ---
    freq_exponent = -3.0 / 8.0 * chirp_mass
    f_t = f0 * tau ** freq_exponent

    # --- Amplitude evolution ---
    A_t = A0 * tau ** (-0.25)

    # --- Phase via trapezoidal cumulative integration ---
    # phi(t_n) = 2*pi * sum_{i=0}^{n-1} (f[i] + f[i+1]) / 2 * dt
    # O(dt^2) global error.
    trapezoids = (f_t[:-1] + f_t[1:]) * 0.5 * dt
    phase = jnp.concatenate([jnp.zeros(1), jnp.cumsum(trapezoids)])
    phase = 2.0 * jnp.pi * phase

    # --- Sum over harmonics ---
    def _add_harmonic(h, k):
        amp_k = A_t * jnp.exp(-harmonic_decay * (k - 1))
        h = h + amp_k * jnp.cos(k * phase)
        return h, None

    ks = jnp.arange(1, n_harmonics + 1)
    h, _ = jax.lax.scan(_add_harmonic, jnp.zeros(N), ks)

    return h


# ------------------------------------------------------------------
# Quick demo / sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Reasonable fake-EMRI parameters
    params = dict(
        f0=1e-3,            # 1 mHz initial frequency
        chirp_mass=1.0,     # standard PN exponent
        t_c=1e6,            # coalescence at 1e6 s (~11.6 days)
        A0=1e-21,           # strain amplitude
        harmonic_decay=1.5, # higher harmonics decay quickly
        n_harmonics=4,
        N=100_000,          # fewer points for quick plot
    )

    h = emri_signal(**params)
    T_obs = 0.9 * params["t_c"]
    t = np.linspace(0, T_obs, params["N"])

    # Test autodiff
    def loss(f0):
        return jnp.sum(_emri_impl(f0, 1.0, 1e6, 1e-21, 1.5, 4, 10_000, 0.9e6))
    grad_f0 = jax.grad(loss)(1e-3)
    print(f"grad(sum(h)) w.r.t. f0 = {grad_f0:.6e}  (autodiff works)")

    from scipy.signal import spectrogram

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # --- Top left: full time-domain waveform ---
    axes[0, 0].plot(t, h, linewidth=0.3)
    axes[0, 0].set_xlabel("t (s)")
    axes[0, 0].set_ylabel("h(t)")
    axes[0, 0].set_title("Full waveform")

    # --- Top right: zoomed-in near the plunge ---
    # Last 2% of observation time (near coalescence)
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
    fs = len(h) / T_obs  # sampling frequency
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
    # Crop to the interesting frequency range
    f_max = params["f0"] * (params["n_harmonics"] + 1) * 5
    axes[1, 1].set_ylim(params["f0"] * 0.5, f_max)
    fig.colorbar(im, ax=axes[1, 1], label="PSD (dB)")

    plt.tight_layout()
    plt.savefig("emri_demo.png", dpi=150)
    print("Saved emri_demo.png")
