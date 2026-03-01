"""
Streaming PCA via SVD with Procrustes-stabilized output.

Provides `PCAProjector`, a streaming dimensionality reduction module that
maintains principal components and eigenvalues via momentum-blended SVD
updates. Procrustes alignment ensures output coefficients have a stable
meaning across updates — critical when feeding into a neural network.

The key idea: eigenvalue-scaled old components are concatenated with new
data, and a single SVD of this matrix yields both updated components and
eigenvalues jointly.

Expects whitened input (zero mean, unit noise variance). The Wiener filter
assumes noise variance = 1.

Example::

    proj = PCAProjector(n_components=32, buffer_size=256, momentum=0.1)

    # Training: stream whitened batches through update()
    for batch in data_loader:
        proj.update(whiten(batch))

    # Inference: get stable k-dimensional coefficients
    coeffs = proj.project(whiten(x))  # (batch_size, 32), ~unit variance
"""

import torch
import numpy as np
from typing import Optional, List


class PCAProjector(torch.nn.Module):
    """
    Streaming PCA projector with Procrustes-stabilized output.

    Maintains eigenbasis (V, Λ) where eigenvalues and components match
    exactly, plus an orthogonal rotation R from eigenbasis to a stable
    output frame.

    State: V ∈ R^{k×D} (eigenbasis), Λ ∈ R^k (eigenvalues), R ∈ O(k).

    Update (when buffer full):
        U = [ √(1-α) · diag(√Λ_old) · V_old ;  √(α/M) · X ]
        SVD(U) → V_new, S² = Λ_new
        Procrustes(V_new, R_old @ V_old) → R_new

    forward(X) → stable k-dim coefficients:
        1. c = X @ V.T              (project onto eigenbasis)
        2. c *= λ/(λ+1)             (Wiener filter, diagonal)
        3. c /= √λ                  (normalize to ~unit variance)
        4. c_out = c @ R.T          (rotate to stable frame)

    reconstruct(X) → D-dim filtered signal (R cancels in round-trip).
    """

    def __init__(
        self,
        n_components: int = 10,
        buffer_size: int = 256,
        momentum: float = 0.1,
        normalize_output: bool = True,
        use_prior: bool = True,
    ) -> None:
        """
        Args:
            n_components: Number of principal components to retain.
            buffer_size: Number of samples to accumulate before an SVD update.
            momentum: Blend factor α for merging new data with old covariance.
            normalize_output: Whether to normalize D-dim reconstructions to unit
                              average variance (only affects forward(), not project()).
            use_prior: Whether to apply Wiener-filter shrinkage.
        """
        super().__init__()
        self.n_components: int = n_components
        self.buffer_size: int = buffer_size
        self.momentum: float = momentum
        self.normalize_output: bool = normalize_output
        self.use_prior: bool = use_prior

        self.buffer: List[torch.Tensor] = []
        self.buffer_counter: int = 0

        # Eigenbasis: V and Λ match (V are eigenvectors, Λ eigenvalues)
        self.components: Optional[torch.Tensor] = None    # (k, D)
        self.eigenvalues: Optional[torch.Tensor] = None   # (k,)

        # Procrustes rotation: eigenbasis → stable output frame
        self._R: Optional[torch.Tensor] = None             # (k, k)

    def update(self, X: torch.Tensor) -> None:
        """
        Accumulate a batch of data in the buffer. When full, update PCA.

        Args:
            X: Whitened input data with shape (batch_size, D).
        """
        batch_size = X.shape[0]
        self.buffer.append(X)
        self.buffer_counter += batch_size

        if self.buffer_counter >= self.buffer_size:
            self._compute_svd_update()
            self.buffer = []
            self.buffer_counter = 0

    @staticmethod
    def _procrustes(V_new, V_old):
        """Align V_new to V_old via orthogonal Procrustes. Both shape (k, D).

        Returns (V_aligned, R) where V_aligned = R @ V_new and R is the
        optimal k×k orthogonal rotation minimizing ||R @ V_new - V_old||_F.
        """
        C = V_old @ V_new.T  # (k, k)
        U, S, Wt = torch.linalg.svd(C)
        R = U @ Wt
        return R @ V_new, R

    def _compute_svd_update(self) -> None:
        """
        Unified eigenvalue-scaled covariance update with Procrustes stabilization.

        Builds U_combined so that U_combined.T @ U_combined equals the
        momentum-blended covariance, then takes a single SVD.
        """
        X = torch.cat(self.buffer, dim=0)  # (M, D)
        M = X.shape[0]
        alpha = self.momentum

        if self.components is None:
            # First update: dual PCA (M < D trick)
            K = X @ X.T / M  # (M, M)
            eigvals, eigvecs = torch.linalg.eigh(K)

            top_indices = torch.argsort(eigvals, descending=True)[:self.n_components]
            Q = eigvecs[:, top_indices]  # (M, k)
            Λ = eigvals[top_indices]     # (k,)

            V = (X.T @ Q) / torch.sqrt(M * Λ.clamp(min=1e-12))  # (D, k)
            V = V.T  # (k, D)

            self.components = V
            self.eigenvalues = Λ
            self._R = torch.eye(self.n_components, dtype=V.dtype, device=V.device)
        else:
            # Eigenvalue-scaled old components
            Λ_sqrt = torch.sqrt(self.eigenvalues.clamp(min=1e-12))
            scaled_old = np.sqrt(1 - alpha) * (Λ_sqrt.unsqueeze(1) * self.components)

            # Scaled new data
            scaled_new = np.sqrt(alpha / M) * X

            # Single SVD of concatenation → eigenbasis + eigenvalues
            U_combined = torch.cat([scaled_old, scaled_new], dim=0)  # (k+M, D)
            U, S, Vt = torch.linalg.svd(U_combined, full_matrices=False)
            V_new = Vt[:self.n_components]       # (k, D)
            Λ_new = S[:self.n_components] ** 2   # (k,)

            # Procrustes: align to previous stable frame (R_old @ V_old)
            V_stable_old = self._R @ self.components
            _, R_new = self._procrustes(V_new, V_stable_old)

            self.components = V_new
            self.eigenvalues = Λ_new
            self._R = R_new

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return stable k-dimensional coefficients with Wiener denoising.

        The computation path:
          1. Project onto eigenbasis: c = X @ V.T        (diagonal covariance)
          2. Wiener filter:           c *= λ/(λ+1)       (shrinkage, diagonal)
          3. Normalize:               c /= sqrt(λ)       (→ ~unit variance)
          4. Rotate to stable frame:  c_stable = c @ R.T  (Procrustes stability)

        Args:
            X: Whitened input of shape (batch_size, D).

        Returns:
            Stable coefficients of shape (batch_size, k), ~unit variance
            for signal-dominated components.
        """
        if self.components is None:
            raise ValueError(
                "SVD components not computed yet. Call update() enough times first."
            )

        # Project onto eigenbasis (diagonal covariance)
        coeffs = X @ self.components.T  # (batch_size, k)

        if self.use_prior and self.eigenvalues is not None:
            # Wiener filter + normalize, diagonal in eigenbasis
            Λ = self.eigenvalues.clamp(min=1e-12)
            coeffs = coeffs * (Λ / (Λ + 1.0) / torch.sqrt(Λ)).unsqueeze(0)

        # Rotate to Procrustes-stable frame
        if self._R is not None:
            coeffs = coeffs @ self._R.T

        return coeffs

    def reconstruct(self, X: torch.Tensor) -> torch.Tensor:
        """
        Filter and reconstruct in original D-dimensional space.

        Wiener filter is diagonal in eigenbasis V. R is irrelevant here
        (cancels in the round-trip V.T @ diag(shrink) @ V).

        Args:
            X: Whitened input of shape (batch_size, D).

        Returns:
            Reconstructed data of shape (batch_size, D).
        """
        if self.components is None:
            raise ValueError(
                "SVD components not computed yet. Call update() enough times first."
            )

        X_proj = X @ self.components.T  # (batch_size, k)

        if self.use_prior:
            shrink = self.eigenvalues / (self.eigenvalues + 1.0)
            X_proj = X_proj * shrink.unsqueeze(0)

        X_reconstructed = X_proj @ self.components

        if self.normalize_output:
            input_dim = X_reconstructed.shape[-1]
            scale_factor = (self.eigenvalues.sum() / input_dim) ** 0.5
            X_reconstructed = X_reconstructed / scale_factor

        return X_reconstructed
