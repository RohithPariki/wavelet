"""
Wavelet Matrix Builder
=======================
Precomputes wavelet basis matrices W, D1W, D2W for a given set of
collocation points and wavelet family parameters.

These are the non-trainable reconstruction matrices from the W-PINN paper.
In MW-PINN, we replace the fixed wavelet evaluations with the meta-wavelet,
but the matrix structure remains the same.
"""

import torch
import math
from scipy.stats import qmc


def build_wavelet_family_1d(domain: tuple, J_range: tuple,
                            gamma: float = 0.5) -> torch.Tensor:
    """
    Build the 1D wavelet family index set.

    For each scale j ∈ J_range, translations k span:
        k ∈ [⌊(a − γ)·2^j⌋, ⌈(b + γ)·2^j⌉]

    Parameters
    ----------
    domain : (lower, upper)
    J_range : (j_min, j_max) — resolution levels as integers.
    gamma : float — translation hyperparameter.

    Returns
    -------
    family : [F, 2] tensor with columns [2^j, k]
    """
    a, b = domain
    j_min, j_max = J_range

    entries = []
    for j in range(j_min, j_max):
        scale = 2.0 ** j
        k_lo = int(math.floor((a - gamma) * scale))
        k_hi = int(math.ceil((b + gamma) * scale))
        for k in range(k_lo, k_hi + 1):
            entries.append([scale, float(k)])

    family = torch.tensor(entries, dtype=torch.float32)
    return family


def build_wavelet_family_2d(domain_x: tuple, domain_y: tuple,
                            Jx_range: tuple, Jy_range: tuple,
                            gamma: float = 0.5) -> torch.Tensor:
    """
    Build the 2D wavelet family index set (tensor product).

    Returns
    -------
    family : [F, 4] tensor with columns [2^jx, 2^jy, kx, ky]
    """
    ax, bx = domain_x
    ay, by = domain_y
    jx_min, jx_max = Jx_range
    jy_min, jy_max = Jy_range

    entries = []
    for jx in range(jx_min, jx_max):
        sx = 2.0 ** jx
        kx_lo = int(math.floor((ax - gamma) * sx))
        kx_hi = int(math.ceil((bx + gamma) * sx))
        for jy in range(jy_min, jy_max):
            sy = 2.0 ** jy
            ky_lo = int(math.floor((ay - gamma) * sy))
            ky_hi = int(math.ceil((by + gamma) * sy))
            for kx in range(kx_lo, kx_hi + 1):
                for ky in range(ky_lo, ky_hi + 1):
                    entries.append([sx, sy, float(kx), float(ky)])

    family = torch.tensor(entries, dtype=torch.float32)
    return family


class WaveletMatrixBuilder:
    """
    Builds wavelet basis matrices for a given meta-wavelet and family.

    Given collocation points {x_i} and family {(j_f, k_f)}, builds:
        W[i,f]   = ψ_θ(j_f · x_i − k_f)
        D1W[i,f] = j_f · ψ_θ'(j_f · x_i − k_f)
        D2W[i,f] = j_f² · ψ_θ''(j_f · x_i − k_f)
    """

    def __init__(self, meta_wavelet, device='cpu'):
        """
        Parameters
        ----------
        meta_wavelet : MetaWavelet instance (or None for fixed wavelets).
        device : torch device.
        """
        self.meta_wavelet = meta_wavelet
        self.device = device

    # ── 1D matrices ────────────────────────────────────────────

    def build_1d(self, x: torch.Tensor, family: torch.Tensor):
        """
        Build W, D1W, D2W for 1D problem.

        Parameters
        ----------
        x : [N] collocation points.
        family : [F, 2] with columns [scale, translate].

        Returns
        -------
        W    : [N, F]
        D1W  : [N, F] first derivative matrix
        D2W  : [N, F] second derivative matrix
        """
        scales = family[:, 0].to(self.device)
        translates = family[:, 1].to(self.device)
        x = x.to(self.device)

        W = self.meta_wavelet.evaluate_basis(x, scales, translates)
        D1W = self.meta_wavelet.evaluate_basis_deriv(x, scales, translates, order=1)
        D2W = self.meta_wavelet.evaluate_basis_deriv(x, scales, translates, order=2)

        return W, D1W, D2W

    # ── 2D matrices ────────────────────────────────────────────

    def build_2d(self, x: torch.Tensor, y: torch.Tensor,
                 family: torch.Tensor):
        """
        Build 2D wavelet matrices.

        Parameters
        ----------
        x, y : [N] collocation points.
        family : [F, 4] with columns [jx, jy, kx, ky].

        Returns
        -------
        W     : [N, F] — basis values
        D1Wx  : [N, F] — ∂/∂x
        D2Wx  : [N, F] — ∂²/∂x²
        D1Wy  : [N, F] — ∂/∂y
        D2Wy  : [N, F] — ∂²/∂y²
        """
        jx = family[:, 0].to(self.device)
        jy = family[:, 1].to(self.device)
        kx = family[:, 2].to(self.device)
        ky = family[:, 3].to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)

        W = self.meta_wavelet.evaluate_basis_2d(x, y, jx, jy, kx, ky)
        D1Wx = self.meta_wavelet.evaluate_basis_2d_dx(x, y, jx, jy, kx, ky, order=1)
        D2Wx = self.meta_wavelet.evaluate_basis_2d_dx(x, y, jx, jy, kx, ky, order=2)
        D1Wy = self.meta_wavelet.evaluate_basis_2d_dy(x, y, jx, jy, kx, ky, order=1)
        D2Wy = self.meta_wavelet.evaluate_basis_2d_dy(x, y, jx, jy, kx, ky, order=2)

        return W, D1Wx, D2Wx, D1Wy, D2Wy


def generate_sobol_points_1d(n: int, domain: tuple, seed: int = 501):
    """Generate quasi-random Sobol sequence points in 1D."""
    a, b = domain
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    pts = sampler.random(n=n)
    pts = torch.tensor(pts[:, 0] * (b - a) + a, dtype=torch.float32)
    return pts


def generate_sobol_points_2d(n: int, domain_x: tuple, domain_y: tuple,
                             seed: int = 501):
    """Generate quasi-random Sobol sequence points in 2D."""
    ax, bx = domain_x
    ay, by = domain_y
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    pts = sampler.random(n=n)
    x = torch.tensor(pts[:, 0] * (bx - ax) + ax, dtype=torch.float32)
    y = torch.tensor(pts[:, 1] * (by - ay) + ay, dtype=torch.float32)
    return x, y
