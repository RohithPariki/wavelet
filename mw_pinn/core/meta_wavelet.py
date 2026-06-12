"""
Meta-Wavelet Module
====================
Learnable mother wavelet: ψ_θ(x) = Σ_{n=1}^{N_H} a_n · φ_n(x)

where φ_n(x) = H_n(x)·exp(-x²/2) are unnormalized Hermite functions.

The key innovation: instead of fixing the wavelet to Gaussian or Mexican Hat,
we learn the optimal linear combination of Hermite-Gaussian wavelets.

The a_n coefficients are trained jointly with the PINN, allowing the model
to discover the problem-optimal wavelet shape automatically.

Derivative identity (exact, no approximation):
    d^k/dx^k [ψ_θ(x)] = Σ a_n · (-1)^k · φ_{n+k}(x)

Fully analytical — no autograd needed at any point.
"""

import torch
import torch.nn as nn
import numpy as np

from .hermite_family import HermiteGaussianFamily


class MetaWavelet(nn.Module):
    """
    Learnable mother wavelet parameterized as a linear combination
    of Hermite-Gaussian basis functions.

    ψ_θ(x) = Σ_{n=1}^{N_H} a_n · ψ^(n)(x)

    Parameters
    ----------
    N_H : int
        Number of Hermite-Gaussian components (ablation variable).
    init_type : str
        Initialization strategy:
        - 'gaussian'    : a_1=1, rest=0 (start as Gaussian wavelet)
        - 'mexican_hat' : a_2=1, rest=0 (start as Mexican Hat)
        - 'uniform'     : equal weights 1/N_H
        - 'random'      : small random perturbation
    normalize_coeffs : bool
        If True, softmax-normalize a_n during forward so they sum to 1
        and stay in a convex hull of basis wavelets. Generally False
        gives better results — let gradient figure out the magnitudes.
    """

    def __init__(self, N_H: int = 4, init_type: str = 'uniform',
                 normalize_coeffs: bool = False):
        super().__init__()
        self.N_H = N_H
        self.normalize_coeffs = normalize_coeffs
        self.hg = HermiteGaussianFamily()

        # The key learnable parameters
        a_init = self._get_initialization(N_H, init_type)
        self.a = nn.Parameter(a_init)

    def _get_initialization(self, N_H: int, init_type: str) -> torch.Tensor:
        a = torch.zeros(N_H)
        if init_type == 'gaussian':
            a[0] = 1.0
        elif init_type == 'mexican_hat':
            if N_H < 2:
                raise ValueError("N_H must be ≥ 2 for mexican_hat init")
            a[1] = 1.0
        elif init_type == 'uniform':
            a = torch.ones(N_H) / N_H
        elif init_type == 'random':
            a = torch.randn(N_H) * 0.1
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        return a

    def _effective_coeffs(self) -> torch.Tensor:
        """Return (possibly normalized) mixing coefficients."""
        if self.normalize_coeffs:
            return torch.softmax(self.a, dim=0)
        return self.a

    # ── 1D evaluation ──────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate meta-wavelet at points x.

        ψ_θ(x) = Σ_{n=1}^{N_H} a_n · ψ^(n)(x)

        Parameters
        ----------
        x : torch.Tensor, arbitrary shape.

        Returns
        -------
        torch.Tensor of same shape as x.
        """
        a = self._effective_coeffs()
        result = torch.zeros_like(x)
        for n in range(1, self.N_H + 1):
            result = result + a[n - 1] * self.hg.psi(x, n)
        return result

    def derivative(self, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """
        Analytical derivative of the meta-wavelet.

        d^k ψ_θ / dx^k = Σ_{n=1}^{N_H} a_n · (-1)^k · φ_{n+k}(x)

        Uses the exact identity: d^k/dx^k[φ_n] = (-1)^k · φ_{n+k}
        Never uses autograd. This is the core advantage.

        Parameters
        ----------
        x : torch.Tensor
        order : int ≥ 1 — derivative order.

        Returns
        -------
        torch.Tensor of same shape as x.
        """
        a = self._effective_coeffs()
        sign = (-1.0) ** order
        result = torch.zeros_like(x)
        for n in range(1, self.N_H + 1):
            result = result + a[n - 1] * sign * self.hg.phi(x, n + order)
        return result

    # ── 2D tensor-product evaluation ───────────────────────────

    def forward_2d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        2D separable meta-wavelet: ψ_θ(x) · ψ_θ(y)

        Both dimensions share the same learned wavelet shape.
        """
        return self.forward(x) * self.forward(y)

    def derivative_2d_x(self, x: torch.Tensor, y: torch.Tensor,
                        order: int = 1) -> torch.Tensor:
        """∂^k/∂x^k [ψ_θ(x)·ψ_θ(y)] = ψ_θ^(k)(x) · ψ_θ(y)"""
        return self.derivative(x, order) * self.forward(y)

    def derivative_2d_y(self, x: torch.Tensor, y: torch.Tensor,
                        order: int = 1) -> torch.Tensor:
        """∂^k/∂y^k [ψ_θ(x)·ψ_θ(y)] = ψ_θ(x) · ψ_θ^(k)(y)"""
        return self.forward(x) * self.derivative(y, order)

    # ── Scaled & translated evaluation ─────────────────────────

    def evaluate_basis(self, x: torch.Tensor,
                       scales: torch.Tensor,
                       translates: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a full family of scaled/translated meta-wavelets.

        Ψ_{j,k}(x) = ψ_θ(2^j · x − k)

        Parameters
        ----------
        x : [N] collocation points
        scales : [F] scale factors (2^j values)
        translates : [F] translation parameters (k values)

        Returns
        -------
        [N, F] matrix of basis function evaluations.
        """
        # z_{i,f} = scales[f] * x[i] - translates[f]
        z = scales[None, :] * x[:, None] - translates[None, :]  # [N, F]
        return self.forward(z)  # broadcasts through the wavelet

    def evaluate_basis_deriv(self, x: torch.Tensor,
                             scales: torch.Tensor,
                             translates: torch.Tensor,
                             order: int = 1) -> torch.Tensor:
        """
        Derivative of the scaled/translated basis:

        d^k/dx^k [Ψ_{j,k}(x)] = (2^j)^k · ψ_θ^(k)(2^j·x − k)

        Returns
        -------
        [N, F] matrix of derivative evaluations.
        """
        z = scales[None, :] * x[:, None] - translates[None, :]
        scale_factor = scales[None, :] ** order
        return scale_factor * self.derivative(z, order)

    # ── 2D scaled/translated basis ─────────────────────────────

    def evaluate_basis_2d(self, x: torch.Tensor, y: torch.Tensor,
                          jx: torch.Tensor, jy: torch.Tensor,
                          kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
        """
        Evaluate 2D tensor-product family:
        Ψ(x,y) = ψ_θ(jx·x − kx) · ψ_θ(jy·y − ky)

        Parameters
        ----------
        x, y : [N] collocation points
        jx, jy : [F] scale parameters per family member
        kx, ky : [F] translate parameters per family member

        Returns
        -------
        [N, F] matrix.
        """
        Zx = jx[None, :] * x[:, None] - kx[None, :]  # [N, F]
        Zy = jy[None, :] * y[:, None] - ky[None, :]   # [N, F]
        return self.forward(Zx) * self.forward(Zy)

    def evaluate_basis_2d_dx(self, x: torch.Tensor, y: torch.Tensor,
                             jx: torch.Tensor, jy: torch.Tensor,
                             kx: torch.Tensor, ky: torch.Tensor,
                             order: int = 1) -> torch.Tensor:
        """∂^k/∂x^k of the 2D basis. Chain rule gives factor jx^order."""
        Zx = jx[None, :] * x[:, None] - kx[None, :]
        Zy = jy[None, :] * y[:, None] - ky[None, :]
        scale_factor = jx[None, :] ** order
        return scale_factor * self.derivative(Zx, order) * self.forward(Zy)

    def evaluate_basis_2d_dy(self, x: torch.Tensor, y: torch.Tensor,
                             jx: torch.Tensor, jy: torch.Tensor,
                             kx: torch.Tensor, ky: torch.Tensor,
                             order: int = 1) -> torch.Tensor:
        """∂^k/∂y^k of the 2D basis. Chain rule gives factor jy^order."""
        Zx = jx[None, :] * x[:, None] - kx[None, :]
        Zy = jy[None, :] * y[:, None] - ky[None, :]
        scale_factor = jy[None, :] ** order
        return self.forward(Zx) * scale_factor * self.derivative(Zy, order)

    # ── Diagnostics ────────────────────────────────────────────

    def get_shape_description(self) -> dict:
        """
        After training, report what wavelet shape was learned.
        Used for the visualization figure in the paper.
        """
        a_vals = self.a.detach().cpu().numpy()
        names = ['Gaussian', 'MexicanHat', '3rd-DOG', '4th-DOG',
                 '5th-DOG', '6th-DOG', '7th-DOG', '8th-DOG']
        dominant_idx = int(np.abs(a_vals).argmax())
        dominant = names[dominant_idx] if dominant_idx < len(names) else f'{dominant_idx+1}th-DOG'

        return {
            'coefficients': a_vals.copy(),
            'dominant_component': dominant,
            'dominant_index': dominant_idx + 1,
            'effective_vanishing_moments': dominant_idx + 1,
            'sparsity': float((np.abs(a_vals) < 1e-6).mean()),
        }

    def verify_derivative(self, tol: float = 1e-4) -> bool:
        """
        Verify that the meta-wavelet's analytical derivative matches autograd.
        """
        x = torch.linspace(-4, 4, 1000)
        x_ad = x.clone().requires_grad_(True)

        # Forward with autograd graph
        psi_val = self.forward(x_ad)
        psi_val.sum().backward()
        autograd_d1 = x_ad.grad.clone()

        # Analytical
        with torch.no_grad():
            analytical_d1 = self.derivative(x, order=1)

        max_err = (autograd_d1 - analytical_d1).abs().max().item()
        passed = max_err < tol
        status = "✓" if passed else "✗ FAILED"
        print(f"MetaWavelet derivative check: max_err={max_err:.2e} {status}")
        return passed
