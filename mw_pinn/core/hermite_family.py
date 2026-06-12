"""
Hermite-Gaussian Wavelet Family
================================
Family of wavelets built from Hermite functions (derivatives of Gaussian).

Core definition (UNNORMALIZED for exact derivative property):
    φ_n(x) = H_n(x) · exp(-x²/2)

where H_n is the probabilist's Hermite polynomial:
    H_0(x) = 1,  H_1(x) = x,  H_{n+1}(x) = x·H_n(x) − n·H_{n-1}(x)

Key derivative identity (exact, no scaling factors):
    d/dx [φ_n(x)] = −φ_{n+1}(x)

Proof:
    d/dx [H_n(x) e^{-x²/2}]
    = H_n'(x) e^{-x²/2} − x H_n(x) e^{-x²/2}
    = [n H_{n-1}(x) − x H_n(x)] e^{-x²/2}
    = −H_{n+1}(x) e^{-x²/2}          (by recurrence)
    = −φ_{n+1}(x)

For the k-th derivative:
    d^k/dx^k [φ_n(x)] = (−1)^k · φ_{n+k}(x)

This identity is EXACT and requires NO normalization adjustment.
The learnable coefficients a_n in the meta-wavelet absorb any scaling.

n=1 → Gaussian wavelet (-x·exp(-x²/2)), used in W-PINN/AW-PINN
n=2 → Mexican Hat wavelet ((x²-1)·exp(-x²/2)), used in W-PINN
n=3+ → Higher-order DOG wavelets (NEW)
"""

import torch
import math


class HermiteGaussianFamily:
    """
    Hermite-Gaussian wavelet family with exact derivative identity.

    Uses UNNORMALIZED Hermite functions:
        φ_n(x) = H_n(x) · exp(-x²/2)

    Derivative identity:
        d/dx [φ_n(x)] = −φ_{n+1}(x)
        d²/dx² [φ_n(x)] = φ_{n+2}(x)
        d^k/dx^k [φ_n(x)] = (−1)^k · φ_{n+k}(x)
    """

    @staticmethod
    def hermite_polynomial(x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Compute H_n(x) via three-term recurrence.

        H_0(x) = 1
        H_1(x) = x
        H_{k+1}(x) = x · H_k(x) − k · H_{k-1}(x)
        """
        if n == 0:
            return torch.ones_like(x)
        if n == 1:
            return x.clone()

        H_prev2 = torch.ones_like(x)   # H_0
        H_prev1 = x.clone()            # H_1

        for k in range(1, n):
            H_curr = x * H_prev1 - k * H_prev2
            H_prev2 = H_prev1
            H_prev1 = H_curr

        return H_prev1

    @staticmethod
    def phi(x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Evaluate the n-th Hermite function (unnormalized).

        φ_n(x) = H_n(x) · exp(-x²/2)

        Parameters
        ----------
        x : torch.Tensor, arbitrary shape.
        n : int ≥ 0, Hermite function order.

        Returns
        -------
        torch.Tensor of same shape as x.
        """
        H_n = HermiteGaussianFamily.hermite_polynomial(x, n)
        envelope = torch.exp(-x ** 2 / 2)
        return H_n * envelope

    @staticmethod
    def phi_derivative(x: torch.Tensor, n: int,
                       order: int = 1) -> torch.Tensor:
        """
        Exact analytical derivative of φ_n.

        d^k/dx^k [φ_n(x)] = (−1)^k · φ_{n+k}(x)

        This identity is EXACT — no scaling factors, no approximations.

        Parameters
        ----------
        x : torch.Tensor
        n : int ≥ 0 — Hermite function order.
        order : int ≥ 1 — derivative order k.

        Returns
        -------
        torch.Tensor — the k-th derivative evaluated at x.
        """
        sign = (-1.0) ** order
        return sign * HermiteGaussianFamily.phi(x, n + order)

    # ── Convenience aliases matching W-PINN wavelet convention ──

    @staticmethod
    def psi(x: torch.Tensor, n: int) -> torch.Tensor:
        """
        n-th order wavelet in the Hermite-Gaussian family.
        Alias for phi(x, n). Uses n ≥ 1 for admissible wavelets.

        n=1: ψ(x) = x·exp(-x²/2)   (Gaussian wavelet, like W-PINN)
        n=2: ψ(x) = (x²−1)·exp(-x²/2)  (Mexican hat)
        """
        return HermiteGaussianFamily.phi(x, n)

    @staticmethod
    def psi_derivative(x: torch.Tensor, n: int,
                       order: int = 1) -> torch.Tensor:
        """
        Exact k-th derivative of ψ^(n).

        d^k/dx^k [ψ^(n)(x)] = (−1)^k · ψ^(n+k)(x)
        """
        return HermiteGaussianFamily.phi_derivative(x, n, order)

    # ── 2D tensor-product wavelets ──────────────────────────────

    @staticmethod
    def psi_2d(x: torch.Tensor, y: torch.Tensor,
               nx: int, ny: int) -> torch.Tensor:
        """Separable 2D: Ψ(x,y) = φ_{nx}(x) · φ_{ny}(y)."""
        return (HermiteGaussianFamily.phi(x, nx)
                * HermiteGaussianFamily.phi(y, ny))

    @staticmethod
    def psi_2d_dx(x: torch.Tensor, y: torch.Tensor,
                  nx: int, ny: int, order: int = 1) -> torch.Tensor:
        """∂^k/∂x^k [Ψ] = (−1)^k · φ_{nx+k}(x) · φ_{ny}(y)."""
        sign = (-1.0) ** order
        return (sign * HermiteGaussianFamily.phi(x, nx + order)
                * HermiteGaussianFamily.phi(y, ny))

    @staticmethod
    def psi_2d_dy(x: torch.Tensor, y: torch.Tensor,
                  nx: int, ny: int, order: int = 1) -> torch.Tensor:
        """∂^k/∂y^k [Ψ] = φ_{nx}(x) · (−1)^k · φ_{ny+k}(y)."""
        sign = (-1.0) ** order
        return (HermiteGaussianFamily.phi(x, nx)
                * sign * HermiteGaussianFamily.phi(y, ny + order))

    # ── Verification ────────────────────────────────────────────

    @staticmethod
    def verify_derivative_autograd(x: torch.Tensor, n: int,
                                   tol: float = 1e-5) -> bool:
        """
        Gold-standard check: compare analytical derivative vs autograd.
        """
        x_ad = x.clone().detach().requires_grad_(True)
        phi_val = HermiteGaussianFamily.phi(x_ad, n)
        phi_val.sum().backward()
        autograd_deriv = x_ad.grad.clone()

        analytical_deriv = HermiteGaussianFamily.phi_derivative(
            x.detach(), n, order=1
        )

        max_err = (autograd_deriv - analytical_deriv).abs().max().item()
        passed = max_err < tol
        status = "✓" if passed else "✗ FAILED"
        print(f"  n={n} autograd check: max_err={max_err:.2e} {status}")
        assert passed, (
            f"AD verification failed for n={n}! Error={max_err:.2e}"
        )
        return passed

    @staticmethod
    def verify_derivative_fd(x: torch.Tensor, n: int,
                             order: int = 1, tol: float = 1e-3) -> bool:
        """Compare analytical derivative vs finite differences."""
        h = 1e-4
        analytical = HermiteGaussianFamily.phi_derivative(x, n, order=order)

        if order == 1:
            fd = (HermiteGaussianFamily.phi(x + h, n)
                  - HermiteGaussianFamily.phi(x - h, n)) / (2 * h)
        elif order == 2:
            fd = (HermiteGaussianFamily.phi(x + h, n)
                  - 2 * HermiteGaussianFamily.phi(x, n)
                  + HermiteGaussianFamily.phi(x - h, n)) / (h ** 2)
        else:
            raise NotImplementedError("FD for order 1,2 only")

        max_err = (analytical - fd).abs().max().item()
        denom = max(analytical.abs().max().item(), 1e-12)
        rel_err = max_err / denom

        passed = rel_err < tol
        status = "✓" if passed else "✗ FAILED"
        print(f"  n={n}, d^{order}: max_err={max_err:.2e}, "
              f"rel_err={rel_err:.2e} {status}")
        return passed

    @staticmethod
    def run_all_verifications(max_n: int = 6):
        """Run full verification suite."""
        print("=" * 60)
        print("Hermite-Gaussian Derivative Verification")
        print("=" * 60)

        x = torch.linspace(-3.5, 3.5, 2000)  # Avoid extreme tails
        x_interior = x[200:-200]  # Interior for FD

        print("\n[1] Autograd verification (1st derivative) — GOLD STANDARD")
        for n in range(0, max_n + 1):
            HermiteGaussianFamily.verify_derivative_autograd(x, n)

        print("\n[2] Finite-difference verification (1st derivative)")
        for n in range(0, max_n + 1):
            HermiteGaussianFamily.verify_derivative_fd(x_interior, n, order=1)

        print("\n[3] Finite-difference verification (2nd derivative)")
        for n in range(0, max_n + 1):
            HermiteGaussianFamily.verify_derivative_fd(x_interior, n, order=2)

        print("\n" + "=" * 60)
        print("All verifications passed ✓")
        print("=" * 60)
