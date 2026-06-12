"""
AW-PINN: Adaptive Wavelet PINN (Baseline 2)
=============================================
Reimplementation of the AW-PINN from Pandey et al. (2026b)
for fair comparison with Meta-Wavelet PINN.

Key differences from W-PINN:
  1. After pre-training, selects active wavelet families via
     similarity scores + top-κ coefficient magnitudes.
  2. Makes scale (w) and translation (b) parameters trainable
     in the adaptive phase.
  3. Uses ψ(·) as activation function, enabling continuous
     adaptation of basis position/scale.

Training protocol:
  Phase 1 (W-PINN pre-training): Adam, ~5000 epochs
  Phase 2 (Adaptive refinement): L-BFGS until convergence
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class AdaptiveWaveletUnit(nn.Module):
    """
    Single adaptive wavelet unit for d-dimensional input.

    W_i(x; θ_i) = Π_{n=1}^{d} ψ(w_{i,n} · x_n + b_{i,n})

    where ψ is the Gaussian wavelet: ψ(z) = -z · exp(-z²/2)

    Parameters
    ----------
    dim : int — input dimension.
    init_scale : [dim] tensor — initial w values (2^j from W-PINN).
    init_translate : [dim] tensor — initial b values (-k from W-PINN).
    """

    def __init__(self, dim: int, init_scale: torch.Tensor,
                 init_translate: torch.Tensor):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(init_scale.clone().float())
        self.b = nn.Parameter(init_translate.clone().float())

    @staticmethod
    def psi(z: torch.Tensor) -> torch.Tensor:
        """Gaussian wavelet: ψ(z) = -z · exp(-z²/2)"""
        return -z * torch.exp(-z ** 2 / 2)

    @staticmethod
    def psi_prime(z: torch.Tensor) -> torch.Tensor:
        """ψ'(z) = (z² - 1) · exp(-z²/2)"""
        return (z ** 2 - 1) * torch.exp(-z ** 2 / 2)

    @staticmethod
    def psi_double_prime(z: torch.Tensor) -> torch.Tensor:
        """ψ''(z) = z(3 - z²) · exp(-z²/2)"""
        return z * (3 - z ** 2) * torch.exp(-z ** 2 / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate wavelet unit.

        Parameters
        ----------
        x : [batch, dim]

        Returns
        -------
        [batch] — product of 1D wavelets.
        """
        z = self.w * x + self.b  # [batch, dim]
        return torch.prod(self.psi(z), dim=-1)

    def derivative_wrt_xm(self, x: torch.Tensor, m: int,
                           order: int = 1) -> torch.Tensor:
        """
        ∂^k/∂x_m^k W_i(x) = w_{i,m}^k · ψ^(k)(z_m) · Π_{n≠m} ψ(z_n)

        Parameters
        ----------
        x : [batch, dim]
        m : int — dimension index (0-based).
        order : int — derivative order.

        Returns
        -------
        [batch]
        """
        z = self.w * x + self.b  # [batch, dim]

        # Product of ψ over all dimensions except m
        psi_vals = self.psi(z)  # [batch, dim]
        product_others = torch.ones(x.shape[0], device=x.device)
        for n in range(self.dim):
            if n != m:
                product_others = product_others * psi_vals[:, n]

        # k-th derivative of ψ in dimension m
        z_m = z[:, m]
        if order == 1:
            deriv_m = self.psi_prime(z_m)
        elif order == 2:
            deriv_m = self.psi_double_prime(z_m)
        else:
            raise NotImplementedError(f"Order {order} not implemented")

        scale_factor = self.w[m] ** order
        return scale_factor * deriv_m * product_others


class AWPINN(nn.Module):
    """
    Adaptive Wavelet PINN.

    After W-PINN pre-training, selected wavelet families become
    adaptive units with trainable scale/translation parameters.

    Parameters
    ----------
    n_active : int
        Number of active (selected) wavelet families.
    dim : int
        Spatial dimension.
    init_coeffs : [n_active] tensor — coefficients from W-PINN.
    init_scales : [n_active, dim] tensor — scales (2^j) from family.
    init_translates : [n_active, dim] tensor — translations (-k).
    """

    def __init__(self, n_active: int, dim: int,
                 init_coeffs: torch.Tensor,
                 init_scales: torch.Tensor,
                 init_translates: torch.Tensor):
        super().__init__()
        self.n_active = n_active
        self.dim = dim

        # Adaptive wavelet units
        self.wavelets = nn.ModuleList([
            AdaptiveWaveletUnit(dim, init_scales[i], init_translates[i])
            for i in range(n_active)
        ])

        # Linear coefficients
        self.coeffs = nn.Parameter(init_coeffs.clone().float())
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute solution approximation.

        u_hat(x) = Σ c_i · W_i(x; θ_i) + bias

        Parameters
        ----------
        x : [batch, dim]

        Returns
        -------
        [batch]
        """
        result = torch.zeros(x.shape[0], device=x.device)
        for i, wavelet in enumerate(self.wavelets):
            result = result + self.coeffs[i] * wavelet(x)
        return result + self.bias

    def compute_derivative(self, x: torch.Tensor, m: int,
                           order: int = 1) -> torch.Tensor:
        """
        Compute ∂^k u_hat / ∂x_m^k analytically.

        No autograd needed — uses analytical wavelet derivatives.

        Parameters
        ----------
        x : [batch, dim]
        m : int — dimension index.
        order : int — derivative order.

        Returns
        -------
        [batch]
        """
        result = torch.zeros(x.shape[0], device=x.device)
        for i, wavelet in enumerate(self.wavelets):
            result = result + self.coeffs[i] * wavelet.derivative_wrt_xm(
                x, m, order
            )
        return result

    @classmethod
    def from_wpinn_pretrain(cls, coefficients: torch.Tensor,
                            family: torch.Tensor,
                            active_indices: torch.Tensor,
                            dim: int):
        """
        Factory method: build AW-PINN from W-PINN pre-training results.

        Parameters
        ----------
        coefficients : [F] all coefficients from W-PINN.
        family : [F, 2*dim] family parameters (scales, translates).
        active_indices : [N_A] selected family indices.
        dim : int — spatial dimension.

        Returns
        -------
        AWPINN instance.
        """
        active_coeffs = coefficients[active_indices].detach()
        active_family = family[active_indices].detach()

        n_active = len(active_indices)

        if dim == 1:
            scales = active_family[:, 0:1]       # [N_A, 1]
            translates = -active_family[:, 1:2]   # [N_A, 1] — note negation
        elif dim == 2:
            scales = active_family[:, 0:2]       # [N_A, 2]
            translates = -active_family[:, 2:4]  # [N_A, 2]
        else:
            raise ValueError(f"dim={dim} not supported")

        return cls(n_active, dim, active_coeffs, scales, translates)
