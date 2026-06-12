"""
Poisson Problem with Highly Localized Source Term
==================================================
∂²u/∂x² + ∂²u/∂y² = f(x,y),   (x,y) ∈ (0,1)²
with Dirichlet boundary conditions.

Exact solution: u(x,y) = 1 + (y² + 10³) exp(−500((x−0.5)² + (y−0.5)²))

The highly localized exponential creates extreme multi-magnitude loss
terms, testing the method's ability to handle localized phenomena.

From AW-PINN paper, Section 4.2.
"""

import torch
import numpy as np
from scipy.stats import qmc

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problems.base import BaseProblem
from core.wavelet_matrices import (build_wavelet_family_2d,
                                    WaveletMatrixBuilder)
from core.loss_functions import PINNLoss


class PoissonProblem(BaseProblem):
    """
    2D Poisson equation with a highly localized source.

    Parameters
    ----------
    sigma : float
        Width of the localized Gaussian source (smaller = harder).
    n_coll : int
        Number of collocation points.
    n_bc : int
        Number of boundary condition points per edge.
    n_test : int
        Grid resolution for test evaluation.
    Jx_range, Jy_range : tuple
        Scale ranges for wavelet family.
    gamma : float
        Translation hyperparameter.
    device : str
    """

    def __init__(self, sigma: float = 500.0,
                 n_coll: int = 10000, n_bc: int = 250,
                 n_val: int = 1000, n_test: int = 200,
                 Jx_range: tuple = (-4, 6),
                 Jy_range: tuple = (-4, 6),
                 gamma: float = 0.5,
                 device: str = 'cpu'):
        super().__init__(device)
        self.sigma = sigma
        self._n_coll = n_coll
        self.n_bc = n_bc
        self.n_val = n_val
        self.n_test = n_test
        self.Jx_range = Jx_range
        self.Jy_range = Jy_range
        self.gamma = gamma

        # Domain
        self.x_lower, self.x_upper = 0.0, 1.0
        self.y_lower, self.y_upper = 0.0, 1.0

        self._generate_points()

    @property
    def input_dim(self):
        return 2

    @property
    def n_collocation(self):
        return self._n_coll

    def analytical_solution(self, x, y):
        """u(x,y) = 1 + (y² + 10³) exp(-σ((x-0.5)² + (y-0.5)²))"""
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return 1.0 + (y ** 2 + 1e3) * torch.exp(-self.sigma * r2)

    def _source_term(self, x, y):
        """Compute f = Δu analytically."""
        s = self.sigma
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        e = torch.exp(-s * r2)
        ym = y ** 2 + 1e3

        # u_xx
        u_xx = e * (-2 * s * ym
                     + 4 * s ** 2 * (x - 0.5) ** 2 * ym)
        # u_yy
        u_yy = e * (2
                     - 2 * s * ym
                     - 8 * s * y * (y - 0.5)  # cross term from chain rule (2 * 2y * -2s(y-0.5))
                     + 4 * s ** 2 * (y - 0.5) ** 2 * ym)

        # Actually let's compute this more carefully with autograd for correctness
        # and use it as the RHS
        return u_xx + u_yy

    def _source_term_autograd(self, x, y):
        """Compute f = Δu using autograd for correctness."""
        x_ad = x.clone().requires_grad_(True)
        y_ad = y.clone().requires_grad_(True)
        u = self.analytical_solution(x_ad, y_ad)

        # u_x
        u_x = torch.autograd.grad(u.sum(), x_ad, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x_ad, create_graph=True)[0]

        # u_y
        u_y = torch.autograd.grad(u.sum(), y_ad, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y_ad, create_graph=True)[0]

        return (u_xx + u_yy).detach()

    def _generate_points(self):
        """Generate all training points."""
        sampler = qmc.Sobol(d=2, scramble=True, seed=501)
        sobol_coll = sampler.random(n=self._n_coll)
        sobol_bc = sampler.random(n=self.n_bc)

        self.x_coll = torch.tensor(
            sobol_coll[:, 0] * (self.x_upper - self.x_lower) + self.x_lower,
            dtype=torch.float32
        ).to(self.device)
        self.y_coll = torch.tensor(
            sobol_coll[:, 1] * (self.y_upper - self.y_lower) + self.y_lower,
            dtype=torch.float32
        ).to(self.device)

        # Boundary points (4 edges)
        x_bc = torch.tensor(
            sobol_bc[:, 0] * (self.x_upper - self.x_lower) + self.x_lower,
            dtype=torch.float32
        ).to(self.device)
        y_bc = torch.tensor(
            sobol_bc[:, 1] * (self.y_upper - self.y_lower) + self.y_lower,
            dtype=torch.float32
        ).to(self.device)

        self.x_bc_left = self.x_lower * torch.ones_like(y_bc)
        self.x_bc_right = self.x_upper * torch.ones_like(y_bc)
        self.y_bc_bottom = self.y_lower * torch.ones_like(x_bc)
        self.y_bc_top = self.y_upper * torch.ones_like(x_bc)
        self.x_bc = x_bc
        self.y_bc = y_bc

        # BC exact values
        self.u_bc_left = self.analytical_solution(self.x_bc_left, self.y_bc)
        self.u_bc_right = self.analytical_solution(self.x_bc_right, self.y_bc)
        self.u_bc_bottom = self.analytical_solution(self.x_bc, self.y_bc_bottom)
        self.u_bc_top = self.analytical_solution(self.x_bc, self.y_bc_top)

        # RHS (use autograd for accuracy)
        self.rhs = self._source_term_autograd(self.x_coll, self.y_coll)

        # Validation
        self.x_val = (torch.rand(self.n_val)
                      * (self.x_upper - self.x_lower) + self.x_lower)
        self.y_val = (torch.rand(self.n_val)
                      * (self.y_upper - self.y_lower) + self.y_lower)
        self.u_val_exact = self.analytical_solution(self.x_val, self.y_val)

        # Test grid
        x_lin = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        y_lin = torch.linspace(self.y_lower, self.y_upper, self.n_test)
        xg, yg = torch.meshgrid(x_lin, y_lin, indexing='ij')
        self.x_test = xg.reshape(-1)
        self.y_test = yg.reshape(-1)
        self.u_test_exact = (self.analytical_solution(self.x_test, self.y_test)
                             .reshape(self.n_test, self.n_test).numpy())

    def build_family(self):
        return build_wavelet_family_2d(
            domain_x=(self.x_lower, self.x_upper),
            domain_y=(self.y_lower, self.y_upper),
            Jx_range=self.Jx_range,
            Jy_range=self.Jy_range,
            gamma=self.gamma
        )

    def get_coordinates(self):
        return {
            'x_collocation': self.x_coll,
            'y_collocation': self.y_coll,
        }

    def get_loss_function(self, pruned=False):
        rhs = self.rhs
        u_bc_left = self.u_bc_left
        u_bc_right = self.u_bc_right
        u_bc_bottom = self.u_bc_bottom
        u_bc_top = self.u_bc_top

        def loss_fn(c, bias, matrices):
            W = matrices['W']
            DW2x = matrices['D2Wx']
            DW2y = matrices['D2Wy']

            u = torch.mv(W, c) + bias
            u_xx = torch.mv(DW2x, c)
            u_yy = torch.mv(DW2y, c)

            pde_loss = torch.mean((u_xx + u_yy - rhs) ** 2)

            # BC at all 4 edges
            bc_loss = (torch.mean((torch.mv(
                           matrices.get('W_bc_left', W[:len(u_bc_left)]), c
                       ) + bias - u_bc_left) ** 2)
                       + torch.mean((torch.mv(
                           matrices.get('W_bc_right', W[:len(u_bc_right)]), c
                       ) + bias - u_bc_right) ** 2)
                       + torch.mean((torch.mv(
                           matrices.get('W_bc_bottom', W[:len(u_bc_bottom)]), c
                       ) + bias - u_bc_bottom) ** 2)
                       + torch.mean((torch.mv(
                           matrices.get('W_bc_top', W[:len(u_bc_top)]), c
                       ) + bias - u_bc_top) ** 2))

            total = pde_loss + bc_loss
            return total, pde_loss, bc_loss

        return loss_fn

    def get_full_loss_function(self, meta_wavelet, family, device='cpu'):
        """Self-contained loss with meta-wavelet matrix building."""
        rhs = self.rhs.to(device)
        u_bc_left = self.u_bc_left.to(device)
        u_bc_right = self.u_bc_right.to(device)
        u_bc_bottom = self.u_bc_bottom.to(device)
        u_bc_top = self.u_bc_top.to(device)

        x_coll = self.x_coll.to(device)
        y_coll = self.y_coll.to(device)
        x_bc_left = self.x_bc_left.to(device)
        x_bc_right = self.x_bc_right.to(device)
        y_bc_bottom = self.y_bc_bottom.to(device)
        y_bc_top = self.y_bc_top.to(device)
        x_bc = self.x_bc.to(device)
        y_bc = self.y_bc.to(device)

        fam = family.to(device)
        jx, jy, kx, ky = fam[:, 0], fam[:, 1], fam[:, 2], fam[:, 3]

        def loss_fn(c, bias):
            W = meta_wavelet.evaluate_basis_2d(x_coll, y_coll, jx, jy, kx, ky)
            DW2x = meta_wavelet.evaluate_basis_2d_dx(
                x_coll, y_coll, jx, jy, kx, ky, order=2
            )
            DW2y = meta_wavelet.evaluate_basis_2d_dy(
                x_coll, y_coll, jx, jy, kx, ky, order=2
            )

            u_xx = torch.mv(DW2x, c)
            u_yy = torch.mv(DW2y, c)
            pde_loss = torch.mean((u_xx + u_yy - rhs) ** 2)

            # BC
            W_l = meta_wavelet.evaluate_basis_2d(
                x_bc_left, y_bc, jx, jy, kx, ky
            )
            W_r = meta_wavelet.evaluate_basis_2d(
                x_bc_right, y_bc, jx, jy, kx, ky
            )
            W_b = meta_wavelet.evaluate_basis_2d(
                x_bc, y_bc_bottom, jx, jy, kx, ky
            )
            W_t = meta_wavelet.evaluate_basis_2d(
                x_bc, y_bc_top, jx, jy, kx, ky
            )

            bc_loss = (
                torch.mean((torch.mv(W_l, c) + bias - u_bc_left) ** 2)
                + torch.mean((torch.mv(W_r, c) + bias - u_bc_right) ** 2)
                + torch.mean((torch.mv(W_b, c) + bias - u_bc_bottom) ** 2)
                + torch.mean((torch.mv(W_t, c) + bias - u_bc_top) ** 2)
            )

            return pde_loss + bc_loss, pde_loss, bc_loss

        return loss_fn

    def evaluate(self, model, family, device='cpu'):
        family = family.to(device)
        jx, jy = family[:, 0], family[:, 1]
        kx, ky = family[:, 2], family[:, 3]

        with torch.no_grad():
            c, bias = model()
            W_val = model.meta_wavelet.evaluate_basis_2d(
                self.x_val.to(device), self.y_val.to(device),
                jx, jy, kx, ky
            )
            u_val_pred = torch.mv(W_val, c) + bias

        rel_l2 = PINNLoss.relative_l2_error(
            u_val_pred.cpu(), self.u_val_exact
        ).item()
        max_err = PINNLoss.max_error(
            u_val_pred.cpu(), self.u_val_exact
        ).item()

        return {'rel_l2_error': rel_l2, 'max_error': max_err}
