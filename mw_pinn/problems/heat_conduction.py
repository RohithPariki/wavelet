"""
Heat Conduction Problem — Primary Benchmark
=============================================
∂u/∂t = ∂²u/∂x² + h(x,t),   (x,t) ∈ (-1,1) × (0,1)
u(x,0) = (1-x²) exp(1/(1+ε)),  x ∈ (-1,1)
u(-1,t) = 0,  u(1,t) = 0,      t ∈ (0,1]

Exact solution: u(x,t) = (1-x²) exp(1/((2t-1)² + ε))

For small ε, the source term and solution exhibit transient behavior
with an extreme gradient near t=0.5. The loss ratio reaches
L_b : L_i : L_r = 1 : 10 : 10⁹ at ε=0.1, making this extremely
challenging for standard PINNs.

This is the primary benchmark used in both the W-PINN and AW-PINN papers.
"""

import torch
import numpy as np
from scipy.stats import qmc

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problems.base import BaseProblem
from core.wavelet_matrices import (build_wavelet_family_2d,
                                    WaveletMatrixBuilder,
                                    generate_sobol_points_2d)
from core.loss_functions import PINNLoss


class HeatConductionProblem(BaseProblem):
    """
    Heat conduction with extreme heat source.

    Parameters
    ----------
    epsilon : float
        Controls sharpness of transient behavior.
        ε=0.15 (mild), ε=0.12, ε=0.11, ε=0.10 (extreme).
    n_coll : int
        Number of collocation points.
    n_bc : int
        Number of boundary/initial condition points.
    n_test : int
        Grid resolution for test evaluation.
    Jx_range, Jt_range : tuple
        Scale ranges for wavelet family.
    gamma : float
        Translation hyperparameter.
    device : str
    """

    def __init__(self, epsilon: float = 0.15,
                 n_coll: int = 10000, n_bc: int = 500,
                 n_val: int = 1000, n_test: int = 200,
                 Jx_range: tuple = (-6, 6),
                 Jt_range: tuple = (-6, 6),
                 gamma: float = 0.2,
                 device: str = 'cpu'):
        super().__init__(device)
        self.epsilon = epsilon
        self._n_coll = n_coll
        self.n_bc = n_bc
        self.n_val = n_val
        self.n_test = n_test
        self.Jx_range = Jx_range
        self.Jt_range = Jt_range
        self.gamma = gamma

        # Domain
        self.x_lower, self.x_upper = -1.0, 1.0
        self.t_lower, self.t_upper = 0.0, 1.0

        # Generate all points
        self._generate_points()

    @property
    def input_dim(self):
        return 2

    @property
    def n_collocation(self):
        return self._n_coll

    def analytical_solution(self, x, t):
        """u(x,t) = (1-x²) exp(1/((2t-1)² + ε))"""
        et = 2 * t - 1
        return (1 - x ** 2) * torch.exp(1.0 / (et ** 2 + self.epsilon))

    def _source_term(self, x, t):
        """h(x,t) = 2[1 + 2(2t-1)(x²-1)/((2t-1)²+ε)²] exp(1/((2t-1)²+ε))"""
        et = 2 * t - 1
        ex = torch.exp(1.0 / (et ** 2 + self.epsilon))
        return 2 * ex * (1 + 2 * et * (x ** 2 - 1) / (et ** 2 + self.epsilon) ** 2)

    def _generate_points(self):
        """Generate all collocation, BC, IC, validation, and test points."""
        # Sobol quasi-random collocation
        sampler = qmc.Sobol(d=2, scramble=True, seed=501)
        sobol_coll = sampler.random(n=self._n_coll)
        sobol_bc = sampler.random(n=self.n_bc)

        self.x_coll = torch.tensor(
            sobol_coll[:, 0] * (self.x_upper - self.x_lower) + self.x_lower,
            dtype=torch.float32
        ).to(self.device)
        self.t_coll = torch.tensor(
            sobol_coll[:, 1] * (self.t_upper - self.t_lower) + self.t_lower,
            dtype=torch.float32
        ).to(self.device)

        # Initial condition (t=0)
        self.x_ic = torch.tensor(
            sobol_bc[:, 0] * (self.x_upper - self.x_lower) + self.x_lower,
            dtype=torch.float32
        ).to(self.device)
        self.t_ic = torch.zeros(self.n_bc, dtype=torch.float32).to(self.device)

        # Boundary conditions
        self.t_bc = torch.tensor(
            sobol_bc[:, 1] * (self.t_upper - self.t_lower) + self.t_lower,
            dtype=torch.float32
        ).to(self.device)
        self.x_bc_left = (self.x_lower
                          * torch.ones(self.n_bc, dtype=torch.float32)
                          .to(self.device))
        self.x_bc_right = (self.x_upper
                           * torch.ones(self.n_bc, dtype=torch.float32)
                           .to(self.device))

        # Exact values
        self.u_ic = self.analytical_solution(self.x_ic, self.t_ic)
        self.u_bc_left = self.analytical_solution(self.x_bc_left, self.t_bc)
        self.u_bc_right = self.analytical_solution(self.x_bc_right, self.t_bc)
        self.rhs = self._source_term(self.x_coll, self.t_coll)

        # Validation points
        self.x_val = (torch.rand(self.n_val)
                      * (self.x_upper - self.x_lower) + self.x_lower)
        self.t_val = (torch.rand(self.n_val)
                      * (self.t_upper - self.t_lower) + self.t_lower)
        self.u_val_exact = self.analytical_solution(self.x_val, self.t_val)

        # Test grid
        x_lin = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        t_lin = torch.linspace(self.t_lower, self.t_upper, self.n_test)
        xg, tg = torch.meshgrid(x_lin, t_lin, indexing='ij')
        self.x_test = xg.reshape(-1)
        self.t_test = tg.reshape(-1)
        self.u_test_exact = (self.analytical_solution(self.x_test, self.t_test)
                             .reshape(self.n_test, self.n_test).numpy())

    def build_family(self):
        """Build 2D wavelet family."""
        return build_wavelet_family_2d(
            domain_x=(self.x_lower, self.x_upper),
            domain_y=(self.t_lower, self.t_upper),
            Jx_range=self.Jx_range,
            Jy_range=self.Jt_range,
            gamma=self.gamma
        )

    def get_coordinates(self):
        """Return coordinate dict for training."""
        return {
            'x_collocation': self.x_coll,
            't_collocation': self.t_coll,
            'x_ic': self.x_ic,
            't_ic': self.t_ic,
            'x_bc_left': self.x_bc_left,
            'x_bc_right': self.x_bc_right,
            't_bc': self.t_bc,
        }

    def get_loss_function(self, pruned=False):
        """
        Return a loss function callable for the heat equation.

        The returned function signature:
            loss_fn(c, bias, matrices_dict) -> (total, pde, bc+ic)
        """
        # Capture problem data in closure
        rhs = self.rhs
        u_ic = self.u_ic
        u_bc_left = self.u_bc_left
        u_bc_right = self.u_bc_right

        # These will be rebuilt with the current meta-wavelet
        x_ic = self.x_ic
        t_ic = self.t_ic
        x_bc_left = self.x_bc_left
        x_bc_right = self.x_bc_right
        t_bc = self.t_bc

        def loss_fn(c, bias, matrices):
            """
            Heat equation loss using precomputed wavelet matrices.

            matrices should contain:
                'W'     : [N_coll, F]
                'D1Wy'  : [N_coll, F]  (d/dt — y-direction is time)
                'D2Wx'  : [N_coll, F]  (d²/dx²)
            """
            W = matrices['W']
            DWt = matrices['D1Wy']    # time derivative (2nd coord)
            DW2x = matrices['D2Wx']   # spatial second derivative

            # PDE residual: u_t - u_xx - h = 0
            u_t = torch.mv(DWt, c)
            u_xx = torch.mv(DW2x, c)
            pde_loss = torch.mean((u_t - u_xx - rhs) ** 2)

            # IC and BC need wavelet matrices at their respective points
            # For simplicity, evaluate directly using meta-wavelet
            # (this is a small overhead since BC/IC sets are small)
            from core.wavelet_matrices import WaveletMatrixBuilder
            meta_w = None
            # Try to get meta_wavelet from the computational graph
            # We need to build BC/IC matrices with the current wavelet

            # Use the same matrices approach: build them once
            # In practice, these are built alongside the main matrices
            # For now, use the W matrix structure

            # IC loss
            u_pred_ic = torch.mv(matrices.get('W_ic', W[:len(u_ic)]), c) + bias
            ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)

            # BC loss
            u_pred_bc_l = torch.mv(
                matrices.get('W_bc_left', W[:len(u_bc_left)]), c
            ) + bias
            u_pred_bc_r = torch.mv(
                matrices.get('W_bc_right', W[:len(u_bc_right)]), c
            ) + bias
            bc_loss = (torch.mean((u_pred_bc_l - u_bc_left) ** 2)
                       + torch.mean((u_pred_bc_r - u_bc_right) ** 2))

            total = pde_loss + ic_loss + bc_loss
            return total, pde_loss, ic_loss + bc_loss

        return loss_fn

    def get_full_loss_function(self, meta_wavelet, family, device='cpu'):
        """
        Return a self-contained loss function that builds all matrices
        internally using the meta_wavelet.

        This is the preferred loss function for training.
        """
        rhs = self.rhs.to(device)
        u_ic = self.u_ic.to(device)
        u_bc_left = self.u_bc_left.to(device)
        u_bc_right = self.u_bc_right.to(device)

        x_coll = self.x_coll.to(device)
        t_coll = self.t_coll.to(device)
        x_ic = self.x_ic.to(device)
        t_ic = self.t_ic.to(device)
        x_bc_left_pts = self.x_bc_left.to(device)
        x_bc_right_pts = self.x_bc_right.to(device)
        t_bc = self.t_bc.to(device)

        fam = family.to(device)
        jx = fam[:, 0]
        jy = fam[:, 1]
        kx = fam[:, 2]
        ky = fam[:, 3]

        def loss_fn(c, bias):
            # Build matrices with current meta-wavelet
            W = meta_wavelet.evaluate_basis_2d(
                x_coll, t_coll, jx, jy, kx, ky
            )
            DWt = meta_wavelet.evaluate_basis_2d_dy(
                x_coll, t_coll, jx, jy, kx, ky, order=1
            )
            DW2x = meta_wavelet.evaluate_basis_2d_dx(
                x_coll, t_coll, jx, jy, kx, ky, order=2
            )

            # PDE
            u_t = torch.mv(DWt, c)
            u_xx = torch.mv(DW2x, c)
            pde_loss = torch.mean((u_t - u_xx - rhs) ** 2)

            # IC
            W_ic = meta_wavelet.evaluate_basis_2d(
                x_ic, t_ic, jx, jy, kx, ky
            )
            u_pred_ic = torch.mv(W_ic, c) + bias
            ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)

            # BC
            W_bc_l = meta_wavelet.evaluate_basis_2d(
                x_bc_left_pts, t_bc, jx, jy, kx, ky
            )
            W_bc_r = meta_wavelet.evaluate_basis_2d(
                x_bc_right_pts, t_bc, jx, jy, kx, ky
            )
            u_pred_bc_l = torch.mv(W_bc_l, c) + bias
            u_pred_bc_r = torch.mv(W_bc_r, c) + bias

            bc_loss = (torch.mean((u_pred_bc_l - u_bc_left) ** 2)
                       + torch.mean((u_pred_bc_r - u_bc_right) ** 2))

            total = pde_loss + ic_loss + bc_loss
            return total, pde_loss, ic_loss + bc_loss

        return loss_fn

    def evaluate(self, model, family, device='cpu'):
        """Evaluate on test grid."""
        family = family.to(device)
        jx = family[:, 0]
        jy = family[:, 1]
        kx = family[:, 2]
        ky = family[:, 3]

        with torch.no_grad():
            c, bias = model()

            W_test = model.meta_wavelet.evaluate_basis_2d(
                self.x_test.to(device), self.t_test.to(device),
                jx, jy, kx, ky
            )
            u_pred = torch.mv(W_test, c) + bias
            u_pred_np = u_pred.cpu().reshape(self.n_test, self.n_test).numpy()

            # Also evaluate on validation set
            W_val = model.meta_wavelet.evaluate_basis_2d(
                self.x_val.to(device), self.t_val.to(device),
                jx, jy, kx, ky
            )
            u_val_pred = torch.mv(W_val, c) + bias

        # Metrics
        rel_l2 = PINNLoss.relative_l2_error(
            u_val_pred.cpu(), self.u_val_exact
        ).item()

        max_err = PINNLoss.max_error(
            u_val_pred.cpu(), self.u_val_exact
        ).item()

        return {
            'rel_l2_error': rel_l2,
            'max_error': max_err,
            'u_pred': u_pred_np,
            'u_exact': self.u_test_exact,
        }

    def __repr__(self):
        return (f"HeatConductionProblem(ε={self.epsilon}, "
                f"n_coll={self._n_coll}, n_bc={self.n_bc})")
