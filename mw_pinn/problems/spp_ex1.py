"""
Singular Perturbation Problem (SPP) Ex1
=======================================
1D ODE:
ε u'' + (1+ε) u' + u = 0

where ε = 2^(-7).
Exact Solution: u(x) = (exp(-x) - exp(-x/ε)) / (exp(-1) - exp(-1/ε))
Boundary Conditions:
u(0) = 0
u(1) = 1

Domain: x ∈ (0,1)
"""

import torch
import numpy as np
from scipy.stats import qmc

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problems.base import BaseProblem
from core.wavelet_matrices import build_wavelet_family_2d
from core.loss_functions import PINNLoss

class SPPEx1Problem(BaseProblem):
    def __init__(self, epsilon: float = 2**(-7),
                 n_coll: int = 1000, n_bc: int = 100,
                 n_val: int = 1000, n_test: int = 10000,
                 Jx_range: tuple = (-4, 4),
                 gamma: float = 0.5,
                 device: str = 'cpu'):
        super().__init__(device)
        self.e = epsilon
        self._n_coll = n_coll
        self.n_bc = n_bc
        self.n_val = n_val
        self.n_test = n_test
        # Model as 1D problem (using dummy y coordinates)
        self.Jx_range = Jx_range
        self.Jy_range = (0, 1)
        self.gamma = gamma

        self.x_lower, self.x_upper = 0.0, 1.0

        self._generate_points()

    @property
    def input_dim(self):
        return 1

    @property
    def n_collocation(self):
        return self._n_coll

    def analytical_solution(self, x):
        """Exact solution u(x) = (exp(-x) - exp(-x/ε)) / (exp(-1) - exp(-1/ε))"""
        e = self.e
        f1 = torch.exp(-x) - torch.exp(-x / e)
        f2 = np.exp(-1.0) - np.exp(-1.0 / e)
        return f1 / f2

    def _generate_points(self):
        sampler = qmc.Sobol(d=1, scramble=True, seed=501)
        sobol_coll = sampler.random(n=self._n_coll)

        self.t_coll = torch.tensor(sobol_coll[:, 0] * (self.x_upper - self.x_lower) + self.x_lower, dtype=torch.float32).to(self.device)
        self.x_coll = torch.zeros_like(self.t_coll) # dummy

        self.t_bc_left = torch.zeros(self.n_bc, dtype=torch.float32).to(self.device)
        self.t_bc_right = torch.ones(self.n_bc, dtype=torch.float32).to(self.device)
        self.x_bc = torch.zeros_like(self.t_bc_left) # dummy

        self.u_bc_left = self.analytical_solution(self.t_bc_left)
        self.u_bc_right = self.analytical_solution(self.t_bc_right)

        self.t_val = (torch.rand(self.n_val) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        self.x_val = torch.zeros_like(self.t_val)
        self.u_val_exact = self.analytical_solution(self.t_val)

        self.t_test = torch.linspace(self.x_lower, self.x_upper, self.n_test).to(self.device)
        self.x_test = torch.zeros_like(self.t_test)
        self.u_test_exact = self.analytical_solution(self.t_test)

    def build_family(self):
        return build_wavelet_family_2d(
            domain_x=(0.0, 0.0),
            domain_y=(self.x_lower, self.x_upper),
            Jx_range=self.Jy_range,
            Jy_range=self.Jx_range,
            gamma=self.gamma
        )

    def get_coordinates(self):
        return {
            'x_collocation': self.x_coll,
            'y_collocation': self.t_coll,
        }

    def get_loss_function(self, pruned=False):
        raise NotImplementedError("Use get_full_loss_function")

    def get_full_loss_function(self, meta_wavelet, family, device='cpu'):
        x_coll = self.x_coll.to(device)
        t_coll = self.t_coll.to(device)
        x_bc = self.x_bc.to(device)
        t_bc_l = self.t_bc_left.to(device)
        t_bc_r = self.t_bc_right.to(device)

        u_bc_l = self.u_bc_left.to(device)
        u_bc_r = self.u_bc_right.to(device)
        e = self.e

        fam = family.to(device)
        if fam.dim() == 1:
            fam = fam.unsqueeze(0)
        jx, jt, kx, kt = fam[:, 0], fam[:, 1], fam[:, 2], fam[:, 3]

        def loss_fn(c, bias):
            W = meta_wavelet.evaluate_basis_2d(x_coll, t_coll, jx, jt, kx, kt)
            DWt = meta_wavelet.evaluate_basis_2d_dy(x_coll, t_coll, jx, jt, kx, kt, order=1)
            DW2t = meta_wavelet.evaluate_basis_2d_dy(x_coll, t_coll, jx, jt, kx, kt, order=2)

            u = torch.mv(W, c) + bias
            u_t = torch.mv(DWt, c)
            u_tt = torch.mv(DW2t, c)

            # ε u'' + (1+ε) u' + u = 0
            pde_res = e * u_tt + (1.0 + e) * u_t + u
            pde_loss = torch.mean(pde_res**2)

            W_l = meta_wavelet.evaluate_basis_2d(x_bc, t_bc_l, jx, jt, kx, kt)
            W_r = meta_wavelet.evaluate_basis_2d(x_bc, t_bc_r, jx, jt, kx, kt)

            u_pred_bc_l = torch.mv(W_l, c) + bias
            u_pred_bc_r = torch.mv(W_r, c) + bias

            bc_loss = torch.mean((u_pred_bc_l - u_bc_l)**2) + torch.mean((u_pred_bc_r - u_bc_r)**2)

            return pde_loss + bc_loss, pde_loss, bc_loss

        return loss_fn

    def evaluate(self, model, family, device='cpu'):
        family = family.to(device)
        if family.dim() == 1:
            family = family.unsqueeze(0)
        jx, jt = family[:, 0], family[:, 1]
        kx, kt = family[:, 2], family[:, 3]

        with torch.no_grad():
            result = model()
            if isinstance(result, tuple) and len(result) == 2:
                c, bias = result
            else:
                c, bias = result, 0.0

            W_val = model.meta_wavelet.evaluate_basis_2d(
                self.x_val.to(device), self.t_val.to(device),
                jx, jt, kx, kt
            )

            u_pred = torch.mv(W_val, c) + bias

            W_test = model.meta_wavelet.evaluate_basis_2d(
                self.x_test.to(device), self.t_test.to(device),
                jx, jt, kx, kt
            )
            u_test_pred = torch.mv(W_test, c) + bias

        return {
            'rel_l2_error': PINNLoss.relative_l2_error(u_pred.cpu(), self.u_val_exact.cpu()).item(),
            'max_error': PINNLoss.max_error(u_pred.cpu(), self.u_val_exact.cpu()).item(),
            'u_pred': u_test_pred.cpu().numpy(),
            'u_exact': self.u_test_exact.cpu().numpy(),
            'x_coords': self.t_test.cpu().numpy()
        }
