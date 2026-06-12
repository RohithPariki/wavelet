"""
Singular Perturbation Problem (SPP) Ex2
=======================================
1D ODE:
ε u'' + (3+t)u' + u² - sin(u) = f(t)

where ε = 2^(-10).
Exact Solution: u(t) = t² + 2 - exp(-t/ε)
Initial Conditions:
u(0) = 1.0
u'(0) = 1/ε

Domain: t ∈ (0,1)
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

class SPPEx2Problem(BaseProblem):
    def __init__(self, epsilon: float = 2**(-10),
                 n_coll: int = 10000, n_bc: int = 200,
                 n_val: int = 1000, n_test: int = 2000,
                 Jt_range: tuple = (-4, 4),
                 gamma: float = 0.5,
                 device: str = 'cpu'):
        super().__init__(device)
        self.e = epsilon
        self._n_coll = n_coll
        self.n_bc = n_bc
        self.n_val = n_val
        self.n_test = n_test
        # We model this as a 1D problem but the framework uses 2D wavelet matrices.
        # We will use dummy x=0 coordinates and Jx_range=(0,1) so it doesn't return empty.
        self.Jx_range = (0, 1)
        self.Jt_range = Jt_range
        self.gamma = gamma

        self.t_lower, self.t_upper = 0.0, 1.0

        self._generate_points()

    @property
    def input_dim(self):
        return 1

    @property
    def n_collocation(self):
        return self._n_coll

    def analytical_solution(self, t):
        """Exact solution u(t) = t² + 2 - exp(-t/ε)"""
        return t**2 + 2.0 - torch.exp(-t / self.e)

    def _right_side(self, t):
        expo = torch.exp(-t / self.e)
        return (2 - expo + t**2)**2 + (3 + t)*(2*t + expo/self.e) + self.e*(2 - expo/self.e**2) - torch.sin(2 - expo + t**2)

    def _generate_points(self):
        sampler = qmc.Sobol(d=1, scramble=True, seed=501)
        sobol_coll = sampler.random(n=self._n_coll)

        self.t_coll = torch.tensor(sobol_coll[:, 0], dtype=torch.float32).to(self.device)
        self.x_coll = torch.zeros_like(self.t_coll) # dummy

        self.t_ic = torch.zeros(self.n_bc, dtype=torch.float32).to(self.device)
        self.x_ic = torch.zeros_like(self.t_ic) # dummy

        self.u_ic = torch.ones_like(self.t_ic) * 1.0
        self.Du_ic = torch.ones_like(self.t_ic) * (1.0 / self.e)

        self.rhs = self._right_side(self.t_coll)

        self.t_val = torch.rand(self.n_val).to(self.device)
        self.x_val = torch.zeros_like(self.t_val)
        self.u_val_exact = self.analytical_solution(self.t_val)

        self.t_test = torch.linspace(self.t_lower, self.t_upper, self.n_test).to(self.device)
        self.x_test = torch.zeros_like(self.t_test)
        self.u_test_exact = self.analytical_solution(self.t_test)

    def build_family(self):
        return build_wavelet_family_2d(
            domain_x=(0.0, 0.0),
            domain_y=(self.t_lower, self.t_upper),
            Jx_range=self.Jx_range,
            Jy_range=self.Jt_range,
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
        x_ic = self.x_ic.to(device)
        t_ic = self.t_ic.to(device)

        rhs = self.rhs.to(device)
        u_ic = self.u_ic.to(device)
        Du_ic = self.Du_ic.to(device)
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

            # ε u'' + (3+t)u' + u² - sin(u) = f(t)
            pde_res = e * u_tt + (3.0 + t_coll) * u_t + u**2 - torch.sin(u) - rhs
            pde_loss = torch.mean(pde_res**2)

            W_ic = meta_wavelet.evaluate_basis_2d(x_ic, t_ic, jx, jt, kx, kt)
            DWt_ic = meta_wavelet.evaluate_basis_2d_dy(x_ic, t_ic, jx, jt, kx, kt, order=1)

            u_pred_ic = torch.mv(W_ic, c) + bias
            u_t_pred_ic = torch.mv(DWt_ic, c)

            bc_loss = torch.mean((u_pred_ic - u_ic)**2) + torch.mean((u_t_pred_ic - Du_ic)**2)

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

        return {
            'rel_l2_error': PINNLoss.relative_l2_error(u_pred.cpu(), self.u_val_exact.cpu()).item(),
            'max_error': PINNLoss.max_error(u_pred.cpu(), self.u_val_exact.cpu()).item()
        }
