"""
Maxwell's Equations (Homogeneous 1D)
======================================
1D Homogeneous Maxwell's equations in vacuum.

∂E/∂t - ∂H/∂x = 0
∂H/∂t - ∂E/∂x = 0

Domain: (x,t) ∈ (0,1)²
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

class MaxwellHomogeneousProblem(BaseProblem):
    def __init__(self, mode: int = 4,
                 n_coll: int = 10000, n_bc: int = 500,
                 n_val: int = 1000, n_test: int = 200,
                 Jx_range: tuple = (-4, 4),
                 Jt_range: tuple = (-4, 4),
                 gamma: float = 0.5,
                 device: str = 'cpu'):
        super().__init__(device)
        self.mode = mode
        self._n_coll = n_coll
        self.n_bc = n_bc
        self.n_val = n_val
        self.n_test = n_test
        self.Jx_range = Jx_range
        self.Jt_range = Jt_range
        self.gamma = gamma

        self.x_lower, self.x_upper = 0.0, 1.0
        self.t_lower, self.t_upper = 0.0, 1.0

        self._generate_points()

    @property
    def input_dim(self):
        return 2

    @property
    def n_collocation(self):
        return self._n_coll

    def analytical_solution(self, x, t):
        """
        Analytical solution.
        E = sin(m*pi*x)*cos(m*pi*t)
        H = -cos(m*pi*x)*sin(m*pi*t)
        """
        p = torch.tensor(np.pi)
        sx = torch.sin(self.mode * p * x)
        st = torch.sin(self.mode * p * t)
        cx = torch.cos(self.mode * p * x)
        ct = torch.cos(self.mode * p * t)
        
        E = sx * ct
        H = -cx * st
        return E, H

    def _generate_points(self):
        sampler = qmc.Sobol(d=2, scramble=True, seed=501)
        sobol_coll = sampler.random(n=self._n_coll)
        sobol_bc = sampler.random(n=self.n_bc)

        self.x_coll = torch.tensor(sobol_coll[:, 0], dtype=torch.float32).to(self.device)
        self.t_coll = torch.tensor(sobol_coll[:, 1], dtype=torch.float32).to(self.device)

        x_bc = torch.tensor(sobol_bc[:, 0], dtype=torch.float32).to(self.device)
        t_bc = torch.tensor(sobol_bc[:, 1], dtype=torch.float32).to(self.device)

        self.x_bc_left = torch.zeros_like(t_bc)
        self.x_bc_right = torch.ones_like(t_bc)
        self.t_ic = torch.zeros_like(x_bc)
        self.t_bc = t_bc
        self.x_ic = x_bc

        self.E_ic, self.H_ic = self.analytical_solution(self.x_ic, self.t_ic)
        self.E_bc_l, self.H_bc_l = self.analytical_solution(self.x_bc_left, self.t_bc)
        self.E_bc_r, self.H_bc_r = self.analytical_solution(self.x_bc_right, self.t_bc)

        self.x_val = torch.rand(self.n_val)
        self.t_val = torch.rand(self.n_val)
        self.E_val, self.H_val = self.analytical_solution(self.x_val, self.t_val)

    def build_family(self):
        return build_wavelet_family_2d(
            domain_x=(self.x_lower, self.x_upper),
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
        x_bc_l = self.x_bc_left.to(device)
        x_bc_r = self.x_bc_right.to(device)
        x_ic = self.x_ic.to(device)
        t_bc = self.t_bc.to(device)
        t_ic = self.t_ic.to(device)

        E_ic, H_ic = self.E_ic.to(device), self.H_ic.to(device)
        E_bc_l, H_bc_l = self.E_bc_l.to(device), self.H_bc_l.to(device)
        E_bc_r, H_bc_r = self.E_bc_r.to(device), self.H_bc_r.to(device)

        fam = family.to(device)
        if fam.dim() == 1:
            fam = fam.unsqueeze(0)
        jx, jt, kx, kt = fam[:, 0], fam[:, 1], fam[:, 2], fam[:, 3]

        def loss_fn(c_E, c_H, bias_E, bias_H):
            W = meta_wavelet.evaluate_basis_2d(x_coll, t_coll, jx, jt, kx, kt)
            DWx = meta_wavelet.evaluate_basis_2d_dx(x_coll, t_coll, jx, jt, kx, kt, order=1)
            DWt = meta_wavelet.evaluate_basis_2d_dy(x_coll, t_coll, jx, jt, kx, kt, order=1)

            E_x = torch.mv(DWx, c_E)
            E_t = torch.mv(DWt, c_E)
            H_x = torch.mv(DWx, c_H)
            H_t = torch.mv(DWt, c_H)

            pde_loss = torch.mean((E_x + H_t)**2) + torch.mean((H_x + E_t)**2)

            W_ic = meta_wavelet.evaluate_basis_2d(x_ic, t_ic, jx, jt, kx, kt)
            W_l = meta_wavelet.evaluate_basis_2d(x_bc_l, t_bc, jx, jt, kx, kt)
            W_r = meta_wavelet.evaluate_basis_2d(x_bc_r, t_bc, jx, jt, kx, kt)
            
            DWx_l = meta_wavelet.evaluate_basis_2d_dx(x_bc_l, t_bc, jx, jt, kx, kt, order=1)
            DWx_r = meta_wavelet.evaluate_basis_2d_dx(x_bc_r, t_bc, jx, jt, kx, kt, order=1)

            bc_loss = (
                torch.mean((torch.mv(W_ic, c_E) + bias_E - E_ic)**2) +
                torch.mean((torch.mv(W_ic, c_H) + bias_H - H_ic)**2) +
                torch.mean((torch.mv(W_l, c_E) + bias_E)**2) +
                torch.mean((torch.mv(W_r, c_E) + bias_E)**2) +
                torch.mean((torch.mv(DWx_l, c_H))**2) +
                torch.mean((torch.mv(DWx_r, c_H))**2)
            )

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
            coeffs, biases = result
            c_E, c_H = coeffs
            bias_E, bias_H = biases

            W_val = model.meta_wavelet.evaluate_basis_2d(
                self.x_val.to(device), self.t_val.to(device),
                jx, jt, kx, kt
            )

            E_pred = torch.mv(W_val, c_E) + bias_E
            H_pred = torch.mv(W_val, c_H) + bias_H

        return {
            'rel_l2_error_E': PINNLoss.relative_l2_error(E_pred.cpu(), self.E_val).item(),
            'rel_l2_error_H': PINNLoss.relative_l2_error(H_pred.cpu(), self.H_val).item(),
            'max_error_E': PINNLoss.max_error(E_pred.cpu(), self.E_val).item(),
            'max_error_H': PINNLoss.max_error(H_pred.cpu(), self.H_val).item()
        }
