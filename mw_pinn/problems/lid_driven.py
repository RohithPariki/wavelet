"""
Lid-Driven Cavity (Steady Navier-Stokes)
========================================
2D Steady Incompressible Navier-Stokes Equations.

u u_x + v u_y + p_x - (1/Re) (u_xx + u_yy) = 0
u v_x + v v_y + p_y - (1/Re) (v_xx + v_yy) = 0
u_x + v_y = 0

Domain: (x,y) ∈ (0,1)²
Boundary Conditions:
Top lid: u=1, v=0
Other walls: u=0, v=0
"""

import torch
import numpy as np
import pandas as pd
from scipy.stats import qmc

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problems.base import BaseProblem
from core.wavelet_matrices import build_wavelet_family_2d
from core.loss_functions import PINNLoss

class LidDrivenProblem(BaseProblem):
    def __init__(self, Re: float = 400.0,
                 n_coll: int = 8000, n_bc: int = 1000,
                 n_val: int = 100, n_test: int = 100,
                 Jx_range: tuple = (-4, 4),
                 Jy_range: tuple = (-4, 4),
                 gamma: float = 0.5,
                 device: str = 'cpu'):
        super().__init__(device)
        self.Re = Re
        self._n_coll = n_coll
        self.n_bc = n_bc
        self.n_val = n_val
        self.n_test = n_test
        self.Jx_range = Jx_range
        self.Jy_range = Jy_range
        self.gamma = gamma

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
        # Lid driven cavity doesn't have a simple analytical solution.
        # It relies on numerical reference data (vel_ref) for validation.
        return torch.zeros_like(x)

    def _generate_points(self):
        sampler = qmc.Sobol(d=2, scramble=True, seed=501)
        sobol_coll = sampler.random(n=self._n_coll)
        sobol_bc = sampler.random(n=self.n_bc)

        self.x_coll = torch.tensor(sobol_coll[:, 0], dtype=torch.float32).to(self.device)
        self.y_coll = torch.tensor(sobol_coll[:, 1], dtype=torch.float32).to(self.device)

        x_bc = torch.tensor(sobol_bc[:, 0], dtype=torch.float32).to(self.device)
        y_bc = torch.tensor(sobol_bc[:, 1], dtype=torch.float32).to(self.device)

        self.x_bc_left = torch.zeros_like(y_bc)
        self.x_bc_right = torch.ones_like(y_bc)
        self.y_bc_bottom = torch.zeros_like(x_bc)
        self.y_bc_top = torch.ones_like(x_bc)
        self.x_bc = x_bc
        self.y_bc = y_bc

        x_lin = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        y_lin = torch.linspace(self.y_lower, self.y_upper, self.n_test)
        xg, yg = torch.meshgrid(x_lin, y_lin, indexing='ij')
        self.x_test = xg.reshape(-1)
        self.y_test = yg.reshape(-1)

        # Try to load reference velocity data for the given Re
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ref_path = os.path.join(base_dir, '1stpaper', 'W-PINN', 'Lid-Driven', f'Re{int(self.Re)}', 'ref_vel.csv')
        if os.path.exists(ref_path):
            try:
                vel_data = pd.read_csv(ref_path, header=None)
                self.vel_ref = torch.tensor(vel_data.values, dtype=torch.float32).to(self.device)
                if self.vel_ref.shape == (self.n_test, self.n_test):
                    self.vel_ref = self.vel_ref.flatten()
            except Exception as e:
                print(f"Could not load reference data: {e}")
                self.vel_ref = None
        else:
            self.vel_ref = None

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
        raise NotImplementedError("Use get_full_loss_function")

    def get_full_loss_function(self, meta_wavelet, family, device='cpu'):
        x_coll = self.x_coll.to(device)
        y_coll = self.y_coll.to(device)
        
        x_bc_l = self.x_bc_left.to(device)
        x_bc_r = self.x_bc_right.to(device)
        y_bc_b = self.y_bc_bottom.to(device)
        y_bc_t = self.y_bc_top.to(device)
        x_bc = self.x_bc.to(device)
        y_bc = self.y_bc.to(device)
        Re = self.Re

        fam = family.to(device)
        if fam.dim() == 1:
            fam = fam.unsqueeze(0)
        jx, jy, kx, ky = fam[:, 0], fam[:, 1], fam[:, 2], fam[:, 3]

        def loss_fn(c_u, c_v, c_p, bias_u, bias_v, bias_p):
            W = meta_wavelet.evaluate_basis_2d(x_coll, y_coll, jx, jy, kx, ky)
            DWx = meta_wavelet.evaluate_basis_2d_dx(x_coll, y_coll, jx, jy, kx, ky, order=1)
            DWy = meta_wavelet.evaluate_basis_2d_dy(x_coll, y_coll, jx, jy, kx, ky, order=1)
            DW2x = meta_wavelet.evaluate_basis_2d_dx(x_coll, y_coll, jx, jy, kx, ky, order=2)
            DW2y = meta_wavelet.evaluate_basis_2d_dy(x_coll, y_coll, jx, jy, kx, ky, order=2)

            u = torch.mv(W, c_u) + bias_u
            u_x = torch.mv(DWx, c_u)
            u_y = torch.mv(DWy, c_u)
            u_xx = torch.mv(DW2x, c_u)
            u_yy = torch.mv(DW2y, c_u)

            v = torch.mv(W, c_v) + bias_v
            v_x = torch.mv(DWx, c_v)
            v_y = torch.mv(DWy, c_v)
            v_xx = torch.mv(DW2x, c_v)
            v_yy = torch.mv(DW2y, c_v)

            p_x = torch.mv(DWx, c_p)
            p_y = torch.mv(DWy, c_p)

            pde_1 = u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy)
            pde_2 = u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy)
            pde_3 = u_x + v_y

            pde_loss = torch.mean(pde_1**2) + torch.mean(pde_2**2) + torch.mean(pde_3**2)

            # BC calculations
            W_l = meta_wavelet.evaluate_basis_2d(x_bc_l, y_bc, jx, jy, kx, ky)
            W_r = meta_wavelet.evaluate_basis_2d(x_bc_r, y_bc, jx, jy, kx, ky)
            W_b = meta_wavelet.evaluate_basis_2d(x_bc, y_bc_b, jx, jy, kx, ky)
            W_t = meta_wavelet.evaluate_basis_2d(x_bc, y_bc_t, jx, jy, kx, ky)

            bc_loss_u = (
                torch.mean((torch.mv(W_l, c_u) + bias_u - 0.0)**2) +
                torch.mean((torch.mv(W_r, c_u) + bias_u - 0.0)**2) +
                torch.mean((torch.mv(W_b, c_u) + bias_u - 0.0)**2) +
                torch.mean((torch.mv(W_t, c_u) + bias_u - 1.0)**2) # Lid is moving at u=1
            )

            bc_loss_v = (
                torch.mean((torch.mv(W_l, c_v) + bias_v - 0.0)**2) +
                torch.mean((torch.mv(W_r, c_v) + bias_v - 0.0)**2) +
                torch.mean((torch.mv(W_b, c_v) + bias_v - 0.0)**2) +
                torch.mean((torch.mv(W_t, c_v) + bias_v - 0.0)**2)
            )

            bc_loss = bc_loss_u + bc_loss_v

            return pde_loss + bc_loss, pde_loss, bc_loss

        return loss_fn

    def evaluate(self, model, family, device='cpu'):
        if self.vel_ref is None:
            return {'rel_l2_error': 0.0, 'max_error': 0.0}

        family = family.to(device)
        if family.dim() == 1:
            family = family.unsqueeze(0)
        jx, jy = family[:, 0], family[:, 1]
        kx, ky = family[:, 2], family[:, 3]

        with torch.no_grad():
            result = model()
            coeffs, biases = result
            c_u, c_v, c_p = coeffs
            bias_u, bias_v, bias_p = biases

            W_test = model.meta_wavelet.evaluate_basis_2d(
                self.x_test.to(device), self.y_test.to(device),
                jx, jy, kx, ky
            )

            u_test = torch.mv(W_test, c_u) + bias_u
            v_test = torch.mv(W_test, c_v) + bias_v
            
            pred_vel = torch.sqrt(u_test**2 + v_test**2)

        return {
            'rel_l2_error': PINNLoss.relative_l2_error(pred_vel.cpu(), self.vel_ref.cpu()).item(),
            'max_error': PINNLoss.max_error(pred_vel.cpu(), self.vel_ref.cpu()).item(),
            'u_pred': pred_vel.cpu().numpy().reshape(self.n_test, self.n_test),
            'u_exact': self.vel_ref.cpu().numpy().reshape(self.n_test, self.n_test)
        }
