"""
Maxwell's Equations (Homogeneous TEz)
======================================
Homogeneous 2D Maxwell's equations in transverse-electric mode.

∂Hz/∂y  = -iω Ex
-∂Hz/∂x = -iω Ey
∂Ex/∂y - ∂Ey/∂x = iω Hz

Domain: (x,y) ∈ (0,1)²
From W-PINN paper.
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


class MaxwellProblem(BaseProblem):
    """
    Homogeneous 2D Maxwell TEz problem.

    Parameters
    ----------
    omega : float
        Angular frequency.
    n_coll : int
        Number of collocation points.
    n_bc : int
        Boundary points per edge.
    n_test : int
        Test grid resolution.
    Jx_range, Jy_range : tuple
        Wavelet scale ranges.
    gamma : float
        Translation hyperparameter.
    device : str
    """

    def __init__(self, omega: float = 2 * np.pi,
                 n_coll: int = 10000, n_bc: int = 250,
                 n_val: int = 1000, n_test: int = 200,
                 Jx_range: tuple = (-4, 6),
                 Jy_range: tuple = (-4, 6),
                 gamma: float = 0.5,
                 device: str = 'cpu'):
        super().__init__(device)
        self.omega = omega
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
        """
        Reference analytical solution for TEz mode.
        Ex, Ey, Hz fields.
        """
        p = torch.tensor(np.pi)
        kx = p
        ky = p
        Ex = torch.sin(kx * x) * torch.cos(ky * y)
        Ey = -torch.cos(kx * x) * torch.sin(ky * y)
        Hz = -(1.0 / self.omega) * (
            kx * torch.cos(kx * x) * torch.sin(ky * y)
            + ky * torch.sin(kx * x) * torch.cos(ky * y)
        )
        return Ex, Ey, Hz

    def _generate_points(self):
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

        # BC
        x_bc = torch.tensor(
            sobol_bc[:, 0], dtype=torch.float32
        ).to(self.device)
        y_bc = torch.tensor(
            sobol_bc[:, 1], dtype=torch.float32
        ).to(self.device)

        self.x_bc_left = torch.zeros_like(y_bc)
        self.x_bc_right = torch.ones_like(y_bc)
        self.y_bc_bottom = torch.zeros_like(x_bc)
        self.y_bc_top = torch.ones_like(x_bc)
        self.x_bc = x_bc
        self.y_bc = y_bc

        # Store BC exact values for all fields
        self.bc_data = {}
        for name, x_pts, y_pts in [
            ('left', self.x_bc_left, self.y_bc),
            ('right', self.x_bc_right, self.y_bc),
            ('bottom', self.x_bc, self.y_bc_bottom),
            ('top', self.x_bc, self.y_bc_top)
        ]:
            ex, ey, hz = self.analytical_solution(x_pts, y_pts)
            self.bc_data[name] = {'Ex': ex, 'Ey': ey, 'Hz': hz,
                                  'x': x_pts, 'y': y_pts}

        # Validation
        self.x_val = torch.rand(self.n_val)
        self.y_val = torch.rand(self.n_val)
        self.ex_val, self.ey_val, self.hz_val = self.analytical_solution(
            self.x_val, self.y_val
        )

        # Test grid
        x_lin = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        y_lin = torch.linspace(self.y_lower, self.y_upper, self.n_test)
        xg, yg = torch.meshgrid(x_lin, y_lin, indexing='ij')
        self.x_test = xg.reshape(-1)
        self.y_test = yg.reshape(-1)

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
        """Placeholder — Maxwell requires multi-output model."""
        raise NotImplementedError(
            "Maxwell requires multi-output model. "
            "Use get_full_loss_function() instead."
        )

    def get_full_loss_function(self, meta_wavelet, family, device='cpu'):
        """Full loss for Maxwell system with 3 field variables."""
        x_coll = self.x_coll.to(device)
        y_coll = self.y_coll.to(device)
        omega = self.omega

        fam = family.to(device)
        jx, jy, kx, ky = fam[:, 0], fam[:, 1], fam[:, 2], fam[:, 3]

        bc = {k: {field: v[field].to(device) for field in ['Ex', 'Ey', 'Hz']}
              | {'x': v['x'].to(device), 'y': v['y'].to(device)}
              for k, v in self.bc_data.items()}

        def loss_fn(c_ex, c_ey, c_hz, bias_ex, bias_ey, bias_hz):
            # Collocation matrices
            DWx = meta_wavelet.evaluate_basis_2d_dx(
                x_coll, y_coll, jx, jy, kx, ky, order=1
            )
            DWy = meta_wavelet.evaluate_basis_2d_dy(
                x_coll, y_coll, jx, jy, kx, ky, order=1
            )
            W = meta_wavelet.evaluate_basis_2d(
                x_coll, y_coll, jx, jy, kx, ky
            )

            Ex = torch.mv(W, c_ex) + bias_ex
            Ey = torch.mv(W, c_ey) + bias_ey
            Hz = torch.mv(W, c_hz) + bias_hz

            dHz_dx = torch.mv(DWx, c_hz)
            dHz_dy = torch.mv(DWy, c_hz)
            dEx_dy = torch.mv(DWy, c_ex)
            dEy_dx = torch.mv(DWx, c_ey)

            r1 = dHz_dy + omega * Ex
            r2 = -dHz_dx + omega * Ey
            r3 = dEx_dy - dEy_dx - omega * Hz

            pde_loss = (torch.mean(r1 ** 2)
                        + torch.mean(r2 ** 2)
                        + torch.mean(r3 ** 2))

            # BC loss
            bc_loss = torch.tensor(0.0, device=device)
            for edge_data in bc.values():
                W_bc = meta_wavelet.evaluate_basis_2d(
                    edge_data['x'], edge_data['y'], jx, jy, kx, ky
                )
                bc_loss += torch.mean(
                    (torch.mv(W_bc, c_ex) + bias_ex - edge_data['Ex']) ** 2
                )
                bc_loss += torch.mean(
                    (torch.mv(W_bc, c_ey) + bias_ey - edge_data['Ey']) ** 2
                )
                bc_loss += torch.mean(
                    (torch.mv(W_bc, c_hz) + bias_hz - edge_data['Hz']) ** 2
                )

            return pde_loss + bc_loss, pde_loss, bc_loss

        return loss_fn

    def evaluate(self, model, family, device='cpu'):
        """Evaluate all three field variables."""
        # For multi-output, model returns (c_ex, c_ey, c_hz, bias_ex, ...)
        family = family.to(device)
        jx, jy = family[:, 0], family[:, 1]
        kx, ky = family[:, 2], family[:, 3]

        with torch.no_grad():
            result = model()
            if isinstance(result, tuple) and len(result) == 2:
                coeffs, biases = result
                c_ex, c_ey, c_hz = coeffs
                bias_ex, bias_ey, bias_hz = biases
            else:
                raise ValueError("Model output format not recognized")

            W_val = model.meta_wavelet.evaluate_basis_2d(
                self.x_val.to(device), self.y_val.to(device),
                jx, jy, kx, ky
            )

            ex_pred = torch.mv(W_val, c_ex) + bias_ex
            ey_pred = torch.mv(W_val, c_ey) + bias_ey
            hz_pred = torch.mv(W_val, c_hz) + bias_hz

        return {
            'rel_l2_error_Ex': PINNLoss.relative_l2_error(
                ex_pred.cpu(), self.ex_val).item(),
            'rel_l2_error_Ey': PINNLoss.relative_l2_error(
                ey_pred.cpu(), self.ey_val).item(),
            'rel_l2_error_Hz': PINNLoss.relative_l2_error(
                hz_pred.cpu(), self.hz_val).item(),
        }
