"""
PDE Loss Functions (AD-Free)
==============================
All loss computations use precomputed wavelet matrices and analytical
derivatives. No autograd is used for PDE residual computation.

Structure:
    u_hat = W @ c + bias          (reconstruction)
    u_t   = D1Wt @ c             (time derivative)
    u_xx  = D2Wx @ c             (second spatial derivative)
    PDE residual = u_t - u_xx - f (example for heat equation)
"""

import torch


class PINNLoss:
    """
    Generic PINN loss computation using wavelet matrices.

    All derivatives are obtained via matrix-vector products with
    precomputed derivative matrices — no autograd.
    """

    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean squared error."""
        return torch.mean((pred - target) ** 2)

    @staticmethod
    def relative_l2_error(pred: torch.Tensor,
                          exact: torch.Tensor) -> torch.Tensor:
        """
        Relative L2 error: ||u - û||₂ / ||u||₂

        This is the primary evaluation metric used in both papers.
        """
        return (torch.norm(exact - pred, p=2)
                / (torch.norm(exact, p=2) + 1e-12))

    @staticmethod
    def max_error(pred: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
        """Maximum absolute pointwise error."""
        return torch.max(torch.abs(exact - pred))

    # ── Wavelet-based reconstruction ───────────────────────────

    @staticmethod
    def reconstruct(W: torch.Tensor, c: torch.Tensor,
                    bias: torch.Tensor) -> torch.Tensor:
        """u_hat = W @ c + bias"""
        return torch.mv(W, c) + bias

    @staticmethod
    def reconstruct_deriv(DW: torch.Tensor,
                          c: torch.Tensor) -> torch.Tensor:
        """Derivative: DW @ c (bias drops out)."""
        return torch.mv(DW, c)

    # ── Problem-specific losses ────────────────────────────────

    @staticmethod
    def heat_equation_loss(c, bias, W, DWt, DW2x, rhs,
                           W_ic, u_ic, W_bc_left, u_bc_left,
                           W_bc_right, u_bc_right):
        """
        Heat equation: u_t = u_xx + f

        Returns
        -------
        total_loss, pde_loss, ic_loss, bc_loss
        """
        # PDE residual
        u_t = torch.mv(DWt, c)
        u_xx = torch.mv(DW2x, c)
        pde_loss = torch.mean((u_t - u_xx - rhs) ** 2)

        # Initial condition
        u_pred_ic = torch.mv(W_ic, c) + bias
        ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)

        # Boundary conditions
        u_pred_bc_l = torch.mv(W_bc_left, c) + bias
        u_pred_bc_r = torch.mv(W_bc_right, c) + bias
        bc_loss = (torch.mean((u_pred_bc_l - u_bc_left) ** 2)
                   + torch.mean((u_pred_bc_r - u_bc_right) ** 2))

        total_loss = pde_loss + ic_loss + bc_loss
        return total_loss, pde_loss, ic_loss, bc_loss

    @staticmethod
    def helmholtz_loss(c, bias, W, DW2x, DW2y, rhs,
                       W_bc_left, u_bc_left, W_bc_right, u_bc_right,
                       W_bc_bottom, u_bc_bottom, W_bc_top, u_bc_top):
        """
        Helmholtz equation: u_xx + u_yy + u = f

        Returns
        -------
        total_loss, pde_loss, bc_loss
        """
        u = torch.mv(W, c) + bias
        u_xx = torch.mv(DW2x, c)
        u_yy = torch.mv(DW2y, c)

        pde_loss = torch.mean((u_xx + u_yy + u - rhs) ** 2)

        # All four boundary edges
        bc_loss = (
            torch.mean((torch.mv(W_bc_left, c) + bias - u_bc_left) ** 2)
            + torch.mean((torch.mv(W_bc_right, c) + bias - u_bc_right) ** 2)
            + torch.mean((torch.mv(W_bc_bottom, c) + bias - u_bc_bottom) ** 2)
            + torch.mean((torch.mv(W_bc_top, c) + bias - u_bc_top) ** 2)
        )

        total_loss = pde_loss + bc_loss
        return total_loss, pde_loss, bc_loss

    @staticmethod
    def poisson_loss(c, bias, W, DW2x, DW2y, rhs,
                     bc_matrices, bc_values):
        """
        Poisson equation: u_xx + u_yy = f

        Parameters
        ----------
        bc_matrices : list of [N_bc, F] matrices for each boundary segment.
        bc_values   : list of [N_bc] tensors with exact BC values.

        Returns
        -------
        total_loss, pde_loss, bc_loss
        """
        u_xx = torch.mv(DW2x, c)
        u_yy = torch.mv(DW2y, c)

        pde_loss = torch.mean((u_xx + u_yy - rhs) ** 2)

        bc_loss = torch.tensor(0.0, device=c.device)
        for W_bc, u_bc in zip(bc_matrices, bc_values):
            u_pred_bc = torch.mv(W_bc, c) + bias
            bc_loss = bc_loss + torch.mean((u_pred_bc - u_bc) ** 2)

        total_loss = pde_loss + bc_loss
        return total_loss, pde_loss, bc_loss

    @staticmethod
    def maxwell_loss_homogeneous(c_ex, c_ey, c_hz,
                                 bias_ex, bias_ey, bias_hz,
                                 W, DWx, DWy,
                                 W_bc_list, bc_vals_ex, bc_vals_ey,
                                 bc_vals_hz, omega=1.0):
        """
        Homogeneous Maxwell TEz equations:
            ∂Hz/∂y = -iω Ex
            -∂Hz/∂x = -iω Ey
            ∂Ex/∂y - ∂Ey/∂x = iω Hz

        Simplified for real-valued fields.

        Returns
        -------
        total_loss, pde_loss, bc_loss
        """
        # Reconstruct field derivatives
        dHz_dy = torch.mv(DWy, c_hz)
        dHz_dx = torch.mv(DWx, c_hz)
        dEx_dy = torch.mv(DWy, c_ex)
        dEy_dx = torch.mv(DWx, c_ey)

        Ex = torch.mv(W, c_ex) + bias_ex
        Ey = torch.mv(W, c_ey) + bias_ey
        Hz = torch.mv(W, c_hz) + bias_hz

        # PDE residuals
        res1 = dHz_dy + omega * Ex
        res2 = -dHz_dx + omega * Ey
        res3 = dEx_dy - dEy_dx - omega * Hz

        pde_loss = (torch.mean(res1 ** 2)
                    + torch.mean(res2 ** 2)
                    + torch.mean(res3 ** 2))

        # Boundary conditions (combined)
        bc_loss = torch.tensor(0.0, device=c_ex.device)
        for W_bc in W_bc_list:
            bc_loss += torch.mean((torch.mv(W_bc, c_ex) + bias_ex) ** 2)
            bc_loss += torch.mean((torch.mv(W_bc, c_ey) + bias_ey) ** 2)

        total_loss = pde_loss + bc_loss
        return total_loss, pde_loss, bc_loss

    @staticmethod
    def spp_loss(c, bias, W, DW1, DW2, rhs, u_ic, du_ic,
                 W_ic, DW_ic, epsilon):
        """
        Singularly perturbed problem:
            ε u'' + b(t) u' + g(u) = f(t)

        Returns
        -------
        total_loss, pde_loss, ic_loss
        """
        u = torch.mv(W, c) + bias
        u_t = torch.mv(DW1, c)
        u_tt = torch.mv(DW2, c)

        pde_loss = torch.mean((epsilon * u_tt + u_t - rhs) ** 2)

        # Initial conditions
        u_pred_ic = torch.mv(W_ic, c) + bias
        du_pred_ic = torch.mv(DW_ic, c)

        ic_loss = (torch.mean((u_pred_ic - u_ic) ** 2)
                   + torch.mean((du_pred_ic - du_ic) ** 2))

        total_loss = pde_loss + ic_loss
        return total_loss, pde_loss, ic_loss
