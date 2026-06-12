"""
MW-PINN: Meta-Wavelet Physics-Informed Neural Network
======================================================
THE MAIN CONTRIBUTION.

Key innovation: learns the optimal mother wavelet shape via
Hermite-Gaussian parameterization, simultaneously with scale,
translation, and wavelet coefficients.

ψ_θ(x) = Σ_{n=1}^{N_H} a_n · ψ^(n)(x)

where ψ^(n) are Hermite-Gaussian wavelets with the crucial property:
    d/dx[ψ^(n)] = ψ^(n+1)

This enables fully analytical derivative computation while the wavelet
shape adapts to the problem. No autograd needed for PDE residuals.

Architecture:
  1. Feature network (shallow NN) → latent features
  2. Coefficient network (deep NN) → wavelet coefficients c
  3. Meta-wavelet layer (LEARNED) → adaptive basis evaluation
  4. Linear reconstruction → solution u_hat = W_θ @ c + bias

Training protocol:
  Phase 1 (Adam + L1): Learn sparse structure + wavelet shape
  Phase 2 (L-BFGS): Fine-tune all parameters to convergence
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.meta_wavelet import MetaWavelet
from core.sparse_selection import SparseWaveletSelector
from core.loss_functions import PINNLoss


class MetaWaveletPINN(nn.Module):
    """
    Full Meta-Wavelet PINN model.

    Combines a two-stage neural network (from W-PINN) with a learnable
    meta-wavelet that replaces the fixed Gaussian/Mexican Hat basis.

    Parameters
    ----------
    n_collocation : int
        Number of collocation points.
    family_size : int
        Number of wavelet basis functions.
    N_H : int
        Number of Hermite-Gaussian components in the meta-wavelet.
    lambda_l1 : float
        L1 regularization weight for sparsity.
    n_hidden_1, n_hidden_2 : int
        Hidden layers in feature/coefficient networks.
    hidden_dim : int
        Width of hidden layers.
    input_dim : int
        Spatial dimension (1 or 2).
    init_type : str
        Meta-wavelet initialization: 'uniform', 'gaussian',
        'mexican_hat', 'random'.
    """

    def __init__(self, n_collocation: int, family_size: int,
                 N_H: int = 4, lambda_l1: float = 1e-4,
                 n_hidden_1: int = 2, n_hidden_2: int = 4,
                 hidden_dim: int = 50, input_dim: int = 2,
                 init_type: str = 'uniform'):
        super().__init__()

        self.n_collocation = n_collocation
        self.family_size = family_size
        self.N_H = N_H
        self.lambda_l1 = lambda_l1
        self.input_dim = input_dim
        self.activation = nn.Tanh()

        # ── The meta-wavelet (shared across all basis functions) ──
        self.meta_wavelet = MetaWavelet(N_H=N_H, init_type=init_type)

        # ── Feature network (Stage 1) ────────────────────────────
        if input_dim >= 2:
            layers_1 = [nn.Linear(input_dim, hidden_dim), self.activation]
            for _ in range(n_hidden_1 - 1):
                layers_1.extend([nn.Linear(hidden_dim, hidden_dim),
                                 self.activation])
            layers_1.append(nn.Linear(hidden_dim, 1))
            self.feature_net = nn.Sequential(*layers_1)
        else:
            self.feature_net = None

        # ── Coefficient network (Stage 2) ─────────────────────────
        layers_2 = [nn.Linear(n_collocation, hidden_dim), self.activation]
        for _ in range(n_hidden_2 - 1):
            layers_2.extend([nn.Linear(hidden_dim, hidden_dim),
                             self.activation])
        layers_2.append(nn.Linear(hidden_dim, family_size))
        self.coeff_net = nn.Sequential(*layers_2)

        # ── Trainable bias ────────────────────────────────────────
        self.bias = nn.Parameter(torch.tensor(0.5))

        # ── Initialization ────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for NN layers."""
        modules = [self.coeff_net]
        if self.feature_net is not None:
            modules.append(self.feature_net)
        for net in modules:
            for m in net:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)

    def get_coefficients(self, *coords):
        """
        Run the two-stage network to produce wavelet coefficients.

        Returns
        -------
        c : [family_size] tensor
        bias : scalar tensor
        """
        if self.input_dim >= 2:
            inputs = torch.stack(coords, dim=-1)
            features = self.feature_net(inputs).squeeze(-1)
        else:
            features = coords[0].reshape(-1)

        c = self.coeff_net(features)
        return c, self.bias

    def build_matrices(self, coords_dict, family):
        """
        Build wavelet matrices using the current meta-wavelet.

        This is called at the beginning of each training step
        (or periodically) to recompute W, DW etc. with the
        current learned wavelet shape.

        Parameters
        ----------
        coords_dict : dict with coordinate tensors.
        family : [F, 2*dim] tensor with family parameters.

        Returns
        -------
        dict of wavelet matrices for each point set.
        """
        from core.wavelet_matrices import WaveletMatrixBuilder
        builder = WaveletMatrixBuilder(self.meta_wavelet,
                                       device=family.device)

        if self.input_dim == 1:
            return builder.build_1d(coords_dict['collocation'],
                                    family)
        else:
            return builder.build_2d(
                coords_dict['x_collocation'],
                coords_dict['y_collocation'],
                family
            )

    def forward_with_matrices(self, W, c, bias):
        """
        Reconstruct solution from precomputed matrices.

        u_hat = W @ c + bias

        Parameters
        ----------
        W : [N, F] wavelet basis matrix.
        c : [F] coefficients.
        bias : scalar.

        Returns
        -------
        [N] solution values.
        """
        return torch.mv(W, c) + bias

    def get_parameter_groups(self, lr_meta=1e-3, lr_nn=1e-4):
        """
        Return parameter groups with different learning rates.

        The meta-wavelet coefficients a_n get a higher learning rate
        than the neural network weights.
        """
        meta_params = list(self.meta_wavelet.parameters())
        nn_params = [p for n, p in self.named_parameters()
                     if 'meta_wavelet' not in n]

        return [
            {'params': meta_params, 'lr': lr_meta},
            {'params': nn_params, 'lr': lr_nn},
        ]


class MWPINNRefinement(nn.Module):
    """
    Post-selection refinement model for MW-PINN.

    After L1-based family selection, this model directly optimizes:
      - Wavelet coefficients c (pruned to active set)
      - Meta-wavelet shape parameters a_n
      - Bias

    The neural network is frozen; only the final parameters are tuned.
    This is ideal for L-BFGS optimization.

    Parameters
    ----------
    initial_coefficients : [N_A] from pre-training (active only).
    initial_bias : scalar.
    meta_wavelet : MetaWavelet instance (shared, continues training).
    """

    def __init__(self, initial_coefficients: torch.Tensor,
                 initial_bias: torch.Tensor,
                 meta_wavelet: MetaWavelet):
        super().__init__()
        self.coefficients = nn.Parameter(
            initial_coefficients.clone().detach()
        )
        self.bias = nn.Parameter(initial_bias.clone().detach())
        # The meta-wavelet keeps training in this phase
        self.meta_wavelet = meta_wavelet

    def forward(self, *coords):
        """Returns coefficients and bias."""
        return self.coefficients, self.bias


# ═══════════════════════════════════════════════════════════════════
#  Two-Phase Training Protocol
# ═══════════════════════════════════════════════════════════════════

def train_mwpinn_phase1(model, loss_fn, coords, family, config,
                         device='cpu', verbose=True):
    """
    Phase 1: Adam training with L1 regularization.

    Learns:
      - Neural network weights (feature + coefficient networks)
      - Meta-wavelet shape parameters a_n
      - Bias

    The L1 penalty on coefficients drives sparsity for subsequent
    family selection.

    Parameters
    ----------
    model : MetaWaveletPINN
    loss_fn : callable(c, bias, matrices_dict) -> (loss, pde, bc, ...)
    coords : dict of coordinate tensors.
    family : [F, 2*dim] wavelet family parameters.
    config : TrainingConfig instance.
    device : torch device.

    Returns
    -------
    model : trained model
    history : dict of training logs
    """
    from core.wavelet_matrices import WaveletMatrixBuilder

    model = model.to(device)
    family = family.to(device)

    # Separate parameter groups for meta-wavelet vs NN
    param_groups = model.get_parameter_groups(
        lr_meta=config.adam_lr_meta,
        lr_nn=config.adam_lr
    )
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.adam_epochs, eta_min=config.adam_lr_min
    )

    builder = WaveletMatrixBuilder(model.meta_wavelet, device=device)

    history = {
        'total_loss': [], 'pde_loss': [], 'bc_loss': [],
        'l1_loss': [], 'wavelet_shape': [], 'lr': [],
    }

    start_time = time.time()
    last_c = None

    for epoch in range(config.adam_epochs):
        optimizer.zero_grad()

        # Get coefficients from the two-stage network
        # The W-PINN architecture uses the collocation points as the input feature vector
        if model.input_dim == 1:
            c, bias = model.get_coefficients(coords['t_collocation'])
        else:
            x_key = 'x_collocation' if 'x_collocation' in coords else 'x'
            y_key = 't_collocation' if 't_collocation' in coords else ('y_collocation' if 'y_collocation' in coords else 'y')
            c, bias = model.get_coefficients(coords[x_key], coords[y_key])

        # Rebuild wavelet matrices with current meta-wavelet shape
        # (meta-wavelet parameters change each epoch)
        matrices = _build_matrices_for_problem(
            builder, coords, family, model.input_dim
        )

        # PDE + BC loss (AD-free)
        total_loss, pde_loss, bc_loss = loss_fn(c, bias, matrices)

        # L1 sparsity penalty
        l1_loss = model.lambda_l1 * torch.norm(c, p=1)
        augmented_loss = total_loss + l1_loss

        augmented_loss.backward()
        optimizer.step()
        scheduler.step()

        last_c = c.detach().clone()

        # Logging
        if epoch % config.log_every == 0:
            history['total_loss'].append(total_loss.item())
            history['pde_loss'].append(pde_loss.item())
            if isinstance(bc_loss, torch.Tensor):
                history['bc_loss'].append(bc_loss.item())
            else:
                history['bc_loss'].append(bc_loss)
            history['l1_loss'].append(l1_loss.item())
            history['wavelet_shape'].append(
                model.meta_wavelet.a.detach().cpu().numpy().copy()
            )
            history['lr'].append(optimizer.param_groups[0]['lr'])

            if verbose and epoch % config.eval_every == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch:5d}/{config.adam_epochs} | "
                      f"Loss: {total_loss.item():.4e} | "
                      f"PDE: {pde_loss.item():.4e} | "
                      f"L1: {l1_loss.item():.4e} | "
                      f"Shape: {model.meta_wavelet.a.detach().cpu().numpy()} | "
                      f"Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Phase 1 complete: {elapsed:.1f}s total")
        print(f"  Final wavelet shape: "
              f"{model.meta_wavelet.get_shape_description()}")

    return model, last_c, history


def train_mwpinn_phase2(refinement_model, loss_fn, coords, family,
                         config, input_dim=2, device='cpu', verbose=True):
    """
    Phase 2: L-BFGS fine-tuning of selected coefficients + wavelet shape.

    Parameters
    ----------
    refinement_model : MWPINNRefinement (with pruned coefficients).
    loss_fn : callable(c, bias, matrices_dict) -> (loss, pde, bc)
    coords : dict
    family : [N_A, 2*dim] pruned family.
    config : TrainingConfig
    device : str

    Returns
    -------
    refinement_model : fine-tuned model.
    history : dict of L-BFGS logs.
    """
    from core.wavelet_matrices import WaveletMatrixBuilder

    refinement_model = refinement_model.to(device)
    family = family.to(device)

    optimizer = torch.optim.LBFGS(
        refinement_model.parameters(),
        lr=config.lbfgs_lr,
        max_iter=config.lbfgs_max_iter,
        max_eval=config.lbfgs_max_iter + 5,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=config.lbfgs_history_size,
        line_search_fn='strong_wolfe'
    )

    builder = WaveletMatrixBuilder(
        refinement_model.meta_wavelet, device=device
    )

    history = {'total_loss': [], 'wavelet_shape': []}
    start_time = time.time()

    for epoch in range(config.lbfgs_epochs):
        def closure():
            optimizer.zero_grad()
            c, bias = refinement_model()

            matrices = _build_matrices_for_problem(
                builder, coords, family, input_dim
            )

            total_loss, pde_loss, bc_loss = loss_fn(c, bias, matrices)
            total_loss.backward()
            return total_loss

        loss_val = optimizer.step(closure)

        if epoch % max(1, config.lbfgs_epochs // 20) == 0:
            history['total_loss'].append(loss_val.item()
                                         if isinstance(loss_val, torch.Tensor)
                                         else loss_val)
            history['wavelet_shape'].append(
                refinement_model.meta_wavelet.a.detach().cpu().numpy().copy()
            )
            if verbose:
                elapsed = time.time() - start_time
                print(f"  L-BFGS epoch {epoch:4d}/{config.lbfgs_epochs} | "
                      f"Loss: {loss_val:.4e} | Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Phase 2 complete: {elapsed:.1f}s total")

    return refinement_model, history


def _build_matrices_for_problem(builder, coords, family, input_dim):
    """
    Dispatch matrix building based on problem dimension.

    Returns a dict of matrices needed by the loss function.
    """
    if input_dim == 1:
        W, D1W, D2W = builder.build_1d(coords['t_collocation'], family)
        return {'W': W, 'D1W': D1W, 'D2W': D2W}
    else:
        # 2D: figure out coordinate keys
        x_key = 'x_collocation' if 'x_collocation' in coords else 'x'
        y_key = ('t_collocation' if 't_collocation' in coords
                 else 'y_collocation' if 'y_collocation' in coords
                 else 'y')

        W, D1Wx, D2Wx, D1Wy, D2Wy = builder.build_2d(
            coords[x_key], coords[y_key], family
        )
        return {
            'W': W, 'D1Wx': D1Wx, 'D2Wx': D2Wx,
            'D1Wy': D1Wy, 'D2Wy': D2Wy
        }


# ═══════════════════════════════════════════════════════════════════
#  Full Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_mwpinn_full_pipeline(problem, config, device='cpu', seed=42,
                              verbose=True):
    """
    Complete MW-PINN training pipeline:
      1. Build wavelet family
      2. Phase 1: Adam + L1 (learn shape + sparse structure)
      3. Family selection (automatic via L1 threshold)
      4. Phase 2: L-BFGS (fine-tune coefficients + shape)
      5. Evaluate on test set

    Parameters
    ----------
    problem : A problem instance (e.g., HeatConductionProblem).
    config : TrainingConfig instance.
    device : str
    seed : int

    Returns
    -------
    results : dict with errors, timing, wavelet shape info.
    """
    from config import set_seed
    set_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"MW-PINN Pipeline (N_H={config.N_H}, lambda_L1={config.lambda_l1})")
        print(f"{'='*60}")

    # ── Step 1: Build family ──────────────────────────────────
    family = problem.build_family()
    coords = problem.get_coordinates()
    family_size = len(family)

    if verbose:
        print(f"\n[1] Family size: {family_size}")

    # ── Step 2: Create model ──────────────────────────────────
    n_coll = problem.n_collocation
    model = MetaWaveletPINN(
        n_collocation=n_coll,
        family_size=family_size,
        N_H=config.N_H,
        lambda_l1=config.lambda_l1,
        input_dim=problem.input_dim,
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[2] Model params: {n_params:,}")

    # ── Step 3: Phase 1 — Adam ────────────────────────────────
    if verbose:
        print(f"\n[3] Phase 1: Adam ({config.adam_epochs} epochs)")

    loss_fn = problem.get_loss_function()

    model, last_c, history_1 = train_mwpinn_phase1(
        model, loss_fn, coords, family, config,
        device=device, verbose=verbose
    )

    # ── Step 4: Family selection ──────────────────────────────
    if verbose:
        print(f"\n[4] Sparse family selection")

    active_idx, sparsity = SparseWaveletSelector.select_active_family(
        last_c, threshold=config.l1_threshold, verbose=verbose
    )

    # Prune family
    family_pruned = family[active_idx]
    c_pruned = last_c[active_idx]

    # ── Step 5: Phase 2 — L-BFGS ─────────────────────────────
    if verbose:
        print(f"\n[5] Phase 2: L-BFGS ({config.lbfgs_epochs} epochs)")

    refinement = MWPINNRefinement(
        c_pruned, model.bias.detach(), model.meta_wavelet
    ).to(device)

    loss_fn_pruned = problem.get_loss_function(pruned=True)

    refinement, history_2 = train_mwpinn_phase2(
        refinement, loss_fn_pruned, coords, family_pruned,
        config, input_dim=problem.input_dim, device=device, verbose=verbose
    )

    # ── Step 6: Evaluate ──────────────────────────────────────
    if verbose:
        print(f"\n[6] Evaluation")

    results = problem.evaluate(refinement, family_pruned, device)

    results['sparsity'] = sparsity
    results['active_family_size'] = len(active_idx)
    results['original_family_size'] = family_size
    results['wavelet_shape'] = refinement.meta_wavelet.get_shape_description()
    results['history_phase1'] = history_1
    results['history_phase2'] = history_2

    if verbose:
        print(f"\n  Relative L2 Error: {results.get('rel_l2_error', 'N/A'):.4e}")
        print(f"  Active family: {results['active_family_size']} / "
              f"{results['original_family_size']}")
        print(f"  Learned wavelet: {results['wavelet_shape']}")
        print(f"{'='*60}\n")

    return results
