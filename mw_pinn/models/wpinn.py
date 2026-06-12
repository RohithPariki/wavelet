"""
W-PINN: Wavelet-based Physics-Informed Neural Network (Baseline)
=================================================================
Clean reimplementation of the W-PINN from Pandey et al. (2026).

Architecture:
  1. Feature network (shallow NN): (x, t) → latent feature per point
  2. Coefficient network (deep NN): features → wavelet coefficients c
  3. Fixed wavelet reconstruction: u_hat = W @ c + bias

This serves as Baseline 1 in all comparisons.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class WPINN(nn.Module):
    """
    Two-stage W-PINN architecture for 2D problems.

    Stage 1 (feature network):
        Maps each (x, y) or (x, t) point through a shallow MLP to
        a scalar feature. Output shape: [n_collocation]

    Stage 2 (coefficient network):
        Maps the feature vector through a deep MLP to produce
        wavelet coefficients. Output shape: [family_size]

    Parameters
    ----------
    n_collocation : int
        Number of collocation points (input dim for stage 2).
    family_size : int
        Number of wavelet basis functions (output dim).
    n_hidden_1 : int
        Hidden layers in feature network.
    n_hidden_2 : int
        Hidden layers in coefficient network.
    hidden_dim : int
        Width of hidden layers.
    input_dim : int
        Spatial dimension (1 for 1D problems, 2 for 2D).
    """

    def __init__(self, n_collocation: int, family_size: int,
                 n_hidden_1: int = 2, n_hidden_2: int = 4,
                 hidden_dim: int = 50, input_dim: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.activation = nn.Tanh()

        # ── Stage 1: Feature network ───────────────────────────
        if input_dim >= 2:
            layers_1 = [nn.Linear(input_dim, hidden_dim), self.activation]
            for _ in range(n_hidden_1 - 1):
                layers_1.extend([nn.Linear(hidden_dim, hidden_dim),
                                 self.activation])
            layers_1.append(nn.Linear(hidden_dim, 1))
            self.feature_net = nn.Sequential(*layers_1)
        else:
            # For 1D problems, skip feature network
            self.feature_net = None

        # ── Stage 2: Coefficient network ───────────────────────
        layers_2 = [nn.Linear(n_collocation, hidden_dim), self.activation]
        for _ in range(n_hidden_2 - 1):
            layers_2.extend([nn.Linear(hidden_dim, hidden_dim),
                             self.activation])
        layers_2.append(nn.Linear(hidden_dim, family_size))
        self.coeff_net = nn.Sequential(*layers_2)

        # ── Trainable bias ─────────────────────────────────────
        self.bias = nn.Parameter(torch.tensor(0.5))

        # ── Weight initialization ──────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization (as in the original paper)."""
        modules = [self.coeff_net]
        if self.feature_net is not None:
            modules.append(self.feature_net)

        for net in modules:
            for m in net:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)

    def forward(self, *coords):
        """
        Forward pass.

        Parameters
        ----------
        *coords : variable number of coordinate tensors.
            For 2D: (x, y) or (x, t), each [n_collocation].
            For 1D: (t,), shape [n_collocation].

        Returns
        -------
        coefficients : [family_size]
        bias : scalar
        """
        if self.input_dim >= 2:
            inputs = torch.stack(coords, dim=-1)  # [N, d]
            features = self.feature_net(inputs).squeeze(-1)  # [N]
        else:
            features = coords[0].reshape(-1)  # [N]

        coefficients = self.coeff_net(features)  # [family_size]
        return coefficients, self.bias


class CoefficientRefinementNet(nn.Module):
    """
    Post-training coefficient refinement (from W-PINN paper).

    Freezes the neural network and directly optimizes the wavelet
    coefficients and bias as learnable parameters.
    This is ideal for L-BFGS optimization.

    Parameters
    ----------
    initial_coefficients : [family_size] tensor from W-PINN training.
    initial_bias : scalar tensor.
    """

    def __init__(self, initial_coefficients: torch.Tensor,
                 initial_bias: torch.Tensor):
        super().__init__()
        self.coefficients = nn.Parameter(initial_coefficients.clone().detach())
        self.bias = nn.Parameter(initial_bias.clone().detach())

    def forward(self, *coords):
        """Returns the learnable coefficients and bias (ignores inputs)."""
        return self.coefficients, self.bias


class WPINNMultiOutput(nn.Module):
    """
    Multi-output W-PINN for systems of PDEs (e.g., Navier-Stokes, Maxwell).

    Shares a single feature network across all outputs, with separate
    coefficient networks for each field variable.

    Parameters
    ----------
    n_outputs : int
        Number of field variables (e.g., 3 for u, v, p in Lid-Driven).
    n_collocation, family_size, etc. : same as WPINN.
    """

    def __init__(self, n_outputs: int, n_collocation: int,
                 family_size: int, n_hidden_1: int = 2,
                 n_hidden_2: int = 4, hidden_dim: int = 50,
                 input_dim: int = 2):
        super().__init__()

        self.n_outputs = n_outputs
        self.activation = nn.Tanh()

        # Shared feature network
        layers_1 = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_hidden_1 - 1):
            layers_1.extend([nn.Linear(hidden_dim, hidden_dim),
                             self.activation])
        layers_1.append(nn.Linear(hidden_dim, 1))
        self.feature_net = nn.Sequential(*layers_1)

        # Separate coefficient networks
        self.coeff_nets = nn.ModuleList()
        for _ in range(n_outputs):
            layers = [nn.Linear(n_collocation, hidden_dim), self.activation]
            for _ in range(n_hidden_2 - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim),
                               self.activation])
            layers.append(nn.Linear(hidden_dim, family_size))
            self.coeff_nets.append(nn.Sequential(*layers))

        # Per-output biases
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(n_outputs)]
        )

        self._init_weights()

    def _init_weights(self):
        for net in [self.feature_net] + list(self.coeff_nets):
            for m in net:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)

    def forward(self, *coords):
        inputs = torch.stack(coords, dim=-1)
        features = self.feature_net(inputs).squeeze(-1)

        coeffs = tuple(net(features) for net in self.coeff_nets)
        biases = tuple(b for b in self.biases)

        return coeffs, biases
