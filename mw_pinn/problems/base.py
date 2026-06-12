"""
Problem Base Class
===================
Abstract interface that all PDE problems implement.
"""

import torch
from abc import ABC, abstractmethod


class BaseProblem(ABC):
    """
    Base class for all PDE benchmark problems.

    Each problem defines:
      - Domain bounds and collocation point generation
      - Analytical solution (for error computation)
      - PDE right-hand side and boundary values
      - Wavelet family construction
      - Loss function (AD-free, using wavelet matrices)
      - Evaluation on test set
    """

    def __init__(self, device='cpu'):
        self.device = device

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Spatial dimension (1 or 2)."""
        ...

    @property
    @abstractmethod
    def n_collocation(self) -> int:
        """Number of collocation points."""
        ...

    @abstractmethod
    def build_family(self) -> torch.Tensor:
        """Build wavelet family index set. Returns [F, 2*dim] tensor."""
        ...

    @abstractmethod
    def get_coordinates(self) -> dict:
        """Return dict of all coordinate tensors (collocation, BC, IC, etc.)."""
        ...

    @abstractmethod
    def get_loss_function(self, pruned=False):
        """
        Return a callable: loss_fn(c, bias, matrices_dict) -> (total, pde, bc)

        If pruned=True, returns the loss function for the pruned family.
        """
        ...

    @abstractmethod
    def evaluate(self, model, family, device) -> dict:
        """
        Evaluate the model on test set.

        Returns dict with at least 'rel_l2_error' and 'max_error'.
        """
        ...

    @abstractmethod
    def analytical_solution(self, *coords) -> torch.Tensor:
        """Exact/reference solution for error computation."""
        ...
