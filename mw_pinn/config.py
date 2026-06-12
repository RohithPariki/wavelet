"""
Global configuration for Meta-Wavelet PINN.
All device, seed, and hyperparameter defaults live here.
"""

import torch
import numpy as np
import random


# ── Device Detection ──────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


DEVICE = get_device()
print(f"[Config] Using device: {DEVICE}")


# ── Reproducibility ──────────────────────────────────────────────
DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ── Default Hyperparameters ──────────────────────────────────────
class TrainingConfig:
    """Default training configuration. Override per-experiment."""

    # Phase 1: Adam pre-training
    adam_epochs: int = 5000
    adam_lr: float = 1e-4
    adam_lr_meta: float = 1e-3       # Higher LR for meta-wavelet a_n
    adam_lr_min: float = 1e-6        # Cosine annealing floor

    # Phase 2: L-BFGS fine-tuning
    lbfgs_epochs: int = 500
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 50

    # Wavelet family
    N_H: int = 4                     # Hermite components
    lambda_l1: float = 1e-4          # L1 sparsity weight
    l1_threshold: float = 1e-8       # Post-training sparsity cutoff

    # Logging
    log_every: int = 100
    eval_every: int = 500
    n_runs: int = 5                  # Independent runs for statistics

    def __repr__(self):
        attrs = {k: v for k, v in vars(type(self)).items()
                 if not k.startswith('_') and not callable(v)}
        lines = [f"  {k} = {v}" for k, v in attrs.items()]
        return "TrainingConfig(\n" + "\n".join(lines) + "\n)"
