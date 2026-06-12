"""
L1-Based Sparse Wavelet Family Selection
==========================================
Replaces AW-PINN's manual threshold/κ selection with principled
compressed-sensing-style automatic sparsity.

This is the second contribution claim:
"Principled L1-based basis selection vs. AW-PINN's heuristic threshold tuning"

The idea:
1. During Adam pre-training, add L1 penalty on wavelet coefficients.
2. After pre-training, coefficients below a threshold are exactly zero
   (or near-zero due to soft-thresholding effect of L1).
3. Select active family = indices where |c_i| > threshold.
4. Fine-tune only the active family with L-BFGS.
"""

import torch
import numpy as np


class SparseWaveletSelector:
    """
    Automatic wavelet family selection via L1 regularization.
    """

    @staticmethod
    def l1_penalty(coefficients: torch.Tensor) -> torch.Tensor:
        """
        L1 norm of coefficients. Add this to the loss during pre-training:
            total_loss = pde_loss + bc_loss + λ · l1_penalty(c)
        """
        return torch.norm(coefficients, p=1)

    @staticmethod
    def add_l1_to_loss(loss: torch.Tensor, coefficients: torch.Tensor,
                       lambda_l1: float) -> torch.Tensor:
        """Convenience: augmented loss with L1 sparsity penalty."""
        return loss + lambda_l1 * torch.norm(coefficients, p=1)

    @staticmethod
    def select_active_family(coefficients: torch.Tensor,
                             threshold: float = 1e-8,
                             min_keep: int = 10,
                             verbose: bool = True) -> tuple:
        """
        Zero-threshold selection after L1-regularized training.
        No manual κ or score threshold needed.

        Parameters
        ----------
        coefficients : [F] tensor of learned wavelet coefficients.
        threshold : float — magnitude cutoff for "active".
        min_keep : int — minimum number of bases to keep (safety).
        verbose : bool — print selection statistics.

        Returns
        -------
        active_indices : [N_A] long tensor of selected indices.
        sparsity : float — fraction of pruned bases.
        """
        c = coefficients.detach().abs()

        # Primary selection: above threshold
        active_mask = c > threshold

        # Safety: ensure we keep at least min_keep
        if active_mask.sum() < min_keep:
            topk = torch.topk(c, min(min_keep, len(c)))
            active_mask = torch.zeros_like(c, dtype=torch.bool)
            active_mask[topk.indices] = True

        active_indices = torch.where(active_mask)[0]
        sparsity = 1.0 - active_mask.float().mean().item()

        if verbose:
            print(f"[SparseSelector] Selected {active_mask.sum().item()} / "
                  f"{len(coefficients)} families "
                  f"(sparsity: {sparsity:.1%})")
            print(f"  Max |c|: {c.max().item():.4e}, "
                  f"Min selected |c|: {c[active_mask].min().item():.4e}")

        return active_indices, sparsity

    @staticmethod
    def similarity_score_selection(coefficients: torch.Tensor,
                                   wavelet_matrix: torch.Tensor,
                                   rhs: torch.Tensor,
                                   score_threshold: float = 0.1,
                                   top_k: int = 100,
                                   verbose: bool = True) -> torch.Tensor:
        """
        AW-PINN-style similarity-based selection (for comparison baseline).

        Computes alignment between each wavelet's response and the PDE
        right-hand side, then selects families with high alignment.

        Parameters
        ----------
        coefficients : [F] learned coefficients.
        wavelet_matrix : [N, F] basis matrix W.
        rhs : [N] PDE right-hand side at collocation points.
        score_threshold : float — normalized score cutoff.
        top_k : int — also keep top-k by coefficient magnitude.

        Returns
        -------
        active_indices : [N_A] long tensor.
        """
        c = coefficients.detach()

        # Response of each wavelet: R_i = c_i * W[:,i]
        responses = c[None, :] * wavelet_matrix  # [N, F]

        # Alignment with RHS
        scores = torch.mv(responses.T, rhs)       # [F]
        scores_norm = scores / (scores.abs().max() + 1e-12)

        # Score-based selection
        score_mask = scores_norm.abs() > score_threshold

        # Top-k by coefficient magnitude
        topk_idx = torch.topk(c.abs(), min(top_k, len(c))).indices

        # Combine
        combined_mask = score_mask.clone()
        combined_mask[topk_idx] = True

        active_indices = torch.where(combined_mask)[0]

        if verbose:
            print(f"[SimilaritySelector] Selected {active_indices.shape[0]} / "
                  f"{len(coefficients)} families")
            print(f"  Score-based: {score_mask.sum().item()}, "
                  f"Top-k: {len(topk_idx)}")

        return active_indices

    @staticmethod
    def prune_matrices(active_indices: torch.Tensor, *matrices):
        """
        Prune wavelet matrices to keep only active columns.

        Parameters
        ----------
        active_indices : [N_A] indices to keep.
        *matrices : variable number of [N, F] matrices to prune.

        Returns
        -------
        tuple of pruned [N, N_A] matrices.
        """
        return tuple(m[:, active_indices] for m in matrices)

    @staticmethod
    def lambda_sweep(train_fn, lambdas=None, verbose=True):
        """
        Find optimal L1 strength via validation error.
        Run once per problem type.

        Parameters
        ----------
        train_fn : callable(lambda_l1) -> (val_error, family_size)
            A function that trains a model with given L1 weight
            and returns the validation error and active family size.
        lambdas : list of float — L1 weights to test.

        Returns
        -------
        dict mapping lambda -> (val_error, family_size)
        """
        if lambdas is None:
            lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        results = {}
        for lam in lambdas:
            val_err, fam_size = train_fn(lam)
            results[lam] = (val_err, fam_size)
            if verbose:
                print(f"  λ={lam:.1e}: L2_err={val_err:.4e}, "
                      f"family_size={fam_size}")

        # Best by validation error
        best_lam = min(results, key=lambda k: results[k][0])
        if verbose:
            print(f"  Best λ = {best_lam:.1e}")

        return results, best_lam
