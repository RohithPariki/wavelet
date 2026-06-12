"""
Heat Conduction Comparison — Main Results Table
=================================================
Table 1 of the paper.

Compares W-PINN, AW-PINN, and MW-PINN on the heat conduction problem
at ε = 0.15, 0.12, 0.11, 0.10.

Trains each method for n_runs independent runs and reports
mean ± std of relative L2 error.
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import set_seed, DEVICE, TrainingConfig
from core.meta_wavelet import MetaWavelet
from core.sparse_selection import SparseWaveletSelector
from core.loss_functions import PINNLoss
from core.wavelet_matrices import WaveletMatrixBuilder
from models.wpinn import WPINN, CoefficientRefinementNet
from models.mwpinn import MetaWaveletPINN, MWPINNRefinement
from problems.heat_conduction import HeatConductionProblem


def run_wpinn_baseline(problem, config, device, seed=42):
    """Run W-PINN baseline (Adam only, then Adam on CoefficientRefinement)."""
    set_seed(seed)

    family = problem.build_family().to(device)
    family_size = len(family)

    # Use a fixed Gaussian wavelet (n=1) for W-PINN baseline
    meta_w = MetaWavelet(N_H=1, init_type='gaussian').to(device)
    # Freeze the meta-wavelet — it's fixed as Gaussian
    for p in meta_w.parameters():
        p.requires_grad = False

    model = WPINN(
        n_collocation=problem.n_collocation,
        family_size=family_size,
        n_hidden_1=2, n_hidden_2=4, hidden_dim=50, input_dim=2
    ).to(device)

    loss_fn = problem.get_full_loss_function(meta_w, family, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)

    # Phase 1: Adam training
    start = time.time()
    last_c = None
    for epoch in range(config.adam_epochs):
        optimizer.zero_grad()
        c, bias = model(problem.x_coll.to(device), problem.t_coll.to(device))
        total_loss, pde_loss, bc_loss = loss_fn(c, bias)
        total_loss.backward()
        optimizer.step()
        last_c = c.detach().clone()

    # Phase 2: Coefficient refinement with Adam
    refine = CoefficientRefinementNet(last_c, model.bias.detach()).to(device)
    refine.meta_wavelet = meta_w  # Attach for evaluation
    opt2 = torch.optim.Adam(refine.parameters(), lr=1e-3)

    for epoch in range(config.adam_epochs // 2):
        opt2.zero_grad()
        c, bias = refine()
        total_loss, _, _ = loss_fn(c, bias)
        total_loss.backward()
        opt2.step()

    elapsed = time.time() - start

    # Evaluate
    with torch.no_grad():
        c, bias = refine()
        jx, jy, kx, ky = family[:, 0], family[:, 1], family[:, 2], family[:, 3]
        W_val = meta_w.evaluate_basis_2d(
            problem.x_val.to(device), problem.t_val.to(device),
            jx, jy, kx, ky
        )
        u_pred = torch.mv(W_val, c) + bias
        rel_l2 = PINNLoss.relative_l2_error(
            u_pred.cpu(), problem.u_val_exact
        ).item()

    return rel_l2, elapsed


def run_wpinn_lbfgs(problem, config, device, seed=42):
    """W-PINN + L-BFGS (improved baseline)."""
    set_seed(seed)

    family = problem.build_family().to(device)
    family_size = len(family)

    meta_w = MetaWavelet(N_H=1, init_type='gaussian').to(device)
    for p in meta_w.parameters():
        p.requires_grad = False

    model = WPINN(
        n_collocation=problem.n_collocation,
        family_size=family_size,
        n_hidden_1=2, n_hidden_2=4, hidden_dim=50, input_dim=2
    ).to(device)

    loss_fn = problem.get_full_loss_function(meta_w, family, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)

    start = time.time()
    last_c = None
    for epoch in range(config.adam_epochs):
        optimizer.zero_grad()
        c, bias = model(problem.x_coll.to(device), problem.t_coll.to(device))
        total_loss, _, _ = loss_fn(c, bias)
        total_loss.backward()
        optimizer.step()
        last_c = c.detach().clone()

    # L-BFGS refinement
    refine = CoefficientRefinementNet(last_c, model.bias.detach()).to(device)
    refine.meta_wavelet = meta_w

    opt_lbfgs = torch.optim.LBFGS(
        refine.parameters(), lr=1.0, max_iter=20,
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=50, line_search_fn='strong_wolfe'
    )

    for epoch in range(config.lbfgs_epochs):
        def closure():
            opt_lbfgs.zero_grad()
            c, bias = refine()
            total_loss, _, _ = loss_fn(c, bias)
            total_loss.backward()
            return total_loss
        opt_lbfgs.step(closure)

    elapsed = time.time() - start

    with torch.no_grad():
        c, bias = refine()
        jx, jy, kx, ky = family[:, 0], family[:, 1], family[:, 2], family[:, 3]
        W_val = meta_w.evaluate_basis_2d(
            problem.x_val.to(device), problem.t_val.to(device),
            jx, jy, kx, ky
        )
        u_pred = torch.mv(W_val, c) + bias
        rel_l2 = PINNLoss.relative_l2_error(
            u_pred.cpu(), problem.u_val_exact
        ).item()

    return rel_l2, elapsed


def run_mwpinn(problem, config, device, N_H=4, seed=42):
    """Full MW-PINN pipeline."""
    set_seed(seed)

    family = problem.build_family().to(device)
    family_size = len(family)

    model = MetaWaveletPINN(
        n_collocation=problem.n_collocation,
        family_size=family_size,
        N_H=N_H,
        lambda_l1=config.lambda_l1,
        input_dim=2,
    ).to(device)

    loss_fn = problem.get_full_loss_function(
        model.meta_wavelet, family, device
    )

    # Phase 1: Adam + L1
    param_groups = model.get_parameter_groups(
        lr_meta=config.adam_lr_meta, lr_nn=config.adam_lr
    )
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.adam_epochs, eta_min=config.adam_lr_min
    )

    start = time.time()
    last_c = None
    for epoch in range(config.adam_epochs):
        optimizer.zero_grad()
        c, bias = model.get_coefficients(
            problem.x_coll.to(device), problem.t_coll.to(device)
        )
        total_loss, pde_loss, bc_loss = loss_fn(c, bias)
        l1 = config.lambda_l1 * torch.norm(c, p=1)
        (total_loss + l1).backward()
        optimizer.step()
        scheduler.step()
        last_c = c.detach().clone()

    # Family selection
    active_idx, sparsity = SparseWaveletSelector.select_active_family(
        last_c, threshold=config.l1_threshold, verbose=False
    )
    family_pruned = family[active_idx]
    c_pruned = last_c[active_idx]

    # Phase 2: L-BFGS
    loss_fn_pruned = problem.get_full_loss_function(
        model.meta_wavelet, family_pruned, device
    )

    refine = MWPINNRefinement(
        c_pruned, model.bias.detach(), model.meta_wavelet
    ).to(device)

    opt_lbfgs = torch.optim.LBFGS(
        refine.parameters(), lr=1.0, max_iter=20,
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=50, line_search_fn='strong_wolfe'
    )

    for epoch in range(config.lbfgs_epochs):
        def closure():
            opt_lbfgs.zero_grad()
            c, bias = refine()
            total_loss, _, _ = loss_fn_pruned(c, bias)
            total_loss.backward()
            return total_loss
        opt_lbfgs.step(closure)

    elapsed = time.time() - start

    # Evaluate
    jx = family_pruned[:, 0]
    jy = family_pruned[:, 1]
    kx = family_pruned[:, 2]
    ky = family_pruned[:, 3]

    with torch.no_grad():
        c, bias = refine()
        W_val = refine.meta_wavelet.evaluate_basis_2d(
            problem.x_val.to(device), problem.t_val.to(device),
            jx, jy, kx, ky
        )
        u_pred = torch.mv(W_val, c) + bias
        rel_l2 = PINNLoss.relative_l2_error(
            u_pred.cpu(), problem.u_val_exact
        ).item()

    shape = refine.meta_wavelet.get_shape_description()

    return rel_l2, elapsed, shape, sparsity, len(active_idx)


def run_comparison_table(n_runs=5, device=None):
    """
    Main comparison: Table 1 of the paper.
    """
    if device is None:
        device = DEVICE

    config = TrainingConfig()
    config.adam_epochs = 5000
    config.lbfgs_epochs = 300

    epsilons = [0.15, 0.12, 0.11, 0.10]
    methods = ['W-PINN', 'W-PINN+LBFGS', 'MW-PINN(N_H=2)',
               'MW-PINN(N_H=3)', 'MW-PINN(N_H=4)']

    results = {eps: {m: [] for m in methods} for eps in epsilons}
    timings = {eps: {m: [] for m in methods} for eps in epsilons}

    for eps in epsilons:
        print(f"\n{'='*60}")
        print(f"ε = {eps}")
        print(f"{'='*60}")

        problem = HeatConductionProblem(epsilon=eps, device=device)

        for run in range(n_runs):
            seed = 42 + run
            print(f"\n  Run {run+1}/{n_runs} (seed={seed})")

            # W-PINN baseline
            err, t = run_wpinn_baseline(problem, config, device, seed)
            results[eps]['W-PINN'].append(err)
            timings[eps]['W-PINN'].append(t)
            print(f"    W-PINN:         {err:.4e} ({t:.1f}s)")

            # W-PINN + L-BFGS
            err, t = run_wpinn_lbfgs(problem, config, device, seed)
            results[eps]['W-PINN+LBFGS'].append(err)
            timings[eps]['W-PINN+LBFGS'].append(t)
            print(f"    W-PINN+LBFGS:   {err:.4e} ({t:.1f}s)")

            # MW-PINN variants
            for N_H in [2, 3, 4]:
                err, t, shape, spar, n_act = run_mwpinn(
                    problem, config, device, N_H=N_H, seed=seed
                )
                key = f'MW-PINN(N_H={N_H})'
                results[eps][key].append(err)
                timings[eps][key].append(t)
                print(f"    {key}: {err:.4e} ({t:.1f}s) "
                      f"[shape={shape['dominant_component']}, "
                      f"active={n_act}]")

    # Print summary table
    print("\n\n" + "=" * 80)
    print("TABLE 1: Heat Conduction Comparison")
    print("=" * 80)
    print(f"{'Method':<20s}", end="")
    for eps in epsilons:
        print(f"  {'ε='+str(eps):>16s}", end="")
    print(f"  {'Avg Time':>10s}")
    print("-" * 80)

    for method in methods:
        print(f"{method:<20s}", end="")
        for eps in epsilons:
            errs = results[eps][method]
            mean = np.mean(errs)
            std = np.std(errs)
            print(f"  {mean:.2e}±{std:.2e}", end="")
        avg_t = np.mean([np.mean(timings[eps][method]) for eps in epsilons])
        print(f"  {avg_t:>8.1f}s")

    return results, timings


if __name__ == '__main__':
    results, timings = run_comparison_table(n_runs=3, device=DEVICE)
