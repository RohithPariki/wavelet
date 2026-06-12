"""
End-to-End Benchmark: MW-PINN vs W-PINN vs W-PINN+LBFGS
=========================================================
Streamlined script that runs all key experiments and produces
a clear verdict on whether MW-PINN improves results.

Tests on Heat Conduction (primary benchmark) at multiple ε values.
Reports: Relative L2 Error, Training Time, Family Size.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from core.hermite_family import HermiteGaussianFamily
from core.meta_wavelet import MetaWavelet
from core.wavelet_matrices import build_wavelet_family_2d
from core.sparse_selection import SparseWaveletSelector
from core.loss_functions import PINNLoss
from problems.heat_conduction import HeatConductionProblem

# ── Device ─────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════
#  HELPER: Build wavelet matrices using a meta-wavelet
# ═══════════════════════════════════════════════════════════════

def build_all_matrices(meta_w, family, problem, device):
    """Build all wavelet matrices for the heat conduction problem."""
    fam = family.to(device)
    jx, jy, kx, ky = fam[:, 0], fam[:, 1], fam[:, 2], fam[:, 3]

    x_c = problem.x_coll.to(device)
    t_c = problem.t_coll.to(device)

    W = meta_w.evaluate_basis_2d(x_c, t_c, jx, jy, kx, ky)
    DWt = meta_w.evaluate_basis_2d_dy(x_c, t_c, jx, jy, kx, ky, order=1)
    DW2x = meta_w.evaluate_basis_2d_dx(x_c, t_c, jx, jy, kx, ky, order=2)

    W_ic = meta_w.evaluate_basis_2d(
        problem.x_ic.to(device), problem.t_ic.to(device), jx, jy, kx, ky
    )
    W_bc_l = meta_w.evaluate_basis_2d(
        problem.x_bc_left.to(device), problem.t_bc.to(device), jx, jy, kx, ky
    )
    W_bc_r = meta_w.evaluate_basis_2d(
        problem.x_bc_right.to(device), problem.t_bc.to(device), jx, jy, kx, ky
    )

    return {
        'W': W, 'DWt': DWt, 'DW2x': DW2x,
        'W_ic': W_ic, 'W_bc_l': W_bc_l, 'W_bc_r': W_bc_r,
    }


def compute_heat_loss(c, bias, mats, rhs, u_ic, u_bc_l, u_bc_r):
    """Heat equation AD-free loss."""
    u_t = torch.mv(mats['DWt'], c)
    u_xx = torch.mv(mats['DW2x'], c)
    pde_loss = torch.mean((u_t - u_xx - rhs) ** 2)

    u_pred_ic = torch.mv(mats['W_ic'], c) + bias
    ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)

    u_pred_bc_l = torch.mv(mats['W_bc_l'], c) + bias
    u_pred_bc_r = torch.mv(mats['W_bc_r'], c) + bias
    bc_loss = (torch.mean((u_pred_bc_l - u_bc_l) ** 2)
               + torch.mean((u_pred_bc_r - u_bc_r) ** 2))

    return pde_loss + ic_loss + bc_loss, pde_loss, ic_loss, bc_loss


def evaluate_model(c, bias, meta_w, family, problem, device):
    """Compute relative L2 error on validation set."""
    fam = family.to(device)
    jx, jy, kx, ky = fam[:, 0], fam[:, 1], fam[:, 2], fam[:, 3]

    with torch.no_grad():
        W_val = meta_w.evaluate_basis_2d(
            problem.x_val.to(device), problem.t_val.to(device),
            jx, jy, kx, ky
        )
        u_pred = torch.mv(W_val, c) + bias
        rel_l2 = PINNLoss.relative_l2_error(
            u_pred.cpu(), problem.u_val_exact
        ).item()
        max_err = PINNLoss.max_error(
            u_pred.cpu(), problem.u_val_exact
        ).item()
    return rel_l2, max_err


# ═══════════════════════════════════════════════════════════════
#  Two-Stage Network
# ═══════════════════════════════════════════════════════════════

class TwoStageNet(nn.Module):
    """Shared two-stage network for all methods."""
    def __init__(self, n_coll, family_size, n_h1=2, n_h2=4, hdim=50):
        super().__init__()
        act = nn.Tanh()

        # Feature net
        layers1 = [nn.Linear(2, hdim), act]
        for _ in range(n_h1 - 1):
            layers1 += [nn.Linear(hdim, hdim), act]
        layers1.append(nn.Linear(hdim, 1))
        self.feat = nn.Sequential(*layers1)

        # Coeff net
        layers2 = [nn.Linear(n_coll, hdim), act]
        for _ in range(n_h2 - 1):
            layers2 += [nn.Linear(hdim, hdim), act]
        layers2.append(nn.Linear(hdim, family_size))
        self.coeff = nn.Sequential(*layers2)

        self.bias = nn.Parameter(torch.tensor(0.5))

        for net in [self.feat, self.coeff]:
            for m in net:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)

    def forward(self, x, t):
        inp = torch.stack([x, t], dim=-1)
        feat = self.feat(inp).squeeze(-1)
        c = self.coeff(feat)
        return c, self.bias


class CoeffRefine(nn.Module):
    """Direct coefficient optimization for L-BFGS phase."""
    def __init__(self, c_init, b_init):
        super().__init__()
        self.c = nn.Parameter(c_init.clone().detach())
        self.b = nn.Parameter(b_init.clone().detach())

    def forward(self):
        return self.c, self.b


# ═══════════════════════════════════════════════════════════════
#  METHOD 1: W-PINN (Fixed Gaussian, Adam only)
# ═══════════════════════════════════════════════════════════════

def run_wpinn(problem, family, adam_epochs=5000, refine_epochs=5000,
              device=DEVICE, seed=42):
    torch.manual_seed(seed)
    family_size = len(family)

    # Fixed Gaussian wavelet (n=1): φ_1(x) = x·exp(-x²/2)
    meta_w = MetaWavelet(N_H=1, init_type='gaussian').to(device)
    for p in meta_w.parameters():
        p.requires_grad = False

    rhs = problem.rhs.to(device)
    u_ic = problem.u_ic.to(device)
    u_bc_l = problem.u_bc_left.to(device)
    u_bc_r = problem.u_bc_right.to(device)
    x_c = problem.x_coll.to(device)
    t_c = problem.t_coll.to(device)

    # Precompute matrices (fixed wavelet → build once)
    mats = build_all_matrices(meta_w, family, problem, device)

    model = TwoStageNet(problem.n_collocation, family_size).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    start = time.time()

    # Phase 1: Adam on NN
    for ep in range(adam_epochs):
        opt.zero_grad()
        c, bias = model(x_c, t_c)
        loss, _, _, _ = compute_heat_loss(c, bias, mats, rhs, u_ic, u_bc_l, u_bc_r)
        loss.backward()
        opt.step()

    last_c = c.detach().clone()
    last_b = model.bias.detach().clone()

    # Phase 2: Adam on coefficients directly
    refine = CoeffRefine(last_c, last_b).to(device)
    opt2 = optim.Adam(refine.parameters(), lr=1e-3)

    for ep in range(refine_epochs):
        opt2.zero_grad()
        c, bias = refine()
        loss, _, _, _ = compute_heat_loss(c, bias, mats, rhs, u_ic, u_bc_l, u_bc_r)
        loss.backward()
        opt2.step()

    elapsed = time.time() - start
    c_final, b_final = refine()
    rel_l2, max_err = evaluate_model(c_final.detach(), b_final.detach(),
                                      meta_w, family, problem, device)

    return {
        'method': 'W-PINN (Adam)',
        'rel_l2': rel_l2, 'max_err': max_err,
        'time': elapsed, 'family_size': family_size,
    }


# ═══════════════════════════════════════════════════════════════
#  METHOD 2: W-PINN + L-BFGS
# ═══════════════════════════════════════════════════════════════

def run_wpinn_lbfgs(problem, family, adam_epochs=5000, lbfgs_epochs=300,
                     device=DEVICE, seed=42):
    torch.manual_seed(seed)
    family_size = len(family)

    meta_w = MetaWavelet(N_H=1, init_type='gaussian').to(device)
    for p in meta_w.parameters():
        p.requires_grad = False

    rhs = problem.rhs.to(device)
    u_ic = problem.u_ic.to(device)
    u_bc_l = problem.u_bc_left.to(device)
    u_bc_r = problem.u_bc_right.to(device)
    x_c = problem.x_coll.to(device)
    t_c = problem.t_coll.to(device)

    mats = build_all_matrices(meta_w, family, problem, device)

    model = TwoStageNet(problem.n_collocation, family_size).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    start = time.time()

    for ep in range(adam_epochs):
        opt.zero_grad()
        c, bias = model(x_c, t_c)
        loss, _, _, _ = compute_heat_loss(c, bias, mats, rhs, u_ic, u_bc_l, u_bc_r)
        loss.backward()
        opt.step()

    last_c = c.detach().clone()
    last_b = model.bias.detach().clone()

    # L-BFGS refinement
    refine = CoeffRefine(last_c, last_b).to(device)
    opt_lbfgs = optim.LBFGS(
        refine.parameters(), lr=1.0, max_iter=20,
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=50, line_search_fn='strong_wolfe'
    )

    for ep in range(lbfgs_epochs):
        def closure():
            opt_lbfgs.zero_grad()
            c, bias = refine()
            loss, _, _, _ = compute_heat_loss(c, bias, mats, rhs, u_ic, u_bc_l, u_bc_r)
            loss.backward()
            return loss
        opt_lbfgs.step(closure)

    elapsed = time.time() - start
    c_final, b_final = refine()
    rel_l2, max_err = evaluate_model(c_final.detach(), b_final.detach(),
                                      meta_w, family, problem, device)

    return {
        'method': 'W-PINN + L-BFGS',
        'rel_l2': rel_l2, 'max_err': max_err,
        'time': elapsed, 'family_size': family_size,
    }


# ═══════════════════════════════════════════════════════════════
#  METHOD 3: MW-PINN (Learnable wavelet shape + L1 + L-BFGS)
# ═══════════════════════════════════════════════════════════════

def run_mwpinn(problem, family, N_H=4, lambda_l1=1e-4,
               adam_epochs=5000, lbfgs_epochs=300,
               device=DEVICE, seed=42):
    torch.manual_seed(seed)
    family_size = len(family)

    # Learnable meta-wavelet
    meta_w = MetaWavelet(N_H=N_H, init_type='uniform').to(device)

    rhs = problem.rhs.to(device)
    u_ic = problem.u_ic.to(device)
    u_bc_l = problem.u_bc_left.to(device)
    u_bc_r = problem.u_bc_right.to(device)
    x_c = problem.x_coll.to(device)
    t_c = problem.t_coll.to(device)
    fam = family.to(device)

    model = TwoStageNet(problem.n_collocation, family_size).to(device)

    # Separate LR for meta-wavelet vs NN
    opt = optim.Adam([
        {'params': meta_w.parameters(), 'lr': 1e-3},
        {'params': model.parameters(), 'lr': 1e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=adam_epochs, eta_min=1e-6
    )

    start = time.time()
    shape_history = []

    # Phase 1: Adam + L1
    for ep in range(adam_epochs):
        opt.zero_grad()

        # Rebuild matrices each step (meta-wavelet changes)
        mats = build_all_matrices(meta_w, fam, problem, device)

        c, bias = model(x_c, t_c)
        loss, pde, ic, bc = compute_heat_loss(
            c, bias, mats, rhs, u_ic, u_bc_l, u_bc_r
        )
        l1 = lambda_l1 * torch.norm(c, p=1)
        (loss + l1).backward()
        opt.step()
        scheduler.step()

        if ep % 500 == 0:
            shape_history.append(meta_w.a.detach().cpu().numpy().copy())
            if ep % 1000 == 0:
                print(f"    [MW ep {ep:5d}] loss={loss.item():.4e} "
                      f"pde={pde.item():.4e} shape={meta_w.a.detach().cpu().numpy()}")

    last_c = c.detach().clone()
    last_b = model.bias.detach().clone()

    # Family selection
    active_idx, sparsity = SparseWaveletSelector.select_active_family(
        last_c, threshold=1e-6, min_keep=20, verbose=True
    )
    fam_pruned = fam[active_idx]
    c_pruned = last_c[active_idx]

    # Rebuild matrices for pruned family
    mats_pruned = build_all_matrices(meta_w, fam_pruned, problem, device)
    for k in mats_pruned:
        mats_pruned[k] = mats_pruned[k].detach()

    # Phase 2: L-BFGS on pruned coefficients (freeze meta-wavelet shape)
    refine = CoeffRefine(c_pruned, last_b).to(device)

    opt_lbfgs = optim.LBFGS(
        refine.parameters(), lr=1.0, max_iter=20,
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=50, line_search_fn='strong_wolfe'
    )

    for ep in range(lbfgs_epochs):
        def closure():
            opt_lbfgs.zero_grad()
            c, bias = refine()
            loss, _, _, _ = compute_heat_loss(
                c, bias, mats_pruned, rhs, u_ic, u_bc_l, u_bc_r
            )
            loss.backward()
            return loss
        opt_lbfgs.step(closure)

    elapsed = time.time() - start
    c_final, b_final = refine()
    rel_l2, max_err = evaluate_model(
        c_final.detach(), b_final.detach(),
        meta_w, fam_pruned, problem, device
    )

    shape_info = meta_w.get_shape_description()

    return {
        'method': f'MW-PINN (N_H={N_H})',
        'rel_l2': rel_l2, 'max_err': max_err,
        'time': elapsed,
        'family_size': family_size,
        'active_size': len(active_idx),
        'sparsity': sparsity,
        'wavelet_shape': shape_info,
        'shape_history': shape_history,
    }


# ═══════════════════════════════════════════════════════════════
#  MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════

def run_full_benchmark():
    print("=" * 70)
    print("META-WAVELET PINN — FULL BENCHMARK")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Use smaller family for faster experiments on CPU
    if DEVICE.type == 'cpu' or DEVICE.type == 'mps':
        Jx_range = (-4, 4)
        Jt_range = (-4, 4)
        n_coll = 5000
        adam_epochs = 300  # Drastically reduced for fast preview
        lbfgs_epochs = 200
        refine_epochs = 300  # Drastically reduced for fast preview
        n_runs = 1  # Fast preliminary test
    else:
        Jx_range = (-6, 6)
        Jt_range = (-6, 6)
        n_coll = 10000
        adam_epochs = 5000
        lbfgs_epochs = 300
        refine_epochs = 5000
        n_runs = 5

    gamma = 0.2
    epsilons = [0.15]  # Only one difficulty for the fast test

    all_results = {}

    for eps in epsilons:
        print(f"\n{'='*70}")
        print(f"  ε = {eps}")
        print(f"{'='*70}")

        problem = HeatConductionProblem(
            epsilon=eps, n_coll=n_coll, n_bc=500,
            Jx_range=Jx_range, Jt_range=Jt_range,
            gamma=gamma, device='cpu'  # points generated on CPU
        )
        family = problem.build_family()
        print(f"  Family size: {len(family)}")

        method_results = {}

        for run in range(n_runs):
            seed = 42 + run
            print(f"\n  ── Run {run+1}/{n_runs} (seed={seed}) ──")

            # Method 1: W-PINN
            print(f"  [1/4] W-PINN (Adam)...")
            r1 = run_wpinn(problem, family, adam_epochs=adam_epochs,
                           refine_epochs=refine_epochs, device=DEVICE, seed=seed)
            print(f"        L2={r1['rel_l2']:.4e}, time={r1['time']:.1f}s")
            method_results.setdefault('W-PINN', []).append(r1)

            # Method 2: W-PINN + L-BFGS
            print(f"  [2/4] W-PINN + L-BFGS...")
            r2 = run_wpinn_lbfgs(problem, family, adam_epochs=adam_epochs,
                                  lbfgs_epochs=lbfgs_epochs, device=DEVICE, seed=seed)
            print(f"        L2={r2['rel_l2']:.4e}, time={r2['time']:.1f}s")
            method_results.setdefault('W-PINN+LBFGS', []).append(r2)

            # Method 3: MW-PINN N_H=2
            print(f"  [3/4] MW-PINN (N_H=2)...")
            r3 = run_mwpinn(problem, family, N_H=2, adam_epochs=adam_epochs,
                             lbfgs_epochs=lbfgs_epochs, device=DEVICE, seed=seed)
            print(f"        L2={r3['rel_l2']:.4e}, time={r3['time']:.1f}s, "
                  f"active={r3['active_size']}, shape={r3['wavelet_shape']['dominant_component']}")
            method_results.setdefault('MW-PINN(NH=2)', []).append(r3)

            # Method 4: MW-PINN N_H=4
            print(f"  [4/4] MW-PINN (N_H=4)...")
            r4 = run_mwpinn(problem, family, N_H=4, adam_epochs=adam_epochs,
                             lbfgs_epochs=lbfgs_epochs, device=DEVICE, seed=seed)
            print(f"        L2={r4['rel_l2']:.4e}, time={r4['time']:.1f}s, "
                  f"active={r4['active_size']}, shape={r4['wavelet_shape']['dominant_component']}")
            method_results.setdefault('MW-PINN(NH=4)', []).append(r4)

        all_results[eps] = method_results

    # ── RESULTS TABLE ──────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("                      RESULTS SUMMARY")
    print("=" * 90)

    header = f"{'Method':<22s}"
    for eps in epsilons:
        header += f" | {'ε='+str(eps):^24s}"
    header += f" | {'Avg Time':>10s}"
    print(header)
    print("-" * 90)

    methods_order = ['W-PINN', 'W-PINN+LBFGS', 'MW-PINN(NH=2)', 'MW-PINN(NH=4)']

    for method in methods_order:
        line = f"{method:<22s}"
        times = []
        for eps in epsilons:
            runs = all_results[eps][method]
            errs = [r['rel_l2'] for r in runs]
            mean_e = np.mean(errs)
            std_e = np.std(errs)
            line += f" | {mean_e:.2e} ± {std_e:.2e}"
            times.extend([r['time'] for r in runs])
        line += f" | {np.mean(times):>8.1f}s"
        print(line)

    # ── ANALYSIS ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("                      ANALYSIS")
    print("=" * 90)

    for eps in epsilons:
        print(f"\n  ε = {eps}:")
        runs = all_results[eps]

        wpinn_errs = [r['rel_l2'] for r in runs['W-PINN']]
        lbfgs_errs = [r['rel_l2'] for r in runs['W-PINN+LBFGS']]
        mw2_errs = [r['rel_l2'] for r in runs['MW-PINN(NH=2)']]
        mw4_errs = [r['rel_l2'] for r in runs['MW-PINN(NH=4)']]

        best_baseline = min(np.mean(wpinn_errs), np.mean(lbfgs_errs))
        best_mw = min(np.mean(mw2_errs), np.mean(mw4_errs))

        improvement = best_baseline / max(best_mw, 1e-20)

        if best_mw < best_baseline:
            print(f"    ✓ MW-PINN WINS: {improvement:.1f}x better than best baseline")
        else:
            print(f"    ✗ Baseline wins: MW-PINN is {1/improvement:.1f}x worse")

        # Sparsity info
        for mw_key in ['MW-PINN(NH=2)', 'MW-PINN(NH=4)']:
            sparsities = [r.get('sparsity', 0) for r in runs[mw_key]]
            active = [r.get('active_size', 0) for r in runs[mw_key]]
            orig = [r.get('family_size', 0) for r in runs[mw_key]]
            print(f"    {mw_key}: avg active family = {np.mean(active):.0f} / "
                  f"{np.mean(orig):.0f} ({np.mean(sparsities)*100:.1f}% pruned)")

        # Learned wavelet shapes
        for mw_key in ['MW-PINN(NH=2)', 'MW-PINN(NH=4)']:
            shapes = [r['wavelet_shape']['coefficients'] for r in runs[mw_key]]
            avg_shape = np.mean(shapes, axis=0)
            dominant = runs[mw_key][0]['wavelet_shape']['dominant_component']
            print(f"    {mw_key}: learned shape = {avg_shape} "
                  f"(dominant: {dominant})")

    # Timing comparison
    print(f"\n  Timing comparison:")
    for method in methods_order:
        all_times = []
        for eps in epsilons:
            all_times.extend([r['time'] for r in all_results[eps][method]])
        print(f"    {method:<22s}: {np.mean(all_times):.1f}s avg")

    # ── VERDICT ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("                      VERDICT")
    print("=" * 90)

    # Check if MW-PINN wins on at least one epsilon
    wins_accuracy = 0
    wins_speed = 0
    for eps in epsilons:
        runs = all_results[eps]
        best_bl = min(np.mean([r['rel_l2'] for r in runs['W-PINN']]),
                      np.mean([r['rel_l2'] for r in runs['W-PINN+LBFGS']]))
        best_mw = min(np.mean([r['rel_l2'] for r in runs['MW-PINN(NH=2)']]),
                      np.mean([r['rel_l2'] for r in runs['MW-PINN(NH=4)']]))
        if best_mw < best_bl * 0.9:  # 10% improvement threshold
            wins_accuracy += 1

        bl_time = np.mean([r['time'] for r in runs['W-PINN+LBFGS']])
        mw_time = min(np.mean([r['time'] for r in runs['MW-PINN(NH=2)']]),
                      np.mean([r['time'] for r in runs['MW-PINN(NH=4)']]))
        if mw_time < bl_time * 0.9:
            wins_speed += 1

    print(f"\n  Accuracy wins: {wins_accuracy}/{len(epsilons)} epsilon values")
    print(f"  Speed wins:    {wins_speed}/{len(epsilons)} epsilon values")

    if wins_accuracy > 0:
        print("\n  ✅ MW-PINN shows accuracy improvement — PAPER IS VIABLE!")
        print("  The learnable wavelet shape provides measurable benefit.")
    elif wins_speed > 0:
        print("\n  ⚡ MW-PINN shows speed improvement (via sparsity)")
        print("  — Paper can focus on computational efficiency angle.")
    else:
        print("\n  ⚠️  No clear improvement yet.")
        print("  Consider: more epochs, different λ_L1, or harder problems (ε=0.10).")

    print("\n" + "=" * 90)
    return all_results


if __name__ == '__main__':
    results = run_full_benchmark()
