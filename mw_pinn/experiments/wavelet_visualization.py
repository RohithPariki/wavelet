"""
Wavelet Shape Visualization — THE KEY FIGURE (Figure 1)
=========================================================
Trains MW-PINN on each PDE type, extracts the learned wavelet shape,
and shows that the meta-wavelet automatically discovers the optimal
basis for each problem type.

This is the most compelling visual result of the paper.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import set_seed, DEVICE, TrainingConfig
from core.hermite_family import HermiteGaussianFamily
from core.meta_wavelet import MetaWavelet
from core.sparse_selection import SparseWaveletSelector
from core.loss_functions import PINNLoss
from models.mwpinn import MetaWaveletPINN, MWPINNRefinement
from problems.heat_conduction import HeatConductionProblem
from problems.poisson import PoissonProblem


def train_and_extract_shape(problem, config, device, N_H=4, seed=42):
    """
    Train MW-PINN on a problem and return the learned wavelet shape
    plus its training history.
    """
    set_seed(seed)

    family = problem.build_family().to(device)
    family_size = len(family)

    model = MetaWaveletPINN(
        n_collocation=problem.n_collocation,
        family_size=family_size,
        N_H=N_H,
        lambda_l1=config.lambda_l1,
        input_dim=problem.input_dim,
    ).to(device)

    loss_fn = problem.get_full_loss_function(
        model.meta_wavelet, family, device
    )

    param_groups = model.get_parameter_groups(
        lr_meta=config.adam_lr_meta, lr_nn=config.adam_lr
    )
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.adam_epochs, eta_min=config.adam_lr_min
    )

    shape_history = []
    loss_history = []

    for epoch in range(config.adam_epochs):
        optimizer.zero_grad()
        c, bias = model.get_coefficients(
            *[getattr(problem, f'x_coll', getattr(problem, 'x_coll', None)),
              getattr(problem, f't_coll', getattr(problem, 'y_coll', None))]
        )
        total_loss, pde_loss, bc_loss = loss_fn(c, bias)
        l1 = config.lambda_l1 * torch.norm(c, p=1)
        (total_loss + l1).backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            shape_history.append(
                model.meta_wavelet.a.detach().cpu().numpy().copy()
            )
            loss_history.append(total_loss.item())

    # Final shape
    final_shape = model.meta_wavelet.get_shape_description()

    return {
        'shape_history': np.array(shape_history),
        'loss_history': np.array(loss_history),
        'final_shape': final_shape,
        'meta_wavelet': model.meta_wavelet,
    }


def generate_figure_1(device=None, N_H=4):
    """
    Generate Figure 1: The most important figure.

    2×N grid showing:
      Top row: Learned wavelet shape overlaid on fixed wavelets
      Bottom row: Evolution of a_n coefficients during training
    """
    # Force CPU to avoid MPS Out-Of-Memory errors on large dense matrices
    device = torch.device('cpu')

    config = TrainingConfig()
    config.adam_epochs = 1000  # Enough to see shape convergence, fast enough for interactive run

    # Define problems
    problems = {
        'Heat Conduction\n(ε=0.15)': HeatConductionProblem(
            epsilon=0.15, device=device
        ),
        'Heat Conduction\n(ε=0.10)': HeatConductionProblem(
            epsilon=0.10, device=device
        ),
        'Poisson\n(localized source)': PoissonProblem(
            sigma=500.0, device=device
        ),
    }

    n_problems = len(problems)
    fig, axes = plt.subplots(2, n_problems, figsize=(5 * n_problems, 8))

    if n_problems == 1:
        axes = axes.reshape(2, 1)

    x_plot = torch.linspace(-4, 4, 500)
    hg = HermiteGaussianFamily()

    # Fixed wavelet references
    psi_gaussian = hg.psi(x_plot, 1).numpy()
    psi_mexican = hg.psi(x_plot, 2).numpy()

    for idx, (name, problem) in enumerate(problems.items()):
        print(f"\nTraining on: {name}")

        result = train_and_extract_shape(
            problem, config, device, N_H=N_H
        )

        # ── Top panel: Learned wavelet shape ───────────────────
        ax = axes[0, idx]

        with torch.no_grad():
            psi_learned = result['meta_wavelet'].to('cpu')(x_plot).numpy()

        ax.plot(x_plot.numpy(), psi_learned, 'b-', linewidth=2.5,
                label='Learned $\\psi_\\theta$', zorder=5)
        ax.plot(x_plot.numpy(), psi_gaussian, 'r--', alpha=0.4,
                linewidth=1.5, label='Gaussian ($n$=1)')
        ax.plot(x_plot.numpy(), psi_mexican, 'g--', alpha=0.4,
                linewidth=1.5, label='Mex. Hat ($n$=2)')

        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\psi(x)$')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-4, 4)

        # ── Bottom panel: Coefficient evolution ────────────────
        ax = axes[1, idx]
        shape_hist = result['shape_history']
        component_names = [f'$a_{n+1}$' for n in range(N_H)]
        colors_cycle = ['#2563eb', '#dc2626', '#16a34a', '#9333ea',
                        '#ea580c', '#0891b2', '#4f46e5', '#be185d']

        for n in range(min(N_H, len(shape_hist[0]))):
            epochs = np.arange(len(shape_hist)) * 50
            ax.plot(epochs, shape_hist[:, n],
                    label=component_names[n],
                    color=colors_cycle[n % len(colors_cycle)],
                    linewidth=1.5)

        ax.set_xlabel('Training epoch')
        ax.set_ylabel('Coefficient value')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.2)

        # Print final shape
        print(f"  Final shape: {result['final_shape']}")

    fig.suptitle(
        'Meta-Wavelet PINN: Learned wavelet shapes per PDE type\n'
        '(Top: learned vs. fixed wavelets | Bottom: coefficient evolution)',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    save_path = os.path.join(
        os.path.dirname(__file__), '..', 'paper', 'figures',
        'fig1_learned_wavelets.pdf'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure 1 saved: {save_path}")

    # Also save PNG for quick preview
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    return fig


if __name__ == '__main__':
    generate_figure_1(device=DEVICE, N_H=4)
