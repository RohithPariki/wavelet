"""
N_H Ablation Study — Key Design Experiment
============================================
Answers: "Does more Hermite components = better?"

Sweeps N_H ∈ {1, 2, 3, 4, 5, 6, 8} and reports:
  - Mean ± std relative L2 error
  - Learned wavelet shape (a_n coefficients)
  - Training time
  - Active family size after L1 selection

This produces Figure 3 of the paper.
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
from experiments.comparison_table import run_mwpinn
from problems.heat_conduction import HeatConductionProblem


def run_nh_ablation(epsilon=0.10, n_runs=10, device=None):
    """
    N_H ablation study.

    Parameters
    ----------
    epsilon : float — heat conduction problem difficulty.
    n_runs : int — independent runs per setting.
    device : str

    Returns
    -------
    results : dict mapping N_H -> {errors, shapes, times, ...}
    """
    if device is None:
        device = DEVICE

    config = TrainingConfig()
    config.adam_epochs = 5000
    config.lbfgs_epochs = 300

    problem = HeatConductionProblem(epsilon=epsilon, device=device)
    NH_values = [1, 2, 3, 4, 5, 6, 8]

    results = {}

    for N_H in NH_values:
        print(f"\n{'='*50}")
        print(f"N_H = {N_H}")
        print(f"{'='*50}")

        errors = []
        times = []
        shapes = []
        sparsities = []
        active_sizes = []

        for run in range(n_runs):
            seed = 42 + run
            err, t, shape, spar, n_act = run_mwpinn(
                problem, config, device, N_H=N_H, seed=seed
            )
            errors.append(err)
            times.append(t)
            shapes.append(shape['coefficients'])
            sparsities.append(spar)
            active_sizes.append(n_act)

            print(f"  Run {run+1}/{n_runs}: err={err:.4e}, "
                  f"time={t:.1f}s, active={n_act}")

        results[N_H] = {
            'errors': np.array(errors),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'times': np.array(times),
            'mean_time': np.mean(times),
            'shapes': np.array(shapes),
            'mean_shape': np.mean(shapes, axis=0),
            'sparsities': np.array(sparsities),
            'active_sizes': np.array(active_sizes),
        }

        print(f"\n  Summary: {np.mean(errors):.4e} ± {np.std(errors):.4e}")
        print(f"  Mean shape: {np.mean(shapes, axis=0)}")

    return results


def plot_nh_ablation(results, save_path=None):
    """
    Generate Figure 3: N_H ablation plot.

    Two panels:
      Left: Relative L2 error vs N_H (with error bars)
      Right: Average learned wavelet coefficients for each N_H
    """
    NH_values = sorted(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Error vs N_H
    means = [results[nh]['mean_error'] for nh in NH_values]
    stds = [results[nh]['std_error'] for nh in NH_values]

    ax1.errorbar(NH_values, means, yerr=stds, fmt='o-', capsize=5,
                 linewidth=2, markersize=8, color='#2563eb')
    ax1.set_yscale('log')
    ax1.set_xlabel('$N_H$ (Hermite components)', fontsize=12)
    ax1.set_ylabel('Relative $L^2$ Error', fontsize=12)
    ax1.set_title('(a) Error vs. $N_H$', fontsize=13)
    ax1.set_xticks(NH_values)
    ax1.grid(True, alpha=0.3)

    # Mark the sweet spot
    best_nh = NH_values[np.argmin(means)]
    ax1.axvline(best_nh, color='red', linestyle='--', alpha=0.5,
                label=f'Best $N_H={best_nh}$')
    ax1.legend(fontsize=11)

    # Panel 2: Learned shapes
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(NH_values)))
    bar_width = 0.12

    for i, nh in enumerate(NH_values):
        shape = results[nh]['mean_shape']
        positions = np.arange(len(shape)) + i * bar_width
        ax2.bar(positions, shape, bar_width, label=f'$N_H={nh}$',
                alpha=0.8)

    ax2.set_xlabel('Component index $n$', fontsize=12)
    ax2.set_ylabel('Coefficient $a_n$', fontsize=12)
    ax2.set_title('(b) Learned wavelet shape', fontsize=13)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(__file__), '..', 'paper', 'figures',
            'fig3_nh_ablation.pdf'
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()

    return fig


def print_ablation_table(results):
    """Print formatted ablation results."""
    print("\n" + "=" * 70)
    print("N_H ABLATION RESULTS")
    print("=" * 70)
    print(f"{'N_H':>5s} | {'Rel L2 Error':>20s} | {'Time (s)':>10s} | "
          f"{'Active Fam':>10s} | {'Dominant':>12s}")
    print("-" * 70)

    for nh in sorted(results.keys()):
        r = results[nh]
        dominant = 'N/A'
        if len(r['mean_shape']) > 0:
            names = ['Gauss', 'MexHat', '3-DOG', '4-DOG',
                     '5-DOG', '6-DOG', '7-DOG', '8-DOG']
            dom_idx = int(np.abs(r['mean_shape']).argmax())
            dominant = names[dom_idx] if dom_idx < len(names) else f'{dom_idx}-DOG'

        print(f"{nh:>5d} | {r['mean_error']:.4e} ± {r['std_error']:.4e} | "
              f"{r['mean_time']:>8.1f}s | "
              f"{np.mean(r['active_sizes']):>8.0f} | "
              f"{dominant:>12s}")


if __name__ == '__main__':
    results = run_nh_ablation(epsilon=0.10, n_runs=5, device=DEVICE)
    plot_nh_ablation(results)
    print_ablation_table(results)
