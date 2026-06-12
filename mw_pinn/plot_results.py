import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from core.hermite_family import HermiteGaussianFamily

def generate_result_figures():
    """Generate visualization from the actual benchmark results."""
    # These are the exact coefficients learned during the benchmark test!
    shape_nh2 = np.array([0.71771777, 0.7148457])
    shape_nh4 = np.array([0.02404217, 0.02867269, 0.47962126, 0.47085106])
    
    # Normalize for visual comparison
    shape_nh2 = shape_nh2 / np.linalg.norm(shape_nh2)
    shape_nh4 = shape_nh4 / np.linalg.norm(shape_nh4)

    x = torch.linspace(-5, 5, 500)
    hg = HermiteGaussianFamily()

    psi_fixed_1 = hg.psi(x, 1).numpy()  # Gaussian wavelet
    psi_fixed_2 = hg.psi(x, 2).numpy()  # Mexican Hat wavelet

    # Compute learned shapes
    psi_learned_2 = sum(shape_nh2[i] * hg.psi(x, i+1).numpy() for i in range(2))
    psi_learned_4 = sum(shape_nh4[i] * hg.psi(x, i+1).numpy() for i in range(4))
    
    # Scale to match visual height of fixed wavelets for easy comparison
    psi_learned_2 = psi_learned_2 / np.max(np.abs(psi_learned_2)) * np.max(np.abs(psi_fixed_1))
    psi_learned_4 = psi_learned_4 / np.max(np.abs(psi_learned_4)) * np.max(np.abs(psi_fixed_2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: N_H = 2
    ax1.plot(x.numpy(), psi_fixed_1, 'r--', alpha=0.5, linewidth=2, label='Fixed Gaussian (W-PINN)')
    ax1.plot(x.numpy(), psi_learned_2, 'b-', linewidth=3, label='Learned MW-PINN ($N_H=2$)')
    ax1.set_title('Learned Wavelet vs Gaussian Baseline', fontsize=14, fontweight='bold')
    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel(r'$\psi(x)$', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)

    # Panel 2: N_H = 4
    ax1.plot(x.numpy(), psi_fixed_2, 'g--', alpha=0.5, linewidth=2, label='Fixed Mexican Hat')
    
    ax2.plot(x.numpy(), psi_fixed_1, 'r--', alpha=0.4, linewidth=2, label='Fixed Gaussian')
    ax2.plot(x.numpy(), psi_fixed_2, 'g--', alpha=0.4, linewidth=2, label='Fixed Mexican Hat')
    ax2.plot(x.numpy(), psi_learned_4, '#8b5cf6', linewidth=3, label='Learned MW-PINN ($N_H=4$)')
    ax2.set_title('Higher-Order Discovery (Dominant: 3rd-DOG)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('$x$', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)

    fig.suptitle('Meta-Wavelet PINN: Automatically Discovered Wavelet Shapes', 
                 fontsize=18, fontweight='bold', y=1.05)

    plt.tight_layout()
    
    save_path = '/Users/jaipreethtiruvaipati/.gemini/antigravity-ide/brain/1c913928-9b2d-4225-8f79-fa9f08acf95e/fig_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    generate_result_figures()
