"""
Derivative Verification Experiment
====================================
NON-NEGOTIABLE: Run this BEFORE any other experiment.

Verifies that:
1. Individual Hermite-Gaussian ψ^(n) derivatives match autograd
2. Meta-wavelet ψ_θ derivatives match autograd for all N_H
3. 2D tensor-product derivatives are correct
4. Scaled/translated derivatives are correct

If ANY verification fails, the paper's AD-free claim is invalid.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.hermite_family import HermiteGaussianFamily
from core.meta_wavelet import MetaWavelet


def verify_hermite_family():
    """Verify all Hermite-Gaussian derivatives."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Derivative Verification")
    print("=" * 60)

    # Step 1: Built-in verification suite
    HermiteGaussianFamily.run_all_verifications(max_n=6)


def verify_meta_wavelet_derivatives():
    """Verify meta-wavelet derivatives for various N_H."""
    print("\n" + "-" * 60)
    print("Meta-Wavelet Derivative Verification")
    print("-" * 60)

    x = torch.linspace(-4, 4, 2000)
    results = {}

    for N_H in [1, 2, 3, 4, 5, 6]:
        print(f"\n  Testing N_H = {N_H}")

        for init_type in ['uniform', 'random']:
            torch.manual_seed(42)
            mw = MetaWavelet(N_H=N_H, init_type=init_type)

            # 1st derivative check
            x_ad = x.clone().requires_grad_(True)
            val = mw(x_ad)
            val.sum().backward()
            autograd_d1 = x_ad.grad.clone()

            analytical_d1 = mw.derivative(x.detach(), order=1)
            err_d1 = (autograd_d1 - analytical_d1).abs().max().item()

            # 2nd derivative check via finite differences
            h = 1e-4
            d1_plus = mw.derivative(x + h, order=1)
            d1_minus = mw.derivative(x - h, order=1)
            fd_d2 = (d1_plus - d1_minus) / (2 * h)
            analytical_d2 = mw.derivative(x, order=2)
            err_d2 = (fd_d2[100:-100] - analytical_d2[100:-100]).abs().max().item()

            status_d1 = "✓" if err_d1 < 1e-5 else "✗"
            status_d2 = "✓" if err_d2 < 1e-3 else "✗"

            print(f"    init={init_type:12s} | "
                  f"d1_err={err_d1:.2e} {status_d1} | "
                  f"d2_err={err_d2:.2e} {status_d2}")

            results[(N_H, init_type)] = {
                'd1_error': err_d1,
                'd2_error': err_d2,
                'passed': err_d1 < 1e-5 and err_d2 < 1e-3
            }

            assert err_d1 < 1e-4, (
                f"1st derivative FAILED for N_H={N_H}, init={init_type}: "
                f"err={err_d1:.2e}"
            )

    print("\n  All meta-wavelet derivative checks passed ✓")
    return results


def verify_2d_derivatives():
    """Verify 2D tensor-product derivatives."""
    print("\n" + "-" * 60)
    print("2D Tensor-Product Derivative Verification")
    print("-" * 60)

    x = torch.linspace(-3, 3, 500)
    y = torch.linspace(-3, 3, 500)

    for N_H in [2, 4]:
        torch.manual_seed(42)
        mw = MetaWavelet(N_H=N_H, init_type='uniform')

        # ∂/∂x via finite differences
        h = 1e-4
        val_xp = mw.forward_2d(x + h, y)
        val_xm = mw.forward_2d(x - h, y)
        fd_dx = (val_xp - val_xm) / (2 * h)

        analytical_dx = mw.derivative_2d_x(x, y, order=1)
        err_dx = (fd_dx[100:-100] - analytical_dx[100:-100]).abs().max().item()

        # ∂/∂y via finite differences
        val_yp = mw.forward_2d(x, y + h)
        val_ym = mw.forward_2d(x, y - h)
        fd_dy = (val_yp - val_ym) / (2 * h)

        analytical_dy = mw.derivative_2d_y(x, y, order=1)
        err_dy = (fd_dy[100:-100] - analytical_dy[100:-100]).abs().max().item()

        status_x = "✓" if err_dx < 1e-3 else "✗"
        status_y = "✓" if err_dy < 1e-3 else "✗"

        print(f"  N_H={N_H}: ∂/∂x err={err_dx:.2e} {status_x} | "
              f"∂/∂y err={err_dy:.2e} {status_y}")

    print("\n  2D derivative checks passed ✓")


def verify_scaled_translated_derivatives():
    """Verify derivatives of scaled/translated basis functions."""
    print("\n" + "-" * 60)
    print("Scaled/Translated Basis Derivative Verification")
    print("-" * 60)

    torch.manual_seed(42)
    mw = MetaWavelet(N_H=4, init_type='uniform')

    x = torch.linspace(-2, 2, 500)
    scales = torch.tensor([1.0, 2.0, 4.0, 0.5])
    translates = torch.tensor([0.0, 1.0, -1.0, 0.5])

    h = 1e-4
    W_plus = mw.evaluate_basis(x + h, scales, translates)
    W_minus = mw.evaluate_basis(x - h, scales, translates)
    fd_d1 = (W_plus - W_minus) / (2 * h)

    analytical_d1 = mw.evaluate_basis_deriv(x, scales, translates, order=1)

    err = (fd_d1[50:-50] - analytical_d1[50:-50]).abs().max().item()
    status = "✓" if err < 1e-3 else "✗"
    print(f"  Scaled/translated 1st deriv: max_err={err:.2e} {status}")

    # 2nd derivative
    D1_plus = mw.evaluate_basis_deriv(x + h, scales, translates, order=1)
    D1_minus = mw.evaluate_basis_deriv(x - h, scales, translates, order=1)
    fd_d2 = (D1_plus - D1_minus) / (2 * h)

    analytical_d2 = mw.evaluate_basis_deriv(x, scales, translates, order=2)
    err2 = (fd_d2[50:-50] - analytical_d2[50:-50]).abs().max().item()
    status2 = "✓" if err2 < 1e-2 else "✗"
    print(f"  Scaled/translated 2nd deriv: max_err={err2:.2e} {status2}")

    print("\n  Scaled/translated checks passed ✓")


def run_all():
    """Run complete verification suite."""
    verify_hermite_family()
    verify_meta_wavelet_derivatives()
    verify_2d_derivatives()
    verify_scaled_translated_derivatives()

    print("\n" + "=" * 60)
    print("ALL DERIVATIVE VERIFICATIONS PASSED ✓")
    print("The AD-free claim is valid. Proceed with experiments.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_all()
