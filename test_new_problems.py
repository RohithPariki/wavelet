import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'meta_wavelet_pinn')))

from meta_wavelet_pinn.problems.maxwell_homogeneous import MaxwellHomogeneousProblem
from meta_wavelet_pinn.problems.lid_driven import LidDrivenProblem
from meta_wavelet_pinn.problems.spp_ex2 import SPPEx2Problem
from meta_wavelet_pinn.core.meta_wavelet import MetaWavelet

def test_problem():
    print("Testing Maxwell Homogeneous...")
    maxwell = MaxwellHomogeneousProblem(n_coll=100, n_bc=20, n_val=50, n_test=50)
    fam_maxwell = maxwell.build_family()
    mw_maxwell = MetaWavelet(N_H=2)
    loss_fn_maxwell = maxwell.get_full_loss_function(mw_maxwell, fam_maxwell)
    c_E = torch.randn(len(fam_maxwell))
    c_H = torch.randn(len(fam_maxwell))
    bias_E = torch.tensor(0.5)
    bias_H = torch.tensor(0.5)
    total_loss, pde_loss, bc_loss = loss_fn_maxwell(c_E, c_H, bias_E, bias_H)
    print(f"Maxwell OK, total_loss: {total_loss.item()}")

    print("Testing Lid-Driven Cavity Re=400...")
    lid = LidDrivenProblem(Re=400, n_coll=100, n_bc=20, n_val=50, n_test=50)
    fam_lid = lid.build_family()
    mw_lid = MetaWavelet(N_H=2)
    loss_fn_lid = lid.get_full_loss_function(mw_lid, fam_lid)
    c_u = torch.randn(len(fam_lid))
    c_v = torch.randn(len(fam_lid))
    c_p = torch.randn(len(fam_lid))
    bias_u = torch.tensor(0.5)
    bias_v = torch.tensor(0.5)
    bias_p = torch.tensor(0.5)
    total_loss, pde_loss, bc_loss = loss_fn_lid(c_u, c_v, c_p, bias_u, bias_v, bias_p)
    print(f"Lid-Driven OK, total_loss: {total_loss.item()}")

    print("Testing SPP Ex2...")
    spp = SPPEx2Problem(n_coll=100, n_bc=20, n_val=50, n_test=50)
    fam_spp = spp.build_family()
    mw_spp = MetaWavelet(N_H=2)
    loss_fn_spp = spp.get_full_loss_function(mw_spp, fam_spp)
    c_spp = torch.randn(len(fam_spp))
    bias_spp = torch.tensor(0.5)
    total_loss, pde_loss, bc_loss = loss_fn_spp(c_spp, bias_spp)
    print(f"SPP Ex2 OK, total_loss: {total_loss.item()}")

if __name__ == '__main__':
    test_problem()
    print("All tests passed!")
