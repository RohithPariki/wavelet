import os
import sys
import torch
import time
from datetime import datetime

# Make sure mw_pinn is in the path
sys.path.insert(0, os.path.dirname(__file__))

from mw_pinn.config import TrainingConfig
from mw_pinn.models.mwpinn import run_mwpinn_full_pipeline
from mw_pinn.problems.heat_conduction import HeatConductionProblem
from mw_pinn.problems.lid_driven import LidDrivenProblem
from mw_pinn.problems.maxwell_heterogeneous import MaxwellProblem
from mw_pinn.problems.maxwell_homogeneous import MaxwellHomogeneousProblem
from mw_pinn.problems.poisson import PoissonProblem
from mw_pinn.problems.spp_ex1 import SPPEx1Problem
from mw_pinn.problems.spp_ex2 import SPPEx2Problem

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting MW-PINN Experiments on Device: {device}")
    
    if device == 'cpu':
        print("\n[WARNING] CUDA is not available. Running on CPU will be extremely slow or cause Out-of-Memory errors due to large wavelet matrices. Please ensure your 64GB GPU is active before running.\n")
    else:
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}\n")

    # Define all problems to run
    experiments = [
        {"name": "Heat Conduction", "problem_class": HeatConductionProblem, "kwargs": {"epsilon": 0.15}},
        {"name": "Maxwell Heterogeneous", "problem_class": MaxwellProblem, "kwargs": {}},
        {"name": "Maxwell Homogeneous", "problem_class": MaxwellHomogeneousProblem, "kwargs": {}},
        {"name": "Lid-Driven Cavity (Re=100)", "problem_class": LidDrivenProblem, "kwargs": {"Re": 100.0}},
        {"name": "Lid-Driven Cavity (Re=400)", "problem_class": LidDrivenProblem, "kwargs": {"Re": 400.0}},
        {"name": "Singular Perturbation (SPP Ex1)", "problem_class": SPPEx1Problem, "kwargs": {}},
        {"name": "Singular Perturbation (SPP Ex2)", "problem_class": SPPEx2Problem, "kwargs": {}},
        {"name": "Helmholtz / Poisson", "problem_class": PoissonProblem, "kwargs": {}}
    ]

    # Shared training config
    config = TrainingConfig()
    config.adam_epochs = 10000
    config.lbfgs_epochs = 500

    results_summary = []

    for exp in experiments:
        name = exp["name"]
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] RUNNING EXPERIMENT: {name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        try:
            problem = exp["problem_class"](**exp["kwargs"], device=device)
            results = run_mwpinn_full_pipeline(problem, config, device=device, verbose=True)
            end_time = time.time()
            
            exec_time_min = (end_time - start_time) / 60.0
            
            # Store the final metrics dynamically based on what the problem evaluate() returned
            metric_keys = [k for k in results.keys() if 'error' in k]
            metric_str = ", ".join([f"{k}: {results[k]:.6f}" for k in metric_keys])
            
            results_summary.append({
                "name": name,
                "status": "SUCCESS",
                "time_min": exec_time_min,
                "metrics": metric_str
            })
            
            print(f"\n[SUCCESS] {name} completed in {exec_time_min:.2f} mins.")
            print(f"[METRICS] {metric_str}")
            
        except Exception as e:
            print(f"\n[FAILED] {name} encountered an error: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                "name": name,
                "status": "FAILED",
                "error": str(e)
            })

    # Print Final Report
    print("\n\n" + "#"*80)
    print("                      FINAL EXPERIMENT REPORT")
    print("#"*80)
    for res in results_summary:
        if res["status"] == "SUCCESS":
            print(f"✅ {res['name']:<30} | {res['time_min']:5.1f} min | {res['metrics']}")
        else:
            print(f"❌ {res['name']:<30} | ERROR: {res['error']}")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
