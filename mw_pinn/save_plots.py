import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def save_solution_plot(problem_name, results, save_dir):
    """
    Generate and save a visual comparison plot of the exact and predicted PDE solutions.
    Handles both 1D and 2D problems automatically.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    u_pred = results.get('u_pred')
    u_exact = results.get('u_exact')
    
    if u_pred is None or u_exact is None:
        print(f"Skipping plot for {problem_name}: u_pred or u_exact not found in results.")
        return

    # Check if 1D or 2D
    is_1d = (u_pred.ndim == 1)
    
    fig = plt.figure(figsize=(20, 15) if not is_1d else (10, 6))
    
    if not is_1d:
        # 2D plotting logic (matches W-PINN base paper format)
        plt.subplot(1, 3, 1)
        plt.imshow(u_exact, cmap='jet', aspect='auto')
        plt.colorbar(shrink=0.3)
        plt.title('Exact', fontsize=18, fontweight='bold')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(u_pred, cmap='jet', aspect='auto')
        plt.colorbar(shrink=0.3)
        plt.title('Predicted', fontsize=18, fontweight='bold')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(u_exact - u_pred), cmap='jet', aspect='auto')
        plt.colorbar(shrink=0.3)
        plt.title('Error', fontsize=18, fontweight='bold')
        plt.axis('off')

        plt.suptitle(f"{problem_name} - 2D Solution", fontsize=24, fontweight='bold')
        
    else:
        # 1D plotting logic
        x = results.get('x_coords', np.linspace(0, 1, len(u_pred)))
        plt.plot(x, u_exact, 'r-', linewidth=3, label='Exact')
        plt.plot(x, u_pred, 'b--', linewidth=3, label='Predicted (MW-PINN)')
        plt.title(f"{problem_name} - 1D Solution", fontsize=18, fontweight='bold')
        plt.xlabel('x', fontsize=14)
        plt.ylabel('u(x)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{problem_name.replace(' ', '_')}_sol.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> Saved visualization to {save_path}")
