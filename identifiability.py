import numpy as np
import matplotlib.pyplot as plt
import os
from data_gen import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, PLDs

def compute_jacobian(cbf, att, eps_cbf=0.1, eps_att=0.01):
    cbf = np.array([cbf])
    att = np.array([att])
    
    base_sig = compute_signals_vec(cbf, att)
    
    cbf_plus = compute_signals_vec(cbf + eps_cbf, att)
    ds_dcbf = (cbf_plus - base_sig) / eps_cbf
    
    att_plus = compute_signals_vec(cbf, att + eps_att)
    ds_datt = (att_plus - base_sig) / eps_att
    
    return np.vstack([ds_dcbf, ds_datt]).T  # shape (4, 2)

def analyze_identifiability():
    os.makedirs("plots/identifiability", exist_ok=True)
    cbf_vals = np.linspace(5, 100, 20)
    att_vals = np.linspace(0.5, 3.0, 20)
    
    cond_grid = np.zeros((len(att_vals), len(cbf_vals)))
    corr_grid = np.zeros((len(att_vals), len(cbf_vals)))
    det_grid = np.zeros((len(att_vals), len(cbf_vals)))
    
    for i, att in enumerate(att_vals):
        for j, cbf in enumerate(cbf_vals):
            J = compute_jacobian(cbf, att)
            
            # Normalize J by parameters to get relative sensitivity
            # J_rel = dS/(dtheta/theta) = theta * dS/dtheta
            J_rel = J.copy()
            J_rel[:, 0] *= cbf
            J_rel[:, 1] *= att
            
            # Condition number
            _, s, _ = np.linalg.svd(J_rel)
            cond = s[0] / (s[-1] + 1e-12)
            cond_grid[i, j] = np.log10(cond)
            
            # Correlation between sensitivity vectors
            v1 = J_rel[:, 0]
            v2 = J_rel[:, 1]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                corr = np.abs(np.dot(v1, v2) / (n1 * n2))
            else:
                corr = 1.0
            corr_grid[i, j] = corr
            
            # FIM determinant ~ |J^T J|
            F = J_rel.T @ J_rel
            det_grid[i, j] = np.log10(np.linalg.det(F) + 1e-12)
            
    # Plot condition number
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.imshow(cond_grid, origin='lower', extent=[5, 100, 0.5, 3.0], aspect='auto', cmap='magma')
    plt.colorbar(label='Log10 Condition Number')
    plt.xlabel('CBF')
    plt.ylabel('ATT')
    plt.title('Ill-Conditioning (High = Bad)')
    
    plt.subplot(132)
    plt.imshow(corr_grid, origin='lower', extent=[5, 100, 0.5, 3.0], aspect='auto', cmap='viridis')
    plt.colorbar(label='Absolute Correlation')
    plt.xlabel('CBF')
    plt.ylabel('ATT')
    plt.title('Sensitivity Correlation (1.0 = Confounded)')
    
    plt.subplot(133)
    plt.imshow(det_grid, origin='lower', extent=[5, 100, 0.5, 3.0], aspect='auto', cmap='plasma')
    plt.colorbar(label='Log10 Fisher Det')
    plt.xlabel('CBF')
    plt.ylabel('ATT')
    plt.title('Fisher Information Det (Low = Bad)')
    
    plt.tight_layout()
    plt.savefig("plots/identifiability/identifiability_analysis.png")
    
if __name__ == "__main__":
    analyze_identifiability()
    print("Identifiability analysis complete.")

