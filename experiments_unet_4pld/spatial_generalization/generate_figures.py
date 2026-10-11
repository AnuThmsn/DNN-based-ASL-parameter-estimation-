"""
Generates publication-quality figures for the ASL U-Net Spatial Generalization Study.
Saved to: D:\\ASL_WORKSPACE\\experiments\\spatial_generalization\\figures\\
"""

import os
import sys
import json
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.spatial_generalization.models_generalization import ASLUNet, ASLHybridUNet, ASLResidualUNet
from experiments_unet_4pld.evaluate import calc_metrics

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15
})

def generate_all_figures():
    ws = get_workspace()
    fig_dir = os.path.join(ws.root, "experiments", "spatial_generalization", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen test datasets
    te_ord = np.load(ws.get_path("data", "test_N100_s7.npz"))
    te_chal = np.load(ws.get_path("data", "challenge_test_N40_s2026.npz"))

    # Load baseline model
    base_ckpt_path = os.path.join(ws.get_path("final_model"), "asl_unet_4pld_final.pth")
    base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    base_stats = base_ckpt["stats"]
    base_model = ASLUNet().to(device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()

    # Load v2 improved model
    v2_dir = os.path.join(ws.root, "final_model_4pld_spatial_v2")
    v2_ckpt_path = os.path.join(v2_dir, "asl_unet_4pld_final.pth")
    if not os.path.exists(v2_ckpt_path):
        print("Model v2 checkpoint not yet found. Skipping figure generation until packaging finishes.")
        return

    with open(os.path.join(v2_dir, "config.json"), "r") as f:
        v2_cfg = json.load(f)
    arch_type = v2_cfg["architecture_type"]
    if arch_type == "ASLHybridUNet":
        v2_model = ASLHybridUNet().to(device)
    elif arch_type == "ASLResidualUNet":
        v2_model = ASLResidualUNet().to(device)
    else:
        v2_model = ASLUNet().to(device)

    v2_ckpt = torch.load(v2_ckpt_path, map_location=device, weights_only=False)
    v2_model.load_state_dict(v2_ckpt["model_state_dict"])
    v2_model.eval()

    with open(os.path.join(v2_dir, "normalization_stats.json"), "r") as f:
        v2_stats = json.load(f)
    for k in ["X_mean", "X_std"]:
        v2_stats[k] = np.array(v2_stats[k], dtype=np.float32)

    # Load point-wise DNN baseline predictions
    dnn_chal_preds_path = os.path.join(ws.root, "experiments", "spatial_generalization", "baseline_reproduced", "dnn_chal_preds.npz")
    if os.path.exists(dnn_chal_preds_path):
        dnn_chal = np.load(dnn_chal_preds_path)
        dnn_cbf_chal = dnn_chal["cbf"]
        dnn_att_chal = dnn_chal["att"]
    else:
        dnn_cbf_chal = None
        dnn_att_chal = None

    # Predict with baseline and v2 on challenge test set
    X_chal = te_chal["X"]
    Y_chal = te_chal["Y"]
    M_chal = te_chal["masks"]
    cases_chal = te_chal["case_labels"]

    # Normalize for baseline
    X_base_n = np.copy(X_chal)
    for c in range(4):
        X_base_n[:, c] = (X_base_n[:, c] - base_stats["X_mean"][c]) / base_stats["X_std"][c]

    # Normalize for v2
    X_v2_n = np.copy(X_chal)
    for c in range(4):
        X_v2_n[:, c] = (X_v2_n[:, c] - v2_stats["X_mean"][c]) / v2_stats["X_std"][c]

    with torch.no_grad():
        tb = torch.tensor(X_base_n, dtype=torch.float32, device=device)
        base_preds = base_model(tb).cpu().numpy()
        base_cbf = np.clip(base_preds[:, 0] * base_stats["cbf_std"] + base_stats["cbf_mean"], 0.0, 100.0)
        base_att = np.clip(base_preds[:, 1] * base_stats["att_std"] + base_stats["att_mean"], 0.5, 3.0)

        tv2 = torch.tensor(X_v2_n, dtype=torch.float32, device=device)
        v2_preds = v2_model(tv2).cpu().numpy()
        v2_cbf = np.clip(v2_preds[:, 0] * v2_stats["cbf_std"] + v2_stats["cbf_mean"], 0.0, 100.0)
        v2_att = np.clip(v2_preds[:, 1] * v2_stats["att_std"] + v2_stats["att_mean"], 0.5, 3.0)

    # Mask outside brain
    for i in range(len(M_chal)):
        m = M_chal[i, 0] <= 0.5
        base_cbf[i][m] = 0.0
        base_att[i][m] = 0.0
        v2_cbf[i][m] = 0.0
        v2_att[i][m] = 0.0

    # -------------------------------------------------------------
    # FIGURE 1: Gallery Comparison across Challenge Syndromes
    # -------------------------------------------------------------
    print("Generating Figure 1: Gallery Comparison...")
    sample_indices = [0, 10, 20, 30]  # One from each syndrome category
    syndrome_titles = ["Focal Ischemia", "Hydrocephalus", "Sharp Discontinuity", "Watershed Margin"]

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for row, (idx, syn) in enumerate(zip(sample_indices, syndrome_titles)):
        gt = Y_chal[idx, 0]
        bc = base_cbf[idx]
        v2c = v2_cbf[idx]
        dnnc = dnn_cbf_chal[idx] if dnn_cbf_chal is not None else np.zeros_like(gt)
        err_base = np.abs(bc - gt) * (M_chal[idx, 0] > 0.5)
        err_v2 = np.abs(v2c - gt) * (M_chal[idx, 0] > 0.5)

        vmax = 80.0
        err_max = 25.0

        # Col 0: Ground Truth
        im0 = axes[row, 0].imshow(gt, cmap="inferno", vmin=0, vmax=vmax)
        axes[row, 0].set_ylabel(syn, fontsize=12, fontweight="bold")
        if row == 0: axes[row, 0].set_title("Ground Truth CBF")

        # Col 1: Original U-Net
        im1 = axes[row, 1].imshow(bc, cmap="inferno", vmin=0, vmax=vmax)
        if row == 0: axes[row, 1].set_title("Original U-Net")

        # Col 2: Point-wise DNN
        im2 = axes[row, 2].imshow(dnnc, cmap="inferno", vmin=0, vmax=vmax)
        if row == 0: axes[row, 2].set_title("Point-wise DNN")

        # Col 3: Improved U-Net (v2)
        im3 = axes[row, 3].imshow(v2c, cmap="inferno", vmin=0, vmax=vmax)
        if row == 0: axes[row, 3].set_title("Improved U-Net (v2)")

        # Col 4: Base Error
        im4 = axes[row, 4].imshow(err_base, cmap="magma", vmin=0, vmax=err_max)
        if row == 0: axes[row, 4].set_title("|Base U-Net Error|")

        # Col 5: v2 Error
        im5 = axes[row, 5].imshow(err_v2, cmap="magma", vmin=0, vmax=err_max)
        if row == 0: axes[row, 5].set_title("|Improved v2 Error|")

        for c in range(6):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    # Colorbars
    cbar_ax1 = fig.add_axes([0.15, 0.05, 0.35, 0.02])
    fig.colorbar(im0, cax=cbar_ax1, orientation="horizontal", label="CBF (ml/100g/min)")
    cbar_ax2 = fig.add_axes([0.65, 0.05, 0.25, 0.02])
    fig.colorbar(im4, cax=cbar_ax2, orientation="horizontal", label="Absolute Error (ml/100g/min)")

    fig.subplots_adjust(bottom=0.12, hspace=0.15, wspace=0.08)
    fig_path1 = os.path.join(fig_dir, "figure1_spatial_comparison_gallery.png")
    plt.savefig(fig_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path1}")

    # -------------------------------------------------------------
    # FIGURE 2: Boundary Error Dissection vs Distance to Edge
    # -------------------------------------------------------------
    print("Generating Figure 2: Boundary Radial Error Profile...")
    # Compute error as function of signed distance from lesion edge
    focal_indices = np.where(cases_chal == "focal_lesion")[0]
    
    # Distance bins from -4 (deep core) to +4 (surrounding parenchyma)
    distances = np.arange(-4, 5)
    base_dist_errs = {d: [] for d in distances}
    v2_dist_errs = {d: [] for d in distances}
    dnn_dist_errs = {d: [] for d in distances}

    for idx in focal_indices:
        gt = Y_chal[idx, 0]
        m = M_chal[idx, 0] > 0.5
        lesion = (gt < 10.0) & m
        
        # Signed distance: negative inside lesion, positive outside
        dt_out = ndi.distance_transform_edt(~lesion)
        dt_in = ndi.distance_transform_edt(lesion)
        signed_dt = np.where(lesion, -dt_in, dt_out)

        err_b = np.abs(base_cbf[idx] - gt)
        err_v = np.abs(v2_cbf[idx] - gt)
        err_d = np.abs(dnn_cbf_chal[idx] - gt) if dnn_cbf_chal is not None else None

        for d in distances:
            bin_mask = (signed_dt >= d - 0.5) & (signed_dt < d + 0.5) & m
            if np.any(bin_mask):
                base_dist_errs[d].extend(err_b[bin_mask])
                v2_dist_errs[d].extend(err_v[bin_mask])
                if err_d is not None:
                    dnn_dist_errs[d].extend(err_d[bin_mask])

    d_vals = []
    base_means, v2_means, dnn_means = [], [], []
    for d in distances:
        if len(base_dist_errs[d]) > 0:
            d_vals.append(d)
            base_means.append(np.mean(base_dist_errs[d]))
            v2_means.append(np.mean(v2_dist_errs[d]))
            if dnn_cbf_chal is not None:
                dnn_means.append(np.mean(dnn_dist_errs[d]))

    plt.figure(figsize=(9, 5.5))
    plt.plot(d_vals, base_means, "o-", color="#D9534F", linewidth=2.5, markersize=7, label="Original 4-PLD U-Net")
    if dnn_cbf_chal is not None:
        plt.plot(d_vals, dnn_means, "s--", color="#F0AD4E", linewidth=2.0, markersize=6, label="Point-wise DNN Baseline")
    plt.plot(d_vals, v2_means, "D-", color="#2E7D32", linewidth=2.5, markersize=7, label="Improved U-Net (v2)")

    plt.axvline(0, color="gray", linestyle=":", linewidth=1.5, label="Lesion Boundary Interface")
    plt.axvspan(-2, 2, color="gray", alpha=0.12, label="Boundary Transition Zone (±2 voxels)")

    plt.xlabel("Signed Distance from Lesion Boundary (voxels)\n[ < 0: Inside Ischemic Core | > 0: Surrounding Parenchyma ]")
    plt.ylabel("Mean Absolute CBF Error (ml/100g/min)")
    plt.title("Spatial Resolution Dissection: Error Across Lesion Interface")
    plt.legend(frameon=True, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path2 = os.path.join(fig_dir, "figure2_boundary_error_profile.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Saved: {fig_path2}")

    # -------------------------------------------------------------
    # FIGURE 3: SNR Robustness Curve
    # -------------------------------------------------------------
    print("Generating Figure 3: SNR Robustness Curve...")
    snrs_eval = [10, 20, 30, 50]
    
    # Evaluate across SNR groups on ordinary test set
    ord_snrs = te_ord["snrs"]
    base_ord_by_snr = []
    v2_ord_by_snr = []

    X_ord = te_ord["X"]
    Y_ord = te_ord["Y"]
    M_ord = te_ord["masks"]

    X_b_ord_n = np.copy(X_ord)
    for c in range(4):
        X_b_ord_n[:, c] = (X_b_ord_n[:, c] - base_stats["X_mean"][c]) / base_stats["X_std"][c]

    X_v_ord_n = np.copy(X_ord)
    for c in range(4):
        X_v_ord_n[:, c] = (X_v_ord_n[:, c] - v2_stats["X_mean"][c]) / v2_stats["X_std"][c]

    with torch.no_grad():
        tb_o = torch.tensor(X_b_ord_n, dtype=torch.float32, device=device)
        base_ord_p = base_model(tb_o).cpu().numpy()
        b_cbf_ord = np.clip(base_ord_p[:, 0] * base_stats["cbf_std"] + base_stats["cbf_mean"], 0.0, 100.0)

        tv_o = torch.tensor(X_v_ord_n, dtype=torch.float32, device=device)
        v2_ord_p = v2_model(tv_o).cpu().numpy()
        v_cbf_ord = np.clip(v2_ord_p[:, 0] * v2_stats["cbf_std"] + v2_stats["cbf_mean"], 0.0, 100.0)

    for snr in snrs_eval:
        s_idx = np.where(ord_snrs == snr)[0]
        if len(s_idx) > 0:
            m = M_ord[s_idx, 0] > 0.5
            err_b = np.sqrt(np.mean((b_cbf_ord[s_idx][m] - Y_ord[s_idx, 0][m])**2))
            err_v = np.sqrt(np.mean((v_cbf_ord[s_idx][m] - Y_ord[s_idx, 0][m])**2))
            base_ord_by_snr.append(err_b)
            v2_ord_by_snr.append(err_v)
        else:
            base_ord_by_snr.append(np.nan)
            v2_ord_by_snr.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(snrs_eval, base_ord_by_snr, "o-", color="#D9534F", linewidth=2.5, markersize=8, label="Original U-Net")
    ax1.plot(snrs_eval, v2_ord_by_snr, "D-", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved U-Net (v2)")
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("CBF RMSE (ml/100g/min)")
    ax1.set_title("Standard Brain Phantoms (Denoising Retention)")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Challenge dataset breakdown by category
    categories = ["Focal Lesion", "Hydrocephalus", "Step Edge", "Watershed"]
    cat_keys = ["focal_lesion", "hydrocephalus", "discontinuous_step", "extreme_watershed"]
    base_cat_errs = []
    v2_cat_errs = []

    for k in cat_keys:
        idx = np.where(cases_chal == k)[0]
        m = M_chal[idx, 0] > 0.5
        eb = np.sqrt(np.mean((base_cbf[idx][m] - Y_chal[idx, 0][m])**2))
        ev = np.sqrt(np.mean((v2_cbf[idx][m] - Y_chal[idx, 0][m])**2))
        base_cat_errs.append(eb)
        v2_cat_errs.append(ev)

    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, base_cat_errs, width, label="Original U-Net", color="#D9534F", alpha=0.85)
    ax2.bar(x + width/2, v2_cat_errs, width, label="Improved U-Net (v2)", color="#2E7D32", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("CBF RMSE (ml/100g/min)")
    ax2.set_title("Challenge Syndromes (Generalization Improvement)")
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path3 = os.path.join(fig_dir, "figure3_snr_and_category_robustness.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Saved: {fig_path3}")

    # -------------------------------------------------------------
    # FIGURE 4: Ground Truth vs Prediction Scatter Plots
    # -------------------------------------------------------------
    print("Generating Figure 4: Scatter Correlations...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # Downsample points for clear visualization
    m_chal_vox = M_chal[:, 0] > 0.5
    gt_chal_cbf = Y_chal[:, 0][m_chal_vox]
    base_chal_cbf = base_cbf[m_chal_vox]
    v2_chal_cbf = v2_cbf[m_chal_vox]

    m_chal_vox_a = M_chal[:, 0] > 0.5
    gt_chal_att = Y_chal[:, 1][m_chal_vox_a]
    base_chal_att = base_att[m_chal_vox_a]
    v2_chal_att = v2_att[m_chal_vox_a]

    sub_idx = np.random.RandomState(42).choice(len(gt_chal_cbf), size=min(15000, len(gt_chal_cbf)), replace=False)

    # Base CBF
    ax = axes[0, 0]
    ax.scatter(gt_chal_cbf[sub_idx], base_chal_cbf[sub_idx], alpha=0.15, s=6, color="#D9534F")
    ax.plot([0, 100], [0, 100], "k--", linewidth=1.5)
    ax.set_title(f"Original U-Net CBF (Challenge)\nRMSE = {np.sqrt(np.mean((base_chal_cbf-gt_chal_cbf)**2)):.2f}")
    ax.set_xlabel("True CBF (ml/100g/min)")
    ax.set_ylabel("Estimated CBF (ml/100g/min)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # v2 CBF
    ax = axes[0, 1]
    ax.scatter(gt_chal_cbf[sub_idx], v2_chal_cbf[sub_idx], alpha=0.15, s=6, color="#2E7D32")
    ax.plot([0, 100], [0, 100], "k--", linewidth=1.5)
    ax.set_title(f"Improved U-Net v2 CBF (Challenge)\nRMSE = {np.sqrt(np.mean((v2_chal_cbf-gt_chal_cbf)**2)):.2f}")
    ax.set_xlabel("True CBF (ml/100g/min)")
    ax.set_ylabel("Estimated CBF (ml/100g/min)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Base ATT
    ax = axes[1, 0]
    ax.scatter(gt_chal_att[sub_idx], base_chal_att[sub_idx], alpha=0.15, s=6, color="#D9534F")
    ax.plot([0.5, 3.0], [0.5, 3.0], "k--", linewidth=1.5)
    ax.set_title(f"Original U-Net ATT (Challenge)\nRMSE = {np.sqrt(np.mean((base_chal_att-gt_chal_att)**2)):.3f} s")
    ax.set_xlabel("True ATT (s)")
    ax.set_ylabel("Estimated ATT (s)")
    ax.set_xlim(0.5, 3.0); ax.set_ylim(0.5, 3.0)
    ax.grid(True, alpha=0.3)

    # v2 ATT
    ax = axes[1, 1]
    ax.scatter(gt_chal_att[sub_idx], v2_chal_att[sub_idx], alpha=0.15, s=6, color="#2E7D32")
    ax.plot([0.5, 3.0], [0.5, 3.0], "k--", linewidth=1.5)
    ax.set_title(f"Improved U-Net v2 ATT (Challenge)\nRMSE = {np.sqrt(np.mean((v2_chal_att-gt_chal_att)**2)):.3f} s")
    ax.set_xlabel("True ATT (s)")
    ax.set_ylabel("Estimated ATT (s)")
    ax.set_xlim(0.5, 3.0); ax.set_ylim(0.5, 3.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path4 = os.path.join(fig_dir, "figure4_scatter_correlation.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"Saved: {fig_path4}")

if __name__ == "__main__":
    generate_all_figures()

