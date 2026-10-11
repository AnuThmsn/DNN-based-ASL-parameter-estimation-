import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import compute_signals_vec, SCALE, SIG_REF_SC, tau
from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.evaluate import (
    evaluate_unet,
    train_pointwise_dnn,
    evaluate_pointwise_dnn,
    evaluate_bayesian_calibrated,
    calc_metrics,
    calc_masked_ssim
)

def generate_generalization_challenge_dataset(n_phantoms_per_case=10, seed=999, img_size=64):
    """
    Generates genuinely unseen spatial geometries and hemodynamic conditions:
    1. Case A (Focal Lesions): Abrupt focal hypoperfusion/hyperperfusion stroke infarcts.
    2. Case B (Hydrocephalus): Severely enlarged ventricles and compressed cortical ribbon.
    3. Case C (High-Frequency / Discontinuous): Step-function perfusion with zero smoothing.
    4. Case D (Extreme Transit): Prolonged watershed ischemia (ATT > 2.6s, CBF < 15).
    """
    rng = np.random.default_rng(seed)
    cases = {}

    for case_name in ["focal_lesion", "hydrocephalus", "discontinuous_step", "extreme_watershed"]:
        X_list, Y_list, mask_list, snr_list = [], [], [], []

        for p_idx in range(n_phantoms_per_case):
            y, x = np.ogrid[:img_size, :img_size]
            center_y, center_x = img_size / 2.0, img_size / 2.0
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            if case_name == "hydrocephalus":
                # Enlarged ventricles: 3x normal radius
                brain_mask = (r < (img_size * 0.44)).astype(np.float32)
                ventricle_mask = (r < (img_size * 0.22)).astype(np.float32)
                gm_mask = ((r >= (img_size * 0.32)) & (r < (img_size * 0.44))).astype(np.float32)
                wm_mask = ((r >= (img_size * 0.22)) & (r < (img_size * 0.32))).astype(np.float32)
            else:
                brain_mask = (r < (img_size * 0.42)).astype(np.float32)
                ventricle_mask = (r < (img_size * 0.10)).astype(np.float32)
                gm_mask = ((r >= (img_size * 0.26)) & (r < (img_size * 0.42))).astype(np.float32)
                wm_mask = ((r >= (img_size * 0.10)) & (r < (img_size * 0.26))).astype(np.float32)

            brain_mask[ventricle_mask > 0] = 0.0

            if case_name == "focal_lesion":
                cbf_map = gm_mask * 65.0 + wm_mask * 25.0
                att_map = gm_mask * 1.2 + wm_mask * 1.8
                # Add sharp focal stroke lesion in right hemisphere
                lesion = ((x - (center_x + 12))**2 + (y - center_y)**2) < 40.0
                cbf_map[lesion & (brain_mask > 0)] = 5.0 # Severe ischemia
                att_map[lesion & (brain_mask > 0)] = 2.8 # Delayed collateral arrival
            elif case_name == "discontinuous_step":
                # Sharp alternating bands across cortex
                step_pattern = ((x // 8) % 2).astype(np.float32)
                cbf_map = (gm_mask * (40.0 + 35.0 * step_pattern) + wm_mask * 22.0) * brain_mask
                att_map = (gm_mask * (1.0 + 0.8 * step_pattern) + wm_mask * 1.7) * brain_mask
            elif case_name == "extreme_watershed":
                # Watershed gradient: severe posterior transit delay
                grad = (y / img_size).astype(np.float32)
                cbf_map = (gm_mask * (60.0 - 45.0 * grad) + wm_mask * 15.0) * brain_mask
                att_map = (gm_mask * (1.2 + 1.6 * grad) + wm_mask * (1.6 + 1.2 * grad)) * brain_mask
            else: # hydrocephalus
                cbf_map = (gm_mask * 60.0 + wm_mask * 24.0) * brain_mask
                att_map = (gm_mask * 1.3 + wm_mask * 1.9) * brain_mask

            cbf_map = np.clip(cbf_map, 0.0, 100.0)
            att_map = np.clip(att_map, 0.5, 3.0)

            # Compute observation with Rician noise (SNR=12.0)
            snr = 12.0
            sig = compute_signals_vec(cbf_map.flatten(), att_map.flatten(), ld=tau) * SCALE
            sig = sig.reshape(img_size, img_size, 4).transpose(2, 0, 1) # [4, H, W]

            sd = SIG_REF_SC / snr
            e1 = rng.normal(0, sd, sig.shape)
            e2 = rng.normal(0, sd, sig.shape)
            e3 = rng.normal(0, sd, sig.shape)
            e4 = rng.normal(0, sd, sig.shape)
            X = np.sqrt((sig + e1)**2 + e2**2) + np.sqrt((sig + e3)**2 + e4**2)
            X = (X * brain_mask[None, ...]).astype(np.float32)

            Y = np.stack([cbf_map, att_map], axis=0).astype(np.float32)
            X_list.append(X)
            Y_list.append(Y)
            mask_list.append(brain_mask[None, ...].astype(np.float32))
            snr_list.append(snr)

        cases[case_name] = {
            "X": np.stack(X_list, axis=0),
            "Y": np.stack(Y_list, axis=0),
            "masks": np.stack(mask_list, axis=0),
            "snrs": np.array(snr_list, dtype=np.float32)
        }
    return cases

def run_spatial_generalization_study(args=None):
    ws = get_workspace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n================ SPATIAL GENERALIZATION STRESS TEST ================")
    print(f"Workspace: {ws.root} | Device: {device}")

    # Load 4-PLD U-Net model from final package
    final_model_dir = ws.get_path("final_model")
    ckpt_path = os.path.join(final_model_dir, "asl_unet_4pld_final.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Final model checkpoint not found at {ckpt_path}. Run 4-PLD experiment first.")

    with open(os.path.join(final_model_dir, "config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(final_model_dir, "normalization_stats.json"), "r") as f:
        stats = json.load(f)

    unet = ASLUNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"]["features"]
    ).to(device)
    unet.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
    unet.eval()

    # Load training data for fitting point-wise DNN baseline
    tr_data = np.load(ws.get_path("data", f"train_N{config['data']['n_train_phantoms']}_s{config['data']['train_seed']}.npz"))
    v_tr = tr_data["masks"][:, 0] > 0.5
    X_tr_vox = tr_data["X"].transpose(0, 2, 3, 1)[v_tr]
    Y_tr_vox = tr_data["Y"].transpose(0, 2, 3, 1)[v_tr]
    if len(X_tr_vox) > 200_000:
        sub_idx = np.random.choice(len(X_tr_vox), size=200_000, replace=False)
        X_tr_vox = X_tr_vox[sub_idx]
        Y_tr_vox = Y_tr_vox[sub_idx]

    print("Fitting Point-wise DNN baseline on training voxels...")
    cbf_mlp, att_mlp = train_pointwise_dnn(X_tr_vox, Y_tr_vox, stats, device, epochs=30)

    # Generate unseen challenge cases
    print("Generating unseen challenge geometries and pathological patterns...")
    challenges = generate_generalization_challenge_dataset(n_phantoms_per_case=10, seed=2026)

    results_table = []
    case_predictions = {}

    for case_name, data in challenges.items():
        Xc, Yc, Mc, Sc = data["X"], data["Y"], data["masks"], data["snrs"]
        valid_m = Mc[:, 0] > 0.5
        gt_c = Yc[:, 0][valid_m]
        gt_a = Yc[:, 1][valid_m]

        # 1. U-Net evaluation
        Xc_norm = np.copy(Xc)
        for c in range(4):
            Xc_norm[:, c] = (Xc[:, c] - stats["X_mean"][c]) / stats["X_std"][c]
        cbf_u, att_u, _ = evaluate_unet(unet, Xc_norm, stats, device)
        mu_c = calc_metrics(gt_c, cbf_u[valid_m])
        mu_a = calc_metrics(gt_a, att_u[valid_m])

        # 2. Point-wise DNN evaluation
        cbf_dnn, att_dnn, _ = evaluate_pointwise_dnn(cbf_mlp, att_mlp, Xc, Mc, stats, device)
        md_c = calc_metrics(gt_c, cbf_dnn[valid_m])
        md_a = calc_metrics(gt_a, att_dnn[valid_m])

        # 3. Calibrated Bayesian evaluation
        cbf_b, att_b, _ = evaluate_bayesian_calibrated(Xc, Sc, Mc, device)
        mb_c = calc_metrics(gt_c, cbf_b[valid_m])
        mb_a = calc_metrics(gt_a, att_b[valid_m])

        # Calculate SSIM
        ssim_u_c = np.mean([calc_masked_ssim(Yc[i, 0], cbf_u[i], Mc[i, 0]) for i in range(len(Xc))])
        ssim_u_a = np.mean([calc_masked_ssim(Yc[i, 1], att_u[i], Mc[i, 0], data_range=2.5) for i in range(len(Xc))])

        case_predictions[case_name] = (Yc, Mc, cbf_u, att_u, cbf_dnn, att_dnn, cbf_b, att_b)

        results_table.append({
            "Challenge_Case": case_name,
            "UNet_CBF_RMSE": round(mu_c["RMSE"], 3),
            "DNN_CBF_RMSE": round(md_c["RMSE"], 3),
            "Bayes_CBF_RMSE": round(mb_c["RMSE"], 3),
            "UNet_ATT_RMSE": round(mu_a["RMSE"], 4),
            "DNN_ATT_RMSE": round(md_a["RMSE"], 4),
            "Bayes_ATT_RMSE": round(mb_a["RMSE"], 4),
            "UNet_CBF_SSIM": round(float(ssim_u_c), 4),
            "UNet_ATT_SSIM": round(float(ssim_u_a), 4),
            "Spatial_Advantage": "Preserved" if (mu_c["RMSE"] < md_c["RMSE"] and mu_a["RMSE"] < md_a["RMSE"]) else "Degraded"
        })

    gen_df = pd.DataFrame(results_table)
    gen_csv = ws.get_path("results", "spatial_generalization_results.csv")
    gen_df.to_csv(gen_csv, index=False)
    # Mirror to repo
    repo_results = os.path.join(parent_dir, "experiments_unet_4pld", "results")
    gen_df.to_csv(os.path.join(repo_results, "spatial_generalization_results.csv"), index=False)

    print("\n--- Spatial Generalization Stress-Test Summary ---")
    print(gen_df.to_string())

    # Generate diagnostic plot comparing unseen cases
    plots_dir = ws.get_path("plots")
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    case_labels = {
        "focal_lesion": "Focal Stroke Infarct",
        "hydrocephalus": "Enlarged Hydrocephalus",
        "discontinuous_step": "Sharp Step Discontinuity",
        "extreme_watershed": "Extreme Watershed Delay"
    }

    for row, (c_key, (Yc, Mc, c_u, a_u, c_d, a_d, c_b, a_b)) in enumerate(case_predictions.items()):
        m_2d = Mc[0, 0] > 0.5
        gt_c_2d = np.where(m_2d, Yc[0, 0], np.nan)
        u_c_2d = np.where(m_2d, c_u[0], np.nan)
        d_c_2d = np.where(m_2d, c_d[0], np.nan)
        b_c_2d = np.where(m_2d, c_b[0], np.nan)

        im0 = axes[row, 0].imshow(gt_c_2d, cmap="hot", vmin=0, vmax=100)
        axes[row, 0].set_title(f"{case_labels[c_key]}\nGround Truth CBF")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(u_c_2d, cmap="hot", vmin=0, vmax=100)
        axes[row, 1].set_title("U-Net Pred CBF")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(d_c_2d, cmap="hot", vmin=0, vmax=100)
        axes[row, 2].set_title("Point-wise DNN Pred CBF")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        im3 = axes[row, 3].imshow(b_c_2d, cmap="hot", vmin=0, vmax=100)
        axes[row, 3].set_title("Bayesian Pred CBF")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

        for col in range(4):
            axes[row, col].axis("off")

    plt.tight_layout()
    gen_plot_ws = os.path.join(plots_dir, "spatial_generalization_cases.png")
    plt.savefig(gen_plot_ws, dpi=200)
    plt.close()
    shutil.copy2(gen_plot_ws, os.path.join(repo_results, "plots", "spatial_generalization_cases.png"))

    print(f"\nSaved generalization plots to: {gen_plot_ws}")
    print("================ GENERALIZATION STUDY COMPLETED ================")
    return gen_df

if __name__ == "__main__":
    run_spatial_generalization_study()

