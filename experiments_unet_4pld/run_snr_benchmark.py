"""
Rigorous SNR-Wise Benchmark of 4-PLD U-Net Models (Original vs v2)
Evaluates across: SNR = [inf, 50, 20, 15, 10, 5]
Compares against historical baseline reference values from comparison PDF:
- Bayesian
- NLS / NLLS
- Point-wise DNN
- Residual ELM
All outputs strictly routed to D:\\ASL_WORKSPACE\\experiments\\snr_benchmark_4pld\\
"""

import os
import sys
import json
import time
import shutil
import hashlib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import compute_signals_vec, SCALE, tau, SIG_REF_SC, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.evaluate import calc_metrics, calc_masked_ssim
from experiments_unet_4pld.spatial_generalization.models_generalization import ASLHybridUNet

# Historical Reference Values from PDF
HISTORICAL_BASELINES = [
    {"Model": "Bayesian", "SNR": 10, "CBF_RMSE": 5.0300, "CBF_MAE": 3.9680, "CBF_CCC": 0.9606, "ATT_RMSE": 0.0826, "ATT_MAE": 0.0683, "ATT_CCC": 0.9914, "Source": "Historical PDF Reference (Unchanged)"},
    {"Model": "NLS", "SNR": 10, "CBF_RMSE": 2.9820, "CBF_MAE": 2.0480, "CBF_CCC": 0.9913, "ATT_RMSE": 0.5062, "ATT_MAE": 0.3411, "ATT_CCC": 0.6770, "Source": "Historical PDF Reference (Unchanged)"},
    {"Model": "Point-wise DNN", "SNR": 10, "CBF_RMSE": 4.2676, "CBF_MAE": 3.2983, "CBF_CCC": 0.9871, "ATT_RMSE": 0.3026, "ATT_MAE": 0.2105, "ATT_CCC": 0.9046, "Source": "Historical PDF Reference (Unchanged)"},
    {"Model": "Residual ELM", "SNR": 10, "CBF_RMSE": 4.2580, "CBF_MAE": 3.2825, "CBF_CCC": 0.9872, "ATT_RMSE": 0.3044, "ATT_MAE": 0.2115, "ATT_CCC": 0.9032, "Source": "Historical PDF Reference (Unchanged)"},
    {"Model": "Bayesian", "SNR": 15, "CBF_RMSE": 3.7590, "CBF_MAE": 2.8490, "CBF_CCC": 0.9671, "ATT_RMSE": 0.0584, "ATT_MAE": 0.0472, "ATT_CCC": 0.9944, "Source": "Historical PDF Reference (Unchanged)"},
    {"Model": "NLS", "SNR": 15, "CBF_RMSE": 2.0620, "CBF_MAE": 1.4140, "CBF_CCC": 0.9958, "ATT_RMSE": 0.4951, "ATT_MAE": 0.3259, "ATT_CCC": 0.6899, "Source": "Historical PDF Reference (Unchanged)"},
    {"Model": "Point-wise DNN", "SNR": 15, "CBF_RMSE": 3.5560, "CBF_MAE": 2.7040, "CBF_CCC": 0.9911, "ATT_RMSE": 0.2625, "ATT_MAE": 0.1802, "ATT_CCC": 0.9293, "Source": "Historical PDF Reference (Unchanged)"},
    {"Model": "Residual ELM", "SNR": 15, "CBF_RMSE": 3.5597, "CBF_MAE": 2.7047, "CBF_CCC": 0.9911, "ATT_RMSE": 0.2629, "ATT_MAE": 0.1806, "ATT_CCC": 0.9288, "Source": "Historical PDF Reference (Unchanged)"},
]

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def calc_lin_ccc(true, pred):
    """
    Computes Lin's Concordance Correlation Coefficient (CCC) using exact population moments:
    CCC = (2 * Cov(y, y_hat)) / (Var(y) + Var(y_hat) + (mu_y - mu_y_hat)^2)
    where Var and Cov use population moments (ddof=0 / 1/N divisor).
    """
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("Input arrays must contain finite values only (found NaN or Inf).")

    if len(true) == 0:
        return 0.0

    mu_x = np.mean(true)
    mu_y = np.mean(pred)
    var_x = np.var(true)
    var_y = np.var(pred)
    cov_xy = np.mean((true - mu_x) * (pred - mu_y))
    denom = var_x + var_y + (mu_x - mu_y) ** 2

    if denom <= 1e-15:
        # Constant arrays: if identical, agreement is perfect 1.0, otherwise 0.0
        return 1.0 if np.allclose(true, pred) else 0.0

    return float((2.0 * cov_xy) / denom)

def generate_snr_dataset(Y_gt, masks, snr, seed=7000):
    """
    Generates paired 4-channel ASL observations matching simulator physics.
    X = mc + ml (sum of two Rician measurements).
    If snr is inf: sd = 0, no noise added -> X = 2 * S.
    If snr is finite: sd = SIG_REF_SC / snr.
    """
    N, _, H, W = Y_gt.shape
    cbf_all = Y_gt[:, 0]
    att_all = Y_gt[:, 1]
    
    # Calculate noiseless Buxton signal
    S_clean = np.zeros((N, 4, H, W), dtype=np.float64)
    for i in range(N):
        valid = masks[i, 0] > 0.5
        if np.any(valid):
            c_vox = cbf_all[i][valid]
            a_vox = att_all[i][valid]
            sig = compute_signals_vec(c_vox, a_vox, ld=tau) * SCALE
            for p in range(4):
                S_clean[i, p][valid] = sig[:, p]

    if snr == float("inf") or snr is None:
        X = (2.0 * S_clean).astype(np.float32)
        meas_sd = 0.0
    else:
        sd = float(SIG_REF_SC / snr)
        rng = np.random.default_rng(seed + int(snr * 10))
        e1 = rng.normal(0.0, 1.0, size=S_clean.shape) * sd
        e2 = rng.normal(0.0, 1.0, size=S_clean.shape) * sd
        e3 = rng.normal(0.0, 1.0, size=S_clean.shape) * sd
        e4 = rng.normal(0.0, 1.0, size=S_clean.shape) * sd

        mc = np.sqrt((S_clean + e1) ** 2 + e2 ** 2)
        ml = np.sqrt((S_clean + e3) ** 2 + e4 ** 2)
        X = (mc + ml).astype(np.float32)

        # Verify noise standard deviation on valid brain voxels
        m_broadcast = np.broadcast_to(masks > 0.5, X.shape)
        meas_sd = float(np.std((X - 2.0 * S_clean)[m_broadcast]))

    # Zero background outside brain mask
    for i in range(N):
        bg = masks[i, 0] <= 0.5
        for c in range(4):
            X[i, c][bg] = 0.0

    return X, meas_sd

def run_snr_benchmark():
    ws = get_workspace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("SNR-WISE BENCHMARK: 4-PLD U-NET (ORIGINAL vs v2) vs HISTORICAL BASELINES")
    print(f"Workspace: {ws.root} | Device: {device}")
    print("=" * 85)

    # 1. Verify Storage
    du_c = shutil.disk_usage("C:").free / (1024 ** 3)
    du_d = shutil.disk_usage(ws.root).free / (1024 ** 3)
    print(f"Storage: Drive C = {du_c:.2f} GB free | Drive D = {du_d:.2f} GB free")
    if du_d < 5.0:
        raise RuntimeError("Insufficient disk space on Drive D!")

    # Setup directories
    exp_dir = os.path.join(ws.root, "experiments", "snr_benchmark_4pld")
    data_dir = os.path.join(exp_dir, "data")
    pred_dir = os.path.join(exp_dir, "predictions")
    met_dir = os.path.join(exp_dir, "metrics")
    fig_dir = os.path.join(exp_dir, "figures")
    for d in [exp_dir, data_dir, pred_dir, met_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Checkpoint Verification
    base_dir = ws.get_path("final_model")
    base_ckpt_path = os.path.join(base_dir, "asl_unet_4pld_final.pth")
    v2_dir = os.path.join(ws.root, "final_model_4pld_spatial_v2")
    v2_ckpt_path = os.path.join(v2_dir, "asl_unet_4pld_final.pth")

    hash_orig = sha256_file(base_ckpt_path)
    hash_v2 = sha256_file(v2_ckpt_path)
    print(f"\n[Model 1] Original 4-PLD U-Net Checkpoint: {base_ckpt_path}")
    print(f"          SHA-256: {hash_orig}")
    print(f"[Model 2] Improved 4-PLD U-Net v2 Checkpoint: {v2_ckpt_path}")
    print(f"          SHA-256: {hash_v2}")

    # Load Normalization Stats
    with open(os.path.join(base_dir, "normalization_stats.json"), "r") as f:
        base_stats = json.load(f)
    with open(os.path.join(v2_dir, "normalization_stats.json"), "r") as f:
        v2_stats = json.load(f)
    for k in ["X_mean", "X_std"]:
        base_stats[k] = np.array(base_stats[k], dtype=np.float32)
        v2_stats[k] = np.array(v2_stats[k], dtype=np.float32)

    # Load Models
    model_orig = ASLUNet().to(device)
    model_orig.load_state_dict(torch.load(base_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    model_orig.eval()

    model_v2 = ASLHybridUNet().to(device)
    model_v2.load_state_dict(torch.load(v2_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    model_v2.eval()

    # 3. Load Frozen Test Phantoms (Ordinary N=100)
    test_npz = ws.get_path("data", "test_N100_s7.npz")
    hash_test = sha256_file(test_npz)
    print(f"\nCanonical Test Phantoms: {test_npz} (SHA-256: {hash_test[:16]}...)")
    te_data = np.load(test_npz)
    Y_gt = te_data["Y"]
    masks = te_data["masks"]
    N_phantoms = len(Y_gt)

    # 4. Define SNR Levels
    SNR_LEVELS = [float("inf"), 50.0, 20.0, 15.0, 10.0, 5.0]
    snr_labels = ["inf", "50", "20", "15", "10", "5"]

    all_phantom_records = []
    all_summary_records = []
    saved_predictions = {}

    for snr, snr_label in zip(SNR_LEVELS, snr_labels):
        print(f"\n>>> Evaluating SNR Condition: {snr_label} (Target SNR = {snr}) <<<")
        X_snr, meas_sd = generate_snr_dataset(Y_gt, masks, snr, seed=7000)
        
        # Save dataset snapshot
        ds_path = os.path.join(data_dir, f"test_N100_snr_{snr_label}.npz")
        np.savez_compressed(ds_path, X=X_snr, Y=Y_gt, masks=masks, snr=snr)
        ds_hash = sha256_file(ds_path)
        print(f"  Generated Observations: {ds_path} (Empirical Noise SD: {meas_sd:.2f}, SHA-256: {ds_hash[:16]}...)")

        # Evaluate Models: (Model Name, Model Instance, Stats, Architecture Version, Checkpoint Hash)
        models_eval = [
            ("Original 4-PLD U-Net", model_orig, base_stats, "ASLUNet_v1", hash_orig),
            ("Improved 4-PLD U-Net v2", model_v2, v2_stats, "ASLHybridUNet_v2", hash_v2)
        ]

        for m_name, net, stats, arch_ver, ckpt_hash in models_eval:
            start_infer = time.time()
            
            # Normalize inputs
            X_norm = np.copy(X_snr)
            for c in range(4):
                X_norm[:, c] = (X_norm[:, c] - stats["X_mean"][c]) / stats["X_std"][c]

            preds_cbf, preds_att = [], []
            with torch.no_grad():
                for i in range(0, N_phantoms, 16):
                    xb = torch.tensor(X_norm[i:i+16], dtype=torch.float32, device=device)
                    out = net(xb).cpu().numpy()
                    c = out[:, 0] * stats["cbf_std"] + stats["cbf_mean"]
                    a = out[:, 1] * stats["att_std"] + stats["att_mean"]
                    preds_cbf.append(np.clip(c, CBF_MIN, CBF_MAX))
                    preds_att.append(np.clip(a, ATT_MIN, ATT_MAX))

            cbf_pred = np.concatenate(preds_cbf, axis=0)
            att_pred = np.concatenate(preds_att, axis=0)
            infer_dur = time.time() - start_infer

            # Mask outside brain
            for i in range(N_phantoms):
                bg = masks[i, 0] <= 0.5
                cbf_pred[i][bg] = 0.0
                att_pred[i][bg] = 0.0

            # Save raw prediction array
            pred_key = f"{m_name.replace(' ', '_')}_snr_{snr_label}"
            saved_predictions[pred_key] = {"cbf": cbf_pred, "att": att_pred, "snr": snr_label}
            np.savez_compressed(os.path.join(pred_dir, f"{pred_key}.npz"), cbf=cbf_pred, att=att_pred)

            # Per-phantom metrics
            phantom_cbf_rmses = []
            phantom_cbf_maes = []
            phantom_cbf_biases = []
            phantom_cbf_r2s = []
            phantom_cbf_cccs = []

            phantom_att_rmses = []
            phantom_att_maes = []
            phantom_att_biases = []
            phantom_att_r2s = []
            phantom_att_cccs = []

            for p_idx in range(N_phantoms):
                vm = masks[p_idx, 0] > 0.5
                gt_c = Y_gt[p_idx, 0][vm]
                gt_a = Y_gt[p_idx, 1][vm]
                pr_c = cbf_pred[p_idx][vm]
                pr_a = att_pred[p_idx][vm]

                mc = calc_metrics(gt_c, pr_c)
                ma = calc_metrics(gt_a, pr_a)
                ccc_c = calc_lin_ccc(gt_c, pr_c)
                ccc_a = calc_lin_ccc(gt_a, pr_a)

                phantom_cbf_rmses.append(mc["RMSE"])
                phantom_cbf_maes.append(mc["MAE"])
                phantom_cbf_biases.append(mc["Bias"])
                phantom_cbf_r2s.append(mc["R2"])
                phantom_cbf_cccs.append(ccc_c)

                phantom_att_rmses.append(ma["RMSE"])
                phantom_att_maes.append(ma["MAE"])
                phantom_att_biases.append(ma["Bias"])
                phantom_att_r2s.append(ma["R2"])
                phantom_att_cccs.append(ccc_a)

                all_phantom_records.append({
                    "model": m_name,
                    "architecture_version": arch_ver,
                    "SNR": snr_label,
                    "test_phantom_id": p_idx,
                    "valid_voxels": int(np.sum(vm)),
                    "CBF_RMSE": mc["RMSE"],
                    "CBF_MAE": mc["MAE"],
                    "CBF_bias": mc["Bias"],
                    "CBF_R2": mc["R2"],
                    "CBF_CCC": ccc_c,
                    "ATT_RMSE": ma["RMSE"],
                    "ATT_MAE": ma["MAE"],
                    "ATT_bias": ma["Bias"],
                    "ATT_R2": ma["R2"],
                    "ATT_CCC": ccc_a,
                    "inference_time_s": infer_dur / N_phantoms,
                    "checkpoint_hash": ckpt_hash,
                    "dataset_hash": ds_hash
                })

            # Summary across all voxels
            vm_all = masks[:, 0] > 0.5
            gt_c_all = Y_gt[:, 0][vm_all]
            gt_a_all = Y_gt[:, 1][vm_all]
            pr_c_all = cbf_pred[vm_all]
            pr_a_all = att_pred[vm_all]

            met_c_all = calc_metrics(gt_c_all, pr_c_all)
            met_a_all = calc_metrics(gt_a_all, pr_a_all)
            ccc_c_all = calc_lin_ccc(gt_c_all, pr_c_all)
            ccc_a_all = calc_lin_ccc(gt_a_all, pr_a_all)

            summary_row = {
                "model": m_name,
                "snr": snr_label,
                "seed": 7000 if snr != float("inf") else 0,
                "n_phantoms": N_phantoms,
                "n_valid_elements": int(np.sum(vm_all)),
                "cbf_rmse": met_c_all["RMSE"],
                "cbf_mae": met_c_all["MAE"],
                "cbf_ccc": ccc_c_all,
                "att_rmse": met_a_all["RMSE"],
                "att_mae": met_a_all["MAE"],
                "att_ccc": ccc_a_all,
                "Model": m_name,
                "Architecture": arch_ver,
                "SNR": snr_label,
                "CBF_RMSE": met_c_all["RMSE"],
                "CBF_MAE": met_c_all["MAE"],
                "CBF_Bias": met_c_all["Bias"],
                "CBF_R2": met_c_all["R2"],
                "CBF_CCC": ccc_c_all,
                "ATT_RMSE": met_a_all["RMSE"],
                "ATT_MAE": met_a_all["MAE"],
                "ATT_Bias": met_a_all["Bias"],
                "ATT_R2": met_a_all["R2"],
                "ATT_CCC": ccc_a_all,
                "CBF_RMSE_mean_phantom": np.mean(phantom_cbf_rmses),
                "CBF_RMSE_std_phantom": np.std(phantom_cbf_rmses),
                "ATT_RMSE_mean_phantom": np.mean(phantom_att_rmses),
                "ATT_RMSE_std_phantom": np.std(phantom_att_rmses),
                "Inference_Time_Total_s": infer_dur,
                "Checkpoint_Hash": ckpt_hash,
                "Dataset_Hash": ds_hash
            }
            all_summary_records.append(summary_row)
            print(f"  [{m_name}] CBF RMSE: {met_c_all['RMSE']:.4f} | CBF MAE: {met_c_all['MAE']:.4f} | CBF CCC: {ccc_c_all:.4f} | "
                  f"ATT RMSE: {met_a_all['RMSE']:.4f} | ATT MAE: {met_a_all['MAE']:.4f} | ATT CCC: {ccc_a_all:.4f}")

    # 5. Save DataFrames
    df_phantom = pd.DataFrame(all_phantom_records)
    df_summary = pd.DataFrame(all_summary_records)
    df_hist = pd.DataFrame(HISTORICAL_BASELINES)

    csv_phantom = os.path.join(met_dir, "snr_wise_unet_per_phantom.csv")
    csv_summary = os.path.join(met_dir, "snr_wise_unet_results.csv")
    csv_hist = os.path.join(met_dir, "snr_wise_historical_baseline_reference.csv")
    df_phantom.to_csv(csv_phantom, index=False)
    df_summary.to_csv(csv_summary, index=False)
    df_hist.to_csv(csv_hist, index=False)

    # 6. Combined Comparison Table for SNR 10 & 15
    combined_rows = []
    # Add historical baselines
    for _, row in df_hist.iterrows():
        combined_rows.append({
            "Model": row["Model"],
            "SNR": str(row["SNR"]),
            "CBF_RMSE": row["CBF_RMSE"],
            "CBF_MAE": row["CBF_MAE"],
            "CBF_CCC": row["CBF_CCC"],
            "ATT_RMSE": row["ATT_RMSE"],
            "ATT_MAE": row["ATT_MAE"],
            "ATT_CCC": row["ATT_CCC"],
            "Provenance": row["Source"]
        })
    # Add new U-Net measurements at SNR 10 and 15
    for _, row in df_summary[df_summary["SNR"].isin(["10", "15"])].iterrows():
        combined_rows.append({
            "Model": row["Model"],
            "SNR": str(row["SNR"]),
            "CBF_RMSE": row["CBF_RMSE"],
            "CBF_MAE": row["CBF_MAE"],
            "CBF_CCC": row["CBF_CCC"],
            "ATT_RMSE": row["ATT_RMSE"],
            "ATT_MAE": row["ATT_MAE"],
            "ATT_CCC": row["ATT_CCC"],
            "Provenance": "Newly Measured (Frozen 4-PLD Spatial Dataset)"
        })

    df_comb = pd.DataFrame(combined_rows).sort_values(by=["SNR", "CBF_RMSE"], ascending=[True, True])
    csv_comb = os.path.join(met_dir, "combined_snr10_snr15_comparison.csv")
    df_comb.to_csv(csv_comb, index=False)

    print("\n" + "=" * 85)
    print("COMBINED COMPARISON TABLE AT SNR 10 & 15 (WITH HISTORICAL BASELINES)")
    print("=" * 85)
    print(df_comb.to_string(index=False))

    # 7. Generate Publication Figures
    generate_publication_figures(df_summary, df_hist, df_phantom, saved_predictions, Y_gt, masks, fig_dir)

    # 8. Generate Reproducibility Manifest
    manifest = {
        "benchmark_name": "SNR-Wise 4-PLD U-Net Evaluation",
        "date_executed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "plds_seconds": [1.525, 2.025, 2.525, 3.025],
            "tau_seconds": 1.8,
            "snr_conditions": ["inf", "50", "20", "15", "10", "5"],
            "n_test_phantoms": N_phantoms,
            "seed_noise_base": 7000
        },
        "model_checkpoints": {
            "original_unet": {
                "path": base_ckpt_path,
                "sha256": hash_orig
            },
            "improved_unet_v2": {
                "path": v2_ckpt_path,
                "sha256": hash_v2
            }
        },
        "dataset_hashes": {
            "canonical_test_N100_s7": hash_test
        },
        "historical_baselines_provenance": "Supplied Comparison PDF (Unmodified)",
        "output_files": {
            "unet_summary_metrics": csv_summary,
            "unet_per_phantom_metrics": csv_phantom,
            "historical_baseline_reference": csv_hist,
            "combined_snr10_15_comparison": csv_comb
        }
    }
    with open(os.path.join(exp_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"\nSaved Reproducibility Manifest: {os.path.join(exp_dir, 'manifest.json')}")

    return df_summary, df_comb

def generate_publication_figures(df_summary, df_hist, df_phantom, saved_preds, Y_gt, masks, fig_dir):
    print("\n--- Generating Publication Figures ---")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.titlesize": 15
    })

    snr_order = ["5", "10", "15", "20", "50", "inf"]
    snr_x = [5, 10, 15, 20, 50, 75] # Map 'inf' to 75 for continuous x-axis display
    snr_tick_labels = ["5", "10", "15", "20", "50", "∞"]

    df_orig = df_summary[df_summary["Model"] == "Original 4-PLD U-Net"].copy()
    df_v2 = df_summary[df_summary["Model"] == "Improved 4-PLD U-Net v2"].copy()

    # Reorder according to snr_order
    df_orig["x"] = df_orig["SNR"].map(dict(zip(snr_order, snr_x)))
    df_v2["x"] = df_v2["SNR"].map(dict(zip(snr_order, snr_x)))
    df_orig = df_orig.sort_values("x")
    df_v2 = df_v2.sort_values("x")

    # 1. Fig 1 & 2 & 5: CBF & ATT RMSE vs SNR Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(df_orig["x"], df_orig["CBF_RMSE"], "o-", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax1.plot(df_v2["x"], df_v2["CBF_RMSE"], "D-", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    # Standalone historical reference points at SNR 10 & 15 (no interpolated lines)
    ax1.scatter([10, 15], [5.0300, 3.7590], color="#E65100", s=110, marker="*", label="Bayesian Ref (PDF)", zorder=5)
    ax1.scatter([10, 15], [2.9820, 2.0620], color="#6A1B9A", s=80, marker="^", label="NLS Ref (PDF)", zorder=5)
    ax1.set_xticks(snr_x)
    ax1.set_xticklabels(snr_tick_labels)
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("CBF RMSE (mL/100 g/min)")
    ax1.set_title("CBF RMSE vs SNR")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    ax2.plot(df_orig["x"], df_orig["ATT_RMSE"], "o-", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax2.plot(df_v2["x"], df_v2["ATT_RMSE"], "D-", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax2.scatter([10, 15], [0.0826, 0.0584], color="#E65100", s=110, marker="*", label="Bayesian Ref (PDF)", zorder=5)
    ax2.scatter([10, 15], [0.5062, 0.4951], color="#6A1B9A", s=80, marker="^", label="NLS Ref (PDF)", zorder=5)
    ax2.set_xticks(snr_x)
    ax2.set_xticklabels(snr_tick_labels)
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax2.set_ylabel("ATT RMSE (seconds)")
    ax2.set_title("ATT RMSE vs SNR")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    plt.tight_layout()
    p1 = os.path.join(fig_dir, "fig1_fig2_fig5_rmse_vs_snr_comparison.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Fig 3 & 4: MAE vs SNR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(df_orig["x"], df_orig["CBF_MAE"], "o--", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax1.plot(df_v2["x"], df_v2["CBF_MAE"], "D--", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax1.scatter([10, 15], [3.9680, 2.8490], color="#E65100", s=110, marker="*", label="Bayesian Ref (PDF)", zorder=5)
    ax1.scatter([10, 15], [2.0480, 1.4140], color="#6A1B9A", s=80, marker="^", label="NLS Ref (PDF)", zorder=5)
    ax1.set_xticks(snr_x)
    ax1.set_xticklabels(snr_tick_labels)
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("CBF MAE (mL/100 g/min)")
    ax1.set_title("CBF MAE vs SNR")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    ax2.plot(df_orig["x"], df_orig["ATT_MAE"], "o--", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax2.plot(df_v2["x"], df_v2["ATT_MAE"], "D--", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax2.scatter([10, 15], [0.0683, 0.0472], color="#E65100", s=110, marker="*", label="Bayesian Ref (PDF)", zorder=5)
    ax2.scatter([10, 15], [0.3411, 0.3259], color="#6A1B9A", s=80, marker="^", label="NLS Ref (PDF)", zorder=5)
    ax2.set_xticks(snr_x)
    ax2.set_xticklabels(snr_tick_labels)
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax2.set_ylabel("ATT MAE (seconds)")
    ax2.set_title("ATT MAE vs SNR")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    plt.tight_layout()
    p2 = os.path.join(fig_dir, "fig3_fig4_mae_vs_snr_comparison.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Fig 6: Mean Signed Bias vs SNR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(df_orig["x"], df_orig["CBF_Bias"], "o-", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax1.plot(df_v2["x"], df_v2["CBF_Bias"], "D-", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1.5)
    ax1.set_xticks(snr_x)
    ax1.set_xticklabels(snr_tick_labels)
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("CBF Mean Signed Bias (mL/100 g/min)")
    ax1.set_title("CBF Bias vs SNR")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    ax2.plot(df_orig["x"], df_orig["ATT_Bias"], "o-", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax2.plot(df_v2["x"], df_v2["ATT_Bias"], "D-", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1.5)
    ax2.set_xticks(snr_x)
    ax2.set_xticklabels(snr_tick_labels)
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax2.set_ylabel("ATT Mean Signed Bias (seconds)")
    ax2.set_title("ATT Bias vs SNR")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    plt.tight_layout()
    p3 = os.path.join(fig_dir, "fig6_bias_vs_snr_comparison.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")

    # 4. Fig 7: Lin's Concordance Correlation Coefficient (CCC) vs SNR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(df_orig["x"], df_orig["CBF_CCC"], "o-", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax1.plot(df_v2["x"], df_v2["CBF_CCC"], "D-", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax1.scatter([10, 15], [0.9606, 0.9671], color="#E65100", s=110, marker="*", label="Bayesian Ref (PDF)", zorder=5)
    ax1.scatter([10, 15], [0.9913, 0.9958], color="#6A1B9A", s=80, marker="^", label="NLS Ref (PDF)", zorder=5)
    ax1.set_xticks(snr_x)
    ax1.set_xticklabels(snr_tick_labels)
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("CBF Lin's CCC (dimensionless)")
    ax1.set_ylim(0.92, 1.005)
    ax1.set_title("CBF Lin's Concordance Correlation Coefficient (CCC)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    ax2.plot(df_orig["x"], df_orig["ATT_CCC"], "o-", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax2.plot(df_v2["x"], df_v2["ATT_CCC"], "D-", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax2.scatter([10, 15], [0.9914, 0.9944], color="#E65100", s=110, marker="*", label="Bayesian Ref (PDF)", zorder=5)
    ax2.scatter([10, 15], [0.6770, 0.6899], color="#6A1B9A", s=80, marker="^", label="NLS Ref (PDF)", zorder=5)
    ax2.set_xticks(snr_x)
    ax2.set_xticklabels(snr_tick_labels)
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax2.set_ylabel("ATT Lin's CCC (dimensionless)")
    ax2.set_ylim(0.60, 1.005)
    ax2.set_title("ATT Lin's Concordance Correlation Coefficient (CCC)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    plt.tight_layout()
    p4 = os.path.join(fig_dir, "fig7_ccc_vs_snr_comparison.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")

    # 5. Fig 8: Historical Baseline Comparison at SNR 10 & 15
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Models to display
    model_names = ["Bayesian (PDF)", "NLS (PDF)", "Point-wise DNN (PDF)", "Residual ELM (PDF)", "Original U-Net (Ours)", "U-Net v2 (Ours)"]
    cbf_snr10 = [5.0300, 2.9820, 4.2676, 4.2580, df_orig[df_orig["SNR"] == "10"]["CBF_RMSE"].values[0], df_v2[df_v2["SNR"] == "10"]["CBF_RMSE"].values[0]]
    cbf_snr15 = [3.7590, 2.0620, 3.5560, 3.5597, df_orig[df_orig["SNR"] == "15"]["CBF_RMSE"].values[0], df_v2[df_v2["SNR"] == "15"]["CBF_RMSE"].values[0]]
    att_snr10 = [0.0826, 0.5062, 0.3026, 0.3044, df_orig[df_orig["SNR"] == "10"]["ATT_RMSE"].values[0], df_v2[df_v2["SNR"] == "10"]["ATT_RMSE"].values[0]]
    att_snr15 = [0.0584, 0.4951, 0.2625, 0.2629, df_orig[df_orig["SNR"] == "15"]["ATT_RMSE"].values[0], df_v2[df_v2["SNR"] == "15"]["ATT_RMSE"].values[0]]

    x = np.arange(len(model_names))
    width = 0.35

    ax1.bar(x - width/2, cbf_snr10, width, label="SNR 10", color="#E65100", alpha=0.85)
    ax1.bar(x + width/2, cbf_snr15, width, label="SNR 15", color="#1E88E5", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("CBF RMSE (ml/100g/min)")
    ax1.set_title("CBF RMSE Benchmark at SNR 10 & 15\n(Historical Reference vs U-Net)")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend()

    ax2.bar(x - width/2, att_snr10, width, label="SNR 10", color="#E65100", alpha=0.85)
    ax2.bar(x + width/2, att_snr15, width, label="SNR 15", color="#1E88E5", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("ATT RMSE (s)")
    ax2.set_title("ATT RMSE Benchmark at SNR 10 & 15\n(Historical Reference vs U-Net)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    plt.tight_layout()
    p5 = os.path.join(fig_dir, "fig8_historical_baseline_comparison_snr10_15.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Saved: {p5}")

    # 6. Fig 9: Representative Slice Error Maps across SNR (50, 15, 10, 5)
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    slice_idx = 0
    gt_cbf_slice = Y_gt[slice_idx, 0]
    gt_att_slice = Y_gt[slice_idx, 1]
    mask_slice = masks[slice_idx, 0] > 0.5

    snrs_vis = ["50", "15", "10", "5"]
    for row, s_val in enumerate(snrs_vis):
        orig_key = f"Original_4-PLD_U-Net_snr_{s_val}"
        v2_key = f"Improved_4-PLD_U-Net_v2_snr_{s_val}"
        
        cbf_o = saved_preds[orig_key]["cbf"][slice_idx]
        cbf_v = saved_preds[v2_key]["cbf"][slice_idx]
        
        err_o = np.abs(cbf_o - gt_cbf_slice) * mask_slice
        err_v = np.abs(cbf_v - gt_cbf_slice) * mask_slice

        # Col 0: Ground Truth
        axes[row, 0].imshow(gt_cbf_slice, cmap="inferno", vmin=0, vmax=80)
        axes[row, 0].set_ylabel(f"SNR {s_val}", fontsize=13, fontweight="bold")
        if row == 0: axes[row, 0].set_title("Ground Truth CBF")

        # Col 1: Original U-Net
        axes[row, 1].imshow(cbf_o, cmap="inferno", vmin=0, vmax=80)
        if row == 0: axes[row, 1].set_title("Original U-Net")

        # Col 2: U-Net v2
        axes[row, 2].imshow(cbf_v, cmap="inferno", vmin=0, vmax=80)
        if row == 0: axes[row, 2].set_title("Improved U-Net v2")

        # Col 3: Error Original
        axes[row, 3].imshow(err_o, cmap="magma", vmin=0, vmax=15)
        if row == 0: axes[row, 3].set_title("|Original Error|")

        # Col 4: Error v2
        axes[row, 4].imshow(err_v, cmap="magma", vmin=0, vmax=15)
        if row == 0: axes[row, 4].set_title("|U-Net v2 Error|")

        for c in range(5):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    plt.tight_layout()
    p6 = os.path.join(fig_dir, "fig9_prediction_error_maps_across_snr.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Saved: {p6}")

def run_challenge_snr_benchmark(df_ordinary_summary):
    ws = get_workspace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 85)
    print("CHALLENGE PHANTOMS (N=40): EVALUATION ACROSS 6 SNR CONDITIONS")
    print("=" * 85)

    exp_dir = os.path.join(ws.root, "experiments", "snr_benchmark_4pld")
    met_dir = os.path.join(exp_dir, "metrics")
    fig_dir = os.path.join(exp_dir, "figures")

    # Load Models & Checkpoints
    base_dir = ws.get_path("final_model")
    v2_dir = os.path.join(ws.root, "final_model_4pld_spatial_v2")

    with open(os.path.join(base_dir, "normalization_stats.json"), "r") as f:
        base_stats = json.load(f)
    with open(os.path.join(v2_dir, "normalization_stats.json"), "r") as f:
        v2_stats = json.load(f)
    for k in ["X_mean", "X_std"]:
        base_stats[k] = np.array(base_stats[k], dtype=np.float32)
        v2_stats[k] = np.array(v2_stats[k], dtype=np.float32)

    model_orig = ASLUNet().to(device)
    model_orig.load_state_dict(torch.load(os.path.join(base_dir, "asl_unet_4pld_final.pth"), map_location=device, weights_only=False)["model_state_dict"])
    model_orig.eval()

    model_v2 = ASLHybridUNet().to(device)
    model_v2.load_state_dict(torch.load(os.path.join(v2_dir, "asl_unet_4pld_final.pth"), map_location=device, weights_only=False)["model_state_dict"])
    model_v2.eval()

    # Load Challenge Phantoms
    chal_npz = ws.get_path("data", "challenge_test_N40_s2026.npz")
    chal_data = np.load(chal_npz)
    Y_chal = chal_data["Y"]
    masks_chal = chal_data["masks"]
    cases_chal = chal_data["case_labels"]
    N_chal = len(Y_chal)

    SNR_LEVELS = [float("inf"), 50.0, 20.0, 15.0, 10.0, 5.0]
    snr_labels = ["inf", "50", "20", "15", "10", "5"]

    challenge_summary = []

    for snr, snr_label in zip(SNR_LEVELS, snr_labels):
        X_snr, _ = generate_snr_dataset(Y_chal, masks_chal, snr, seed=8000)

        for m_name, net, stats, arch_ver in [
            ("Original 4-PLD U-Net", model_orig, base_stats, "ASLUNet_v1"),
            ("Improved 4-PLD U-Net v2", model_v2, v2_stats, "ASLHybridUNet_v2")
        ]:
            X_norm = np.copy(X_snr)
            for c in range(4):
                X_norm[:, c] = (X_norm[:, c] - stats["X_mean"][c]) / stats["X_std"][c]

            preds_cbf, preds_att = [], []
            with torch.no_grad():
                for i in range(0, N_chal, 16):
                    xb = torch.tensor(X_norm[i:i+16], dtype=torch.float32, device=device)
                    out = net(xb).cpu().numpy()
                    c = out[:, 0] * stats["cbf_std"] + stats["cbf_mean"]
                    a = out[:, 1] * stats["att_std"] + stats["att_mean"]
                    preds_cbf.append(np.clip(c, CBF_MIN, CBF_MAX))
                    preds_att.append(np.clip(a, ATT_MIN, ATT_MAX))

            cbf_pred = np.concatenate(preds_cbf, axis=0)
            att_pred = np.concatenate(preds_att, axis=0)

            for i in range(N_chal):
                bg = masks_chal[i, 0] <= 0.5
                cbf_pred[i][bg] = 0.0
                att_pred[i][bg] = 0.0

            vm_all = masks_chal[:, 0] > 0.5
            met_c_all = calc_metrics(Y_chal[:, 0][vm_all], cbf_pred[vm_all])
            met_a_all = calc_metrics(Y_chal[:, 1][vm_all], att_pred[vm_all])

            # Per-case breakdown
            case_metrics = {}
            for c_name in ["focal_lesion", "hydrocephalus", "discontinuous_step", "extreme_watershed"]:
                c_idx = np.where(cases_chal == c_name)[0]
                c_vm = masks_chal[c_idx, 0] > 0.5
                mc = calc_metrics(Y_chal[c_idx, 0][c_vm], cbf_pred[c_idx][c_vm])
                case_metrics[f"{c_name}_CBF_RMSE"] = mc["RMSE"]

            challenge_summary.append({
                "Model": m_name,
                "Architecture": arch_ver,
                "SNR": snr_label,
                "Challenge_CBF_RMSE": met_c_all["RMSE"],
                "Challenge_CBF_MAE": met_c_all["MAE"],
                "Challenge_ATT_RMSE": met_a_all["RMSE"],
                "Challenge_ATT_MAE": met_a_all["MAE"],
                "Focal_CBF_RMSE": case_metrics["focal_lesion_CBF_RMSE"],
                "Hydro_CBF_RMSE": case_metrics["hydrocephalus_CBF_RMSE"],
                "Step_CBF_RMSE": case_metrics["discontinuous_step_CBF_RMSE"],
                "Watershed_CBF_RMSE": case_metrics["extreme_watershed_CBF_RMSE"]
            })
            print(f"  [{m_name} | SNR {snr_label:3s}] Chal CBF RMSE: {met_c_all['RMSE']:.3f} | Focal: {case_metrics['focal_lesion_CBF_RMSE']:.3f} | Step: {case_metrics['discontinuous_step_CBF_RMSE']:.3f}")

    df_chal = pd.DataFrame(challenge_summary)
    df_chal.to_csv(os.path.join(met_dir, "snr_wise_challenge_results.csv"), index=False)

    # Figure 10: Ordinary vs Challenge comparison across SNRs
    snr_order = ["5", "10", "15", "20", "50", "inf"]
    snr_x = [5, 10, 15, 20, 50, 75]
    snr_tick_labels = ["5", "10", "15", "20", "50", "∞"]

    df_chal_orig = df_chal[df_chal["Model"] == "Original 4-PLD U-Net"].copy()
    df_chal_v2 = df_chal[df_chal["Model"] == "Improved 4-PLD U-Net v2"].copy()
    df_ord_orig = df_ordinary_summary[df_ordinary_summary["Model"] == "Original 4-PLD U-Net"].copy()
    df_ord_v2 = df_ordinary_summary[df_ordinary_summary["Model"] == "Improved 4-PLD U-Net v2"].copy()

    for d in [df_chal_orig, df_chal_v2, df_ord_orig, df_ord_v2]:
        d["x"] = d["SNR"].map(dict(zip(snr_order, snr_x)))
        d.sort_values("x", inplace=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Panel 1: Ordinary Smooth Phantoms
    ax1.plot(df_ord_orig["x"], df_ord_orig["CBF_RMSE"], "o-", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax1.plot(df_ord_v2["x"], df_ord_v2["CBF_RMSE"], "D-", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax1.set_xticks(snr_x)
    ax1.set_xticklabels(snr_tick_labels)
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("CBF RMSE (ml/100g/min)")
    ax1.set_title("Standard Brain Parenchyma (N=100)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    # Panel 2: Spatial Challenge Phantoms
    ax2.plot(df_chal_orig["x"], df_chal_orig["Challenge_CBF_RMSE"], "o--", color="#D9534F", linewidth=2.5, markersize=8, label="Original 4-PLD U-Net")
    ax2.plot(df_chal_v2["x"], df_chal_v2["Challenge_CBF_RMSE"], "D--", color="#2E7D32", linewidth=2.5, markersize=8, label="Improved 4-PLD U-Net v2")
    ax2.set_xticks(snr_x)
    ax2.set_xticklabels(snr_tick_labels)
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax2.set_ylabel("CBF RMSE (ml/100g/min)")
    ax2.set_title("Spatial Challenge Phantoms (N=40)\n(Focal Ischemia, Hydrocephalus, Step, Watershed)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    plt.tight_layout()
    p10 = os.path.join(fig_dir, "fig10_ordinary_vs_challenge_by_snr.png")
    plt.savefig(p10, dpi=300)
    plt.close()
    print(f"Saved: {p10}")

if __name__ == "__main__":
    df_summary, df_comb = run_snr_benchmark()
    run_challenge_snr_benchmark(df_summary)
