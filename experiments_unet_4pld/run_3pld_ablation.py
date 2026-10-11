import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_gen import (
    PLDs, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SIG_REF_SC, tau, alpha, beta, T1t, T1a, lmbda
)
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.dataset import (
    SpatialASLDataset,
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)
from experiments_unet_4pld.train import MaskedLoss
from experiments_unet_4pld.evaluate import (
    evaluate_unet,
    calc_metrics,
    calc_masked_ssim,
    calc_masked_psnr,
    train_pointwise_dnn,
    evaluate_pointwise_dnn
)

def compute_signals_vec_plds(cbf, att, plds, ld=tau):
    """Buxton ASL model vector evaluation for arbitrary PLD subset."""
    f_per_s = (cbf / (6000.0 * lmbda))[:, None]
    delta   = att[:, None]
    pld     = np.asarray(plds)[None, :]
    prefix  = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s
    e_att   = np.exp(-delta / T1a)
    t1      = np.exp(-np.maximum(pld - delta,      0.0) / T1t)
    t2      = np.exp(-np.maximum(ld  + pld - delta, 0.0) / T1t)
    return prefix * e_att * (t1 - t2)

def evaluate_bayesian_plds(
    X_test,
    snrs_test,
    masks_test,
    plds,
    device=None,
    sig_ref_sc=SIG_REF_SC,
    grid_res=100,
    batch_size=8192
):
    """Calibrated GPU Bayesian grid estimator for arbitrary PLD subset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    N, C, H, W = X_test.shape
    cbf_out = np.zeros((N, H, W), dtype=np.float32)
    att_out = np.zeros((N, H, W), dtype=np.float32)

    cbf_grid = np.linspace(CBF_MIN, CBF_MAX, grid_res, dtype=np.float32)
    att_grid = np.linspace(ATT_MIN, ATT_MAX, grid_res, dtype=np.float32)
    CC, AA = np.meshgrid(cbf_grid, att_grid)
    grid_cbf_t = torch.tensor(CC.flatten(), dtype=torch.float32, device=device)
    grid_att_t = torch.tensor(AA.flatten(), dtype=torch.float32, device=device)

    # Expected signal matching Simulator A observation: 2.0 * compute_signals_vec_plds * SCALE
    grid_sig_np = 2.0 * compute_signals_vec_plds(CC.flatten(), AA.flatten(), plds=plds, ld=tau) * SCALE
    grid_sig_t = torch.tensor(grid_sig_np, dtype=torch.float32, device=device) # [G, C]
    norm_S = torch.sum(grid_sig_t ** 2, dim=1, keepdim=True).T # [1, G]

    with torch.no_grad():
        for i in range(N):
            valid = masks_test[i, 0] > 0.5
            if not np.any(valid):
                continue
            X_vox = X_test[i][:, valid].T # [V, C]
            V = len(X_vox)
            snr = float(snrs_test[i])
            sigma_eff = (np.sqrt(2.0) * sig_ref_sc / snr) if (snr < 1e5 and snr > 0) else 1e-4

            cbf_preds_i = []
            att_preds_i = []

            for v_start in range(0, V, batch_size):
                v_end = min(v_start + batch_size, V)
                xb = torch.tensor(X_vox[v_start:v_end], dtype=torch.float32, device=device)
                norm_X = torch.sum(xb ** 2, dim=1, keepdim=True)
                sq_diff = norm_X - 2.0 * torch.matmul(xb, grid_sig_t.T) + norm_S
                log_L = -0.5 * sq_diff / (sigma_eff ** 2)
                log_L -= torch.max(log_L, dim=1, keepdim=True)[0]
                L = torch.exp(log_L)
                P = L / (torch.sum(L, dim=1, keepdim=True) + 1e-12)

                cb_p = torch.matmul(P, grid_cbf_t).cpu().numpy()
                at_p = torch.matmul(P, grid_att_t).cpu().numpy()
                cbf_preds_i.append(cb_p)
                att_preds_i.append(at_p)

            cbf_out[i][valid] = np.clip(np.concatenate(cbf_preds_i), CBF_MIN, CBF_MAX)
            att_out[i][valid] = np.clip(np.concatenate(att_preds_i), ATT_MIN, ATT_MAX)

    duration = time.time() - start_time
    return cbf_out, att_out, duration

def train_unet_subset_single_seed(
    train_data,
    val_data,
    stats,
    config,
    seed,
    device,
    save_dir,
    subset_key
):
    """Trains 3-PLD ASLUNet for a specific subset and seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_tr_n, Y_tr_n, masks_tr, snrs_tr = train_data
    X_val_n, Y_val_n, masks_val, snrs_val = val_data

    batch_size = config["training"]["batch_size"]
    train_ds = SpatialASLDataset(X_tr_n, Y_tr_n, masks_tr, snrs_tr)
    val_ds = SpatialASLDataset(X_val_n, Y_val_n, masks_val, snrs_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ASLUNet(
        in_channels=3,
        out_channels=2,
        features=config["model"]["features"]
    ).to(device)

    criterion = MaskedLoss(loss_type="l1")
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_factor = config["training"].get("lr_factor", 0.5)
    lr_patience = config["training"].get("lr_patience", 5)
    min_lr = config["training"].get("min_lr", 1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )

    max_epochs = config["training"]["max_epochs"]
    early_stop_patience = config["training"]["patience"]

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        n_tr_batches = 0
        for xb, yb, mb, _ in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_accum += loss.item()
            n_tr_batches += 1

        train_loss = train_loss_accum / max(1, n_tr_batches)

        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb, mb, _ in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb, mb)
                val_loss_accum += loss.item()
                n_val_batches += 1

        val_loss = val_loss_accum / max(1, n_val_batches)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_ckpt_path = os.path.join(ckpt_dir, f"unet_3pld_{subset_key}_best_seed_{seed}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "stats": stats,
                "subset_key": subset_key,
                "seed": seed
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    duration = time.time() - start_time
    return best_ckpt_path, best_epoch, duration

def main():
    parser = argparse.ArgumentParser(description="Run 3-PLD Ablation Study")
    parser.add_argument("--config", type=str, default="experiments_unet_4pld/config.json")
    parser.add_argument("--quick", action="store_true", help="Run seed 42 only for rapid verification")
    args = parser.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(args.config))
    data_dir = os.path.join(exp_dir, "data")
    results_dir = os.path.join(exp_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    seeds = [42] if args.quick else config["training"]["seeds"]

    # Load frozen data
    n_tr = config["data"]["n_train_phantoms"]
    n_val = config["data"]["n_val_phantoms"]
    n_te = config["data"]["n_test_phantoms"]

    tr_data = np.load(os.path.join(data_dir, f"train_N{n_tr}_s{config['data']['train_seed']}.npz"))
    val_data = np.load(os.path.join(data_dir, f"val_N{n_val}_s{config['data']['val_seed']}.npz"))
    te_data = np.load(os.path.join(data_dir, f"test_N{n_te}_s{config['data']['test_seed']}.npz"))

    X_tr_full, Y_tr, masks_tr, snrs_tr = tr_data["X"], tr_data["Y"], tr_data["masks"], tr_data["snrs"]
    X_val_full, Y_val, masks_val, snrs_val = val_data["X"], val_data["Y"], val_data["masks"], val_data["snrs"]
    X_te_full, Y_te, masks_te, snrs_te = te_data["X"], te_data["Y"], te_data["masks"], te_data["snrs"]

    valid_mask_te = masks_te[:, 0] > 0.5
    gt_cbf_te = Y_te[:, 0][valid_mask_te]
    gt_att_te = Y_te[:, 1][valid_mask_te]

    # Baseline 4-PLD U-Net metrics for comparison
    seed_csv_4pld = os.path.join(results_dir, "per_seed_metrics.csv")
    if os.path.exists(seed_csv_4pld):
        df_4pld = pd.read_csv(seed_csv_4pld)
        baseline_4pld_cbf = df_4pld["CBF_RMSE"].mean()
        baseline_4pld_att = df_4pld["ATT_RMSE"].mean()
    else:
        baseline_4pld_cbf = 0.9912
        baseline_4pld_att = 0.0569

    print(f"\n================ 3-PLD ABLATION STUDY ================")
    print(f"4-PLD U-Net Baseline: CBF RMSE = {baseline_4pld_cbf:.4f}, ATT RMSE = {baseline_4pld_att:.4f}")
    print(f"Seeds to test: {seeds}")

    SUBSETS = {
        "drop_P4": {
            "name": "Subset 1 (Drop P4 = 3.025s)",
            "indices": [0, 1, 2],
            "plds": [1.525, 2.025, 2.525],
            "description": "Retains early wash-in and mid PLDs; drops late wash-out"
        },
        "drop_P3": {
            "name": "Subset 2 (Drop P3 = 2.525s)",
            "indices": [0, 1, 3],
            "plds": [1.525, 2.025, 3.025],
            "description": "Retains maximum time span [1.525, 3.025s]; drops mid-late PLD"
        },
        "drop_P2": {
            "name": "Subset 3 (Drop P2 = 2.025s)",
            "indices": [0, 2, 3],
            "plds": [1.525, 2.525, 3.025],
            "description": "Retains wide span; drops mid-early PLD"
        },
        "drop_P1": {
            "name": "Subset 4 (Drop P1 = 1.525s)",
            "indices": [1, 2, 3],
            "plds": [2.025, 2.525, 3.025],
            "description": "Drops earliest PLD (loss of early wash-in information)"
        }
    }

    all_seed_records = []
    subset_summary_records = []
    subset_predictions = {}

    for s_key, s_info in SUBSETS.items():
        print(f"\n>>> Evaluating {s_info['name']} | PLDs: {s_info['plds']} <<<")
        idx = s_info["indices"]
        plds = s_info["plds"]

        X_tr_sub = X_tr_full[:, idx]
        X_val_sub = X_val_full[:, idx]
        X_te_sub = X_te_full[:, idx]

        # Normalization computed strictly on training phantoms
        stats_sub = compute_normalization_stats(X_tr_sub, Y_tr, masks_tr)
        X_tr_n, Y_tr_n = normalize_dataset(X_tr_sub, Y_tr, masks_tr, stats_sub)
        X_val_n, Y_val_n = normalize_dataset(X_val_sub, Y_val, masks_val, stats_sub)
        X_te_n, Y_te_n = normalize_dataset(X_te_sub, Y_te, masks_te, stats_sub)

        seed_cbf_rmses = []
        seed_att_rmses = []
        best_sub_model = None
        best_sub_cbf = float("inf")

        for seed in seeds:
            print(f"  Training Seed {seed}...")
            ckpt_path, best_ep, dur = train_unet_subset_single_seed(
                train_data=(X_tr_n, Y_tr_n, masks_tr, snrs_tr),
                val_data=(X_val_n, Y_val_n, masks_val, snrs_val),
                stats=stats_sub,
                config=config,
                seed=seed,
                device=device,
                save_dir=exp_dir,
                subset_key=s_key
            )

            # Evaluate model
            ckpt = torch.load(ckpt_path, weights_only=False)
            eval_model = ASLUNet(in_channels=3, out_channels=2, features=config["model"]["features"]).to(device)
            eval_model.load_state_dict(ckpt["model_state_dict"])
            eval_model.eval()

            cbf_p, att_p, t_inf = evaluate_unet(eval_model, X_te_n, stats_sub, device)
            met_c = calc_metrics(gt_cbf_te, cbf_p[valid_mask_te])
            met_a = calc_metrics(gt_att_te, att_p[valid_mask_te])

            rec = {
                "Subset": s_key,
                "Subset_Name": s_info["name"],
                "Seed": seed,
                "Best_Epoch": best_ep,
                "Train_Time_s": round(dur, 1),
                "CBF_RMSE": met_c["RMSE"],
                "CBF_MAE": met_c["MAE"],
                "CBF_R2": met_c["R2"],
                "CBF_CCC": met_c["CCC"],
                "ATT_RMSE": met_a["RMSE"],
                "ATT_MAE": met_a["MAE"],
                "ATT_R2": met_a["R2"],
                "ATT_CCC": met_a["CCC"]
            }
            all_seed_records.append(rec)
            seed_cbf_rmses.append(met_c["RMSE"])
            seed_att_rmses.append(met_a["RMSE"])

            if met_c["RMSE"] < best_sub_cbf:
                best_sub_cbf = met_c["RMSE"]
                best_sub_model = eval_model
                subset_predictions[s_key] = (cbf_p, att_p)

            print(f"    Seed {seed}: CBF RMSE = {met_c['RMSE']:.4f}, ATT RMSE = {met_a['RMSE']:.4f} (Best Ep: {best_ep})")

        # Evaluate Bayesian baseline on this 3-PLD subset
        print(f"  Evaluating Calibrated Bayesian on {s_key}...")
        cbf_b, att_b, _ = evaluate_bayesian_plds(X_te_sub, snrs_te, masks_te, plds=plds, device=device)
        met_c_b = calc_metrics(gt_cbf_te, cbf_b[valid_mask_te])
        met_a_b = calc_metrics(gt_att_te, att_b[valid_mask_te])

        # Evaluate Point-wise DNN on this 3-PLD subset
        print(f"  Training and evaluating Point-wise DNN on {s_key}...")
        v_tr = masks_tr[:, 0] > 0.5
        X_tr_vox_sub = X_tr_sub.transpose(0, 2, 3, 1)[v_tr]
        Y_tr_vox = Y_tr.transpose(0, 2, 3, 1)[v_tr]
        if len(X_tr_vox_sub) > 200_000:
            s_idx = np.random.choice(len(X_tr_vox_sub), size=200_000, replace=False)
            X_tr_vox_sub = X_tr_vox_sub[s_idx]
            Y_tr_vox = Y_tr_vox[s_idx]
        mlp_c, mlp_a = train_pointwise_dnn(X_tr_vox_sub, Y_tr_vox, stats_sub, device, epochs=30)
        cbf_dnn, att_dnn, _ = evaluate_pointwise_dnn(mlp_c, mlp_a, X_te_sub, masks_te, stats_sub, device)
        met_c_dnn = calc_metrics(gt_cbf_te, cbf_dnn[valid_mask_te])
        met_a_dnn = calc_metrics(gt_att_te, att_dnn[valid_mask_te])

        # Calculate SSIM and PSNR for U-Net
        cbf_best_p, att_best_p = subset_predictions[s_key]
        ssim_c_list, ssim_a_list = [], []
        psnr_c_list, psnr_a_list = [], []
        for i in range(len(X_te_sub)):
            m_2d = masks_te[i, 0]
            ssim_c_list.append(calc_masked_ssim(Y_te[i, 0], cbf_best_p[i], m_2d, data_range=100.0))
            ssim_a_list.append(calc_masked_ssim(Y_te[i, 1], att_best_p[i], m_2d, data_range=2.5))
            psnr_c_list.append(calc_masked_psnr(Y_te[i, 0], cbf_best_p[i], m_2d, data_range=100.0))
            psnr_a_list.append(calc_masked_psnr(Y_te[i, 1], att_best_p[i], m_2d, data_range=2.5))

        mean_cbf_rmse = float(np.mean(seed_cbf_rmses))
        std_cbf_rmse = float(np.std(seed_cbf_rmses)) if len(seed_cbf_rmses) > 1 else 0.0
        mean_att_rmse = float(np.mean(seed_att_rmses))
        std_att_rmse = float(np.std(seed_att_rmses)) if len(seed_att_rmses) > 1 else 0.0

        rel_diff_cbf = 100.0 * (mean_cbf_rmse - baseline_4pld_cbf) / baseline_4pld_cbf
        rel_diff_att = 100.0 * (mean_att_rmse - baseline_4pld_att) / baseline_4pld_att

        # Comparability verdict against prespecified +10% margin
        comparable_cbf = (rel_diff_cbf <= 10.0)
        comparable_att = (rel_diff_att <= 10.0)
        overall_comparable = comparable_cbf and comparable_att

        subset_summary_records.append({
            "Subset": s_key,
            "Subset_Name": s_info["name"],
            "PLDs": str(plds),
            "Description": s_info["description"],
            "UNet_CBF_RMSE": f"{mean_cbf_rmse:.4f} ± {std_cbf_rmse:.4f}",
            "UNet_CBF_RMSE_mean": mean_cbf_rmse,
            "CBF_Rel_Change_%": round(rel_diff_cbf, 2),
            "CBF_Comparable": comparable_cbf,
            "UNet_ATT_RMSE": f"{mean_att_rmse:.4f} ± {std_att_rmse:.4f}",
            "UNet_ATT_RMSE_mean": mean_att_rmse,
            "ATT_Rel_Change_%": round(rel_diff_att, 2),
            "ATT_Comparable": comparable_att,
            "Overall_Status": "Comparable (Within +10%)" if overall_comparable else "Degraded (> +10%)",
            "CBF_SSIM": round(float(np.mean(ssim_c_list)), 4),
            "CBF_PSNR": round(float(np.mean(psnr_c_list)), 2),
            "ATT_SSIM": round(float(np.mean(ssim_a_list)), 4),
            "ATT_PSNR": round(float(np.mean(psnr_a_list)), 2),
            "Bayesian_CBF_RMSE": round(met_c_b["RMSE"], 4),
            "Bayesian_ATT_RMSE": round(met_a_b["RMSE"], 4),
            "DNN_CBF_RMSE": round(met_c_dnn["RMSE"], 4),
            "DNN_ATT_RMSE": round(met_a_dnn["RMSE"], 4),
        })

    # Save CSVs
    per_seed_df = pd.DataFrame(all_seed_records)
    summary_df = pd.DataFrame(subset_summary_records)

    per_seed_csv = os.path.join(results_dir, "ablation_3pld_per_seed.csv")
    summary_csv = os.path.join(results_dir, "ablation_3pld_summary.csv")
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n================ 3-PLD ABLATION SUMMARY ================")
    print(summary_df[["Subset", "PLDs", "UNet_CBF_RMSE", "CBF_Rel_Change_%", "UNet_ATT_RMSE", "ATT_Rel_Change_%", "Overall_Status"]].to_string())

    # ----------------- PLOTTING ABLATION FIGURES -----------------
    print("\nGenerating ablation summary plots...")
    # 1. Bar Chart: 4-PLD vs 3-PLD Subsets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sub_labels = ["4-PLD Baseline"] + [SUBSETS[k]["name"].replace("Subset ", "S") for k in SUBSETS.keys()]
    cbf_vals_unet = [baseline_4pld_cbf] + [r["UNet_CBF_RMSE_mean"] for r in subset_summary_records]
    att_vals_unet = [baseline_4pld_att] + [r["UNet_ATT_RMSE_mean"] for r in subset_summary_records]

    # Reference threshold lines (+10% margin)
    cbf_thresh = baseline_4pld_cbf * 1.10
    att_thresh = baseline_4pld_att * 1.10

    x = np.arange(len(sub_labels))
    width = 0.55

    # CBF Plot
    bars1 = ax1.bar(x, cbf_vals_unet, width, color=["#1f77b4"] + ["#2ca02c" if v <= cbf_thresh else "#d62728" for v in cbf_vals_unet[1:]])
    ax1.axhline(baseline_4pld_cbf, color="blue", linestyle="--", alpha=0.7, label=f"4-PLD Baseline ({baseline_4pld_cbf:.3f})")
    ax1.axhline(cbf_thresh, color="red", linestyle=":", alpha=0.8, label=f"+10% Margin ({cbf_thresh:.3f})")
    ax1.set_ylabel("CBF RMSE (ml/100g/min)")
    ax1.set_title("CBF Estimation: 4-PLD vs 3-PLD Subsets")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_labels, rotation=25, ha="right", fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)

    # ATT Plot
    bars2 = ax2.bar(x, att_vals_unet, width, color=["#1f77b4"] + ["#2ca02c" if v <= att_thresh else "#d62728" for v in att_vals_unet[1:]])
    ax2.axhline(baseline_4pld_att, color="blue", linestyle="--", alpha=0.7, label=f"4-PLD Baseline ({baseline_4pld_att:.4f} s)")
    ax2.axhline(att_thresh, color="red", linestyle=":", alpha=0.8, label=f"+10% Margin ({att_thresh:.4f} s)")
    ax2.set_ylabel("ATT RMSE (s)")
    ax2.set_title("ATT Estimation: 4-PLD vs 3-PLD Subsets")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub_labels, rotation=25, ha="right", fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f"{yval:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ablation_3pld_barchart.png"), dpi=200)
    plt.close()

    # 2. Visual Prediction Maps Comparison on Sample Phantom
    s_idx = 0
    mask_0 = masks_te[s_idx, 0] > 0.5
    gt_c_0 = np.where(mask_0, Y_te[s_idx, 0], np.nan)
    gt_a_0 = np.where(mask_0, Y_te[s_idx, 1], np.nan)

    # Load 4-PLD best prediction
    cbf_4pld_path = os.path.join(results_dir, "predictions", "cbf_unet.npy")
    att_4pld_path = os.path.join(results_dir, "predictions", "att_unet.npy")
    if os.path.exists(cbf_4pld_path):
        cbf_4pld_0 = np.where(mask_0, np.load(cbf_4pld_path)[s_idx], np.nan)
        att_4pld_0 = np.where(mask_0, np.load(att_4pld_path)[s_idx], np.nan)
    else:
        cbf_4pld_0, att_4pld_0 = gt_c_0, gt_a_0

    # Best 3-PLD subset and Worst 3-PLD subset
    sorted_subsets = sorted(subset_summary_records, key=lambda x: x["UNet_CBF_RMSE_mean"])
    best_key = sorted_subsets[0]["Subset"]
    worst_key = sorted_subsets[-1]["Subset"]

    cbf_best_3pld_0 = np.where(mask_0, subset_predictions[best_key][0][s_idx], np.nan)
    att_best_3pld_0 = np.where(mask_0, subset_predictions[best_key][1][s_idx], np.nan)
    cbf_worst_3pld_0 = np.where(mask_0, subset_predictions[worst_key][0][s_idx], np.nan)
    att_worst_3pld_0 = np.where(mask_0, subset_predictions[worst_key][1][s_idx], np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    # Top Row: CBF
    im0 = axes[0, 0].imshow(gt_c_0, cmap="hot", vmin=0, vmax=100)
    axes[0, 0].set_title("Ground Truth CBF")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(cbf_4pld_0, cmap="hot", vmin=0, vmax=100)
    axes[0, 1].set_title(f"4-PLD U-Net (RMSE {baseline_4pld_cbf:.2f})")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(cbf_best_3pld_0, cmap="hot", vmin=0, vmax=100)
    axes[0, 2].set_title(f"Best 3-PLD ({best_key})\n(RMSE {sorted_subsets[0]['UNet_CBF_RMSE_mean']:.2f})")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im3 = axes[0, 3].imshow(cbf_worst_3pld_0, cmap="hot", vmin=0, vmax=100)
    axes[0, 3].set_title(f"Worst 3-PLD ({worst_key})\n(RMSE {sorted_subsets[-1]['UNet_CBF_RMSE_mean']:.2f})")
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Bottom Row: ATT
    im4 = axes[1, 0].imshow(gt_a_0, cmap="viridis", vmin=0.5, vmax=3.0)
    axes[1, 0].set_title("Ground Truth ATT")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(att_4pld_0, cmap="viridis", vmin=0.5, vmax=3.0)
    axes[1, 1].set_title(f"4-PLD U-Net (RMSE {baseline_4pld_att:.3f})")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im6 = axes[1, 2].imshow(att_best_3pld_0, cmap="viridis", vmin=0.5, vmax=3.0)
    axes[1, 2].set_title(f"Best 3-PLD ({best_key})\n(RMSE {sorted_subsets[0]['UNet_ATT_RMSE_mean']:.3f})")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im7 = axes[1, 3].imshow(att_worst_3pld_0, cmap="viridis", vmin=0.5, vmax=3.0)
    axes[1, 3].set_title(f"Worst 3-PLD ({worst_key})\n(RMSE {sorted_subsets[-1]['UNet_ATT_RMSE_mean']:.3f})")
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ablation_3pld_sample_maps.png"), dpi=200)
    plt.close()

    print("\n================ ABLATION STUDY FINISHED SUCCESSFULLY ================")

if __name__ == "__main__":
    main()

