"""
Execution Script: Training-Set Size Generalization Study for 4-PLD ASL Spatial U-Net
Investigates model capacity, overfitting, and generalization across nested training sizes:
N in [100, 200, 400, 600, 1200] phantoms across 3 seeds [42, 123, 2024].
Fixed independent validation cohort (N=100) and test cohort (N=100).
Storage-safe: strictly contained under D:\ASL_WORKSPACE.
"""

import os
import sys
import time
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.dataset import (
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)
from experiments_unet_4pld.train import train_unet_single_seed
from experiments_unet_4pld.run_snr_benchmark import calc_metrics, calc_lin_ccc
from experiments_unet_4pld.generate_spatial_data import generate_spatial_dataset

def sha256_file(filepath):
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def evaluate_model_on_test(model, X_te, Y_te, masks_te, stats, device):
    """
    Evaluates model on fixed test cohort and calculates comprehensive physical metrics.
    """
    model.eval()
    N_te = len(X_te)
    X_norm = np.copy(X_te)
    for c in range(4):
        X_norm[:, c] = (X_norm[:, c] - stats["X_mean"][c]) / stats["X_std"][c]

    preds_cbf, preds_att = [], []
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, N_te, 16):
            xb = torch.tensor(X_norm[i:i+16], dtype=torch.float32, device=device)
            out = model(xb).cpu().numpy()
            c = out[:, 0] * stats["cbf_std"] + stats["cbf_mean"]
            a = out[:, 1] * stats["att_std"] + stats["att_mean"]
            preds_cbf.append(np.clip(c, 0.0, 100.0))
            preds_att.append(np.clip(a, 0.5, 3.0))

    infer_time = time.time() - start_time
    cbf_pred = np.concatenate(preds_cbf, axis=0)
    att_pred = np.concatenate(preds_att, axis=0)

    # Valid brain mask
    valid_mask = masks_te[:, 0] > 0.5
    gt_cbf = Y_te[:, 0][valid_mask]
    gt_att = Y_te[:, 1][valid_mask]
    pr_cbf = cbf_pred[valid_mask]
    pr_att = att_pred[valid_mask]

    mc = calc_metrics(gt_cbf, pr_cbf)
    ma = calc_metrics(gt_att, pr_att)
    ccc_c = calc_lin_ccc(gt_cbf, pr_cbf)
    ccc_a = calc_lin_ccc(gt_att, pr_att)

    return {
        "cbf_rmse": mc["RMSE"],
        "cbf_mae": mc["MAE"],
        "cbf_bias": mc["Bias"],
        "cbf_r2": mc["R2"],
        "cbf_ccc": ccc_c,
        "att_rmse": ma["RMSE"],
        "att_mae": ma["MAE"],
        "att_bias": ma["Bias"],
        "att_r2": ma["R2"],
        "att_ccc": ccc_a,
        "infer_time_s": infer_time
    }

def main():
    parser = argparse.ArgumentParser(description="Training Size Study for 4-PLD ASL Spatial U-Net")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 400, 600, 1200], help="Training sizes")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024], help="Random seeds")
    args = parser.parse_args()

    # 1. Setup Managed Workspace & Storage Guard
    ws = get_workspace(min_free_gb=5.0)
    ws.check_preflight_storage(required_gb=3.0)
    redir_info = ws.verify_storage_redirection()
    print("Storage Redirection Verified:", redir_info)

    study_dir = os.path.join(ws.root, "experiments", "training_size_study")
    ckpt_dir = os.path.join(study_dir, "checkpoints")
    hist_dir = os.path.join(study_dir, "training_histories")
    curve_dir = os.path.join(study_dir, "learning_curves")
    met_dir = os.path.join(study_dir, "metrics")
    for d in [ckpt_dir, hist_dir, curve_dir, met_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Load Fixed Independent Validation & Test Cohorts
    print("\n--- Loading Fixed Independent Evaluation Cohorts ---")
    val_path = os.path.join(ws.root, "data", "val_N100_s99.npz")
    test_path = os.path.join(ws.root, "data", "test_N100_s7.npz")
    
    val_data_raw = np.load(val_path)
    X_val_raw, Y_val_raw, masks_val = val_data_raw["X"], val_data_raw["Y"], val_data_raw["masks"]
    snrs_val = val_data_raw["snrs"] if "snrs" in val_data_raw else np.full(len(X_val_raw), 15.0, dtype=np.float32)

    test_data_raw = np.load(test_path)
    X_te_raw, Y_te_raw, masks_te = test_data_raw["X"], test_data_raw["Y"], test_data_raw["masks"]

    print(f"Validation Cohort: {val_path} ({len(X_val_raw)} phantoms, SHA256: {sha256_file(val_path)[:16]}...)")
    print(f"Test Cohort:       {test_path} ({len(X_te_raw)} phantoms, SHA256: {sha256_file(test_path)[:16]}...)")

    # 3. Load or Build Nested Master Training Pool
    train_600_path = os.path.join(ws.root, "data", "train_N600_s42.npz")
    train_600_raw = np.load(train_600_path)
    X_tr_600 = train_600_raw["X"]
    Y_tr_600 = train_600_raw["Y"]
    masks_tr_600 = train_600_raw["masks"]
    snrs_tr_600 = train_600_raw["snrs"] if "snrs" in train_600_raw else np.full(len(X_tr_600), 15.0, dtype=np.float32)

    max_req_size = max(args.sizes)
    if max_req_size > 600:
        nested_1200_path = os.path.join(ws.root, "data", "train_nested_N1200_s42.npz")
        if os.path.exists(nested_1200_path):
            print(f"Reusing existing nested 1200-phantom dataset: {nested_1200_path}")
            n1200_raw = np.load(nested_1200_path)
            X_tr_pool = n1200_raw["X"]
            Y_tr_pool = n1200_raw["Y"]
            masks_tr_pool = n1200_raw["masks"]
            snrs_tr_pool = n1200_raw["snrs"]
        else:
            print(f"Generating full nested master pool of {max_req_size} phantoms...")
            ws.check_preflight_storage(required_gb=2.0)
            X_tr_pool, Y_tr_pool, masks_tr_pool, snrs_tr_pool, _ = generate_spatial_dataset(
                n_phantoms=max_req_size,
                seed=42,
                img_size=64
            )
            # Verify exact nesting with 600 phantoms
            assert np.array_equal(X_tr_pool[:600], X_tr_600), "FATAL: Nested dataset first 600 phantoms do not match train_N600_s42.npz!"
            np.savez_compressed(nested_1200_path, X=X_tr_pool, Y=Y_tr_pool, masks=masks_tr_pool, snrs=snrs_tr_pool)
            print(f"Saved verified nested dataset: {nested_1200_path} (SHA-256: {sha256_file(nested_1200_path)[:16]}...)")
    else:
        X_tr_pool, Y_tr_pool, masks_tr_pool, snrs_tr_pool = X_tr_600, Y_tr_600, masks_tr_600, snrs_tr_600

    # Base training configuration
    with open("experiments_unet_4pld/config.json", "r") as f:
        base_config = json.load(f)

    results_records = []
    manifest_records = []

    # Build Phantom Data Manifest
    for p_i in range(len(X_tr_pool)):
        vm = int(np.sum(masks_tr_pool[p_i, 0] > 0.5))
        manifest_records.append({
            "phantom_id": f"train_{p_i:04d}",
            "split": "train",
            "generation_seed": 42,
            "stream_index": p_i,
            "valid_voxels": vm,
            "in_N100": p_i < 100,
            "in_N200": p_i < 200,
            "in_N400": p_i < 400,
            "in_N600": p_i < 600,
            "in_N1200": p_i < 1200
        })

    for p_i in range(len(X_val_raw)):
        vm = int(np.sum(masks_val[p_i, 0] > 0.5))
        manifest_records.append({
            "phantom_id": f"val_{p_i:04d}",
            "split": "val",
            "generation_seed": 99,
            "stream_index": p_i,
            "valid_voxels": vm,
            "in_N100": True,
            "in_N200": True,
            "in_N400": True,
            "in_N600": True,
            "in_N1200": True
        })

    for p_i in range(len(X_te_raw)):
        vm = int(np.sum(masks_te[p_i, 0] > 0.5))
        manifest_records.append({
            "phantom_id": f"test_{p_i:04d}",
            "split": "test",
            "generation_seed": 7,
            "stream_index": p_i,
            "valid_voxels": vm,
            "in_N100": True,
            "in_N200": True,
            "in_N400": True,
            "in_N600": True,
            "in_N1200": True
        })

    manifest_df = pd.DataFrame(manifest_records)
    manifest_csv = os.path.join(study_dir, "data_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Saved complete Data Manifest: {manifest_csv} ({len(manifest_df)} total phantoms cataloged)")

    # 4. Iterate Over Training Sizes & Seeds
    for N in args.sizes:
        print(f"\n{'='*70}\n>>> EXAMINING TRAINING-SET SIZE: N = {N} PHANTOMS <<<\n{'='*70}")
        X_sub = X_tr_pool[:N]
        Y_sub = Y_tr_pool[:N]
        masks_sub = masks_tr_pool[:N]
        snrs_sub = snrs_tr_pool[:N]

        # Compute stats STRICTLY on this specific training size
        stats_N = compute_normalization_stats(X_sub, Y_sub, masks_sub)
        X_sub_n, Y_sub_n = normalize_dataset(X_sub, Y_sub, masks_sub, stats_N)
        X_val_n, Y_val_n = normalize_dataset(X_val_raw, Y_val_raw, masks_val, stats_N)

        for seed in args.seeds:
            print(f"\n--- Condition: N = {N} | Seed = {seed} ---")
            ckpt_path = os.path.join(ckpt_dir, f"unet_N{N}_seed_{seed}.pth")
            hist_csv = os.path.join(hist_dir, f"history_N{N}_seed_{seed}.csv")

            # Check if this exact run can be reused from existing baseline
            can_reuse_baseline = False
            if N == 600:
                existing_ckpt = os.path.join(ws.root, "checkpoints", f"unet_best_seed_{seed}.pth")
                existing_hist = os.path.join(ws.root, "results", "logs", f"history_seed_{seed}.csv")
                if os.path.exists(existing_ckpt) and os.path.exists(existing_hist):
                    can_reuse_baseline = True
                    print(f"Reusing verified baseline run: {existing_ckpt}")
                    # Copy / link into study directory
                    torch.save(torch.load(existing_ckpt, weights_only=False), ckpt_path)
                    hist_df = pd.read_csv(existing_hist)
                    hist_df.to_csv(hist_csv, index=False)

            if not can_reuse_baseline:
                ws.check_preflight_storage(required_gb=1.0)
                cfg = dict(base_config)
                cfg["data"]["n_train_phantoms"] = N

                start_tr = time.time()
                _, best_vloss, hist_df, train_dur = train_unet_single_seed(
                    train_data=(X_sub_n, Y_sub_n, masks_sub, snrs_sub),
                    val_data=(X_val_n, Y_val_n, masks_val, snrs_val),
                    stats=stats_N,
                    config=cfg,
                    seed=seed,
                    device=device,
                    save_dir=study_dir
                )
                # Move to standard naming
                orig_saved_ckpt = os.path.join(study_dir, "checkpoints", f"unet_best_seed_{seed}.pth")
                if os.path.exists(orig_saved_ckpt):
                    os.replace(orig_saved_ckpt, ckpt_path)
                hist_df.to_csv(hist_csv, index=False)

            # Load checkpoint for final test evaluation
            ckpt_loaded = torch.load(ckpt_path, weights_only=False, map_location=device)
            eval_model = ASLUNet(
                in_channels=base_config["model"]["in_channels"],
                out_channels=base_config["model"]["out_channels"],
                features=base_config["model"]["features"]
            ).to(device)
            eval_model.load_state_dict(ckpt_loaded["model_state_dict"])

            test_metrics = evaluate_model_on_test(eval_model, X_te_raw, Y_te_raw, masks_te, stats_N, device)

            # Read history for best epoch and curves
            hist_df = pd.read_csv(hist_csv)
            best_epoch = int(ckpt_loaded["epoch"])
            best_val_loss = float(ckpt_loaded["val_loss"])
            train_loss_best = float(hist_df[hist_df["epoch"] == best_epoch]["train_loss"].values[0])
            gen_gap = best_val_loss - train_loss_best

            # Save individual learning curve plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            ax1.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss (L1)", color="#1976D2")
            ax1.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss (L1)", color="#F57C00")
            ax1.axvline(best_epoch, linestyle="--", color="gray", label=f"Best Ep {best_epoch}")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Masked Loss (Standardized)")
            ax1.set_title(f"Loss Curves (N={N}, Seed {seed})\nGap at Best: {gen_gap:+.4f}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(hist_df["epoch"], hist_df["val_cbf_rmse"], label="Val CBF RMSE (mL/100g/min)", color="#D32F2F")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("CBF RMSE", color="#D32F2F")
            ax2.tick_params(axis="y", labelcolor="#D32F2F")
            ax2.grid(True, alpha=0.3)

            ax2_r = ax2.twinx()
            ax2_r.plot(hist_df["epoch"], hist_df["val_att_rmse"], label="Val ATT RMSE (s)", color="#388E3C")
            ax2_r.set_ylabel("ATT RMSE (s)", color="#388E3C")
            ax2_r.tick_params(axis="y", labelcolor="#388E3C")
            ax2.set_title(f"Validation Convergence (N={N}, Seed {seed})")

            plt.tight_layout()
            curve_png = os.path.join(curve_dir, f"learning_curve_N{N}_seed_{seed}.png")
            plt.savefig(curve_png, dpi=180)
            plt.close()

            row = {
                "training_size": N,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_loss_at_best": train_loss_best,
                "generalization_gap": gen_gap,
                "cbf_rmse": test_metrics["cbf_rmse"],
                "cbf_mae": test_metrics["cbf_mae"],
                "cbf_bias": test_metrics["cbf_bias"],
                "cbf_r2": test_metrics["cbf_r2"],
                "cbf_ccc": test_metrics["cbf_ccc"],
                "att_rmse": test_metrics["att_rmse"],
                "att_mae": test_metrics["att_mae"],
                "att_bias": test_metrics["att_bias"],
                "att_r2": test_metrics["att_r2"],
                "att_ccc": test_metrics["att_ccc"],
                "checkpoint_path": ckpt_path,
                "checkpoint_sha256": sha256_file(ckpt_path),
                "history_path": hist_csv
            }
            results_records.append(row)
            print(f"Completed N={N}, Seed {seed} -> Test CBF RMSE: {test_metrics['cbf_rmse']:.4f}, ATT RMSE: {test_metrics['att_rmse']:.4f}, Val Loss: {best_val_loss:.4f}")

    # 5. Save Final Results Table
    results_df = pd.DataFrame(results_records)
    results_csv = os.path.join(study_dir, "training_size_results.csv")
    results_df.to_csv(results_csv, index=False)
    print("\n" + "="*85)
    print("TRAINING-SIZE STUDY SUMMARY TABLE")
    print("="*85)
    print(results_df[["training_size", "seed", "best_epoch", "best_val_loss", "generalization_gap", "cbf_rmse", "cbf_ccc", "att_rmse", "att_ccc"]].to_string())

    # 6. Generate Summary Scaling Curves Figure (training_size_comparison.png)
    summary_stats = results_df.groupby("training_size").agg({
        "cbf_rmse": ["mean", "std"],
        "cbf_mae": ["mean", "std"],
        "cbf_ccc": ["mean", "std"],
        "att_rmse": ["mean", "std"],
        "att_mae": ["mean", "std"],
        "att_ccc": ["mean", "std"],
        "generalization_gap": ["mean", "std"]
    }).reset_index()

    sizes = summary_stats["training_size"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: CBF RMSE vs Training Size
    cbf_rmse_m = summary_stats[("cbf_rmse", "mean")].values
    cbf_rmse_s = summary_stats[("cbf_rmse", "std")].values
    axes[0, 0].errorbar(sizes, cbf_rmse_m, yerr=cbf_rmse_s, fmt="o-", color="#D32F2F", linewidth=2.5, markersize=8, capsize=5, label="CBF RMSE (Mean ± SD)")
    axes[0, 0].set_xlabel("Training Set Size (Phantoms)")
    axes[0, 0].set_ylabel("CBF RMSE (mL/100 g/min)")
    axes[0, 0].set_title("CBF RMSE vs Training Sample Size")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Panel B: ATT RMSE vs Training Size
    att_rmse_m = summary_stats[("att_rmse", "mean")].values
    att_rmse_s = summary_stats[("att_rmse", "std")].values
    axes[0, 1].errorbar(sizes, att_rmse_m, yerr=att_rmse_s, fmt="s-", color="#1976D2", linewidth=2.5, markersize=8, capsize=5, label="ATT RMSE (Mean ± SD)")
    axes[0, 1].set_xlabel("Training Set Size (Phantoms)")
    axes[0, 1].set_ylabel("ATT RMSE (seconds)")
    axes[0, 1].set_title("ATT RMSE vs Training Sample Size")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Panel C: Lin's Concordance Correlation Coefficient (CCC)
    cbf_ccc_m = summary_stats[("cbf_ccc", "mean")].values
    cbf_ccc_s = summary_stats[("cbf_ccc", "std")].values
    att_ccc_m = summary_stats[("att_ccc", "mean")].values
    att_ccc_s = summary_stats[("att_ccc", "std")].values
    axes[1, 0].errorbar(sizes, cbf_ccc_m, yerr=cbf_ccc_s, fmt="o-", color="#388E3C", linewidth=2.5, markersize=8, capsize=5, label="CBF CCC")
    axes[1, 0].errorbar(sizes, att_ccc_m, yerr=att_ccc_s, fmt="^-", color="#7B1FA2", linewidth=2.5, markersize=8, capsize=5, label="ATT CCC")
    axes[1, 0].set_xlabel("Training Set Size (Phantoms)")
    axes[1, 0].set_ylabel("Lin's CCC (dimensionless)")
    axes[1, 0].set_ylim(0.97, 1.002)
    axes[1, 0].set_title("Lin's CCC vs Training Sample Size")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Panel D: Generalization Gap (Val Loss - Train Loss)
    gap_m = summary_stats[("generalization_gap", "mean")].values
    gap_s = summary_stats[("generalization_gap", "std")].values
    axes[1, 1].errorbar(sizes, gap_m, yerr=gap_s, fmt="d-", color="#E65100", linewidth=2.5, markersize=8, capsize=5, label="Gen Gap (Val - Train Loss)")
    axes[1, 1].axhline(0, color="gray", linestyle=":")
    axes[1, 1].set_xlabel("Training Set Size (Phantoms)")
    axes[1, 1].set_ylabel("Generalization Gap (Masked L1)")
    axes[1, 1].set_title("Generalization Gap vs Training Sample Size")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    comparison_png = os.path.join(study_dir, "training_size_comparison.png")
    plt.savefig(comparison_png, dpi=300)
    plt.close()
    print(f"Saved Publication Figure: {comparison_png}")

    # 7. Generate Reproducibility Manifest
    rep_manifest = {
        "study_name": "Training-Set Size Generalization Study",
        "date_executed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sizes_evaluated": [int(s) for s in args.sizes],
        "seeds_evaluated": [int(s) for s in args.seeds],
        "datasets": {
            "val_cohort": {"path": val_path, "sha256": sha256_file(val_path)},
            "test_cohort": {"path": test_path, "sha256": sha256_file(test_path)},
            "data_manifest": {"path": manifest_csv, "sha256": sha256_file(manifest_csv)}
        },
        "output_files": {
            "results_csv": results_csv,
            "comparison_plot": comparison_png,
            "learning_curves_dir": curve_dir,
            "training_histories_dir": hist_dir
        },
        "system": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
    }
    rep_json = os.path.join(study_dir, "reproducibility_manifest.json")
    with open(rep_json, "w") as f:
        json.dump(rep_manifest, f, indent=2)
    print(f"Saved Reproducibility Manifest: {rep_json}")

if __name__ == "__main__":
    main()

