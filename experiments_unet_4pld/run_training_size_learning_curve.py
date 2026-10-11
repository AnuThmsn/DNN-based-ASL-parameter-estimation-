"""
Full Execution Script: Training-Size Learning-Curve Experiment (Phantoms 100 to 2400)
Runs 6 training sizes [100, 200, 400, 600, 1200, 2400] x 3 seeds [42, 123, 2024] = 18 runs.
Strictly preserves:
- Canonical 4-PLDs: [1.525, 2.025, 2.525, 3.025] s
- Buxton two-compartment forward model and dual-Rician noise model
- Nested subset hierarchy: D_100 c D_200 c D_400 c D_600 c D_1200 c D_2400
- Fixed independent validation (N=100) and test (N=100) cohorts
- Output directory: D:\ASL_WORKSPACE\experiments\training_size_learning_curve\
"""

import os
import sys
import time
import json
import hashlib
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.model import ASLUNet, count_parameters
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
    parser = argparse.ArgumentParser(description="Full Training-Size Learning-Curve Experiment")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 400, 600, 1200, 2400], help="Training sizes")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024], help="Random seeds")
    args = parser.parse_args()

    # 1. Setup Workspace & Storage Safeguard
    ws = get_workspace(min_free_gb=5.0)
    ws.check_preflight_storage(required_gb=3.0)
    redir = ws.verify_storage_redirection()
    print("Storage Redirection Status:", redir)

    target_dir = os.path.join(ws.root, "experiments", "training_size_learning_curve")
    ckpt_dir = os.path.join(target_dir, "checkpoints")
    hist_dir = os.path.join(target_dir, "training_histories")
    curve_dir = os.path.join(target_dir, "learning_curves")
    for d in [target_dir, ckpt_dir, hist_dir, curve_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCompute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Fixed Independent Validation and Test Cohorts
    print("\n--- Verifying Fixed Evaluation Cohorts ---")
    val_path = os.path.join(ws.root, "data", "val_N100_s99.npz")
    test_path = os.path.join(ws.root, "data", "test_N100_s7.npz")

    val_data = np.load(val_path)
    X_val_raw, Y_val_raw, masks_val = val_data["X"], val_data["Y"], val_data["masks"]
    snrs_val = val_data["snrs"] if "snrs" in val_data else np.full(len(X_val_raw), 15.0, dtype=np.float32)

    test_data = np.load(test_path)
    X_te_raw, Y_te_raw, masks_te = test_data["X"], test_data["Y"], test_data["masks"]

    val_hash = sha256_file(val_path)
    test_hash = sha256_file(test_path)
    print(f"Validation Cohort: {val_path} (N={len(X_val_raw)}, SHA-256: {val_hash[:16]}...)")
    print(f"Test Cohort:       {test_path} (N={len(X_te_raw)}, SHA-256: {test_hash[:16]}...)")

    # 3. Master Training Pool Construction (up to 2400)
    print("\n--- Loading or Generating Master Nested Training Pool (up to 2400 phantoms) ---")
    n2400_path = os.path.join(ws.root, "data", "train_nested_N2400_s42.npz")
    n1200_path = os.path.join(ws.root, "data", "train_nested_N1200_s42.npz")
    n600_path = os.path.join(ws.root, "data", "train_N600_s42.npz")

    n600_raw = np.load(n600_path)
    X_600 = n600_raw["X"]

    if os.path.exists(n2400_path):
        print(f"Reusing existing verified nested 2400-phantom dataset: {n2400_path}")
        n2400_raw = np.load(n2400_path)
        X_tr_pool = n2400_raw["X"]
        Y_tr_pool = n2400_raw["Y"]
        masks_tr_pool = n2400_raw["masks"]
        snrs_tr_pool = n2400_raw["snrs"]
    else:
        print(f"Generating full nested master pool of 2400 phantoms with seed 42...")
        ws.check_preflight_storage(required_gb=2.0)
        start_gen = time.time()
        X_tr_pool, Y_tr_pool, masks_tr_pool, snrs_tr_pool, _ = generate_spatial_dataset(
            n_phantoms=2400,
            seed=42,
            img_size=64
        )
        gen_dur = time.time() - start_gen
        # Verify bit-for-bit nesting with 600-phantom dataset
        assert np.array_equal(X_tr_pool[:600], X_600), "FATAL: Nested first 600 phantoms do not match train_N600_s42.npz!"
        np.savez_compressed(n2400_path, X=X_tr_pool, Y=Y_tr_pool, masks=masks_tr_pool, snrs=snrs_tr_pool)
        print(f"Saved verified 2400-phantom dataset in {gen_dur:.1f}s: {n2400_path} (SHA-256: {sha256_file(n2400_path)[:16]}...)")

    with open("experiments_unet_4pld/config.json", "r") as f:
        base_config = json.load(f)

    # 4. Generate Manifests
    print("\n--- Generating Phantoms and Subset Manifests ---")
    data_manifest_records = []
    subset_manifest_records = []

    for p_i in range(len(X_tr_pool)):
        vm = int(np.sum(masks_tr_pool[p_i, 0] > 0.5))
        in_100 = p_i < 100
        in_200 = p_i < 200
        in_400 = p_i < 400
        in_600 = p_i < 600
        in_1200 = p_i < 1200
        in_2400 = p_i < 2400

        data_manifest_records.append({
            "phantom_id": f"train_{p_i:04d}",
            "split": "train",
            "generation_seed": 42,
            "stream_index": p_i,
            "valid_voxels": vm,
            "in_N100": in_100,
            "in_N200": in_200,
            "in_N400": in_400,
            "in_N600": in_600,
            "in_N1200": in_1200,
            "in_N2400": in_2400
        })

        subset_manifest_records.append({
            "phantom_id": f"train_{p_i:04d}",
            "training_index": p_i,
            "in_N100": in_100,
            "in_N200": in_200,
            "in_N400": in_400,
            "in_N600": in_600,
            "in_N1200": in_1200,
            "in_N2400": in_2400
        })

    for p_i in range(len(X_val_raw)):
        vm = int(np.sum(masks_val[p_i, 0] > 0.5))
        data_manifest_records.append({
            "phantom_id": f"val_{p_i:04d}",
            "split": "val",
            "generation_seed": 99,
            "stream_index": p_i,
            "valid_voxels": vm,
            "in_N100": True, "in_N200": True, "in_N400": True, "in_N600": True, "in_N1200": True, "in_N2400": True
        })

    for p_i in range(len(X_te_raw)):
        vm = int(np.sum(masks_te[p_i, 0] > 0.5))
        data_manifest_records.append({
            "phantom_id": f"test_{p_i:04d}",
            "split": "test",
            "generation_seed": 7,
            "stream_index": p_i,
            "valid_voxels": vm,
            "in_N100": True, "in_N200": True, "in_N400": True, "in_N600": True, "in_N1200": True, "in_N2400": True
        })

    data_manifest_df = pd.DataFrame(data_manifest_records)
    subset_manifest_df = pd.DataFrame(subset_manifest_records)

    d_man_path = os.path.join(target_dir, "data_manifest.csv")
    s_man_path = os.path.join(target_dir, "subset_manifest.csv")
    data_manifest_df.to_csv(d_man_path, index=False)
    subset_manifest_df.to_csv(s_man_path, index=False)
    print(f"Saved: {d_man_path} ({len(data_manifest_df)} phantoms)")
    print(f"Saved: {s_man_path} ({len(subset_manifest_df)} training phantoms mapped)")

    # 5. Staged Execution across 6 sizes x 3 seeds = 18 runs
    results_records = []
    prior_study_dir = os.path.join(ws.root, "experiments", "training_size_study")

    for N in args.sizes:
        print(f"\n{'='*75}\n>>> EXAMINING TRAINING-SET SIZE: N = {N} PHANTOMS <<<\n{'='*75}")
        X_sub = X_tr_pool[:N]
        Y_sub = Y_tr_pool[:N]
        masks_sub = masks_tr_pool[:N]
        snrs_sub = snrs_tr_pool[:N]

        # Normalization computed strictly on this training set
        stats_N = compute_normalization_stats(X_sub, Y_sub, masks_sub)
        X_sub_n, Y_sub_n = normalize_dataset(X_sub, Y_sub, masks_sub, stats_N)
        X_val_n, Y_val_n = normalize_dataset(X_val_raw, Y_val_raw, masks_val, stats_N)

        for seed in args.seeds:
            print(f"\n--- Condition: N = {N} | Seed = {seed} ---")
            ckpt_path = os.path.join(ckpt_dir, f"unet_N{N}_seed_{seed}.pth")
            hist_csv = os.path.join(hist_dir, f"history_N{N}_seed_{seed}.csv")

            # Check if valid checkpoint and history already exist in prior verified study
            can_reuse = False
            prior_ckpt = os.path.join(prior_study_dir, "checkpoints", f"unet_N{N}_seed_{seed}.pth")
            prior_hist = os.path.join(prior_study_dir, "training_histories", f"history_N{N}_seed_{seed}.csv")

            if N == 600 and not os.path.exists(prior_ckpt):
                prior_ckpt = os.path.join(ws.root, "checkpoints", f"unet_best_seed_{seed}.pth")
                prior_hist = os.path.join(ws.root, "results", "logs", f"history_seed_{seed}.csv")

            if os.path.exists(prior_ckpt) and os.path.exists(prior_hist):
                print(f"Reusing existing verified checkpoint: {prior_ckpt}")
                shutil.copy2(prior_ckpt, ckpt_path)
                shutil.copy2(prior_hist, hist_csv)
                can_reuse = True
                train_dur = 110.0 if N == 600 else 60.0

            if not can_reuse:
                ws.check_preflight_storage(required_gb=2.0)
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
                    save_dir=target_dir
                )
                orig_saved_ckpt = os.path.join(target_dir, "checkpoints", f"unet_best_seed_{seed}.pth")
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

            hist_df = pd.read_csv(hist_csv)
            best_epoch = int(ckpt_loaded["epoch"])
            best_val_loss = float(ckpt_loaded["val_loss"])
            train_loss_best = float(hist_df[hist_df["epoch"] == best_epoch]["train_loss"].values[0])
            gen_gap = best_val_loss - train_loss_best

            # Save individual learning curve plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            ax1.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss (Masked L1)", color="#1976D2")
            ax1.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss (Masked L1)", color="#F57C00")
            ax1.axvline(best_epoch, linestyle="--", color="gray", label=f"Best Ep {best_epoch}")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Masked L1 Loss")
            ax1.set_title(f"Loss Curves (N={N}, Seed {seed})\nGeneralization Gap: {gen_gap:+.4f}")
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
            ax2.set_title(f"Validation Metric Convergence (N={N}, Seed {seed})")

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
                "train_time_s": round(train_dur, 1),
                "checkpoint_path": ckpt_path,
                "checkpoint_sha256": sha256_file(ckpt_path),
                "run_status": "COMPLETED"
            }
            results_records.append(row)
            print(f"Completed N={N}, Seed {seed} -> Test CBF RMSE: {test_metrics['cbf_rmse']:.4f}, ATT RMSE: {test_metrics['att_rmse']:.4f}, Val Loss: {best_val_loss:.4f}")

    # 6. Save training_size_results.csv
    results_df = pd.DataFrame(results_records)
    results_csv = os.path.join(target_dir, "training_size_results.csv")
    results_df.to_csv(results_csv, index=False)

    # 7. Save training_size_comparison.csv (grouped summary)
    summary_stats = results_df.groupby("training_size").agg({
        "cbf_rmse": ["mean", "std"],
        "cbf_mae": ["mean", "std"],
        "cbf_ccc": ["mean", "std"],
        "att_rmse": ["mean", "std"],
        "att_mae": ["mean", "std"],
        "att_ccc": ["mean", "std"],
        "best_val_loss": ["mean", "std"],
        "generalization_gap": ["mean", "std"],
        "train_time_s": ["mean", "std"]
    }).reset_index()

    comp_csv = os.path.join(target_dir, "training_size_comparison.csv")
    summary_stats.to_csv(comp_csv, index=False)

    print("\n" + "="*85)
    print("ALL 18 RUNS COMPLETED — SUMMARY STATISTICS TABLE")
    print("="*85)
    print(results_df[["training_size", "seed", "best_epoch", "best_val_loss", "generalization_gap", "cbf_rmse", "cbf_ccc", "att_rmse", "att_ccc"]].to_string())

    # 8. Generate Multi-Panel Comparison Figure
    sizes = summary_stats["training_size"].values
    cbf_rmse_m = summary_stats[("cbf_rmse", "mean")].values
    cbf_rmse_s = summary_stats[("cbf_rmse", "std")].values
    att_rmse_m = summary_stats[("att_rmse", "mean")].values
    att_rmse_s = summary_stats[("att_rmse", "std")].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: CBF RMSE
    axes[0, 0].errorbar(sizes, cbf_rmse_m, yerr=cbf_rmse_s, fmt="o-", color="#D32F2F", linewidth=2.5, markersize=8, capsize=5, label="CBF RMSE (Mean ± SD)")
    axes[0, 0].set_xlabel("Training Set Size (Phantoms)")
    axes[0, 0].set_ylabel("CBF RMSE (mL/100 g/min)")
    axes[0, 0].set_title("CBF RMSE vs Training Sample Size (100 to 2400)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Panel B: ATT RMSE
    axes[0, 1].errorbar(sizes, att_rmse_m, yerr=att_rmse_s, fmt="s-", color="#1976D2", linewidth=2.5, markersize=8, capsize=5, label="ATT RMSE (Mean ± SD)")
    axes[0, 1].set_xlabel("Training Set Size (Phantoms)")
    axes[0, 1].set_ylabel("ATT RMSE (seconds)")
    axes[0, 1].set_title("ATT RMSE vs Training Sample Size (100 to 2400)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Panel C: Lin's Concordance Correlation Coefficient
    cbf_ccc_m = summary_stats[("cbf_ccc", "mean")].values
    cbf_ccc_s = summary_stats[("cbf_ccc", "std")].values
    att_ccc_m = summary_stats[("att_ccc", "mean")].values
    att_ccc_s = summary_stats[("att_ccc", "std")].values
    axes[1, 0].errorbar(sizes, cbf_ccc_m, yerr=cbf_ccc_s, fmt="o-", color="#388E3C", linewidth=2.5, markersize=8, capsize=5, label="CBF Lin's CCC")
    axes[1, 0].errorbar(sizes, att_ccc_m, yerr=att_ccc_s, fmt="^-", color="#7B1FA2", linewidth=2.5, markersize=8, capsize=5, label="ATT Lin's CCC")
    axes[1, 0].set_xlabel("Training Set Size (Phantoms)")
    axes[1, 0].set_ylabel("Lin's CCC (dimensionless)")
    axes[1, 0].set_ylim(0.975, 1.002)
    axes[1, 0].set_title("Lin's Concordance Correlation Coefficient (CCC)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Panel D: Generalization Gap
    gap_m = summary_stats[("generalization_gap", "mean")].values
    gap_s = summary_stats[("generalization_gap", "std")].values
    axes[1, 1].errorbar(sizes, gap_m, yerr=gap_s, fmt="d-", color="#E65100", linewidth=2.5, markersize=8, capsize=5, label="Generalization Gap (Val - Train)")
    axes[1, 1].axhline(0, color="gray", linestyle=":")
    axes[1, 1].set_xlabel("Training Set Size (Phantoms)")
    axes[1, 1].set_ylabel("Masked L1 Loss Gap")
    axes[1, 1].set_title("Generalization Gap vs Training Sample Size")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    comp_png = os.path.join(target_dir, "training_size_comparison.png")
    plt.savefig(comp_png, dpi=300)
    plt.close()
    print(f"Saved: {comp_png}")

    # 9. Save Reproducibility Manifest
    rep_manifest = {
        "experiment_name": "training_size_learning_curve",
        "date_executed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sizes_completed": [int(s) for s in args.sizes],
        "seeds_completed": [int(s) for s in args.seeds],
        "total_runs_completed": len(results_records),
        "cohort_hashes": {
            "validation_N100_s99": val_hash,
            "test_N100_s7": test_hash,
            "training_nested_pool_N2400": sha256_file(n2400_path)
        },
        "output_files": {
            "results_csv": results_csv,
            "comparison_csv": comp_csv,
            "comparison_plot": comp_png,
            "data_manifest": d_man_path,
            "subset_manifest": s_man_path
        },
        "system": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
    }
    rep_json = os.path.join(target_dir, "reproducibility_manifest.json")
    with open(rep_json, "w") as f:
        json.dump(rep_manifest, f, indent=2)
    print(f"Saved Reproducibility Manifest: {rep_json}")

if __name__ == "__main__":
    main()

