import os
import sys
import json
import time
import argparse
import shutil
import numpy as np
import pandas as pd
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.dataset_cache import get_or_create_dataset
from experiments_unet_4pld.dataset import (
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.train import train_unet_single_seed
from experiments_unet_4pld.evaluate import run_full_evaluation, evaluate_unet, calc_metrics
from experiments_unet_4pld.package_model import package_final_four_pld_model

def main():
    parser = argparse.ArgumentParser(description="Run Storage-Safe 4-PLD Spatial U-Net ASL Experiment")
    parser.add_argument("--config", type=str, default="experiments_unet_4pld/config.json", help="Path to config.json")
    parser.add_argument("--workspace", type=str, default=None, help="Custom workspace directory (e.g., D:\\ASL_WORKSPACE)")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick smoke test with minimal phantoms and epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of seeds to run")
    args = parser.parse_args()

    # 1. Initialize Centralized Workspace Manager (Protects C: Drive)
    ws = get_workspace(custom_path=args.workspace, min_free_gb=5.0)
    print("\n================ STORAGE & WORKSPACE CONFIGURATION ================")
    print(f"Managed Workspace Root: {ws.root}")
    for name, p in ws.subdirs.items():
        print(f"  [{name.upper()}] -> {p}")

    # Pre-flight disk space safety check
    ws.check_preflight_storage(required_gb=2.0)

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    seeds = args.seeds if args.seeds is not None else config["training"]["seeds"]

    if args.smoke_test:
        print("\n*** RUNNING SMOKE TEST MODE ***")
        n_tr = 20
        n_val = 6
        n_te = 6
        config["training"]["max_epochs"] = 3
        config["training"]["patience"] = 3
        seeds = [42]
    else:
        n_tr = config["data"]["n_train_phantoms"]
        n_val = config["data"]["n_val_phantoms"]
        n_te = config["data"]["n_test_phantoms"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCompute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Dataset Retrieval & Verification (Zero Redundant Generation)
    print("\n================ DATASET VALIDATION & LOADING ================")
    X_tr, Y_tr, masks_tr, snrs_tr = get_or_create_dataset(
        split_name="train", n_phantoms=n_tr, seed=config["data"]["train_seed"],
        config=config, ws=ws
    )
    X_val, Y_val, masks_val, snrs_val = get_or_create_dataset(
        split_name="val", n_phantoms=n_val, seed=config["data"]["val_seed"],
        config=config, ws=ws
    )
    X_te, Y_te, masks_te, snrs_te = get_or_create_dataset(
        split_name="test", n_phantoms=n_te, seed=config["data"]["test_seed"],
        config=config, ws=ws, fixed_snrs=config["data"]["test_snr_targets"]
    )

    # 3. Normalization parameters computed STRICTLY from training set
    print("\nComputing normalization statistics from training set...")
    stats = compute_normalization_stats(X_tr, Y_tr, masks_tr)
    stats_json_path = ws.get_path("data", "normalization_stats.json")
    stats_serializable = {
        "X_mean": stats["X_mean"].tolist(),
        "X_std": stats["X_std"].tolist(),
        "cbf_mean": stats["cbf_mean"],
        "cbf_std": stats["cbf_std"],
        "att_mean": stats["att_mean"],
        "att_std": stats["att_std"],
    }
    with open(stats_json_path, "w") as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"Training Target Stats: CBF Mean={stats['cbf_mean']:.2f}, Std={stats['cbf_std']:.2f} | "
          f"ATT Mean={stats['att_mean']:.3f}, Std={stats['att_std']:.3f}")

    X_tr_n, Y_tr_n = normalize_dataset(X_tr, Y_tr, masks_tr, stats)
    X_val_n, Y_val_n = normalize_dataset(X_val, Y_val, masks_val, stats)
    X_te_n, Y_te_n = normalize_dataset(X_te, Y_te, masks_te, stats)

    # 4. Multi-Seed Training with Strict Validation-Only Model Selection
    seed_records = []
    lowest_val_loss = float("inf")
    best_val_seed = seeds[0]
    best_ckpt_for_benchmark = None
    best_history_csv = None

    valid_mask_te = masks_te[:, 0] > 0.5
    gt_cbf_te = Y_te[:, 0][valid_mask_te]
    gt_att_te = Y_te[:, 1][valid_mask_te]

    for seed in seeds:
        print(f"\n================ TRAINING SEED {seed} ================")
        ckpt_path, best_val_loss, hist_df, duration = train_unet_single_seed(
            train_data=(X_tr_n, Y_tr_n, masks_tr, snrs_tr),
            val_data=(X_val_n, Y_val_n, masks_val, snrs_val),
            stats=stats,
            config=config,
            seed=seed,
            device=device,
            save_dir=ws.root
        )

        # STRICT VALIDATION-ONLY SELECTION: Checkpoint chosen by validation loss, NEVER test performance
        if best_val_loss < lowest_val_loss:
            lowest_val_loss = best_val_loss
            best_val_seed = seed
            best_ckpt_for_benchmark = ckpt_path
            best_history_csv = ws.get_path("logs", f"history_seed_{seed}.csv")

        # Evaluate on test set for reporting purposes
        ckpt = torch.load(ckpt_path, weights_only=False)
        eval_model = ASLUNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            features=config["model"]["features"]
        ).to(device)
        eval_model.load_state_dict(ckpt["model_state_dict"])
        eval_model.eval()

        cbf_pred, att_pred, t_inf = evaluate_unet(eval_model, X_te_n, stats, device)
        met_c = calc_metrics(gt_cbf_te, cbf_pred[valid_mask_te])
        met_a = calc_metrics(gt_att_te, att_pred[valid_mask_te])

        rec = {
            "Seed": seed,
            "Best_Epoch": ckpt["epoch"],
            "Best_Val_Loss": round(best_val_loss, 4),
            "Train_Time_s": round(duration, 1),
            "CBF_RMSE": met_c["RMSE"],
            "CBF_MAE": met_c["MAE"],
            "CBF_Bias": met_c["Bias"],
            "CBF_R2": met_c["R2"],
            "CBF_CCC": met_c["CCC"],
            "ATT_RMSE": met_a["RMSE"],
            "ATT_MAE": met_a["MAE"],
            "ATT_Bias": met_a["Bias"],
            "ATT_R2": met_a["R2"],
            "ATT_CCC": met_a["CCC"]
        }
        seed_records.append(rec)
        print(f"Seed {seed} Test Results: Val Loss = {best_val_loss:.4f} | CBF RMSE = {met_c['RMSE']:.4f}, ATT RMSE = {met_a['RMSE']:.4f}")

    seed_df = pd.DataFrame(seed_records)
    seed_csv_ws = ws.get_path("results", "per_seed_metrics.csv")
    seed_df.to_csv(seed_csv_ws, index=False)

    # Mirror metadata CSV to repo results for tracking
    repo_results_dir = os.path.join(parent_dir, "experiments_unet_4pld", "results")
    os.makedirs(repo_results_dir, exist_ok=True)
    shutil.copy2(seed_csv_ws, os.path.join(repo_results_dir, "per_seed_metrics.csv"))

    print("\n--- Per-Seed Verification Summary (Selected by Validation Loss) ---")
    print(seed_df.to_string())
    print(f"\n[Validation Policy] Selected Model: Seed {best_val_seed} with Lowest Val Loss = {lowest_val_loss:.4f}")

    # 5. Full Comparative Benchmark Using the Validation-Selected Model
    print(f"\n--- Running Full Comparative Benchmark (Model Selected by Val Loss: Seed {best_val_seed}) ---")
    benchmark_model = ASLUNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"]["features"]
    ).to(device)
    benchmark_model.load_state_dict(torch.load(best_ckpt_for_benchmark, weights_only=False)["model_state_dict"])
    benchmark_model.eval()

    # Flatten training voxels for point-wise DNN baseline
    v_tr = masks_tr[:, 0] > 0.5
    X_tr_vox = X_tr.transpose(0, 2, 3, 1)[v_tr]
    Y_tr_vox = Y_tr.transpose(0, 2, 3, 1)[v_tr]
    if len(X_tr_vox) > 200_000:
        sub_idx = np.random.choice(len(X_tr_vox), size=200_000, replace=False)
        X_tr_vox = X_tr_vox[sub_idx]
        Y_tr_vox = Y_tr_vox[sub_idx]

    run_full_evaluation(
        unet_model=benchmark_model,
        X_test=X_te,
        Y_test=Y_te,
        masks_test=masks_te,
        snrs_test=snrs_te,
        stats=stats,
        device=device,
        X_tr_vox=X_tr_vox,
        Y_tr_vox=Y_tr_vox,
        out_dir=ws.get_path("results")
    )

    # Mirror result tables and plots to repo results directory
    for item in ["matched_benchmark_25_phantoms.csv", "master_comparison.csv",
                 "paired_statistical_tests.csv", "per_phantom_metrics.csv",
                 "snr_stratified_metrics.csv", "spatial_metrics.json"]:
        src = ws.get_path("results", item)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(repo_results_dir, item))

    # Mirror plots
    repo_plots_dir = os.path.join(repo_results_dir, "plots")
    os.makedirs(repo_plots_dir, exist_ok=True)
    for plot_file in os.listdir(ws.get_path("plots")):
        src = os.path.join(ws.get_path("plots"), plot_file)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(repo_plots_dir, plot_file))

    # 6. Deliverable Packaging: Self-Contained Final 4-PLD Artifact
    print("\n================ PACKAGING FINAL FOUR-PLD MODEL ================")
    final_test_row = seed_df[seed_df["Seed"] == best_val_seed].iloc[0].to_dict()
    final_model_dir = ws.get_path("final_model")
    package_final_four_pld_model(
        selected_ckpt_path=best_ckpt_for_benchmark,
        config=config,
        stats=stats,
        history_csv_path=best_history_csv,
        test_metrics=final_test_row,
        output_dir=final_model_dir
    )

    # 7. Self-Verify Inference Script in a Clean Subprocess
    print("\nVerifying inference script in clean execution...")
    inf_script = os.path.join(final_model_dir, "inference.py")
    test_ret = os.system(f'python "{inf_script}"')
    if test_ret != 0:
        raise RuntimeError("Inference self-verification failed!")

    # 8. Report Final Storage Consumption
    print("\n================ STORAGE CONSUMPTION SUMMARY ================")
    summary = ws.get_storage_summary()
    print(f"Total Workspace Size: {summary['total_mb']:.2f} MB")
    for k, v in summary.items():
        if k not in ["total_mb", "workspace_free_gb"]:
            print(f"  - {k}: {v:.2f} MB")
    print(f"Remaining Free Space on Workspace Drive: {summary['workspace_free_gb']:.2f} GB")

    print("\n================ 4-PLD BENCHMARK COMPLETED SUCCESSFULLY ================")

if __name__ == "__main__":
    main()
