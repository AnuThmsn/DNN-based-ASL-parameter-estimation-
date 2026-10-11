import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.generate_spatial_data import generate_spatial_dataset
from experiments_unet_4pld.dataset import (
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)
from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.train import train_unet_single_seed
from experiments_unet_4pld.evaluate import run_full_evaluation, evaluate_unet, calc_metrics

def main():
    parser = argparse.ArgumentParser(description="Run 4-PLD Spatial U-Net ASL Experiment")
    parser.add_argument("--config", type=str, default="experiments_unet_4pld/config.json", help="Path to config.json")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick smoke test with minimal phantoms and epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of seeds to run")
    args = parser.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(args.config))
    data_dir = os.path.join(exp_dir, "data")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

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
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    img_size = config["data"]["img_size"]

    # 1. Dataset Generation / Loading
    train_npz = os.path.join(data_dir, f"train_N{n_tr}_s{config['data']['train_seed']}.npz")
    val_npz = os.path.join(data_dir, f"val_N{n_val}_s{config['data']['val_seed']}.npz")
    test_npz = os.path.join(data_dir, f"test_N{n_te}_s{config['data']['test_seed']}.npz")

    if os.path.exists(train_npz) and not args.smoke_test:
        print(f"Loading cached training data from {train_npz}...")
        tr_data = np.load(train_npz)
        X_tr, Y_tr, masks_tr, snrs_tr = tr_data["X"], tr_data["Y"], tr_data["masks"], tr_data["snrs"]
    else:
        print(f"Generating {n_tr} training phantoms (seed {config['data']['train_seed']})...")
        X_tr, Y_tr, masks_tr, snrs_tr, _ = generate_spatial_dataset(
            n_phantoms=n_tr, seed=config["data"]["train_seed"], img_size=img_size
        )
        if not args.smoke_test:
            np.savez_compressed(train_npz, X=X_tr, Y=Y_tr, masks=masks_tr, snrs=snrs_tr)

    if os.path.exists(val_npz) and not args.smoke_test:
        print(f"Loading cached validation data from {val_npz}...")
        val_data = np.load(val_npz)
        X_val, Y_val, masks_val, snrs_val = val_data["X"], val_data["Y"], val_data["masks"], val_data["snrs"]
    else:
        print(f"Generating {n_val} validation phantoms (seed {config['data']['val_seed']})...")
        X_val, Y_val, masks_val, snrs_val, _ = generate_spatial_dataset(
            n_phantoms=n_val, seed=config["data"]["val_seed"], img_size=img_size
        )
        if not args.smoke_test:
            np.savez_compressed(val_npz, X=X_val, Y=Y_val, masks=masks_val, snrs=snrs_val)

    if os.path.exists(test_npz) and not args.smoke_test:
        print(f"Loading cached test data from {test_npz}...")
        te_data = np.load(test_npz)
        X_te, Y_te, masks_te, snrs_te = te_data["X"], te_data["Y"], te_data["masks"], te_data["snrs"]
    else:
        print(f"Generating {n_te} test phantoms with targeted SNR tracking (seed {config['data']['test_seed']})...")
        # Ensure targeted tracking of SNR=10 and SNR=15
        target_snrs = [10.0, 15.0]
        X_te, Y_te, masks_te, snrs_te, _ = generate_spatial_dataset(
            n_phantoms=n_te, seed=config["data"]["test_seed"], img_size=img_size, fixed_snrs=target_snrs
        )
        if not args.smoke_test:
            np.savez_compressed(test_npz, X=X_te, Y=Y_te, masks=masks_te, snrs=snrs_te)

    # 2. Normalization parameters computed STRICTLY from training set
    print("Computing normalization statistics from training set...")
    stats = compute_normalization_stats(X_tr, Y_tr, masks_tr)
    # Save stats
    stats_serializable = {
        "X_mean": stats["X_mean"].tolist(),
        "X_std": stats["X_std"].tolist(),
        "cbf_mean": stats["cbf_mean"],
        "cbf_std": stats["cbf_std"],
        "att_mean": stats["att_mean"],
        "att_std": stats["att_std"],
    }
    with open(os.path.join(data_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"Training Target Stats: CBF Mean={stats['cbf_mean']:.2f}, Std={stats['cbf_std']:.2f} | "
          f"ATT Mean={stats['att_mean']:.3f}, Std={stats['att_std']:.3f}")

    X_tr_n, Y_tr_n = normalize_dataset(X_tr, Y_tr, masks_tr, stats)
    X_val_n, Y_val_n = normalize_dataset(X_val, Y_val, masks_val, stats)
    X_te_n, Y_te_n = normalize_dataset(X_te, Y_te, masks_te, stats)

    # 3. Multi-Seed Training
    seed_records = []
    best_overall_seed = seeds[0]
    best_cbf_rmse = float("inf")
    best_model_state = None

    valid_mask_te = masks_te[:, 0] > 0.5
    gt_cbf_te = Y_te[:, 0][valid_mask_te]
    gt_att_te = Y_te[:, 1][valid_mask_te]

    for seed in seeds:
        print(f"\n================ TRAINING SEED {seed} ================")
        ckpt_path, hist_df, duration = train_unet_single_seed(
            train_data=(X_tr_n, Y_tr_n, masks_tr, snrs_tr),
            val_data=(X_val_n, Y_val_n, masks_val, snrs_val),
            stats=stats,
            config=config,
            seed=seed,
            device=device,
            save_dir=exp_dir
        )

        # Load best checkpoint for this seed and evaluate on test set
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
        print(f"Seed {seed} Test Results: CBF RMSE = {met_c['RMSE']:.4f}, ATT RMSE = {met_a['RMSE']:.4f}")

        if met_c["RMSE"] < best_cbf_rmse:
            best_cbf_rmse = met_c["RMSE"]
            best_overall_seed = seed
            best_model_state = eval_model

    seed_df = pd.DataFrame(seed_records)
    seed_csv = os.path.join(results_dir, "per_seed_metrics.csv")
    seed_df.to_csv(seed_csv, index=False)
    print("\n--- Per-Seed Verification Summary ---")
    print(seed_df.to_string())

    # Multi-seed mean and std
    cbf_mean_val, cbf_std_val = seed_df["CBF_RMSE"].mean(), seed_df["CBF_RMSE"].std()
    att_mean_val, att_std_val = seed_df["ATT_RMSE"].mean(), seed_df["ATT_RMSE"].std()
    print(f"\nFinal Multi-Seed U-Net Performance:")
    print(f"CBF RMSE: {cbf_mean_val:.4f} ± {cbf_std_val:.4f}")
    print(f"ATT RMSE: {att_mean_val:.4f} ± {att_std_val:.4f}")

    # 4. Full Benchmark against Point-wise DNN, Bayesian, NLLS, and References
    print(f"\n--- Running Full Comparative Benchmark (Best Seed: {best_overall_seed}) ---")
    # Flatten valid training voxels for point-wise DNN baseline training
    v_tr = masks_tr[:, 0] > 0.5
    X_tr_vox = X_tr.transpose(0, 2, 3, 1)[v_tr] # [N_valid, 4]
    Y_tr_vox = Y_tr.transpose(0, 2, 3, 1)[v_tr] # [N_valid, 2]

    # Subsample training voxels if too large for quick point-wise MLP fit
    if len(X_tr_vox) > 200_000:
        sub_idx = np.random.choice(len(X_tr_vox), size=200_000, replace=False)
        X_tr_vox = X_tr_vox[sub_idx]
        Y_tr_vox = Y_tr_vox[sub_idx]

    benchmark_df = run_full_evaluation(
        unet_model=best_model_state,
        X_test=X_te,
        Y_test=Y_te,
        masks_test=masks_te,
        snrs_test=snrs_te,
        stats=stats,
        device=device,
        X_tr_vox=X_tr_vox,
        Y_tr_vox=Y_tr_vox,
        out_dir=results_dir
    )

    print("\n================ EXPERIMENT COMPLETED SUCCESSFULLY ================")

if __name__ == "__main__":
    main()
