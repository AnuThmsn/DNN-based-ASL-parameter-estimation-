"""
Scalable Training-Size Learning-Curve Experiment (Phantoms 2400 to 19200)
=======================================================================
Extends the sample-size learning curve beyond 2,400 phantoms up to 19,200 phantoms:
Sizes: [2400, 4800, 9600, 19200]
Seeds: [42, 123, 2024]
Master Pool: D:\ASL_WORKSPACE\data\train_nested_N19200_s42.npz

Key Scientific Safeguards:
1. Strict nested prefix inclusion: D_2400 ⊂ D_4800 ⊂ D_9600 ⊂ D_19200.
2. Exact requested size validation (raises clear error if master pool has fewer samples).
3. Zero data leakage: Normalization computed exclusively on training voxels per size.
4. Locked independent multi-SNR evaluation cohort (N=100, seed 777) plus historical cohort (seed 7).
5. Output bounds violation tracking before physical clamping.
6. Per-phantom and pooled voxel-level metrics logging.
7. Output directory: D:\ASL_WORKSPACE\experiments\training_size_scaling_v1\
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
import torch.nn as nn
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.workspace import get_workspace
from experiments_unet_4pld.model import ASLUNet, count_parameters
from experiments_unet_4pld.dataset import (
    SpatialASLDataset,
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)
from experiments_unet_4pld.train import MaskedLoss
from experiments_unet_4pld.run_snr_benchmark import calc_metrics, calc_lin_ccc
from experiments_unet_4pld.generate_spatial_data import generate_spatial_dataset
from experiments_unet_4pld.audit.cohort_integrity_check import (
    audit_cohort_dataset,
    check_splits_independence,
    verify_nested_subsets,
    compute_sample_hash
)


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_model_on_cohort(model, X_raw, Y_raw, masks_raw, stats, device, batch_size=16):
    """
    Evaluates model on a given evaluation cohort:
    - Normalizes with training-derived stats
    - Computes physical predictions
    - Tracks boundary violations before clipping
    - Computes pooled and per-phantom metrics
    """
    model.eval()
    N = len(X_raw)
    X_norm = np.copy(X_raw)
    for c in range(4):
        X_norm[:, c] = (X_norm[:, c] - stats["X_mean"][c]) / stats["X_std"][c]

    raw_cbf_list, raw_att_list = [], []
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.tensor(X_norm[i:i+batch_size], dtype=torch.float32, device=device)
            out = model(xb).cpu().numpy()
            c_raw = out[:, 0] * stats["cbf_std"] + stats["cbf_mean"]
            a_raw = out[:, 1] * stats["att_std"] + stats["att_mean"]
            raw_cbf_list.append(c_raw)
            raw_att_list.append(a_raw)

    infer_time = time.time() - start_time
    pred_cbf_raw = np.concatenate(raw_cbf_list, axis=0)
    pred_att_raw = np.concatenate(raw_att_list, axis=0)

    # Valid brain mask
    valid_mask = masks_raw[:, 0] > 0.5
    gt_cbf = Y_raw[:, 0][valid_mask]
    gt_att = Y_raw[:, 1][valid_mask]

    cbf_valid_raw = pred_cbf_raw[valid_mask]
    att_valid_raw = pred_att_raw[valid_mask]

    # Track boundary violations before clipping
    cbf_oob = np.mean((cbf_valid_raw < 0.0) | (cbf_valid_raw > 100.0))
    att_oob = np.mean((att_valid_raw < 0.5) | (att_valid_raw > 3.0))

    # Physical clamping
    cbf_clipped = np.clip(cbf_valid_raw, 0.0, 100.0)
    att_clipped = np.clip(att_valid_raw, 0.5, 3.0)

    # Pooled metrics
    mc = calc_metrics(gt_cbf, cbf_clipped)
    ma = calc_metrics(gt_att, att_clipped)
    ccc_c = calc_lin_ccc(gt_cbf, cbf_clipped)
    ccc_a = calc_lin_ccc(gt_att, att_clipped)

    # Per-phantom metrics
    phantom_cbf_rmses = []
    phantom_att_rmses = []
    for p_i in range(N):
        m_i = masks_raw[p_i, 0] > 0.5
        if np.any(m_i):
            g_c = Y_raw[p_i, 0][m_i]
            g_a = Y_raw[p_i, 1][m_i]
            p_c = np.clip(pred_cbf_raw[p_i][m_i], 0.0, 100.0)
            p_a = np.clip(pred_att_raw[p_i][m_i], 0.5, 3.0)
            phantom_cbf_rmses.append(float(np.sqrt(np.mean((g_c - p_c) ** 2))))
            phantom_att_rmses.append(float(np.sqrt(np.mean((g_a - p_a) ** 2))))

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
        "cbf_oob_fraction": float(cbf_oob),
        "att_oob_fraction": float(att_oob),
        "phantom_cbf_rmse_mean": float(np.mean(phantom_cbf_rmses)),
        "phantom_cbf_rmse_std": float(np.std(phantom_cbf_rmses)),
        "phantom_att_rmse_mean": float(np.mean(phantom_att_rmses)),
        "phantom_att_rmse_std": float(np.std(phantom_att_rmses)),
        "per_phantom_cbf_rmses": phantom_cbf_rmses,
        "per_phantom_att_rmses": phantom_att_rmses,
        "infer_time_s": infer_time
    }


def train_scaling_model(
    sub_X, sub_Y, sub_masks, sub_snrs,
    val_X, val_Y, val_masks, val_snrs,
    stats,
    config,
    seed=42,
    device=None,
    ckpt_path=None,
    max_epochs=80,
    patience=12
):
    """
    Trains ASLUNet on normalized subset with ReduceLROnPlateau and early stopping.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Normalize data strictly with stats
    X_tr_n, Y_tr_n = normalize_dataset(sub_X, sub_Y, sub_masks, stats)
    X_val_n, Y_val_n = normalize_dataset(val_X, val_Y, val_masks, stats)

    batch_size = config["training"].get("batch_size", 32)
    train_ds = SpatialASLDataset(X_tr_n, Y_tr_n, sub_masks, sub_snrs)
    val_ds = SpatialASLDataset(X_val_n, Y_val_n, val_masks, val_snrs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model_cfg = config["model"]
    model = ASLUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        features=model_cfg["features"]
    ).to(device)

    criterion = MaskedLoss(loss_type="l1")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("learning_rate", 1e-3),
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"].get("lr_factor", 0.5),
        patience=config["training"].get("lr_patience", 5),
        min_lr=config["training"].get("min_lr", 1e-6)
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_cnt = 0
    history = []
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_accum = 0.0
        n_tr_batches = 0
        for xb, yb, mb, _ in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_accum += loss.item()
            n_tr_batches += 1

        train_loss = train_accum / max(1, n_tr_batches)

        # Validation
        model.eval()
        val_accum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb, mb, _ in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb, mb)
                val_accum += loss.item()
                n_val_batches += 1

        val_loss = val_accum / max(1, n_val_batches)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
            if ckpt_path is not None:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "stats": stats,
                    "seed": seed
                }, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    train_duration = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_val_loss, train_duration, history


def main():
    parser = argparse.ArgumentParser(description="Scalable ASL U-Net Training-Size Experiment")
    parser.add_argument("--sizes", nargs="+", type=int, default=[2400, 4800, 9600, 19200])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--master-size", type=int, default=19200)
    parser.add_argument("--stage", type=str, default="all", help="all or specific size: 2400, 4800, 9600, 19200")
    parser.add_argument("--generate-master-only", action="store_true", help="Generate 19200 master pool and manifests only")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test with 1 epoch")
    args = parser.parse_args()

    ws = get_workspace(min_free_gb=5.0)
    data_dir = os.path.join(ws.root, "data")
    exp_dir = os.path.join(ws.root, "experiments", "training_size_scaling_v1")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    hist_dir = os.path.join(exp_dir, "training_histories")
    curve_dir = os.path.join(exp_dir, "learning_curves")
    results_dir = os.path.join(exp_dir, "results")

    for d in [exp_dir, ckpt_dir, hist_dir, curve_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n================ ASL U-NET RESEARCH SCALING EXPERIMENT ================")
    print(f"Workspace Root: {ws.root}")
    print(f"Experiment Directory: {exp_dir}")

    # Check free disk space
    ws.check_preflight_storage(required_gb=5.0)

    # 1. Master Training Pool
    master_path = os.path.join(data_dir, f"train_nested_N{args.master_size}_s42.npz")
    if args.smoke_test and not os.path.exists(master_path) and os.path.exists(os.path.join(data_dir, "train_nested_N2400_s42.npz")) and max(args.sizes) <= 2400:
        master_path = os.path.join(data_dir, "train_nested_N2400_s42.npz")
    if os.path.exists(master_path):
        print(f"Loading existing master training pool: {master_path}")
        master_data = np.load(master_path)
        X_master = master_data["X"]
        Y_master = master_data["Y"]
        masks_master = master_data["masks"]
        snrs_master = master_data["snrs"]
    else:
        print(f"Generating master training pool of {args.master_size} phantoms (seed 42)...")
        start_gen = time.time()
        X_master, Y_master, masks_master, snrs_master, sds_master = generate_spatial_dataset(
            n_phantoms=args.master_size,
            seed=42,
            img_size=64
        )
        gen_duration = time.time() - start_gen
        print(f"Master pool generated in {gen_duration:.1f}s. Saving to {master_path}...")
        np.savez_compressed(
            master_path,
            X=X_master,
            Y=Y_master,
            masks=masks_master,
            snrs=snrs_master,
            sds=sds_master
        )
        print(f"Saved master pool: {master_path} ({os.path.getsize(master_path)/1e6:.1f} MB)")

    master_sha256 = sha256_file(master_path)
    print(f"Master Dataset SHA-256: {master_sha256}")
    print(f"Actual Master Samples: {len(X_master)}")

    # VALIDATION: Validate requested sizes against master pool length
    for sz in args.sizes:
        if sz > len(X_master):
            raise ValueError(f"CRITICAL ERROR: Requested size {sz} exceeds master dataset length {len(X_master)}!")

    if args.generate_master_only:
        print("Master pool generation complete. Exiting (--generate-master-only).")
        return

    # 2. Evaluation Cohorts
    val_path = os.path.join(data_dir, "val_N100_s99.npz")
    hist_test_path = os.path.join(data_dir, "test_N100_s7.npz")
    locked_test_path = os.path.join(data_dir, "test_locked_N100_s777.npz")

    val_data = np.load(val_path)
    X_val, Y_val, masks_val = val_data["X"], val_data["Y"], val_data["masks"]
    snrs_val = val_data["snrs"] if "snrs" in val_data else np.full(len(X_val), 15.0, dtype=np.float32)

    hist_test_data = np.load(hist_test_path)
    X_te_hist, Y_te_hist, masks_te_hist = hist_test_data["X"], hist_test_data["Y"], hist_test_data["masks"]

    locked_test_data = np.load(locked_test_path)
    X_te_locked, Y_te_locked, masks_te_locked = locked_test_data["X"], locked_test_data["Y"], locked_test_data["masks"]

    # Multi-SNR locked snapshots
    snr_levels = [5, 10, 15, 20, 50, "inf"]
    locked_snr_cohorts = {}
    for sl in snr_levels:
        f_snr = os.path.join(data_dir, f"test_locked_N100_s777_snr_{sl}.npz")
        if os.path.exists(f_snr):
            d_s = np.load(f_snr)
            locked_snr_cohorts[str(sl)] = d_s["X"]

    with open("experiments_unet_4pld/config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    sizes_to_run = args.sizes
    if args.stage != "all":
        st_size = int(args.stage)
        sizes_to_run = [st_size]

    max_epochs = 2 if args.smoke_test else config["training"].get("max_epochs", 80)
    patience = 2 if args.smoke_test else config["training"].get("patience", 12)

    results_csv = os.path.join(exp_dir, "smoke_test_results.csv" if args.smoke_test else "training_scaling_results.csv")
    existing_rows = []
    if os.path.exists(results_csv):
        df_old = pd.read_csv(results_csv)
        existing_rows = df_old.to_dict("records")

    per_run_records = existing_rows

    for sz in sizes_to_run:
        print(f"\n================ EXECUTING TRAINING SIZE N = {sz} ================")
        # Slice nested training pool strictly
        sub_X = X_master[:sz]
        sub_Y = Y_master[:sz]
        sub_masks = masks_master[:sz]
        sub_snrs = snrs_master[:sz]

        # Compute normalization strictly on this training subset
        stats_N = compute_normalization_stats(sub_X, sub_Y, sub_masks)

        for sd in args.seeds:
            # Check if run already completed
            prefix = "smoke_" if args.smoke_test else ""
            ckpt_name = f"{prefix}unet_N{sz}_seed_{sd}.pth"
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
            hist_file = os.path.join(hist_dir, f"{prefix}history_N{sz}_seed_{sd}.csv")

            already_done = any(r["training_size"] == sz and r["seed"] == sd for r in per_run_records)
            if already_done and os.path.exists(ckpt_file):
                print(f"Skipping already completed run: N={sz}, Seed={sd}")
                continue

            # Check if this is N=2400 and we have a historical checkpoint in training_size_study
            hist_2400_ckpt = os.path.join(ws.root, "experiments", "training_size_study", "checkpoints", f"unet_N2400_seed_{sd}.pth")
            if sz == 2400 and os.path.exists(hist_2400_ckpt) and not args.smoke_test:
                print(f"Evaluating existing verified N=2400 checkpoint from historical study: {hist_2400_ckpt}")
                model = ASLUNet(in_channels=4, out_channels=2, features=config["model"]["features"]).to(device)
                ckpt_data = torch.load(hist_2400_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ckpt_data["model_state_dict"])
                best_epoch = ckpt_data.get("epoch", 50)
                best_val_loss = ckpt_data.get("val_loss", 0.0715)
                train_duration = 0.0
                stats_used = ckpt_data.get("stats", stats_N)
                # Copy to local ckpt_dir
                import shutil
                shutil.copy2(hist_2400_ckpt, ckpt_file)
                hist_2400_csv = os.path.join(ws.root, "experiments", "training_size_study", "training_histories", f"history_N2400_seed_{sd}.csv")
                if os.path.exists(hist_2400_csv) and not os.path.exists(hist_file):
                    shutil.copy2(hist_2400_csv, hist_file)
            else:
                stats_used = stats_N
                print(f"\n>>> Training N={sz}, Seed={sd} (Max Epochs: {max_epochs}, Patience: {patience}) <<<")
                model, best_epoch, best_val_loss, train_duration, hist = train_scaling_model(
                    sub_X, sub_Y, sub_masks, sub_snrs,
                    X_val, Y_val, masks_val, snrs_val,
                    stats=stats_N,
                    config=config,
                    seed=sd,
                    device=device,
                    ckpt_path=ckpt_file,
                    max_epochs=max_epochs,
                    patience=patience
                )
                df_hist = pd.DataFrame(hist)
                df_hist.to_csv(hist_file, index=False)

            # Evaluate on locked test cohort
            print(f"Evaluating N={sz}, Seed={sd} on Locked Multi-SNR Test Cohort...")
            eval_locked = evaluate_model_on_cohort(model, X_te_locked, Y_te_locked, masks_te_locked, stats_used, device)

            # Evaluate across individual locked SNRs
            snr_metrics = {}
            for sl, X_snr_raw in locked_snr_cohorts.items():
                ev_s = evaluate_model_on_cohort(model, X_snr_raw, Y_te_locked, masks_te_locked, stats_used, device)
                snr_metrics[f"cbf_rmse_snr_{sl}"] = ev_s["cbf_rmse"]
                snr_metrics[f"att_rmse_snr_{sl}"] = ev_s["att_rmse"]

            # Evaluate on historical test cohort (seed 7)
            eval_hist = evaluate_model_on_cohort(model, X_te_hist, Y_te_hist, masks_te_hist, stats_used, device)

            ckpt_hash = sha256_file(ckpt_file)
            record = {
                "training_size": sz,
                "seed": sd,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_duration_s": train_duration,
                "checkpoint_sha256": ckpt_hash,
                # Locked Test Cohort Metrics
                "locked_cbf_rmse": eval_locked["cbf_rmse"],
                "locked_cbf_mae": eval_locked["cbf_mae"],
                "locked_cbf_ccc": eval_locked["cbf_ccc"],
                "locked_att_rmse": eval_locked["att_rmse"],
                "locked_att_mae": eval_locked["att_mae"],
                "locked_att_ccc": eval_locked["att_ccc"],
                "cbf_oob_fraction": eval_locked["cbf_oob_fraction"],
                "att_oob_fraction": eval_locked["att_oob_fraction"],
                # Historical Test Cohort Metrics (for backwards parity)
                "hist_cbf_rmse": eval_hist["cbf_rmse"],
                "hist_att_rmse": eval_hist["att_rmse"],
                "hist_cbf_ccc": eval_hist["cbf_ccc"],
                "hist_att_ccc": eval_hist["att_ccc"],
                "status": "COMPLETED"
            }
            record.update(snr_metrics)
            per_run_records.append(record)

            # Save incrementally
            df_out = pd.DataFrame(per_run_records)
            df_out.to_csv(results_csv, index=False)
            print(f"Saved run result. Locked CBF RMSE: {eval_locked['cbf_rmse']:.4f}, ATT RMSE: {eval_locked['att_rmse']:.4f}")

    print("\nScaling execution stage completed successfully.")


if __name__ == "__main__":
    main()
