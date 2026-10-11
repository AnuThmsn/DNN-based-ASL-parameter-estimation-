import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments_unet_4pld.model import ASLUNet
from experiments_unet_4pld.dataset import (
    SpatialASLDataset,
    compute_normalization_stats,
    normalize_dataset,
    denormalize_predictions
)

class MaskedLoss(nn.Module):
    """
    Standardized masked regression loss (L1 or MSE).
    Computes loss STRICTLY on valid brain voxels specified by mask [B, 1, H, W].
    """
    def __init__(self, loss_type="l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, mask):
        # pred: [B, 2, H, W], target: [B, 2, H, W], mask: [B, 1, H, W]
        # Expand mask across parameter channels: [B, 2, H, W]
        m = mask.expand_as(pred)
        diff = pred - target
        if self.loss_type == "l1":
            err = torch.abs(diff) * m
        elif self.loss_type == "mse":
            err = (diff ** 2) * m
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        total_valid = torch.sum(m) + 1e-8
        loss = torch.sum(err) / total_valid
        return loss

def train_unet_single_seed(
    train_data,
    val_data,
    stats,
    config,
    seed=42,
    device=None,
    save_dir="experiments_unet_4pld"
):
    """
    Trains ASLUNet on synthetic spatial data for a single seed with early stopping
    and ReduceLROnPlateau scheduling.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Unpack data
    X_tr_n, Y_tr_n, masks_tr, snrs_tr = train_data
    X_val_n, Y_val_n, masks_val, snrs_val = val_data

    # DataLoaders
    batch_size = config["training"]["batch_size"]
    train_ds = SpatialASLDataset(X_tr_n, Y_tr_n, masks_tr, snrs_tr)
    val_ds = SpatialASLDataset(X_val_n, Y_val_n, masks_val, snrs_val)

    # Use persistent DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_cfg = config["model"]
    model = ASLUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        features=model_cfg["features"]
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
    log_dir = os.path.join(save_dir, "results", "logs")
    plot_dir = os.path.join(save_dir, "results", "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    start_train_time = time.time()
    print(f"\n--- Starting ASLUNet Training (Seed {seed}, Device: {device}) ---")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Max Epochs: {max_epochs}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        n_train_batches = 0

        for xb, yb, mb, _ in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item()
            n_train_batches += 1

        train_loss = train_loss_accum / max(1, n_train_batches)

        # Validation loop
        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0
        all_val_preds = []
        all_val_gt = []
        all_val_masks = []

        with torch.no_grad():
            for xb, yb, mb, _ in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb, mb)
                val_loss_accum += loss.item()
                n_val_batches += 1

                all_val_preds.append(preds.cpu())
                all_val_gt.append(yb.cpu())
                all_val_masks.append(mb.cpu())

        val_loss = val_loss_accum / max(1, n_val_batches)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Compute validation RMSE in physical units
        val_preds_arr = torch.cat(all_val_preds, dim=0).numpy()
        val_gt_arr = torch.cat(all_val_gt, dim=0).numpy()
        val_masks_arr = torch.cat(all_val_masks, dim=0).numpy()

        cbf_pred_phys, att_pred_phys = denormalize_predictions(val_preds_arr, stats)
        cbf_gt_phys, att_gt_phys = denormalize_predictions(val_gt_arr, stats)

        val_valid_mask = val_masks_arr[:, 0] > 0.5
        cbf_rmse = float(np.sqrt(np.mean((cbf_pred_phys[val_valid_mask] - cbf_gt_phys[val_valid_mask]) ** 2)))
        att_rmse = float(np.sqrt(np.mean((att_pred_phys[val_valid_mask] - att_gt_phys[val_valid_mask]) ** 2)))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cbf_rmse": cbf_rmse,
            "val_att_rmse": att_rmse,
            "lr": current_lr
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:02d}/{max_epochs} | Tr Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val CBF RMSE: {cbf_rmse:.3f} | Val ATT RMSE: {att_rmse:.4f} | LR: {current_lr:.1e}")

        # Checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            best_ckpt_path = os.path.join(ckpt_dir, f"unet_best_seed_{seed}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_cbf_rmse": cbf_rmse,
                "val_att_rmse": att_rmse,
                "stats": stats,
                "config": config,
                "seed": seed
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val Loss {best_val_loss:.4f}")
                break

    train_duration = time.time() - start_train_time
    print(f"Training completed in {train_duration:.1f}s. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")

    # Save history dataframe
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(log_dir, f"history_seed_{seed}.csv")
    history_df.to_csv(history_csv, index=False)

    # Plot convergence curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train Masked L1", color="blue")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="Val Masked L1", color="orange")
    ax1.axvline(best_epoch, linestyle="--", color="gray", label=f"Best Ep ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Standardized Masked L1 Loss")
    ax1.set_title(f"Loss Curves (Seed {seed})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history_df["epoch"], history_df["val_cbf_rmse"], label="Val CBF RMSE (ml/100g/min)", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("CBF RMSE", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.grid(True, alpha=0.3)

    ax2_r = ax2.twinx()
    ax2_r.plot(history_df["epoch"], history_df["val_att_rmse"], label="Val ATT RMSE (s)", color="green")
    ax2_r.set_ylabel("ATT RMSE", color="green")
    ax2_r.tick_params(axis="y", labelcolor="green")
    ax2.set_title(f"Validation Metric Convergence (Seed {seed})")

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"loss_curve_seed_{seed}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return best_ckpt_path, best_val_loss, history_df, train_duration

if __name__ == "__main__":
    print("Testing train script with dummy data...")
    cfg = {
        "model": {"in_channels": 4, "out_channels": 2, "features": [16, 32, 64, 128]},
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 2,
            "patience": 2,
            "lr_factor": 0.5,
            "lr_patience": 1,
            "min_lr": 1e-6
        }
    }
    dummy_X = np.random.randn(8, 4, 32, 32).astype(np.float32)
    dummy_Y = np.random.uniform(20, 80, (8, 2, 32, 32)).astype(np.float32)
    dummy_M = np.ones((8, 1, 32, 32), dtype=np.float32)
    dummy_S = np.full(8, 10.0, dtype=np.float32)
    stats = compute_normalization_stats(dummy_X, dummy_Y, dummy_M)
    X_n, Y_n = normalize_dataset(dummy_X, dummy_Y, dummy_M, stats)

    tr_d = (X_n, Y_n, dummy_M, dummy_S)
    val_d = (X_n, Y_n, dummy_M, dummy_S)

    ckpt, vloss, hist, dur = train_unet_single_seed(tr_d, val_d, stats, cfg, seed=42)
    print("Train module smoke test passed successfully!")
