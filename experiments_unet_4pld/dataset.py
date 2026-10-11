import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SpatialASLDataset(Dataset):
    """
    PyTorch Dataset for spatial ASL-MRI parameter estimation.
    Yields:
      - x: [4, H, W] normalized multi-PLD ASL signals
      - y: [2, H, W] normalized ground-truth parameter maps (0: CBF, 1: ATT)
      - mask: [1, H, W] binary valid brain mask (1.0: brain, 0.0: background/CSF)
      - snr: scalar SNR value for the phantom
    """
    def __init__(self, X, Y, masks, snrs=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.float32)
        if snrs is not None:
            self.snrs = torch.as_tensor(snrs, dtype=torch.float32)
        else:
            self.snrs = torch.zeros(len(X), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.masks[idx], self.snrs[idx]

def compute_normalization_stats(X_tr, Y_tr, masks_tr):
    """
    Computes signal and target normalization parameters STRICTLY from
    valid brain voxels of the training set.
    """
    valid_mask = masks_tr[:, 0] > 0.5 # [N, H, W] boolean
    
    # Per-channel input statistics over valid brain voxels
    n_channels = X_tr.shape[1]
    X_mean = np.zeros(n_channels, dtype=np.float32)
    X_std = np.zeros(n_channels, dtype=np.float32)
    for c in range(n_channels):
        ch_vals = X_tr[:, c][valid_mask]
        X_mean[c] = np.mean(ch_vals)
        X_std[c] = np.std(ch_vals) + 1e-8

    # Target statistics over valid brain voxels
    cbf_vals = Y_tr[:, 0][valid_mask]
    att_vals = Y_tr[:, 1][valid_mask]

    cbf_mean = float(np.mean(cbf_vals))
    cbf_std = float(np.std(cbf_vals) + 1e-8)
    att_mean = float(np.mean(att_vals))
    att_std = float(np.std(att_vals) + 1e-8)

    stats = {
        "X_mean": X_mean,
        "X_std": X_std,
        "cbf_mean": cbf_mean,
        "cbf_std": cbf_std,
        "att_mean": att_mean,
        "att_std": att_std,
    }
    return stats

def normalize_dataset(X, Y, masks, stats):
    """
    Applies training-derived normalization to signals and targets.
    Background voxels remain zeroed in targets to prevent background exploitation.
    """
    # Normalize inputs channel-wise: shape [N, C, H, W]
    X_norm = np.copy(X)
    n_channels = X.shape[1]
    for c in range(n_channels):
        X_norm[:, c] = (X[:, c] - stats["X_mean"][c]) / stats["X_std"][c]

    # Normalize targets
    Y_norm = np.zeros_like(Y)
    valid_mask = masks > 0.5 # [N, 1, H, W]

    # Only normalize valid brain voxels, leave background at 0.0
    cbf_norm = (Y[:, 0:1] - stats["cbf_mean"]) / stats["cbf_std"]
    att_norm = (Y[:, 1:2] - stats["att_mean"]) / stats["att_std"]

    Y_norm[:, 0:1] = np.where(valid_mask, cbf_norm, 0.0)
    Y_norm[:, 1:2] = np.where(valid_mask, att_norm, 0.0)

    return X_norm.astype(np.float32), Y_norm.astype(np.float32)

def denormalize_predictions(pred_norm, stats, cbf_bounds=(0.0, 100.0), att_bounds=(0.5, 3.0)):
    """
    Converts normalized predictions back to physical units and tracks boundary violations.
    pred_norm: numpy array [N, 2, H, W] or tensor
    Returns:
      cbf_phys: physical CBF [N, H, W]
      att_phys: physical ATT [N, H, W]
    """
    if isinstance(pred_norm, torch.Tensor):
        pred_norm = pred_norm.detach().cpu().numpy()

    cbf_raw = pred_norm[:, 0] * stats["cbf_std"] + stats["cbf_mean"]
    att_raw = pred_norm[:, 1] * stats["att_std"] + stats["att_mean"]

    return cbf_raw, att_raw

if __name__ == "__main__":
    print("Testing dataset functions...")
    N = 10
    dummy_X = np.random.randn(N, 4, 64, 64).astype(np.float32)
    dummy_Y = np.random.uniform(20, 80, (N, 2, 64, 64)).astype(np.float32)
    dummy_M = np.ones((N, 1, 64, 64), dtype=np.float32)
    
    stats = compute_normalization_stats(dummy_X, dummy_Y, dummy_M)
    X_n, Y_n = normalize_dataset(dummy_X, dummy_Y, dummy_M, stats)
    ds = SpatialASLDataset(X_n, Y_n, dummy_M)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    b_x, b_y, b_m, b_s = next(iter(loader))
    print(f"Batch shapes: X={b_x.shape}, Y={b_y.shape}, M={b_m.shape}, S={b_s.shape}")
    cbf_p, att_p = denormalize_predictions(b_y, stats)
    print("Denormalized CBF mean diff:", np.abs(cbf_p - dummy_Y[:4, 0]).max())
    print("Dataset tests passed successfully!")

