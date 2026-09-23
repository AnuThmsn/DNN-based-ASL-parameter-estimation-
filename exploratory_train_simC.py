import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_gen import SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, SD_MAX, SIG_REF_SC
def calc_rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
import os

# Use Simulator C (Direct Gaussian Difference)
def generate_data_simC(N_total, rng, n_noise_levels=51):
    cbf_gt = rng.uniform(CBF_MIN, CBF_MAX, N_total).astype(np.float64)
    att_gt = rng.uniform(ATT_MIN, ATT_MAX, N_total).astype(np.float64)
    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]
    
    with np.errstate(divide='ignore'):
        snr_arr = np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)
        
    from data_gen import compute_signals_vec
    delta_M = compute_signals_vec(cbf_gt, att_gt) * SCALE
    sd = sd_arr[:, None]
    
    # Gaussian noise (equivalent to standard ASL difference assumption)
    X = (delta_M + rng.normal(0.0, 1.0, size=delta_M.shape) * sd).astype(np.float32)
    Y = np.stack([cbf_gt, att_gt], axis=1).astype(np.float32)
    return X, Y, snr_arr

rng_tr = np.random.default_rng(1)
rng_val = np.random.default_rng(2)
rng_test = np.random.default_rng(7)

X_tr, Y_tr, _ = generate_data_simC(200_000, rng_tr, 100) # smaller dataset for speed
X_val, Y_val, _ = generate_data_simC(20_000, rng_val, 100)
X_test, Y_test, snr_test = generate_data_simC(10_000, rng_test, 51)

X_mean = X_tr.mean(axis=0, keepdims=True)
X_std = X_tr.std(axis=0, keepdims=True) + 1e-8
Y_mean = Y_tr.mean(axis=0, keepdims=True)
Y_std = Y_tr.std(axis=0, keepdims=True) + 1e-8

X_tr_n = (X_tr - X_mean) / X_std
X_val_n = (X_val - X_mean) / X_std
X_test_n = (X_test - X_mean) / X_std
Y_tr_n = (Y_tr - Y_mean) / Y_std
Y_val_n = (Y_val - Y_mean) / Y_std

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SharedNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU()
        )
        self.cbf_head = nn.Sequential(
            nn.Linear(128, 32), nn.ELU(),
            nn.Linear(32, 1)
        )
        self.att_head = nn.Sequential(
            nn.Linear(128, 32), nn.ELU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        s = self.shared(x)
        c = self.cbf_head(s)
        a = self.att_head(s)
        return torch.cat([c, a], dim=1)

net = SharedNet(4).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
crit = nn.L1Loss()

X_tr_t = torch.tensor(X_tr_n, device=device)
Y_tr_t = torch.tensor(Y_tr_n, device=device)
X_val_t = torch.tensor(X_val_n, device=device)
Y_val_t = torch.tensor(Y_val_n, device=device)

print("Training DNN on Correct Simulator C...")
best_val = float('inf')
for epoch in range(40):
    net.train()
    perm = torch.randperm(X_tr_t.size(0))
    for i in range(0, X_tr_t.size(0), 4096):
        idx = perm[i:i+4096]
        opt.zero_grad()
        loss = crit(net(X_tr_t[idx]), Y_tr_t[idx])
        loss.backward()
        opt.step()
        
    net.eval()
    with torch.no_grad():
        val_loss = crit(net(X_val_t), Y_val_t).item()
    if val_loss < best_val:
        best_val = val_loss
        torch.save(net.state_dict(), "simC_best.pth")

net.load_state_dict(torch.load("simC_best.pth"))
net.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test_n, device=device)
    preds_n = []
    for i in range(0, X_test_t.size(0), 4096):
        preds_n.append(net(X_test_t[i:i+4096]).cpu().numpy())
    preds_n = np.concatenate(preds_n, axis=0)

preds = preds_n * Y_std + Y_mean
cbf_pred = np.clip(preds[:, 0], CBF_MIN, CBF_MAX)
att_pred = np.clip(preds[:, 1], ATT_MIN, ATT_MAX)

mask10 = (snr_test >= 9) & (snr_test <= 11)
cbf_rmse_10 = calc_rmse(Y_test[mask10, 0], cbf_pred[mask10])
att_rmse_10 = calc_rmse(Y_test[mask10, 1], att_pred[mask10])

print(f"DNN (Sim C) SNR=10 -> CBF RMSE: {cbf_rmse_10:.4f}, ATT RMSE: {att_rmse_10:.4f}")
